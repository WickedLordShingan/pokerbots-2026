'''
Sneak Peek Hold'em Bot — v7  (COMPETITION BUILD)
==================================================

Key facts learned from reading engine.py:
  - 1000 rounds, 30 second TOTAL time bank → ~30ms average per query
  - Auction: winner pays loser's bid (not their own!). Tie = both pay own bid
    and both see a card. This changes auction math completely.
  - After auction resolves, PokerState gives us state.bids = [p0_bid, p1_bid]
    so we read opponent's exact bid every hand.
  - Invalid action = forced fold/check. Never submit illegal raise amounts.

Speed strategy (30ms budget):
  - ZERO Monte Carlo anywhere. Equity = lookup table + bucket classifier.
  - All auction math is O(1) arithmetic.
  - Personality classifier is cached — only recomputes at hand boundaries.
  - Full try/except wrapper — a bug never costs a round disqualification.

Training integration:
  - learned_params.json (written by train_from_logs.py) loads at __init__.
  - Provides opp_bluff_prior AND calibrated_priors which override the 
    PersonalityClassifier.PRIORS with values from real competition logs.
  - Run: python train_from_logs.py --bot-name YourBotName after each log scrape.

Personality system:
  - COCKY: wide range, bluffs, overbids auctions, rarely folds
  - SAFE:  tight, value-only, underbids auctions, folds to pressure
  - LOSING: erratic, high-variance sizing, inconsistent
  Counter-strategies adjust all thresholds and auction multipliers.

Auction strategy (corrected for actual engine second-price rules):
  - You win by bidding more than opponent. You pay THEIR bid (not yours).
  - So optimal strategy = bid your true info valuation.
  - Sniper heuristic: bid just above predicted opp bid if cheaper than valuation.
  - Info value = uncertainty * pot * fraction (uncertainty peaks at equity=0.5).
'''

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot
from pkbot.states import BIG_BLIND, SMALL_BLIND, STARTING_STACK

import math
import json
import os

# =============================================================================
# PREFLOP EQUITY TABLE
# =============================================================================
RANK_ORDER = '23456789TJQKA'

PREFLOP_EQUITY = {
    ('A','A','o'):0.85,('K','K','o'):0.82,('Q','Q','o'):0.80,('J','J','o'):0.77,
    ('T','T','o'):0.75,('9','9','o'):0.72,('8','8','o'):0.69,('7','7','o'):0.66,
    ('6','6','o'):0.63,('5','5','o'):0.60,('4','4','o'):0.57,('3','3','o'):0.54,
    ('2','2','o'):0.50,
    ('A','K','s'):0.67,('A','Q','s'):0.66,('A','J','s'):0.65,('A','T','s'):0.64,
    ('A','9','s'):0.62,('A','8','s'):0.61,('A','7','s'):0.60,('A','6','s'):0.59,
    ('A','5','s'):0.59,('A','4','s'):0.58,('A','3','s'):0.58,('A','2','s'):0.57,
    ('A','K','o'):0.65,('A','Q','o'):0.64,('A','J','o'):0.63,('A','T','o'):0.62,
    ('A','9','o'):0.60,('A','8','o'):0.59,('A','7','o'):0.58,('A','6','o'):0.57,
    ('A','5','o'):0.57,('A','4','o'):0.56,('A','3','o'):0.55,('A','2','o'):0.54,
    ('K','Q','s'):0.63,('K','J','s'):0.62,('K','T','s'):0.61,('K','9','s'):0.58,
    ('K','8','s'):0.57,('K','7','s'):0.56,('K','6','s'):0.55,('K','4','s'):0.53,
    ('K','3','s'):0.52,('K','2','s'):0.51,
    ('K','Q','o'):0.61,('K','J','o'):0.60,('K','T','o'):0.59,('K','9','o'):0.56,
    ('K','8','o'):0.54,('K','7','o'):0.53,('K','6','o'):0.52,('K','5','o'):0.51,
    ('K','4','o'):0.50,('K','3','o'):0.49,('K','2','o'):0.48,
    ('Q','J','s'):0.60,('Q','T','s'):0.59,('Q','9','s'):0.56,('Q','8','s'):0.54,
    ('Q','7','s'):0.53,('Q','6','s'):0.52,('Q','5','s'):0.51,('Q','4','s'):0.50,
    ('Q','3','s'):0.49,('Q','2','s'):0.48,
    ('Q','J','o'):0.58,('Q','T','o'):0.56,('Q','9','o'):0.54,('Q','8','o'):0.52,
    ('Q','7','o'):0.50,('Q','6','o'):0.49,('Q','5','o'):0.48,('Q','4','o'):0.47,
    ('Q','3','o'):0.46,('Q','2','o'):0.45,
    ('J','T','s'):0.58,('J','9','s'):0.55,('J','8','s'):0.53,('J','7','s'):0.51,
    ('J','6','s'):0.50,('J','5','s'):0.49,('J','4','s'):0.48,('J','3','s'):0.47,
    ('J','2','s'):0.46,
    ('J','T','o'):0.55,('J','9','o'):0.53,('J','8','o'):0.50,('J','7','o'):0.48,
    ('J','6','o'):0.47,('J','5','o'):0.46,('J','4','o'):0.45,('J','3','o'):0.44,
    ('J','2','o'):0.43,
    ('T','9','s'):0.56,('T','8','s'):0.54,('T','7','s'):0.52,('T','6','s'):0.50,
    ('T','5','s'):0.49,('T','4','s'):0.48,('T','3','s'):0.47,('T','2','s'):0.46,
    ('T','9','o'):0.53,('T','8','o'):0.51,('T','7','o'):0.49,('T','6','o'):0.47,
    ('T','5','o'):0.46,('T','4','o'):0.45,('T','3','o'):0.44,('T','2','o'):0.43,
    ('9','8','s'):0.55,('9','7','s'):0.53,('9','6','s'):0.51,('9','5','s'):0.49,
    ('9','4','s'):0.48,('9','3','s'):0.47,('9','2','s'):0.46,
    ('9','8','o'):0.52,('9','7','o'):0.50,('9','6','o'):0.48,('9','5','o'):0.46,
    ('9','4','o'):0.45,('9','3','o'):0.44,('9','2','o'):0.43,
    ('8','7','s'):0.54,('8','6','s'):0.52,('8','5','s'):0.50,('8','4','s'):0.48,
    ('8','3','s'):0.47,('8','2','s'):0.46,
    ('8','7','o'):0.51,('8','6','o'):0.49,('8','5','o'):0.47,('8','4','o'):0.45,
    ('8','3','o'):0.44,('8','2','o'):0.43,
    ('7','6','s'):0.53,('7','5','s'):0.51,('7','4','s'):0.49,('7','3','s'):0.47,
    ('7','2','s'):0.46,('7','6','o'):0.50,('7','5','o'):0.48,('7','4','o'):0.46,
    ('7','3','o'):0.44,('7','2','o'):0.43,
    ('6','5','s'):0.52,('6','4','s'):0.50,('6','3','s'):0.48,('6','2','s'):0.46,
    ('6','5','o'):0.49,('6','4','o'):0.47,('6','3','o'):0.45,('6','2','o'):0.43,
    ('5','4','s'):0.51,('5','3','s'):0.49,('5','2','s'):0.47,
    ('5','4','o'):0.48,('5','3','o'):0.46,('5','2','o'):0.44,
    ('4','3','s'):0.49,('4','2','s'):0.47,('4','3','o'):0.46,('4','2','o'):0.44,
    ('3','2','s'):0.47,('3','2','o'):0.44,
}

def _rank_idx(r):
    return RANK_ORDER.index(r)

def preflop_equity(hole_cards):
    h1, h2 = hole_cards[0], hole_cards[1]
    suited = 's' if h1[1] == h2[1] else 'o'
    ranks  = sorted([h1[0], h2[0]], key=_rank_idx, reverse=True)
    return PREFLOP_EQUITY.get((ranks[0], ranks[1], suited), 0.48)


# =============================================================================
# POSTFLOP HAND BUCKETER (O(n), no eval7)
# 0=trash 1=weak-pair/draw 2=top-mid-pair 3=flush/straight/trips 4=monsters
# =============================================================================

def _postflop_bucket(hole_cards, board):
    if not board:
        return 0
    all_cards   = hole_cards + board
    ranks       = [c[0] for c in all_cards]
    suits       = [c[1] for c in all_cards]
    hole_ranks  = [c[0] for c in hole_cards]
    board_ranks = [c[0] for c in board]

    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1

    flush      = any(v >= 5 for v in suit_counts.values())
    flush_draw = any(v == 4 for v in suit_counts.values())

    unique_sorted = sorted({_rank_idx(r) for r in ranks})
    consec = max_consec = 1
    for i in range(1, len(unique_sorted)):
        if unique_sorted[i] == unique_sorted[i-1] + 1:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 1
    straight      = max_consec >= 5
    straight_draw = max_consec == 4

    trips_plus   = [r for r, c in rank_counts.items() if c >= 3]
    pairs        = [r for r, c in rank_counts.items() if c == 2]
    fh_or_better = bool(trips_plus and (len(trips_plus) >= 2 or pairs))

    my_pair_ranks = []
    for r in rank_counts:
        if r in hole_ranks and rank_counts[r] >= 2:
            my_pair_ranks.append(_rank_idx(r))
    if len(hole_ranks) >= 2 and hole_ranks[0] == hole_ranks[1]:
        my_pair_ranks.append(_rank_idx(hole_ranks[0]))

    board_high = max((_rank_idx(r) for r in board_ranks), default=0)

    if fh_or_better or (flush and straight):
        return 4
    if flush or straight or trips_plus:
        return 3
    if my_pair_ranks:
        best = max(my_pair_ranks)
        if best >= board_high or best >= board_high - 2:
            return 2
        return 1
    if flush_draw or straight_draw:
        return 1
    return 0


# =============================================================================
# EQUITY ESTIMATION (pure lookup, zero Monte Carlo)
# =============================================================================

_BUCKET_EQUITY = [0.23, 0.41, 0.58, 0.76, 0.92]

def _revealed_card_adj(opp_card, board):
    if not opp_card or not board:
        return 0.0
    r = opp_card[0]
    idx = _rank_idx(r)
    if r in [c[0] for c in board]:
        return -0.08
    if idx >= _rank_idx('Q'):
        return -0.04
    if idx <= _rank_idx('5'):
        return +0.03
    return 0.0

def estimate_equity(hole_cards, board, opp_known_card=None):
    if not board:
        return preflop_equity(hole_cards)
    bucket = _postflop_bucket(hole_cards, board)
    equity = _BUCKET_EQUITY[bucket]
    equity += _revealed_card_adj(opp_known_card, board)
    return max(0.05, min(0.97, equity))


# =============================================================================
# PERSONALITY CLASSIFIER
# =============================================================================

COCKY   = 'COCKY'
SAFE    = 'SAFE'
LOSING  = 'LOSING'
UNKNOWN = 'UNKNOWN'
MIN_HANDS_TO_CLASSIFY = 8


class PersonalityClassifier:
    PRIORS = {
        COCKY:  {'vpip':(0.80,0.12),'aggression':(0.65,0.12),'fold_to_bet':(0.20,0.10),'auction_ratio':(0.22,0.10),'bet_sizing':(0.80,0.20)},
        SAFE:   {'vpip':(0.30,0.10),'aggression':(0.25,0.10),'fold_to_bet':(0.65,0.12),'auction_ratio':(0.05,0.04),'bet_sizing':(0.45,0.15)},
        LOSING: {'vpip':(0.60,0.18),'aggression':(0.50,0.20),'fold_to_bet':(0.40,0.18),'auction_ratio':(0.15,0.12),'bet_sizing':(0.65,0.30)},
    }

    def __init__(self):
        self.hands_seen = 0; self.vpip_count = 0
        self.total_actions = 0; self.aggressive_actions = 0
        self.times_faced_bet = 0; self.folds_to_bet = 0
        self.auction_ratios = []; self.bet_size_ratios = []
        self._cache = UNKNOWN; self._cache_valid = False

    def record_hand_vpip(self, did_vpip):
        self.hands_seen += 1
        if did_vpip: self.vpip_count += 1
        self._cache_valid = False

    def record_action(self, atype):
        self.total_actions += 1
        if atype in ('bet','raise'): self.aggressive_actions += 1
        self._cache_valid = False

    def record_faced_bet(self, did_fold):
        self.times_faced_bet += 1
        if did_fold: self.folds_to_bet += 1
        self._cache_valid = False

    def record_auction_ratio(self, bid, pot):
        if pot > 0: self.auction_ratios.append(min(bid/pot, 3.0))
        self._cache_valid = False

    def record_bet_sizing(self, bet, pot):
        if pot > 0: self.bet_size_ratios.append(min(bet/pot, 3.0))
        self._cache_valid = False

    @staticmethod
    def _lp(x, mu, sig):
        return -0.5*((x-mu)/sig)**2 - math.log(sig)

    def classify(self):
        if self._cache_valid: return self._cache
        if self.hands_seen < MIN_HANDS_TO_CLASSIFY: return UNKNOWN
        sd = lambda n,d: n/d if d>0 else 0.5
        feats = {
            'vpip':         sd(self.vpip_count, self.hands_seen),
            'aggression':   sd(self.aggressive_actions, max(self.total_actions,1)),
            'fold_to_bet':  sd(self.folds_to_bet, max(self.times_faced_bet,1)),
            'auction_ratio': sum(self.auction_ratios)/len(self.auction_ratios) if self.auction_ratios else 0.10,
            'bet_sizing':    sum(self.bet_size_ratios)/len(self.bet_size_ratios) if self.bet_size_ratios else 0.55,
        }
        scores = {lb: sum(self._lp(feats[f],mu,sg) for f,(mu,sg) in pr.items())
                  for lb,pr in self.PRIORS.items()}
        best = max(scores, key=scores.get)
        self._cache = best; self._cache_valid = True
        return best


# =============================================================================
# COUNTER-STRATEGY PROFILES
# =============================================================================

COUNTER_PROFILES = {
    COCKY:  {'call_eq_adj':-0.05, 'open_eq_adj':+0.05, 'raise_eq_adj':+0.03,
             'bluff_div':2.5,  'fold_adj':0.02,  'auction_mult':0.70},
    SAFE:   {'call_eq_adj':+0.03, 'open_eq_adj':-0.08, 'raise_eq_adj':-0.05,
             'bluff_div':0.60, 'fold_adj':-0.02, 'auction_mult':1.40},
    LOSING: {'call_eq_adj':-0.04, 'open_eq_adj':+0.02, 'raise_eq_adj':+0.01,
             'bluff_div':1.30, 'fold_adj':0.00,  'auction_mult':1.00},
    UNKNOWN:{'call_eq_adj':0.00,  'open_eq_adj':0.00,  'raise_eq_adj':0.00,
             'bluff_div':1.00, 'fold_adj':0.00,  'auction_mult':1.00},
}


# =============================================================================
# AUCTION STRATEGY  (second-price / sniper model)
# =============================================================================
# ENGINE RULE: highest bidder wins and pays the LOWER (opponent's) bid.
# Tie: both pay their own bid and both see a card.
# → This is Vickrey-like: bid your true valuation of information.
# → Sniper: if opp typically bids X, bid X+delta to win while paying X.

INFO_VALUE_FRAC = 0.28   # fraction of pot info is worth at peak uncertainty (eq=0.5)
SNIPE_DELTA     = 2      # chips above predicted opp bid
MAX_BID_FRAC    = 0.06   # never more than 6% of stack
MAX_BID_BB      = 18     # hard ceiling in big blinds


def _predict_opp_bid(history):
    if not history: return 0.0
    n = len(history)
    weights = [1.25**i for i in range(n)]
    return sum(b*w for b,w in zip(history,weights)) / sum(weights)


def compute_auction_bid(hole_cards, board, pot, my_chips,
                        personality, chip_lead,
                        opp_bid_history, auction_wins, auction_total):
    equity      = estimate_equity(hole_cards, board)
    uncertainty = 1.0 - abs(2.0*equity - 1.0)
    if uncertainty < 0.12:
        return 0

    effective_pot  = max(pot, BIG_BLIND * 3)
    info_value     = uncertainty * effective_pot * INFO_VALUE_FRAC

    pers_mult = COUNTER_PROFILES[personality]['auction_mult']

    if chip_lead < -300:   sess_mult = 1.30
    elif chip_lead < -100: sess_mult = 1.15
    elif chip_lead > 300:  sess_mult = 0.75
    elif chip_lead > 100:  sess_mult = 0.88
    else:                  sess_mult = 1.00

    if auction_total >= 6:
        wr = auction_wins / auction_total
        win_mult = 1.20 if wr < 0.30 else (0.85 if wr > 0.75 else 1.00)
    else:
        win_mult = 1.00

    max_pay = info_value * pers_mult * sess_mult * win_mult

    predicted_opp = _predict_opp_bid(opp_bid_history)
    snipe         = predicted_opp + SNIPE_DELTA

    if snipe <= max_pay:
        chosen = snipe       # cheap win
    elif not opp_bid_history:
        chosen = max_pay     # no history — bid valuation
    else:
        chosen = 0           # their bid > our valuation — don't auction

    cap   = min(my_chips * MAX_BID_FRAC, MAX_BID_BB * BIG_BLIND)
    return max(0, int(min(chosen, cap)))


# =============================================================================
# HAND STATE TRACKER
# =============================================================================

class HandState:
    __slots__ = ['my_hand','opp_actions','opp_bets','opp_vpip','opp_auction_bid','last_aggressor']
    def __init__(self, my_hand):
        self.my_hand          = my_hand
        self.opp_actions      = []
        self.opp_bets         = []
        self.opp_vpip         = False
        self.opp_auction_bid  = None
        self.last_aggressor   = ''

    def opp_bet(self, amt, pot):
        self.opp_actions.append('bet'); self.opp_bets.append((amt,pot)); self.opp_vpip = True
    def opp_raise(self, amt, pot):
        self.opp_actions.append('raise'); self.opp_bets.append((amt,pot)); self.opp_vpip = True
    def opp_call(self):
        self.opp_actions.append('call'); self.opp_vpip = True
    def opp_check(self):
        self.opp_actions.append('check')
    def opp_fold(self):
        self.opp_actions.append('fold')


# =============================================================================
# MAIN BOT
# =============================================================================

class Player(BaseBot):

    def __init__(self):
        self.classifier       = PersonalityClassifier()
        self.opp_bid_history  = []
        self.auction_total    = 0
        self.auction_wins     = 0
        self.opp_showdowns    = 0
        self.opp_bluffs       = 0
        self.opp_bluff_ratio  = self._load_learned_params()

        # Per-hand
        self.opp_known_card   = None
        self.opp_auction_bid  = None
        self.hand_state       = None
        self._prev_pot        = 0
        self._prev_opp_wager  = 0
        self._counted_win     = False

    # ---- Learned params ----

    def _load_learned_params(self):
        """
        Load trained parameters. Strategy:

        1. BAKED-IN DEFAULTS (always active, even on competition server with no JSON):
           These are values trained from real competition logs and hardcoded here.
           Update these constants every time you run train_from_logs.py locally.

        2. JSON OVERRIDE (if learned_params.json exists alongside bot.py):
           If found, overrides the baked-in values. Useful for local testing
           and for competitions that allow file submissions alongside bot.py.

        This means the bot runs identically whether or not the JSON file is present.
        TO UPDATE AFTER TRAINING: replace the _BAKED constants below with the
        values printed by train_from_logs.py, then resubmit bot.py.
        """

        # ----------------------------------------------------------------
        # BAKED-IN TRAINED VALUES — update these after each training run
        # Format: replace with output from train_from_logs.py --verbose
        # ----------------------------------------------------------------
        _BAKED_BLUFF_PRIOR = 0.6846   # from learned_params.json opp_bluff_prior

        # Calibrated personality priors from training (mu, sigma per feature).
        # If you haven't trained yet, these are the theory-based defaults.
        # After running train_from_logs.py, copy the "Calibrated prior means" output here.
        _BAKED_PRIORS = {
            COCKY:  {'vpip':(0.80,0.12),'aggression':(0.65,0.12),
                     'fold_to_bet':(0.20,0.10),'auction_ratio':(0.22,0.10),'bet_sizing':(0.80,0.20)},
            SAFE:   {'vpip':(0.30,0.10),'aggression':(0.25,0.10),
                     'fold_to_bet':(0.65,0.12),'auction_ratio':(0.05,0.04),'bet_sizing':(0.45,0.15)},
            LOSING: {'vpip':(0.60,0.18),'aggression':(0.50,0.20),
                     'fold_to_bet':(0.40,0.18),'auction_ratio':(0.15,0.12),'bet_sizing':(0.65,0.30)},
        }
        # ----------------------------------------------------------------

        # Apply baked-in priors as the baseline
        PersonalityClassifier.PRIORS = _BAKED_PRIORS
        bluff = _BAKED_BLUFF_PRIOR

        # Attempt to load JSON — overrides baked values if present
        path = os.path.join(os.path.dirname(__file__), 'learned_params.json')
        try:
            with open(path) as f:
                data = json.load(f)

            json_bluff = float(data.get('opp_bluff_prior', bluff))
            bluff = max(0.0, min(1.0, json_bluff))

            cal     = data.get('calibrated_priors', {})
            valid_l = {COCKY, SAFE, LOSING}
            valid_f = {'vpip','aggression','fold_to_bet','auction_ratio','bet_sizing'}
            if cal and set(cal.keys()) == valid_l:
                new_p = {}
                ok    = True
                for lb, fm in cal.items():
                    new_p[lb] = {}
                    for feat, val in fm.items():
                        if feat not in valid_f: continue
                        if isinstance(val, (list, tuple)) and len(val) == 2:
                            mu, sig = float(val[0]), float(val[1])
                            if 0 <= mu <= 1 and 0.01 <= sig <= 1:
                                new_p[lb][feat] = (mu, sig)
                    for feat in valid_f:
                        if feat not in new_p[lb]:
                            new_p[lb][feat] = _BAKED_PRIORS[lb][feat]
                    if len(new_p[lb]) != len(valid_f):
                        ok = False; break
                if ok:
                    PersonalityClassifier.PRIORS = new_p

        except Exception:
            pass  # JSON missing or malformed — baked values already applied above

        return bluff

    # ---- Hand lifecycle ----

    def on_hand_start(self, game_info, current_state):
        self.opp_known_card  = None
        self.opp_auction_bid = None
        self.hand_state      = HandState(my_hand=list(current_state.my_hand))
        self._prev_pot       = current_state.pot
        self._prev_opp_wager = current_state.opp_wager
        self._counted_win    = False

    def on_hand_end(self, game_info, current_state):
        hs = self.hand_state
        if not hs: return

        self.classifier.record_hand_vpip(hs.opp_vpip)
        for at in hs.opp_actions:
            self.classifier.record_action(at)

        prev = None
        for at in hs.opp_actions:
            if prev in ('bet','raise') and at in ('call','fold'):
                self.classifier.record_faced_bet(did_fold=(at=='fold'))
            prev = at

        # Opponent folded early to our aggression
        if hs.last_aggressor == 'us' and current_state.payoff > 0 and current_state.street != 'river':
            self.classifier.record_faced_bet(did_fold=True)

        if hs.opp_auction_bid is not None:
            self.classifier.record_auction_ratio(hs.opp_auction_bid, max(self._prev_pot, BIG_BLIND*3))

        for bet, pot in hs.opp_bets:
            self.classifier.record_bet_sizing(bet, pot)

        # Showdown bluff tracking
        opp_cards = current_state.opp_revealed_cards
        board     = current_state.board
        if opp_cards and len(opp_cards) >= 2 and len(board) >= 3:
            self.opp_showdowns += 1
            bucket = _postflop_bucket(list(opp_cards), list(board))
            if bucket <= 1 and current_state.pot >= 200:
                self.opp_bluffs += 1
            pw = max(5.0, 15.0 - 0.5*self.opp_showdowns)
            self.opp_bluff_ratio = (pw*0.20 + self.opp_bluffs) / (pw + self.opp_showdowns)

    # ---- Main decision ----

    def get_move(self, game_info, current_state):
        try:
            return self._move(game_info, current_state)
        except Exception:
            if current_state.street == 'auction':
                return ActionBid(0)
            return ActionCheck() if current_state.can_act(ActionCheck) else ActionFold()

    def _move(self, game_info, current_state):
        street       = current_state.street
        hole_cards   = list(current_state.my_hand)
        board        = list(current_state.board)
        pot          = current_state.pot
        my_chips     = current_state.my_chips
        opp_chips    = current_state.opp_chips
        cost_to_call = current_state.cost_to_call
        opp_wager    = current_state.opp_wager
        is_bb        = current_state.is_bb

        # ---- Read opp auction bid from state.bids (available post-auction) ----
        bids = getattr(current_state, 'bids', None)
        if bids and len(bids) == 2 and all(b is not None for b in bids) and self.opp_auction_bid is None:
            my_idx  = 1 if is_bb else 0
            opp_bid = bids[1 - my_idx]
            if opp_bid is not None:
                self.opp_auction_bid = int(opp_bid)
                if self.hand_state:
                    self.hand_state.opp_auction_bid = self.opp_auction_bid
                self.opp_bid_history.append(self.opp_auction_bid)
                self.classifier.record_auction_ratio(self.opp_auction_bid, max(self._prev_pot, BIG_BLIND*3))

        # ---- Capture revealed card ----
        if current_state.opp_revealed_cards and not self.opp_known_card:
            self.opp_known_card = current_state.opp_revealed_cards[0]
            if not self._counted_win:
                self.auction_wins += 1
                self._counted_win  = True

        # ---- Detect opp actions via wager delta ----
        if self.hand_state and street not in ('auction', 'pre-flop') and opp_wager > self._prev_opp_wager:
            delta = opp_wager - self._prev_opp_wager
            if self._prev_opp_wager == 0:
                self.hand_state.opp_bet(delta, self._prev_pot)
            else:
                self.hand_state.opp_raise(delta, self._prev_pot)
        self._prev_pot       = pot
        self._prev_opp_wager = opp_wager

        personality = self.classifier.classify()
        chip_lead   = my_chips - opp_chips

        # ================================================================
        # AUCTION
        # ================================================================
        if street == 'auction':
            self.auction_total += 1
            bid = compute_auction_bid(
                hole_cards, board, pot, my_chips,
                personality, chip_lead,
                self.opp_bid_history, self.auction_wins, self.auction_total,
            )
            return ActionBid(bid)

        # ================================================================
        # BETTING
        # ================================================================
        equity     = estimate_equity(hole_cards, board, self.opp_known_card)
        bluff_bonus = max(-0.12, min(0.12, self.opp_bluff_ratio - 0.20))
        adj_equity  = max(0.05, min(0.97, equity + bluff_bonus))

        return self._decide(current_state, adj_equity, equity,
                            street, pot, cost_to_call, personality,
                            my_chips, chip_lead)

    def _decide(self, state, adj_eq, raw_eq, street, pot,
                cost_to_call, personality, my_chips, chip_lead):
        p        = COUNTER_PROFILES[personality]
        fold_adj = p['fold_adj']
        call_adj = p['call_eq_adj']
        open_adj = p['open_eq_adj']
        raise_adj= p['raise_eq_adj']
        bdiv     = p['bluff_div']

        # ---- Facing a bet ----
        if cost_to_call > 0:
            pot_odds    = cost_to_call / max(pot + cost_to_call, 1)
            call_thresh = pot_odds + fold_adj - call_adj
            if adj_eq < call_thresh:
                return ActionFold() if state.can_act(ActionFold) else ActionCheck()

            if state.can_act(ActionRaise) and adj_eq > (0.68 + raise_adj):
                min_r, max_r = state.raise_bounds
                frac   = 0.75 if adj_eq > 0.82 else 0.50
                amount = max(min_r, min(max_r, int(pot * frac)))
                if self.hand_state: self.hand_state.last_aggressor = 'us'
                return ActionRaise(amount)

            return ActionCall()

        # ---- Open action ----
        if state.can_act(ActionRaise):
            base_thresh  = 0.56 + open_adj
            bluff_thresh = max(0.35, base_thresh / max(bdiv, 0.1))
            if adj_eq > bluff_thresh:
                min_r, max_r = state.raise_bounds
                frac = 0.50 if raw_eq < 0.75 else 0.67
                # Overbet vs SAFE on late streets — they fold to big bets
                if personality == SAFE and street in ('turn','river') and raw_eq > 0.60:
                    frac = 1.00
                amount = max(min_r, min(max_r, int(pot * frac)))
                if self.hand_state: self.hand_state.last_aggressor = 'us'
                return ActionRaise(amount)

        return ActionCheck()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
