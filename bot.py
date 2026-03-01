'''
Sneak Peek Hold'em Bot — v6
===========================

Major additions over v5:

1. PERSONALITY SYSTEM
   Three opponent archetypes:
     COCKY  — overconfident, bets wide, bluffs often, overbids auctions.
               Bets frequently with weak hands, rarely folds to pressure.
     SAFE   — tight-passive, only plays strong hands, folds to aggression.
               Low VPIP, rarely bluffs, low auction bids.
     LOSING — erratic, tilt-driven, inconsistent sizing, emotional patterns.
               Swings between over-aggression and passivity.

2. PERSONALITY CLASSIFIER (lightweight ML via Bayesian feature scoring)
   Tracks per-hand metrics across the session:
     - VPIP (voluntarily put chips in pot)
     - Aggression frequency (bets+raises / total actions)
     - Fold-to-bet frequency
     - Auction bid ratio (bid / pot)
     - Average bet/pot sizing ratio
   Uses naive Bayes log-likelihood scoring — no external ML libs needed,
   works in real-time as data accumulates.

3. COUNTER-PERSONALITY STRATEGIES
   COCKY  → Play SAFE against them: tighten ranges, don't bluff, let them
             hang themselves, call-down with medium hands.
   SAFE   → Play COCKY against them: bluff more, steal pots, force folds,
             overbid auctions to bully.
   LOSING → Play SAFE-leaning: exploit tilt bluffs by calling wider,
             but don't overcommit since they're unpredictable.
   UNKNOWN → Balanced default until enough data (< 6 hands).

4. ADAPTIVE AUCTION BIDDING
   Bid scales based on:
     (a) Information value of the card (EV-delta model from v5)
     (b) Session context: behind on chips → bid more aggressively
         to gain info edge; ahead → bid conservatively, protect lead.
     (c) Opponent personality: bid high vs SAFE (they underbid, easy wins),
         bid low vs COCKY (they overbid, let them waste chips).
     (d) Auction win rate history: if being outbid consistently, nudge up.

5. HAND STATE TRACKING
   Per-hand tracking of: actions, bet sizes, streets played, auction result.
   All flushed into classifier on hand end.
'''

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot
from pkbot.states import BIG_BLIND, SMALL_BLIND, STARTING_STACK

import random
import math

# ---------------------------------------------------------------------------
# Preflop equity lookup
# ---------------------------------------------------------------------------
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

def preflop_equity(hole_cards):
    h1, h2 = hole_cards[0], hole_cards[1]
    r1, r2 = h1[0], h2[0]
    suited = 's' if h1[1] == h2[1] else 'o'
    ranks = sorted([r1, r2], key=lambda x: RANK_ORDER.index(x), reverse=True)
    return PREFLOP_EQUITY.get((ranks[0], ranks[1], suited), 0.48)


# ===========================================================================
# PERSONALITY SYSTEM
# ===========================================================================

COCKY   = 'COCKY'
SAFE    = 'SAFE'
LOSING  = 'LOSING'
UNKNOWN = 'UNKNOWN'

MIN_HANDS_FOR_CLASSIFY = 6


class PersonalityClassifier:
    """
    Lightweight Bayesian personality classifier using Gaussian log-likelihoods.

    Features observed:
      vpip          — how often opp voluntarily enters pot
      aggression    — (bets+raises) / total_actions
      fold_to_bet   — folds / times_faced_bet
      auction_ratio — avg(bid/pot) in auctions
      bet_sizing    — avg(bet/pot) when they bet

    Reference priors (mu, sigma) calibrated from poker theory for each archetype.
    """

    PRIORS = {
        COCKY:  {'vpip': (0.80, 0.12), 'aggression': (0.65, 0.12),
                 'fold_to_bet': (0.20, 0.10), 'auction_ratio': (0.22, 0.10),
                 'bet_sizing':  (0.80, 0.20)},
        SAFE:   {'vpip': (0.30, 0.10), 'aggression': (0.25, 0.10),
                 'fold_to_bet': (0.65, 0.12), 'auction_ratio': (0.05, 0.04),
                 'bet_sizing':  (0.45, 0.15)},
        LOSING: {'vpip': (0.60, 0.18), 'aggression': (0.50, 0.20),
                 'fold_to_bet': (0.40, 0.18), 'auction_ratio': (0.15, 0.12),
                 'bet_sizing':  (0.65, 0.30)},
    }

    def __init__(self):
        self.hands_seen         = 0
        self.vpip_count         = 0
        self.total_actions      = 0
        self.aggressive_actions = 0
        self.times_faced_bet    = 0
        self.folds_to_bet       = 0
        self.auction_ratios     = []
        self.bet_size_ratios    = []

    def record_hand_vpip(self, did_vpip):
        self.hands_seen += 1
        if did_vpip:
            self.vpip_count += 1

    def record_action(self, action_type):
        self.total_actions += 1
        if action_type in ('bet', 'raise'):
            self.aggressive_actions += 1

    def record_faced_bet(self, did_fold):
        self.times_faced_bet += 1
        if did_fold:
            self.folds_to_bet += 1

    def record_auction_bid(self, bid, pot):
        if pot > 0:
            self.auction_ratios.append(bid / pot)

    def record_bet_size(self, bet, pot):
        if pot > 0:
            self.bet_size_ratios.append(bet / pot)

    def _features(self):
        safe_div = lambda n, d, default=0.5: n / d if d > 0 else default
        return {
            'vpip':          safe_div(self.vpip_count, self.hands_seen, 0.50),
            'aggression':    safe_div(self.aggressive_actions, max(self.total_actions,1), 0.40),
            'fold_to_bet':   safe_div(self.folds_to_bet, max(self.times_faced_bet,1), 0.40),
            'auction_ratio': sum(self.auction_ratios)/len(self.auction_ratios) if self.auction_ratios else 0.10,
            'bet_sizing':    sum(self.bet_size_ratios)/len(self.bet_size_ratios) if self.bet_size_ratios else 0.55,
        }

    @staticmethod
    def _log_prob(x, mu, sigma):
        return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma)

    def classify(self):
        if self.hands_seen < MIN_HANDS_FOR_CLASSIFY:
            return UNKNOWN, {COCKY: 0.33, SAFE: 0.33, LOSING: 0.33}

        features   = self._features()
        log_scores = {}
        for label, priors in self.PRIORS.items():
            log_scores[label] = sum(
                self._log_prob(features[f], mu, sig)
                for f, (mu, sig) in priors.items()
            )

        max_s = max(log_scores.values())
        exp_s = {k: math.exp(v - max_s) for k, v in log_scores.items()}
        total = sum(exp_s.values())
        conf  = {k: v / total for k, v in exp_s.items()}
        best  = max(conf, key=conf.get)
        return best, conf


# ===========================================================================
# COUNTER-STRATEGY PROFILES
# ===========================================================================

COUNTER_PROFILES = {
    # vs COCKY: slow-play, call down, don't bluff into them, let them spew
    COCKY: {
        'call_equity_adj':    -0.05,
        'bet_equity_adj':     +0.05,
        'raise_equity_adj':   +0.03,
        'bluff_multiplier':    0.40,
        'fold_threshold_adj':  0.03,
        'auction_bid_mult':    0.70,
    },
    # vs SAFE: be the aggressor, steal pots, overbid auctions to bully
    SAFE: {
        'call_equity_adj':    +0.02,
        'bet_equity_adj':     -0.08,
        'raise_equity_adj':   -0.05,
        'bluff_multiplier':    2.00,
        'fold_threshold_adj': -0.02,
        'auction_bid_mult':    1.40,
    },
    # vs LOSING: call wider to catch tilt-bluffs, don't overbet
    LOSING: {
        'call_equity_adj':    -0.04,
        'bet_equity_adj':     +0.02,
        'raise_equity_adj':   +0.02,
        'bluff_multiplier':    0.70,
        'fold_threshold_adj':  0.00,
        'auction_bid_mult':    1.00,
    },
    UNKNOWN: {
        'call_equity_adj':     0.00,
        'bet_equity_adj':      0.00,
        'raise_equity_adj':    0.00,
        'bluff_multiplier':    1.00,
        'fold_threshold_adj':  0.00,
        'auction_bid_mult':    1.00,
    },
}


# ===========================================================================
# VILLAIN RANGE
# ===========================================================================

def villain_range(pot, opp_auction_bid=None, personality=UNKNOWN):
    if personality == COCKY:
        base = "22+, A2s+, K2s+, Q5s+, J7s+, T7s+, 97s+, 87s, A2o+, K7o+, Q8o+, J9o+, T9o"
    elif personality == SAFE:
        base = "99+, ATs+, KQs, AJo+, KQo"
    elif personality == LOSING:
        base = "33+, A5s+, K8s+, Q9s+, JTs, ATo+, KJo+"
    else:
        if pot > 400:
            base = "77+, ATs+, KTs+, AJo+, KQo"
        elif pot > 200:
            base = "55+, A9s+, K9s+, QTs+, JTs, ATo+, KJo+, QJo"
        else:
            base = "22+, A2s+, K2s+, Q7s+, J8s+, T8s+, 98s, A2o+, K9o+, Q9o+, JTo"

    if opp_auction_bid is not None and opp_auction_bid > pot * 0.10:
        return eval7.HandRange("44+, A8s+, K9s+, QTs+, JTs, T9s, A9o+, KTo+, QJo")

    return eval7.HandRange(base)


# ===========================================================================
# EQUITY COMPUTATION
# ===========================================================================

def compute_equity(hole_cards, board, opp_known_card=None, pot=30,
                   opp_auction_bid=None, personality=UNKNOWN, trials=50):
    my_cards    = [eval7.Card(c) for c in hole_cards]
    board_cards = [eval7.Card(c) for c in board]

    if not board_cards:
        return preflop_equity(hole_cards)

    if opp_known_card:
        return _mc_known_card(my_cards, board_cards, eval7.Card(opp_known_card), trials=trials * 6)

    try:
        vrange = villain_range(pot, opp_auction_bid, personality)
        return eval7.py_hand_vs_range_monte_carlo(my_cards, vrange, board_cards, trials)
    except Exception:
        return _mc_uniform(my_cards, board_cards, trials=trials * 4)


def _mc_known_card(my_hand, board_cards, opp_c, trials=300):
    used      = set(str(c) for c in my_hand + board_cards + [opp_c])
    deck      = [c for c in eval7.Deck() if str(c) not in used]
    remaining = 5 - len(board_cards)
    wins = 0.0
    for _ in range(trials):
        random.shuffle(deck)
        full_board = board_cards + deck[1: 1 + remaining]
        ms = eval7.evaluate(my_hand + full_board)
        os = eval7.evaluate([opp_c, deck[0]] + full_board)
        if ms > os:    wins += 1.0
        elif ms == os: wins += 0.5
    return wins / trials


def _mc_uniform(my_hand, board_cards, trials=200):
    used      = set(str(c) for c in my_hand + board_cards)
    deck      = [c for c in eval7.Deck() if str(c) not in used]
    remaining = 5 - len(board_cards)
    wins = 0.0
    for _ in range(trials):
        random.shuffle(deck)
        full_board = board_cards + deck[2: 2 + remaining]
        ms = eval7.evaluate(my_hand + full_board)
        os = eval7.evaluate([deck[0], deck[1]] + full_board)
        if ms > os:    wins += 1.0
        elif ms == os: wins += 0.5
    return wins / trials


# ===========================================================================
# AUCTION BID: EV-DELTA + PERSONALITY + SESSION CONTEXT
# ===========================================================================

INFO_DISCOUNT    = 0.60
BID_ABS_MAX_BB   = 25
BID_STACK_FRAC   = 0.08
MIN_EQUITY_DELTA = 0.03


def _sample_equity_delta(hole_cards, board, pot, trials=40):
    """Estimate average |equity shift| from seeing one random opponent card."""
    my_cards    = [eval7.Card(c) for c in hole_cards]
    board_cards = [eval7.Card(c) for c in board]
    used        = set(str(c) for c in my_cards + board_cards)
    deck        = [c for c in eval7.Deck() if str(c) not in used]

    eq_blind = compute_equity(hole_cards, board, pot=pot, trials=30)

    total_delta = 0.0
    for _ in range(trials):
        random.shuffle(deck)
        eq_with = _mc_known_card(my_cards, board_cards, deck[0], trials=40)
        total_delta += abs(eq_with - eq_blind)

    return total_delta / trials, eq_blind


def compute_bid(hole_cards, board, pot, my_chips, opp_chips,
                personality=UNKNOWN, chip_lead=0,
                auction_wins=0, auction_total=0, time_budget=10.0):
    """
    Full-context auction bid:

      base_bid      = delta_ev * pot * INFO_DISCOUNT
      final_bid     = base_bid * personality_mult * session_mult * auction_hist_mult

    All capped at min(8% stack, 25BB).
    """
    trials = 40 if time_budget > 5.0 else 20
    avg_delta, eq_blind = _sample_equity_delta(hole_cards, board, pot, trials=trials)

    if avg_delta < MIN_EQUITY_DELTA:
        return 0

    certainty = 1.0 - abs(2.0 * eq_blind - 1.0)
    if certainty < 0.10:
        return 0

    effective_pot = max(pot, BIG_BLIND * 4)
    base_bid      = avg_delta * effective_pot * certainty * INFO_DISCOUNT

    # Personality multiplier
    pers_mult = COUNTER_PROFILES.get(personality, COUNTER_PROFILES[UNKNOWN])['auction_bid_mult']

    # Session chip lead multiplier
    if chip_lead < -100:
        session_mult = 1.25    # behind — need info edge more
    elif chip_lead < -50:
        session_mult = 1.10
    elif chip_lead > 200:
        session_mult = 0.80    # comfortably ahead — protect lead
    elif chip_lead > 100:
        session_mult = 0.90
    else:
        session_mult = 1.00

    # Auction win rate feedback (avoid over/underpaying)
    if auction_total >= 5:
        win_rate = auction_wins / auction_total
        if win_rate < 0.35:
            auction_mult = 1.15   # being outbid too often
        elif win_rate > 0.70:
            auction_mult = 0.88   # winning too cheaply, we can cut back
        else:
            auction_mult = 1.00
    else:
        auction_mult = 1.00

    final_bid = base_bid * pers_mult * session_mult * auction_mult

    # Hard variance caps
    stack_cap = my_chips * BID_STACK_FRAC
    abs_cap   = BID_ABS_MAX_BB * BIG_BLIND
    return max(0, int(min(final_bid, stack_cap, abs_cap)))


# ===========================================================================
# EV HELPERS — personality-aware fold probability
# ===========================================================================

BASE_FOLD_PROB = {'pre-flop': 0.45, 'flop': 0.38, 'turn': 0.28, 'river': 0.20}


def fold_probability(street, bet, pot, personality=UNKNOWN):
    base     = BASE_FOLD_PROB.get(street, 0.33)
    if pot <= 0:
        return base
    size_adj = max(0.0, (bet / pot - 0.33) * 0.20)
    pers_adj = {COCKY: -0.15, SAFE: +0.15, LOSING: +0.05}.get(personality, 0.0)
    return min(max(base + size_adj + pers_adj, 0.05), 0.85)


def ev_call(equity, pot, cost):
    return equity * (pot + cost) - cost


def ev_bet(equity, pot, bet, street='flop', personality=UNKNOWN):
    pot = max(pot, BIG_BLIND * 2)
    fp  = fold_probability(street, bet, pot, personality)
    return fp * pot + (1 - fp) * (equity * (pot + bet) - bet)


def best_bet_size(equity, pot, min_raise, max_raise, street='flop', personality=UNKNOWN):
    pot   = max(pot, BIG_BLIND * 2)
    fracs = [0.33, 0.5, 0.67, 0.75, 1.0]
    if personality == SAFE:
        fracs.append(1.5)    # vs SAFE, also try overbet sizing
    best_ev, best_amt = -1e9, min_raise
    for frac in fracs:
        amt = max(min_raise, min(max_raise, int(frac * pot)))
        ev  = ev_bet(equity, pot, amt, street, personality)
        if ev > best_ev:
            best_ev, best_amt = ev, amt
    return best_amt, best_ev


# ===========================================================================
# LIGHTWEIGHT HAND EVALUATION (NO MONTE CARLO)
# ===========================================================================

def _rank_idx(r: str) -> int:
    return RANK_ORDER.index(r)


def _postflop_bucket(hole_cards, board):
    """
    Cheap hand strength bucketing:
      0 = trash / very weak
      1 = weak pair or decent draw
      2 = top/mid pair or strong draw
      3 = two pair / set / flush / straight
      4 = full house+ (monsters)
    """
    if not board:
        return 0

    all_cards = hole_cards + board
    ranks = [c[0] for c in all_cards]
    suits = [c[1] for c in all_cards]
    hole_ranks = [c[0] for c in hole_cards]
    board_ranks = [c[0] for c in board]

    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1

    # Flush / flush draw
    flush = any(cnt >= 5 for cnt in suit_counts.values())
    flush_draw = any(cnt == 4 for cnt in suit_counts.values())

    # Straight-ish
    unique_ranks = sorted({_rank_idx(r) for r in ranks})
    consec = 1
    max_consec = 1
    for i in range(1, len(unique_ranks)):
        if unique_ranks[i] == unique_ranks[i - 1] + 1:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 1
    straight = max_consec >= 5
    straight_draw = max_consec == 4

    # Pairs / trips / full house / quads
    trips_plus = [r for r, c in rank_counts.items() if c >= 3]
    pairs = [r for r, c in rank_counts.items() if c == 2]
    full_house_or_better = bool(trips_plus and (len(trips_plus) >= 2 or pairs))

    # Do we participate in the made hand?
    my_pair_ranks = []
    for r in rank_counts:
        if r in hole_ranks and rank_counts[r] >= 2:
            my_pair_ranks.append(_rank_idx(r))

    pocket_pair = hole_ranks[0] == hole_ranks[1]
    if pocket_pair:
        my_pair_ranks.append(_rank_idx(hole_ranks[0]))

    board_high = max(_rank_idx(r) for r in board_ranks) if board_ranks else 0

    # Monster bucket
    if full_house_or_better or (flush and straight):
        return 4

    # Strong made hands
    if flush or straight or trips_plus:
        return 3

    # Pair quality
    if my_pair_ranks:
        best_pair = max(my_pair_ranks)
        if best_pair >= board_high:
            return 2  # top pair or overpair
        if best_pair >= board_high - 2:
            return 2  # mid pair
        return 1      # weak/bottom pair

    # Draws only
    if flush_draw or straight_draw:
        return 1

    # Completely whiffed
    return 0


def _revealed_card_adjustment(opp_card, board):
    """
    Very cheap adjustment: if revealed card looks scary relative to board, nudge down our equity a bit.
    """
    if not opp_card or not board:
        return 0.0
    r = opp_card[0]
    board_ranks = [c[0] for c in board]
    idx = _rank_idx(r)

    adj = 0.0
    if r in board_ranks:
        # Opp has trips / strong pair potential
        adj -= 0.07
    elif idx >= _rank_idx('Q'):
        # High card that's not on board still slightly scary
        adj -= 0.03
    elif idx <= _rank_idx('6'):
        # Tiny card is usually good news for us
        adj += 0.03
    return adj


def estimate_equity(hole_cards, board, opp_known_card=None):
    """
    Map (hand, board) → rough equity in [0,1] without Monte Carlo.
    """
    if not board:
        return preflop_equity(hole_cards)

    bucket = _postflop_bucket(hole_cards, board)
    # Base equities for buckets 0..4
    base_table = [0.25, 0.42, 0.58, 0.78, 0.93]
    equity = base_table[bucket]
    equity += _revealed_card_adjustment(opp_known_card, board)
    return max(0.05, min(0.97, equity))


# ===========================================================================
# SIMPLE AUCTION BIDDING (NO MONTE CARLO)
# ===========================================================================

def compute_simple_bid(hole_cards, board, pot, my_chips, opp_chips,
                       personality=UNKNOWN, chip_lead=0,
                       auction_wins=0, auction_total=0):
    """
    Very fast auction strategy:
      - Value of info highest when our equity is around 50%.
      - Scale by opponent personality and chip situation.
      - Hard caps as fraction of stack and in BBs.
    """
    if pot <= 0 or my_chips <= BIG_BLIND * 2:
        return 0

    equity = estimate_equity(hole_cards, board, opp_known_card=None)

    # Peak info value at equity ≈ 0.5
    info_factor = 1.0 - 2.0 * abs(equity - 0.5)  # in [0, 1]
    if info_factor < 0.1:
        return 0

    # Base: up to ~30% of pot when info_factor = 1
    base_bid = pot * (0.30 * info_factor)

    # Personality multiplier
    pers_mult = COUNTER_PROFILES.get(personality, COUNTER_PROFILES[UNKNOWN])['auction_bid_mult']

    # Session chip lead multiplier
    if chip_lead < -300:
        sess_mult = 1.30
    elif chip_lead < -150:
        sess_mult = 1.15
    elif chip_lead > 300:
        sess_mult = 0.75
    elif chip_lead > 150:
        sess_mult = 0.90
    else:
        sess_mult = 1.00

    # Auction history feedback
    if auction_total >= 5:
        win_rate = auction_wins / auction_total
        if win_rate < 0.35:
            hist_mult = 1.15  # being outbid too often
        elif win_rate > 0.70:
            hist_mult = 0.90  # probably overpaying
        else:
            hist_mult = 1.00
    else:
        hist_mult = 1.00

    raw_bid = base_bid * pers_mult * sess_mult * hist_mult

    # Hard caps to keep variance and time safe
    stack_cap = my_chips * 0.05
    abs_cap = 15 * BIG_BLIND
    bid = int(min(raw_bid, stack_cap, abs_cap))

    return max(0, bid)


# ===========================================================================
# HAND STATE TRACKER
# ===========================================================================

class HandState:
    """
    Tracks everything observable about an opponent in one hand.
    Flushed into PersonalityClassifier on hand end.
    """
    def __init__(self, my_hand):
        self.my_hand         = my_hand     # our hole cards this hand
        self.opp_actions     = []          # list of action_type strings
        self.opp_bets        = []          # list of (bet, pot) tuples
        self.opp_vpip        = False
        self.opp_auction_bid = None
        self.opp_folded      = False

    def record_opp_action(self, action_type, bet=None, pot=None):
        self.opp_actions.append(action_type)
        if action_type in ('bet', 'raise') and bet is not None and pot:
            self.opp_bets.append((bet, pot))
        if action_type in ('call', 'raise', 'bet'):
            self.opp_vpip = True
        if action_type == 'fold':
            self.opp_folded = True


# ===========================================================================
# MAIN BOT
# ===========================================================================

class Player(BaseBot):

    def __init__(self):
        # Persistent session state
        self.classifier        = PersonalityClassifier()
        self.session_my_chips  = STARTING_STACK
        self.session_opp_chips = STARTING_STACK
        self.auction_total     = 0
        self.auction_wins      = 0
        self._auction_won_this_hand = False

        # Long-run villain bluff tracking (from showdowns)
        self.opp_showdowns_seen = 0
        self.opp_bluffs_seen    = 0
        self.opp_bluff_ratio    = 0.20  # prior: 20% of big pots are bluffs

        # Per-hand state
        self.opp_known_card  = None
        self.opp_auction_bid = None
        self.hand_state      = None

    # ------------------------------------------------------------------
    # Hand lifecycle
    # ------------------------------------------------------------------

    def on_hand_start(self, game_info, current_state):
        self.opp_known_card         = None
        self.opp_auction_bid        = None
        self._auction_won_this_hand = False
        self.hand_state             = HandState(my_hand=current_state.my_hand)

    def on_hand_end(self, game_info, current_state):
        hs = self.hand_state
        if hs is None:
            return

        # Flush observations into classifier
        self.classifier.record_hand_vpip(hs.opp_vpip)
        for at in hs.opp_actions:
            self.classifier.record_action(at)

        # Approximate faced-bet events from action sequence
        prev = None
        for at in hs.opp_actions:
            if prev in ('bet', 'raise') and at in ('call', 'fold'):
                self.classifier.record_faced_bet(did_fold=(at == 'fold'))
            prev = at

        if hs.opp_auction_bid is not None:
            self.classifier.record_auction_bid(
                hs.opp_auction_bid, max(BIG_BLIND * 4, 30)
            )

        for bet, pot in hs.opp_bets:
            self.classifier.record_bet_size(bet, pot)

        # Early-fold detection: if hand ended before river and we got paid,
        # opponent almost surely folded to aggression.
        if current_state.street != 'river' and current_state.payoff > 0:
            self.classifier.record_faced_bet(did_fold=True)

        # Track showdown bluffiness (heuristic, no eval7).
        opp_cards = current_state.opp_revealed_cards
        board = current_state.board
        if opp_cards and len(opp_cards) >= 2 and len(board) >= 3:
            self.opp_showdowns_seen += 1
            opp_bucket = _postflop_bucket(opp_cards, board)
            pot_size = current_state.pot
            is_bluff = (opp_bucket <= 1 and pot_size >= 200)
            if is_bluff:
                self.opp_bluffs_seen += 1

            prior_weight = max(5.0, 15.0 - 0.5 * self.opp_showdowns_seen)
            self.opp_bluff_ratio = (
                (prior_weight * 0.20) + self.opp_bluffs_seen
            ) / (prior_weight + self.opp_showdowns_seen)

        # Update chip tracking for session context
        self.session_my_chips  = current_state.my_chips
        self.session_opp_chips = current_state.opp_chips

    # ------------------------------------------------------------------
    # Main decision
    # ------------------------------------------------------------------

    def get_move(self, game_info, current_state):
        street       = current_state.street
        hole_cards   = current_state.my_hand
        board        = current_state.board
        pot          = current_state.pot
        my_chips     = current_state.my_chips
        opp_chips    = current_state.opp_chips
        cost_to_call = current_state.cost_to_call
        time_bank    = game_info.time_bank

        # Capture opp auction bid
        if hasattr(current_state, 'opp_auction_bid') and current_state.opp_auction_bid is not None:
            self.opp_auction_bid = current_state.opp_auction_bid
            if self.hand_state:
                self.hand_state.opp_auction_bid = self.opp_auction_bid

        # Capture revealed card (we won the auction)
        if current_state.opp_revealed_cards and not self.opp_known_card:
            self.opp_known_card = current_state.opp_revealed_cards[0]
            if not self._auction_won_this_hand:
                self.auction_wins           += 1
                self._auction_won_this_hand  = True

        # Classify opponent personality
        personality, _confidence = self.classifier.classify()

        # Session chip lead (positive = we're ahead)
        chip_lead = my_chips - opp_chips

        # ---- Auction ----
        if street == 'auction':
            self.auction_total += 1
            bid = compute_simple_bid(
                hole_cards, board, pot, my_chips, opp_chips,
                personality   = personality,
                chip_lead     = chip_lead,
                auction_wins  = self.auction_wins,
                auction_total = self.auction_total,
            )
            return ActionBid(bid)

        # ---- Betting ----
        equity = estimate_equity(
            hole_cards, board,
            opp_known_card=self.opp_known_card,
        )

        return self._act(current_state, equity, street, pot, cost_to_call, personality)

    # ------------------------------------------------------------------
    # Action logic
    # ------------------------------------------------------------------

    def _act(self, state, equity, street, pot, cost_to_call, personality):
        profile = COUNTER_PROFILES.get(personality, COUNTER_PROFILES[UNKNOWN])
        fold_adj = profile['fold_threshold_adj']
        raise_adj = profile['raise_equity_adj']
        bet_adj = profile['bet_equity_adj']
        bluff_m = profile['bluff_multiplier']

        # Global bluffiness adjustment from showdown data
        bluff_bonus = max(-0.15, min(0.15, self.opp_bluff_ratio - 0.20))
        eff_equity = max(0.0, min(1.0, equity + bluff_bonus))

        # ---- Facing a bet ----
        if cost_to_call > 0:
            if self.hand_state:
                self.hand_state.record_opp_action('bet', bet=cost_to_call, pot=pot)

            # Simple pot-odds based decision with personality tweak
            total_pot = pot + cost_to_call
            pot_odds = cost_to_call / max(total_pot, 1)
            continue_threshold = pot_odds - fold_adj

            if eff_equity + fold_adj < pot_odds:
                return ActionFold() if state.can_act(ActionFold) else ActionCheck()

            # Occasionally raise for value with strong hands
            if state.can_act(ActionRaise) and eff_equity > (0.70 + raise_adj):
                min_r, max_r = state.raise_bounds
                target_frac = 0.75 if eff_equity > 0.85 else 0.5
                amount = int(pot * target_frac)
                amount = max(min_r, min(max_r, amount))
                return ActionRaise(amount)

            return ActionCall()

        # ---- Open action (no bet facing us) ----
        if state.can_act(ActionRaise):
            # Bluff multiplier: higher bluff_m = we lower the equity threshold to bet
            raw_thresh = 0.55 + bet_adj
            effective_thresh = max(0.35, raw_thresh / max(bluff_m, 0.1))

            if eff_equity > effective_thresh:
                min_r, max_r = state.raise_bounds
                target_frac = 0.5 if eff_equity < 0.75 else 0.75
                amount = int(pot * target_frac)
                amount = max(min_r, min(max_r, amount))
                return ActionRaise(amount)

        # Never fold for free
        return ActionCheck()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
