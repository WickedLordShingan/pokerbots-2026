'''
Sneak Peek Hold'em Bot — v8 (COMPETITION BUILD)
=================================================

Time budget: 30s total / 1000 rounds / ~5 queries per round = ~6ms per query.
v7 used ~0.1ms. This version spends ~2-4ms where it matters most.

Where we spend computation:
  1. EQUITY: Monte Carlo via eval7 on flop/turn/river (street-cached — run once
     per street, reused for all actions on that street). On preflop: instant
     lookup table. When we know opp's card: full MC against their exact hand.
     Budget: ~2ms once per street.

  2. BET SIZING: EV-maximising size search across 6 candidate fractions.
     Uses personality-aware fold probability model. Picks highest EV size
     rather than mechanically using 50%/67%.
     Budget: ~0.2ms (6 multiplications).

  3. TIME-ADAPTIVE MC ITERATIONS: scales iterations based on remaining time bank
     so we always have a safety reserve. Never runs MC when time_bank < 2s.

  4. AUCTION: still O(1) — auction happens once per hand, MC not needed.

Everything else (personality classify, hand state, bluff tracking) unchanged.
'''

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot
from pkbot.states import BIG_BLIND, SMALL_BLIND, STARTING_STACK

import math
import json
import os
import random

try:
    import eval7
    EVAL7_AVAILABLE = True
except ImportError:
    EVAL7_AVAILABLE = False

# =============================================================================
# PREFLOP EQUITY TABLE (instant lookup)
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
    suited  = 's' if h1[1] == h2[1] else 'o'
    ranks   = sorted([h1[0], h2[0]], key=_rank_idx, reverse=True)
    return PREFLOP_EQUITY.get((ranks[0], ranks[1], suited), 0.48)


# =============================================================================
# POSTFLOP BUCKET (fallback when eval7 unavailable or time is short)
# 0=trash 1=weak-pair/draw 2=top-mid-pair 3=flush/straight/trips 4=monsters
# =============================================================================

def _postflop_bucket(hole_cards, board):
    if not board: return 0
    all_cards   = hole_cards + board
    ranks       = [c[0] for c in all_cards]
    suits       = [c[1] for c in all_cards]
    hole_ranks  = [c[0] for c in hole_cards]
    board_ranks = [c[0] for c in board]

    rc = {}
    for r in ranks: rc[r] = rc.get(r, 0) + 1
    sc = {}
    for s in suits: sc[s] = sc.get(s, 0) + 1

    flush      = any(v >= 5 for v in sc.values())
    flush_draw = any(v == 4 for v in sc.values())

    uniq = sorted({_rank_idx(r) for r in ranks})
    c = mc = 1
    for i in range(1, len(uniq)):
        if uniq[i] == uniq[i-1]+1: c += 1; mc = max(mc, c)
        else: c = 1
    straight      = mc >= 5
    straight_draw = mc == 4

    trips  = [r for r,n in rc.items() if n >= 3]
    pairs  = [r for r,n in rc.items() if n == 2]
    fhplus = bool(trips and (len(trips) >= 2 or pairs))

    my_pairs = [_rank_idx(r) for r in rc if r in hole_ranks and rc[r] >= 2]
    if len(hole_ranks) >= 2 and hole_ranks[0] == hole_ranks[1]:
        my_pairs.append(_rank_idx(hole_ranks[0]))
    bh = max((_rank_idx(r) for r in board_ranks), default=0)

    if fhplus or (flush and straight): return 4
    if flush or straight or trips:     return 3
    if my_pairs:
        best = max(my_pairs)
        return 2 if (best >= bh or best >= bh-2) else 1
    if flush_draw or straight_draw:    return 1
    return 0

_BUCKET_EQ = [0.23, 0.41, 0.58, 0.76, 0.92]

def _bucket_equity(hole_cards, board, opp_known_card=None):
    eq = _BUCKET_EQ[_postflop_bucket(hole_cards, board)]
    if opp_known_card:
        r = opp_known_card[0]
        idx = _rank_idx(r)
        if r in [c[0] for c in board]: eq -= 0.08
        elif idx >= _rank_idx('Q'):    eq -= 0.04
        elif idx <= _rank_idx('5'):    eq += 0.03
    return max(0.05, min(0.97, eq))


# =============================================================================
# MONTE CARLO EQUITY  (eval7-powered, street-cached)
# =============================================================================

def _mc_vs_range(my_cards, board_cards, v_range, iters):
    """MC against a HandRange. Returns equity in [0,1]."""
    try:
        return eval7.py_hand_vs_range_monte_carlo(my_cards, v_range, board_cards, iters)
    except Exception:
        return None

def _mc_known_card(my_cards, board_cards, opp_card, iters):
    """
    MC when we know one of opponent's cards exactly.
    Complete remaining board cards and deal opp one random card from
    the remaining deck to form their full hand.
    """
    used    = set(str(c) for c in my_cards + board_cards + [opp_card])
    deck    = [c for c in eval7.Deck() if str(c) not in used]
    remain  = 5 - len(board_cards)
    wins    = 0.0
    for _ in range(iters):
        random.shuffle(deck)
        full_board = board_cards + deck[1:1+remain]
        ms = eval7.evaluate(my_cards + full_board)
        os = eval7.evaluate([opp_card, deck[0]] + full_board)
        if ms > os:    wins += 1.0
        elif ms == os: wins += 0.5
    return wins / iters

def _villain_range(pot, personality, opp_auction_bid):
    """Return eval7.HandRange appropriate for current context."""
    COCKY   = 'COCKY';  SAFE = 'SAFE';  LOSING = 'LOSING'
    if personality == COCKY:
        base = "22+,A2s+,K2s+,Q5s+,J7s+,T7s+,97s+,87s,A2o+,K7o+,Q8o+,J9o+,T9o"
    elif personality == SAFE:
        base = "99+,ATs+,KQs,AJo+,KQo"
    elif personality == LOSING:
        base = "33+,A5s+,K8s+,Q9s+,JTs,ATo+,KJo+"
    else:
        # Unknown — range based on pot size (larger pot = tighter range)
        if pot > 400:   base = "77+,ATs+,KTs+,AJo+,KQo"
        elif pot > 200: base = "55+,A9s+,K9s+,QTs+,JTs,ATo+,KJo+,QJo"
        else:           base = "22+,A2s+,K2s+,Q7s+,J8s+,T8s+,98s,A2o+,K9o+,Q9o+"

    # If opp bid high in auction, they likely have a medium-strong hand
    if opp_auction_bid and opp_auction_bid > 15:
        base = "44+,A8s+,K9s+,QTs+,JTs,T9s,A9o+,KTo+,QJo"

    try:
        return eval7.HandRange(base)
    except Exception:
        return eval7.HandRange("22+,A2s+,K2s+")

def compute_mc_equity(hole_cards, board, opp_known_card, pot,
                      personality, opp_auction_bid, iters):
    """
    Full Monte Carlo equity. Returns float in [0,1] or None on failure.
    Only called when eval7 is available and time_bank is healthy.
    """
    try:
        my_cards    = [eval7.Card(c) for c in hole_cards]
        board_cards = [eval7.Card(c) for c in board]

        if opp_known_card:
            opp_c = eval7.Card(opp_known_card)
            return _mc_known_card(my_cards, board_cards, opp_c, iters)

        v_range = _villain_range(pot, personality, opp_auction_bid)
        result  = _mc_vs_range(my_cards, board_cards, v_range, iters)
        return result
    except Exception:
        return None


# =============================================================================
# UNIFIED EQUITY ESTIMATOR
# =============================================================================

def estimate_equity(hole_cards, board, opp_known_card=None,
                    pot=30, personality='UNKNOWN', opp_auction_bid=None,
                    time_bank=30.0, street_cache=None):
    """
    Returns best available equity estimate given time constraints.

    street_cache: dict with keys 'street', 'equity' — reuses cached value
                  if street hasn't changed since last call.

    Priority:
      preflop → instant lookup table
      flop/turn/river, time ok → MC (result stored in street_cache)
      flop/turn/river, time low → bucket fallback
    """
    if not board:
        return preflop_equity(hole_cards)

    # Return cached value if street hasn't changed
    if street_cache is not None:
        cached_eq = street_cache.get('equity')
        if cached_eq is not None:
            return cached_eq

    # Determine MC iteration count from time budget
    # Reserve 2s as safety floor; scale iterations with available time
    if not EVAL7_AVAILABLE or time_bank < 2.0:
        eq = _bucket_equity(hole_cards, board, opp_known_card)
    else:
        if time_bank > 20.0:   iters = 200
        elif time_bank > 10.0: iters = 150
        elif time_bank > 5.0:  iters = 100
        elif time_bank > 3.0:  iters = 60
        else:                  iters = 30

        eq = compute_mc_equity(hole_cards, board, opp_known_card,
                                pot, personality, opp_auction_bid, iters)
        if eq is None:
            eq = _bucket_equity(hole_cards, board, opp_known_card)

    eq = max(0.05, min(0.97, eq))
    if street_cache is not None:
        street_cache['equity'] = eq
    return eq


# =============================================================================
# EV-MAXIMISING BET SIZER
# =============================================================================

# Fold probability model: how likely is opponent to fold to a bet of size `bet`
# into pot `pot` given street and personality?
_BASE_FOLD = {'pre-flop': 0.42, 'flop': 0.36, 'turn': 0.27, 'river': 0.19}
_PERS_FOLD = {'COCKY': -0.14, 'SAFE': +0.14, 'LOSING': +0.04, 'UNKNOWN': 0.0}

def fold_prob(street, bet, pot, personality):
    base     = _BASE_FOLD.get(street, 0.30)
    size_adj = max(0.0, (bet / max(pot, 1) - 0.33) * 0.22)
    pers_adj = _PERS_FOLD.get(personality, 0.0)
    return min(max(base + size_adj + pers_adj, 0.04), 0.88)

def ev_bet(equity, pot, bet, street, personality):
    fp = fold_prob(street, bet, pot, personality)
    return fp * pot + (1 - fp) * (equity * (pot + bet) - bet)

def ev_call(equity, pot, cost):
    return equity * (pot + cost) - cost

def best_raise_amount(equity, pot, min_r, max_r, street, personality):
    """
    Search 6 candidate bet fractions, return the one with highest EV.
    Also tries overbet (1.5x pot) vs SAFE opponents on late streets.
    """
    fracs = [0.33, 0.50, 0.67, 0.75, 1.0]
    if personality == 'SAFE' and street in ('turn', 'river') and equity > 0.58:
        fracs.append(1.5)   # SAFE players fold to overbets

    best_ev  = -1e9
    best_amt = min_r
    pot_eff  = max(pot, BIG_BLIND * 2)

    for frac in fracs:
        amt = int(frac * pot_eff)
        amt = max(min_r, min(max_r, amt))
        ev  = ev_bet(equity, pot_eff, amt, street, personality)
        if ev > best_ev:
            best_ev  = ev
            best_amt = amt

    return best_amt, best_ev


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
        self.hands_seen = 0;        self.vpip_count = 0
        self.total_actions = 0;     self.aggressive_actions = 0
        self.times_faced_bet = 0;   self.folds_to_bet = 0
        self.auction_ratios = [];   self.bet_size_ratios = []
        self._cache = UNKNOWN;      self._cache_valid = False

    def record_hand_vpip(self, v):
        self.hands_seen += 1
        if v: self.vpip_count += 1
        self._cache_valid = False

    def record_action(self, a):
        self.total_actions += 1
        if a in ('bet','raise'): self.aggressive_actions += 1
        self._cache_valid = False

    def record_faced_bet(self, folded):
        self.times_faced_bet += 1
        if folded: self.folds_to_bet += 1
        self._cache_valid = False

    def record_auction_ratio(self, bid, pot):
        if pot > 0: self.auction_ratios.append(min(bid/pot, 3.0))
        self._cache_valid = False

    def record_bet_sizing(self, bet, pot):
        if pot > 0: self.bet_size_ratios.append(min(bet/pot, 3.0))
        self._cache_valid = False

    @staticmethod
    def _lp(x, mu, sig):
        return -0.5 * ((x-mu)/sig)**2 - math.log(sig)

    def classify(self):
        if self._cache_valid: return self._cache
        if self.hands_seen < MIN_HANDS_TO_CLASSIFY: return UNKNOWN
        sd = lambda n,d: n/d if d > 0 else 0.5
        f = {
            'vpip':          sd(self.vpip_count, self.hands_seen),
            'aggression':    sd(self.aggressive_actions, max(self.total_actions,1)),
            'fold_to_bet':   sd(self.folds_to_bet, max(self.times_faced_bet,1)),
            'auction_ratio': sum(self.auction_ratios)/len(self.auction_ratios) if self.auction_ratios else 0.10,
            'bet_sizing':    sum(self.bet_size_ratios)/len(self.bet_size_ratios) if self.bet_size_ratios else 0.55,
        }
        scores = {lb: sum(self._lp(f[feat],mu,sg) for feat,(mu,sg) in pr.items())
                  for lb,pr in self.PRIORS.items()}
        best = max(scores, key=scores.get)
        self._cache = best; self._cache_valid = True
        return best


# =============================================================================
# COUNTER-STRATEGY PROFILES
# =============================================================================

COUNTER_PROFILES = {
    COCKY:   {'call_eq_adj':-0.05,'open_eq_adj':+0.05,'raise_eq_adj':+0.03,
              'bluff_div':2.5,  'fold_adj':0.02,  'auction_mult':0.70},
    SAFE:    {'call_eq_adj':+0.03,'open_eq_adj':-0.08,'raise_eq_adj':-0.05,
              'bluff_div':0.60, 'fold_adj':-0.02, 'auction_mult':1.40},
    LOSING:  {'call_eq_adj':-0.04,'open_eq_adj':+0.02,'raise_eq_adj':+0.01,
              'bluff_div':1.30, 'fold_adj':0.00,  'auction_mult':1.00},
    UNKNOWN: {'call_eq_adj':0.00, 'open_eq_adj':0.00, 'raise_eq_adj':0.00,
              'bluff_div':1.00, 'fold_adj':0.00,  'auction_mult':1.00},
}


# =============================================================================
# AUCTION STRATEGY (second-price / sniper model)
# ENGINE: highest bidder wins and pays opponent's (lower) bid.
# =============================================================================

INFO_VALUE_FRAC = 0.28
SNIPE_DELTA     = 2
MAX_BID_FRAC    = 0.06
MAX_BID_BB      = 18

def _predict_opp_bid(history):
    if not history: return 0.0
    weights = [1.25**i for i in range(len(history))]
    return sum(b*w for b,w in zip(history, weights)) / sum(weights)

def compute_auction_bid(hole_cards, board, pot, my_chips,
                        personality, chip_lead,
                        opp_bid_history, auction_wins, auction_total,
                        time_bank):
    # Use MC for auction equity too if we have time — more accurate than bucket
    eq = estimate_equity(hole_cards, board,
                         pot=pot, personality=personality,
                         time_bank=time_bank, street_cache=None)
    uncertainty = 1.0 - abs(2.0*eq - 1.0)
    if uncertainty < 0.12:
        return 0

    effective_pot = max(pot, BIG_BLIND * 3)
    info_value    = uncertainty * effective_pot * INFO_VALUE_FRAC

    pers_mult = COUNTER_PROFILES[personality]['auction_mult']

    if chip_lead < -300:    sess = 1.30
    elif chip_lead < -100:  sess = 1.15
    elif chip_lead > 300:   sess = 0.75
    elif chip_lead > 100:   sess = 0.88
    else:                   sess = 1.00

    if auction_total >= 6:
        wr = auction_wins / auction_total
        wm = 1.20 if wr < 0.30 else (0.85 if wr > 0.75 else 1.00)
    else:
        wm = 1.00

    max_pay       = info_value * pers_mult * sess * wm
    predicted_opp = _predict_opp_bid(opp_bid_history)
    snipe         = predicted_opp + SNIPE_DELTA

    if snipe <= max_pay:      chosen = snipe
    elif not opp_bid_history: chosen = max_pay
    else:                     chosen = 0

    cap = min(my_chips * MAX_BID_FRAC, MAX_BID_BB * BIG_BLIND)
    return max(0, int(min(chosen, cap)))


# =============================================================================
# HAND STATE TRACKER
# =============================================================================

class HandState:
    __slots__ = ['my_hand','opp_actions','opp_bets','opp_vpip',
                 'opp_auction_bid','last_aggressor']
    def __init__(self, my_hand):
        self.my_hand         = my_hand
        self.opp_actions     = []
        self.opp_bets        = []
        self.opp_vpip        = False
        self.opp_auction_bid = None
        self.last_aggressor  = ''

    def opp_bet(self, a, p):   self.opp_actions.append('bet');   self.opp_bets.append((a,p)); self.opp_vpip=True
    def opp_raise(self, a, p): self.opp_actions.append('raise'); self.opp_bets.append((a,p)); self.opp_vpip=True
    def opp_call(self):        self.opp_actions.append('call');  self.opp_vpip=True
    def opp_check(self):       self.opp_actions.append('check')
    def opp_fold(self):        self.opp_actions.append('fold')


# =============================================================================
# MAIN BOT
# =============================================================================

class Player(BaseBot):

    def __init__(self):
        self.classifier      = PersonalityClassifier()
        self.opp_bid_history = []
        self.auction_total   = 0
        self.auction_wins    = 0
        self.opp_showdowns   = 0
        self.opp_bluffs      = 0
        self.opp_bluff_ratio = self._load_learned_params()

        # Per-hand state
        self.opp_known_card   = None
        self.opp_auction_bid  = None
        self.hand_state       = None
        self._prev_pot        = 0
        self._prev_opp_wager  = 0
        self._counted_win     = False

        # Street equity cache — reuse MC result across multiple actions
        # on the same street: {'street': str, 'equity': float}
        self._eq_cache        = {}

    # ---- Learned params ----

    def _load_learned_params(self):
        # ----------------------------------------------------------------
        # BAKED-IN TRAINED VALUES — auto-updated by train_from_logs.py
        # ----------------------------------------------------------------
        _BAKED_BLUFF_PRIOR = 0.6846

        _BAKED_PRIORS = {
            COCKY:  {'vpip':(0.80,0.12),'aggression':(0.65,0.12),
                     'fold_to_bet':(0.20,0.10),'auction_ratio':(0.22,0.10),'bet_sizing':(0.80,0.20)},
            SAFE:   {'vpip':(0.30,0.10),'aggression':(0.25,0.10),
                     'fold_to_bet':(0.65,0.12),'auction_ratio':(0.05,0.04),'bet_sizing':(0.45,0.15)},
            LOSING: {'vpip':(0.60,0.18),'aggression':(0.50,0.20),
                     'fold_to_bet':(0.40,0.18),'auction_ratio':(0.15,0.12),'bet_sizing':(0.65,0.30)},
        }
        # ----------------------------------------------------------------

        PersonalityClassifier.PRIORS = _BAKED_PRIORS
        bluff = _BAKED_BLUFF_PRIOR

        path = os.path.join(os.path.dirname(__file__), 'learned_params.json')
        try:
            with open(path) as f:
                data = json.load(f)
            bluff = max(0.0, min(1.0, float(data.get('opp_bluff_prior', bluff))))
            cal   = data.get('calibrated_priors', {})
            valid_l = {COCKY, SAFE, LOSING}
            valid_f = {'vpip','aggression','fold_to_bet','auction_ratio','bet_sizing'}
            if cal and set(cal.keys()) == valid_l:
                new_p = {}; ok = True
                for lb, fm in cal.items():
                    new_p[lb] = {}
                    for feat, val in fm.items():
                        if feat not in valid_f: continue
                        if isinstance(val,(list,tuple)) and len(val)==2:
                            mu,sig = float(val[0]),float(val[1])
                            if 0<=mu<=1 and 0.01<=sig<=1:
                                new_p[lb][feat]=(mu,sig)
                    for feat in valid_f:
                        if feat not in new_p[lb]:
                            new_p[lb][feat] = _BAKED_PRIORS[lb][feat]
                    if len(new_p[lb]) != len(valid_f): ok=False; break
                if ok: PersonalityClassifier.PRIORS = new_p
        except Exception:
            pass

        return bluff

    # ---- Hand lifecycle ----

    def on_hand_start(self, game_info, current_state):
        self.opp_known_card  = None
        self.opp_auction_bid = None
        self.hand_state      = HandState(list(current_state.my_hand))
        self._prev_pot       = current_state.pot
        self._prev_opp_wager = current_state.opp_wager
        self._counted_win    = False
        self._eq_cache       = {}   # clear street equity cache each hand

    def on_hand_end(self, game_info, current_state):
        hs = self.hand_state
        if not hs: return

        self.classifier.record_hand_vpip(hs.opp_vpip)
        for a in hs.opp_actions:
            self.classifier.record_action(a)

        prev = None
        for a in hs.opp_actions:
            if prev in ('bet','raise') and a in ('call','fold'):
                self.classifier.record_faced_bet(did_fold=(a=='fold'))
            prev = a

        if hs.last_aggressor=='us' and current_state.payoff>0 and current_state.street!='river':
            self.classifier.record_faced_bet(did_fold=True)

        if hs.opp_auction_bid is not None:
            self.classifier.record_auction_ratio(hs.opp_auction_bid,
                                                  max(self._prev_pot, BIG_BLIND*3))
        for bet, pot in hs.opp_bets:
            self.classifier.record_bet_sizing(bet, pot)

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
            if current_state.street == 'auction': return ActionBid(0)
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
        time_bank    = game_info.time_bank

        # Reset equity cache when street changes
        if self._eq_cache.get('street') != street:
            self._eq_cache = {'street': street, 'equity': None}

        # ---- Read opp auction bid from state.bids ----
        bids = getattr(current_state, 'bids', None)
        if (bids and len(bids)==2 and all(b is not None for b in bids)
                and self.opp_auction_bid is None):
            my_idx  = 1 if is_bb else 0
            opp_bid = bids[1 - my_idx]
            if opp_bid is not None:
                self.opp_auction_bid = int(opp_bid)
                if self.hand_state:
                    self.hand_state.opp_auction_bid = self.opp_auction_bid
                self.opp_bid_history.append(self.opp_auction_bid)
                self.classifier.record_auction_ratio(
                    self.opp_auction_bid, max(self._prev_pot, BIG_BLIND*3))

        # ---- Capture revealed card ----
        if current_state.opp_revealed_cards and not self.opp_known_card:
            self.opp_known_card = current_state.opp_revealed_cards[0]
            # Revealed card changes equity — invalidate cache
            self._eq_cache = {'street': street, 'equity': None}
            if not self._counted_win:
                self.auction_wins += 1
                self._counted_win  = True

        # ---- Detect opp actions via wager delta ----
        if (self.hand_state and street not in ('auction','pre-flop')
                and opp_wager > self._prev_opp_wager):
            delta = opp_wager - self._prev_opp_wager
            if self._prev_opp_wager == 0: self.hand_state.opp_bet(delta, self._prev_pot)
            else:                          self.hand_state.opp_raise(delta, self._prev_pot)
        self._prev_pot       = pot
        self._prev_opp_wager = opp_wager

        personality = self.classifier.classify()
        chip_lead   = my_chips - opp_chips

        # ==================================================================
        # AUCTION
        # ==================================================================
        if street == 'auction':
            self.auction_total += 1
            bid = compute_auction_bid(
                hole_cards, board, pot, my_chips,
                personality, chip_lead,
                self.opp_bid_history, self.auction_wins, self.auction_total,
                time_bank,
            )
            return ActionBid(bid)

        # ==================================================================
        # BETTING — compute equity once, cache for rest of street
        # ==================================================================
        equity = estimate_equity(
            hole_cards, board,
            opp_known_card  = self.opp_known_card,
            pot             = pot,
            personality     = personality,
            opp_auction_bid = self.opp_auction_bid,
            time_bank       = time_bank,
            street_cache    = self._eq_cache,
        )

        # Adjust for opponent bluff tendency
        bluff_bonus = max(-0.12, min(0.12, self.opp_bluff_ratio - 0.20))
        adj_equity  = max(0.05, min(0.97, equity + bluff_bonus))

        return self._decide(current_state, adj_equity, equity,
                            street, pot, cost_to_call, personality,
                            my_chips, chip_lead)

    # ---- Decision logic ----

    def _decide(self, state, adj_eq, raw_eq, street, pot,
                cost_to_call, personality, my_chips, chip_lead):
        p         = COUNTER_PROFILES[personality]
        fold_adj  = p['fold_adj']
        call_adj  = p['call_eq_adj']
        open_adj  = p['open_eq_adj']
        raise_adj = p['raise_eq_adj']
        bdiv      = p['bluff_div']

        # ---- Facing a bet ----
        if cost_to_call > 0:
            pot_odds    = cost_to_call / max(pot + cost_to_call, 1)
            call_thresh = pot_odds + fold_adj - call_adj

            if adj_eq < call_thresh:
                return ActionFold() if state.can_act(ActionFold) else ActionCheck()

            # Value raise: use EV comparison to decide
            if state.can_act(ActionRaise):
                min_r, max_r = state.raise_bounds
                raise_thresh = 0.68 + raise_adj
                if adj_eq > raise_thresh:
                    amt, rev = best_raise_amount(adj_eq, pot, min_r, max_r, street, personality)
                    ecall    = ev_call(adj_eq, pot, cost_to_call)
                    # Only raise if it's meaningfully better than calling
                    if rev > ecall * 1.05:
                        if self.hand_state: self.hand_state.last_aggressor = 'us'
                        return ActionRaise(amt)

            return ActionCall()

        # ---- Open action ----
        if state.can_act(ActionRaise):
            base_thresh  = 0.56 + open_adj
            bluff_thresh = max(0.35, base_thresh / max(bdiv, 0.1))

            if adj_eq > bluff_thresh:
                min_r, max_r = state.raise_bounds
                amt, _       = best_raise_amount(adj_eq, pot, min_r, max_r, street, personality)
                if self.hand_state: self.hand_state.last_aggressor = 'us'
                return ActionRaise(amt)

        return ActionCheck()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
