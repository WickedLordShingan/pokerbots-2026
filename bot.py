'''
The 1500 ELO Smart Rock (0.015s Speed Tune)
Features: Smart Auction Sniping, Bluff Tracking, 50-Iteration Cap.
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import random
import eval7
import traceback

# THE COMPLETE 169-HAND PREFLOP EQUITY DICTIONARY
PREFLOP_EQUITY = {
    ('A', 'A', 'o'): 0.85, ('K', 'K', 'o'): 0.82, ('Q', 'Q', 'o'): 0.80, ('J', 'J', 'o'): 0.77,
    ('T', 'T', 'o'): 0.75, ('9', '9', 'o'): 0.72, ('8', '8', 'o'): 0.69, ('7', '7', 'o'): 0.66,
    ('6', '6', 'o'): 0.63, ('5', '5', 'o'): 0.60, ('4', '4', 'o'): 0.57, ('3', '3', 'o'): 0.54,
    ('2', '2', 'o'): 0.50,
    ('A', 'K', 's'): 0.67, ('A', 'Q', 's'): 0.66, ('A', 'J', 's'): 0.65, ('A', 'T', 's'): 0.64,
    ('A', '9', 's'): 0.62, ('A', '8', 's'): 0.61, ('A', '7', 's'): 0.60, ('A', '6', 's'): 0.59,
    ('A', '5', 's'): 0.59, ('A', '4', 's'): 0.58, ('A', '3', 's'): 0.58, ('A', '2', 's'): 0.57,
    ('A', 'K', 'o'): 0.65, ('A', 'Q', 'o'): 0.64, ('A', 'J', 'o'): 0.63, ('A', 'T', 'o'): 0.62,
    ('A', '9', 'o'): 0.60, ('A', '8', 'o'): 0.59, ('A', '7', 'o'): 0.58, ('A', '6', 'o'): 0.57,
    ('A', '5', 'o'): 0.57, ('A', '4', 'o'): 0.56, ('A', '3', 'o'): 0.55, ('A', '2', 'o'): 0.54,
    ('K', 'Q', 's'): 0.63, ('K', 'J', 's'): 0.62, ('K', 'T', 's'): 0.61, ('K', '9', 's'): 0.58,
    ('K', '8', 's'): 0.57, ('K', '7', 's'): 0.56, ('K', '6', 's'): 0.55, ('K', '5', 's'): 0.54,
    ('K', '4', 's'): 0.53, ('K', '3', 's'): 0.52, ('K', '2', 's'): 0.51,
    ('K', 'Q', 'o'): 0.61, ('K', 'J', 'o'): 0.60, ('K', 'T', 'o'): 0.59, ('K', '9', 'o'): 0.56,
    ('K', '8', 'o'): 0.54, ('K', '7', 'o'): 0.53, ('K', '6', 'o'): 0.52, ('K', '5', 'o'): 0.51,
    ('K', '4', 'o'): 0.50, ('K', '3', 'o'): 0.49, ('K', '2', 'o'): 0.48,
    ('Q', 'J', 's'): 0.60, ('Q', 'T', 's'): 0.59, ('Q', '9', 's'): 0.56, ('Q', '8', 's'): 0.54,
    ('Q', '7', 's'): 0.53, ('Q', '6', 's'): 0.52, ('Q', '5', 's'): 0.51, ('Q', '4', 's'): 0.50,
    ('Q', '3', 's'): 0.49, ('Q', '2', 's'): 0.48,
    ('Q', 'J', 'o'): 0.58, ('Q', 'T', 'o'): 0.56, ('Q', '9', 'o'): 0.54, ('Q', '8', 'o'): 0.52,
    ('Q', '7', 'o'): 0.50, ('Q', '6', 'o'): 0.49, ('Q', '5', 'o'): 0.48, ('Q', '4', 'o'): 0.47,
    ('Q', '3', 'o'): 0.46, ('Q', '2', 'o'): 0.45,
    ('J', 'T', 's'): 0.58, ('J', '9', 's'): 0.55, ('J', '8', 's'): 0.53, ('J', '7', 's'): 0.51,
    ('J', '6', 's'): 0.50, ('J', '5', 's'): 0.49, ('J', '4', 's'): 0.48, ('J', '3', 's'): 0.47,
    ('J', '2', 's'): 0.46,
    ('J', 'T', 'o'): 0.55, ('J', '9', 'o'): 0.53, ('J', '8', 'o'): 0.50, ('J', '7', 'o'): 0.48,
    ('J', '6', 'o'): 0.47, ('J', '5', 'o'): 0.46, ('J', '4', 'o'): 0.45, ('J', '3', 'o'): 0.44,
    ('J', '2', 'o'): 0.43,
    ('T', '9', 's'): 0.56, ('T', '8', 's'): 0.54, ('T', '7', 's'): 0.52, ('T', '6', 's'): 0.50,
    ('T', '5', 's'): 0.49, ('T', '4', 's'): 0.48, ('T', '3', 's'): 0.47, ('T', '2', 's'): 0.46,
    ('T', '9', 'o'): 0.53, ('T', '8', 'o'): 0.51, ('T', '7', 'o'): 0.49, ('T', '6', 'o'): 0.47,
    ('T', '5', 'o'): 0.46, ('T', '4', 'o'): 0.45, ('T', '3', 'o'): 0.44, ('T', '2', 'o'): 0.43,
    ('9', '8', 's'): 0.55, ('9', '7', 's'): 0.53, ('9', '6', 's'): 0.51, ('9', '5', 's'): 0.49,
    ('9', '4', 's'): 0.48, ('9', '3', 's'): 0.47, ('9', '2', 's'): 0.46,
    ('9', '8', 'o'): 0.52, ('9', '7', 'o'): 0.50, ('9', '6', 'o'): 0.48, ('9', '5', 'o'): 0.46,
    ('9', '4', 'o'): 0.45, ('9', '3', 'o'): 0.44, ('9', '2', 'o'): 0.43,
    ('8', '7', 's'): 0.54, ('8', '6', 's'): 0.52, ('8', '5', 's'): 0.50, ('8', '4', 's'): 0.48,
    ('8', '3', 's'): 0.47, ('8', '2', 's'): 0.46,
    ('8', '7', 'o'): 0.51, ('8', '6', 'o'): 0.49, ('8', '5', 'o'): 0.47, ('8', '4', 'o'): 0.45,
    ('8', '3', 'o'): 0.44, ('8', '2', 'o'): 0.43,
    ('7', '6', 's'): 0.53, ('7', '5', 's'): 0.51, ('7', '4', 's'): 0.49, ('7', '3', 's'): 0.47,
    ('7', '2', 's'): 0.46,
    ('7', '6', 'o'): 0.50, ('7', '5', 'o'): 0.48, ('7', '4', 'o'): 0.46, ('7', '3', 'o'): 0.44,
    ('7', '2', 'o'): 0.43,
    ('6', '5', 's'): 0.52, ('6', '4', 's'): 0.50, ('6', '3', 's'): 0.48, ('6', '2', 's'): 0.46,
    ('6', '5', 'o'): 0.49, ('6', '4', 'o'): 0.47, ('6', '3', 'o'): 0.45, ('6', '2', 'o'): 0.43,
    ('5', '4', 's'): 0.51, ('5', '3', 's'): 0.49, ('5', '2', 's'): 0.47,
    ('5', '4', 'o'): 0.48, ('5', '3', 'o'): 0.46, ('5', '2', 'o'): 0.44,
    ('4', '3', 's'): 0.49, ('4', '2', 's'): 0.47,
    ('4', '3', 'o'): 0.46, ('4', '2', 'o'): 0.44,
    ('3', '2', 's'): 0.47,
    ('3', '2', 'o'): 0.44
}

class Player(BaseBot):
    def __init__(self) -> None:
        self.preflop_cache = PREFLOP_EQUITY
        self.current_street_equity = 0.5
        self.last_street = None
        
        # --- THE MEMORY ENGINE ---
        self.opp_bid_history = []
        self.pot_before_auction = 0
        self.auction_happened_this_hand = False
        
        self.opp_showdowns_seen = 0
        self.opp_bluffs_seen = 0
        self.opp_bluff_ratio = 0.0

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.auction_happened_this_hand = False
        self.pot_before_auction = 0

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        try:
            # BLUFF TRACKER: Did they show up with garbage?
            if current_state.opp_revealed_cards and len(current_state.board) >= 3:
                self.opp_showdowns_seen += 1
                opp_cards = [eval7.Card(c) for c in current_state.opp_revealed_cards]
                board = [eval7.Card(c) for c in current_state.board]
                
                # Throttled to 20 iters to keep hand_end processing ultra-fast
                opp_equity = eval7.py_hand_vs_range_monte_carlo(opp_cards, eval7.HandRange("22+"), board, 20)
                
                if opp_equity < 0.35: # They were caught bluffing!
                    self.opp_bluffs_seen += 1
                
                if self.opp_showdowns_seen >= 5: # Start trusting the stat after 5 showdowns
                    self.opp_bluff_ratio = self.opp_bluffs_seen / self.opp_showdowns_seen
        except:
            pass
            
        self.last_street = None

    def calculate_equity(self, current_state: PokerState, time_bank: float) -> float:
        # Pre-flop instant lookup
        if len(current_state.board) == 0:
            h1, h2 = current_state.my_hand[0], current_state.my_hand[1]
            ranks = sorted([h1[0], h2[0]], key=lambda x: "23456789TJQKA".index(x), reverse=True)
            suited = 's' if h1[1] == h2[1] else 'o'
            return self.preflop_cache.get((ranks[0], ranks[1], suited), 0.40)
        
        # Street Caching (Massive time saver)
        if current_state.street == self.last_street: 
            return self.current_street_equity

        my_cards = [eval7.Card(c) for c in current_state.my_hand]
        board_cards = [eval7.Card(c) for c in current_state.board]
        
        # Dynamic Range mapping
        if current_state.pot > 200:
            v_range = eval7.HandRange("55+, A8s+, K9s+, QTs+, ATo+, KQo") 
        else:
            v_range = eval7.HandRange("22+, A2s+, K2s+, Q2s+, J2s+, T2s+, A2o+, K5o+")
            
        # Target narrow range if we won the auction
        if current_state.opp_revealed_cards:
            known = current_state.opp_revealed_cards[0]
            v_range = eval7.HandRange(f"({known[0]}X)") 
            
        # --- PRECISE 0.015s TIME THROTTLE ---
        # Caps at 50 iterations to perfectly balance speed and intelligence
        if time_bank > 10.0:
            iterations = 50 
        elif time_bank > 5.0:
            iterations = 30
        else:
            iterations = 10 # Panic mode to guarantee survival
            
        equity = eval7.py_hand_vs_range_monte_carlo(my_cards, v_range, board_cards, iterations)
        
        self.last_street = current_state.street
        self.current_street_equity = equity
        return equity

    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        try:
            # TIME SHIELD SHIM
            if game_info.time_bank < 1.0:
                return ActionCheck() if current_state.can_act(ActionCheck) else ActionFold()

            pure_equity = self.calculate_equity(current_state, game_info.time_bank)
            cost_to_call = getattr(current_state, 'cost_to_call', 0)
            pot = current_state.pot

            # --- 1. PREFLOP NIT FILTER ---
            if current_state.street == 'preflop':
                if cost_to_call > 20 and pure_equity < 0.60:
                    return ActionFold()
                if cost_to_call > 0 and pure_equity < 0.52:
                    return ActionFold()

            # --- 2. THE MONTE CARLO TRAP FIX ---
            effective_equity = pure_equity
            if current_state.street == 'flop':
                effective_equity -= 0.12 
            elif current_state.street == 'turn':
                effective_equity -= 0.05 

            # --- 3. THE SMART AUCTION ---
            if current_state.street == 'auction':
                self.pot_before_auction = pot
                self.auction_happened_this_hand = True
                
                if pure_equity > 0.85 or pure_equity < 0.40:
                    return ActionBid(0)
                
                # Sniper bidding: exactly 2 chips more than their average
                if len(self.opp_bid_history) > 3:
                    avg_opp_bid = sum(self.opp_bid_history[-5:]) / len(self.opp_bid_history[-5:])
                    target_bid = int(avg_opp_bid + 2)
                else:
                    target_bid = int(pot * 0.15)
                
                max_bid = int(pot * 0.30) # Never bankrupt yourself for info
                return ActionBid(max(2, min(target_bid, max_bid)))

            # Track Auction Cost Post-Flop
            if self.auction_happened_this_hand and current_state.street != 'auction':
                cost = pot - self.pot_before_auction
                if cost > 0:
                    self.opp_bid_history.append(cost)
                self.auction_happened_this_hand = False

            # --- 4. TIGHT-AGGRESSIVE ACTION + BLUFF CATCHING ---
            
            # Value Betting / Raising
            if current_state.can_act(ActionRaise):
                min_r, max_r = current_state.raise_bounds
                if effective_equity > 0.75:
                    bet = int(pot * 0.75) 
                    return ActionRaise(max(min_r, min(bet, max_r)))
                elif effective_equity > 0.65:
                    bet = int(pot * 0.50) 
                    return ActionRaise(max(min_r, min(bet, max_r)))

            # Calling Engine
            if current_state.can_act(ActionCall):
                pot_odds = cost_to_call / max(1, (pot + cost_to_call))
                
                # Boost equity if we know this opponent bluffs a lot
                adjusted_equity = effective_equity + (self.opp_bluff_ratio * 0.15)
                
                if adjusted_equity > (pot_odds + 0.10):
                    return ActionCall()
            
            return ActionCheck() if current_state.can_act(ActionCheck) else ActionFold()

        except Exception as e:
            # Absolute fallback to prevent disqualification
            traceback.print_exc()
            return ActionCheck() if current_state.can_act(ActionCheck) else ActionFold()

if __name__ == '__main__':
    run_bot(Player(), parse_args())