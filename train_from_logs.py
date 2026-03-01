#!/usr/bin/env python3
"""
train_from_logs.py — Offline trainer for bot.py
=================================================

What this does:
  1. Parses all .log / .log.gz files from logs/competition/
  2. Extracts opponent behavioral stats:
       vpip, aggression, fold-to-bet, auction bid ratio, bet sizing, bluff rate
  3. Classifies each opponent as COCKY / SAFE / LOSING
  4. Calibrates PersonalityClassifier priors from real observed data
  5. Writes learned_params.json (backup / local use)
  6. AUTO-PATCHES bot.py — rewrites _BAKED_BLUFF_PRIOR and _BAKED_PRIORS
     directly in the source so the bot works on the competition server
     without needing any JSON file at all.

Usage:
  python train_from_logs.py --bot-name YourBotName
  python train_from_logs.py --bot-name YourBotName --verbose
  python train_from_logs.py --bot-name YourBotName --bot-file ./bot.py --logs-dir ./logs/competition

After running this, just submit bot.py — no JSON needed.
"""

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
RANK_ORDER = "23456789TJQKA"

def rank_idx(r):
    try:    return RANK_ORDER.index(r)
    except: return 0

def postflop_bucket(hole_cards, board):
    if not board or len(hole_cards) < 2:
        return 0
    all_cards   = hole_cards + board
    ranks       = [c[0] for c in all_cards]
    suits       = [c[1] for c in all_cards]
    hole_ranks  = [c[0] for c in hole_cards]
    board_ranks = [c[0] for c in board]

    rank_counts = {}
    for r in ranks: rank_counts[r] = rank_counts.get(r,0)+1
    suit_counts = {}
    for s in suits: suit_counts[s] = suit_counts.get(s,0)+1

    flush      = any(v>=5 for v in suit_counts.values())
    flush_draw = any(v==4 for v in suit_counts.values())
    unique     = sorted({rank_idx(r) for r in ranks})
    consec = max_c = 1
    for i in range(1,len(unique)):
        if unique[i]==unique[i-1]+1: consec+=1; max_c=max(max_c,consec)
        else: consec=1
    straight      = max_c>=5
    straight_draw = max_c==4
    trips_plus    = [r for r,c in rank_counts.items() if c>=3]
    pairs         = [r for r,c in rank_counts.items() if c==2]
    fh            = bool(trips_plus and (len(trips_plus)>=2 or pairs))
    my_pairs      = [rank_idx(r) for r in rank_counts if r in hole_ranks and rank_counts[r]>=2]
    if len(hole_ranks)>=2 and hole_ranks[0]==hole_ranks[1]:
        my_pairs.append(rank_idx(hole_ranks[0]))
    board_high = max((rank_idx(r) for r in board_ranks), default=0)

    if fh or (flush and straight): return 4
    if flush or straight or trips_plus: return 3
    if my_pairs:
        best=max(my_pairs)
        return 2 if (best>=board_high or best>=board_high-2) else 1
    if flush_draw or straight_draw: return 1
    return 0

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

CARD_RE     = re.compile(r'[2-9TJQKA][cdsh]')
BOARD_RE    = re.compile(r'^(Flop|Turn|River) \[([^\]]+)\]')
SHOWS_RE    = re.compile(r'^(.+?)\s+shows\s+\[([^\]]+)\]')
RAISE_RE    = re.compile(r'^(.+?)\s+(raises?|bets?)\s+(?:to\s+)?(\d+)', re.I)
CALL_RE     = re.compile(r'^(.+?)\s+calls?\s+(\d+)', re.I)
CHECK_RE    = re.compile(r'^(.+?)\s+checks?', re.I)
FOLD_RE     = re.compile(r'^(.+?)\s+folds?', re.I)
BID_RE      = re.compile(r'^(.+?)\s+bids?\s+(\d+)', re.I)
BLIND_RE    = re.compile(r'^(.+?)\s+posts\s+(small|big)\s+blind', re.I)
NEW_HAND_RE = re.compile(r'^(Hand|Round)\s+#?\d+', re.I)
POT_RE      = re.compile(r'pot[:\s]+(\d+)', re.I)
AUCTION_RE  = re.compile(r'Auction.*?(\w+)\s+bids\s+(\d+).*?(\w+)\s+bids\s+(\d+)', re.I)

def parse_cards(s):
    return CARD_RE.findall(s)

class PlayerStats:
    def __init__(self, name):
        self.name            = name
        self.hands_dealt     = 0
        self.hands_vpip      = 0
        self.total_actions   = 0
        self.aggressive_acts = 0
        self.times_faced_bet = 0
        self.folded_to_bet   = 0
        self.auction_ratios  = []
        self.bet_size_ratios = []
        self.showdowns       = 0
        self.bluff_showdowns = 0

    def vpip_rate(self):    return self.hands_vpip/self.hands_dealt if self.hands_dealt else 0.5
    def agg_freq(self):     return self.aggressive_acts/max(self.total_actions,1)
    def fold_rate(self):    return self.folded_to_bet/max(self.times_faced_bet,1)
    def avg_auction(self):  return sum(self.auction_ratios)/len(self.auction_ratios) if self.auction_ratios else 0.10
    def avg_bet(self):      return sum(self.bet_size_ratios)/len(self.bet_size_ratios) if self.bet_size_ratios else 0.55
    def bluff_rate(self):   return self.bluff_showdowns/max(self.showdowns,1)

def parse_log_file(path, hero_name):
    players = {}
    try:
        opener = gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',encoding='utf-8',errors='ignore') as f:
            lines = f.readlines()
    except OSError:
        return players

    def ep(name):
        if name and name not in players:
            players[name] = PlayerStats(name)

    board=[]; pot=0; street='pre-flop'
    hand_players=set(); vpip_hand=set()
    last_bet=''; shows={}

    for raw in lines:
        line = raw.strip()
        if not line: continue

        if NEW_HAND_RE.match(line):
            for p in hand_players:
                ep(p); players[p].hands_dealt+=1
                if p in vpip_hand: players[p].hands_vpip+=1
            board=[]; pot=0; street='pre-flop'
            hand_players=set(); vpip_hand=set(); last_bet=''; shows={}
            continue

        m=BOARD_RE.match(line)
        if m: board=parse_cards(m.group(2)); continue

        m=POT_RE.search(line)
        if m: pot=max(pot,int(m.group(1)))

        m=AUCTION_RE.search(line)
        if m:
            for pn,bid in [(m.group(1),int(m.group(2))),(m.group(3),int(m.group(4)))]:
                ep(pn); hand_players.add(pn)
                if pn!=hero_name: players[pn].auction_ratios.append(bid/max(pot,20))
            continue

        m=BLIND_RE.match(line)
        if m: pn=m.group(1).strip(); ep(pn); hand_players.add(pn); continue

        m=RAISE_RE.match(line)
        if m:
            pn=m.group(1).strip(); amt=int(m.group(3))
            ep(pn); hand_players.add(pn); vpip_hand.add(pn); last_bet=pn
            if pn!=hero_name:
                players[pn].total_actions+=1; players[pn].aggressive_acts+=1
                players[pn].bet_size_ratios.append(amt/max(pot,20))
            continue

        m=CALL_RE.match(line)
        if m:
            pn=m.group(1).strip(); ep(pn); hand_players.add(pn); vpip_hand.add(pn)
            if pn!=hero_name:
                players[pn].total_actions+=1
                if last_bet and last_bet!=pn: players[pn].times_faced_bet+=1
            continue

        m=CHECK_RE.match(line)
        if m:
            pn=m.group(1).strip(); ep(pn); hand_players.add(pn)
            if pn!=hero_name: players[pn].total_actions+=1
            continue

        m=FOLD_RE.match(line)
        if m:
            pn=m.group(1).strip(); ep(pn); hand_players.add(pn)
            if pn!=hero_name:
                players[pn].total_actions+=1
                if last_bet and last_bet!=pn:
                    players[pn].times_faced_bet+=1; players[pn].folded_to_bet+=1
            continue

        m=BID_RE.match(line)
        if m and street=='auction':
            pn=m.group(1).strip(); amt=int(m.group(2))
            ep(pn); hand_players.add(pn)
            if pn!=hero_name: players[pn].auction_ratios.append(amt/max(pot,20))
            continue

        m=SHOWS_RE.match(line)
        if m:
            pn=m.group(1).strip(); cards=parse_cards(m.group(2))
            if len(cards)>=2: shows[pn]=cards
            if len(shows)>=1 and len(board)>=3:
                for shower,hole in shows.items():
                    if shower==hero_name: continue
                    ep(shower)
                    bucket=postflop_bucket(hole,board)
                    players[shower].showdowns+=1
                    if bucket<=1 and pot>=200: players[shower].bluff_showdowns+=1
            continue

    # EOF flush
    for p in hand_players:
        ep(p); players[p].hands_dealt+=1
        if p in vpip_hand: players[p].hands_vpip+=1

    return players

# ---------------------------------------------------------------------------
# Naive Bayes classifier (same as bot.py)
# ---------------------------------------------------------------------------

ORIG_PRIORS = {
    'COCKY':  {'vpip':(0.80,0.12),'aggression':(0.65,0.12),'fold_to_bet':(0.20,0.10),'auction_ratio':(0.22,0.10),'bet_sizing':(0.80,0.20)},
    'SAFE':   {'vpip':(0.30,0.10),'aggression':(0.25,0.10),'fold_to_bet':(0.65,0.12),'auction_ratio':(0.05,0.04),'bet_sizing':(0.45,0.15)},
    'LOSING': {'vpip':(0.60,0.18),'aggression':(0.50,0.20),'fold_to_bet':(0.40,0.18),'auction_ratio':(0.15,0.12),'bet_sizing':(0.65,0.30)},
}
FEATURES = ['vpip','aggression','fold_to_bet','auction_ratio','bet_sizing']

def lp(x,mu,sig): return -0.5*((x-mu)/sig)**2 - math.log(sig)

def classify(ps):
    feats={'vpip':ps.vpip_rate(),'aggression':ps.agg_freq(),
           'fold_to_bet':ps.fold_rate(),'auction_ratio':ps.avg_auction(),'bet_sizing':ps.avg_bet()}
    scores={lb:sum(lp(feats[f],mu,sg) for f,(mu,sg) in pr.items()) for lb,pr in ORIG_PRIORS.items()}
    best=max(scores,key=scores.get)
    mx=max(scores.values())
    exp_s={k:math.exp(v-mx) for k,v in scores.items()}
    tot=sum(exp_s.values())
    return best, {k:v/tot for k,v in exp_s.items()}

def calibrate_priors(qualified, min_hands=15):
    buckets = defaultdict(list)
    for ps in qualified:
        if ps.hands_dealt < min_hands: continue
        label, conf = classify(ps)
        if conf[label] > 0.45: buckets[label].append(ps)

    result = {}
    for label, orig_feats in ORIG_PRIORS.items():
        result[label] = {}
        members = buckets.get(label, [])
        for feat, (orig_mu, orig_sig) in orig_feats.items():
            vals = []
            for ps in members:
                v = {'vpip':ps.vpip_rate(),'aggression':ps.agg_freq(),
                     'fold_to_bet':ps.fold_rate(),'auction_ratio':ps.avg_auction(),
                     'bet_sizing':ps.avg_bet()}[feat]
                vals.append(v)
            if len(vals) >= 3:
                emp_mu = sum(vals)/len(vals)
                alpha  = min(1.0, len(vals)/20.0)
                blended_mu  = round(alpha*emp_mu + (1-alpha)*orig_mu, 4)
                blended_sig = round(orig_sig * max(0.80, 1.0 - 0.01*len(vals)), 4)
            else:
                blended_mu, blended_sig = orig_mu, orig_sig
            result[label][feat] = [blended_mu, blended_sig]
    return result

# ---------------------------------------------------------------------------
# AUTO-PATCH bot.py  ← the key new feature
# ---------------------------------------------------------------------------

def patch_bot(bot_path: Path, bluff_prior: float, calibrated_priors: dict) -> bool:
    """
    Rewrites the _BAKED_BLUFF_PRIOR and _BAKED_PRIORS constants inside bot.py
    so the trained values are embedded in the source code itself.
    The bot then works on any server without needing learned_params.json.

    Returns True if patch was applied, False if the markers weren't found
    (meaning bot.py doesn't have the expected structure).
    """
    if not bot_path.exists():
        print(f"[WARN] bot.py not found at {bot_path} — skipping patch")
        return False

    source = bot_path.read_text(encoding='utf-8')

    # --- Patch bluff prior ---
    bluff_pattern = re.compile(
        r'(_BAKED_BLUFF_PRIOR\s*=\s*)[\d.]+',
        re.MULTILINE
    )
    if not bluff_pattern.search(source):
        print("[WARN] _BAKED_BLUFF_PRIOR marker not found in bot.py — skipping bluff patch")
        bluff_ok = False
    else:
        source = bluff_pattern.sub(
            lambda m: f"{m.group(1)}{bluff_prior:.4f}",
            source
        )
        bluff_ok = True

    # --- Patch personality priors ---
    # We replace the entire _BAKED_PRIORS = { ... } block.
    # The block starts at "_BAKED_PRIORS = {" and ends at the matching closing "}"
    # We detect it by finding the line and then tracking brace depth.
    priors_start = re.search(r'(\s*)_BAKED_PRIORS\s*=\s*\{', source)
    if not priors_start:
        print("[WARN] _BAKED_PRIORS marker not found in bot.py — skipping priors patch")
        priors_ok = False
    else:
        # Find where the dict ends by tracking brace depth
        start_pos = priors_start.start()
        brace_pos = source.index('{', priors_start.start())
        depth = 0
        end_pos = brace_pos
        for i, ch in enumerate(source[brace_pos:], start=brace_pos):
            if ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break

        indent = '        '   # 8 spaces — matches method body indentation in bot.py

        def fmt_priors(cal):
            lines = [f'{indent}_BAKED_PRIORS = {{']
            labels = ['COCKY', 'SAFE', 'LOSING']
            for lb in labels:
                feats = cal[lb]
                lines.append(f'{indent}    {lb}:  {{')
                feat_items = list(feats.items())
                for i,(feat,(mu,sig)) in enumerate(feat_items):
                    comma = ',' if i < len(feat_items)-1 else ''
                    lines.append(f"{indent}             '{feat}':({mu},{sig}){comma}")
                lines.append(f'{indent}    }},')
            lines.append(f'{indent}}}')
            return '\n'.join(lines)

        new_block = fmt_priors(calibrated_priors)
        source = source[:start_pos] + new_block + source[end_pos+1:]
        priors_ok = True

    if bluff_ok or priors_ok:
        bot_path.write_text(source, encoding='utf-8')
        print(f"[OK] Patched bot.py:")
        if bluff_ok:
            print(f"     _BAKED_BLUFF_PRIOR = {bluff_prior:.4f}")
        if priors_ok:
            print(f"     _BAKED_PRIORS updated with calibrated values")
        return True

    return False

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train bot.py from competition logs and auto-patch source")
    parser.add_argument('--logs-dir',  default='logs/competition')
    parser.add_argument('--bot-name',  default='', help="Your bot's name in logs")
    parser.add_argument('--bot-file',  default='./bot.py', help="Path to bot.py to patch")
    parser.add_argument('--output',    default='learned_params.json')
    parser.add_argument('--min-hands', type=int, default=15)
    parser.add_argument('--verbose',   action='store_true')
    parser.add_argument('--no-patch',  action='store_true', help="Skip patching bot.py (JSON only)")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"[ERROR] Logs dir not found: {logs_dir}")
        print("  Run: python download_logs.py --matches-url https://YOUR_SITE/matches")
        return

    log_files = sorted(list(logs_dir.glob('*.log')) + list(logs_dir.glob('*.log.gz')))
    if not log_files:
        print(f"[ERROR] No log files in {logs_dir}")
        return

    print(f"Parsing {len(log_files)} log file(s)...")

    global_stats = {}
    for path in log_files:
        for name, ps in parse_log_file(path, args.bot_name).items():
            if name == args.bot_name: continue
            if name not in global_stats:
                global_stats[name] = PlayerStats(name)
            gs = global_stats[name]
            gs.hands_dealt     += ps.hands_dealt
            gs.hands_vpip      += ps.hands_vpip
            gs.total_actions   += ps.total_actions
            gs.aggressive_acts += ps.aggressive_acts
            gs.times_faced_bet += ps.times_faced_bet
            gs.folded_to_bet   += ps.folded_to_bet
            gs.showdowns       += ps.showdowns
            gs.bluff_showdowns += ps.bluff_showdowns
            gs.auction_ratios.extend(ps.auction_ratios)
            gs.bet_size_ratios.extend(ps.bet_size_ratios)

    if not global_stats:
        print("[WARN] No opponent data found. Check --bot-name and log format.")
        return

    all_ps    = list(global_stats.values())
    qualified = [ps for ps in all_ps if ps.hands_dealt >= args.min_hands]

    # Bluff prior
    total_sd  = sum(ps.showdowns for ps in all_ps)
    total_bl  = sum(ps.bluff_showdowns for ps in all_ps)
    bluff_prior = max(0.0, min(1.0, total_bl/total_sd if total_sd>0 else 0.20))

    print(f"\nOpponents: {len(all_ps)} total, {len(qualified)} with ≥{args.min_hands} hands")
    print(f"Bluff prior: {bluff_prior:.4f}  ({total_bl}/{total_sd} showdowns)")

    # Calibrate priors
    calibrated = calibrate_priors(qualified, args.min_hands)

    if args.verbose:
        print("\n--- Per-opponent stats ---")
        print(f"{'Name':<20} {'H':>4} {'VPIP':>5} {'Agg':>5} {'F2B':>5} {'AucR':>5} {'Bet':>5} {'Blf':>5} {'Label'}")
        print("-"*75)
        for ps in sorted(qualified, key=lambda x: -x.hands_dealt):
            label, conf = classify(ps)
            print(f"{ps.name:<20} {ps.hands_dealt:>4}  {ps.vpip_rate():.2f}  {ps.agg_freq():.2f}  "
                  f"{ps.fold_rate():.2f}  {ps.avg_auction():.2f}  {ps.avg_bet():.2f}  "
                  f"{ps.bluff_rate():.2f}  {label}({conf[label]:.0%})")

        print("\n--- Calibrated priors vs original ---")
        for lb, feats in calibrated.items():
            print(f"  {lb}:")
            for feat, (mu, sig) in feats.items():
                orig_mu = ORIG_PRIORS[lb][feat][0]
                flag = " ←" if abs(mu - orig_mu) > 0.01 else ""
                print(f"    {feat:>14}: mu={mu:.3f} (was {orig_mu:.3f}){flag}  σ={sig:.3f}")

    # Write JSON (backup + for local use)
    out = {
        'opp_bluff_prior': round(bluff_prior, 4),
        'calibrated_priors': calibrated,
        'opponent_count': len(all_ps),
        'qualified_count': len(qualified),
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}")

    # Auto-patch bot.py
    if not args.no_patch:
        print(f"\nPatching {args.bot_file}...")
        patched = patch_bot(Path(args.bot_file), bluff_prior, calibrated)
        if patched:
            print(f"\nDone. Submit bot.py — learned values are now baked in.")
            print(f"No learned_params.json needed on the competition server.")
        else:
            print(f"\nCould not patch bot.py. Update _BAKED_BLUFF_PRIOR and _BAKED_PRIORS manually.")
    else:
        print("\n--no-patch set. bot.py not modified.")

if __name__ == '__main__':
    main()
