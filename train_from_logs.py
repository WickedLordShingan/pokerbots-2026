#!/usr/bin/env python3
"""
train_from_logs.py — Full offline trainer for bot.py v6
========================================================

What this does:
  1. Reads all .log / .log.gz match files from logs/competition/
  2. Parses every hand and extracts opponent behavioral stats:
       - VPIP rate
       - Aggression frequency (bets+raises / total voluntary actions)
       - Fold-to-bet rate
       - Auction bid ratio (bid / pot)
       - Average bet sizing (bet / pot)
       - Bluff rate at showdown (weak hand shown in big pot)
  3. Computes which personality archetype (COCKY / SAFE / LOSING) the
     log opponent's stats best fit.
  4. Calibrates the Gaussian prior means (mu) used by PersonalityClassifier
     from real observed data. Sigmas are slightly tightened if sample is large.
  5. Writes learned_params.json which bot.py loads at startup via _load_bluff_prior.

Usage:
  # First, scrape logs from competition site:
  python download_logs.py --matches-url https://YOUR_SITE/matches --headless

  # Then train:
  python train_from_logs.py --bot-name YourBotName

  # Check output:
  cat learned_params.json

The bot reads learned_params.json at startup — no restart needed between runs.
Just run this after downloading new logs and your next bot session benefits.

Log format assumed (IIT engine):
  - Lines are plain text
  - Blinds:   "BotA posts small blind 1"  /  "BotB posts big blind 2"
  - Actions:  "BotA raises to 10"  /  "BotB calls 10"  /  "BotA checks"
              "BotA folds"  /  "BotA bids 15"
  - Board:    "Flop [Ah Kd 5s]"  /  "Turn [Ah Kd 5s 2c]"  /  "River [...]"
  - Showdown: "BotA shows [Ah Kd]"
  - Result:   "BotA wins 42"  /  "BotA wins 0"  (0 = push or fold)
  - Auction:  "Auction: BotA bids 15, BotB bids 8. BotA wins."
              OR lines with "bids" in auction context
"""

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Card / hand helpers (mirrors bot.py, no eval7 dependency)
# ---------------------------------------------------------------------------

RANK_ORDER = "23456789TJQKA"


def rank_idx(r: str) -> int:
    try:
        return RANK_ORDER.index(r)
    except ValueError:
        return 0


def postflop_bucket(hole_cards: list[str], board: list[str]) -> int:
    """
    0 = trash, 1 = weak pair/draw, 2 = top/mid pair, 3 = flush/straight/trips, 4 = monsters
    Mirrors _postflop_bucket in bot.py exactly.
    """
    if not board or len(hole_cards) < 2:
        return 0

    all_cards  = hole_cards + board
    ranks      = [c[0] for c in all_cards]
    suits      = [c[1] for c in all_cards]
    hole_ranks = [c[0] for c in hole_cards]
    board_ranks = [c[0] for c in board]

    rank_counts: dict[str, int] = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    suit_counts: dict[str, int] = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1

    flush      = any(v >= 5 for v in suit_counts.values())
    flush_draw = any(v == 4 for v in suit_counts.values())

    unique_sorted = sorted({rank_idx(r) for r in ranks})
    consec = max_consec = 1
    for i in range(1, len(unique_sorted)):
        if unique_sorted[i] == unique_sorted[i - 1] + 1:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 1
    straight      = max_consec >= 5
    straight_draw = max_consec == 4

    trips_plus = [r for r, c in rank_counts.items() if c >= 3]
    pairs      = [r for r, c in rank_counts.items() if c == 2]
    fh_or_better = bool(trips_plus and (len(trips_plus) >= 2 or pairs))

    my_pair_ranks = []
    for r in rank_counts:
        if r in hole_ranks and rank_counts[r] >= 2:
            my_pair_ranks.append(rank_idx(r))
    if hole_ranks[0] == hole_ranks[1]:
        my_pair_ranks.append(rank_idx(hole_ranks[0]))

    board_high = max((rank_idx(r) for r in board_ranks), default=0)

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


# ---------------------------------------------------------------------------
# Regex patterns for log parsing
# ---------------------------------------------------------------------------

BOARD_RE    = re.compile(r"^(Flop|Turn|River) \[([^\]]+)\]")
SHOWS_RE    = re.compile(r"^(.+?)\s+shows\s+\[([^\]]+)\]")
WINS_RE     = re.compile(r"^(.+?)\s+wins\s+(\d+)")
RAISE_RE    = re.compile(r"^(.+?)\s+(raises?|bets?)\s+(?:to\s+)?(\d+)", re.I)
CALL_RE     = re.compile(r"^(.+?)\s+calls?\s+(\d+)", re.I)
CHECK_RE    = re.compile(r"^(.+?)\s+checks?", re.I)
FOLD_RE     = re.compile(r"^(.+?)\s+folds?", re.I)
BID_RE      = re.compile(r"^(.+?)\s+bids?\s+(\d+)", re.I)
BLIND_RE    = re.compile(r"^(.+?)\s+posts\s+(small|big)\s+blind\s+(\d+)", re.I)
AUCTION_RE  = re.compile(r"Auction.*?(\w+)\s+bids\s+(\d+).*?(\w+)\s+bids\s+(\d+)", re.I)
NEW_HAND_RE = re.compile(r"^(Hand|Round)\s+#?\d+", re.I)
POT_RE      = re.compile(r"pot[:\s]+(\d+)", re.I)
STREET_RE   = re.compile(r"^(Preflop|Pre-flop|Flop|Turn|River|Auction)\b", re.I)

CARD_RE     = re.compile(r"[2-9TJQKA][cdsh]")


def parse_cards(s: str) -> list[str]:
    return CARD_RE.findall(s)


# ---------------------------------------------------------------------------
# Per-player stats accumulator
# ---------------------------------------------------------------------------

class PlayerStats:
    def __init__(self, name: str):
        self.name            = name
        # VPIP
        self.hands_dealt     = 0
        self.hands_vpip      = 0     # voluntarily put chips in (not just blinds)
        # Aggression
        self.total_actions   = 0
        self.aggressive_acts = 0     # bets + raises
        # Fold-to-bet
        self.faced_bet       = 0
        self.folded_to_bet   = 0
        # Auction
        self.auction_bids    = []    # list of bid/pot ratios
        # Bet sizing
        self.bet_sizes       = []    # list of bet/pot ratios
        # Showdown bluff tracking
        self.showdowns       = 0
        self.bluff_showdowns = 0     # weak hand (bucket ≤ 1) in pot ≥ 200

    def vpip_rate(self) -> float:
        return self.hands_vpip / self.hands_dealt if self.hands_dealt else 0.50

    def aggression_freq(self) -> float:
        return self.aggressive_acts / max(self.total_actions, 1)

    def fold_to_bet_rate(self) -> float:
        return self.folded_to_bet / max(self.faced_bet, 1)

    def avg_auction_ratio(self) -> float:
        return sum(self.auction_bids) / len(self.auction_bids) if self.auction_bids else 0.10

    def avg_bet_sizing(self) -> float:
        return sum(self.bet_sizes) / len(self.bet_sizes) if self.bet_sizes else 0.55

    def bluff_rate(self) -> float:
        return self.bluff_showdowns / max(self.showdowns, 1)

    def feature_dict(self) -> dict:
        return {
            "vpip":          self.vpip_rate(),
            "aggression":    self.aggression_freq(),
            "fold_to_bet":   self.fold_to_bet_rate(),
            "auction_ratio": self.avg_auction_ratio(),
            "bet_sizing":    self.avg_bet_sizing(),
            "bluff_rate":    self.bluff_rate(),
            "hands":         self.hands_dealt,
            "showdowns":     self.showdowns,
        }


# ---------------------------------------------------------------------------
# Log file parser
# ---------------------------------------------------------------------------

def parse_log_file(path: Path, hero_name: str) -> dict[str, PlayerStats]:
    """
    Parse one log file and return {player_name: PlayerStats}.
    We focus on the opponent (not hero) stats.
    """
    players: dict[str, PlayerStats] = {}

    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return players

    # -- Hand state --
    board:        list[str] = []
    pot:          int       = 0
    street:       str       = "pre-flop"
    hand_players: set[str]  = set()
    vpip_this_hand: set[str] = set()
    prev_action:  dict[str, str] = {}  # last action per player this hand
    last_bet_player: str = ""
    shows:        dict[str, list[str]] = {}   # player → hole cards at showdown
    in_hand:      bool = True

    def ensure_player(name: str):
        if name and name not in players:
            players[name] = PlayerStats(name)

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # ---- New hand ----
        if NEW_HAND_RE.match(line):
            # Close out previous hand
            for pname in hand_players:
                ensure_player(pname)
                players[pname].hands_dealt += 1
                if pname in vpip_this_hand:
                    players[pname].hands_vpip += 1

            # Reset
            board            = []
            pot              = 0
            street           = "pre-flop"
            hand_players     = set()
            vpip_this_hand   = set()
            prev_action      = {}
            last_bet_player  = ""
            shows            = {}
            in_hand          = True
            continue

        # ---- Street change ----
        m_street = STREET_RE.match(line)
        if m_street:
            s = m_street.group(1).lower()
            street = {"preflop": "pre-flop", "pre-flop": "pre-flop",
                      "flop": "flop", "turn": "turn", "river": "river",
                      "auction": "auction"}.get(s, street)

        # ---- Board ----
        m_board = BOARD_RE.match(line)
        if m_board:
            board = parse_cards(m_board.group(2))

        # ---- Pot (rough) ----
        m_pot = POT_RE.search(line)
        if m_pot:
            pot = max(pot, int(m_pot.group(1)))

        # ---- Auction line ----
        m_auction = AUCTION_RE.search(line)
        if m_auction:
            p1, b1, p2, b2 = m_auction.group(1), int(m_auction.group(2)), m_auction.group(3), int(m_auction.group(4))
            for pname, bid in [(p1, b1), (p2, b2)]:
                ensure_player(pname)
                hand_players.add(pname)
                eff_pot = max(pot, 20)
                players[pname].auction_bids.append(bid / eff_pot)
            continue

        # ---- Blind posts (not VPIP, but track hand membership) ----
        m_blind = BLIND_RE.match(line)
        if m_blind:
            pname = m_blind.group(1).strip()
            ensure_player(pname)
            hand_players.add(pname)
            continue

        # ---- Raises / Bets ----
        m_raise = RAISE_RE.match(line)
        if m_raise:
            pname   = m_raise.group(1).strip()
            amount  = int(m_raise.group(3))
            ensure_player(pname)
            hand_players.add(pname)
            if pname != hero_name:
                players[pname].total_actions   += 1
                players[pname].aggressive_acts += 1
                eff_pot = max(pot, 20)
                players[pname].bet_sizes.append(amount / eff_pot)
            vpip_this_hand.add(pname)
            last_bet_player = pname
            prev_action[pname] = "raise"
            continue

        # ---- Calls ----
        m_call = CALL_RE.match(line)
        if m_call:
            pname = m_call.group(1).strip()
            ensure_player(pname)
            hand_players.add(pname)
            if pname != hero_name:
                players[pname].total_actions += 1
                # Was facing a bet?
                if last_bet_player and last_bet_player != pname:
                    players[pname].faced_bet += 1
            vpip_this_hand.add(pname)
            prev_action[pname] = "call"
            continue

        # ---- Checks ----
        m_check = CHECK_RE.match(line)
        if m_check:
            pname = m_check.group(1).strip()
            ensure_player(pname)
            hand_players.add(pname)
            if pname != hero_name:
                players[pname].total_actions += 1
            prev_action[pname] = "check"
            continue

        # ---- Folds ----
        m_fold = FOLD_RE.match(line)
        if m_fold:
            pname = m_fold.group(1).strip()
            ensure_player(pname)
            hand_players.add(pname)
            if pname != hero_name:
                players[pname].total_actions += 1
                if last_bet_player and last_bet_player != pname:
                    players[pname].faced_bet    += 1
                    players[pname].folded_to_bet += 1
            prev_action[pname] = "fold"
            continue

        # ---- Bids (auction, separate line format) ----
        m_bid = BID_RE.match(line)
        if m_bid and street == "auction":
            pname  = m_bid.group(1).strip()
            amount = int(m_bid.group(2))
            ensure_player(pname)
            hand_players.add(pname)
            eff_pot = max(pot, 20)
            players[pname].auction_bids.append(amount / eff_pot)
            continue

        # ---- Showdown: player shows cards ----
        m_show = SHOWS_RE.match(line)
        if m_show:
            pname = m_show.group(1).strip()
            cards = parse_cards(m_show.group(2))
            if len(cards) >= 2:
                shows[pname] = cards

            # If we now have at least one opponent showing and board ≥ 3, classify
            if len(shows) >= 1 and len(board) >= 3:
                for shower, hole in shows.items():
                    if shower == hero_name:
                        continue
                    ensure_player(shower)
                    bucket   = postflop_bucket(hole, board)
                    is_bluff = bucket <= 1 and pot >= 200
                    players[shower].showdowns += 1
                    if is_bluff:
                        players[shower].bluff_showdowns += 1
            continue

    # Final hand close (EOF)
    for pname in hand_players:
        ensure_player(pname)
        players[pname].hands_dealt += 1
        if pname in vpip_this_hand:
            players[pname].hands_vpip += 1

    return players


# ---------------------------------------------------------------------------
# Personality classification (same model as bot.py)
# ---------------------------------------------------------------------------

PRIORS = {
    "COCKY":  {"vpip": (0.80, 0.12), "aggression": (0.65, 0.12),
               "fold_to_bet": (0.20, 0.10), "auction_ratio": (0.22, 0.10),
               "bet_sizing":  (0.80, 0.20)},
    "SAFE":   {"vpip": (0.30, 0.10), "aggression": (0.25, 0.10),
               "fold_to_bet": (0.65, 0.12), "auction_ratio": (0.05, 0.04),
               "bet_sizing":  (0.45, 0.15)},
    "LOSING": {"vpip": (0.60, 0.18), "aggression": (0.50, 0.20),
               "fold_to_bet": (0.40, 0.18), "auction_ratio": (0.15, 0.12),
               "bet_sizing":  (0.65, 0.30)},
}

FEATURE_KEYS = ["vpip", "aggression", "fold_to_bet", "auction_ratio", "bet_sizing"]


def log_prob(x: float, mu: float, sigma: float) -> float:
    return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma)


def classify_stats(stats: PlayerStats) -> tuple[str, dict[str, float]]:
    feats = {
        "vpip":          stats.vpip_rate(),
        "aggression":    stats.aggression_freq(),
        "fold_to_bet":   stats.fold_to_bet_rate(),
        "auction_ratio": stats.avg_auction_ratio(),
        "bet_sizing":    stats.avg_bet_sizing(),
    }

    log_scores = {}
    for label, prior in PRIORS.items():
        log_scores[label] = sum(log_prob(feats[f], mu, sig) for f, (mu, sig) in prior.items())

    max_s  = max(log_scores.values())
    exp_s  = {k: math.exp(v - max_s) for k, v in log_scores.items()}
    total  = sum(exp_s.values())
    conf   = {k: v / total for k, v in exp_s.items()}
    best   = max(conf, key=conf.get)
    return best, conf


# ---------------------------------------------------------------------------
# Calibrate prior means from observed opponent stats
# ---------------------------------------------------------------------------

def calibrate_priors(all_stats: list[PlayerStats], min_hands: int = 20) -> dict:
    """
    For each personality class, gather observed feature values from players
    with enough hands, and compute empirical means. If sample is too small,
    keep original prior means. Tighten sigma if sample is large (more confident).

    Returns a dict matching the PRIORS structure for JSON output.
    """
    # Bucket each opponent into a personality by their observed stats
    buckets: dict[str, list[PlayerStats]] = defaultdict(list)
    for ps in all_stats:
        if ps.hands_dealt < min_hands:
            continue
        label, conf = classify_stats(ps)
        # Only use confident classifications
        if conf[label] > 0.45:
            buckets[label].append(ps)

    calibrated = {}
    for label, prior_feats in PRIORS.items():
        calibrated[label] = {}
        members = buckets.get(label, [])

        for feat, (orig_mu, orig_sigma) in prior_feats.items():
            vals = []
            for ps in members:
                feat_val = {
                    "vpip":          ps.vpip_rate(),
                    "aggression":    ps.aggression_freq(),
                    "fold_to_bet":   ps.fold_to_bet_rate(),
                    "auction_ratio": ps.avg_auction_ratio(),
                    "bet_sizing":    ps.avg_bet_sizing(),
                }[feat]
                vals.append(feat_val)

            if len(vals) >= 3:
                emp_mu    = sum(vals) / len(vals)
                # Blend empirical mean with prior (shrinkage estimator)
                # More data → more weight on empirical
                alpha = min(1.0, len(vals) / 20.0)   # full trust at 20+ observations
                blended_mu = alpha * emp_mu + (1 - alpha) * orig_mu
                # Slightly tighten sigma if we have many observations
                tighten = max(0.80, 1.0 - 0.01 * len(vals))
                blended_sigma = orig_sigma * tighten
            else:
                blended_mu    = orig_mu
                blended_sigma = orig_sigma

            calibrated[label][feat] = [round(blended_mu, 4), round(blended_sigma, 4)]

    return calibrated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train bot.py priors from competition match logs")
    parser.add_argument("--logs-dir",  default="logs/competition",
                        help="Directory with .log / .log.gz files")
    parser.add_argument("--bot-name",  default="",
                        help="Your bot's name in logs (used to separate hero vs villain stats)")
    parser.add_argument("--output",    default="learned_params.json",
                        help="Output JSON file (read by bot.py at startup)")
    parser.add_argument("--min-hands", type=int, default=15,
                        help="Min hands for an opponent to be included in calibration")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print per-opponent breakdown")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"[ERROR] Logs directory not found: {logs_dir}")
        print("  Run: python download_logs.py --matches-url https://YOUR_SITE/matches")
        return

    log_files = sorted(list(logs_dir.glob("*.log")) + list(logs_dir.glob("*.log.gz")))
    if not log_files:
        print(f"[ERROR] No .log or .log.gz files found in {logs_dir}")
        return

    print(f"Found {len(log_files)} log file(s). Parsing...")

    # Aggregate stats across all logs, per player name
    global_stats: dict[str, PlayerStats] = {}

    for path in log_files:
        file_stats = parse_log_file(path, args.bot_name)
        for name, ps in file_stats.items():
            if name == args.bot_name:
                continue   # skip our own bot
            if name not in global_stats:
                global_stats[name] = PlayerStats(name)
            gs = global_stats[name]
            gs.hands_dealt     += ps.hands_dealt
            gs.hands_vpip      += ps.hands_vpip
            gs.total_actions   += ps.total_actions
            gs.aggressive_acts += ps.aggressive_acts
            gs.faced_bet       += ps.faced_bet
            gs.folded_to_bet   += ps.folded_to_bet
            gs.showdowns       += ps.showdowns
            gs.bluff_showdowns += ps.bluff_showdowns
            gs.auction_bids.extend(ps.auction_bids)
            gs.bet_sizes.extend(ps.bet_sizes)

    if not global_stats:
        print("[WARN] No opponent data found. Check --bot-name and log format.")
        return

    all_players = list(global_stats.values())
    qualified   = [ps for ps in all_players if ps.hands_dealt >= args.min_hands]

    print(f"\nOpponents found: {len(all_players)}")
    print(f"Opponents with ≥{args.min_hands} hands: {len(qualified)}")

    # ---- Global bluff prior ----
    total_showdowns = sum(ps.showdowns for ps in all_players)
    total_bluffs    = sum(ps.bluff_showdowns for ps in all_players)
    bluff_prior     = total_bluffs / total_showdowns if total_showdowns > 0 else 0.20
    bluff_prior     = max(0.0, min(1.0, bluff_prior))

    print(f"\nGlobal bluff prior:  {bluff_prior:.3f}  ({total_bluffs}/{total_showdowns} showdowns)")

    # ---- Per-opponent breakdown ----
    personality_counts: dict[str, int] = defaultdict(int)

    if args.verbose:
        print("\n--- Per-opponent stats ---")
        print(f"{'Name':<20} {'Hands':>5} {'VPIP':>6} {'Agg':>5} {'F2B':>5} {'AucR':>5} {'Bet':>5} {'BluffR':>6} {'Class'}")
        print("-" * 80)

    for ps in sorted(qualified, key=lambda x: -x.hands_dealt):
        label, conf = classify_stats(ps)
        personality_counts[label] += 1
        if args.verbose:
            f = ps.feature_dict()
            print(
                f"{ps.name:<20} {f['hands']:>5d}  "
                f"{f['vpip']:>5.2f}  {f['aggression']:>5.2f}  "
                f"{f['fold_to_bet']:>5.2f}  {f['auction_ratio']:>5.2f}  "
                f"{f['bet_sizing']:>5.2f}  {f['bluff_rate']:>5.2f}  "
                f"{label} ({conf[label]:.0%})"
            )

    print(f"\nPersonality distribution (among ≥{args.min_hands}-hand opponents):")
    for label in ("COCKY", "SAFE", "LOSING"):
        print(f"  {label:>6}: {personality_counts[label]:>3} players")

    # ---- Calibrate priors ----
    calibrated = calibrate_priors(qualified, min_hands=args.min_hands)

    print("\nCalibrated prior means vs original:")
    for label, feats in calibrated.items():
        orig = PRIORS[label]
        print(f"  {label}:")
        for feat, (mu, sigma) in feats.items():
            orig_mu = orig[feat][0]
            delta = mu - orig_mu
            flag  = " ←updated" if abs(delta) > 0.01 else ""
            print(f"    {feat:>14}: mu={mu:.3f} (orig {orig_mu:.3f}{', Δ' + f'{delta:+.3f}' if flag else ''}){flag}  σ={sigma:.3f}")

    # ---- Write output ----
    output = {
        "opp_bluff_prior":         round(bluff_prior, 4),
        "total_showdowns_seen":    total_showdowns,
        "total_bluffs_seen":       total_bluffs,
        "opponent_count":          len(all_players),
        "qualified_opponent_count": len(qualified),
        "personality_counts":      dict(personality_counts),
        "calibrated_priors":       calibrated,
        # Summary of per-opponent observed features for debugging
        "opponent_summaries": [
            {
                "name":        ps.name,
                "hands":       ps.hands_dealt,
                "personality": classify_stats(ps)[0],
                **{k: round(v, 3) for k, v in ps.feature_dict().items() if k != "hands"},
            }
            for ps in sorted(qualified, key=lambda x: -x.hands_dealt)
        ],
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}  ({out_path.stat().st_size} bytes)")
    print("\nYour bot will load these priors at startup via _load_bluff_prior().")
    print("No restart needed — just ensure learned_params.json is in the same folder as bot.py.")


if __name__ == "__main__":
    main()
