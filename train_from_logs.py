#!/usr/bin/env python3
"""
Offline trainer: estimate opponent bluff prior from competition logs.

Assumptions about log format (based on IIT engine logs):
  - Text lines, possibly gzipped (.log.gz)
  - Board lines like: "Flop [Ah Kd Qs ..." or "Turn [..]" or "River [..]"
  - Showdown lines like: "<Name> shows [Ah Kd]"

We:
  - Walk logs/competition/*.log.gz (or a custom --logs-dir)
  - For each showdown where we see opponent hole cards and a board with >=3 cards,
    classify their hand into a coarse bucket (0..4) using the same bucketer as bot.py.
  - Treat bucket <= 1 as "bluffy / weak hand shown".
  - Aggregate across all opponents and write learned_params.json with:
        {"opp_bluff_prior": <float in [0,1]>}

Run:
  python train_from_logs.py --bot-name YourBotName
"""

import argparse
import gzip
import json
import re
from pathlib import Path

RANK_ORDER = "23456789TJQKA"


def rank_idx(r: str) -> int:
    return RANK_ORDER.index(r)


def postflop_bucket(hole_cards, board):
    """
    Mirror of _postflop_bucket in bot.py, simplified.
    """
    if not board or len(hole_cards) < 2:
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

    flush = any(cnt >= 5 for cnt in suit_counts.values())
    flush_draw = any(cnt == 4 for cnt in suit_counts.values())

    unique_ranks = sorted({rank_idx(r) for r in ranks})
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

    trips_plus = [r for r, c in rank_counts.items() if c >= 3]
    pairs = [r for r, c in rank_counts.items() if c == 2]
    full_house_or_better = bool(trips_plus and (len(trips_plus) >= 2 or pairs))

    my_pair_ranks = []
    for r in rank_counts:
        if r in hole_ranks and rank_counts[r] >= 2:
            my_pair_ranks.append(rank_idx(r))

    pocket_pair = hole_ranks[0] == hole_ranks[1]
    if pocket_pair:
        my_pair_ranks.append(rank_idx(hole_ranks[0]))

    board_high = max(rank_idx(r) for r in board_ranks) if board_ranks else 0

    if full_house_or_better or (flush and straight):
        return 4
    if flush or straight or trips_plus:
        return 3

    if my_pair_ranks:
        best_pair = max(my_pair_ranks)
        if best_pair >= board_high:
            return 2
        if best_pair >= board_high - 2:
            return 2
        return 1

    if flush_draw or straight_draw:
        return 1

    return 0


BOARD_RE = re.compile(r"^(Flop|Turn|River) \[([2-9TJQKA][cdhs](?: [2-9TJQKA][cdhs])*)")
SHOWS_RE = re.compile(r"^(.+?) shows \[([2-9TJQKA][cdhs](?: [2-9TJQKA][cdhs])*)")


def parse_log(path: Path, hero_name: str, stats: dict):
    """
    Update stats dict in-place with bluff events from one log file.
    """
    try:
        if path.suffix == ".gz":
            f = gzip.open(path, "rt", encoding="utf-8", errors="ignore")
        else:
            f = path.open("rt", encoding="utf-8", errors="ignore")
    except OSError:
        return

    with f:
        board = []
        pending_shows = []  # list of (name, [cards])

        for line in f:
            line = line.strip()
            if not line:
                continue

            m_board = BOARD_RE.match(line)
            if m_board:
                cards_str = m_board.group(2)
                board = cards_str.split()
                continue

            m_show = SHOWS_RE.match(line)
            if m_show:
                name = m_show.group(1).strip()
                cards = m_show.group(2).split()
                pending_shows.append((name, cards))

                # When both players have shown, evaluate opponent bucket
                if len(pending_shows) >= 2 and board:
                    # identify opponent (not hero_name)
                    (n1, c1), (n2, c2) = pending_shows[-2], pending_shows[-1]
                    if hero_name and n1 == hero_name:
                        opp_cards = c2
                        opp_name = n2
                    elif hero_name and n2 == hero_name:
                        opp_cards = c1
                        opp_name = n1
                    else:
                        # we don't know which side we are; just pick the second
                        opp_cards = c2
                        opp_name = n2

                    bucket = postflop_bucket(opp_cards, board)
                    stats["showdowns"] += 1
                    if bucket <= 1:
                        stats["bluffs"] += 1
                    stats["by_opponent"][opp_name] = stats["by_opponent"].get(opp_name, 0) + (1 if bucket <= 1 else 0)

    return


def main():
    parser = argparse.ArgumentParser(description="Train bluff prior from competition logs")
    parser.add_argument(
        "--logs-dir",
        default="logs/competition",
        help="Directory containing .log or .log.gz files (default: logs/competition)",
    )
    parser.add_argument(
        "--bot-name",
        default="",
        help="Your bot's name as it appears in logs (optional, improves opponent detection)",
    )
    parser.add_argument(
        "--output",
        default="learned_params.json",
        help="Where to write learned parameters (default: learned_params.json)",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"Logs dir not found: {logs_dir}")
        return

    log_files = sorted(list(logs_dir.glob("*.log")) + list(logs_dir.glob("*.log.gz")))
    if not log_files:
        print(f"No .log or .log.gz files in {logs_dir}")
        return

    stats = {"showdowns": 0, "bluffs": 0, "by_opponent": {}}

    for path in log_files:
        print(f"Parsing {path.name}...")
        parse_log(path, args.bot_name, stats)

    if stats["showdowns"] == 0:
        print("No usable showdowns found; leaving opp_bluff_prior unchanged.")
        return

    bluff_ratio = stats["bluffs"] / stats["showdowns"]
    bluff_ratio = max(0.0, min(1.0, bluff_ratio))

    print(f"Total showdowns: {stats['showdowns']}")
    print(f"Bluffy showdowns (weak hand): {stats['bluffs']}")
    print(f"Inferred opp_bluff_prior: {bluff_ratio:.3f}")

    out = {"opp_bluff_prior": bluff_ratio}
    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

