"""Bias check: where does our model disagree with Kalshi on average?

For each market type, compares our model's probability with Kalshi's fair price across
every contract on today's board. A single bet disagreeing is the whole point of a model;
a whole market type leaning the same way (say, every quarter spread 8% low) means a
modelling mistake, not an edge. Run after run.py:

    python market_check.py            # market types with 5+ contracts
    python market_check.py --min 20
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path

DATA = Path(__file__).resolve().parent / "site" / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=5)
    args = ap.parse_args()
    legs = json.loads((DATA / "picks.json").read_text())["legs"]
    by = collections.defaultdict(list)
    for l in legs:
        if l.get("p_model") is None or (l.get("kalshi") or {}).get("buy") != "yes":
            continue
        by[(l["sport"], l.get("market_name") or l["market"])].append(l["p_model"] - l["p_market"])
    rows = sorted(((k, v) for k, v in by.items() if len(v) >= args.min), key=lambda kv: -abs(st.mean(kv[1])))
    print(f"{'league':14} {'market':30} {'n':>5} {'bias':>7} {'avg gap':>8}")
    for (sport, market), v in rows:
        flag = "  <- check" if abs(st.mean(v)) >= 0.04 else ""
        print(f"{sport:14} {market[:30]:30} {len(v):5d} {st.mean(v):+7.3f} {st.mean(abs(x) for x in v):8.3f}{flag}")


if __name__ == "__main__":
    main()
