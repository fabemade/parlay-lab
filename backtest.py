"""Walk-forward backtest + calibration.

For each sport we step through the history one week at a time: fit ratings using only
games *before* that week, predict the week's games, then compare to what happened.
Nothing from the future leaks into a prediction.

Outputs (data/calibration.json), used by the live model:
  margin_k / total_k : how much to stretch or shrink predicted margins and totals
                       (ridge shrinkage tends to make predictions too timid)
  margin_sd / total_sd: the real spread of outcomes around our predictions
and prints accuracy, Brier score and log loss versus a naive home-team baseline.

Usage: python backtest.py [--sports nfl,mlb]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

from engine import espn, ratings
from engine.dist import Outcome
from engine.sports import SPORTS

CAL_PATH = Path(__file__).resolve().parent / "data" / "calibration.json"


def backtest(sport, today: dt.date, min_train: int = 60):
    hist = [g for g in espn.update_history(sport, today, log=lambda *a: None)
            if ratings.competitive(g)]
    if len(hist) < min_train * 2:
        return None
    hist.sort(key=lambda g: g["date"])
    start = dt.date.fromisoformat(hist[min_train]["date"][:10])
    rows = []
    week = start
    while week < today:
        nxt = week + dt.timedelta(days=7)
        train = [g for g in hist if g["date"][:10] < week.isoformat()]
        test = [g for g in hist if week.isoformat() <= g["date"][:10] < nxt.isoformat()]
        if test and len(train) >= min_train:
            R = ratings.fit(sport, train, week)
            for g in test:
                if R.games.get(g["home"], 0) < 3 or R.games.get(g["away"], 0) < 3:
                    continue
                eh, ea = R.expected(g["home"], g["away"], g.get("neutral"))
                rows.append((eh, ea, g["hs"], g["as"]))
        week = nxt
    return np.array(rows)


def evaluate(sport, rows: np.ndarray) -> dict:
    eh, ea, hs, as_ = rows.T
    pm, am = eh - ea, hs - as_
    pt, at = eh + ea, hs + as_
    # calibration slopes (least squares, centred)
    k_m = float(np.dot(pm - pm.mean(), am - am.mean()) / np.dot(pm - pm.mean(), pm - pm.mean()))
    k_t = float(np.dot(pt - pt.mean(), at - at.mean()) / max(np.dot(pt - pt.mean(), pt - pt.mean()), 1e-9))
    k_m, k_t = float(np.clip(k_m, 0.7, 1.6)), float(np.clip(k_t, 0.3, 1.5))

    def probs(k):
        out = []
        for h, a in zip(eh, ea):
            m, t = (h - a) * k, h + a
            h2, a2 = max((t + m) / 2, 0.05), max((t - m) / 2, 0.05)
            o = Outcome(sport, h2, a2)
            ph = o.p_home_win()
            # soccer: score only decided games, so condition on "not a draw"
            out.append(ph / (ph + o.p_away_win()) if sport.three_way else ph)
        return np.clip(np.array(out), 1e-4, 1 - 1e-4)

    decided = am != 0
    y = (am > 0).astype(float)[decided]
    base = np.full(decided.sum(), y.mean())
    res = {"n": int(len(rows)), "margin_k": round(k_m, 3), "total_k": round(k_t, 3),
           "home_win_rate": round(float(y.mean()), 3)}
    for label, k in (("raw", 1.0), ("calibrated", k_m)):
        p = probs(k)[decided]
        res[f"brier_{label}"] = round(float(np.mean((p - y) ** 2)), 4)
        res[f"logloss_{label}"] = round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4)
        res[f"accuracy_{label}"] = round(float(np.mean((p > 0.5) == (y == 1))), 3)
    res["brier_baseline"] = round(float(np.mean((base - y) ** 2)), 4)
    # residual spreads after calibration
    res["margin_sd"] = round(float(np.std(am - (pm - pm.mean()) * k_m - pm.mean())), 2)
    res["total_sd"] = round(float(np.std(at - (pt.mean() + (pt - pt.mean()) * k_t))), 2)
    res["total_mean_pred"] = round(float(pt.mean()), 3)
    # calibration table: predicted bucket -> actual win rate
    p = probs(k_m)[decided]
    table = []
    for lo in np.arange(0.2, 0.8, 0.1):
        m = (p >= lo) & (p < lo + 0.1)
        if m.sum() >= 15:
            table.append([f"{lo:.0%}-{lo + 0.1:.0%}", int(m.sum()), round(float(y[m].mean()), 3)])
    res["calibration_table"] = table
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sports")
    ap.add_argument("--date")
    args = ap.parse_args()
    today = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    keys = args.sports.split(",") if args.sports else list(SPORTS)
    cal = json.loads(CAL_PATH.read_text()) if CAL_PATH.exists() else {}
    for key in keys:
        sport = SPORTS[key]
        rows = backtest(sport, today)
        if rows is None or len(rows) < 100:
            print(f"{sport.name}: not enough history")
            continue
        r = evaluate(sport, rows)
        cal[key] = {k: r[k] for k in ("margin_k", "total_k", "margin_sd", "total_sd", "n")}
        cal[key]["updated"] = today.isoformat()
        print(f"\n{sport.name}  ({r['n']} games predicted out-of-sample)")
        print(f"  accuracy  raw {r['accuracy_raw']:.1%}  calibrated {r['accuracy_calibrated']:.1%}   (home team wins {r['home_win_rate']:.1%})")
        print(f"  brier     raw {r['brier_raw']}  calibrated {r['brier_calibrated']}  baseline {r['brier_baseline']}  (lower is better)")
        print(f"  log loss  raw {r['logloss_raw']}  calibrated {r['logloss_calibrated']}")
        print(f"  margin stretch k={r['margin_k']}  total stretch k={r['total_k']}  margin sd {r['margin_sd']}  total sd {r['total_sd']}")
        for row in r["calibration_table"]:
            print(f"    predicted {row[0]:>9}: {row[1]:4d} games, actual {row[2]:.1%}")
    CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAL_PATH.write_text(json.dumps(cal, indent=1))


if __name__ == "__main__":
    main()
