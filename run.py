"""Daily pipeline: fetch data -> fit models -> price every bet -> build parlays -> publish.

Usage:
  python run.py                       # today + tomorrow, all sports in season
  python run.py --sports nfl,mlb      # only some sports
  python run.py --date 2026-10-04     # a specific day
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

from engine import espn, grade, kalshi, mlb, parlay, ratings
from engine.picks import GameModel, build_legs, finalize
from engine.sports import SPORTS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "site" / "data" / "picks.json"
ET = ZoneInfo("America/New_York")


def log(*a):
    print(*a, flush=True)


def run_sport(sport, days: list[dt.date], today: dt.date) -> tuple[list[dict], dict, list[dict]]:
    found = [g for d in days for g in espn.upcoming(sport, d)]
    games = [g for g in found if g.get("odds") and g.get("season_type") != 1]
    hist = espn.update_history(sport, today, log=log)
    results = {g["id"]: g for g in hist}
    if not games:
        # say why a league is missing from the app instead of dropping it silently
        no_odds = sum(1 for g in found if not g.get("odds"))
        preseason = sum(1 for g in found if g.get("odds") and g.get("season_type") == 1)
        log(f"  {sport.name}: no games to price ({len(found)} upcoming, {no_odds} without betting lines, "
            f"{preseason} preseason)")
        return [], results, []
    log(f"  {sport.name}: {len(games)} upcoming games with odds, {len(hist)} games of history")
    R = ratings.fit(sport, hist, today)
    extras: dict = {}
    if sport.key == "mlb":
        extras["pitchers"] = {}
        for d in days:
            extras["pitchers"].update(mlb.probable_pitchers(d))
        extras["team_fip"] = mlb.team_fip(today.year)
        extras["parks"] = mlb.park_factors(hist)
    with ThreadPoolExecutor(max_workers=10) as ex:
        extras["ctx"] = dict(zip([g["id"] for g in games],
                                 ex.map(lambda g: espn.context(sport, g["id"]), games)))
    last = ratings.last_game_dates([g for g in hist if ratings.competitive(g)])
    legs, game_rows = [], []
    for g in games:
        try:
            gm = GameModel(sport, g, R, [h for h in hist if ratings.competitive(h)], last, extras)
            gl = build_legs(gm)
        except Exception as e:  # one bad game must never sink the whole run
            log(f"    ! skipped {g.get('short')}: {e}")
            continue
        legs += gl
        game_rows.append({"id": g["id"], "sport": sport.key, "short": g["short"], "name": g["name"],
                          "start": g["date"], "home": g["home"], "away": g["away"],
                          "exp_home": round(gm.exp[0], 2), "exp_away": round(gm.exp[1], 2),
                          "venue": g.get("venue"), "note": g.get("note")})
    return legs, results, game_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default: today, US Eastern)")
    ap.add_argument("--days", type=int, default=4, help="how many days ahead to include")
    ap.add_argument("--sports", help="comma-separated keys, e.g. nfl,mlb,nba (default: all)")
    args = ap.parse_args()

    now = dt.datetime.now(ET)
    today = dt.date.fromisoformat(args.date) if args.date else now.date()
    days = [today + dt.timedelta(days=i) for i in range(args.days)]
    keys = args.sports.split(",") if args.sports else list(SPORTS)
    t0 = time.time()

    all_legs, all_games, results_by_sport = [], [], {}
    for key in keys:
        sport = SPORTS[key]
        log(f"[{sport.name}]")
        try:
            legs, results, games = run_sport(sport, days, today)
        except Exception as e:
            log(f"  ! {sport.name} failed: {e}")
            continue
        all_legs += legs
        all_games += games
        results_by_sport[key] = results

    finalize(all_legs)
    # drop games that already started
    now_utc = dt.datetime.now(dt.timezone.utc)
    all_legs = [l for l in all_legs if dt.datetime.fromisoformat(l["start"].replace("Z", "+00:00")) > now_utc]
    all_legs.sort(key=lambda l: (-l["p"] - l["edge"]))
    log("[Kalshi]")
    try:
        kalshi.attach(all_games, all_legs, log=log)
    except Exception as e:  # links are a convenience; never let them sink the run
        log(f"  ! Kalshi lookup failed: {e}")

    featured = {}
    # Only feature a parlay when every leg is +EV and the whole ticket still has a
    # realistic chance. On thin or sharply priced slates, showing nothing is the right call.
    for n, target, floor in ((2, 100, 0.30), (3, 150, 0.18), (4, 250, 0.10)):
        p = parlay.build(all_legs, n, mode="safest", target_american=target, min_edge=0.005)
        if p and p["p"] >= floor:
            featured[f"safest_{n}"] = p
    v = parlay.build(all_legs, 3, mode="value", min_hit=0.2, min_edge=0.005)
    if v:
        featured["value_3"] = v
    straights = [l["id"] for l in sorted(parlay.eligible(all_legs, min_edge=0.005),
                                         key=lambda l: -l["ev"]) if l["p"] >= 0.45][:8]

    featured_ids = {lid for p in featured.values() for lid in p["legs"]} | set(straights)
    grade.grade_all(results_by_sport)
    if featured_ids:
        grade.record_picks(today.isoformat(), all_legs, list(featured.values()), featured_ids)

    payload = {
        "generated_at": now.isoformat(timespec="minutes"),
        "date": today.isoformat(),
        "sports": sorted({l["sport"] for l in all_legs}),
        "sport_names": {k: SPORTS[k].name for k in {l["sport"] for l in all_legs}},
        "games": all_games,
        "legs": all_legs,
        "featured": featured,
        "straights": straights,
        "record": grade.summary(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    log(f"\n{len(all_legs)} bets priced across {len(all_games)} games in {time.time() - t0:.0f}s -> {OUT.relative_to(ROOT)}")

    by_id = {l["id"]: l for l in all_legs}
    for name, p in featured.items():
        log(f"\n== {name}: {p['american']:+d}  ({p['p'] * 100:.0f}% to hit by our model)")
        for lid in p["legs"]:
            l = by_id[lid]
            log(f"   [{l['grade']}] {l['sport_name']:<16} {l['short']:<14} {l['selection']:<18} {l['odds_str']:>5}  {l['headline']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
