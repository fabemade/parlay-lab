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
import math
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
STRAIGHTS_PER_LEAGUE = 5


def log(*a):
    print(*a, flush=True)


def run_sport(sport, days: list[dt.date], today: dt.date) -> tuple[list[dict], dict, list[dict], dict]:
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
        return [], results, [], {}
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
    legs, game_rows, models = [], [], {}
    for g in games:
        try:
            gm = GameModel(sport, g, R, [h for h in hist if ratings.competitive(h)], last, extras)
            gl = build_legs(gm)
        except Exception as e:  # one bad game must never sink the whole run
            log(f"    ! skipped {g.get('short')}: {e}")
            continue
        legs += gl
        models[g["id"]] = gm
        game_rows.append({"id": g["id"], "sport": sport.key, "short": g["short"], "name": g["name"],
                          "start": g["date"], "home": g["home"], "away": g["away"],
                          "exp_home": round(gm.exp[0], 2), "exp_away": round(gm.exp[1], 2),
                          "venue": g.get("venue"), "note": g.get("note"), "broadcast": g.get("broadcast"),
                          "context": gm.game_lines()})
    return legs, results, game_rows, models


GAME_FIELDS = ("home", "away", "game", "short", "start", "venue", "broadcast", "note", "context",
               "sport_name", "headline", "link")


def slim(leg: dict) -> dict:
    """Drop per-game fields repeated on every leg; the app re-attaches them from `games`."""
    out = {k: v for k, v in leg.items() if k not in GAME_FIELDS and v is not None}
    if leg.get("group", "Game lines") == "Game lines":
        out.pop("group", None)
    return out


MAX_DISAGREEMENT = 0.15   # model vs Kalshi gap above which we assume the model is missing news


def candidate(l: dict) -> bool:
    """Bets allowed into Top Picks.

    Only bet types with a track record (see grade.trusted): full-game lines are calibrated
    by the backtest, other types must first hit as often as predicted over enough graded
    predictions. And when our model disagrees with Kalshi by a lot, the usual reason is
    news the model can't see (injury, lineup, pitch count), not a giant edge, so skip it.
    """
    if not l.get("modeled", True) or not grade.trusted(l["sport"], l.get("group")):
        return False
    pm = l.get("p_model")
    return pm is None or abs(pm - l["p_market"]) <= MAX_DISAGREEMENT


def pick_top(all_legs: list[dict], today: dt.date) -> tuple[dict, list[str]]:
    """The day's featured parlays and straight bets, from games today and tomorrow only."""
    window = {today, today + dt.timedelta(days=1)}
    legs = [l for l in all_legs if candidate(l)
            and dt.datetime.fromisoformat(l["start"].replace("Z", "+00:00")).astimezone(ET).date() in window]
    featured, used = {}, set()
    # Only feature a parlay when every leg is +EV and the whole ticket still has a
    # realistic chance. On thin or sharply priced slates, showing nothing is the right call.
    # Each featured parlay uses different legs, so one miss can't sink every ticket.
    fresh = lambda pool: [l for l in pool if l["id"] not in used]
    specs = [("safest_2", legs, dict(n=2, mode="safest", target_american=100), 0.30),
             ("safest_3", legs, dict(n=3, mode="safest", target_american=150), 0.18),
             ("props_3", [l for l in legs if l.get("market") == "prop"], dict(n=3, mode="safest", target_american=150), 0.18),
             ("value_3", legs, dict(n=3, mode="value", min_hit=0.2), 0.0),
             ("safest_4", legs, dict(n=4, mode="safest", target_american=250), 0.10)]
    for name, pool, kw, floor in specs:
        n = kw.pop("n")
        p = parlay.build(fresh(pool), n, min_edge=0.005, **kw)
        if p and p["p"] >= floor:
            featured[name] = p
            used.update(p["legs"])
    # straight bets: the best few per league, at most one per game, so one game going
    # wrong (or two strikes on the same player) can't take out several picks at once
    straights, per, games = [], {}, set()
    for l in sorted(parlay.eligible(legs, min_edge=0.005), key=lambda l: -l["ev"]):
        if l["p"] >= 0.45 and per.get(l["sport"], 0) < STRAIGHTS_PER_LEAGUE and l["game_id"] not in games:
            straights.append(l["id"])
            games.add(l["game_id"])
            per[l["sport"]] = per.get(l["sport"], 0) + 1
    return featured, straights


def validate(legs: list[dict], featured: dict) -> list[str]:
    """Every published number must follow from Kalshi's price and our probability.

    Catches bugs before they reach the app: odds that don't match the price, an edge that
    doesn't equal probability minus break-even, a parlay whose payout isn't the product of
    its legs. Any problem stops the publish (the site keeps the last good board).
    """
    from engine.odds import american_to_decimal, decimal_to_american
    problems = []
    by_id = {l["id"]: l for l in legs}
    for l in legs:
        where = f"{l['id']} ({l.get('selection')})"
        p, be = l["p"], l.get("p_breakeven")
        if not 0 < p < 1:
            problems.append(f"{where}: probability {p} outside (0, 1)")
        if l.get("p_model") is not None:
            lo, hi = sorted((l["p_model"], l["p_market"]))
            if not lo - 2e-3 <= p <= hi + 2e-3:
                problems.append(f"{where}: blended {p} not between model {l['p_model']} and market {l['p_market']}")
        k = l.get("kalshi")
        if k and be is not None:
            price = k["price"] / 100
            if abs(be - (price + 0.07 * price * (1 - price))) > 0.006:
                problems.append(f"{where}: break-even {be} doesn't match {k['price']}¢ plus fee")
            # American odds are whole numbers: near -150 one point is ~0.15% of probability, and at
            # +3000 a 4th-decimal rounding of the break-even moves the odds a few points. Accept
            # either agreement: within one odds point, or within 0.01% of probability.
            if (abs(l["odds"] - decimal_to_american(1 / be)) > 1
                    and abs(1 / american_to_decimal(l["odds"]) - be) > 1e-4):
                problems.append(f"{where}: odds {l['odds']} don't match break-even {be}")
            if abs(l["edge"] - (p - be)) > 1.5e-3 or abs(l["ev"] - (p / be - 1)) > 3e-3:
                problems.append(f"{where}: edge/EV inconsistent with probability and price")
    for name, par in featured.items():
        ls = [by_id.get(i) for i in par["legs"]]
        if None in ls:
            problems.append(f"{name}: references a bet that isn't published")
            continue
        if len({l["game_id"] for l in ls}) < len(ls):
            problems.append(f"{name}: two legs from the same game")
        pp = math.prod(l["p"] for l in ls)
        dd = math.prod(american_to_decimal(l["odds"]) for l in ls)
        if abs(pp - par["p"]) > 2e-3 or abs(dd - par["decimal"]) > 0.02 * dd:
            problems.append(f"{name}: parlay odds/probability aren't the product of its legs")
        if any(l["edge"] < 0.005 for l in ls):
            problems.append(f"{name}: includes a leg without an edge")
    return problems


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

    all_legs, all_games, results_by_sport, all_models = [], [], {}, {}
    for key in keys:
        sport = SPORTS[key]
        log(f"[{sport.name}]")
        try:
            legs, results, games, models = run_sport(sport, days, today)
        except Exception as e:
            log(f"  ! {sport.name} failed: {e}")
            continue
        all_legs += legs
        all_games += games
        all_models.update(models)
        results_by_sport[key] = results

    # The app only offers what Kalshi lists: every leg is a Kalshi contract at Kalshi's price.
    # DraftKings lines still feed the model (market blend, reasons) for each game.
    log("[Kalshi]")
    try:
        kalshi_legs = kalshi.build_legs(all_models, all_legs, log=log)
    except Exception as e:
        log(f"  ! Kalshi lookup failed, publishing sportsbook lines instead: {e}")
        kalshi_legs = None
    if kalshi_legs is not None:
        all_legs = kalshi_legs
    finalize(all_legs)
    # drop games that already started
    now_utc = dt.datetime.now(dt.timezone.utc)
    all_legs = [l for l in all_legs if dt.datetime.fromisoformat(l["start"].replace("Z", "+00:00")) > now_utc]
    all_legs.sort(key=lambda l: (-l["p"] - l["edge"]))

    legs_by_id = {l["id"]: l for l in all_legs}

    # ---- Top Picks: chosen once a day (first refresh after grade.LOCK_HOUR Eastern) from
    # games today and tomorrow, then frozen so the record grades exactly what was shown.
    day = today.isoformat()
    top_log = grade.load(day)
    featured, straights = {}, []
    # only a full run (every league) may lock the day's picks
    if top_log is None and not args.sports and (args.date or now.hour >= grade.LOCK_HOUR):
        featured, straights = pick_top(all_legs, today)
        problems = validate(all_legs, featured)
        if problems:
            for msg in problems[:25]:
                log("  ✗ " + msg)
            log(f"\n{len(problems)} consistency problems; not publishing this board.")
            return 1
        top_log = grade.lock_picks(day, now.isoformat(timespec="minutes"), legs_by_id, featured, straights)
        grade.record_calibration(day, all_legs)
        log(f"  Locked today's Top Picks: {len(featured)} parlays, {len(straights)} straight bets")
    problems = validate(all_legs, {})
    if problems:
        for msg in problems[:25]:
            log("  ✗ " + msg)
        log(f"\n{len(problems)} consistency problems; not publishing this board.")
        return 1
    log(f"  ✓ consistency check passed ({len(all_legs)} bets)")
    grade.update_closing(legs_by_id)
    grade.grade_all(results_by_sport)
    grade.feedback()

    # Modeled bets go in picks.json (the builder searches them). Bets we can only list at
    # Kalshi's price go in one small file per game, loaded when that game is opened.
    game_ids = {l["game_id"] for l in all_legs}
    all_games = [g for g in all_games if g["id"] in game_ids]
    for l in all_legs:   # the app labels bet types that haven't earned a track record yet
        if l.get("modeled", True) and not grade.trusted(l["sport"], l.get("group")):
            l["untested"] = True
    main_legs = [slim(l) for l in all_legs if l.get("modeled", True)]
    other: dict[str, list] = {}
    for l in all_legs:
        if not l.get("modeled", True):
            other.setdefault(l["game_id"], []).append(slim(l))
    games_dir = OUT.parent / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    for f in games_dir.glob("*.json"):
        f.unlink()
    for gid, ls in other.items():
        (games_dir / f"{gid}.json").write_text(json.dumps(ls, separators=(",", ":")))
    for g in all_games:
        g["n_other"] = len(other.get(g["id"], []))
    payload = {
        "generated_at": now.isoformat(timespec="minutes"),
        "date": today.isoformat(),
        "sports": [k for k in SPORTS if k in {l["sport"] for l in all_legs}],
        "sport_names": {k: SPORTS[k].name for k in {l["sport"] for l in all_legs}},
        "games": all_games,
        "legs": main_legs,
        "top": grade.top_payload(top_log or grade.latest(), legs_by_id),
        "record": grade.summary(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    log(f"\n{len(all_legs)} bets across {len(all_games)} games ({len(main_legs)} modeled, "
        f"{len(all_legs) - len(main_legs)} at market price) in {time.time() - t0:.0f}s -> {OUT.relative_to(ROOT)} "
        f"({OUT.stat().st_size / 1e6:.1f} MB)")

    for p in (payload["top"] or {}).get("parlays", []):
        log(f"\n== {p['name']}: {p['american']:+d}  ({p['p'] * 100:.0f}% to hit)  result: {p['result'] or 'pending'}")
        for l in p["legs"]:
            log(f"   [{l.get('grade')}] {l['sport']:<14} {l['short']:<14} {l['selection']:<40} {l.get('result') or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
