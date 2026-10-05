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
from engine.odds import american_to_decimal
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
    def ctx(g):   # matchup extras (injuries, predictor); a malformed one must not sink the league
        try:
            return espn.context(sport, g["id"])
        except Exception as e:
            log(f"    ! no matchup context for {g.get('short')}: {e}")
            return {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        extras["ctx"] = dict(zip([g["id"] for g in games], ex.map(ctx, games)))
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
    # Full-game lines only: props and period bets "earn" trust from many correlated
    # predictions over a few dozen games, and props went 7-13 when we actually picked them.
    if (l.get("group") or grade.TESTED) != grade.TESTED:
        return False
    if not l.get("modeled", True) or not grade.trusted(l["sport"], l.get("group")):
        return False
    pm = l.get("p_model")
    return pm is None or abs(pm - l["p_market"]) <= MAX_DISAGREEMENT


# Locks: bets both our model and Kalshi rate very likely. Nothing is guaranteed, so the bar
# is high and the payout is small: an 85% lock still misses about 1 time in 7.
LOCK_P = 0.80              # our (trained) probability
LOCK_MARKET_P = 0.78       # and Kalshi's must agree
LOCK_AGREE = 0.08          # model and market within 8 points of each other
LOCK_MAX_PRICE = 95        # cents; above that the payout is too small to be worth a slot
LOCK_MIN_EDGE = -0.03      # can sit a little under Kalshi's price, never far under
LOCKS_MAX, LOCKS_PER_SPORT = 12, 3
# Kalshi lists dozens of over/under strikes per game, so ranked purely by probability the
# card fills with "at least 1 goal"-type totals at 95c. Winners and spreads come first;
# totals are capped so the locks stay varied.
LOCK_TOTALS_MAX, LOCK_TOTALS_PER_SPORT = 5, 2


def is_lock(l: dict) -> bool:
    k = l.get("kalshi") or {}
    pm = l.get("p_model")
    return (candidate(l) and l["p"] >= LOCK_P and l["p_market"] >= LOCK_MARKET_P
            and pm is not None and abs(pm - l["p_market"]) <= LOCK_AGREE
            and l["edge"] >= LOCK_MIN_EDGE and k.get("price", 100) <= LOCK_MAX_PRICE)


TOP_P, TOP_MIN_EDGE = 0.55, -0.02     # Top picks: likely, and priced near fair or better
TOP_STRAIGHTS, TOP_PER_SPORT = 10, 2


def _varied(cands: list[dict], n_max: int, per_sport: int, games: set, totals_max: int,
            totals_per_sport: int) -> list[str]:
    """Likeliest first, one per game, a few per sport, and winners/spreads before totals:
    Kalshi lists dozens of total strikes per game, which would otherwise crowd everything out."""
    out, per, totals = [], {}, {}
    by_conf = sorted(cands, key=lambda l: (-l["p"], -l["edge"]))
    for l in [l for l in by_conf if l["market"] != "total"] + [l for l in by_conf if l["market"] == "total"]:
        if per.get(l["sport"], 0) >= per_sport or l["game_id"] in games or len(out) >= n_max:
            continue
        if l["market"] == "total" and (sum(totals.values()) >= totals_max
                                       or totals.get(l["sport"], 0) >= totals_per_sport):
            continue
        out.append(l["id"])
        games.add(l["game_id"])
        per[l["sport"]] = per.get(l["sport"], 0) + 1
        if l["market"] == "total":
            totals[l["sport"]] = totals.get(l["sport"], 0) + 1
    return out


def pick_top(all_legs: list[dict], today: dt.date) -> tuple[dict, list[str], list[str]]:
    """The day's card: Top picks and Locks, for games before the next day's card.

    Picks are chosen by how likely they are, not by how far our model sits from Kalshi's
    price: graded results showed the "edge" picks hit exactly what Kalshi said (50%), not
    what we said (54%). Selecting on disagreement mostly selects the model's noise.

    Top picks: mixed parlays (2, 3, 4 legs, across sports) and straight bets, each leg 55%+
    and priced near fair or better.
    Locks: bets both our model and Kalshi rate 80%+, as singles and lock parlays.
    No leg is reused between parlays, and no game appears in both lists of straights.
    """
    # Each card covers games until the next card (24h). With "today and tomorrow", a
    # Saturday game was picked on Friday's card and again on Saturday's, and counted twice.
    until = dt.datetime.combine(today + dt.timedelta(days=1), dt.time(grade.LOCK_HOUR), ET)
    legs = [l for l in all_legs if candidate(l)
            and dt.datetime.fromisoformat(l["start"].replace("Z", "+00:00")) < until]
    lock_pool = [l for l in legs if is_lock(l)]
    top_pool = [l for l in legs if l["p"] >= TOP_P and l["edge"] >= TOP_MIN_EDGE and not is_lock(l)]
    featured, used = {}, set()
    fresh = lambda ls: [l for l in ls if l["id"] not in used]

    def band(pool: list[dict], n: int, target: float) -> list[dict]:
        # A likeliest-first search never looks past the shortest prices, which can't reach a
        # payout target together (two 95c legs pay about -900). An n-leg ticket paying
        # `target` needs legs costing at most target_decimal ** (-1/n) each.
        cap = american_to_decimal(target) ** (-1 / n)
        return [l for l in pool if l["p_breakeven"] <= cap + 1e-9]

    def mix(pool, n):   # one leg per sport when the slate allows
        k = len({l["sport"] for l in pool})
        return 1 if k >= n else 2
    kw = dict(mode="safest", min_odds=-5000, max_odds=300, pool=45)
    specs = [
        # name, tier, pool, legs, minimum payout, minimum hit chance
        ("lock_2", "lock", lock_pool, 2, -300, 0.65),
        ("lock_3", "lock", lock_pool, 3, -200, 0.55),
        ("lock_4", "lock", lock_pool, 4, 100, 0.45),
        ("best_2", "best", top_pool, 2, 100, 0.35),
        ("best_3", "best", top_pool, 3, 200, 0.24),
        ("best_4", "best", top_pool, 4, 350, 0.15),
    ]
    for name, tier, pool, n, target, floor in specs:
        cands = fresh(band(pool, n, target))
        p = parlay.build(cands, n, target_american=target, max_per_sport=mix(cands, n),
                         min_edge=LOCK_MIN_EDGE if tier == "lock" else TOP_MIN_EDGE, **kw)
        if p and p["p"] >= floor:
            featured[name] = {**p, "tier": tier}
            used.update(p["legs"])
    games: set = set()
    locks = _varied(lock_pool, LOCKS_MAX, LOCKS_PER_SPORT, games, LOCK_TOTALS_MAX, LOCK_TOTALS_PER_SPORT)
    top = _varied(top_pool, TOP_STRAIGHTS, TOP_PER_SPORT, games, 4, 1)
    return featured, top, locks


def apply_training(legs: list[dict]) -> int:
    """Our final probability for each modeled bet, after the calibration learned from results."""
    from engine.picks import confidence, summarize
    changed = 0
    for l in legs:
        if l.get("p_model") is None or not l.get("modeled", True):
            continue
        q = round(grade.learned_p(l["p"], l["p_market"], l.get("group")), 4)
        if q == l["p"]:
            continue
        be = l.get("p_breakeven") or (1 / (1 + (l["odds"] / 100 if l["odds"] > 0 else 100 / -l["odds"])))
        l.update(p_raw=l["p"], p=q, edge=round(q - be, 4), ev=round(q / be - 1, 4))
        l["grade"], l["headline"] = confidence(l), summarize(l)
        changed += 1
    return changed


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
            raw = l.get("p_raw", p)   # trained calibration may move the final number past the blend
            if not lo - 2e-3 <= raw <= hi + 2e-3:
                problems.append(f"{where}: blended {raw} not between model {l['p_model']} and market {l['p_market']}")
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
        floor = {"lock": LOCK_MIN_EDGE, "best": -0.02}.get(par.get("tier"), 0.005)
        if any(l["edge"] < floor - 1e-9 for l in ls):
            problems.append(f"{name}: includes a leg below its tier's edge floor")
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
            import traceback
            where = traceback.extract_tb(e.__traceback__)[-1]
            log(f"  ! {sport.name} failed: {e} ({Path(where.filename).name}:{where.lineno} {where.line})")
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
    learned = grade.train()
    for g, prm in learned["groups"].items():
        log(f"  trained {g}: {prm['n']} graded bets over {prm['games']} games -> k={prm['k']} T={prm['T']}"
            + (f" (loss {prm['loss_before']} -> {prm['loss_after']})" if "loss_before" in prm else f" ({prm.get('note', '')})"))
    log(f"  applied trained calibration to {apply_training(all_legs)} bets")
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
        featured, top, locks = pick_top(all_legs, today)
        problems = validate(all_legs, featured)
        if problems:
            for msg in problems[:25]:
                log("  ✗ " + msg)
            log(f"\n{len(problems)} consistency problems; not publishing this board.")
            return 1
        top_log = grade.lock_picks(day, now.isoformat(timespec="minutes"), legs_by_id, featured, [], best=top, locks=locks)
        grade.record_calibration(day, all_legs)
        log(f"  Locked today's card: {len(top)} top straights, {sum(p['tier'] == 'best' for p in featured.values())} top parlays, "
            f"{len(locks)} locks, {sum(p['tier'] == 'lock' for p in featured.values())} lock parlays")
    elif top_log is not None and "straights_best" not in top_log and not args.sports:
        # a day locked before the Top picks tier existed: add that tier from games not yet started
        featured, top, _ = pick_top(all_legs, today)
        featured = {n: p for n, p in featured.items() if p["tier"] == "best"}
        if not validate(all_legs, featured):
            top_log = grade.add_best_tier(day, legs_by_id, featured, top)
            log(f"  Added Top picks to today's card: {len(top_log['straights_best'])} straights, "
                f"{sum(p.get('tier') == 'best' for p in top_log['parlays'])} parlays")
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
