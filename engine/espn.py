"""ESPN public API client: schedules, results, DraftKings odds, injuries, matchup context.

Past days are cached on disk forever (results don't change), so the first run is slow
and every run after that only fetches today.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .odds import parse_american
from .sports import Sport

BASE = "https://site.api.espn.com/apis/site/v2/sports"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_DIR = DATA_DIR / "history"

_session = requests.Session()


def _get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (400, 404):
                return {}
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return {}


def scoreboard(sport: Sport, day: dt.date) -> dict:
    params = {"dates": day.strftime("%Y%m%d"), **sport.params}
    return _get(f"{BASE}/{sport.path}/scoreboard", params)


def summary(sport: Sport, event_id: str) -> dict:
    return _get(f"{BASE}/{sport.path}/summary", {"event": event_id})


# ---------------------------------------------------------------- parsing helpers

def _team(c: dict) -> dict:
    t = c["team"]
    return {"id": t["id"], "name": t.get("displayName", t.get("name")),
            "abbr": t.get("abbreviation", ""), "logo": t.get("logo", "")}


def _record(c: dict, kind: str) -> str | None:
    for r in c.get("records", []) or []:
        if r.get("type") == kind or r.get("name") == kind:
            return r.get("summary")
    return None


def parse_result(event: dict) -> dict | None:
    """Compact finished-game record used to fit team ratings."""
    comp = event["competitions"][0]
    if not comp["status"]["type"].get("completed"):
        return None
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    if "home" not in sides or "away" not in sides:
        return None
    try:
        hs, as_ = float(sides["home"]["score"]), float(sides["away"]["score"])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "id": event["id"], "date": event["date"],
        "season_type": (event.get("season") or {}).get("type"),
        "neutral": bool(comp.get("neutralSite")),
        "home": sides["home"]["team"]["id"], "away": sides["away"]["team"]["id"],
        "home_name": sides["home"]["team"].get("displayName"),
        "away_name": sides["away"]["team"].get("displayName"),
        "hs": hs, "as": as_,
    }


def _line(block: dict | None, key: str = "odds"):
    """Latest pre-game value from an ESPN odds block (current > close > open)."""
    if not block:
        return None
    for stage in ("current", "close", "open"):
        v = (block.get(stage) or {}).get(key)
        if v is not None:
            return v
    return None


def _open(block: dict | None, key: str = "odds"):
    return ((block or {}).get("open") or {}).get(key)


def _num(line: str | None) -> float | None:
    if line is None:
        return None
    try:
        return float(str(line).lstrip("ou").replace("+", ""))
    except ValueError:
        return None


def parse_odds(comp: dict) -> dict | None:
    """Pull moneyline / spread / total (with opening lines) from the DraftKings block."""
    odds_list = comp.get("odds") or []
    if not odds_list:
        return None
    o = odds_list[0]
    ml, ps, tot = o.get("moneyline") or {}, o.get("pointSpread") or {}, o.get("total") or {}
    out = {
        "provider": (o.get("provider") or {}).get("name", "DraftKings"),
        "ml_home": parse_american(_line(ml.get("home"))),
        "ml_away": parse_american(_line(ml.get("away"))),
        "ml_draw": parse_american(_line(ml.get("draw"))),
        "ml_home_open": parse_american(_open(ml.get("home"))),
        "ml_away_open": parse_american(_open(ml.get("away"))),
        "spread_home": _num(_line(ps.get("home"), "line")),
        "spread_home_odds": parse_american(_line(ps.get("home"))),
        "spread_away_odds": parse_american(_line(ps.get("away"))),
        "spread_home_open": _num(_open(ps.get("home"), "line")),
        "total": _num(_line(tot.get("over"), "line")),
        "over_odds": parse_american(_line(tot.get("over"))),
        "under_odds": parse_american(_line(tot.get("under"))),
        "total_open": _num(_open(tot.get("over"), "line")),
    }
    if out["ml_draw"] is None and (o.get("drawOdds") or {}).get("moneyLine") is not None:
        out["ml_draw"] = float(o["drawOdds"]["moneyLine"])
    if out["total"] is None and o.get("overUnder") is not None:
        out["total"] = float(o["overUnder"])
    # deep links to the bet slip, when present
    links = {}
    for key, block in (("ml_home", ml.get("home")), ("ml_away", ml.get("away")),
                       ("ml_draw", ml.get("draw")), ("spread_home", ps.get("home")),
                       ("spread_away", ps.get("away")), ("over", tot.get("over")),
                       ("under", tot.get("under"))):
        for stage in ("current", "close", "open"):
            href = (((block or {}).get(stage) or {}).get("link") or {}).get("href")
            if href:
                links[key] = href
                break
    out["links"] = links
    return out


def parse_upcoming(event: dict) -> dict | None:
    comp = event["competitions"][0]
    state = comp["status"]["type"].get("state")
    if state != "pre":
        return None
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    if "home" not in sides or "away" not in sides:
        return None
    game = {
        "id": event["id"], "date": event["date"], "name": event.get("name"),
        "short": event.get("shortName"),
        "season_type": (event.get("season") or {}).get("type"),
        "neutral": bool(comp.get("neutralSite")),
        "venue": (comp.get("venue") or {}).get("fullName"),
        "home": _team(sides["home"]), "away": _team(sides["away"]),
        "odds": parse_odds(comp),
        "broadcast": ", ".join(n for b in comp.get("broadcasts") or [] for n in (b or {}).get("names") or []),
        "note": ((comp.get("notes") or [{}])[0] or {}).get("headline"),
    }
    for side in ("home", "away"):
        c = sides[side]
        game[side]["record"] = _record(c, "total")
        game[side]["home_record"] = _record(c, "home")
        game[side]["road_record"] = _record(c, "road")
        probs = c.get("probables") or []
        if probs:
            a = probs[0].get("athlete") or {}
            game[side]["probable"] = {"name": a.get("fullName"), "espn_id": a.get("id")}
    return game


# ---------------------------------------------------------------- history cache

def _history_path(sport: Sport) -> Path:
    return HISTORY_DIR / f"{sport.history_key or sport.key}.json"


def load_history(sport: Sport) -> dict:
    p = _history_path(sport)
    if p.exists():
        return json.loads(p.read_text())
    return {"days": {}}


def update_history(sport: Sport, today: dt.date, log=print) -> list[dict]:
    """Make sure every past day in the rating window is cached; return all results."""
    hist = load_history(sport)
    days = hist["days"]
    start = today - dt.timedelta(days=sport.history_days)
    # Only days that are at least 2 days old are treated as final and cached.
    final_cutoff = today - dt.timedelta(days=2)
    wanted = [start + dt.timedelta(days=i) for i in range((today - start).days)]
    missing = [d for d in wanted if d.strftime("%Y%m%d") not in days]
    if missing:
        log(f"  {sport.name}: fetching {len(missing)} days of results…")

        paths = (sport.path,) + tuple(p for p in sport.history_paths if p != sport.path)

        def fetch(d):
            results, ok = [], False
            for path in paths:
                sb = _get(f"{BASE}/{path}/scoreboard", {"dates": d.strftime("%Y%m%d"), **sport.params})
                ok = ok or bool(sb)
                results += [r for e in sb.get("events", []) if (r := parse_result(e))]
            return d, results, ok

        with ThreadPoolExecutor(max_workers=12) as ex:
            for d, results, ok in ex.map(fetch, missing):
                key = d.strftime("%Y%m%d")
                if not ok:
                    continue
                if d <= final_cutoff:
                    days[key] = results
                else:
                    hist.setdefault("recent", {})[key] = results
    # drop days that fell out of the window
    for k in list(days):
        if k < start.strftime("%Y%m%d"):
            del days[k]
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    recent = hist.pop("recent", {})
    _history_path(sport).write_text(json.dumps(hist, separators=(",", ":")))
    games = [g for day in sorted(days) for g in days[day]]
    games += [g for day in sorted(recent) for g in recent[day]]
    seen, uniq = set(), []
    for g in games:
        if g["id"] not in seen:
            seen.add(g["id"])
            uniq.append(g)
    return uniq


def upcoming(sport: Sport, day: dt.date) -> list[dict]:
    sb = scoreboard(sport, day)
    return [g for e in sb.get("events", []) if (g := parse_upcoming(e))]


# ---------------------------------------------------------------- matchup context

INJURY_OUT = {"out", "injured reserve", "ir", "doubtful", "suspension", "60-day-il",
              "15-day-il", "10-day-il", "7-day-il", "day-to-day", "questionable"}


def context(sport: Sport, event_id: str) -> dict:
    """Injuries, last-5 form, ESPN matchup predictor, season series, ATS records."""
    s = summary(sport, event_id)
    if not s:
        return {}
    ctx: dict = {"injuries": {}, "last5": {}, "predictor": None, "series": None, "ats": {}}
    for block in s.get("injuries") or []:
        tid = (block.get("team") or {}).get("id")
        items = []
        for inj in block.get("injuries") or []:
            a = inj.get("athlete") or {}
            status = (inj.get("status") or "").strip()
            items.append({
                "name": a.get("displayName"),
                "pos": (a.get("position") or {}).get("abbreviation"),
                "status": status,
                "detail": ((inj.get("details") or {}).get("type") or ""),
            })
        ctx["injuries"][tid] = items
    for block in s.get("lastFiveGames") or []:
        tid = (block.get("team") or {}).get("id")
        ctx["last5"][tid] = [
            {"result": e.get("gameResult"), "score": e.get("score"),
             "opp": (e.get("opponent") or {}).get("abbreviation"), "at_vs": e.get("atVs")}
            for e in block.get("events") or []
        ]
    pred = s.get("predictor")
    if pred and pred.get("homeTeam", {}).get("gameProjection"):
        try:
            ctx["predictor"] = {
                "home": float(pred["homeTeam"]["gameProjection"]) / 100,
                "away": float(pred["awayTeam"]["gameProjection"]) / 100,
            }
        except (TypeError, ValueError):
            pass
    for ser in s.get("seasonseries") or []:
        if ser.get("summary"):
            ctx["series"] = ser["summary"]
            break
    for block in s.get("againstTheSpread") or []:
        tid = (block.get("team") or {}).get("id")
        ctx["ats"][tid] = {r.get("type"): r.get("summary") for r in block.get("records") or []}
    # starting goalies (NHL)
    goalies = s.get("goalies")
    if goalies:
        ctx["goalies"] = goalies
    return ctx
