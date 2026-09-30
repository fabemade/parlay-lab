"""Kalshi game markets: find each game's Kalshi page and Yes price so the app can link to it.

Read-only public endpoints (no account or API key). Kalshi combos can't be pre-filled from
outside the app, so the goal is just to put the right market one tap away. Anything that
fails here only means a game shows no Kalshi link; it never affects the picks.
"""
from __future__ import annotations

import datetime as dt
import re
from zoneinfo import ZoneInfo

import requests

API = "https://api.elections.kalshi.com/trade-api/v2"
ET = ZoneInfo("America/New_York")

# Kalshi "game winner" series per league. A wrong or missing ticker just finds no events,
# and the log line for that league says so.
SERIES = {
    "nfl": "KXNFLGAME", "cfb": "KXNCAAFGAME", "nba": "KXNBAGAME", "wnba": "KXWNBAGAME",
    "ncaab": "KXNCAAMBGAME", "mlb": "KXMLBGAME", "nhl": "KXNHLGAME",
    "soccer_eng.1": "KXEPLGAME", "soccer_esp.1": "KXLALIGAGAME", "soccer_ita.1": "KXSERIEAGAME",
    "soccer_ger.1": "KXBUNDESLIGAGAME", "soccer_fra.1": "KXLIGUE1GAME", "soccer_usa.1": "KXMLSGAME",
    "soccer_uefa.champions": "KXUCLGAME", "soccer_mex.1": "KXLIGAMXGAME",
}
MONTHS = {m: i for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}

_session = requests.Session()


def _get(path: str, params: dict | None = None) -> dict:
    r = _session.get(API + path, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _events(series: str) -> list[dict]:
    out, cursor = [], None
    for _ in range(10):
        params = {"series_ticker": series, "status": "open", "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        d = _get("/events", params)
        out += d.get("events") or []
        cursor = d.get("cursor")
        if not cursor:
            break
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "game").lower()).strip("-") or "game"


def _words(s: str | None) -> list[str]:
    w = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower().replace("&", " and ")).split()
    if w and w[-1] == "st":  # "Texas St." -> "texas state" (but leave "St. Louis" alone)
        w[-1] = "state"
    return w


def _is_team(label: str | None, team: dict) -> bool:
    """Kalshi labels teams by city ("Pittsburgh"), full name, or abbreviation."""
    lab = " ".join(_words(label))
    if not lab:
        return False
    if lab == (team.get("abbr") or "").lower():
        return True
    name = " ".join(_words(team.get("name")))
    return f" {lab} " in f" {name} "


def _event_day(event_ticker: str) -> dt.date | None:
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", event_ticker or "")
    if not m or m.group(2) not in MONTHS:
        return None
    try:
        return dt.date(2000 + int(m.group(1)), MONTHS[m.group(2)], int(m.group(3)))
    except ValueError:
        return None


def _cents(market: dict) -> int | None:
    if market.get("yes_ask") not in (None, 0):
        return int(market["yes_ask"])
    if market.get("yes_ask_dollars"):
        try:
            return round(float(market["yes_ask_dollars"]) * 100)
        except ValueError:
            return None
    return None


def attach(games: list[dict], legs: list[dict], log=print) -> None:
    """Add leg["kalshi"] = {"url", "price"?} for every leg whose game is listed on Kalshi."""
    by_sport: dict[str, list[dict]] = {}
    for g in games:
        by_sport.setdefault(g["sport"], []).append(g)
    found: dict[str, dict] = {}  # game id -> {"url", "prices": {"home"/"away"/"draw": cents}}

    for sport, sport_games in by_sport.items():
        series = SERIES.get(sport)
        if not series:
            continue
        try:
            events = _events(series)
            try:
                title = (_get(f"/series/{series}").get("series") or {}).get("title")
            except requests.RequestException:
                title = None
        except requests.RequestException as e:
            log(f"  Kalshi {series}: request failed ({e})")
            continue
        matched = 0
        for g in sport_games:
            day = dt.datetime.fromisoformat(g["start"].replace("Z", "+00:00")).astimezone(ET).date()
            for ev in events:
                ev_day = _event_day(ev.get("event_ticker", ""))
                if ev_day and abs((ev_day - day).days) > 1:
                    continue
                prices, sides = {}, set()
                for m in ev.get("markets") or []:
                    label = m.get("yes_sub_title") or m.get("subtitle")
                    code = (m.get("ticker") or "").rsplit("-", 1)[-1]  # market tickers end in a team code
                    for side in ("home", "away"):
                        if _is_team(label, g[side]) or (code and code == (g[side].get("abbr") or "").upper()):
                            prices[side] = _cents(m)
                            sides.add(side)
                    if re.fullmatch(r"(tie|draw)", (label or "").strip().lower()):
                        prices["draw"] = _cents(m)
                if sides == {"home", "away"}:
                    ticker = ev["event_ticker"]
                    found[g["id"]] = {
                        "url": f"https://kalshi.com/markets/{series.lower()}/{_slug(title)}/{ticker.lower()}",
                        "prices": prices,
                    }
                    matched += 1
                    break
        log(f"  Kalshi {series}: {len(events)} open events, matched {matched} of {len(sport_games)} games")

    for leg in legs:
        k = found.get(leg["game_id"])
        if not k:
            continue
        leg["kalshi"] = {"url": k["url"]}
        if leg["market"] == "ml" and k["prices"].get(leg["side"]):
            leg["kalshi"]["price"] = k["prices"][leg["side"]]


def probe(log=print) -> None:
    """Temporary: log what Kalshi's spread/total markets look like, to build matching on."""
    import json as _json
    leagues = ["NFL", "NCAAF", "MLB", "NHL", "WNBA", "MLS", "NBA", "EPL"]
    for lg in leagues:
        for kind in ("SPREAD", "TOTAL", "RUNLINE", "PUCKLINE", "OU"):
            series = f"KX{lg}{kind}"
            try:
                evs = _events(series)
            except requests.RequestException as e:
                continue
            if not evs:
                continue
            ev = evs[0]
            ms = ev.get("markets") or []
            log(f"  PROBE {series}: {len(evs)} events; event={_json.dumps({k: ev.get(k) for k in ('event_ticker', 'title', 'sub_title')})}")
            for m in ms[:3]:
                keep = {k: m.get(k) for k in ("ticker", "title", "subtitle", "yes_sub_title", "no_sub_title",
                                               "floor_strike", "cap_strike", "strike_type", "yes_ask", "yes_ask_dollars",
                                               "no_ask", "no_ask_dollars", "custom_strike")}
                log(f"    {_json.dumps(keep)}")
