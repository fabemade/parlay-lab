"""Kalshi markets: the bets the app offers are the contracts Kalshi actually lists.

Kalshi's game contracts differ from a sportsbook's: spreads only come as "TEAM wins by
over X" (several X per team), totals as "Over X" (an under is buying No), and each has
its own price. So instead of DraftKings' single line per market, every leg here is one
Kalshi contract at Kalshi's price, and our score distribution prices whatever strike
Kalshi lists.

Read-only public endpoints (no account or API key). Kalshi combos can't be pre-filled
from outside Kalshi's app, so each leg links to its market instead.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import requests

from .odds import blend, decimal_to_american, logit

API = "https://api.elections.kalshi.com/trade-api/v2"
ET = ZoneInfo("America/New_York")

# Series tickers per league: game winner, spread, total. A wrong ticker just finds no
# events, and the log line for that league says so.
LEAGUE = {
    "nfl": "NFL", "cfb": "NCAAF", "nba": "NBA", "wnba": "WNBA", "ncaab": "NCAAMB", "mlb": "MLB",
    "nhl": "NHL", "soccer_eng.1": "EPL", "soccer_esp.1": "LALIGA", "soccer_ita.1": "SERIEA",
    "soccer_ger.1": "BUNDESLIGA", "soccer_fra.1": "LIGUE1", "soccer_usa.1": "MLS",
    "soccer_uefa.champions": "UCL", "soccer_mex.1": "LIGAMX",
    "soccer_uefa.europa": "UEL", "soccer_uefa.europa.conf": "UECL",
    "soccer_ned.1": "EREDIVISIE", "soccer_por.1": "LIGAPORTUGAL", "soccer_bel.1": "BELGIANPL",
    "soccer_tur.1": "SUPERLIG", "soccer_sco.1": "SCOTTISHPREM",
    "soccer_eng.2": "EFLCHAMPIONSHIP", "soccer_eng.3": "EFLL1", "soccer_ger.2": "BUNDESLIGA2",
    "soccer_esp.2": "LALIGA2", "soccer_ita.2": "SERIEB", "soccer_fra.2": "LIGUE2",
    "soccer_bra.1": "BRASILEIRO", "soccer_arg.1": "ARGPREMDIV",
    "soccer_conmebol.libertadores": "CONMEBOLLIB", "soccer_ksa.1": "SAUDIPL",
}
KINDS = ("GAME", "SPREAD", "TOTAL")
MONTHS = {m: i for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}
FEE = 0.07              # Kalshi taker fee per contract: 0.07 x P x (1 - P)
PRICE_RANGE = (0.10, 0.90)
STRIKES_PER_SIDE = 3    # Kalshi lists many margins; keep a spread of them per team/side

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


# ESPN shortens Brazilian clubs by state ("Atlético-MG"); Kalshi spells it out.
STATE_SUFFIX = {"mg": "mineiro", "pr": "paranaense", "go": "goianiense"}


def _words(s: str | None) -> list[str]:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()  # São -> Sao
    w = re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("&", " and ")).split()
    w = [STATE_SUFFIX.get(x, x) if i else x for i, x in enumerate(w)]
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


def _matchup(event_ticker: str) -> str:
    """'KXNFLSPREAD-26OCT01PITCLE' -> '26OCT01PITCLE' (shared by a game's winner/spread/total events)."""
    return event_ticker.split("-", 1)[-1]


def _ask(market: dict, side: str) -> float | None:
    cents = market.get(f"{side}_ask")
    if cents not in (None, 0):
        return cents / 100
    try:
        v = float(market.get(f"{side}_ask_dollars") or "nan")
    except ValueError:
        return None
    return None if math.isnan(v) else v


def _cost(price: float) -> float:
    return price + FEE * price * (1 - price)


def _spread_out(items: list, k: int) -> list:
    if len(items) <= k:
        return items
    return [items[i] for i in sorted({round(x) for x in np.linspace(0, len(items) - 1, k)})]


class _Series:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.events = _events(ticker)
        try:
            self.title = (_get(f"/series/{ticker}").get("series") or {}).get("title")
        except requests.RequestException:
            self.title = None

    def url(self, event_ticker: str) -> str:
        return f"https://kalshi.com/markets/{self.ticker.lower()}/{_slug(self.title)}/{event_ticker.lower()}"


def _match_game(g: dict, events: list[dict]) -> tuple[dict, dict] | None:
    """Find the game's winner event; return it and a Kalshi team code -> side map."""
    day = dt.datetime.fromisoformat(g["date"].replace("Z", "+00:00")).astimezone(ET).date()
    for ev in events:
        ev_day = _event_day(ev.get("event_ticker", ""))
        if ev_day and abs((ev_day - day).days) > 1:
            continue
        codes = {}
        for m in ev.get("markets") or []:
            label = m.get("yes_sub_title") or m.get("subtitle")
            code = (m.get("ticker") or "").rsplit("-", 1)[-1]
            for side in ("home", "away"):
                if _is_team(label, g[side]) or code == (g[side].get("abbr") or "").upper():
                    codes[code] = side
        if set(codes.values()) == {"home", "away"}:
            return ev, codes
    return None


def build_legs(game_models: dict, dk_legs: list[dict], log=print) -> list[dict]:
    """One leg per Kalshi contract worth showing, priced at Kalshi's ask."""
    by_sport: dict[str, list] = {}
    for gm in game_models.values():
        by_sport.setdefault(gm.sport.key, []).append(gm)
    dk_reasons = {(l["game_id"], l["market"], l["side"]): l["reasons"] for l in dk_legs}
    legs: list[dict] = []

    for sport, gms in by_sport.items():
        lg = LEAGUE.get(sport)
        if not lg:
            continue
        try:
            series = {k: _Series(f"KX{lg}{k}") for k in KINDS}
        except requests.RequestException as e:
            log(f"  Kalshi {lg}: request failed ({e})")
            continue
        by_matchup = {k: {_matchup(e["event_ticker"]): e for e in s.events} for k, s in series.items()}
        matched = n_before = 0
        for gm in gms:
            hit = _match_game(gm.g, series["GAME"].events)
            if not hit:
                continue
            matched += 1
            ev, codes = hit
            key = _matchup(ev["event_ticker"])
            before = len(legs)
            legs += _game_legs(gm, ev, codes, series, by_matchup, key, dk_reasons)
            n_before += len(legs) - before
        counts = ", ".join(f"{k.lower()} {len(s.events)}" for k, s in series.items())
        log(f"  Kalshi {lg}: open events ({counts}); matched {matched} of {len(gms)} games -> {n_before} contracts")
    return legs


def _game_legs(gm, ev, codes, series, by_matchup, key, dk_reasons) -> list[dict]:
    g, out, sport = gm.g, gm.out, gm.sport
    base = {"sport": sport.key, "sport_name": sport.name, "game_id": g["id"], "game": g["name"],
            "short": g["short"], "start": g["date"], "home": g["home"], "away": g["away"],
            "venue": g.get("venue"), "broadcast": g.get("broadcast"), "note": g.get("note"),
            "context": gm.game_lines(), "link": None, "p_push": 0.0}
    abbr = {s: g[s]["abbr"] or g[s]["name"] for s in ("home", "away")}
    legs = []

    def add(market, side, line, selection, contract, price, p_model, p_fair, reasons, url, ticker, buy):
        if price is None or not (PRICE_RANGE[0] <= price <= PRICE_RANGE[1]):
            return
        favoured = side if side in ("home", "away") and p_model > p_fair else None
        w = gm.weight(favoured, market)
        w = w / (1 + (logit(p_model) - logit(p_fair)) ** 2)   # big disagreements: trust the market
        p = blend(p_model, p_fair, w)
        cost = _cost(price)
        legs.append({**base, "id": f"{g['id']}:{ticker}:{buy}", "market": market, "side": side, "line": line,
                     "selection": selection, "odds": decimal_to_american(1 / cost),
                     "p": round(p, 4), "p_model": round(p_model, 4), "p_market": round(p_fair, 4),
                     "p_breakeven": round(cost, 4), "edge": round(p - cost, 4), "ev": round(p / cost - 1, 4),
                     "w_model": round(w, 3), "reasons": reasons,
                     "kalshi": {"url": url, "price": round(price * 100), "contract": contract, "buy": buy}})

    # ---------------- winner (and tie in soccer)
    ms = ev.get("markets") or []
    asks = {m["ticker"]: _ask(m, "yes") for m in ms}
    total_ask = sum(a for a in asks.values() if a) or 1
    if sport.three_way:
        model = {"home": out.p_home_win(), "away": out.p_away_win(), "draw": out.p_draw()}
    else:
        mh = out.p_home_win()
        pred = gm.ctx.get("predictor")
        if pred:
            mh = 0.5 * mh + 0.5 * pred["home"] / max(pred["home"] + pred["away"], 1e-9)
        model = {"home": mh, "away": 1 - mh}
    url = series["GAME"].url(ev["event_ticker"])
    for m in ms:
        code = m["ticker"].rsplit("-", 1)[-1]
        label = (m.get("yes_sub_title") or "").strip()
        side = codes.get(code) or ("draw" if re.fullmatch(r"(tie|draw)", label.lower()) else None)
        if side not in model or not asks.get(m["ticker"]):
            continue
        sel = "Draw" if side == "draw" else f"{abbr[side]} to win"
        reasons = dk_reasons.get((g["id"], "ml", side)) or (gm.team_line(side) if side != "draw" else [])
        add("ml", side, None, sel, f"Yes · {label or sel}", asks[m["ticker"]], model[side],
            asks[m["ticker"]] / total_ask, reasons, url, m["ticker"], "yes")

    # ---------------- spreads: "TEAM wins by over X"
    sev = by_matchup["SPREAD"].get(key)
    if sev:
        url = series["SPREAD"].url(sev["event_ticker"])
        per_side: dict[str, list] = {"home": [], "away": []}
        for m in sev.get("markets") or []:
            code = re.sub(r"\d+$", "", m["ticker"].rsplit("-", 1)[-1])
            side, x = codes.get(code), m.get("floor_strike")
            yes, no = _ask(m, "yes"), _ask(m, "no")
            if side and x is not None and yes and no:
                per_side[side].append((float(x), m, yes, no))
        for side, rows in per_side.items():
            rows = [r for r in sorted(rows, key=lambda r: r[0]) if PRICE_RANGE[0] <= r[2] <= PRICE_RANGE[1]]
            for x, m, yes, no in _spread_out(rows, STRIKES_PER_SIDE):
                if side == "home":
                    p_model = out.p_margin_gt(x)[0]
                else:
                    gt, eq = out.p_margin_gt(-x)
                    p_model = max(0.0, 1 - gt - eq)
                reasons = dk_reasons.get((g["id"], "spread", side)) or gm.team_line(side)
                add("spread", side, -x, f"{abbr[side]} by {x:g}+", f"Yes · {m.get('yes_sub_title') or m.get('title')}",
                    yes, p_model, yes / (yes + no), reasons, url, m["ticker"], "yes")

    # ---------------- totals: "Over X" (Yes) and under (No)
    tev = by_matchup["TOTAL"].get(key)
    if tev:
        url = series["TOTAL"].url(tev["event_ticker"])
        rows = []
        for m in tev.get("markets") or []:
            x, yes, no = m.get("floor_strike"), _ask(m, "yes"), _ask(m, "no")
            if x is not None and yes and no:
                rows.append((float(x), m, yes, no))
        rows.sort(key=lambda r: r[0])
        reasons = dk_reasons.get((g["id"], "total", "over")) or []
        over = [r for r in rows if PRICE_RANGE[0] <= r[2] <= PRICE_RANGE[1]]
        under = [r for r in rows if PRICE_RANGE[0] <= r[3] <= PRICE_RANGE[1]]
        for x, m, yes, no in _spread_out(over, STRIKES_PER_SIDE):
            gt = out.p_total_gt(x)[0]
            add("total", "over", x, f"Over {x:g}", f"Yes · Over {x:g}", yes, gt, yes / (yes + no),
                reasons, url, m["ticker"], "yes")
        for x, m, yes, no in _spread_out(under, STRIKES_PER_SIDE):
            gt, eq = out.p_total_gt(x)
            add("total", "under", x, f"Under {x:g}", f"No · Over {x:g}", no, max(0.0, 1 - gt - eq), no / (yes + no),
                reasons, url, m["ticker"], "no")
    return legs
