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
import json
import math
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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


_lock = threading.Lock()
_next_slot = [0.0]
RATE = 9.0              # requests per second; Kalshi's public API rate-limits bursts


def _get(path: str, params: dict | None = None) -> dict:
    """GET with a shared rate limit and retries on 429 / transient errors."""
    for attempt in range(6):
        with _lock:
            now = time.monotonic()
            wait = _next_slot[0] - now
            _next_slot[0] = max(now, _next_slot[0]) + 1 / RATE
        if wait > 0:
            time.sleep(wait)
        try:
            r = _session.get(API + path, params=params, timeout=20)
        except requests.RequestException:
            time.sleep(1 + attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise requests.RequestException(f"Kalshi {path} kept failing")


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


def build_legs(game_models: dict, dk_legs: list[dict], log=print, extras: bool = True) -> list[dict]:
    """One leg per Kalshi contract worth showing, priced at Kalshi's ask."""
    from .players import STATS, PlayerBook, family

    by_sport: dict[str, list] = {}
    for gm in game_models.values():
        by_sport.setdefault(gm.sport.key, []).append(gm)
    dk_reasons = {(l["game_id"], l["market"], l["side"]): l["reasons"] for l in dk_legs}
    legs: list[dict] = []
    leagues = [LEAGUE[s] for s in by_sport if s in LEAGUE]
    found = discover(leagues, log=log) if extras else {}

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
        hits = {}
        for gm in gms:
            hit = _match_game(gm.g, series["GAME"].events)
            if hit:
                hits[gm.g["id"]] = hit
        n_core = n_extra = 0
        for gm in gms:
            if gm.g["id"] not in hits:
                continue
            ev, codes = hits[gm.g["id"]]
            before = len(legs)
            legs += _game_legs(gm, ev, codes, series, by_matchup, _matchup(ev["event_ticker"]), dk_reasons)
            n_core += len(legs) - before

        # ---- every other per-game market for the matched games
        titles = found.get(lg, {})
        suffixes = [x for x in titles if x not in KINDS]
        if hits and suffixes:
            keys = {_key(ev["event_ticker"]) for ev, _ in hits.values()}
            with ThreadPoolExecutor(max_workers=8) as ex:
                fetched = dict(zip(suffixes, ex.map(lambda x: _safe_events(f"KX{lg}{x}"), suffixes)))
            by_key: dict[str, dict[str, list]] = {}
            for x, evs in fetched.items():
                for e in evs:
                    k = _key(e["event_ticker"])
                    if k in keys:
                        by_key.setdefault(k, {}).setdefault(x, []).append(e)
            # player game logs, fetched in parallel before pricing
            fam = family(sport)
            book = PlayerBook(gms[0].sport) if STATS.get(fam) else None
            if book:
                want = []
                for gm in gms:
                    if gm.g["id"] not in hits:
                        continue
                    k = _key(hits[gm.g["id"]][0]["event_ticker"])
                    tids = [gm.g["home"]["id"], gm.g["away"]["id"]]
                    for x, evs in by_key.get(k, {}).items():
                        if x not in STATS[fam]:
                            continue
                        for e in evs:
                            for m in e.get("markets") or []:
                                lab = m.get("yes_sub_title") or ""
                                if ":" in lab:
                                    want.append((lab.split(":")[0].strip(), tids))
                with ThreadPoolExecutor(max_workers=12) as ex:
                    list(ex.map(lambda t: book.roster(t), {t for _, tids in want for t in tids}))
                ids = [a["id"] for a in (book.find(n, t) for n, t in {(n, tuple(t)) for n, t in want}) if a]
                book.prefetch(ids)
            for gm in gms:
                if gm.g["id"] not in hits:
                    continue
                ev, codes = hits[gm.g["id"]]
                before = len(legs)
                try:
                    legs += extra_legs(gm, by_key.get(_key(ev["event_ticker"]), {}), titles, codes, lg, book, dk_reasons)
                except Exception as e:  # one odd market must not sink the league
                    log(f"    ! {gm.g['short']}: extra markets failed: {e}")
                n_extra += len(legs) - before
        log(f"  Kalshi {lg}: matched {len(hits)} of {len(gms)} games -> {n_core} core + {n_extra} other contracts "
            f"({len(suffixes)} market types)")
    return debias(legs, log)


def debias(legs: list[dict], log=print, min_n: int = 8) -> list[dict]:
    """Remove a market type's shared lean before looking for edges.

    If our model sits, say, 7% above Kalshi on *every* 2nd-quarter total, that's a modelling
    error for that market type, not seven separate edges. For each (league, market type)
    with enough contracts we measure the median log-odds gap between model and market on
    the Yes side and subtract it, keeping only the contract-to-contract differences.
    Full-game winner/spread/total are left alone: the backtest already calibrates those.
    """
    from .odds import inv_logit
    groups: dict[tuple, list[float]] = {}
    for l in legs:
        if l.get("group") != "Game lines" and l.get("p_model") is not None and l["kalshi"]["buy"] == "yes":
            groups.setdefault((l["sport"], l.get("market_name")), []).append(logit(l["p_model"]) - logit(l["p_market"]))
    # median, so a few extreme longshot contracts can't drag the whole market type
    shift = {k: float(np.median(v)) for k, v in groups.items() if len(v) >= min_n}
    moved = 0
    for l in legs:
        s = shift.get((l["sport"], l.get("market_name")))
        if not s or l.get("p_model") is None or l.get("group") == "Game lines":
            continue
        pm = l["p_model"]
        pm = inv_logit(logit(pm) - s) if l["kalshi"]["buy"] == "yes" else 1 - inv_logit(logit(1 - pm) - s)
        p = blend(pm, l["p_market"], l["w_model"])
        cost = l["p_breakeven"]
        l.update(p_model=round(pm, 4), p=round(p, 4), edge=round(p - cost, 4), ev=round(p / cost - 1, 4))
        moved += 1
    big = sorted(((k, v) for k, v in shift.items() if abs(v) > 0.2), key=lambda kv: -abs(kv[1]))[:5]
    if big:
        log("  de-biased market types (largest): " + ", ".join(f"{k[0]} {k[1]} {v:+.2f}" for k, v in big))
    return legs


def _safe_events(series: str) -> list[dict]:
    try:
        return _events(series)
    except requests.RequestException:
        return []


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
                     "w_model": round(w, 3), "reasons": reasons, "group": "Game lines", "modeled": True,
                     "market_name": {"ml": "Winner", "spread": "Spread", "total": "Total"}.get(market, market),
                     "kalshi": {"url": url, "price": round(price * 100), "contract": contract, "buy": buy,
                                "ticker": ticker}})

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


# ============================================================================ every other market
# Beyond winner / spread / total, Kalshi lists dozens of per-game markets per league:
# halves, quarters, periods, first-N innings, team totals, player props, specials. We find
# them by scanning each league's series once every couple of days (cached), then price
# every contract: with a model where we have one, at Kalshi's own price where we don't.

SERIES_CACHE = Path(__file__).resolve().parent.parent / "data" / "kalshi_series.json"
DISCOVERY_MAX_AGE = dt.timedelta(days=2)
GAME_EVENT = re.compile(r"^[A-Z0-9]+-\d{2}[A-Z]{3}\d{2}(\d{4})?[A-Z]")
PLAYER_CODE = re.compile(r"^([A-Z]{2,5}?)([A-Z]+\d+)(?:-(\d+(?:P\d+)?))?$")
PROP_PRICE_RANGE = (0.04, 0.96)
PERIOD_RE = re.compile(r"^(1H|2H|1Q|2Q|3Q|4Q|1P|2P|3P|F3|F5|F7)")


def _key(event_ticker: str) -> str:
    """'KXMLBKS-26SEP301400PHIATL' / '...-26SEP301400PHIATL-8' -> '26SEP301400PHIATL'."""
    parts = event_ticker.split("-")
    return parts[1] if len(parts) > 1 else ""


def discover(leagues: list[str], log=print) -> dict:
    """{league: {suffix: series title}} for every series with per-game events."""
    cache = {}
    if SERIES_CACHE.exists():
        cache = json.loads(SERIES_CACHE.read_text())
    fresh = cache.get("updated") and dt.datetime.fromisoformat(cache["updated"]) > dt.datetime.now(dt.timezone.utc) - DISCOVERY_MAX_AGE
    have = cache.get("leagues", {})
    todo = [lg for lg in leagues if not fresh or lg not in have]
    if not todo:
        return have
    log(f"  Kalshi: discovering per-game markets for {len(todo)} leagues (cached for 2 days)…")
    all_series = _get("/series", {"category": "Sports"}).get("series") or []
    titles = {s["ticker"]: s.get("title") for s in all_series}

    def probe(tk):
        try:
            d = _get("/events", {"series_ticker": tk, "status": "open", "limit": 25})
        except requests.RequestException:
            return tk, False
        return tk, any(GAME_EVENT.match(e["event_ticker"]) for e in d.get("events") or [])

    other_prefixes = sorted(set(LEAGUE.values()), key=len, reverse=True)
    for lg in todo:
        mine = []
        for tk in titles:
            if not tk.startswith("KX" + lg):
                continue
            # skip series that belong to a longer league code ("LALIGA2..." isn't La Liga)
            if any(p != lg and p.startswith(lg) and tk.startswith("KX" + p) for p in other_prefixes):
                continue
            mine.append(tk)
        with ThreadPoolExecutor(max_workers=8) as ex:
            hits = [tk for tk, ok in ex.map(probe, mine) if ok]
        have[lg] = {tk[2 + len(lg):]: titles.get(tk) for tk in hits}
    if not fresh:
        cache["updated"] = dt.datetime.now(dt.timezone.utc).isoformat()
    cache["leagues"] = have
    SERIES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SERIES_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return have


def _market_title(ev: dict) -> str:
    t = ev.get("title") or ""
    return t.split(": ", 1)[1] if ": " in t else t


def _group(suffix: str, is_player: bool) -> str:
    if is_player:
        return "Player props"
    m = PERIOD_RE.match(suffix)
    if m:
        p = m.group(1)
        return {"1H": "1st half", "2H": "2nd half", "1Q": "1st quarter", "2Q": "2nd quarter",
                "3Q": "3rd quarter", "4Q": "4th quarter", "1P": "1st period", "2P": "2nd period",
                "3P": "3rd period", "F3": "First 3 innings", "F5": "First 5 innings",
                "F7": "First 7 innings"}[p]
    if "TEAM" in suffix:
        return "Team props"
    return "Game specials"


def extra_legs(gm, ev_by_suffix: dict[str, list[dict]], titles: dict, codes: dict, league: str,
               book, dk_reasons) -> list[dict]:
    """Every contract in every non-core market for one game."""
    from .markets import Scoreline, p_first_inning_run, p_goal_first_minutes, period_share
    from .players import STATS, PropModel, family

    g, sport = gm.g, gm.sport
    abbr = {s: g[s]["abbr"] or g[s]["name"] for s in ("home", "away")}
    side_of = {c: s for c, s in codes.items()}
    team_ids = [g["home"]["id"], g["away"]["id"]]
    fam = family(sport.key)
    base = {"sport": sport.key, "sport_name": sport.name, "game_id": g["id"], "game": g["name"],
            "short": g["short"], "start": g["date"], "home": g["home"], "away": g["away"],
            "venue": g.get("venue"), "broadcast": g.get("broadcast"), "note": g.get("note"),
            "link": None, "p_push": 0.0}
    full = Scoreline(sport, *gm.exp)
    legs: list[dict] = []
    # team scoring multipliers for player props: tonight's projection vs the team's norm
    mult = {}
    for s, e in (("home", gm.exp[0]), ("away", gm.exp[1])):
        pf = (gm.form.get(s) or {}).get("season_pf")
        mult[s] = e / pf if pf else 1.0

    def add(suffix, ev, m, buy, selection, price, p_model, p_fair, reasons, *, market, side=None,
            line=None, player=None, w_mult=1.0, low=PRICE_RANGE[0], high=PRICE_RANGE[1]):
        if price is None or not (low <= price <= high):
            return
        cost = _cost(price)
        if p_model is None:
            p, w, modeled = p_fair, 0.0, False
        else:
            p_model = min(max(p_model, 1e-4), 1 - 1e-4)
            w = sport.w_model * w_mult
            if gm.low_sample:
                w *= 0.3
            w = w / (1 + (logit(p_model) - logit(p_fair)) ** 2)
            p, modeled = blend(p_model, p_fair, w), True
        url = f"https://kalshi.com/markets/kx{league.lower()}{suffix.lower()}/{_slug(titles.get(suffix))}/{ev['event_ticker'].lower()}"
        legs.append({**base, "id": f"{g['id']}:{m['ticker']}:{buy}", "market": market, "side": side,
                     "line": line, "selection": selection, "group": _group(suffix, player is not None),
                     "market_name": _market_title(ev), "player": player, "modeled": modeled,
                     "odds": decimal_to_american(1 / cost), "p": round(p, 4),
                     "p_model": round(p_model, 4) if p_model is not None else None,
                     "p_market": round(p_fair, 4), "p_breakeven": round(cost, 4),
                     "edge": round(p - cost, 4), "ev": round(p / cost - 1, 4), "w_model": round(w, 3),
                     "reasons": reasons, "context": [],
                     "kalshi": {"url": url, "price": round(price * 100), "ticker": m["ticker"],
                                "contract": f"{'Yes' if buy == 'yes' else 'No'} · {m.get('yes_sub_title') or m.get('title')}",
                                "buy": buy}})

    def fair(yes, no):
        return yes / (yes + no) if yes and no else yes

    def both_sides(suffix, ev, m, sel_yes, sel_no, p_yes_model, reasons, **kw):
        yes, no = _ask(m, "yes"), _ask(m, "no")
        if yes:
            add(suffix, ev, m, "yes", sel_yes, yes, p_yes_model, fair(yes, no), reasons, **kw)
        if no and sel_no:
            kw = dict(kw)
            if kw.get("side") in ("over", "yes"):
                kw["side"] = "under" if kw["side"] == "over" else "no"
            add(suffix, ev, m, "no", sel_no, no, None if p_yes_model is None else 1 - p_yes_model,
                fair(no, yes), reasons, **kw)

    for suffix, events in ev_by_suffix.items():
        pm = PERIOD_RE.match(suffix)
        period = pm.group(1) if pm else None
        rest = suffix[len(period):] if period else suffix
        share = period_share(sport.key, period) if period else None
        sl = Scoreline(sport, *gm.exp, share=share) if share else None
        pname = _group(suffix, False)
        for ev in events:
            for m in ev.get("markets") or []:
                tail = m["ticker"].split("-", 2)[-1] if m["ticker"].count("-") >= 2 else m["ticker"]
                tail = tail[len(_key(ev["event_ticker"])) + 1:] if tail.startswith(_key(ev["event_ticker"])) else tail
                label = (m.get("yes_sub_title") or m.get("title") or "").strip()
                x = m.get("floor_strike")
                team = re.sub(r"\d+$", "", tail)
                tside = side_of.get(team)
                pc = PLAYER_CODE.match(tail)
                is_player = bool(pc and not tside and ":" in label and not label.lower().startswith(("over", "under")))
                player = label.split(":")[0].strip() if is_player else (label if (pc and not tside and not x and "TIE" not in tail and "NONE" not in tail and suffix.startswith(("FIRST", "TEAMFIRST"))) else None)

                # ---- player props ----------------------------------------------------
                if is_player and x is not None:
                    reasons, p_model = [], None
                    stat = STATS.get(fam, {}).get(suffix)
                    ath = book.find(player, team_ids) if (book and stat) else None
                    if ath:
                        own = "home" if ath["team"] == g["home"]["id"] else "away"
                        opp = "away" if own == "home" else "home"
                        sens = stat[2]
                        pmod = PropModel(suffix, fam, book.gamelog(ath["id"]),
                                         mult[own] if sens >= 0 else mult[opp], g[opp]["id"], own == "home",
                                         postseason=g.get("season_type") == 3)
                        if pmod.usable:
                            p_model = pmod.p_at_least(float(x))
                            reasons = pmod.reasons(player, float(x), abbr[opp])
                    stat_label = stat[3] if stat else _market_title(ev).lower()
                    if float(x) % 1 == 0.5:          # "N+" ladder: Yes = N or more, No = N-1 or fewer
                        n = math.floor(float(x)) + 1
                        sel_yes = f"{player} {n}+ {stat_label}"
                        sel_no = f"{player}: no {stat_label}" if n == 1 else f"{player} {n - 1} or fewer {stat_label}"
                    else:
                        sel_yes, sel_no = f"{player} over {x:g} {stat_label}", f"{player} under {x:g} {stat_label}"
                    both_sides(suffix, ev, m, sel_yes, sel_no,
                               p_model, reasons or ["No game log available for this player, so this is priced at Kalshi's market"],
                               market="prop", side="over", line=float(x), player=player, w_mult=1.15,
                               low=PROP_PRICE_RANGE[0], high=PROP_PRICE_RANGE[1])
                    continue

                # ---- period / game winners (three-way with ties) ---------------------
                if rest in ("", "WINNER") and period and sl:
                    s = tside or ("draw" if tail in ("TIE", "DRAW") else None)
                    pmodel = {"home": sl.p_home(), "away": sl.p_away(), "draw": sl.p_tie()}.get(s)
                    sel = f"{abbr[s]} wins {pname}" if s in ("home", "away") else f"{pname} tied"
                    reasons = [f"{pname}: model projects {abbr['away']} {sl.ma:.1f} – {abbr['home']} {sl.mh:.1f}",
                               f"Win {sl.p_home() * 100:.0f}% / tie {sl.p_tie() * 100:.0f}% / win {sl.p_away() * 100:.0f}% ({abbr['home']} / tie / {abbr['away']})"]
                    yes, no = _ask(m, "yes"), _ask(m, "no")
                    if yes:
                        add(suffix, ev, m, "yes", sel, yes, pmodel if s else None, fair(yes, no), reasons,
                            market="period_ml", side=s, w_mult=0.6)
                    continue

                # ---- period spreads --------------------------------------------------
                if rest == "SPREAD" and sl and tside and x is not None:
                    both_sides(suffix, ev, m, f"{abbr[tside]} wins {pname} by {x:g}+", f"{abbr[tside]} not by {x:g}+ ({pname})",
                               sl.p_margin_gt(tside, float(x)),
                               [f"{pname}: projected margin {abbr[tside]} {((sl.mh - sl.ma) if tside == 'home' else (sl.ma - sl.mh)):+.1f}"],
                               market="period_spread", side=tside, line=float(x), w_mult=0.6)
                    continue

                # ---- period / game totals --------------------------------------------
                if rest == "TOTAL" and sl and x is not None:
                    both_sides(suffix, ev, m, f"{pname} over {x:g}", f"{pname} under {x:g}", sl.p_total_gt(float(x)),
                               [f"{pname}: projected total {sl.mh + sl.ma:.1f}"],
                               market="period_total", side="over", line=float(x), w_mult=0.5)
                    continue

                # ---- team totals (full game or period) -------------------------------
                if rest == "TEAMTOTAL" and tside and x is not None:
                    s_line = sl or full
                    both_sides(suffix, ev, m, f"{abbr[tside]} over {x:g}" + (f" ({pname})" if sl else ""),
                               f"{abbr[tside]} under {x:g}" + (f" ({pname})" if sl else ""),
                               s_line.p_team_gt(tside, float(x)),
                               [f"Projected {abbr[tside]}: {(s_line.mh if tside == 'home' else s_line.ma):.1f}"] +
                               (gm.team_line(tside)[:2] if not sl else []),
                               market="team_total", side="over", line=float(x), w_mult=0.5)
                    continue

                # ---- both teams to score ---------------------------------------------
                if rest == "BTTS":
                    s_line = sl or (Scoreline(sport, *gm.exp) if not sport.three_way else full)
                    both_sides(suffix, ev, m, f"Both teams score{' (' + pname + ')' if sl else ''}",
                               f"Not both teams score{' (' + pname + ')' if sl else ''}",
                               s_line.p_both_score(1), [f"Projected {abbr['away']} {s_line.ma:.1f} – {abbr['home']} {s_line.mh:.1f}"],
                               market="btts", side="yes", w_mult=0.6)
                    continue
                if suffix == "BOTH" and x is not None:   # NFL: both teams score N+
                    both_sides(suffix, ev, m, f"Both teams {x:g}+ pts", f"Not both {x:g}+ pts", full.p_both_score(float(x)),
                               [f"Projected {abbr['away']} {full.ma:.1f} – {abbr['home']} {full.mh:.1f}"],
                               market="special", side="yes", w_mult=0.5)
                    continue

                # ---- overtime / extra innings / ties ---------------------------------
                if suffix in ("OT", "OVERTIME", "EXTRAS", "TIE"):
                    if sport.kind == "poisson" and not sport.three_way:
                        p_ot = float(gm.out.M[gm.out.diff == 0].sum())
                    elif sport.kind == "gaussian" and not hasattr(gm.out, "kp"):
                        p_ot = float(norm_pdf0(gm.out.margin, sport.margin_sd))
                    elif hasattr(gm.out, "kp"):
                        p_ot = float(gm.out.kp[gm.out.ks == 0].sum()) * (0.2 if suffix == "TIE" else 1.0)
                    else:
                        p_ot = None
                    name = {"EXTRAS": "Extra innings", "TIE": "Game ends tied"}.get(suffix, "Overtime")
                    both_sides(suffix, ev, m, name, f"No {name.lower()}", p_ot,
                               [f"Regulation tie chance from the projected score distribution: {p_ot * 100:.0f}%"] if p_ot else [],
                               market="special", side="yes", w_mult=0.5)
                    continue

                # ---- MLB run in the 1st / NHL goal in first 10 minutes ---------------
                if suffix == "RFI":
                    p = p_first_inning_run(*gm.exp)
                    both_sides(suffix, ev, m, "Run in the 1st inning (YRFI)", "No run in the 1st (NRFI)", p,
                               [f"Projected runs: {abbr['away']} {gm.exp[1]:.1f}, {abbr['home']} {gm.exp[0]:.1f}"] +
                               [n for s in ("away", "home") for n in gm.notes[s] if "Starter" in n],
                               market="special", side="yes", w_mult=0.5)
                    continue
                if suffix == "F10G":
                    p = p_goal_first_minutes(sum(gm.exp), 10)
                    both_sides(suffix, ev, m, "Goal in first 10 minutes", "No goal in first 10 minutes", p,
                               [f"Projected total goals {sum(gm.exp):.1f}"], market="special", side="yes", w_mult=0.5)
                    continue

                # ---- soccer correct score / first team to score ----------------------
                if rest == "SCORE":
                    sm = re.fullmatch(r"([A-Z]+?)(\d+)([A-Z]+?)(\d+)", tail)
                    p = None
                    if sm and side_of.get(sm.group(1)) and side_of.get(sm.group(3)):
                        goals = {side_of[sm.group(1)]: int(sm.group(2)), side_of[sm.group(3)]: int(sm.group(4))}
                        src = sl or full
                        p = src.p_exact(goals.get("home", 0), goals.get("away", 0))
                        if not sl and sport.three_way:
                            p = float(gm.out.M[goals.get("home", 0), goals.get("away", 0)]) if goals.get("home", 0) < gm.out.M.shape[0] and goals.get("away", 0) < gm.out.M.shape[1] else p
                    yes, no = _ask(m, "yes"), _ask(m, "no")
                    if yes:
                        add(suffix, ev, m, "yes", f"Correct score{' (' + pname + ')' if sl else ''}: {label}", yes, p, fair(yes, no),
                            [f"Projected {abbr['away']} {(sl or full).ma:.2f} – {abbr['home']} {(sl or full).mh:.2f} goals"],
                            market="correct_score", w_mult=0.5, low=0.03)
                    continue
                if suffix in ("FTTS", "FIRSTTDTEAM"):
                    s = tside
                    p = full.p_first_to_score(s) if s else (full.p_exact(0, 0) if tail in ("NONE",) else None)
                    yes, no = _ask(m, "yes"), _ask(m, "no")
                    if yes:
                        add(suffix, ev, m, "yes", f"{abbr[s]} scores first" if s else label, yes, p if suffix == "FTTS" else None,
                            fair(yes, no), [f"Scoring rates: {abbr['home']} {full.mh:.1f}, {abbr['away']} {full.ma:.1f}"],
                            market="special", side=s, w_mult=0.4)
                    continue

                # ---- winning margin bands --------------------------------------------
                if suffix in ("WINMARGIN", "MOV"):
                    bm = re.fullmatch(r"([A-Z]+?)(\d+)TO(\d+)", tail)
                    bp = re.fullmatch(r"([A-Z]+?)(\d+)(?:PLUS|OVER)", tail)
                    p = None
                    if (bm or bp) and side_of.get((bm or bp).group(1)):
                        s = side_of[(bm or bp).group(1)]
                        lo = int((bm or bp).group(2))
                        hi = int(bm.group(3)) if bm else 999
                        def gt(v):
                            return gm.out.p_margin_gt(v)[0]
                        if s == "home":
                            p = gt(lo - 0.5) - (gt(hi + 0.5) if hi < 999 else 0)
                        else:
                            p = (1 - gt(-lo + 0.5) - 0) - ((1 - gt(-hi - 0.5)) if hi < 999 else 0)
                            p = max(p, 0.0)
                    elif tail == "TIE" and hasattr(gm.out, "kp"):
                        p = float(gm.out.kp[gm.out.ks == 0].sum()) * 0.2
                    yes, no = _ask(m, "yes"), _ask(m, "no")
                    if yes:
                        add(suffix, ev, m, "yes", f"Winning margin: {label}", yes, p, fair(yes, no),
                            [f"Projected margin {abbr['home']} {gm.out.margin:+.1f}"], market="special", w_mult=0.5, low=0.03)
                    continue

                # ---- everything else: listed at Kalshi's price -----------------------
                sel = f"{_market_title(ev)}: {label}" if label and _market_title(ev).lower() not in label.lower() else (label or _market_title(ev))
                yes, no = _ask(m, "yes"), _ask(m, "no")
                if yes:
                    add(suffix, ev, m, "yes", sel, yes, None, fair(yes, no),
                        ["Priced at Kalshi's market. We don't model this bet type yet, so there's no edge estimate"],
                        market="other", player=player, low=0.03, high=0.97)
    return legs


def norm_pdf0(mean: float, sd: float) -> float:
    """P(|margin| < 0.5) for a normal margin: the chance regulation ends level."""
    from scipy.stats import norm as _n
    return _n.cdf(0.5, mean, sd) - _n.cdf(-0.5, mean, sd)
