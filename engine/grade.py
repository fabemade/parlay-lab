"""The daily Top Picks: lock them once a day, grade them, keep the record.

Each morning (first refresh after LOCK_HOUR Eastern) the day's Top Picks are chosen from
games today and tomorrow and frozen in data/log/<date>.json with the price at lock time.
Later refreshes only update each leg's latest price (closing-line value) and grade legs
as Kalshi settles them. A parlay is graded once every leg is settled, and the legs that
missed are kept so the Record tab can show exactly what broke a ticket.

Graded results also feed back into the model: if a bet type keeps hitting less often than
we said it would, the model leans harder on the market for it (see feedback()).
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import requests

from .odds import american_to_decimal, implied_prob

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "data" / "log"
FEEDBACK_PATH = ROOT / "data" / "feedback.json"
CALIB_DIR = ROOT / "data" / "calib"
TESTED = "Game lines"   # full-game winner/spread/total: calibrated by the walk-forward backtest
TRUST_MIN_N = 150       # graded predictions a bet type needs before it can be a Top Pick
TRUST_MIN_Z = -1.0      # ...and it must not be hitting clearly below what we predicted
LOCK_HOUR = 11          # Eastern: by 11am most lineups, starters and prop markets are posted
RECORD_DAYS = 30

SNAPSHOT = ("id", "sport", "game_id", "short", "game", "start", "market", "side", "line", "selection",
            "group", "market_name", "player", "odds", "p", "p_model", "p_market", "p_breakeven", "edge",
            "ev", "grade", "modeled", "reasons")
LIVE_FIELDS = ("odds", "p", "p_model", "p_market", "p_breakeven", "edge", "ev", "grade")


def _path(day: str) -> Path:
    return LOG_DIR / f"{day}.json"


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _write(day: str, log: dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _path(day).write_text(json.dumps(log, indent=1))


def _locked_logs() -> list[dict]:
    logs = [_read(p) for p in sorted(LOG_DIR.glob("*.json"))]
    return [l for l in logs if l and l.get("locked_at")]


def load(day: str) -> dict | None:
    log = _read(_path(day))
    return log if log and log.get("locked_at") else None


def latest() -> dict | None:
    logs = _locked_logs()
    return logs[-1] if logs else None


def _snapshot(leg: dict) -> dict:
    s = {k: leg.get(k) for k in SNAPSHOT if leg.get(k) is not None}
    for side in ("home", "away"):
        if leg.get(side):
            s[side] = {"name": leg[side].get("name"), "abbr": leg[side].get("abbr")}
    if leg.get("kalshi"):
        s["kalshi"] = {k: leg["kalshi"].get(k) for k in ("url", "price", "ticker", "contract", "buy")}
    s["result"] = None
    s["close_odds"] = leg.get("odds")
    return s


def lock_picks(day: str, locked_at: str, legs_by_id: dict, featured: dict, straights: list[str]) -> dict:
    ids = {i for p in featured.values() for i in p["legs"]} | set(straights)
    log = {
        "date": day, "locked_at": locked_at,
        "legs": {i: _snapshot(legs_by_id[i]) for i in ids},
        "parlays": [{**p, "name": name, "result": None} for name, p in featured.items()],
        "straights": straights,
    }
    _write(day, log)
    return log


def update_closing(legs_by_id: dict):
    """Track each pending leg's latest pre-game price (closing-line value)."""
    now = dt.datetime.now(dt.timezone.utc)
    for log in _locked_logs():
        changed = False
        for lid, leg in log["legs"].items():
            cur = legs_by_id.get(lid)
            if cur and leg["result"] is None and dt.datetime.fromisoformat(leg["start"].replace("Z", "+00:00")) > now:
                leg["close_odds"] = cur["odds"]
                changed = True
        if changed:
            _write(log["date"], log)


# ---------------------------------------------------------------- calibration log
# Top Picks only grade what we chose, which says little about a bet type we rarely pick.
# So at lock time we also record every candidate bet's prediction (each contract the model
# prices with a non-negative edge) and grade them all from Kalshi's settlements. That is
# how a bet type like player props earns (or loses) its place in Top Picks.

def record_calibration(day: str, legs: list[dict]):
    rows = {}
    for l in legs:
        k = l.get("kalshi") or {}
        if l.get("p_model") is None or l["edge"] < 0 or not k.get("ticker"):
            continue
        rows[l["id"]] = {"t": k["ticker"], "b": k.get("buy", "yes"), "s": l["sport"],
                         "g": l.get("group") or TESTED, "p": l["p"], "pm": l["p_market"],
                         "st": l["start"], "r": None}
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    (CALIB_DIR / f"{day}.json").write_text(json.dumps(rows, separators=(",", ":")))


def _calib_files() -> list[Path]:
    return sorted(CALIB_DIR.glob("*.json")) if CALIB_DIR.exists() else []


def _grade_calibration(max_events: int = 250):
    """Settle pending calibration rows, one Kalshi request per event (not per contract)."""
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)
    asked = 0
    for path in _calib_files():
        rows = json.loads(path.read_text())
        pending: dict[str, list[dict]] = {}
        for r in rows.values():
            if r["r"] is None and dt.datetime.fromisoformat(r["st"].replace("Z", "+00:00")) < cutoff:
                pending.setdefault("-".join(r["t"].split("-")[:2]), []).append(r)
        changed = False
        for event, rs in pending.items():
            if asked >= max_events:
                break
            asked += 1
            try:
                resp = requests.get(f"{KALSHI_API}/markets", params={"event_ticker": event, "limit": 1000}, timeout=20)
                markets = {m["ticker"]: m for m in (resp.json().get("markets") or [])} if resp.ok else {}
            except (requests.RequestException, ValueError):
                continue
            for r in rs:
                m = markets.get(r["t"]) or {}
                res = (m.get("result") or "").lower()
                if res in ("yes", "no"):
                    r["r"] = "W" if res == r["b"] else "L"
                elif m.get("status") in ("settled", "finalized"):
                    r["r"] = "P"
                changed = changed or r["r"] is not None
        if changed:
            path.write_text(json.dumps(rows, separators=(",", ":")))


def _calib_rows(days: int = 45) -> list[dict]:
    out = []
    for path in _calib_files()[-days:]:
        out += [r for r in json.loads(path.read_text()).values() if r["r"] in ("W", "L")]
    return out


# ---------------------------------------------------------------- grading

def _result(leg: dict, g: dict) -> str:
    hs, as_ = g["hs"], g["as"]
    m = leg["market"]
    if m == "ml":
        if leg["side"] == "draw":
            return "W" if hs == as_ else "L"
        if hs == as_:
            return "P"
        return "W" if (hs > as_) == (leg["side"] == "home") else "L"
    if m == "spread":
        v = ((hs - as_) if leg["side"] == "home" else (as_ - hs)) + leg["line"]
        return "W" if v > 0 else ("P" if v == 0 else "L")
    if m == "total":
        t = hs + as_
        if t == leg["line"]:
            return "P"
        return "W" if (t > leg["line"]) == (leg["side"] == "over") else "L"
    return "P"


def _kalshi_result(ticker: str, buy: str, start: str) -> str | None:
    """W / L / P from Kalshi's settled market, or None while it's still open."""
    try:
        if dt.datetime.fromisoformat(start.replace("Z", "+00:00")) > dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2):
            return None
        r = requests.get(f"{KALSHI_API}/markets/{ticker}", timeout=20)
        if not r.ok:
            return None
        m = r.json().get("market") or {}
    except (requests.RequestException, ValueError):
        return None
    res = (m.get("result") or "").lower()
    if res in ("yes", "no"):
        return "W" if res == buy else "L"
    if m.get("status") in ("settled", "finalized") and res in ("void", "all_no", ""):
        return "P"
    return None


def grade_all(results_by_sport: dict[str, dict[str, dict]]):
    _grade_calibration()
    for log in _locked_logs():
        changed = False
        for leg in log["legs"].values():
            if leg["result"] is not None:
                continue
            k = leg.get("kalshi") or {}
            r = None
            if k.get("ticker"):
                r = _kalshi_result(k["ticker"], k.get("buy", "yes"), leg["start"])
            else:
                g = results_by_sport.get(leg["sport"], {}).get(leg["game_id"])
                r = _result(leg, g) if g else None
            if r:
                leg["result"] = r
                changed = True
        for p in log["parlays"]:
            rs = [log["legs"][i]["result"] for i in p["legs"]]
            missed = [i for i, r in zip(p["legs"], rs) if r == "L"]
            # a parlay is lost the moment any leg misses; won once every leg is in
            if missed:
                res = "L"
            elif all(r in ("W", "P") for r in rs):
                res = "W" if "W" in rs else "P"
            else:
                res = None
            if res != p["result"] or missed != p.get("missed", []):
                p["result"], p["missed"] = res, missed
                changed = True
        if changed:
            _write(log["date"], log)


# ---------------------------------------------------------------- what the app shows

def _merged(leg: dict, cur: dict | None) -> dict:
    """The locked leg, with today's latest price when the bet is still listed."""
    out = dict(leg)
    out["locked_odds"] = leg["odds"]
    if leg.get("kalshi"):
        out["locked_price"] = leg["kalshi"].get("price")
    if cur and leg["result"] is None:
        for k in LIVE_FIELDS:
            if cur.get(k) is not None:
                out[k] = cur[k]
        if cur.get("kalshi"):
            out["kalshi"] = {**leg.get("kalshi", {}), "price": cur["kalshi"]["price"]}
    return out


def top_payload(log: dict | None, legs_by_id: dict) -> dict | None:
    if not log:
        return None
    legs = {i: _merged(l, legs_by_id.get(i)) for i, l in log["legs"].items()}
    return {
        "date": log["date"], "locked_at": log["locked_at"],
        "parlays": [{**p, "legs": [legs[i] for i in p["legs"]]} for p in log["parlays"]],
        "straights": [legs[i] for i in log["straights"]],
    }


def _stats(rows, odds_key="odds", p_key="p"):
    rows = [r for r in rows if r.get("result") in ("W", "L")]
    if not rows:
        return {"n": 0}
    w = sum(r["result"] == "W" for r in rows)
    profit = sum((american_to_decimal(r[odds_key]) - 1) if r["result"] == "W" else -1 for r in rows)
    return {"n": len(rows), "wins": w, "hit_rate": round(w / len(rows), 3),
            "expected_hit_rate": round(sum(r[p_key] for r in rows) / len(rows), 3),
            "units": round(profit, 2), "roi": round(profit / len(rows), 3)}


def summary() -> dict:
    logs = _locked_logs()
    straights = [log["legs"][i] for log in logs for i in log["straights"]]
    parlays = [p for log in logs for p in log["parlays"]]
    all_legs = [l for log in logs for l in log["legs"].values()]
    by = lambda rows, key: {k: _stats([r for r in rows if key(r) == k]) for k in sorted({key(r) for r in rows})}
    clv = [implied_prob(l["close_odds"]) - implied_prob(l["odds"]) for l in all_legs
           if l.get("close_odds") and l["result"] is not None]
    days = []
    for log in reversed(logs[-RECORD_DAYS:]):
        legs = log["legs"]
        days.append({
            "date": log["date"], "locked_at": log["locked_at"],
            "parlays": [{**p, "legs": [legs[i] for i in p["legs"]]} for p in log["parlays"]],
            "straights": [legs[i] for i in log["straights"]],
        })
    return {
        "straight": _stats(straights),
        "parlays": _stats(parlays, odds_key="american"),
        "all_legs": _stats(all_legs),
        "by_grade": by(all_legs, lambda l: l.get("grade", "?")),
        "by_type": by(all_legs, lambda l: l.get("group") or "Game lines"),
        "by_sport": by(all_legs, lambda l: l["sport"]),
        "avg_clv": round(sum(clv) / len(clv), 4) if clv else None,
        "calibration": _calibration_summary(),
        "days": days,
    }


def _calibration_summary() -> dict:
    """Every prediction we logged, by bet type: how often it hit vs how often we said."""
    out = {}
    for r in _calib_rows():
        out.setdefault(r["g"], []).append(r)
    res = {}
    for g, rs in out.items():
        w = sum(r["r"] == "W" for r in rs)
        res[g] = {"n": len(rs), "hit_rate": round(w / len(rs), 3),
                  "expected_hit_rate": round(sum(r["p"] for r in rs) / len(rs), 3),
                  "market_hit_rate": round(sum(r["pm"] for r in rs) / len(rs), 3),
                  "trusted": g == TESTED or _is_trusted(_z_stats(rs))}
    return res


# ---------------------------------------------------------------- learning from results

def _z_stats(rows: list[dict]) -> dict:
    exp = sum(r["p"] for r in rows)
    var = sum(r["p"] * (1 - r["p"]) for r in rows) or 1
    wins = sum(r.get("result", r.get("r")) == "W" for r in rows)
    return {"n": len(rows), "z": round((wins - exp) / math.sqrt(var), 2)}


def _is_trusted(st: dict) -> bool:
    return st["n"] >= TRUST_MIN_N and st["z"] >= TRUST_MIN_Z


def feedback(min_n: int = 30) -> dict:
    """Per (league, bet type): how much to trust the model, from graded results.

    Uses every logged prediction (the calibration log), not just the Top Picks. If a bet
    type hits clearly less often than we predicted (more than 1.5 standard errors below),
    the model is overconfident there, so its weight against the market is halved (quartered
    past 2.5). Bet types other than full-game lines start at half weight and need
    TRUST_MIN_N graded predictions without underperforming before they get full weight and
    can appear in Top Picks. Rewritten after every run into data/feedback.json.
    """
    rows: dict[str, list] = {}
    for r in _calib_rows():
        rows.setdefault(f"{r['s']}|{r['g']}", []).append(r)
    for log in _locked_logs():          # older days logged before the calibration log existed
        for l in log["legs"].values():
            if l["result"] in ("W", "L"):
                key = f"{l['sport']}|{l.get('group') or TESTED}"
                rows.setdefault(key + "#picks", []).append({"p": l["p"], "r": l["result"]})
    out = {}
    for key, ls in rows.items():
        if key.endswith("#picks"):
            base = key[:-6]
            if base in rows:            # the calibration log covers it
                continue
            key = base
        st = _z_stats(ls)
        group = key.split("|", 1)[1]
        mult = 0.25 if st["n"] >= min_n and st["z"] < -2.5 else 0.5 if st["n"] >= min_n and st["z"] < -1.5 else 1.0
        trusted = group == TESTED or _is_trusted(st)
        if not trusted:
            mult *= 0.5
        out[key] = {**st, "w_mult": mult, "trusted": trusted}
    FEEDBACK_PATH.write_text(json.dumps(out, indent=1, sort_keys=True))
    feedback_mult.cache = out
    return out


def _feedback() -> dict:
    if not hasattr(feedback_mult, "cache"):
        feedback_mult.cache = json.loads(FEEDBACK_PATH.read_text()) if FEEDBACK_PATH.exists() else {}
    return feedback_mult.cache


def feedback_mult(sport: str, group: str | None) -> float:
    g = group or TESTED
    default = 1.0 if g == TESTED else 0.5     # untested bet types start at half weight
    return _feedback().get(f"{sport}|{g}", {}).get("w_mult", default)


def trusted(sport: str, group: str | None) -> bool:
    """Can this bet type be a Top Pick? Full-game lines yes; others once they've earned it."""
    g = group or TESTED
    return g == TESTED or bool(_feedback().get(f"{sport}|{g}", {}).get("trusted"))
