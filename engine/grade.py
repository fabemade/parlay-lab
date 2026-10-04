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

from .odds import american_to_decimal, implied_prob, inv_logit, logit

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "data" / "log"
FEEDBACK_PATH = ROOT / "data" / "feedback.json"
LEARNED_PATH = ROOT / "data" / "learned.json"
TRAIN_MIN_ROWS = 100     # graded predictions a bet type needs before its calibration is fitted
CALIB_DIR = ROOT / "data" / "calib"
TESTED = "Game lines"   # full-game winner/spread/total: calibrated by the walk-forward backtest
TRUST_MIN_N = 150       # graded predictions a bet type needs before it can be a Top Pick
TRUST_MIN_Z = -1.0      # ...and it must not be hitting clearly below what we predicted
LOCK_HOUR = 10          # Eastern: picks lock on the first refresh from 10am
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


def lock_picks(day: str, locked_at: str, legs_by_id: dict, featured: dict, straights: list[str],
               best: list[str] = (), locks: list[str] = ()) -> dict:
    ids = {i for p in featured.values() for i in p["legs"]} | set(straights) | set(best) | set(locks)
    log = {
        "date": day, "locked_at": locked_at,
        "legs": {i: _snapshot(legs_by_id[i]) for i in ids},
        "parlays": [{**p, "name": name, "result": None} for name, p in featured.items()],
        "straights": straights,
        "straights_best": list(best),
        "locks": list(locks),
    }
    _write(day, log)
    return log


def add_best_tier(day: str, legs_by_id: dict, featured: dict, best: list[str]) -> dict | None:
    """Add the Best-available tier to a day locked before that tier existed, leaving the
    already-locked picks untouched. Only legs that haven't started can be added."""
    log = load(day)
    if not log or "straights_best" in log:
        return log
    taken = set(log["legs"])
    parlays = [{**p, "name": n, "result": None} for n, p in featured.items()
               if p.get("tier") == "best" and not taken & set(p["legs"])]
    best = [i for i in best if i not in taken]
    for i in {i for p in parlays for i in p["legs"]} | set(best):
        log["legs"][i] = _snapshot(legs_by_id[i])
    log["parlays"] += parlays
    log["straights_best"] = best
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
        # candidates with an edge, plus every high-probability bet (the Locks population)
        if l.get("p_model") is None or (l["edge"] < 0 and l["p"] < 0.75) or not k.get("ticker"):
            continue
        # "p" is the model's blend before learned calibration, so retraining doesn't compound;
        # "pc" is what we showed
        rows[l["id"]] = {"t": k["ticker"], "b": k.get("buy", "yes"), "s": l["sport"],
                         "g": l.get("group") or TESTED, "p": l.get("p_raw", l["p"]), "pc": l["p"],
                         "pm": l["p_market"], "st": l["start"], "r": None}
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
        "straights_best": [legs[i] for i in log.get("straights_best", [])],
        "locks": [legs[i] for i in log.get("locks", [])],
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
    best_straights = [log["legs"][i] for log in logs for i in log.get("straights_best", [])]
    parlays = [p for log in logs for p in log["parlays"] if p.get("tier", "edge") == "edge"]
    best_parlays = [p for log in logs for p in log["parlays"] if p.get("tier") == "best"]
    locks = [log["legs"][i] for log in logs for i in log.get("locks", [])]
    lock_parlays = [p for log in logs for p in log["parlays"] if p.get("tier") == "lock"]
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
            "straights_best": [legs[i] for i in log.get("straights_best", [])],
            "locks": [legs[i] for i in log.get("locks", [])],
        })
    return {
        "straight": _stats(straights),
        "parlays": _stats(parlays, odds_key="american"),
        "straight_best": _stats(best_straights),
        "parlays_best": _stats(best_parlays, odds_key="american"),
        "locks": _stats(locks),
        "parlays_lock": _stats(lock_parlays, odds_key="american"),
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
                  "expected_hit_rate": round(sum(r.get("pc", r["p"]) for r in rs) / len(rs), 3),
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
        trusted = (group == TESTED and key.split("|", 1)[0] in _calibrated_sports()) or _is_trusted(st)
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


def _calibrated_sports() -> set[str]:
    if not hasattr(_calibrated_sports, "cache"):
        p = ROOT / "data" / "calibration.json"
        _calibrated_sports.cache = set(json.loads(p.read_text())) if p.exists() else set()
    return _calibrated_sports.cache


def trusted(sport: str, group: str | None) -> bool:
    """Can this bet type be a Top Pick? Full-game lines in a league the backtest has
    calibrated, yes; other bet types (or new leagues) once they've earned it."""
    g = group or TESTED
    if g == TESTED and sport in _calibrated_sports():
        return True
    return bool(_feedback().get(f"{sport}|{g}", {}).get("trusted"))


# ---------------------------------------------------------------- training on our own results

def train(days: int = 60) -> dict:
    """Refit the final calibration on every graded prediction (runs each refresh).

    Two numbers per bet type, fitted by maximum likelihood:
      k: how far to follow the model when it disagrees with Kalshi (1 = as now, 0 = market only)
      T: a temperature on Kalshi's own probability (<1 = favourites win less often than priced)
    final = inv_logit(T * logit(market) + k * (logit(ours) - logit(market)))
    Each game counts once however many contracts it had, and both numbers are pulled toward
    "no change" (k=1, T=1), so a few days of results nudge the model instead of swinging it.
    """
    import numpy as np
    rows: dict[str, list] = {}
    for path in _calib_files()[-days:]:
        for r in json.loads(path.read_text()).values():
            if r["r"] in ("W", "L"):
                rows.setdefault(r["g"], []).append(r)
    out = {"trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "groups": {}}
    for g, rs in rows.items():
        games = [r["t"].split("-")[1] if "-" in r["t"] else r["t"] for r in rs]
        info = {"n": len(rs), "games": len(set(games))}
        if len(rs) < TRAIN_MIN_ROWS:
            out["groups"][g] = {**info, "k": 1.0, "T": 1.0, "note": f"needs {TRAIN_MIN_ROWS} graded"}
            continue
        count = {x: games.count(x) for x in set(games)}
        w = np.array([1 / count[x] for x in games])
        y = np.array([r["r"] == "W" for r in rs], float)
        lp = np.array([logit(min(max(r["p"], 1e-3), 1 - 1e-3)) for r in rs])
        lm = np.array([logit(min(max(r["pm"], 1e-3), 1 - 1e-3)) for r in rs])

        def nll(k, T):
            q = np.clip(1 / (1 + np.exp(-(T * lm + k * (lp - lm)))), 1e-6, 1 - 1e-6)
            return float(-(w * (y * np.log(q) + (1 - y) * np.log(1 - q))).sum())
        best = min(((nll(k, T) + 0.5 * ((k - 1) / 0.5) ** 2 + 0.5 * ((T - 1) / 0.1) ** 2, k, T)
                    for k in np.arange(0, 2.51, 0.05) for T in np.arange(0.8, 1.201, 0.01)))
        _, k, T = best
        out["groups"][g] = {**info, "k": round(float(k), 2), "T": round(float(T), 2),
                            "loss_before": round(nll(1, 1), 3), "loss_after": round(nll(k, T), 3)}
    LEARNED_PATH.write_text(json.dumps(out, indent=1, sort_keys=True))
    learned_p.cache = out
    return out


def learned_p(p: float, p_market: float, group: str | None) -> float:
    """Our probability after the calibration learned from graded results."""
    if not hasattr(learned_p, "cache"):
        learned_p.cache = json.loads(LEARNED_PATH.read_text()) if LEARNED_PATH.exists() else {}
    prm = learned_p.cache.get("groups", {}).get(group or TESTED)
    if not prm or (prm["k"] == 1.0 and prm["T"] == 1.0):
        return p
    lp, lm = logit(min(max(p, 1e-4), 1 - 1e-4)), logit(min(max(p_market, 1e-4), 1 - 1e-4))
    return inv_logit(prm["T"] * lm + prm["k"] * (lp - lm))
