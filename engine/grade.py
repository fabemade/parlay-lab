"""Keep score: log every published pick, then grade it once the game is final.

This is how we find out whether the model is actually any good. Hit rate alone can
mislead (favourites hit often and still lose money), so we also track profit and
calibration (do our 70% picks hit ~70% of the time?).
"""
from __future__ import annotations

import json
from pathlib import Path

import datetime as dt

import requests

from .odds import american_to_decimal, implied_prob

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "log"


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"legs": {}, "parlays": []}


def record_picks(day: str, legs: list[dict], parlays: list[dict], featured_ids: set[str]):
    """Store today's featured legs. Re-runs keep the first price and track the latest as 'close'."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{day}.json"
    log = _load(path)
    by_id = {l["id"]: l for l in legs}
    for lid in featured_ids:
        leg = by_id[lid]
        slim = {k: leg.get(k) for k in ("id", "sport", "game_id", "short", "start", "market", "side",
                                        "selection", "line", "odds", "p", "p_market", "grade", "group")}
        if leg.get("kalshi"):
            slim["kalshi"] = {"ticker": leg["kalshi"].get("ticker"), "buy": leg["kalshi"].get("buy")}
        if lid in log["legs"]:
            log["legs"][lid]["close_odds"] = leg["odds"]
        else:
            log["legs"][lid] = {**slim, "close_odds": leg["odds"], "result": None}
    known = {tuple(p["legs"]) for p in log["parlays"]}
    for p in parlays:
        if tuple(p["legs"]) not in known:
            log["parlays"].append({**p, "result": None})
    path.write_text(json.dumps(log, indent=1))


def _result(leg: dict, g: dict) -> str:
    hs, as_ = g["hs"], g["as"]
    m = leg["market"]
    if m == "ml":
        if leg["side"] == "draw":
            return "W" if hs == as_ else "L"
        if hs == as_:
            return "P"
        home_won = hs > as_
        return "W" if home_won == (leg["side"] == "home") else "L"
    if m == "spread":
        margin = (hs - as_) if leg["side"] == "home" else (as_ - hs)
        v = margin + leg["line"]
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
        if dt.datetime.fromisoformat(start.replace("Z", "+00:00")) > dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=3):
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
    """results_by_sport: sport -> game_id -> finished game record."""
    for path in sorted(LOG_DIR.glob("*.json")):
        log = _load(path)
        changed = False
        for leg in log["legs"].values():
            if leg["result"] is None:
                k = leg.get("kalshi") or {}
                if k.get("ticker"):
                    # Kalshi's own settlement grades every contract type (props included)
                    r = _kalshi_result(k["ticker"], k.get("buy", "yes"), leg["start"])
                    if r:
                        leg["result"] = r
                        changed = True
                    continue
                g = results_by_sport.get(leg["sport"], {}).get(leg["game_id"])
                if g:
                    leg["result"] = _result(leg, g)
                    changed = True
        for p in log["parlays"]:
            if p["result"] is None:
                rs = [log["legs"].get(l, {}).get("result") for l in p["legs"]]
                if "L" in rs:
                    p["result"] = "L"
                elif all(r in ("W", "P") for r in rs):
                    p["result"] = "W" if "W" in rs else "P"
                if p["result"]:
                    changed = True
        if changed:
            path.write_text(json.dumps(log, indent=1))


def summary() -> dict:
    """Aggregate track record across every logged day."""
    legs, parlays = [], []
    for path in sorted(LOG_DIR.glob("*.json")):
        log = _load(path)
        legs += [l for l in log["legs"].values() if l["result"] in ("W", "L")]
        parlays += [p for p in log["parlays"] if p["result"] in ("W", "L")]

    def stats(rows, odds_key="odds", p_key="p"):
        if not rows:
            return {"n": 0}
        w = sum(r["result"] == "W" for r in rows)
        profit = sum((american_to_decimal(r[odds_key]) - 1) if r["result"] == "W" else -1 for r in rows)
        return {"n": len(rows), "wins": w, "hit_rate": round(w / len(rows), 3),
                "expected_hit_rate": round(sum(r[p_key] for r in rows) / len(rows), 3),
                "units": round(profit, 2), "roi": round(profit / len(rows), 3)}

    by_grade = {g: stats([l for l in legs if l.get("grade") == g]) for g in "ABCD"}
    by_sport = {}
    for l in legs:
        by_sport.setdefault(l["sport"], []).append(l)
    clv = [implied_prob(l["close_odds"]) - implied_prob(l["odds"]) for l in legs if l.get("close_odds")]
    return {
        "straight": stats(legs),
        "by_grade": by_grade,
        "by_sport": {k: stats(v) for k, v in by_sport.items()},
        "parlays": stats(parlays, odds_key="american"),
        "avg_clv": round(sum(clv) / len(clv), 4) if clv else None,
        "recent": sorted(legs, key=lambda l: l["start"], reverse=True)[:30],
    }
