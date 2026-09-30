"""MLB specifics from the official MLB Stats API: starting pitchers and park factors.

The starting pitcher is the single biggest per-game factor in baseball. The team
ratings describe a team's *average* pitching; this module adjusts each game for who
actually starts, using FIP (fielding-independent pitching: strikeouts, walks and home
runs, which predict future run prevention better than ERA does).
"""
from __future__ import annotations

import datetime as dt
import unicodedata

import requests

S = "https://statsapi.mlb.com/api/v1"
FIP_CONST = 3.10
PRIOR_IP = 50.0          # shrink small samples toward the team average by this many innings


def _get(path: str, params: dict) -> dict:
    try:
        r = requests.get(f"{S}/{path}", params=params, timeout=30)
        return r.json() if r.ok else {}
    except requests.RequestException:
        return {}


def _norm(name: str) -> str:
    return unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower().strip()


def _ip(s) -> float:
    s = str(s or "0")
    whole, _, frac = s.partition(".")
    return int(whole or 0) + int(frac or 0) / 3


def fip(stat: dict) -> float | None:
    ip = _ip(stat.get("inningsPitched"))
    if ip <= 0:
        return None
    return (13 * stat.get("homeRuns", 0) + 3 * (stat.get("baseOnBalls", 0) + stat.get("hitByPitch", 0))
            - 2 * stat.get("strikeOuts", 0)) / ip + FIP_CONST


def team_fip(season: int) -> dict[str, float]:
    d = _get("teams/stats", {"stats": "season", "group": "pitching", "season": season, "sportIds": 1})
    out = {}
    for sp in (d.get("stats") or [{}])[0].get("splits", []):
        f = fip(sp["stat"])
        if f:
            out[_norm(sp["team"]["name"])] = f
    return out


def probable_pitchers(day: dt.date) -> dict[str, dict]:
    """Map normalised 'away@home' team names -> pitcher details for both sides."""
    d = _get("schedule", {"sportId": 1, "date": day.isoformat(), "hydrate": "probablePitcher,team"})
    games = {}
    ids = []
    for date in d.get("dates", []):
        for g in date.get("games", []):
            a, h = g["teams"]["away"], g["teams"]["home"]
            key = f"{_norm(a['team']['name'])}@{_norm(h['team']['name'])}"
            games[key] = {"away": a.get("probablePitcher"), "home": h.get("probablePitcher"),
                          "venue": g.get("venue", {}).get("name")}
            ids += [p["id"] for p in (a.get("probablePitcher"), h.get("probablePitcher")) if p]
    if not ids:
        return games
    people = _get("people", {"personIds": ",".join(map(str, ids)),
                             "hydrate": f"stats(group=[pitching],type=[season,gameLog],season={day.year})"})
    info = {}
    for p in people.get("people", []):
        season, log = {}, []
        for st in p.get("stats", []):
            kind = st["type"]["displayName"]
            if kind == "season" and st["splits"]:
                season = st["splits"][0]["stat"]
            elif kind == "gameLog":
                log = [s["stat"] for s in st["splits"] if s["stat"].get("gamesStarted")]
        starts = season.get("gamesStarted", 0) or 0
        ip = _ip(season.get("inningsPitched"))
        last = log[-5:]
        last_ip = sum(_ip(s.get("inningsPitched")) for s in last)
        last_er = sum(s.get("earnedRuns", 0) for s in last)
        info[p["id"]] = {
            "name": p["fullName"], "hand": (p.get("pitchHand") or {}).get("code"),
            "era": season.get("era"), "whip": season.get("whip"), "ip": round(ip, 1),
            "starts": starts, "fip": fip(season) if ip else None,
            "k": season.get("strikeOuts"), "bb": season.get("baseOnBalls"),
            "ip_per_start": ip / starts if starts else None,
            "last5_era": round(9 * last_er / last_ip, 2) if last_ip else None,
            "last5": [s.get("summary") for s in last],
        }
    for g in games.values():
        for side in ("away", "home"):
            if g[side]:
                g[side] = info.get(g[side]["id"], {"name": g[side].get("fullName")})
    return games


def starter_multiplier(p: dict | None, team_fip_value: float | None) -> tuple[float, str | None]:
    """Multiplier on the *opponent's* expected runs, and a one-line note."""
    if not p or p.get("fip") is None or not team_fip_value:
        return 1.0, None
    ip = p.get("ip") or 0
    shrunk = (p["fip"] * ip + team_fip_value * PRIOR_IP) / (ip + PRIOR_IP)
    share = min(max((p.get("ip_per_start") or 5.0), 3.0), 7.0) / 9
    mult = 1 + share * (shrunk - team_fip_value) / team_fip_value
    mult = min(max(mult, 0.75), 1.30)
    return mult, (f"{p['name']} ({p.get('hand') or '?'}HP): {p.get('era')} ERA, "
                  f"{p['fip']:.2f} FIP over {ip:.0f} IP"
                  + (f", {p['last5_era']} ERA last 5 starts" if p.get("last5_era") is not None else ""))


def park_factors(games: list[dict]) -> dict[str, float]:
    """Runs per game in each team's home park vs its road games, regressed halfway to 1."""
    home_runs, home_n, road_runs, road_n = {}, {}, {}, {}
    for g in games:
        if g.get("season_type") != 2:
            continue
        t = g["hs"] + g["as"]
        home_runs[g["home"]] = home_runs.get(g["home"], 0) + t
        home_n[g["home"]] = home_n.get(g["home"], 0) + 1
        road_runs[g["away"]] = road_runs.get(g["away"], 0) + t
        road_n[g["away"]] = road_n.get(g["away"], 0) + 1
    out = {}
    for team in home_runs:
        if home_n[team] < 20 or road_n.get(team, 0) < 20:
            continue
        raw = (home_runs[team] / home_n[team]) / (road_runs[team] / road_n[team])
        out[team] = 1 + 0.5 * (raw - 1)
    return out
