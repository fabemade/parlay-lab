"""Player props: model a player's stat line from their own game logs.

For each player Kalshi lists, we pull the ESPN game log (this season, plus last season when
the sample is small), then:

  1. take a recency-weighted average (recent games count more; last season counts half),
  2. scale it to tonight's matchup: if our game model projects this team to score 15%
     more than it usually does, scoring and volume stats move up with it,
  3. turn that into a full distribution: negative binomial for counts (hits, goals,
     strikeouts, receptions), normal for yardage and big totals (passing yards, points),
     using the player's own game-to-game spread, shrunk toward typical values,
  4. read off P(stat >= N) for every "N+" line Kalshi lists.

The result is blended with Kalshi's price like every other leg, so a model that doesn't
know about a benching or a minutes restriction can't run away from the market.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from scipy.stats import nbinom, norm, poisson

SITE = "https://site.api.espn.com/apis/site/v2/sports"
WEB = "https://site.web.api.espn.com/apis/common/v3/sports"
HALF_LIFE_GAMES = 8
PITCHER_VOLUME = {"KS", "OUTS", "HA", "WA"}
PLAYOFF_STARTER_WORKLOAD = 0.88
MIN_GAMES = 4

_session = requests.Session()


def _get(url: str, params: dict | None = None) -> dict:
    for _ in range(3):
        try:
            r = _session.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (400, 404):
                return {}
        except requests.RequestException:
            pass
    return {}


def norm_name(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s.replace("-", " "))
    words = [w for w in s.split() if w not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(words)


def _num(v) -> float | None:
    """ESPN stat cell -> number ('8-16' -> 8 made, '5:25' -> 5.42 minutes)."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s in ("-", "--"):
        return None
    if ":" in s:
        m, _, sec = s.partition(":")
        try:
            return int(m) + int(sec) / 60
        except ValueError:
            return None
    if re.fullmatch(r"-?\d+-\d+(-\d+)?", s):
        return float(s.split("-")[0])
    try:
        return float(s)
    except ValueError:
        return None


def _outs(ip: float | None) -> float | None:
    """Innings pitched notation (6.2 = 6 and 2/3) -> outs."""
    if ip is None:
        return None
    whole = int(ip)
    return whole * 3 + round((ip - whole) * 10)


# Kalshi stat code -> (how to read it from a game log row, distribution family, context sensitivity)
# context: how strongly the stat moves with the team's projected scoring (1 = proportionally)
def _g(k):
    return lambda s: s.get(k)


def _sum(*ks):
    return lambda s: None if any(s.get(k) is None for k in ks) else sum(s[k] for k in ks)


STATS = {
    "mlb": {
        "HIT": (_g("hits"), "count", 0.5, "hits"),
        "HR": (_g("homeRuns"), "count", 0.8, "home runs"),
        "TB": (lambda s: None if s.get("hits") is None else s["hits"] + s.get("doubles", 0) + 2 * s.get("triples", 0) + 3 * s.get("homeRuns", 0), "count", 0.6, "total bases"),
        "RBI": (_g("RBIs"), "count", 0.9, "RBIs"),
        "SB": (_g("stolenBases"), "count", 0.2, "stolen bases"),
        "HRR": (_sum("hits", "runs", "RBIs"), "count", 0.8, "hits + runs + RBIs"),
        "KS": (_g("strikeouts"), "count", 0.0, "strikeouts"),
        "OUTS": (lambda s: _outs(s.get("innings")), "count", 0.0, "outs recorded"),
        "HA": (_g("hits"), "count", -0.6, "hits allowed"),
        "ERA": (_g("earnedRuns"), "count", -0.9, "earned runs"),
        "WA": (_g("walks"), "count", 0.0, "walks allowed"),
    },
    "hockey": {
        "GOAL": (_g("goals"), "count", 0.9, "goals"),
        "AST": (_g("assists"), "count", 0.8, "assists"),
        "PTS": (_g("points"), "count", 0.9, "points"),
        "SOG": (_g("shotsTotal"), "count", 0.5, "shots"),
        "SAVE": (_g("saves"), "normal", -0.6, "saves"),
    },
    "basketball": {
        "PTS": (_g("points"), "normal", 0.7, "points"),
        "REB": (_g("totalRebounds"), "count", 0.3, "rebounds"),
        "AST": (_g("assists"), "count", 0.5, "assists"),
        "3PT": (_g("threePointFieldGoalsMade-threePointFieldGoalsAttempted"), "count", 0.6, "threes"),
        "STL": (_g("steals"), "count", 0.2, "steals"),
        "BLK": (_g("blocks"), "count", 0.2, "blocks"),
        "PRA": (_sum("points", "totalRebounds", "assists"), "normal", 0.6, "pts + reb + ast"),
        "PR": (_sum("points", "totalRebounds"), "normal", 0.6, "pts + reb"),
        "PA": (_sum("points", "assists"), "normal", 0.6, "pts + ast"),
        "RA": (_sum("totalRebounds", "assists"), "normal", 0.4, "reb + ast"),
    },
    "football": {
        "PASSYDS": (_g("passingYards"), "normal", 0.6, "passing yards"),
        "PASSTDS": (_g("passingTouchdowns"), "count", 0.9, "passing TDs"),
        "PASSATT": (_g("passingAttempts"), "normal", 0.2, "pass attempts"),
        "PASSCOMP": (_g("completions"), "normal", 0.3, "completions"),
        "PASSINT": (_g("interceptions"), "count", 0.0, "interceptions"),
        "RSHYDS": (_g("rushingYards"), "normal", 0.5, "rushing yards"),
        "RSHATT": (_g("rushingAttempts"), "normal", 0.4, "rush attempts"),
        "REC": (_g("receptions"), "count", 0.4, "receptions"),
        "RECYDS": (_g("receivingYards"), "normal", 0.5, "receiving yards"),
        "RRYDS": (lambda s: (s.get("rushingYards") or 0) + (s.get("receivingYards") or 0)
                  if (s.get("rushingYards") is not None or s.get("receivingYards") is not None) else None,
                  "normal", 0.5, "rush + rec yards"),
        "TD": (lambda s: (s.get("rushingTouchdowns") or 0) + (s.get("receivingTouchdowns") or 0)
               if (s.get("rushingTouchdowns") is not None or s.get("receivingTouchdowns") is not None) else None,
               "count", 0.9, "touchdowns"),
        "LONGREC": (_g("longReception"), "normal", 0.3, "longest reception"),
        "LONGRSH": (_g("longRushing"), "normal", 0.3, "longest rush"),
    },
    "soccer": {
        "GOAL": (_g("totalGoals"), "count", 0.9, "goals"),
        "SOT": (_g("shotsOnTarget"), "count", 0.6, "shots on target"),
        "SHOT": (_g("totalShots"), "count", 0.5, "shots"),
        "AST": (_g("goalAssists"), "count", 0.8, "assists"),
        "SAVE": (_g("saves"), "count", -0.6, "saves"),
    },
}


def family(sport_key: str) -> str:
    if sport_key == "mlb":
        return "mlb"
    if sport_key == "nhl":
        return "hockey"
    if sport_key in ("nba", "wnba", "ncaab"):
        return "basketball"
    if sport_key in ("nfl", "cfb"):
        return "football"
    return "soccer"


class PlayerBook:
    """Rosters and game logs for one sport, fetched once per run."""

    def __init__(self, sport):
        self.sport = sport
        self.fam = family(sport.key)
        self._rosters: dict[str, dict[str, dict]] = {}
        self._logs: dict[str, list[dict]] = {}

    # ---------------- rosters
    def roster(self, team_id: str) -> dict[str, dict]:
        if team_id not in self._rosters:
            d = _get(f"{SITE}/{self.sport.path}/teams/{team_id}/roster")
            people = {}
            for a in d.get("athletes") or []:
                for p in (a.get("items") if isinstance(a, dict) and "items" in a else [a]):
                    if p.get("id"):
                        people[norm_name(p.get("fullName") or p.get("displayName"))] = {
                            "id": p["id"], "name": p.get("fullName") or p.get("displayName"),
                            "pos": (p.get("position") or {}).get("abbreviation"), "team": team_id}
            self._rosters[team_id] = people
        return self._rosters[team_id]

    def find(self, name: str, team_ids: list[str]) -> dict | None:
        key = norm_name(name)
        for t in team_ids:
            r = self.roster(t)
            if key in r:
                return r[key]
        # fall back to first initial + last name ("C. Sanchez" / nicknames)
        parts = key.split()
        if len(parts) >= 2:
            cands = [p for t in team_ids for n, p in self.roster(t).items()
                     if n.split() and n.split()[-1] == parts[-1] and n[0] == parts[0][0]]
            if len(cands) == 1:
                return cands[0]
        return None

    # ---------------- game logs
    def _parse(self, j: dict, weight: float) -> list[dict]:
        names = j.get("names") or []
        meta = j.get("events") or {}
        rows = []
        for st in j.get("seasonTypes") or []:
            if "preseason" in (st.get("displayName") or "").lower():
                continue
            for cat in st.get("categories") or []:
                for ev in cat.get("events") or []:
                    vals = ev.get("stats") or []
                    if len(vals) != len(names):
                        continue
                    m = meta.get(ev.get("eventId")) or {}
                    rows.append({
                        "event": ev.get("eventId"), "date": (m.get("gameDate") or "")[:10],
                        "opp": (m.get("opponent") or {}).get("id"),
                        "opp_abbr": (m.get("opponent") or {}).get("abbreviation"),
                        "home": m.get("atVs") == "vs", "w": weight,
                        "s": {k: _num(v) for k, v in zip(names, vals)},
                    })
        return rows

    def gamelog(self, athlete_id: str) -> list[dict]:
        if athlete_id in self._logs:
            return self._logs[athlete_id]
        url = f"{WEB}/{self.sport.path}/athletes/{athlete_id}/gamelog"
        j = _get(url)
        rows = self._parse(j, 1.0)
        if len(rows) < 12:
            seasons = next((f.get("options") for f in j.get("filters") or [] if f.get("name") == "season"), []) or []
            if len(seasons) > 1:
                rows += self._parse(_get(url, {"season": seasons[1]["value"]}), 0.5)
        seen, uniq = set(), []
        for r in sorted(rows, key=lambda r: r["date"], reverse=True):
            if r["event"] not in seen:
                seen.add(r["event"])
                uniq.append(r)
        self._logs[athlete_id] = uniq
        return uniq

    def prefetch(self, athlete_ids: list[str]):
        todo = [a for a in set(athlete_ids) if a not in self._logs]
        with ThreadPoolExecutor(max_workers=12) as ex:
            list(ex.map(self.gamelog, todo))


# ---------------------------------------------------------------- the prop model

class PropModel:
    """Distribution of one player's stat tonight."""

    def __init__(self, stat_code: str, fam: str, log: list[dict], context_mult: float,
                 opp_id: str | None, is_home: bool, postseason: bool = False):
        fn, self.kind, sens, self.label = STATS[fam][stat_code]
        self.code = stat_code
        rows = []
        for r in log:
            v = fn(r["s"])
            if v is None:
                continue
            if fam == "mlb" and stat_code in ("KS", "OUTS", "HA", "ERA", "WA"):
                if (_outs(r["s"].get("innings")) or 0) < 9:   # starts only (3+ innings)
                    continue
            if fam == "basketball" and (r["s"].get("minutes") or 0) < 5:
                continue
            rows.append((v, r))
        rows = rows[:30]
        self.n = len(rows)
        self.values = np.array([v for v, _ in rows], float)
        self.rows = [r for _, r in rows]
        if self.n == 0:
            self.mu = 0.0
            return
        w = np.array([0.5 ** (i / HALF_LIFE_GAMES) * r["w"] for i, r in enumerate(self.rows)])
        self.w = w / w.sum()
        base = float(np.dot(self.w, self.values))
        # The caller passes the right team's multiplier: own team for scoring stats, the
        # opponent's for stats a pitcher or goalie gives up (negative sensitivity).
        self.sens = sens
        self.context = max(0.7, min(1.4, context_mult)) ** abs(sens)
        self.base = base
        self.mu = max(base * self.context, 0.01)
        self.workload = 1.0
        if postseason and fam == "mlb" and stat_code in PITCHER_VOLUME:
            # playoff managers pull starters earlier (quicker hooks, deeper bullpens)
            self.workload = PLAYOFF_STARTER_WORKLOAD
            self.mu *= self.workload
        var = float(np.dot(self.w, (self.values - base) ** 2))
        k = 6.0   # prior strength, in games
        if self.kind == "count":
            prior_var = self.mu * (1 + 0.15 * self.mu)
        else:
            prior_var = (0.45 * self.mu + 3) ** 2
        self.var = (self.n * var + k * prior_var) / (self.n + k)
        self.opp_id, self.is_home = opp_id, is_home

    @property
    def usable(self) -> bool:
        return self.n >= MIN_GAMES

    def p_at_least(self, x: float) -> float:
        """P(stat > x) where x is Kalshi's floor strike (e.g. 4.5 for '5+')."""
        if self.kind == "count":
            k = math.floor(x) + 1
            mu, var = self.mu, max(self.var, self.mu * 1.0001)
            if var <= mu * 1.01:
                return float(poisson.sf(k - 1, mu))
            r = mu * mu / (var - mu)
            return float(nbinom.sf(k - 1, r, r / (r + mu)))
        sd = math.sqrt(max(self.var, 1e-6))
        return float(norm.sf(x, self.mu, sd))

    def reasons(self, player: str, x: float, opp_abbr: str | None) -> list[str]:
        if not self.n:
            return []
        need = math.floor(x) + 1 if self.kind == "count" else x
        need_s = f"{need:g}+" if self.kind == "count" else f"over {x:g}"
        last = self.values[:10]
        hit = int(np.sum(last > x))
        out = [f"{player}: {np.mean(last):.1f} {self.label} per game over the last {len(last)}, "
               f"{need_s} in {hit} of {len(last)}"]
        if self.n > len(last):
            out.append(f"Longer sample: {np.mean(self.values):.1f} per game over {self.n} games, "
                       f"{need_s} in {int(np.sum(self.values > x))} of {self.n}")
        where = [v for v, r in zip(self.values, self.rows) if r["home"] == self.is_home]
        if len(where) >= 3:
            out.append(f"{'At home' if self.is_home else 'On the road'}: {np.mean(where):.1f} per game ({len(where)} games)")
        vs = [v for v, r in zip(self.values, self.rows) if self.opp_id and r["opp"] == self.opp_id]
        if vs:
            out.append(f"Against {opp_abbr or 'this opponent'}: {', '.join(f'{v:g}' for v in vs[:5])} in the last {min(len(vs), 5)} meeting{'s' if len(vs) > 1 else ''}")
        if abs(self.context - 1) >= 0.03:
            out.append(f"Matchup adjustment: {(self.context - 1) * 100:+.0f}% (the game model projects "
                       f"{'more' if self.context > 1 else 'less'} {'scoring' if self.context > 1 else 'scoring'} than this team's usual)")
        if self.workload != 1.0:
            out.append("Playoff game: starters get pulled earlier, so workload is projected about 12% lower")
        out.append(f"Projection tonight: {self.mu:.1f} {self.label}")
        return out
