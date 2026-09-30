"""Team strength ratings fitted from every game in the window.

Each team gets an offensive and a defensive rating, estimated together so that
strength of schedule is handled automatically (beating good teams counts more).
Recent games count more (exponential decay) and ratings are shrunk toward league
average (ridge penalty) so small samples don't produce wild numbers.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .sports import Sport


@dataclass
class Ratings:
    kind: str
    mu: float                       # league baseline (points, or log goals/runs)
    home: float                     # home advantage
    off: dict[str, float]
    deff: dict[str, float]
    games: dict[str, int]           # games played in window per team
    names: dict[str, str]

    def expected(self, home_id: str, away_id: str, neutral: bool = False) -> tuple[float, float]:
        """Expected score (home, away)."""
        oh, dh = self.off.get(home_id, 0.0), self.deff.get(home_id, 0.0)
        oa, da = self.off.get(away_id, 0.0), self.deff.get(away_id, 0.0)
        h = 0.0 if neutral else self.home
        if self.kind == "gaussian":
            return self.mu + h / 2 + oh - da, self.mu - h / 2 + oa - dh
        return math.exp(self.mu + h + oh - da), math.exp(self.mu + oa - dh)

    def rank(self, team_id: str) -> tuple[int, int] | None:
        """(offense rank, defense rank) among teams with enough games."""
        eligible = [t for t, n in self.games.items() if n >= 3]
        if team_id not in eligible:
            return None
        o = sorted(eligible, key=lambda t: -self.off[t]).index(team_id) + 1
        d = sorted(eligible, key=lambda t: -self.deff[t]).index(team_id) + 1
        return o, d

    def n_teams(self) -> int:
        return sum(1 for n in self.games.values() if n >= 3)


def competitive(g: dict) -> bool:
    """Everything except preseason/exhibition games (ESPN season type 1)."""
    return g.get("season_type") != 1


def _weights(games: list[dict], today: dt.date, half_life: float) -> np.ndarray:
    ages = np.array([(today - dt.date.fromisoformat(g["date"][:10])).days for g in games], float)
    return 0.5 ** (np.maximum(ages, 0) / half_life)


def fit(sport: Sport, games: list[dict], today: dt.date) -> Ratings:
    games = [g for g in games if competitive(g) and dt.date.fromisoformat(g["date"][:10]) < today]
    teams = sorted({g["home"] for g in games} | {g["away"] for g in games})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    names = {}
    counts = {t: 0 for t in teams}
    for g in games:
        names[g["home"]] = g.get("home_name") or g["home"]
        names[g["away"]] = g.get("away_name") or g["away"]
        counts[g["home"]] += 1
        counts[g["away"]] += 1
    if not games:
        return Ratings(sport.kind, 0.0, sport.home_adv_prior, {}, {}, {}, {})
    w = _weights(games, today, sport.half_life_days)
    H = np.array([idx[g["home"]] for g in games])
    A = np.array([idx[g["away"]] for g in games])
    neutral = np.array([0.0 if g.get("neutral") else 1.0 for g in games])
    hs = np.array([g["hs"] for g in games])
    as_ = np.array([g["as"] for g in games])

    if sport.kind == "gaussian":
        # rows: home score and away score. params: mu, home, off[n], def[n]
        m = len(games)
        p = 2 + 2 * n
        X = np.zeros((2 * m, p))
        y = np.concatenate([hs, as_])
        r = np.arange(m)
        X[r, 0] = 1
        X[r, 1] = neutral / 2
        X[r, 2 + H] = 1
        X[r, 2 + n + A] = -1
        X[m + r, 0] = 1
        X[m + r, 1] = -neutral / 2
        X[m + r, 2 + A] = 1
        X[m + r, 2 + n + H] = -1
        sw = np.sqrt(np.concatenate([w, w]))
        Xw, yw = X * sw[:, None], y * sw
        # ridge rows: ratings -> 0, home advantage -> prior
        lam = sport.ridge
        R = np.zeros((p, p))
        R[np.arange(2, p), np.arange(2, p)] = math.sqrt(lam)
        R[1, 1] = math.sqrt(lam)
        ry = np.zeros(p)
        ry[1] = math.sqrt(lam) * sport.home_adv_prior
        beta, *_ = np.linalg.lstsq(np.vstack([Xw, R]), np.concatenate([yw, ry]), rcond=None)
        mu, home = beta[0], beta[1]
        off, deff = beta[2:2 + n], beta[2 + n:]
    else:
        lam = sport.ridge
        mean_goals = max(float(np.average(np.concatenate([hs, as_]), weights=np.concatenate([w, w]))), 0.1)

        def nll(theta):
            mu, home = theta[0], theta[1]
            off, deff = theta[2:2 + n], theta[2 + n:]
            eta_h = mu + home * neutral + off[H] - deff[A]
            eta_a = mu + off[A] - deff[H]
            lh, la = np.exp(eta_h), np.exp(eta_a)
            f = np.sum(w * (lh - hs * eta_h)) + np.sum(w * (la - as_ * eta_a))
            f += lam * (np.sum(off ** 2) + np.sum(deff ** 2)) + lam * 0.5 * (home - sport.home_adv_prior) ** 2
            rh, ra = w * (lh - hs), w * (la - as_)
            g = np.zeros_like(theta)
            g[0] = rh.sum() + ra.sum()
            g[1] = (rh * neutral).sum() + lam * (home - sport.home_adv_prior)
            g_off = np.bincount(H, rh, n) + np.bincount(A, ra, n) + 2 * lam * off
            g_def = -np.bincount(A, rh, n) - np.bincount(H, ra, n) + 2 * lam * deff
            g[2:2 + n], g[2 + n:] = g_off, g_def
            return f, g

        theta0 = np.zeros(2 + 2 * n)
        theta0[0], theta0[1] = math.log(mean_goals), sport.home_adv_prior
        res = minimize(nll, theta0, jac=True, method="L-BFGS-B")
        mu, home = res.x[0], res.x[1]
        off, deff = res.x[2:2 + n], res.x[2 + n:]

    return Ratings(sport.kind, float(mu), float(home),
                   {t: float(off[i]) for t, i in idx.items()},
                   {t: float(deff[i]) for t, i in idx.items()},
                   counts, names)


def last_game_dates(games: list[dict]) -> dict[str, str]:
    last: dict[str, str] = {}
    for g in games:
        for t in (g["home"], g["away"]):
            if g["date"] > last.get(t, ""):
                last[t] = g["date"]
    return last


def team_form(games: list[dict], team_id: str, n: int = 10) -> dict:
    """Recent results, scoring averages and streak for a team."""
    mine = sorted([g for g in games if team_id in (g["home"], g["away"])], key=lambda g: g["date"])
    if not mine:
        return {}
    # current season = everything after the last off-season gap (> 45 days)
    for i in range(len(mine) - 1, 0, -1):
        gap = dt.date.fromisoformat(mine[i]["date"][:10]) - dt.date.fromisoformat(mine[i - 1]["date"][:10])
        if gap.days > 45:
            mine = mine[i:]
            break
    rows = []
    for g in mine:
        home = g["home"] == team_id
        pf, pa = (g["hs"], g["as"]) if home else (g["as"], g["hs"])
        rows.append({"pf": pf, "pa": pa, "home": home, "won": pf > pa, "tie": pf == pa})
    recent = rows[-n:]
    streak_kind, streak = None, 0
    for r in reversed(rows):
        k = "W" if r["won"] else ("T" if r["tie"] else "L")
        if streak_kind in (None, k):
            streak_kind, streak = k, streak + 1
        else:
            break
    home_rows = [r for r in rows if r["home"]]
    away_rows = [r for r in rows if not r["home"]]

    def avg(rs, k):
        return round(sum(r[k] for r in rs) / len(rs), 2) if rs else None

    stale = (dt.date.today() - dt.date.fromisoformat(mine[-1]["date"][:10])).days > 45
    return {
        "season_label": "last season" if stale else "season",
        "n": len(rows),
        "last_n": len(recent),
        "last_wins": sum(r["won"] for r in recent),
        "last_pf": avg(recent, "pf"), "last_pa": avg(recent, "pa"),
        "season_pf": avg(rows, "pf"), "season_pa": avg(rows, "pa"),
        "home_pf": avg(home_rows, "pf"), "home_pa": avg(home_rows, "pa"),
        "away_pf": avg(away_rows, "pf"), "away_pa": avg(away_rows, "pa"),
        "streak": f"{streak_kind}{streak}" if streak_kind else None,
    }


def head_to_head(games: list[dict], a: str, b: str) -> list[dict]:
    return [g for g in games if {g["home"], g["away"]} == {a, b}][-5:]
