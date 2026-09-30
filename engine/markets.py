"""Probabilities for everything beyond the full-game winner / spread / total:
periods (halves, quarters, periods, first-N innings), team totals and game specials.

Each period gets its own score distribution, derived from the full-game projection:

  football    each team's points = 7 x Poisson(touchdowns) + 3 x Poisson(field goals),
              which reproduces the lumpy scores (0, 3, 7, 10, 14) and the frequent
              0-0 / 7-7 quarters that a bell curve misses
  basketball  normal margins and totals, spread scaled by sqrt(period length)
  hockey,     independent Poisson / negative binomial scoring per team, with the
  baseball,   expected rate scaled to the period (NHL 2nd periods run highest, MLB
  soccer      first-inning scoring runs slightly above average, soccer 2nd halves
              produce more goals than 1st halves)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import nbinom, norm, poisson

# Share of full-game scoring in each period (league-wide scoring splits).
PERIODS = {
    "nfl": {"1Q": .20, "2Q": .30, "3Q": .21, "4Q": .28, "1H": .50, "2H": .49},
    "cfb": {"1Q": .24, "2Q": .28, "3Q": .24, "4Q": .24, "1H": .52, "2H": .48},
    "nba": {"1Q": .25, "2Q": .25, "3Q": .25, "4Q": .245, "1H": .50, "2H": .495},
    "wnba": {"1Q": .25, "2Q": .25, "3Q": .25, "4Q": .245, "1H": .50, "2H": .495},
    "ncaab": {"1H": .48, "2H": .52},
    "nhl": {"1P": .30, "2P": .35, "3P": .35},
    "mlb": {"F3": .345, "F5": .565, "F7": .785, "1I": .12},
    "soccer": {"1H": .45, "2H": .55},
}
TD_SHARE = 0.77   # share of NFL/CFB points that come from touchdowns (incl. extra points)


def period_share(sport_key: str, period: str) -> float | None:
    table = PERIODS.get("soccer" if sport_key.startswith("soccer_") else sport_key, {})
    return table.get(period)


def _football_pmf(mu: float, kmax: int = 90) -> np.ndarray:
    lam_td, lam_fg = TD_SHARE * mu / 7, (1 - TD_SHARE) * mu / 3
    td = poisson.pmf(np.arange(kmax // 7 + 1), lam_td)
    fg = poisson.pmf(np.arange(kmax // 3 + 1), lam_fg)
    p = np.zeros(kmax + 1)
    for i, a in enumerate(td):
        for j, b in enumerate(fg):
            s = 7 * i + 3 * j
            if s <= kmax:
                p[s] += a * b
    return p / p.sum()


def _count_pmf(mu: float, dispersion: float, kmax: int) -> np.ndarray:
    k = np.arange(kmax + 1)
    if dispersion <= 0:
        p = poisson.pmf(k, mu)
    else:
        r = 1 / dispersion
        p = nbinom.pmf(k, r, r / (r + mu))
    return p / p.sum()


class Scoreline:
    """Joint distribution of (home, away) score for a game or a period of it."""

    def __init__(self, sport, exp_home: float, exp_away: float, share: float = 1.0):
        self.sport = sport
        self.mh, self.ma = exp_home * share, exp_away * share
        self.share = share
        key = sport.key
        if key in ("nfl", "cfb"):
            ph, pa = _football_pmf(self.mh), _football_pmf(self.ma)
            self.kind = "matrix"
        elif sport.kind == "poisson":
            kmax = 25 if key == "mlb" else 12
            disp = sport.dispersion if share >= 0.99 else sport.dispersion + {"mlb": 0.35, "nhl": 0.12}.get(key, 0.0)
            ph, pa = _count_pmf(max(self.mh, 1e-3), disp, kmax), _count_pmf(max(self.ma, 1e-3), disp, kmax)
            self.kind = "matrix"
        else:
            self.kind = "normal"
            # single quarters/halves are noisier than sqrt-scaling of the full game implies
            # (runs, rotations, end-of-period fouling); checked against Kalshi's period lines
            wide = 1.0 if share >= 0.99 else (1.06 if share >= 0.45 else 1.14)
            self.m_sd = sport.margin_sd * math.sqrt(share) * wide
            self.t_sd = sport.total_sd * math.sqrt(share) * wide
            self.team_sd = math.sqrt((self.m_sd ** 2 + self.t_sd ** 2) / 4)
            return
        self.M = np.outer(ph, pa)
        i, j = np.indices(self.M.shape)
        self.i, self.j = i, j

    # -------- outcomes of this period
    def p_home(self) -> float:
        if self.kind == "matrix":
            return float(self.M[self.i > self.j].sum())
        return float(norm.sf(0.5, self.mh - self.ma, self.m_sd))

    def p_away(self) -> float:
        if self.kind == "matrix":
            return float(self.M[self.i < self.j].sum())
        return float(norm.cdf(-0.5, self.mh - self.ma, self.m_sd))

    def p_tie(self) -> float:
        return max(0.0, 1 - self.p_home() - self.p_away())

    def p_margin_gt(self, side: str, x: float) -> float:
        """P(side wins by more than x)."""
        if self.kind == "matrix":
            d = (self.i - self.j) if side == "home" else (self.j - self.i)
            return float(self.M[d > x].sum())
        m = (self.mh - self.ma) if side == "home" else (self.ma - self.mh)
        return float(norm.sf(x, m, self.m_sd))

    def p_total_gt(self, x: float) -> float:
        if self.kind == "matrix":
            return float(self.M[(self.i + self.j) > x].sum())
        return float(norm.sf(x, self.mh + self.ma, self.t_sd))

    def p_team_gt(self, side: str, x: float) -> float:
        if self.kind == "matrix":
            s = self.i if side == "home" else self.j
            return float(self.M[s > x].sum())
        return float(norm.sf(x, self.mh if side == "home" else self.ma, self.team_sd))

    def p_both_score(self, at_least: float = 1) -> float:
        if self.kind == "matrix":
            return float(self.M[(self.i >= at_least) & (self.j >= at_least)].sum())
        return float(norm.sf(at_least - 0.5, self.mh, self.team_sd) * norm.sf(at_least - 0.5, self.ma, self.team_sd))

    def p_exact(self, h: int, a: int) -> float:
        if self.kind == "matrix" and h < self.M.shape[0] and a < self.M.shape[1]:
            return float(self.M[h, a])
        return 0.0

    def p_first_to_score(self, side: str) -> float:
        """Rough: each team's chance to score first is its share of the scoring rate."""
        tot = self.mh + self.ma
        if tot <= 0:
            return 0.0
        none = self.p_exact(0, 0) if self.kind == "matrix" else 0.0
        return (1 - none) * ((self.mh if side == "home" else self.ma) / tot)


def p_first_inning_run(exp_home: float, exp_away: float) -> float:
    """MLB 'run in the 1st inning' (YRFI). Innings are streaky: most are scoreless, some
    are crooked numbers, so a heavily over-dispersed count fits far better than Poisson."""
    r = 0.55
    p0 = 1.0
    for mu in (exp_home * PERIODS["mlb"]["1I"], exp_away * PERIODS["mlb"]["1I"]):
        p0 *= (1 + mu / r) ** (-r)
    return 1 - p0


def p_goal_first_minutes(exp_total: float, minutes: float, game_minutes: float = 60) -> float:
    return 1 - math.exp(-exp_total * minutes / game_minutes * 0.9)
