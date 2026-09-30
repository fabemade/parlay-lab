"""Turn expected scores into probabilities for every bet type."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import nbinom, norm, poisson

from .sports import Sport

# NFL games cluster on these final margins (field goals and touchdowns). A plain
# normal curve misses that, which matters most for spreads near 3 and 7.
NFL_KEY_WEIGHTS = {0: 0.25, 1: 0.9, 2: 0.9, 3: 2.5, 4: 1.3, 6: 1.3, 7: 1.8, 8: 1.1, 10: 1.4, 14: 1.3}
ENG_SHIFT = 0.22       # share of regulation one-goal NHL results that become two-goal (empty net)
WALKOFF_SHIFT = 0.08
NHL_TIE_BOOST = 1.18   # scales regulation-tie cells up to match the league's ~23% overtime rate   # share of 2+ run MLB home wins that end as one-run wins instead


class Outcome:
    """Joint distribution over (home score - away score) and total, as needed per sport."""

    def __init__(self, sport: Sport, exp_home: float, exp_away: float):
        self.sport = sport
        self.exp_home, self.exp_away = exp_home, exp_away
        self.margin = exp_home - exp_away
        self.total_mean = exp_home + exp_away
        if sport.kind == "poisson":
            self._build_matrix()
        elif sport.key in ("nfl", "cfb"):
            self._build_football_margin()

    # ---------- low scoring: full score matrix
    def _pmf(self, lam: float, kmax: int) -> np.ndarray:
        k = np.arange(kmax + 1)
        d = self.sport.dispersion
        if d <= 0:
            p = poisson.pmf(k, lam)
        else:
            r = 1 / d
            p = nbinom.pmf(k, r, r / (r + lam))
        return p / p.sum()

    def _build_matrix(self):
        kmax = 25 if self.sport.key == "mlb" else 12
        ph, pa = self._pmf(self.exp_home, kmax), self._pmf(self.exp_away, kmax)
        M = np.outer(ph, pa)
        if self.sport.three_way:
            # Dixon-Coles correction: low scores (0-0, 1-1) happen more often than
            # independent Poisson predicts; 1-0 / 0-1 slightly less.
            rho = -0.08
            lh, la = self.exp_home, self.exp_away
            M[0, 0] *= 1 - lh * la * rho
            M[0, 1] *= 1 + lh * rho
            M[1, 0] *= 1 + la * rho
            M[1, 1] *= 1 - rho
            M /= M.sum()
        if self.sport.key == "nhl":
            # Tied games after regulation happen more often than independent scoring predicts
            # (score effects: the trailing team pushes, the leader sits back). Real OT rate ~23%.
            M = M.copy()
            idx = np.arange(min(M.shape))
            M[idx, idx] *= NHL_TIE_BOOST
            M /= M.sum()
            # Empty-net goals: a team trailing by one late pulls its goalie, and about a
            # fifth of those games end as two-goal wins. Plain Poisson misses this, which
            # makes +1.5 puck lines look safer than they are.
            M = M.copy()
            for i in range(M.shape[0] - 1):
                for j in range(M.shape[1] - 1):
                    if i - j == 1:
                        moved = M[i, j] * ENG_SHIFT
                        M[i, j] -= moved
                        M[i + 1, j] += moved
                    elif j - i == 1:
                        moved = M[i, j] * ENG_SHIFT
                        M[i, j] -= moved
                        M[i, j + 1] += moved
        if self.sport.key == "mlb":
            # The home team skips the bottom of the 9th when it is ahead and walk-offs
            # stop the game, so home wins are by exactly one run more often than
            # independent run counts suggest. This matters for -1.5 run lines.
            M = M.copy()
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if i - j >= 2 and i >= 1:
                        moved = M[i, j] * WALKOFF_SHIFT
                        M[i, j] -= moved
                        M[j + 1, j] += moved
        self.M = M
        i, j = np.indices(M.shape)
        self.diff = i - j
        self.tot = i + j

    # ---------- football: discrete margin with key numbers
    def _build_football_margin(self):
        ks = np.arange(-80, 81)
        sd = self.sport.margin_sd
        p = norm.cdf(ks + 0.5, self.margin, sd) - norm.cdf(ks - 0.5, self.margin, sd)
        if self.sport.key == "nfl":
            wts = np.array([NFL_KEY_WEIGHTS.get(abs(int(k)), 1.0) for k in ks])
            p = p * wts
        self.ks, self.kp = ks, p / p.sum()

    # ---------- queries
    def p_home_win(self) -> float:
        """Probability home team wins (full game incl. OT / extra innings)."""
        s = self.sport
        if s.kind == "poisson":
            win = self.M[self.diff > 0].sum()
            tie = self.M[self.diff == 0].sum()
            if s.three_way:
                return float(win)
            # extra innings / overtime: split ties, leaning slightly to the stronger side
            share = 0.52 if s.key == "mlb" else 0.5 + 0.1 * math.tanh(self.margin)
            return float(win + tie * share)
        if hasattr(self, "kp"):
            win = self.kp[self.ks > 0].sum()
            tie = self.kp[self.ks == 0].sum()
            return float(win + tie * 0.5)
        return float(norm.cdf(self.margin / s.margin_sd))

    def p_draw(self) -> float:
        return float(self.M[self.diff == 0].sum()) if self.sport.kind == "poisson" else 0.0

    def p_away_win(self) -> float:
        if self.sport.three_way:
            return float(self.M[self.diff < 0].sum())
        return 1 - self.p_home_win()

    def p_margin_gt(self, x: float) -> tuple[float, float]:
        """(P(home margin > x), P(home margin == x)) in regulation/final score."""
        s = self.sport
        if s.kind == "poisson":
            M, diff = self.M, self.diff
            if not s.three_way:
                # ties get resolved by 1 run/goal in extras/OT
                share = 0.52 if s.key == "mlb" else 0.5 + 0.1 * math.tanh(self.margin)
                tie = M[diff == 0].sum()
                d = {}
                for k in np.unique(diff):
                    d[int(k)] = float(M[diff == k].sum())
                d[0] = 0.0
                d[1] = d.get(1, 0) + tie * share
                d[-1] = d.get(-1, 0) + tie * (1 - share)
                gt = sum(v for k, v in d.items() if k > x)
                eq = d.get(x, 0.0) if float(x).is_integer() else 0.0
                return gt, eq
            gt = float(M[diff > x].sum())
            eq = float(M[diff == x].sum()) if float(x).is_integer() else 0.0
            return gt, eq
        if hasattr(self, "kp"):
            gt = float(self.kp[self.ks > x].sum())
            eq = float(self.kp[self.ks == x].sum()) if float(x).is_integer() else 0.0
            return gt, eq
        sd = s.margin_sd
        if float(x).is_integer():
            return float(1 - norm.cdf(x + 0.5, self.margin, sd)), float(
                norm.cdf(x + 0.5, self.margin, sd) - norm.cdf(x - 0.5, self.margin, sd))
        return float(1 - norm.cdf(x, self.margin, sd)), 0.0

    def p_total_gt(self, line: float) -> tuple[float, float]:
        """(P(total > line), P(total == line))."""
        s = self.sport
        if s.kind == "poisson":
            gt = float(self.M[self.tot > line].sum())
            eq = float(self.M[self.tot == line].sum()) if float(line).is_integer() else 0.0
            return gt, eq
        sd = s.total_sd
        if float(line).is_integer():
            return float(1 - norm.cdf(line + 0.5, self.total_mean, sd)), float(
                norm.cdf(line + 0.5, self.total_mean, sd) - norm.cdf(line - 0.5, self.total_mean, sd))
        return float(1 - norm.cdf(line, self.total_mean, sd)), 0.0
