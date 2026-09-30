"""Odds math: conversions, removing the bookmaker's margin (vig), parlay pricing."""
from __future__ import annotations

import math


def american_to_decimal(american: float) -> float:
    return 1 + (american / 100 if american > 0 else 100 / -american)


def decimal_to_american(dec: float) -> int:
    if dec <= 1:
        return 0
    return round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1))


def implied_prob(american: float) -> float:
    """Raw implied probability, still including the book's vig."""
    return 1 / american_to_decimal(american)


def devig_power(american_prices: list[float]) -> list[float]:
    """Remove vig with the power method: find k so that sum(p_i^k) == 1.

    Unlike naive normalisation, this shifts more of the margin onto longshots,
    which matches the well-documented favourite-longshot bias in betting markets.
    """
    raw = [implied_prob(a) for a in american_prices]
    if sum(raw) <= 1:
        s = sum(raw)
        return [p / s for p in raw]
    lo, hi = 1.0, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if sum(p ** k for p in raw) > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    fair = [p ** k for p in raw]
    s = sum(fair)
    return [p / s for p in fair]


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def inv_logit(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def blend(p_model: float, p_market: float, w_model: float) -> float:
    """Blend two probabilities in log-odds space (a standard way to combine forecasts)."""
    return inv_logit(w_model * logit(p_model) + (1 - w_model) * logit(p_market))


def expected_value(prob: float, american: float) -> float:
    """Expected profit per $1 staked."""
    return prob * (american_to_decimal(american) - 1) - (1 - prob)


def kelly_fraction(prob: float, american: float) -> float:
    b = american_to_decimal(american) - 1
    return max(0.0, (b * prob - (1 - prob)) / b)


def parse_american(s) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().upper()
    if s in ("EVEN", "EV"):
        return 100.0
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return None
