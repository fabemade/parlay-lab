"""Parlay construction.

Rules (the same ones the website's interactive builder uses):
  * at most one leg per game (same-game legs are correlated and books price that in)
  * every leg must be priced inside the allowed odds range (no -900 "locks")
  * every leg must have non-negative edge: our probability >= the market's fair probability
  * "safest" mode: highest chance of hitting that still pays at least the target odds
  * "value" mode: highest expected return, with a floor on hit chance
"""
from __future__ import annotations

import itertools
import math

from .odds import american_to_decimal, decimal_to_american


def eligible(legs: list[dict], min_odds: float = -250, max_odds: float = 200,
             min_edge: float = 0.0, sports: set[str] | None = None) -> list[dict]:
    out = []
    for leg in legs:
        a = leg["odds"]
        if a is None or a < min_odds or a > max_odds:
            continue
        if leg["edge"] < min_edge:
            continue
        if sports and leg["sport"] not in sports:
            continue
        out.append(leg)
    return out


def price(combo: list[dict]) -> tuple[float, float]:
    """(hit probability, decimal odds). Legs in different games are treated as independent."""
    p = math.prod(leg["p"] for leg in combo)
    d = math.prod(american_to_decimal(leg["odds"]) for leg in combo)
    return p, d


def build(legs: list[dict], n: int, mode: str = "safest", target_american: float = 100,
          min_hit: float = 0.2, pool: int = 22, max_per_sport: int | None = None, **filters) -> dict | None:
    cands = eligible(legs, **filters)
    key = (lambda l: l["p"]) if mode == "safest" else (lambda l: l["ev"])
    # keep the best two options per game so the search can still diversify
    per_game: dict[str, list[dict]] = {}
    for leg in sorted(cands, key=key, reverse=True):
        per_game.setdefault(leg["game_id"], [])
        if len(per_game[leg["game_id"]]) < 2:
            per_game[leg["game_id"]].append(leg)
    cands = sorted((l for ls in per_game.values() for l in ls), key=key, reverse=True)
    if max_per_sport:   # keep the best few per sport so a mixed ticket is possible
        per_sport: dict[str, list] = {}
        for l in cands:
            per_sport.setdefault(l["sport"], []).append(l)
        cands = sorted((l for ls in per_sport.values() for l in ls[:6]), key=key, reverse=True)
    cands = cands[:pool]
    target_dec = american_to_decimal(target_american) if n > 1 else 1.0
    best, best_score = None, -1e9
    for combo in itertools.combinations(cands, n):
        if len({l["game_id"] for l in combo}) < n:
            continue
        if max_per_sport and max(sum(l["sport"] == s for l in combo) for s in {l["sport"] for l in combo}) > max_per_sport:
            continue
        p, d = price(combo)
        if mode == "safest":
            if d < target_dec:
                continue
            score = p
        else:
            if p < min_hit:
                continue
            score = p * d - 1
        if score > best_score:
            best, best_score = combo, score
    if not best:
        return None
    p, d = price(best)
    return {"legs": [l["id"] for l in best], "n": n, "mode": mode, "p": round(p, 4),
            "decimal": round(d, 3), "american": decimal_to_american(d),
            "ev": round(p * d - 1, 4)}
