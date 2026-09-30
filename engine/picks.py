"""Build every candidate bet ("leg") for upcoming games, with probabilities and reasons."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from functools import lru_cache
from pathlib import Path

from . import espn, mlb, ratings
from .dist import Outcome
from .odds import blend, devig_power, expected_value, implied_prob
from .sports import Sport

KEY_POSITIONS = {"nfl": {"QB"}, "cfb": {"QB"}, "nhl": {"G"}, "nba": set(), "wnba": set(),
                 "ncaab": set(), "mlb": set()}
OUT_STATUSES = ("out", "injured reserve", "doubtful", "suspension", "il")
CAL_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration.json"


@lru_cache(maxsize=1)
def calibration() -> dict:
    return json.loads(CAL_PATH.read_text()) if CAL_PATH.exists() else {}


def calibrate(sport: Sport, R: ratings.Ratings, eh: float, ea: float) -> tuple[Sport, float, float]:
    """Apply backtest-derived stretch factors and outcome spreads (see backtest.py)."""
    cal = calibration().get(sport.key)
    if not cal:
        return sport, eh, ea
    if sport.kind == "gaussian":
        league_total = 2 * R.mu
        sport = dataclasses.replace(sport, margin_sd=cal["margin_sd"], total_sd=cal["total_sd"])
    else:
        league_total = math.exp(R.mu + R.home) + math.exp(R.mu)
    m = (eh - ea) * cal["margin_k"]
    t = league_total + (eh + ea - league_total) * cal["total_k"]
    lo = 0.05 if sport.kind == "poisson" else 0.0
    return sport, max((t + m) / 2, lo), max((t - m) / 2, lo)


def _fmt_odds(a: float | None) -> str:
    if a is None:
        return "—"
    return f"+{a:.0f}" if a > 0 else f"{a:.0f}"


def _is_out(status: str) -> bool:
    s = (status or "").lower()
    return any(k in s for k in OUT_STATUSES)


def _rest_days(last: dict[str, str], team_id: str, kickoff: str) -> int | None:
    if team_id not in last:
        return None
    return (dt.date.fromisoformat(kickoff[:10]) - dt.date.fromisoformat(last[team_id][:10])).days


class GameModel:
    """Everything the engine knows about one upcoming game."""

    def __init__(self, sport: Sport, game: dict, R: ratings.Ratings, hist: list[dict],
                 last_played: dict[str, str], extras: dict):
        h, a = game["home"]["id"], game["away"]["id"]
        self.notes: dict[str, list[str]] = {"home": [], "away": [], "game": []}
        eh, ea = R.expected(h, a, game["neutral"])
        sport, eh, ea = calibrate(sport, R, eh, ea)
        self.sport, self.g, self.R, self.hist = sport, game, R, hist
        self.base = (eh, ea)

        # --- rest / schedule spot
        self.rest = {"home": _rest_days(last_played, h, game["date"]),
                     "away": _rest_days(last_played, a, game["date"])}
        if sport.key in ("nba", "wnba", "nhl", "ncaab"):
            for side in ("home", "away"):
                if self.rest[side] == 1:
                    if sport.kind == "gaussian":
                        pen = 1.2
                        if side == "home":
                            eh -= pen / 2; ea += pen / 2
                        else:
                            ea -= pen / 2; eh += pen / 2
                    else:
                        if side == "home":
                            eh *= 0.96; ea *= 1.04
                        else:
                            ea *= 0.96; eh *= 1.04
                    self.notes[side].append("Playing the 2nd night of a back-to-back (historically worth about -1 to -1.5 pts in the NBA; tired legs and often the backup goalie in the NHL)")

        # --- MLB: starting pitchers and park
        self.pitchers = {}
        if sport.key == "mlb":
            key = f"{mlb._norm(game['away']['name'])}@{mlb._norm(game['home']['name'])}"
            pp = extras.get("pitchers", {}).get(key)
            tf = extras.get("team_fip", {})
            if pp:
                for side, opp in (("home", "away"), ("away", "home")):
                    p = pp.get(side)
                    m, note = mlb.starter_multiplier(p, tf.get(mlb._norm(game[side]["name"])))
                    if side == "home":
                        ea *= m
                    else:
                        eh *= m
                    if note:
                        self.pitchers[side] = p
                        tag = "below" if m < 0.99 else ("above" if m > 1.01 else "in line with")
                        self.notes[side].append(f"Starter {note}. Projects {tag} the team's usual pitching ({(m - 1) * 100:+.0f}% runs allowed)")
            pf = extras.get("parks", {}).get(h)
            if pf and not game["neutral"]:
                eh *= pf ** 0.75
                ea *= pf ** 0.75
                if abs(pf - 1) > 0.03:
                    self.notes["game"].append(
                        f"Park factor at {game.get('venue') or 'this ballpark'}: {(pf - 1) * 100:+.0f}% runs vs an average park")

        self.exp = (max(eh, 0.05), max(ea, 0.05))
        self.out = Outcome(sport, *self.exp)
        self.ctx: dict = extras.get("ctx", {}).get(game["id"], {})
        self.form = {s: ratings.team_form(hist, game[s]["id"]) for s in ("home", "away")}
        self.low_sample = min(R.games.get(h, 0), R.games.get(a, 0)) < 4

    # -------------------------------------------------------------- helpers
    def key_injuries(self, side: str) -> list[dict]:
        tid = self.g[side]["id"]
        return [i for i in self.ctx.get("injuries", {}).get(tid, []) if _is_out(i["status"])]

    def weight(self, favoured_side: str | None, market: str = "ml") -> float:
        """How much to trust our model vs the market for this game and market."""
        w = self.sport.w_model * self.sport.market_w.get(market, 1.0)
        if self.low_sample:
            w *= 0.3
        # Our ratings don't know about today's injuries; the market does. If we like a
        # team that is missing a key player, lean much harder on the market.
        if favoured_side:
            keys = [i for i in self.key_injuries(favoured_side)
                    if i.get("pos") in KEY_POSITIONS.get(self.sport.key, set())]
            if keys:
                w *= 0.25
        return w

    def team_line(self, side: str) -> list[str]:
        """Reasons describing one team."""
        g, R, f = self.g, self.R, self.form[side]
        t = g[side]
        lines = []
        rk = R.rank(t["id"])
        unit = {"mlb": "runs", "nhl": "goals"}.get(self.sport.key, "goals" if self.sport.three_way else "pts")
        if rk:
            lines.append(f"{t['abbr'] or t['name']} ranks #{rk[0]} in offense and #{rk[1]} in defense of {R.n_teams()} (schedule-adjusted, recent games weighted more)")
        if f.get("n"):
            s = f"Last {f['last_n']}: {f['last_wins']}-{f['last_n'] - f['last_wins']}, scoring {f['last_pf']} and allowing {f['last_pa']} {unit}/game (season: {f['season_pf']} / {f['season_pa']})"
            if f.get("streak") and int(f["streak"][1:]) >= 3:
                s += f". Current streak: {f['streak']}"
            lines.append(s)
            split_pf, split_pa = (f["home_pf"], f["home_pa"]) if side == "home" else (f["away_pf"], f["away_pa"])
            if split_pf is not None:
                lines.append(f"{'At home' if side == 'home' else 'On the road'}: {split_pf} scored / {split_pa} allowed per game")
        if t.get("record"):
            lines.append(f"Record {t['record']}" + (f" (home {t['home_record']})" if side == "home" and t.get("home_record") else "")
                         + (f" (road {t['road_record']})" if side == "away" and t.get("road_record") else ""))
        inj = self.key_injuries(side)
        if inj:
            names = ", ".join(f"{i['name']} ({i['pos']}, {i['status']})" for i in inj[:5])
            lines.append(f"Injuries: {names}" + (f" +{len(inj) - 5} more" if len(inj) > 5 else ""))
        if self.rest[side] and self.sport.key in ("nfl", "cfb") and self.rest[side] >= 13:
            lines.append("Coming off a bye week (extra rest)")
        lines += self.notes[side]
        return lines

    def game_lines(self) -> list[str]:
        g, (eh, ea) = self.g, self.exp
        unit = {"mlb": "runs", "nhl": "goals"}.get(self.sport.key, "goals" if self.sport.three_way else "pts")
        lines = [f"Model projection: {g['away']['abbr']} {ea:.1f} – {g['home']['abbr']} {eh:.1f} {unit} "
                 f"(total {eh + ea:.1f}, margin {eh - ea:+.1f} home)"]
        pred = self.ctx.get("predictor")
        if pred:
            lines.append(f"ESPN matchup predictor (independent model): {g['home']['abbr']} {pred['home'] * 100:.0f}% / {g['away']['abbr']} {pred['away'] * 100:.0f}%")
        h2h = ratings.head_to_head(self.hist, g["home"]["id"], g["away"]["id"])
        if h2h:
            hw = sum(1 for x in h2h if (x["hs"] > x["as"]) == (x["home"] == g["home"]["id"]))
            lines.append(f"Recent head-to-head: {g['home']['abbr']} {hw}-{len(h2h) - hw} in last {len(h2h)} meetings (avg total {sum(x['hs'] + x['as'] for x in h2h) / len(h2h):.1f})")
        if self.ctx.get("series"):
            lines.append(f"Season series: {self.ctx['series']}")
        lines += self.notes["game"]
        if self.low_sample:
            lines.append("⚠ Limited data on at least one team, so the pick leans mostly on the market price")
        return lines


def _movement_note(label: str, open_: float | None, now: float | None, is_line: bool = False) -> str | None:
    if open_ is None or now is None or open_ == now:
        return None
    if is_line:
        return f"Line moved from {open_:+g} to {now:+g} since opening ({label})"
    p0, p1 = implied_prob(open_), implied_prob(now)
    direction = "toward" if p1 > p0 else "away from"
    return f"Price moved {_fmt_odds(open_)} → {_fmt_odds(now)} since opening, money moving {direction} {label}"


def build_legs(gm: GameModel) -> list[dict]:
    sport, g, o, out = gm.sport, gm.g, gm.g.get("odds"), gm.out
    if not o:
        return []
    H, A = g["home"]["abbr"] or g["home"]["name"], g["away"]["abbr"] or g["away"]["name"]
    legs = []
    base = {"sport": sport.key, "sport_name": sport.name, "game_id": g["id"], "game": g["name"],
            "short": g["short"], "start": g["date"], "home": g["home"], "away": g["away"],
            "venue": g.get("venue"), "broadcast": g.get("broadcast"), "note": g.get("note")}
    shared = gm.game_lines()

    def add(market, side, selection, price, p_model, p_fair, reasons, link_key, line=None, p_push=0.0,
            favoured_side=None):
        if price is None or p_fair is None:
            return
        w = gm.weight(favoured_side if p_model > p_fair else None, market)
        p = blend(p_model, p_fair, w)
        legs.append({**base, "id": f"{g['id']}:{market}:{side}", "market": market, "side": side,
                     "selection": selection, "line": line, "odds": price,
                     "p": round(p, 4), "p_model": round(p_model, 4), "p_market": round(p_fair, 4),
                     "p_push": round(p_push, 4), "edge": round(p - p_fair, 4),
                     "ev": round(expected_value(p, price), 4), "w_model": round(w, 3),
                     "reasons": reasons, "context": shared,
                     "link": (o.get("links") or {}).get(link_key)})

    # ---------------- moneyline
    pred = gm.ctx.get("predictor")
    if sport.three_way and None not in (o["ml_home"], o["ml_away"], o["ml_draw"]):
        fh, fd, fa = devig_power([o["ml_home"], o["ml_draw"], o["ml_away"]])
        mh, md, ma = out.p_home_win(), out.p_draw(), out.p_away_win()
        add("ml", "home", f"{H} to win", o["ml_home"], mh, fh, gm.team_line("home"), "ml_home", favoured_side="home")
        add("ml", "away", f"{A} to win", o["ml_away"], ma, fa, gm.team_line("away"), "ml_away", favoured_side="away")
        add("ml", "draw", "Draw", o["ml_draw"], md, fd, ["Low-scoring matchups and evenly rated teams draw more often"], "ml_draw")
    elif o["ml_home"] is not None and o["ml_away"] is not None:
        fh, fa = devig_power([o["ml_home"], o["ml_away"]])
        mh = out.p_home_win()
        if pred:  # ensemble our ratings with ESPN's independent model
            mh = 0.5 * mh + 0.5 * pred["home"] / max(pred["home"] + pred["away"], 1e-9)
        mv_h = _movement_note(H, o.get("ml_home_open"), o["ml_home"])
        mv_a = _movement_note(A, o.get("ml_away_open"), o["ml_away"])
        add("ml", "home", f"{H} ML", o["ml_home"], mh, fh, gm.team_line("home") + ([mv_h] if mv_h else []), "ml_home", favoured_side="home")
        add("ml", "away", f"{A} ML", o["ml_away"], 1 - mh, fa, gm.team_line("away") + ([mv_a] if mv_a else []), "ml_away", favoured_side="away")

    # ---------------- spread (run line / puck line / handicap)
    sh = o.get("spread_home")
    if sh is not None and o.get("spread_home_odds") is not None and o.get("spread_away_odds") is not None:
        fh, fa = devig_power([o["spread_home_odds"], o["spread_away_odds"]])
        gt, eq = out.p_margin_gt(-sh)          # home covers if margin > -spread
        lt = max(0.0, 1 - gt - eq)
        denom = max(gt + lt, 1e-9)
        mv = _movement_note("spread", o.get("spread_home_open"), sh, is_line=True)
        mvl = [mv] if mv else []
        add("spread", "home", f"{H} {sh:+g}", o["spread_home_odds"], gt / denom, fh,
            gm.team_line("home") + mvl, "spread_home", line=sh, p_push=eq, favoured_side="home")
        add("spread", "away", f"{A} {-sh:+g}", o["spread_away_odds"], lt / denom, fa,
            gm.team_line("away") + mvl, "spread_away", line=-sh, p_push=eq, favoured_side="away")

    # ---------------- total
    tl = o.get("total")
    if tl is not None and o.get("over_odds") is not None and o.get("under_odds") is not None:
        fo, fu = devig_power([o["over_odds"], o["under_odds"]])
        gt, eq = out.p_total_gt(tl)
        lt = max(0.0, 1 - gt - eq)
        denom = max(gt + lt, 1e-9)
        tot_reasons = []
        for side in ("away", "home"):
            f, t = gm.form[side], g[side]
            if f.get("n"):
                tot_reasons.append(f"{t['abbr']}: {f['season_pf']} scored / {f['season_pa']} allowed per game this season; last {f['last_n']}: {f['last_pf']} / {f['last_pa']}")
        tot_reasons += [n for s in ("home", "away") for n in gm.notes[s] if "Starter" in n]
        mv = _movement_note("total", o.get("total_open"), tl, is_line=True)
        if mv:
            tot_reasons.append(mv)
        add("total", "over", f"Over {tl:g}", o["over_odds"], gt / denom, fo, tot_reasons, "over", line=tl, p_push=eq)
        add("total", "under", f"Under {tl:g}", o["under_odds"], lt / denom, fu, tot_reasons, "under", line=tl, p_push=eq)
    return legs


def confidence(leg: dict) -> str:
    """Letter grade from hit probability and edge."""
    p, e = leg["p"], leg["edge"]
    score = p + 1.5 * e
    if score >= 0.72:
        return "A"
    if score >= 0.64:
        return "B"
    if score >= 0.56:
        return "C"
    return "D"


def summarize(leg: dict) -> str:
    """One-sentence headline for a leg."""
    edge = leg["edge"] * 100
    s = f"{leg['p'] * 100:.0f}% to hit by our model vs {leg['p_market'] * 100:.0f}% implied by the market"
    if edge >= 1:
        s += f" (+{edge:.1f}% edge)"
    elif edge <= -1:
        s += " (market is more confident than we are)"
    return s


def finalize(legs: list[dict]) -> list[dict]:
    for leg in legs:
        leg["grade"] = confidence(leg)
        leg["headline"] = summarize(leg)
        leg["odds_str"] = _fmt_odds(leg["odds"])
    return legs
