# How Parlay Lab decides

This document explains what the model does, why, and what the evidence says about each
choice. The most important thing up front:

> **Sportsbook prices are the best free forecast that exists.** Closing lines at major books
> already fold in injuries, weather, sharp bettors' money and the books' own models. A
> model that ignores the market loses. A model that *blends* with the market and only acts
> where the data clearly disagrees is the realistic path to an edge.

No model hits 100%. Books take a margin (the "vig", about 4.5% on a -110/-110 line) on every
bet, and a parlay multiplies it: a 3-leg parlay of -110 legs carries roughly 13% hold. The
aim is to find legs where the true probability beats the break-even price, and to measure
honestly whether we do.

---

## 1. Data (all free)

| Source | What we use |
|---|---|
| ESPN public API (`site.api.espn.com`) | Schedules, final scores, DraftKings moneyline, spread and total with **opening and current prices**, injury reports, last-5 form, ESPN's matchup predictor (FPI/BPI), season series, probable pitchers |
| MLB Stats API (`statsapi.mlb.com`, official) | Probable starters, pitcher season stats and game logs, team pitching |
| Our own logs | Every featured pick with its price at publication and its latest price (for closing-line value), graded automatically |

Past days are cached permanently in `data/history/`, so each run only fetches today.

## 2. Team ratings

Each team gets an **offense** and a **defense** rating, fit on all games in a rolling window
(about 1–1.5 seasons):

- **Schedule-adjusted.** All teams are fit together, so a 30-point win over a bad team counts
  less than a close win over a good one (a ridge-regularised version of the Massey/SRS approach).
- **Recency-weighted.** Exponential decay with a sport-specific half-life (60 days for MLB,
  up to 180 for soccer), so current form matters without overreacting to one game.
- **Shrunk toward average** (ridge penalty), so small samples don't produce extreme numbers.
  Early in a season this also carries last season's ratings forward as the prior.
- **Home advantage** is estimated from the data per league, anchored to a research-based prior.
  NFL home edge has fallen to about 1.5 points since 2020; soccer and MLS home edges are larger.

Two model families:

- **Gaussian (NFL, CFB, NBA, WNBA, NCAAB):** points = baseline + offense − opponent defense ±
  home. Margins and totals are roughly normal.
- **Poisson / negative binomial (MLB, NHL, soccer):** scoring rates are multiplicative
  (log-linear), which fits low-scoring count data better.

## 3. Game adjustments

| Factor | Treatment | Evidence |
|---|---|---|
| **MLB starting pitcher** | Opponent run rate scaled by the starter's FIP relative to his team's staff, weighted by the share of innings he usually pitches and shrunk toward the team average by 50 IP | FIP (strikeouts, walks, HR) predicts future run prevention better than ERA; starters cover about 55–65% of innings |
| **MLB park factor** | Home/road run ratio, regressed 50% to average | Coors, Fenway and GABP inflate scoring; one season is noisy, so we regress |
| **Back-to-backs (NBA/WNBA/NHL/NCAAB)** | About −1.2 pts (basketball) or −4% goals / +4% allowed (hockey) | Well-documented fatigue effect; NHL teams usually start the backup goalie |
| **Playoffs** | Totals scaled down (MLB ×0.90, NHL ×0.93, NBA ×0.97) | Aces pitch more, benches shorten, pace slows |
| **Key injuries** | Not modelled as a rating change. If we like a team that is missing its QB or goalie, model weight drops to 25% | The market prices injuries far better than a heuristic can |
| **Rest / bye** | Shown in the reasoning | |
| **Streaks, head-to-head, home/road splits** | Shown in the reasoning, not added on top of ratings | Team-level "hot hand" and head-to-head history are mostly noise beyond what schedule-adjusted, recency-weighted ratings already capture. Adding them again double-counts |

## 4. From ratings to probabilities

The expected score becomes a **full distribution of outcomes**, and every market is priced
from that distribution:

- **NFL key numbers:** margins pile up on 3 and 7 (and 10, 14, 6, 4). A plain normal curve
  misses this, so the discrete margin distribution is reweighted. This matters most for
  spreads like −2.5 vs −3.5.
- **NHL empty-net goals:** about a fifth of one-goal regulation games become two-goal games
  once the trailing team pulls its goalie. Without this correction, +1.5 puck lines look
  much safer than they are. (Our first live run showed exactly that bias.)
- **MLB walk-offs:** the home team doesn't bat in the 9th when it's ahead, and walk-offs end
  games, so home wins land on exactly one run more often. This affects −1.5 run lines.
- **Soccer:** Dixon-Coles correction for low scores (0-0 and 1-1 happen more than
  independent Poisson predicts), with a proper three-way win/draw/win.
- **Overtime / extra innings:** regulation ties are split between the teams.

## 5. Blending with the market

1. **Remove the vig** from DraftKings prices using the *power method*, which puts more of
   the margin on longshots, matching the documented favourite-longshot bias.
2. **Blend** our probability with the fair market probability in log-odds space. The base
   weight on our model is 30%, lowered when:
   - the market is one our model handles poorly (totals ×0.35; MLB/NHL run and puck lines ×0.6)
   - a team has little data, or it's early season (fewer than 4 games this year)
   - a favoured team is missing its QB or goalie
   - the line has moved against the side since opening (a sharp-money signal)
   - the model and market disagree by a lot (weight ÷ (1 + gap²) in log-odds). Big
     disagreements usually mean we're missing information, not that we found a 15% edge.
3. For moneylines where ESPN publishes one (NFL, CFB, NBA, MLB), our rating model is first averaged 50/50 with ESPN's
   independent matchup predictor, a simple ensemble of two models.

**Edge** = our probability − the break-even probability *of the price you actually pay*
(vig included). Positive edge means positive expected value.

**Grades:** A = edge ≥ 2% and more likely than not · B = edge ≥ 0.5% · C = within 2% of
break-even · D = the price is too steep.

## 6. Building parlays

- **At most one leg per game.** Same-game legs are correlated (a favourite covering and the
  over, say), and books price that correlation into same-game parlays.
- **Legs are priced between −250 and +200 by default.** Heavy juice legs (−900 "locks")
  add almost nothing to the payout and still lose sometimes.
- **Safest mode:** among all combinations, pick the one with the highest chance of hitting
  that still pays at least the target (e.g. +150), with every leg +EV.
- **Value mode:** highest expected return, with a floor on hit chance.
- Legs from different games are treated as independent, so the parlay probability is the
  product of the legs.
- On thin or sharply priced slates the app features nothing. Passing is a valid bet.

## 7. Calibration and backtest (`backtest.py`)

Walk-forward: step through history one week at a time, fit only on games *before* that
week, and predict it. Nothing from the future leaks in. The results (Sep 30, 2026, ratings
model alone, without market blending or pitcher adjustments):

| League | Games | Winner accuracy | Brier (model) | Brier (baseline) |
|---|---:|---:|---:|---:|
| NFL | 268 | 64.9% | 0.225 | 0.249 |
| College football | 876 | 67.9% | 0.199 | 0.238 |
| NBA | 893 | 68.0% | 0.210 | 0.247 |
| WNBA | 263 | 68.4% | 0.193 | 0.245 |
| MLB | 2,285 | 54.7% | 0.247 | 0.249 |
| NHL | 866 | 55.9% | 0.245 | 0.250 |
| Premier League* | 362 | 65.8% | 0.213 | 0.245 |
| La Liga* | 381 | 66.4% | 0.211 | 0.232 |
| Serie A* | 361 | 68.0% | 0.200 | 0.249 |
| Bundesliga* | 269 | 73.3% | 0.185 | 0.237 |
| Ligue 1* | 281 | 65.7% | 0.215 | 0.242 |
| MLS* | 501 | 65.6% | 0.227 | 0.235 |
| Liga MX* | 323 | 64.6% | 0.221 | 0.236 |
| Champions League* | 123 | 68.6% | 0.210 | 0.240 |

| Europa League* | 118 | 66.0% | 0.207 | 0.236 |
| Eredivisie* | 294 | 67.6% | 0.205 | 0.245 |
| Primeira Liga* | 295 | 78.9% | 0.164 | 0.243 |
| Belgian Pro League* | 290 | 61.4% | 0.222 | 0.246 |
| Süper Lig* | 290 | 67.8% | 0.207 | 0.239 |
| Scottish Premiership* | 198 | 67.3% | 0.204 | 0.241 |
| Championship* | 571 | 58.7% | 0.234 | 0.244 |
| League One* | 543 | 61.7% | 0.231 | 0.241 |
| 2. Bundesliga* | 276 | 59.2% | 0.234 | 0.236 |
| LaLiga 2* | 462 | 58.9% | 0.230 | 0.243 |
| Serie B* | 359 | 69.4% | 0.205 | 0.226 |
| Ligue 2* | 285 | 61.7% | 0.226 | 0.247 |
| Brasileirão* | 411 | 68.1% | 0.207 | 0.217 |
| Argentine Primera* | 536 | 60.2% | 0.236 | 0.242 |
| Saudi Pro League* | 299 | 72.7% | 0.173 | 0.243 |

\*Soccer scored on games that didn't end in a draw. Conference League and Copa
Libertadores don't have enough history to backtest yet, so they lean mostly on the market.
Second divisions are more balanced (hence lower accuracy) but more thinly traded, so
market prices there are softer.

Takeaways that shaped the model:

- **Every league beats the baseline,** but MLB and NHL only barely: single games in those
  sports are close to coin flips for everyone, books included. Edges there come from
  starting pitchers and goalies, not team ratings.
- **Totals are our weakest area.** The backtest says predicted totals need shrinking 35–70%
  toward the league average (stretch factor k = 0.3–0.65). That's why totals lean hard on the
  market.
- **NBA predictions were too timid (k = 1.19); MLB/NHL too bold (k ≈ 0.7).** These factors
  are stored in `data/calibration.json` and applied automatically. The workflow recomputes
  them on every run.

What the backtest can't tell us yet is whether we **beat the closing line**, because ESPN
doesn't keep historical odds. So the app logs its own prices from now on. The tracked stats
are hit rate vs expected hit rate (calibration), profit in units, and **closing-line value
(CLV)**, the best-known early indicator of a real edge.

## 8. Every Kalshi market: props, periods, specials

The app lists every per-game contract Kalshi offers. For a typical NFL game that's about
1,000 contracts across 60+ market types. Kalshi's sports series are scanned every two days
(`data/kalshi_series.json`) to find which ones hold per-game markets, and each contract is
priced one of three ways.

**Player props** (`engine/players.py`): points, rebounds, assists, threes and combos;
passing/rushing/receiving yards, attempts, completions, receptions, TDs, INTs; hits, HRs,
total bases, RBIs, H+R+RBI, SBs; pitcher strikeouts, outs, hits/walks/earned runs allowed;
NHL goals, assists, points, saves; soccer goals, shots and shots on target.
- Each player's ESPN game log (this season, plus last season at half weight when the sample
  is small), with recent games weighted more (half-life of 8 games).
- Scaled to tonight's matchup. If the game model projects a team 10% above its usual
  scoring, its players' scoring stats move up with it. Pitcher and goalie "allowed" stats
  move with the *opponent's* projection.
- Counts (hits, goals, strikeouts, receptions) use a negative binomial fit to the player's
  own game-to-game spread. Yardage and big totals use a normal distribution. Both are
  shrunk toward typical spreads so a hot five-game stretch doesn't dominate.
- MLB playoff starters are projected about 12% lower in workload (quicker hooks).
- The reasons show the recent average, the hit rate at that exact line over the last 10 and
  the longer sample, home/road splits, recent games against this opponent, the matchup
  adjustment and tonight's projection.

**Periods and team markets** (`engine/markets.py`): quarter, half, period and
first-3/5/7-innings winners, spreads and totals, team totals, both-teams-to-score,
overtime/extra innings, correct score, winning-margin bands, YRFI/NRFI, and a goal in the
first 10 minutes. Each period gets its own score distribution:
- **Football:** each team's points = 7 × Poisson(TDs) + 3 × Poisson(FGs), which reproduces
  scores clustering on 0, 3, 7, 10, 14 and the frequent 0-0 / 7-7 quarters.
- **Basketball:** normal margins and totals, slightly wider for single quarters than simple
  scaling suggests.
- **Low-scoring sports:** scaled Poisson / negative binomial rates. NHL 2nd periods run
  highest, and regulation ties are boosted to the real ~23% overtime rate.

**Everything else** (first TD scorer, fantasy ladders, sacks, corners…) is listed at
Kalshi's own price with grade **M** (market only): available for your slip, never
presented as an edge.

**De-biasing.** After pricing, the model's median log-odds gap to Kalshi is measured for each
(league, market type) with 8+ contracts and removed. A model that runs 7% high on every
2nd-quarter total has a modelling error for that market type, not seven edges. What's left
are the contract-to-contract differences. `python market_check.py` prints the remaining
lean per market type after each run.

**Grading.** Every logged pick is graded from Kalshi's own settlement, so props and specials
are scored as reliably as game lines.

## 9. Keeping prices honest

- **Refresh cadence:** the board is rebuilt every 30 minutes through the betting day.
- **Publish-time check:** before anything goes live, every bet is re-derived. The
  break-even must equal Kalshi's price plus its fee, the odds must match that break-even,
  edge must equal probability minus break-even, and each blended probability must sit
  between the model's and the market's. Every featured parlay's hit chance and payout
  must be the product of its legs, with one leg per game and every leg +EV. Any failure
  stops the publish, and the last good board stays up.
- **Live re-pricing in the app:** with the price relay deployed, building a parlay first
  re-prices about 400 candidate bets from Kalshi's current quotes, searches, then makes
  sure every chosen leg was re-priced (and searches again if one wasn't). A price move
  shifts the fair market probability by exactly the move in log-odds, and it's re-blended
  with the model at the same weight the engine used, so the app and engine agree. Moved
  prices show "was 42¢", and closed markets are dropped.

## 10. Roadmap

- **Props:** minutes/usage projections and opponent positional defense (e.g. yards allowed to
  WRs); confirmed lineups and batting order for MLB
- **NHL starting goalies** (confirmed starters and save % above expected)
- **NFL/CFB:** QB-adjusted ratings, weather for totals (wind above 15 mph lowers scoring),
  travel and time-zone effects
- **Soccer:** expected goals (xG) instead of goals once a free xG source is wired in
- **Multi-book line shopping:** the same bet is often priced differently across books, which
  is free edge
- **Correlated same-game parlays**, modelled explicitly
