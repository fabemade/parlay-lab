# Parlay Lab

A free, data-driven model that builds straight bets and parlays across the NFL, college
football, NBA, WNBA, college basketball, MLB, NHL and eight soccer leagues. Every pick
comes with the reasoning behind it: team ratings, form, splits, pitchers, injuries, line
movement and the model's probability vs the market's.

See [METHODOLOGY.md](METHODOLOGY.md) for how it works and what the backtests show.

## Use it

**On your phone:** open https://fabemade.github.io/parlay-lab/ in Safari, then tap Share and
"Add to Home Screen". It refreshes itself four times a day.

**On your Mac:**

```bash
./lab.sh            # refresh picks for all sports and open the app in your browser
./lab.sh nfl,mlb    # only some sports
```

Or step by step:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py         # price the next 4 days, build parlays, write site/data/picks.json
.venv/bin/python backtest.py    # walk-forward backtest + recalibration
.venv/bin/python -m http.server 8765 --directory site   # then open http://localhost:8765
```

`market_check.py` shows, per market type, how far the model leans from Kalshi on average
(run it after `run.py`).

`run.py` options: `--sports nfl,mlb,nhl`, `--date 2026-10-04`, `--days 2`.
Sport keys: `nfl cfb nba ncaab wnba mlb nhl soccer_eng.1 soccer_esp.1 soccer_ita.1
soccer_ger.1 soccer_fra.1 soccer_usa.1 soccer_uefa.champions soccer_mex.1`.

## Daily Top Picks

Each day at the first refresh after 10 AM Eastern, the app picks that day's featured
parlays and straight bets from games today and tomorrow and locks them. The Record tab
lists every day's picks, grades each leg from Kalshi's settlement, and marks the legs that
broke a parlay. Results feed back into how much the model trusts itself per bet type
(`data/feedback.json`).

## Live prices (optional, free)

Before building a parlay, and when you open your slip or Top picks, the app can re-check
every candidate bet against Kalshi's current price. It re-prices each one with the same
math as the engine and drops anything Kalshi has closed. Kalshi only answers browsers on
kalshi.com, so this goes through a tiny read-only relay, `worker/kalshi-prices.js`, on
Cloudflare's free plan:

1. Create a free account at https://dash.cloudflare.com/sign-up
2. **Workers & Pages → Create → Create Worker**, name it `parlay-lab-prices`, **Deploy**
3. **Edit code**, replace everything with `worker/kalshi-prices.js`, **Deploy**
4. Put the worker's URL (`https://parlay-lab-prices.<you>.workers.dev`) in
   `DEFAULT_PRICE_PROXY` near the top of the script in `site/index.html`

Without it, the app shows how old its prices are (the board refreshes every 30 minutes).

## How it stays up to date

`.github/workflows/refresh.yml` runs every 30 minutes from about 8am to 1:30am Eastern on
GitHub Actions (free). It recalibrates, prices the board, checks that every number is
consistent (odds, break-even, edge and parlay payouts all have to follow from Kalshi's
price and our probability, or nothing is published), grades finished picks, commits the
small history/log files, and publishes `site/` to GitHub Pages. You can also trigger it by hand from the repo's Actions tab.
In the repo's Settings → Pages, set **Source** to **GitHub Actions** so the workflow's deploy is
the one that's live (a root `index.html` also forwards to `site/` if Pages serves the branch instead).

## Layout

```
engine/
  espn.py      ESPN data: schedules, results cache, odds, injuries, context
  mlb.py       MLB Stats API: probable pitchers (FIP), park factors
  ratings.py   schedule-adjusted, recency-weighted offense/defense ratings
  dist.py      score distributions per sport -> win / spread / total probabilities
  picks.py     game adjustments, market blending, reasons, grades
  parlay.py    parlay search
  grade.py     pick log, grading, track record
  odds.py      odds conversions, de-vig, EV
  kalshi.py    Kalshi markets: discovery, matching, pricing every contract, de-biasing
  players.py   player props from ESPN game logs
  markets.py   periods, team totals and specials
  sports.py    per-sport settings
run.py         daily pipeline
backtest.py    walk-forward backtest and calibration
market_check.py  model-vs-Kalshi lean per market type
site/          the phone web app (static; reads site/data/picks.json)
data/          results history, calibration, pick logs
```

Please gamble responsibly. No model wins every bet. If it stops being fun, call 1-800-GAMBLER.
