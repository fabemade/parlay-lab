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

`run.py` options: `--sports nfl,mlb,nhl`, `--date 2026-10-04`, `--days 2`.
Sport keys: `nfl cfb nba ncaab wnba mlb nhl soccer_eng.1 soccer_esp.1 soccer_ita.1
soccer_ger.1 soccer_fra.1 soccer_usa.1 soccer_uefa.champions soccer_mex.1`.

## How it stays up to date

`.github/workflows/refresh.yml` runs four times a day on GitHub Actions (free). It
recalibrates, prices the board, grades finished picks, commits the data, and publishes
`site/` to GitHub Pages. You can also trigger it by hand from the repo's Actions tab.

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
  sports.py    per-sport settings
run.py         daily pipeline
backtest.py    walk-forward backtest and calibration
site/          the phone web app (static; reads site/data/picks.json)
data/          results history, calibration, pick logs
```

Please gamble responsibly. No model wins every bet. If it stops being fun, call 1-800-GAMBLER.
