#!/bin/bash
# One command to refresh picks and open the app locally.
#   ./lab.sh              all sports, next 4 days
#   ./lab.sh nfl,mlb      only some sports
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run: setting up Python environment…"
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
if [ -n "$1" ]; then .venv/bin/python run.py --sports "$1"; else .venv/bin/python run.py; fi
echo
echo "Opening http://localhost:8765 (press Ctrl+C to stop)"
(sleep 1 && open http://localhost:8765) &
.venv/bin/python -m http.server 8765 --directory site
