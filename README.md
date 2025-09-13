# Portfolio Dashboard — Step 1 (Clean Start)

A **simple and clean** Streamlit dashboard to view your portfolio snapshot
(stocks + options), current market value, and unrealized P/L with a pie chart
distribution by symbol.

This is the very first step — minimal code, minimal deps, easy to extend.

> **Disclaimer:** For research/education only. Not financial advice.

---

## Features (Step 1)
- Load positions from CSV (sample provided)
- Fetch **stock** last price via `yfinance`
- Fetch **option** prices via `yfinance` option chains (last price; fallback to mid of bid/ask)
- Compute market value, cost basis, and unrealized P/L
- Pie chart of portfolio value by symbol
- Simple summary metrics

---

## Quickstart

```bash
# 1) Create venv and install deps (Python 3.10+ recommended)
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# 2) Copy the example positions file and edit it with your holdings
cp data/positions.example.csv data/positions.csv

# 3) Run the dashboard
streamlit run app/app.py
```

Open http://localhost:8501 in your browser.

---

## Positions file format

The app expects a CSV at `data/positions.csv` (you can upload a file in the UI as well).

Columns:
- `type` — `stock` or `option`
- `symbol` — equity ticker (for stock) — *leave blank for options*
- `quantity` — shares (stock) or contracts (option)
- `avg_price` — per-share (stock) or per-contract premium (option)
- `underlying` — (option only) e.g., `SPY`
- `expiration` — (option only) ISO date, e.g., `2025-12-19`
- `strike` — (option only) numeric, e.g., `450`
- `right` — (option only) `C` or `P`
- `multiplier` — (option only) typically `100`

> If your option `expiration` isn't exactly listed by Yahoo, the app looks for
> the **nearest** available expiration to estimate price.

See `data/positions.example.csv` for a ready-to-edit template.

---

## Create a new GitHub repo (from this folder)

**With GitHub CLI (recommended):**
```bash
gh repo create portfolio-dash --public --source=. --remote=origin --push
```

**Manual Git commands:**
```bash
git init
git add .
git commit -m "step1: minimal portfolio dashboard (stocks + options)"
git branch -M main
git remote add origin git@github.com:<your-username>/<repo-name>.git  # or https://github.com/<user>/<repo>.git
git push -u origin main
```

---

## Next Steps (small, incremental)
- Step 2: Add historical P/L curve and a simple positions editor.
- Step 3: Add sectors/industries and distribution by sector.
- Step 4: Import parsers for common broker CSV exports.
- Step 5: Persist settings; refresh scheduling; basic caching controls.
- Step 6: Parameterized views (accounts/tags), tearsheet, notes per position.
- Step 7: Live alert tiles (price triggers) and watchlists.
