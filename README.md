# BETAVUS ⚽

Football Goal Probability Engine.

BETAVUS is a mobile-friendly football dashboard for the following eight leagues:

- Premier League
- Championship
- Turkish Süper Lig
- LaLiga
- Bundesliga
- Serie A
- Ligue 1
- Eredivisie

## Architecture

There is **no serverless backend** and **no API key** in the live app.

`openfootball/football.json → GitHub Actions → predictions.json → static site`

Fixtures and historical results come from the open-data project
[`openfootball/football.json`](https://github.com/openfootball/football.json)
(`{season}/{code}.json`, e.g. `2026-27/en.1.json`). Season files are mirrored
under `data/cache/openfootball/` so the build is reproducible offline.

The daily job ([`scripts/update_predictions.py`](scripts/update_predictions.py))
downloads the current season plus the last three completed ones for each league,
builds a weighted home/away goals model and a head-to-head record, and commits the
resulting `predictions.json`. The static site only reads that JSON file
(`index.html` fetches the repo-relative `predictions.json`).

### Leagues and sources

| League | openfootball file |
| --- | --- |
| Premier League | `2026-27/en.1.json` |
| Championship | `2026-27/en.2.json` |
| Turkish Süper Lig | football-data only (openfootball has no current fixtures) |
| LaLiga | `2026-27/es.1.json` |
| Bundesliga | `2026-27/de.1.json` |
| Serie A | `2026-27/it.1.json` |
| Ligue 1 | `2026-27/fr.1.json` |
| Eredivisie | `2026-27/nl.1.json` |

## Model

The first BETAVUS model combines:

1. Each team's home / away goals-for and goals-against rates
2. Four seasons of results, weighted toward the most recent (1.0 / 0.7 / 0.45 / 0.30)
3. League-average fallback for newly promoted teams with no top-flight history
4. Head-to-head record, last 8 meetings, blended in at 28% when 2+ meetings exist
5. Poisson goal distribution on the combined expected goals

The dashboard displays Over 0.5, Over 1.5 and Over 2.5 goal probabilities. Each
row in `predictions.json` carries a `basis` field (`form`, `form+h2h`,
`partial-form`, `league-avg`) showing how much real data backs it.

## Match stats (click a fixture)

A second data layer powers the drawer that opens when a fixture row is clicked:
head-to-head history, recent form, current-season aggregates, and a
**per-match goal-average chart** (both teams' total goals per game over the last
five completed seasons, drawn as inline SVG).

`football-data.co.uk → CSVs → data/match-stats.json → drawer`

- [`scripts/fetch_football_data.py`](scripts/fetch_football_data.py) downloads the
  *Main Leagues* CSVs (top division only) into `data/football-data/<DIV>/<SEASON>.csv`
  for eight seasons. These raw CSVs are the local stats database (FT/HT results,
  shots, corners, cards, 1X2 / Over-Under / Asian-handicap odds).
- [`scripts/build_match_stats.py`](scripts/build_match_stats.py) maps each fixture's
  teams onto the football-data naming, then writes per-`match_id` H2H + form +
  season summaries to `data/match-stats.json`, which `index.html` fetches once.

| League | football-data division | CSV |
| --- | --- | --- |
| Premier League | `E0` | `mmz4281/<season>/E0.csv` |
| Championship | `E1` | `mmz4281/<season>/E1.csv` |
| Turkish Süper Lig | `T1` | `mmz4281/<season>/T1.csv` |
| LaLiga | `SP1` | `mmz4281/<season>/SP1.csv` |
| Bundesliga | `D1` | `mmz4281/<season>/D1.csv` |
| Serie A | `I1` | `mmz4281/<season>/I1.csv` |
| Ligue 1 | `F1` | `mmz4281/<season>/F1.csv` |
| Eredivisie | `N1` | `mmz4281/<season>/N1.csv` |

Use the apex domain `football-data.co.uk` (the `www` host currently 503s).

## Backtest (model validation — off-site)

[`scripts/backtest.py`](scripts/backtest.py) walk-forward tests the goal model:
each target season (2021/22 → 2025/26) is predicted using **only the seasons
before it** (up to four, no result leakage), then scored against what actually
happened, over ~13k matches in all eight leagues. Output `data/backtest.json` is rendered both by the **Model doğruluğu** tab in the app and by the standalone
[`backtest.html`](backtest.html) (`/backtest.html`): matches tested plus
0.5/1.5/2.5 Üst direction accuracy, sliceable by league and season.

Headline: the model is **well calibrated** (a stated 60% comes in near 60%) and
**unbiased** on expected goals (bias ≈ −0.04), but single-match discrimination
over the base-rate baseline is modest (Over 2.5 Brier skill ≈ +1.8%). Marketing
copy should lean on *calibration*, never on a hit-rate guarantee.

Run it with `python scripts/backtest.py`; it is **not** part of the daily
workflow.

## Deployment

The repository is connected to the static deployment. Any push to `main` triggers a new static deployment.

<!-- BETAVUS data pipeline verified 2026-09-09 -->
<!-- Calculation run requested 2026-09-09 -->
