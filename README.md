# BETAVUS ⚽

Football Goal Probability Engine.

BETAVUS is a mobile-friendly football dashboard for the following six leagues:

- Premier League
- LaLiga
- Bundesliga
- Serie A
- Ligue 1
- Eredivisie

## Architecture

There is **no serverless backend** in the live app.

`API-Football → GitHub Actions → predictions.json → Vercel static site`

The API key is used only inside the GitHub Actions secret `API_FOOTBALL_KEY`. It is never sent to the browser.

The daily job refreshes fixtures and calculations, keeps historical/H2H data in `data/cache`, and commits the resulting `predictions.json`. The Vercel site only reads that JSON file.

## Model

The first BETAVUS model combines:

1. H2H last 10 matches
2. Each team's completed matches in the last 365 days
3. Recent five matches with higher weights
4. Home/away split
5. Five-season league goal baseline
6. Poisson goal distribution

The dashboard displays Over 0.5, Over 1.5 and Over 2.5 goal probabilities.

## Deployment

The repository is connected to Vercel. Any push to `main` triggers a new static deployment.

<!-- BETAVUS data pipeline verified 2026-09-09 -->
