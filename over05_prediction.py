
import os
import json
import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

import requests


# =========================
# Configuration
# =========================

# Read API key from environment (recommended).
# If you must hardcode for local testing, replace the fallback string below,
# but DO NOT commit secrets to public repositories.
API_KEY = os.getenv("API_FOOTBALL_KEY") or "REPLACE_WITH_YOUR_API_FOOTBALL_KEY"

BASE_URL = "https://v3.football.api-sports.io"

# Supported Leagues (API-Football IDs)
LEAGUE_IDS: Dict[str, int] = {
    "Premier League": 39,
    "Championship": 40,
    "Serie A": 135,
    "Bundesliga": 78,
    "La Liga": 140,
    "Primeira Liga": 94,
}

# Simple baselines for demo (replace with your trained model when ready)
LEAGUE_BASELINES: Dict[str, float] = {
    "Premier League": 0.95,
    "Championship": 0.93,
    "Serie A": 0.95,
    "Bundesliga": 0.96,
    "La Liga": 0.94,
    "Primeira Liga": 0.94,
}

ULTRA_TH = 0.98
HIGH_TH = 0.95


# =========================
# HTTP helpers
# =========================

def _headers() -> Dict[str, str]:
    key = API_KEY.strip() if API_KEY else ""
    if not key or key == "REPLACE_WITH_YOUR_API_FOOTBALL_KEY":
        raise RuntimeError(
            "API key missing. Set environment variable API_FOOTBALL_KEY."
        )
    return {"x-apisports-key": key}


def _get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """GET wrapper with light retry/backoff."""
    url = f"{BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=_headers(), params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            last_exc = Exception(f"HTTP {resp.status_code}: {resp.text[:300]}")
        except Exception as e:
            last_exc = e
        time.sleep(1 + attempt)  # simple backoff
    if last_exc:
        raise last_exc
    return {}


# =========================
# API calls
# =========================

def get_fixtures(
    league_id: int,
    date_from_utc: Optional[str],
    date_to_utc: Optional[str],
    season: Optional[int] = 2025,
    status: Optional[str] = None,  # None → do not filter by status
) -> List[Dict[str, Any]]:
    """
    Fetch fixtures from API-Football for the given league and date range (YYYY-MM-DD).
    Returns API 'response' list.
    """
    params: Dict[str, Any] = {"league": league_id}
    if season is not None:
        params["season"] = season
    if date_from_utc:
        params["from"] = date_from_utc
    if date_to_utc:
        params["to"] = date_to_utc
    if status:
        params["status"] = status

    data = _get("fixtures", params)
    return data.get("response", [])


# =========================
# Scoring (placeholder)
# =========================

def _baseline_probability(league_name: str) -> float:
    return float(LEAGUE_BASELINES.get(league_name, 0.94))


def _light_time_adjustment(kickoff_iso: str, base: float) -> float:
    """
    Tiny variation to avoid identical numbers in demo mode:
    +0.005 if kickoff <24h, -0.005 if kickoff >14d.
    """
    try:
        dt = datetime.fromisoformat(kickoff_iso.replace("Z", "+00:00"))
    except Exception:
        return base
    now = datetime.now(timezone.utc)
    delta = dt - now
    if delta <= timedelta(hours=24):
        return min(base + 0.005, 0.995)
    if delta >= timedelta(days=14):
        return max(base - 0.005, 0.90)
    return base


def _label_for(p: float) -> str:
    if p >= ULTRA_TH:
        return "ULTRA"
    if p >= HIGH_TH:
        return "HIGH"
    return ""


# =========================
# Public API for app.py
# =========================

def run_week_predictions(
    leagues: Optional[List[str]] = None,
    date_from_utc: Optional[str] = None,
    date_to_utc: Optional[str] = None,
    season: Optional[int] = 2025,
    save_json: bool = True,
    output_path: str = "predictions.json",
) -> List[Dict[str, Any]]:
    """
    Main entry point used by app.py and CLI.
    - leagues: subset of league names; None = all supported
    - date_from_utc / date_to_utc: YYYY-MM-DD strings; defaults to next 14 days if missing
    - season: integer like 2025
    - save_json: write predictions.json
    Returns a list of normalized dicts with keys:
      match_id, league, kickoff_utc, home, away, p_over_0_5, label
    """
    # Default date window = next 14 days
    if not date_from_utc or not date_to_utc:
        today = datetime.now(timezone.utc).date()
        if not date_from_utc:
            date_from_utc = today.isoformat()
        if not date_to_utc:
            date_to_utc = (today + timedelta(days=14)).isoformat()

    target_leagues = leagues or list(LEAGUE_IDS.keys())
    results: List[Dict[str, Any]] = []

    for league_name in target_leagues:
        league_id = LEAGUE_IDS.get(league_name)
        if not league_id:
            # Skip unknown league names silently
            continue

        fixtures = get_fixtures(
            league_id=league_id,
            date_from_utc=date_from_utc,
            date_to_utc=date_to_utc,
            season=season,
            status=None,  # be permissive; let API return anything in the window
        )

        # Normalize + Score
        for fx in fixtures:
            try:
                fixture_id = fx["fixture"]["id"]
                kickoff_iso = fx["fixture"]["date"]  # ISO8601
                home = fx["teams"]["home"]["name"]
                away = fx["teams"]["away"]["name"]
            except Exception:
                continue

            p = _baseline_probability(league_name)
            p = _light_time_adjustment(kickoff_iso, p)

            results.append({
                "match_id": fixture_id,
                "league": league_name,
                "kickoff_utc": kickoff_iso.replace("+00:00", "Z"),
                "home": home,
                "away": away,
                "p_over_0_5": round(float(p), 3),
                "label": _label_for(p),
            })

        # Friendly to API rate limits
        time.sleep(0.35)

    # Sort by probability desc
    results.sort(key=lambda r: r.get("p_over_0_5", 0.0), reverse=True)

    if save_json:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


# =========================
# CLI
# =========================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Over 0.5 predictions JSON (API-Football).")
    p.add_argument("--leagues", type=str, default="", help="Comma-separated league names (empty=all).")
    p.add_argument("--from", dest="date_from_utc", type=str, default="", help="Start date (UTC) YYYY-MM-DD.")
    p.add_argument("--to", dest="date_to_utc", type=str, default="", help="End date (UTC) YYYY-MM-DD.")
    p.add_argument("--season", type=int, default=2025, help="Season year (e.g., 2025).")
    p.add_argument("--out", dest="output_path", type=str, default="predictions.json", help="Output JSON path.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    leagues_arg = [s.strip() for s in args.leagues.split(",") if s.strip()] if args.leagues else None

    preds = run_week_predictions(
        leagues=leagues_arg,
        date_from_utc=args.date_from_utc or None,
        date_to_utc=args.date_to_utc or None,
        season=args.season,
        save_json=True,
        output_path=args.output_path,
    )

    print(f"Predictions written to {args.output_path}. Items: {len(preds)}")
