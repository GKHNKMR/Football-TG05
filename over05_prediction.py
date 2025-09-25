import os
import json
import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

import requests
import pandas as pd


# =========================
# Configuration
# =========================

# API key (hardcoded as requested). Prefer env var in the future for security.
API_KEY = os.getenv("API_FOOTBALL_KEY") or "a8b70416d123b5cc9a82aae8ea5ec065"

# API base
BASE_URL = "https://v3.football.api-sports.io"

# Supported leagues (API-Football league IDs)
LEAGUE_IDS: Dict[str, int] = {
    "Premier League": 39,
    "Championship": 40,
    "Serie A": 135,
    "Bundesliga": 78,
    "La Liga": 140,
    "Primeira Liga": 94,
}

# Simple league baselines for P(Over 0.5) – replace with model outputs when ready
LEAGUE_BASELINES: Dict[str, float] = {
    "Premier League": 0.95,
    "Championship": 0.93,
    "Serie A": 0.95,
    "Bundesliga": 0.96,
    "La Liga": 0.94,
    "Primeira Liga": 0.94,
}

# Label thresholds
ULTRA_TH = 0.98
HIGH_TH = 0.95


# =========================
# Low-level HTTP
# =========================

def _headers() -> Dict[str, str]:
    if not API_KEY or API_KEY.strip() == "":
        raise RuntimeError(
            "API key is missing. Set env var API_FOOTBALL_KEY or keep the hardcoded key."
        )
    return {"x-apisports-key": API_KEY}


def _get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """GET wrapper with basic retry/backoff."""
    url = f"{BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=_headers(), params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            # Rate-limit or transient issues: brief backoff
            time.sleep(1 + attempt)
            last_exc = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            last_exc = e
            time.sleep(1 + attempt)
    if last_exc:
        raise last_exc
    return {}


# =========================
# Data Fetch
# =========================

def get_fixtures(
    league_id: int,
    date_from_utc: Optional[str] = None,
    date_to_utc: Optional[str] = None,
    season: Optional[int] = None,
    status: str = "NS,1H,HT,2H,ET,BT,P"  # upcoming + in-play statuses (keeps near-future too)
) -> List[Dict[str, Any]]:
    """
    Fetch fixtures for a league. Filter by date range (YYYY-MM-DD) and season if provided.
    Returns API 'response' array (list of fixtures).
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
# Feature / Prediction (Placeholder)
# =========================

def _baseline_probability(league_name: str) -> float:
    """Return a baseline probability for Over 0.5 by league."""
    return float(LEAGUE_BASELINES.get(league_name, 0.94))


def _light_adjustment_by_time_to_kickoff(kickoff_iso: str, base: float) -> float:
    """
    Tiny demo-only adjustment to avoid identical numbers:
    - If the match is soon (< 24h), add +0.005 (capped to 0.995)
    - If it's far (> 14 days), subtract -0.005 (floored to 0.90)
    Replace with real model logic later.
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


def _label_for_prob(p: float) -> str:
    if p >= ULTRA_TH:
        return "ULTRA"
    if p >= HIGH_TH:
        return "HIGH"
    return ""


# =========================
# Public API
# =========================

def run_week_predictions(
    leagues: Optional[List[str]] = None,
    date_from_utc: Optional[str] = None,
    date_to_utc: Optional[str] = None,
    season: Optional[int] = None,
    save_json: bool = True,
    output_path: str = "predictions.json",
) -> List[Dict[str, Any]]:
    """
    Main entry point (used by app.py or CLI).
    - leagues: list of league names; None = all supported
    - date_from_utc / date_to_utc: filter fixtures by UTC date ("YYYY-MM-DD")
    - season: optional season year (e.g., 2025)
    - save_json: write predictions.json
    Returns: list of normalized prediction dicts.
    """
    # Defaults: next 7 days if not provided
    if not date_from_utc or not date_to_utc:
        today = datetime.now(timezone.utc).date()
        if not date_from_utc:
            date_from_utc = today.isoformat()
        if not date_to_utc:
            date_to_utc = (today + timedelta(days=7)).isoformat()

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
        )

        # Normalize & score
        for fx in fixtures:
            try:
                fixture_id = fx["fixture"]["id"]
                kickoff_iso = fx["fixture"]["date"]
                home = fx["teams"]["home"]["name"]
                away = fx["teams"]["away"]["name"]
            except Exception:
                # Skip malformed fixture
                continue

            # --- Placeholder probability logic (swap with your trained model later) ---
            p = _baseline_probability(league_name)
            p = _light_adjustment_by_time_to_kickoff(kickoff_iso, p)
            # ----------------------------------------------------------------------------

            results.append({
                "match_id": fixture_id,
                "league": league_name,
                "kickoff_utc": kickoff_iso.replace("+00:00", "Z"),
                "home": home,
                "away": away,
                "p_over_0_5": round(float(p), 3),
                "label": _label_for_prob(p),
            })

        # be nice to the API (avoid hammering)
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
    parser = argparse.ArgumentParser(description="Generate Over 0.5 predictions JSON.")
    parser.add_argument("--leagues", type=str, default="", help="Comma-separated league names (empty=all).")
    parser.add_argument("--from", dest="date_from_utc", type=str, default="", help="Start date (UTC) YYYY-MM-DD.")
    parser.add_argument("--to", dest="date_to_utc", type=str, default="", help="End date (UTC) YYYY-MM-DD.")
    parser.add_argument("--season", type=int, default=None, help="Season year (e.g., 2025).")
    parser.add_argument("--out", dest="output_path", type=str, default="predictions.json", help="Output JSON path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Build league list
    leagues_arg = [s.strip() for s in args.leagues.split(",") if s.strip()] if args.leagues else None

    # Run
    preds = run_week_predictions(
        leagues=leagues_arg,
        date_from_utc=args.date_from_utc or None,
        date_to_utc=args.date_to_utc or None,
        season=args.season,
        save_json=True,
        output_path=args.output_path,
    )

    print(f"Predictions written to {args.output_path}. Items: {len(preds)}")
