"""Stdlib-only helpers for blending Opta xG into the live goal model.

See scripts/fetch_opta_xg.py for how data/opta-xg.json is built (needs
duckdb - deliberately NOT imported here, so update_predictions.py/
backtest.py/build_results.py keep running on a bare `python`) and
scripts/tune_xg_weight.py for how XG_WEIGHT_BY_LEAGUE was chosen: a
walk-forward backtest across all 9 leagues found the improvement noise-level
(~0.0002-0.0016 Brier points) everywhere except Premier League, so that's
the only league blending anything in - every other league behaves exactly
as it did before this file existed.
"""

import json
import re
from pathlib import Path

OPTA_XG_FILE = Path("data/opta-xg.json")

XG_WEIGHT_BY_LEAGUE = {
    "Premier League": 1.0,
}


def load_xg_by_div():
    """div -> list of {"home","away","home_xg","away_xg","date","season"},
    all football-data.co.uk short team names."""
    if not OPTA_XG_FILE.exists():
        return {}
    rows = json.loads(OPTA_XG_FILE.read_text(encoding="utf-8"))
    out = {}
    for r in rows:
        out.setdefault(r["div"], []).append(r)
    return out


def to_opta_season(code):
    """'2627' or '2026-27' or '2026-2027' -> '2026-2027' (Opta's own format)."""
    m = re.match(r"^(\d{2})(\d{2})$", code)
    if m:
        return f"20{m.group(1)}-20{m.group(2)}"
    m = re.match(r"^(\d{4})-(\d{2})$", code)
    if m:
        return f"{m.group(1)}-20{m.group(2)}"
    return code


def xg_seasons_for(div, codes_and_weights, rename=None, before_date=None, xg_by_div=None):
    """div: BETAVUS division code (e.g. "E0").
    codes_and_weights: [(season_code, weight), ...], any spelling
    to_opta_season() understands - matched against the caller's own season
    weighting scheme so training data lines up exactly with the goals side.
    rename: optional f(fd_name) -> name, to convert the football-data.co.uk
    team name into whatever convention the caller's LeagueModel is keyed by
    (e.g. openfootball's long names, via teams.to_pretty).
    before_date: optional ISO date string - keep only matches strictly
    before it (for build_results.py's leak-free per-match reconstruction).
    Returns a list shaped for LeagueModel's xg_seasons param."""
    by_div = load_xg_by_div() if xg_by_div is None else xg_by_div
    rows = by_div.get(div, [])
    by_season = {}
    for r in rows:
        by_season.setdefault(r["season"], []).append(r)
    out = []
    for code, w in codes_and_weights:
        season = to_opta_season(code)
        matches = by_season.get(season, [])
        if before_date is not None:
            matches = [m for m in matches if m["date"] < before_date]
        if rename:
            matches = [{**m, "home": rename(m["home"]), "away": rename(m["away"])} for m in matches]
        out.append((matches, w))
    return out
