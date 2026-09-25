"""One-off analysis: does a bad-weather (heavy rain / strong wind) damping
actually help, out of sample - the "Hava Sartlari" third of improvement
item D (Football Goal Analyst roadmap: referee / weather / pitch effects).

Stadium coordinates: scripts/stadium_geo.py (Wikidata-resolved, ~75%
coverage - a match involving an unresolved club just gets no weather
signal, same graceful-degradation idiom as fetch_key_players.py). Weather:
scripts/weather_data.py (Meteostat bulk data - Open-Meteo's archive API is
unreachable from this network, see that module's docstring), collapsed to
one precipitation total + max wind speed per day at the HOME team's nearest
station (a day-level signal, not hour-exact, since lining up a station's
local hour against football-data.co.uk's own local kickoff time isn't
reliably supported by either source).

severity(day) = max(precip_score, wind_score) in [0, 1]:
  precip_score saturates at 10mm/day (a genuinely wet matchday), wind_score
  starts above 20 km/h sustained and saturates at 50 km/h - a single
  "was conditions bad enough to matter" signal rather than two separate
  free parameters, kept deliberately simple for a first test.

Same walk-forward Brier grid-search discipline as every other tune_*.py
script here: train the goal model on the 4 prior seasons, damp BOTH teams'
lambda by strength * severity for the target match (bad weather affects
whoever's playing, not just the home side), keep a strength only if it
beats 0 by more than IMPROVE_MIN.

Writes data/weather-strengths.json; nothing reads it back automatically
until the finding is worth wiring into goals_model.py by hand.
"""

import csv
from datetime import datetime
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goals_model import LeagueModel, dc_score_grid  # noqa: E402
from stadium_geo import coords_for_league  # noqa: E402
from weather_data import daily_weather_for_coord  # noqa: E402
from opta_xg import fd_team_names  # noqa: E402

CSV_DIR = Path("data/football-data")
OUT_FILE = Path("data/weather-strengths.json")

DIVISIONS = {
    "E0": "Premier League", "E1": "Championship", "SP1": "LaLiga", "D1": "Bundesliga",
    "I1": "Serie A", "F1": "Ligue 1", "N1": "Eredivisie", "T1": "Turkish Süper Lig",
    "P1": "Primeira Liga", "B1": "Belgian Pro League",
}
ALL_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]
TARGET_SEASONS = ["2122", "2223", "2324", "2425", "2526"]
PRIOR_WEIGHTS = [1.0, 0.7, 0.45, 0.30]
LINES = [0.5, 1.5, 2.5]
GRID = [round(0.02 * i, 2) for i in range(11)]  # 0.0 .. 0.20
IMPROVE_MIN = 0.0015
PRECIP_SATURATE_MM = 10.0
WIND_START_KMH = 20.0
WIND_SATURATE_KMH = 50.0


def severity(precip_mm, wind_kmh):
    p = max(0.0, min(1.0, precip_mm / PRECIP_SATURATE_MM))
    w = max(0.0, min(1.0, (wind_kmh - WIND_START_KMH) / (WIND_SATURATE_KMH - WIND_START_KMH)))
    return max(p, w)


def load_division(div):
    rows = []
    for season in ALL_SEASONS:
        path = CSV_DIR / div / f"{season}.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                try:
                    hg, ag = int(r["FTHG"]), int(r["FTAG"])
                except (KeyError, ValueError):
                    continue
                try:
                    date = datetime.strptime(r["Date"].strip(), "%d/%m/%Y").date()
                except (KeyError, ValueError):
                    continue
                rows.append({
                    "season": season, "date_iso": date.isoformat(),
                    "home": r["HomeTeam"].strip(), "away": r["AwayTeam"].strip(),
                    "hg": hg, "ag": ag, "total": hg + ag,
                })
    return rows


def predictions_for_target(league, div, by_code, target, coords):
    ti = ALL_SEASONS.index(target)
    priors = ALL_SEASONS[max(0, ti - 4):ti][::-1]
    if len(priors) < 2:
        return None
    seasons = [(by_code.get(p, []), w) for p, w in zip(priors, PRIOR_WEIGHTS)]
    model = LeagueModel(seasons)
    target_matches = by_code.get(target, [])
    if not target_matches:
        return None

    out = []
    for m in target_matches:
        pred = model.predict(m["home"], m["away"])
        sev = 0.0
        coord = coords.get(m["home"])
        if coord:
            w = daily_weather_for_coord(coord[0], coord[1], m["date_iso"])
            if w:
                sev = severity(*w)
        out.append({"lam_h": pred["lam_home"], "lam_a": pred["lam_away"], "rho": pred["rho"],
                     "severity": sev, "total": m["total"]})
    return out


def brier_for_strength(cached_predictions, strength):
    briers = {line: [] for line in LINES}
    for c in cached_predictions:
        lam_h, lam_a = c["lam_h"], c["lam_a"]
        if strength and c["severity"]:
            damp = strength * c["severity"]
            lam_h *= (1 - damp)
            lam_a *= (1 - damp)
        grid = dc_score_grid(lam_h, lam_a, c["rho"])

        def over(n):
            return max(0.0, min(1.0, sum(p for (x, y), p in grid.items() if x + y > n)))

        for line, p in zip(LINES, (over(0), over(1), over(2))):
            outcome = 1 if c["total"] > line else 0
            briers[line].append((p - outcome) ** 2)

    avg_per_line = [sum(v) / len(v) for v in briers.values()]
    return sum(avg_per_line) / len(avg_per_line), len(cached_predictions)


def main():
    strengths = {}
    print(f"{'League':16} {'baseline':>9} {'best-s':>7} {'best':>9} {'delta':>8}  n")
    for div, league in DIVISIONS.items():
        rows = load_division(div)
        by_code = {}
        for m in rows:
            by_code.setdefault(m["season"], []).append(m)
        fd_names = {m["home"] for m in rows} | {m["away"] for m in rows}
        coords = coords_for_league(league, fd_names)
        n_with_coord = sum(1 for m in rows if m["home"] in coords)
        print(f"  [{league}] {len(coords)}/{len(fd_names)} clubs geolocated, "
              f"{n_with_coord}/{len(rows)} rows have a home-team coord")

        cached_by_target = {}
        for target in TARGET_SEASONS:
            c = predictions_for_target(league, div, by_code, target, coords)
            if c is not None:
                cached_by_target[target] = c

        per_strength = {}
        n_total = 0
        for s in GRID:
            total, n = 0.0, 0
            for cached in cached_by_target.values():
                b, nt = brier_for_strength(cached, s)
                total += b * nt
                n += nt
            if n:
                per_strength[s] = total / n
                n_total = n
        if not per_strength:
            print(f"{league:16} {'--':>9} {'--':>7} {'--':>9} {'--':>8}  0")
            strengths[league] = 0.0
            continue

        print("   curve:", {s: round(v, 4) for s, v in per_strength.items()})
        baseline = per_strength.get(0.0)
        best_s = min(per_strength, key=per_strength.get)
        best = per_strength[best_s]
        delta = baseline - best if baseline is not None else 0.0
        chosen = 0.0 if (best_s == 0.0 or delta < IMPROVE_MIN) else best_s
        strengths[league] = chosen
        print(f"{league:16} {baseline:9.4f} {best_s:7.2f} {best:9.4f} {delta:8.4f}  {n_total}"
              f"{'  -> kept 0.0 (no real gain)' if chosen == 0.0 and best_s != 0.0 else ''}")

    OUT_FILE.write_text(json.dumps(strengths, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nWrote {OUT_FILE}: {strengths}")


if __name__ == "__main__":
    main()
