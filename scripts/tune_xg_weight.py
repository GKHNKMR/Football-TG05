"""One-off analysis: does blending Opta xG into the goal model actually help,
out of sample, and if so by how much per league?

Reuses backtest.py's exact walk-forward scheme (train on the 4 prior seasons
only, predict the target season fresh, never touch its own outcomes) so the
comparison is apples-to-apples with the existing pure-goals model. For each
league, tries a grid of xg_weight values and keeps whichever minimizes
out-of-sample Brier score averaged across the three O/U lines - but only if
it beats xg_weight=0 by more than IMPROVE_MIN, so a league where xG doesn't
genuinely help stays on pure goals instead of chasing backtest noise.

Writes data/xg-weights.json ({"Premier League": 0.0, ...}) as a record of the
result. Nothing currently reads that file back - the measured gain across
all 9 leagues (see git history for a run's output) was 0.0002-0.0016 Brier
points, i.e. noise, so this was NOT wired into update_predictions.py/
build_results.py/backtest.py. Kept for re-checking later (more seasons of
data, or a smarter partial-blend design) rather than re-deriving from
scratch. Needs `pip install duckdb`.
"""

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goals_model import LeagueModel  # noqa: E402

CSV_DIR = Path("data/football-data")
OPTA_FILE = Path("data/opta-xg.json")
OUT_FILE = Path("data/xg-weights.json")

DIVISIONS = {
    "E0": "Premier League", "E1": "Championship", "SP1": "LaLiga", "D1": "Bundesliga",
    "I1": "Serie A", "F1": "Ligue 1", "N1": "Eredivisie", "T1": "Turkish Süper Lig",
    "P1": "Primeira Liga", "B1": "Belgian Pro League",
}
ALL_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]
TARGET_SEASONS = ["2122", "2223", "2324", "2425", "2526"]
PRIOR_WEIGHTS = [1.0, 0.7, 0.45, 0.30]
LINES = [0.5, 1.5, 2.5]
GRID = [round(0.1 * i, 1) for i in range(11)]  # 0.0 .. 1.0 - see the whole curve
IMPROVE_MIN = 0.0015  # brier points; below this, call it noise and keep 0.0
EPS = 1e-9

CODE_TO_OPTA_SEASON = {c: f"20{c[:2]}-20{c[2:]}" for c in ALL_SEASONS}


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
                    "season": season, "date": date.isoformat(),
                    "home": r["HomeTeam"].strip(), "away": r["AwayTeam"].strip(),
                    "hg": hg, "ag": ag, "total": hg + ag,
                })
    return rows


def load_opta_by_div_season():
    if not OPTA_FILE.exists():
        return {}
    rows = json.loads(OPTA_FILE.read_text(encoding="utf-8"))
    out = {}
    season_to_code = {v: k for k, v in CODE_TO_OPTA_SEASON.items()}
    for r in rows:
        code = season_to_code.get(r["season"])
        if not code:
            continue
        out.setdefault(r["div"], {}).setdefault(code, []).append(r)
    return out


def brier_for_weight(by_code, xg_by_code, target, xg_weight):
    ti = ALL_SEASONS.index(target)
    priors = ALL_SEASONS[max(0, ti - 4):ti][::-1]
    if len(priors) < 2:
        return None
    seasons = [(by_code.get(p, []), w) for p, w in zip(priors, PRIOR_WEIGHTS)]
    xg_seasons = [(xg_by_code.get(p, []), w) for p, w in zip(priors, PRIOR_WEIGHTS)]
    model = LeagueModel(seasons, xg_seasons=xg_seasons, xg_weight=xg_weight)
    target_matches = by_code.get(target, [])
    if not target_matches:
        return None
    briers = []
    for line in LINES:
        pk = {0.5: "p_over_0_5", 1.5: "p_over_1_5", 2.5: "p_over_2_5"}[line]
        total_brier, n = 0.0, 0
        for m in target_matches:
            pred = model.predict(m["home"], m["away"])
            p = pred[pk]
            o = 1 if m["total"] > line else 0
            total_brier += (p - o) ** 2
            n += 1
        briers.append(total_brier / n)
    return sum(briers) / len(briers), len(target_matches)


def main():
    opta_by_div = load_opta_by_div_season()
    weights = {}
    print(f"{'League':16} {'baseline':>9} {'best-w':>7} {'best':>9} {'delta':>8}  n")
    for div, league in DIVISIONS.items():
        rows = load_division(div)
        by_code = {}
        for m in rows:
            by_code.setdefault(m["season"], []).append(m)
        xg_by_code = opta_by_div.get(div, {})

        per_weight = {}
        n_total = 0
        for w in GRID:
            total, n = 0.0, 0
            for target in TARGET_SEASONS:
                r = brier_for_weight(by_code, xg_by_code, target, w)
                if r is None:
                    continue
                b, nt = r
                total += b * nt
                n += nt
            if n:
                per_weight[w] = total / n
                n_total = n
        if not per_weight:
            print(f"{league:16} {'--':>9} {'--':>7} {'--':>9} {'--':>8}  0")
            weights[league] = 0.0
            continue

        print("   curve:", {w: round(v, 4) for w, v in per_weight.items()})
        baseline = per_weight.get(0.0)
        best_w = min(per_weight, key=per_weight.get)
        best = per_weight[best_w]
        delta = baseline - best if baseline is not None else 0.0
        if best_w == 0.0 or delta < IMPROVE_MIN:
            chosen = 0.0
        else:
            chosen = best_w
        weights[league] = chosen
        print(f"{league:16} {baseline:9.4f} {best_w:7.1f} {best:9.4f} {delta:8.4f}  {n_total}"
              f"{'  -> kept 0.0 (no real gain)' if chosen == 0.0 and best_w != 0.0 else ''}")

    OUT_FILE.write_text(json.dumps(weights, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nWrote {OUT_FILE}: {weights}")


if __name__ == "__main__":
    main()
