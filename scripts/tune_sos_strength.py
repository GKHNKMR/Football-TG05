"""One-off analysis: does a Strength-of-Schedule correction (see
"Strength of Schedule (SoS).txt") actually help, out of sample, and if so
how strongly per league?

Same walk-forward scheme as backtest.py/tune_xg_weight.py: train on the 4
prior seasons only, predict the target season fresh. For each league,
tries a grid of sos_strength values and keeps whichever minimizes
out-of-sample Brier score averaged across the three O/U lines - but only if
it beats sos_strength=0 by more than IMPROVE_MIN, so a league where SoS
doesn't genuinely help stays uncorrected instead of chasing backtest noise.

Writes data/sos-strengths.json as a record of the result; nothing reads it
back automatically (mirrors tune_xg_weight.py - wire the finding into
xg_blend.py-style config by hand once it's actually worth shipping).
"""

import csv
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goals_model import LeagueModel  # noqa: E402

CSV_DIR = Path("data/football-data")
OUT_FILE = Path("data/sos-strengths.json")

DIVISIONS = {
    "E0": "Premier League", "E1": "Championship", "SP1": "LaLiga", "D1": "Bundesliga",
    "I1": "Serie A", "F1": "Ligue 1", "N1": "Eredivisie", "T1": "Turkish Süper Lig",
    "P1": "Primeira Liga", "B1": "Belgian Pro League",
}
ALL_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]
TARGET_SEASONS = ["2122", "2223", "2324", "2425", "2526"]
PRIOR_WEIGHTS = [1.0, 0.7, 0.45, 0.30]
LINES = [0.5, 1.5, 2.5]
GRID = [round(0.1 * i, 1) for i in range(11)]  # 0.0 .. 1.0
IMPROVE_MIN = 0.0015  # brier points; below this, call it noise and keep 0.0


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


def brier_for_strength(by_code, target, sos_strength):
    ti = ALL_SEASONS.index(target)
    priors = ALL_SEASONS[max(0, ti - 4):ti][::-1]
    if len(priors) < 2:
        return None
    seasons = [(by_code.get(p, []), w) for p, w in zip(priors, PRIOR_WEIGHTS)]
    model = LeagueModel(seasons, sos_strength=sos_strength)
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
    strengths = {}
    print(f"{'League':16} {'baseline':>9} {'best-s':>7} {'best':>9} {'delta':>8}  n")
    for div, league in DIVISIONS.items():
        rows = load_division(div)
        by_code = {}
        for m in rows:
            by_code.setdefault(m["season"], []).append(m)

        per_strength = {}
        n_total = 0
        for s in GRID:
            total, n = 0.0, 0
            for target in TARGET_SEASONS:
                r = brier_for_strength(by_code, target, s)
                if r is None:
                    continue
                b, nt = r
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
        print(f"{league:16} {baseline:9.4f} {best_s:7.1f} {best:9.4f} {delta:8.4f}  {n_total}"
              f"{'  -> kept 0.0 (no real gain)' if chosen == 0.0 and best_s != 0.0 else ''}")

    OUT_FILE.write_text(json.dumps(strengths, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nWrote {OUT_FILE}: {strengths}")


if __name__ == "__main__":
    main()
