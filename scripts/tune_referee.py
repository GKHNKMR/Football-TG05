"""One-off analysis: does a referee-tendency adjustment actually help, out
of sample - the "Hakem" third of improvement item D (Football Goal Analyst
roadmap: referee / weather / pitch effects).

Restricted to Premier League + Championship: football-data.co.uk's CSVs
only carry a Referee column for the two English divisions (verified by
inspecting every division's header row) - the other 7 leagues in BETAVUS
have no referee data to test this on at all, and even if they did, a
referee appointment is usually only announced a few days before kickoff
(same live-wiring problem as scripts/fetch_lineups.py's missing-key-player
signal), so this stays a two-league, backtest-only question for now.

Method: for each walk-forward target season, using the SAME LeagueModel
already built from the 4 prior seasons (identical to backtest.py/
tune_sos_strength.py), compute each referee's average residual - actual
total goals minus that model's own exp_goals - across their appearances in
those same 4 prior seasons (a referee with too small a prior sample,
MIN_REF_MATCHES, gets no adjustment at all rather than a noisy one). For a
target-season match, that referee's residual is blended into the total
(same "rescale lam_home/lam_away keeping their ratio" idiom goals_model.py
already uses for the H2H blend), scaled by a grid-searched strength - kept
only if it beats strength=0 by more than IMPROVE_MIN (same policy as every
other tune_*.py script here).

Writes data/referee-strengths.json; nothing reads it back automatically
until the finding is worth wiring into goals_model.py by hand.
"""

import csv
from datetime import datetime
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goals_model import LeagueModel, dc_score_grid  # noqa: E402

CSV_DIR = Path("data/football-data")
OUT_FILE = Path("data/referee-strengths.json")

DIVISIONS = {"E0": "Premier League", "E1": "Championship"}
ALL_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]
TARGET_SEASONS = ["2122", "2223", "2324", "2425", "2526"]
PRIOR_WEIGHTS = [1.0, 0.7, 0.45, 0.30]
LINES = [0.5, 1.5, 2.5]
GRID = [round(0.1 * i, 1) for i in range(11)]  # 0.0 .. 1.0
MIN_REF_MATCHES = 15
IMPROVE_MIN = 0.0015


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
                    datetime.strptime(r["Date"].strip(), "%d/%m/%Y")
                except (KeyError, ValueError):
                    continue
                rows.append({
                    "season": season, "home": r["HomeTeam"].strip(), "away": r["AwayTeam"].strip(),
                    "hg": hg, "ag": ag, "total": hg + ag,
                    "referee": (r.get("Referee") or "").strip(),
                })
    return rows


def referee_residuals(model, prior_matches):
    """{referee: mean(actual_total - model.exp_goals) over prior matches},
    for referees with at least MIN_REF_MATCHES prior appearances."""
    sums, counts = {}, {}
    for m in prior_matches:
        ref = m["referee"]
        if not ref:
            continue
        pred = model.predict(m["home"], m["away"])
        resid = m["total"] - pred["exp_goals"]
        sums[ref] = sums.get(ref, 0.0) + resid
        counts[ref] = counts.get(ref, 0) + 1
    return {ref: sums[ref] / counts[ref] for ref in sums if counts[ref] >= MIN_REF_MATCHES}


def predictions_for_target(by_code, target):
    ti = ALL_SEASONS.index(target)
    priors = ALL_SEASONS[max(0, ti - 4):ti][::-1]
    if len(priors) < 2:
        return None
    seasons = [(by_code.get(p, []), w) for p, w in zip(priors, PRIOR_WEIGHTS)]
    model = LeagueModel(seasons)
    target_matches = by_code.get(target, [])
    if not target_matches:
        return None
    prior_matches = [m for p in priors for m in by_code.get(p, [])]
    residuals = referee_residuals(model, prior_matches)

    out = []
    for m in target_matches:
        pred = model.predict(m["home"], m["away"])
        out.append({"lam_h": pred["lam_home"], "lam_a": pred["lam_away"], "rho": pred["rho"],
                     "resid": residuals.get(m["referee"]), "total": m["total"]})
    return out


def apply_strength(lam_h, lam_a, resid, strength):
    if not resid or not strength:
        return lam_h, lam_a
    base_total = lam_h + lam_a
    new_total = max(0.30, min(6.0, base_total + strength * resid))
    if base_total > 1e-6:
        scale = new_total / base_total
        return lam_h * scale, lam_a * scale
    return new_total / 2, new_total / 2


def brier_for_strength(cached_predictions, strength):
    briers = {line: [] for line in LINES}
    for c in cached_predictions:
        lam_h, lam_a = apply_strength(c["lam_h"], c["lam_a"], c["resid"], strength)
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

        cached_by_target = {}
        for target in TARGET_SEASONS:
            c = predictions_for_target(by_code, target)
            if c is not None:
                cached_by_target[target] = c
        n_with_ref = sum(1 for cached in cached_by_target.values() for c in cached if c["resid"] is not None)
        n_all = sum(len(cached) for cached in cached_by_target.values())

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

        print(f"   ({n_with_ref}/{n_all} target matches have a referee with >= {MIN_REF_MATCHES} prior appearances)")
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
