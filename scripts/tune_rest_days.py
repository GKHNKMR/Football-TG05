"""One-off analysis: does a short-rest (fixture-congestion) damping actually
help, out of sample - the "Dinlenme Suresi (Rest Days)" half of improvement
item C (Football Goal Analyst roadmap: motivation / fixture congestion /
rotation risk).

Same walk-forward scheme as tune_sos_strength.py/tune_xg_weight.py: train the
goal model on the 4 prior seasons only, predict the target season fresh -
but "rest days" itself (days since that team's LAST match in this division,
including any midweek/rearranged round already present in the fixture list)
is computed straight from the target season's own known match dates, which
is not a leak: a real deployment also knows every upcoming fixture's kickoff
date and each team's most recent match date ahead of time (see
scripts/update_predictions.py). Only match OUTCOMES from the target season
stay out of the model.

Rest days also folds in each team's UEFA Champions/Europa/Conference League
match dates for that season (scripts/euro_fixtures.py, ESPN-sourced,
cached - see that module for how team identity is resolved across
providers), so a "played away in the Champions League on Tuesday" gap counts
toward Saturday's rest days exactly like a rearranged domestic midweek round
does - closing the gap this script's first version (domestic-only) flagged
as understating the real hypothesis.

For each league, tries a grid of rest_strength values (a single damping
multiplier applied to whichever side has short rest) and keeps whichever
minimizes out-of-sample Brier score averaged across the three O/U lines -
but only if it beats rest_strength=0 by more than IMPROVE_MIN, so a league
where this doesn't genuinely help stays uncorrected instead of chasing
backtest noise (identical policy to tune_sos_strength.py).

Writes data/rest-days-strengths.json as a record of the result; nothing
reads it back automatically until the finding is worth wiring into
goals_model.py by hand (mirrors tune_sos_strength.py/tune_xg_weight.py).
"""

import csv
import sys
import json
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goals_model import LeagueModel, dc_score_grid  # noqa: E402
from euro_fixtures import euro_dates_for_league_season  # noqa: E402

CSV_DIR = Path("data/football-data")
OUT_FILE = Path("data/rest-days-strengths.json")

DIVISIONS = {
    "E0": "Premier League", "E1": "Championship", "SP1": "LaLiga", "D1": "Bundesliga",
    "I1": "Serie A", "F1": "Ligue 1", "N1": "Eredivisie", "T1": "Turkish Süper Lig",
    "P1": "Primeira Liga",
}
ALL_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]
TARGET_SEASONS = ["2122", "2223", "2324", "2425", "2526"]
PRIOR_WEIGHTS = [1.0, 0.7, 0.45, 0.30]
LINES = [0.5, 1.5, 2.5]
GRID = [round(0.02 * i, 2) for i in range(11)]  # 0.0 .. 0.20
REST_THRESHOLD = 5   # days; at/above this, no penalty at all
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
                    "season": season, "date": date, "date_iso": date.isoformat(),
                    "home": r["HomeTeam"].strip(), "away": r["AwayTeam"].strip(),
                    "hg": hg, "ag": ag, "total": hg + ag,
                })
    return rows


def rest_days_by_match(season_matches, euro_dates_by_team=None):
    """season_matches sorted by date -> {(date_iso, home, away): (rest_home, rest_away)}.
    rest_* is the gap to that team's most recent EARLIER match in this
    division's own fixture list OR (if euro_dates_by_team is given) any of
    its UEFA competition matches that season - whichever is more recent.
    None means no earlier match is known at all (start of season)."""
    timeline = {}
    for m in season_matches:
        timeline.setdefault(m["home"], set()).add(m["date"])
        timeline.setdefault(m["away"], set()).add(m["date"])
    for team, dates in (euro_dates_by_team or {}).items():
        for d in dates:
            try:
                timeline.setdefault(team, set()).add(date.fromisoformat(d))
            except ValueError:
                continue
    timeline = {team: sorted(dates) for team, dates in timeline.items()}

    def most_recent_before(team, match_date):
        prev = None
        for d in timeline.get(team, ()):
            if d >= match_date:
                break
            prev = d
        return prev

    out = {}
    for m in season_matches:
        key = (m["date_iso"], m["home"], m["away"])
        lp_h = most_recent_before(m["home"], m["date"])
        lp_a = most_recent_before(m["away"], m["date"])
        rh = (m["date"] - lp_h).days if lp_h else None
        ra = (m["date"] - lp_a).days if lp_a else None
        out[key] = (rh, ra)
    return out


def damp(rest, strength):
    if rest is None or rest >= REST_THRESHOLD:
        return 0.0
    return strength * (REST_THRESHOLD - rest) / REST_THRESHOLD


def predictions_for_target(league, by_code, target):
    """One model build + one predict() per match for this target season -
    reused across the whole strength grid below, since lam/rho don't depend
    on the rest-days strength being tested."""
    ti = ALL_SEASONS.index(target)
    priors = ALL_SEASONS[max(0, ti - 4):ti][::-1]
    if len(priors) < 2:
        return None
    seasons = [(by_code.get(p, []), w) for p, w in zip(priors, PRIOR_WEIGHTS)]
    model = LeagueModel(seasons)
    target_matches = by_code.get(target, [])
    if not target_matches:
        return None
    fd_names = {m["home"] for m in target_matches} | {m["away"] for m in target_matches}
    euro_dates = euro_dates_for_league_season(league, target, fd_names)
    rest_lookup = rest_days_by_match(target_matches, euro_dates)

    out = []
    for m in target_matches:
        pred = model.predict(m["home"], m["away"])
        rh, ra = rest_lookup[(m["date_iso"], m["home"], m["away"])]
        out.append({"lam_h": pred["lam_home"], "lam_a": pred["lam_away"], "rho": pred["rho"],
                     "rest_h": rh, "rest_a": ra, "total": m["total"]})
    return out


def brier_for_strength(cached_predictions, strength):
    briers = {line: [] for line in LINES}
    for c in cached_predictions:
        lam_h, lam_a = c["lam_h"], c["lam_a"]
        if strength:
            lam_h *= (1 - damp(c["rest_h"], strength))
            lam_a *= (1 - damp(c["rest_a"], strength))
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
            c = predictions_for_target(league, by_code, target)
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
