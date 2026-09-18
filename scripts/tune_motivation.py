"""One-off analysis: does a "nothing left to play for" (mid-table, safe from
both relegation and continental qualification) damping actually help, out of
sample - the "Motivasyon Katsayisi" half of improvement item C (Football
Goal Analyst roadmap item C: motivation / fixture congestion / rotation
risk; the fixture-congestion half is scripts/tune_rest_days.py).

Same walk-forward scheme as tune_sos_strength.py/tune_rest_days.py: train
the goal model on the 4 prior seasons only, predict the target season
fresh. "Stakes" itself is reconstructed leak-free from the target season's
OWN prior matches only (a live standings table as of the day before each
fixture, using only results already played by then - no different from a
real deployment knowing the current table before an upcoming match, see
scripts/update_predictions.py).

Per-league relegation-zone and continental-qualification-zone sizes (see
LEAGUE_ZONES below) were verified league by league rather than assumed
uniform - see the comment there for sourcing and the simplifications made
(folding a relegation play-off spot into the zone size, and using one flat
"top N" count per league rather than modelling each season's exact,
sometimes-fluctuating UEFA slot allocation).

For each league, tries a grid of motivation_strength values (damping applied
to a team's lambda once it is mathematically/practically clear of BOTH
zones, scaled by how many points-worth of games remain) and keeps whichever
minimizes out-of-sample Brier score - but only if it beats strength=0 by
more than IMPROVE_MIN, identical policy to tune_sos_strength.py/
tune_rest_days.py.

Writes data/motivation-strengths.json; nothing reads it back automatically
until the finding is worth wiring into goals_model.py by hand.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goals_model import LeagueModel, dc_score_grid  # noqa: E402

CSV_DIR = Path("data/football-data")
OUT_FILE = Path("data/motivation-strengths.json")

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
IMPROVE_MIN = 0.0015
MIN_GAMES_PLAYED = 8  # before this, the table is too noisy/short to mean anything

# relegation: teams counted as "still fighting to survive" (a straight
# bottom-N drop is folded together with any playoff-decided spot, e.g.
# Bundesliga's 16th-place relegation playoff counts as if inside the zone).
# top: teams counted as "still fighting for something continental" (Europe)
# or, for the Championship, promotion (automatic + play-off spots).
# Verified via web search against 2025-26-season sources (uefa.com European
# Performance Spot pages, premierleague.com/bundesliga.com/beIN Sports/
# groundhopperguides.com season-qualification write-ups, Wikipedia season
# pages) rather than assumed - but simplified to one flat count per league:
# real slot counts fluctuate slightly season to season (e.g. UEFA grants a
# bonus 5th Champions League slot each cycle to whichever two countries had
# the best aggregate coefficient performance that season - England and Spain
# for 2025-26/26-27, not a fixed structural entitlement), and England's
# Conference League slot is nominally cup-route (League Cup winner) rather
# than table-position - both are folded into the flat "top" count below
# since the effect on a continuous stakes SIGNAL (not a hard cutoff) is
# minor at the margin.
LEAGUE_ZONES = {
    "Premier League": {"relegation": 3, "top": 7},    # 4 CL + up to 1 bonus CL + 1 EL + 1 ECL
    "Championship": {"relegation": 3, "top": 6},       # 2 automatic + 4 play-off promotion spots
    "LaLiga": {"relegation": 3, "top": 7},             # 4 CL + up to 1 bonus CL + 1 EL + 1 ECL
    "Bundesliga": {"relegation": 3, "top": 7},         # 4 CL + 2 EL + 1 ECL; 16th relegation play-off folded in
    "Serie A": {"relegation": 3, "top": 7},            # 4 CL + 2 EL + 1 ECL
    "Ligue 1": {"relegation": 3, "top": 6},            # 4 CL + 1 EL + 1 ECL; 16th relegation play-off folded in
    "Eredivisie": {"relegation": 3, "top": 5},         # 3 CL (incl. qualifying) + 1 EL + 1 ECL; 16th play-off folded in
    "Turkish Süper Lig": {"relegation": 3, "top": 4},  # 2 CL (incl. qualifying) + 1 EL + 1 ECL
    "Primeira Liga": {"relegation": 3, "top": 4},      # 2 CL (incl. qualifying) + 1 EL + 1 ECL; 16th play-off folded in
}


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


def stakes_by_match(season_matches, zones):
    """season_matches sorted by date -> {(date_iso, home, away): (stakes_home, stakes_away)}.
    stakes_* in [0, 1]: 0 = still alive for relegation or the top zone (or
    the table isn't meaningful yet), 1 = mathematically/practically settled
    into a mid-table position with nothing left to gain or lose."""
    matches = sorted(season_matches, key=lambda r: r["date"])
    total_games = {}
    for m in matches:
        total_games[m["home"]] = total_games.get(m["home"], 0) + 1
        total_games[m["away"]] = total_games.get(m["away"], 0) + 1

    table = {}  # team -> [points, gf, ga, played]

    def entry(team):
        return table.setdefault(team, [0, 0, 0, 0])

    def standings():
        # sorted by points desc, then goal difference, then goals for - a
        # standard (if not perfectly official) tie-break, fine for a
        # continuous signal that isn't shown to users as an exact position
        rows = [(team, pts, gf - ga, gf) for team, (pts, gf, ga, played) in table.items() if played]
        rows.sort(key=lambda r: (-r[1], -r[2], -r[3]))
        return [(team, pts) for team, pts, _gd, _gf in rows]

    out = {}
    by_date = {}
    for m in matches:
        by_date.setdefault(m["date_iso"], []).append(m)

    for date_iso in sorted(by_date):
        table_snapshot = standings()  # as of BEFORE today's matches - leak-free
        n = len(table_snapshot)
        points_at = {pos: pts for pos, (_team, pts) in enumerate(table_snapshot, start=1)}
        pos_of = {team: pos for pos, (team, _pts) in enumerate(table_snapshot, start=1)}

        for m in by_date[date_iso]:
            def team_stakes(team):
                played = entry(team)[3]
                if played < MIN_GAMES_PLAYED or team not in pos_of or n < zones["relegation"] + zones["top"] + 1:
                    return 0.0
                pos = pos_of[team]
                releg_boundary = n - zones["relegation"]
                if pos > releg_boundary or pos <= zones["top"]:
                    return 0.0  # already IN a zone - definitely still motivated
                pts = table[team][0]
                gap_releg = pts - points_at.get(releg_boundary, pts)
                gap_top = points_at.get(zones["top"], pts) - pts
                games_left = total_games.get(team, 0) - played
                yardstick = max(1, games_left * 3)
                margin = min(gap_releg, gap_top)
                return max(0.0, min(1.0, margin / yardstick))

            out[(m["date_iso"], m["home"], m["away"])] = (team_stakes(m["home"]), team_stakes(m["away"]))

        # now apply today's results to the live table
        for m in by_date[date_iso]:
            h, a = entry(m["home"]), entry(m["away"])
            h[1] += m["hg"]; h[2] += m["ag"]; h[3] += 1
            a[1] += m["ag"]; a[2] += m["hg"]; a[3] += 1
            if m["hg"] > m["ag"]:
                h[0] += 3
            elif m["hg"] < m["ag"]:
                a[0] += 3
            else:
                h[0] += 1; a[0] += 1
    return out


def predictions_for_target(league, by_code, target):
    ti = ALL_SEASONS.index(target)
    priors = ALL_SEASONS[max(0, ti - 4):ti][::-1]
    if len(priors) < 2:
        return None
    seasons = [(by_code.get(p, []), w) for p, w in zip(priors, PRIOR_WEIGHTS)]
    model = LeagueModel(seasons)
    target_matches = by_code.get(target, [])
    if not target_matches:
        return None
    zones = LEAGUE_ZONES.get(league)
    if not zones:
        return None
    stakes_lookup = stakes_by_match(target_matches, zones)

    out = []
    for m in target_matches:
        pred = model.predict(m["home"], m["away"])
        sh, sa = stakes_lookup[(m["date_iso"], m["home"], m["away"])]
        out.append({"lam_h": pred["lam_home"], "lam_a": pred["lam_away"], "rho": pred["rho"],
                     "stakes_h": sh, "stakes_a": sa, "total": m["total"]})
    return out


def brier_for_strength(cached_predictions, strength):
    briers = {line: [] for line in LINES}
    for c in cached_predictions:
        lam_h, lam_a = c["lam_h"], c["lam_a"]
        if strength:
            lam_h *= (1 - strength * c["stakes_h"])
            lam_a *= (1 - strength * c["stakes_a"])
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
