"""Score BETAVUS's own predictions against real results -> data/results.json.

Two sources of graded predictions, same grading:

  * Archived   - the real pre-kickoff call. Each run merges predictions.json into
    data/predictions-archive.json, keeping the FIRST prediction seen per
    match_id, and grades any whose kickoff is now past.
  * Reconstructed - for every completed match inside the window that is not in
    the archive, the model is re-run walk-forward with a per-match cutoff so no
    result leaks in (same LeagueModel + recency weights as the live model and
    the backtest). Marked "reconstructed": true.

The window spans the whole previous season plus the current one, so the
"Sonuçlar" tab reports a full-season track record, not just the last few weeks.
Historical scores come from the football-data.co.uk CSVs (complete, exact) - the
same source scripts/backtest.py uses - not openfootball, whose past-season files
in this mirror are only partially filled.
data/results.json holds every graded prediction plus hit-rate aggregates.
Archive is pruned to ARCHIVE_DAYS.
"""

import csv
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teams import DIVISIONS, DIV_BY_LEAGUE, to_fd, to_pretty  # noqa: E402
from backtest import LeagueModel, poisson_over  # noqa: E402

PRED_FILE = Path("predictions.json")
ARCHIVE_FILE = Path("data/predictions-archive.json")
CSV_DIR = Path("data/football-data")
OUT_FILE = Path("data/results.json")

ARCHIVE_DAYS = 90
DATE_SLACK = 2

# graded window: the previous full season plus the current one
WINDOW_START = date(2025, 7, 1)
WINDOW_LABEL = "2025/26 sezonu + bu sezon"
# football-data.co.uk season codes, oldest -> newest. The last two are the ones
# we reconstruct graded results for; the earlier ones only feed the model.
FD_SEASONS = ["2223", "2324", "2425", "2526", "2627"]
RECON_TARGETS = ["2526", "2627"]
# season being reconstructed gets weight 1.0; the three before it 0.7/0.45/0.30
RECON_WEIGHTS = [1.0, 0.7, 0.45, 0.30]
LINES = [(0.5, "p_over_0_5"), (1.5, "p_over_1_5"), (2.5, "p_over_2_5")]
# "high confidence" thresholds = the Vurgu levels highlighted on the site; the
# Sonuçlar tab reports how the picks above these did (the risk-reduced view).
HI_MIN = {"05": 0.95, "15": 0.85, "25": 0.75}
LINES = [(0.5, "p_over_0_5"), (1.5, "p_over_1_5"), (2.5, "p_over_2_5")]
# "high confidence" thresholds = the Vurgu levels highlighted on the site; the
# Sonuçlar tab reports how the picks above these did (the risk-reduced view).
HI_MIN = {"05": 0.95, "15": 0.85, "25": 0.75}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def implied_o25(o, u):
    if not o or not u:
        return None
    io, iu = 1 / o, 1 / u
    return io / (io + iu)


# ---------------------------------------------------------------- archive ------

def load_archive():
    if ARCHIVE_FILE.exists():
        try:
            return json.loads(ARCHIVE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def merge_predictions(archive):
    preds = json.loads(PRED_FILE.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    for p in preds:
        mid = p["match_id"]
        if mid not in archive:
            archive[mid] = {
                "match_id": mid, "league": p["league"], "kickoff_utc": p["kickoff_utc"],
                "home": p["home"], "away": p["away"],
                "pred_lambda": p.get("exp_goals"), "basis": p.get("basis"),
                "p_over_0_5": p["p_over_0_5"], "p_over_1_5": p["p_over_1_5"],
                "p_over_2_5": p["p_over_2_5"], "market": p.get("market"),
                "first_seen": now,
            }
        elif p.get("market") and not archive[mid].get("market"):
            archive[mid]["market"] = p["market"]
    cutoff = (datetime.now(timezone.utc) - timedelta(days=ARCHIVE_DAYS)).isoformat()
    return {k: v for k, v in archive.items() if v["kickoff_utc"] >= cutoff}


# ---------------------------------------------------------------- actuals ------

def load_actuals():
    """(league, fd_home, fd_away, date) -> {score, total, odds} over the window."""
    out = {}
    for league, div in DIV_BY_LEAGUE.items():
        for season in RECON_TARGETS:
            path = CSV_DIR / div / f"{season}.csv"
            if not path.exists():
                continue
            with path.open(encoding="utf-8-sig") as fh:
                for r in csv.DictReader(fh):
                    try:
                        hg, ag = int(r["FTHG"]), int(r["FTAG"])
                        d = datetime.strptime(r["Date"].strip(), "%d/%m/%Y").date()
                    except (KeyError, ValueError):
                        continue
                    out[(league, r["HomeTeam"].strip(), r["AwayTeam"].strip(), d)] = {
                        "score": f"{hg}-{ag}", "total": hg + ag,
                        "o25_odds": _f(r.get("Avg>2.5")), "u25_odds": _f(r.get("Avg<2.5")),
                    }
    return out


def find_actual(actuals, league, fd_home, fd_away, d):
    for delta in range(-DATE_SLACK, DATE_SLACK + 1):
        hit = actuals.get((league, fd_home, fd_away, d + timedelta(days=delta)))
        if hit:
            return hit
    return None


# ---------------------------------------------------------------- grading -----

def grade(row, total, score, o25_odds, u25_odds):
    g = dict(row)
    g.update({"score": score, "total": total, "o25_odds": o25_odds})
    hits, truth = {}, []
    for line, key in LINES:
        p = row[key]
        over = total > line
        # round the call the same way the % is shown, so "50%" never contradicts
        # the ✓/✗ next to it
        hits[f"hit_{str(line).replace('.', '')}"] = int((round(p, 2) >= 0.5) == over)
        truth.append(p if over else 1 - p)
    g["hits"] = hits
    g["success_pct"] = round(sum(truth) / len(truth) * 100, 1)
    g["lambda_err"] = (round(abs(row["pred_lambda"] - total), 2)
                       if row.get("pred_lambda") is not None else None)
    mk = row.get("market") or {}
    imp = mk.get("o25_implied")
    if imp is None:
        imp = implied_o25(o25_odds, u25_odds)
    if imp is not None:
        g["mkt_o25_implied"] = round(imp, 4)
        g["edge25"] = round(row["p_over_2_5"] - imp, 4)
        g["value_hit"] = int((row["p_over_2_5"] >= imp) == (total > 2.5))
    return g


def aggregate(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    agg = {"n": n}
    for line, key in LINES:
        tag = str(line).replace('.', '')
        hk = f"hit_{tag}"
        agg[f"acc_{tag}"] = round(100 * sum(r["hits"][hk] for r in rows) / n, 1)
        # same, but only the high-confidence picks for this line
        hi = [r for r in rows if r.get(key, 0) >= HI_MIN[tag]]
        if hi:
            agg[f"hi_n_{tag}"] = len(hi)
            agg[f"hi_acc_{tag}"] = round(
                100 * sum(r["hits"][hk] for r in hi) / len(hi), 1)
    agg["success_pct"] = round(sum(r["success_pct"] for r in rows) / n, 1)
    val = [r for r in rows if "value_hit" in r]
    if val:
        agg["value_acc"] = round(100 * sum(r["value_hit"] for r in val) / len(val), 1)
        agg["value_n"] = len(val)
    errs = [r["lambda_err"] for r in rows if r.get("lambda_err") is not None]
    if errs:
        agg["lambda_mae"] = round(sum(errs) / len(errs), 2)
    return agg


# ------------------------------------------------------------ reconstruction --

def load_division(div):
    """Completed matches for one division across FD_SEASONS, oldest first."""
    rows = []
    for season in FD_SEASONS:
        path = CSV_DIR / div / f"{season}.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                try:
                    hg, ag = int(r["FTHG"]), int(r["FTAG"])
                    d = datetime.strptime(r["Date"].strip(), "%d/%m/%Y").date()
                except (KeyError, ValueError):
                    continue
                rows.append({
                    "season": season, "date": d.isoformat(),
                    "home": r["HomeTeam"].strip(), "away": r["AwayTeam"].strip(),
                    "hg": hg, "ag": ag, "total": hg + ag,
                })
    rows.sort(key=lambda m: m["date"])
    return rows


def reconstruct(actuals, already, start, today):
    """Walk-forward, leak-free model call for every completed match in the window
    that the archive does not already hold. Same LeagueModel + recency weights as
    the live predictor; every training match is strictly older than the one being
    predicted, so nothing leaks in."""
    rows = []
    for div, (league, _lid) in DIVISIONS.items():
        by_code = {}
        for m in load_division(div):
            by_code.setdefault(m["season"], []).append(m)
        n = 0
        for target in RECON_TARGETS:
            ti = FD_SEASONS.index(target)
            plan_codes = FD_SEASONS[max(0, ti - 3):ti + 1][::-1]  # target first
            plan = list(zip(plan_codes, RECON_WEIGHTS))
            for m in by_code.get(target, []):
                try:
                    d = date.fromisoformat(m["date"])
                except ValueError:
                    continue
                if not (start <= d < today):
                    continue
                home = to_pretty(league, m["home"])
                away = to_pretty(league, m["away"])
                if (league, home, away, d) in already:
                    continue
                model = LeagueModel([
                    ([x for x in by_code.get(code, []) if x["date"] < m["date"]], w)
                    for code, w in plan
                ])
                lam = model.predict(m["home"], m["away"])
                fd = find_actual(actuals, league, m["home"], m["away"], d) or {}
                n += 1
                row = {
                    "match_id": f"{div}-{d.isoformat()}-R{n:03d}",
                    "league": league, "kickoff_utc": f"{d.isoformat()}T12:00:00Z",
                    "home": home, "away": away,
                    "pred_lambda": round(lam, 3), "basis": None,
                    "reconstructed": True,
                    "p_over_0_5": round(poisson_over(lam, 0), 4),
                    "p_over_1_5": round(poisson_over(lam, 1), 4),
                    "p_over_2_5": round(poisson_over(lam, 2), 4),
                    "market": None,
                }
                rows.append(grade(row, m["total"], f"{m['hg']}-{m['ag']}",
                                  fd.get("o25_odds"), fd.get("u25_odds")))
    return rows


def main():
    archive = merge_predictions(load_archive())
    ARCHIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE_FILE.write_text(
        json.dumps(archive, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"archive: {len(archive)} predictions")

    actuals = load_actuals()
    print(f"actuals ({'+'.join(RECON_TARGETS)}): {len(actuals)}")

    now = datetime.now(timezone.utc)
    graded = []
    for a in archive.values():
        ko = datetime.fromisoformat(a["kickoff_utc"].replace("Z", "+00:00"))
        if ko >= now or ko.date() < WINDOW_START:
            continue
        actual = find_actual(actuals, a["league"],
                             to_fd(a["league"], a["home"]),
                             to_fd(a["league"], a["away"]), ko.date())
        if actual:
            graded.append(grade(a, actual["total"], actual["score"],
                                actual["o25_odds"], actual["u25_odds"]))

    already = {(r["league"], r["home"], r["away"],
               datetime.fromisoformat(r["kickoff_utc"].replace("Z", "+00:00")).date())
              for r in graded}
    recon = reconstruct(actuals, already, WINDOW_START, now.date())
    graded.extend(recon)
    graded.sort(key=lambda r: r["kickoff_utc"], reverse=True)

    payload = {
        "generated_at": now.isoformat(),
        "span": WINDOW_LABEL,
        "window_start": WINDOW_START.isoformat(),
        "reconstructed_count": len(recon),
        "overall": aggregate(graded),
        "matches": graded,
    }
    OUT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    o = payload["overall"]
    print(f"graded {len(graded)} ({len(recon)} reconstructed) - "
          f"O2.5 acc {o.get('acc_25')}%  success {o.get('success_pct')}%  "
          f"lambda MAE {o.get('lambda_mae')}")


if __name__ == "__main__":
    main()
