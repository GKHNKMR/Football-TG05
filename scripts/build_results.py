"""Score BETAVUS's own predictions against real results -> data/results.json.

Two sources of graded predictions, same model, same grading:

  * Archived   - the real pre-kickoff call. Each run merges predictions.json into
    data/predictions-archive.json, keeping the FIRST prediction seen per
    match_id, and grades any whose kickoff is now past.
  * Reconstructed - for recently-played matches not in the archive yet (e.g. the
    first weeks after launch), the model is re-run with a per-match cutoff so no
    result leaks in. Marked "reconstructed": true.

data/results.json holds the last RESULT_DAYS of graded predictions plus hit-rate
aggregates; the "Sonuçlar" tab reads it. Archive is pruned to ARCHIVE_DAYS.
"""

import csv
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teams import DIV_BY_LEAGUE, to_fd  # noqa: E402
from update_predictions import (  # noqa: E402
    LEAGUES, SEASONS, LeagueModel, clean_name, ft_goals, load_season,
)

PRED_FILE = Path("predictions.json")
ARCHIVE_FILE = Path("data/predictions-archive.json")
CSV_DIR = Path("data/football-data")
OUT_FILE = Path("data/results.json")

CURRENT_SEASON = "2627"
ARCHIVE_DAYS = 80
RESULT_DAYS = 21
DATE_SLACK = 2
LINES = [(0.5, "p_over_0_5"), (1.5, "p_over_1_5"), (2.5, "p_over_2_5")]


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
    """(league, fd_home, fd_away, date) -> {score, total, odds}, current season."""
    out = {}
    for league, div in DIV_BY_LEAGUE.items():
        path = CSV_DIR / div / f"{CURRENT_SEASON}.csv"
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
        hits[f"hit_{str(line).replace('.', '')}"] = int((p >= 0.5) == over)
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
    for line, _ in LINES:
        k = f"hit_{str(line).replace('.', '')}"
        agg[f"acc_{str(line).replace('.', '')}"] = round(
            100 * sum(r["hits"][k] for r in rows) / n, 1)
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

def reconstruct(actuals, already, cutoff, today):
    """Model's leak-free call for recently-played matches not in the archive."""
    rows = []
    for lid, (stem, name, code, _tz) in LEAGUES.items():
        seasons_raw = [(load_season(stem, s), w) for s, w in SEASONS]
        n = 0
        for m in seasons_raw[0][0]:
            g = ft_goals(m)
            if not g or not m.get("date"):
                continue
            try:
                d = date.fromisoformat(m["date"])
            except ValueError:
                continue
            if not (cutoff <= d < today):
                continue
            home, away = clean_name(m["team1"]), clean_name(m["team2"])
            if (name, home, away, d) in already:
                continue
            actual = find_actual(actuals, name, to_fd(name, home), to_fd(name, away), d)
            if not actual:
                continue
            model = LeagueModel(
                [([x for x in ms if x.get("date", "") < m["date"]], w)
                 for ms, w in seasons_raw])
            pred = model.predict(m["team1"], m["team2"])
            n += 1
            row = {
                "match_id": f"{code}-{d.isoformat()}-R{n:02d}",
                "league": name, "kickoff_utc": f"{d.isoformat()}T12:00:00Z",
                "home": home, "away": away,
                "pred_lambda": pred["exp_goals"], "basis": pred.get("basis"),
                "reconstructed": True,
                "p_over_0_5": pred["p_over_0_5"],
                "p_over_1_5": pred["p_over_1_5"],
                "p_over_2_5": pred["p_over_2_5"],
                "market": None,
            }
            rows.append(grade(row, actual["total"], actual["score"],
                              actual["o25_odds"], actual["u25_odds"]))
    return rows


def main():
    archive = merge_predictions(load_archive())
    ARCHIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE_FILE.write_text(
        json.dumps(archive, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"archive: {len(archive)} predictions")

    actuals = load_actuals()
    print(f"actuals (current season): {len(actuals)}")

    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=RESULT_DAYS)).date()
    graded = []
    for a in archive.values():
        ko = datetime.fromisoformat(a["kickoff_utc"].replace("Z", "+00:00"))
        if ko >= now or ko.date() < cutoff:
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
    recon = reconstruct(actuals, already, cutoff, now.date())
    graded.extend(recon)
    graded.sort(key=lambda r: r["kickoff_utc"], reverse=True)

    week_cut = (now - timedelta(days=7)).isoformat()
    payload = {
        "generated_at": now.isoformat(),
        "result_days": RESULT_DAYS,
        "reconstructed_count": len(recon),
        "overall": aggregate(graded),
        "last_week": aggregate([r for r in graded if r["kickoff_utc"] >= week_cut]),
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
