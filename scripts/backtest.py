"""Walk-forward backtest of the BETAVUS goal model.

For each target season we build the model from the FOUR seasons before it only
(nearest prior weighted 1.0, then 0.7 / 0.45 / 0.30 - the same recency weights the
live model uses), predict every match as if the fixture list had just been
released, then score those predictions against what actually happened.

Nothing from the target season feeds the model - no result leakage.

Model math is identical to scripts/update_predictions.py (LeagueModel):
    lam = (home_gf + away_ga)/2 + (away_gf + home_ga)/2
    if >=2 H2H meetings:  lam = 0.72*lam + 0.28*(avg H2H total goals)
    P(over n) from Poisson(lam)

Historical data: the football-data.co.uk CSVs already mirrored under
data/football-data/ (exact scores, eight seasons, all six leagues).

Output: data/backtest.json  ->  rendered by backtest.html (standalone screen).
"""

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

CSV_DIR = Path("data/football-data")
OUT_FILE = Path("data/backtest.json")

DIVISIONS = {
    "E0": "Premier League", "SP1": "LaLiga", "D1": "Bundesliga",
    "I1": "Serie A", "F1": "Ligue 1", "N1": "Eredivisie",
}
ALL_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]
TARGET_SEASONS = ["2324", "2425", "2526"]
PRIOR_WEIGHTS = [1.0, 0.7, 0.45, 0.30]  # nearest prior season first
H2H_MAX = 8
LINES = [0.5, 1.5, 2.5]
EPS = 1e-9


def season_label(code):
    return f"20{code[:2]}/{code[2:]}"


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
    rows.sort(key=lambda m: m["date"])
    return rows


class LeagueModel:
    """Weighted home/away goal rates + H2H record - mirrors update_predictions.py."""

    def __init__(self, seasons):  # seasons: list of (matches, weight)
        self.hgf, self.hga, self.agf, self.aga = {}, {}, {}, {}
        self.h2h = {}
        hs = [0.0, 0.0]
        as_ = [0.0, 0.0]
        for matches, w in seasons:
            for m in matches:
                h, a, hg, ag = m["home"], m["away"], m["hg"], m["ag"]
                self._add(self.hgf, h, hg, w)
                self._add(self.hga, h, ag, w)
                self._add(self.agf, a, ag, w)
                self._add(self.aga, a, hg, w)
                hs[0] += hg * w
                hs[1] += w
                as_[0] += ag * w
                as_[1] += w
                self.h2h.setdefault(frozenset((h, a)), []).append((m["date"], hg + ag))
        self.base_home = hs[0] / hs[1] if hs[1] else 1.5
        self.base_away = as_[0] / as_[1] if as_[1] else 1.1

    @staticmethod
    def _add(store, key, value, weight):
        e = store.setdefault(key, [0.0, 0.0])
        e[0] += value * weight
        e[1] += weight

    @staticmethod
    def _avg(store, key, fallback):
        e = store.get(key)
        return e[0] / e[1] if e and e[1] else fallback

    def predict(self, home, away):
        hgf = self._avg(self.hgf, home, self.base_home)
        hga = self._avg(self.hga, home, self.base_away)
        agf = self._avg(self.agf, away, self.base_away)
        aga = self._avg(self.aga, away, self.base_home)
        lam = (hgf + aga) / 2 + (agf + hga) / 2
        pair = sorted(self.h2h.get(frozenset((home, away)), []), reverse=True)[:H2H_MAX]
        if len(pair) >= 2:
            lam = 0.72 * lam + 0.28 * (sum(tg for _, tg in pair) / len(pair))
        return max(0.30, min(6.0, lam))


def poisson_over(lam, n):
    term = math.exp(-lam)
    cdf = term
    for k in range(1, n + 1):
        term *= lam / k
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


def calibration(pairs, bins=10):
    """pairs: list of (p, outcome 0/1) -> per-bin predicted vs actual, plus ECE."""
    buckets = [[] for _ in range(bins)]
    for p, o in pairs:
        idx = min(bins - 1, int(p * bins))
        buckets[idx].append((p, o))
    rows = []
    ece = 0.0
    n = len(pairs) or 1
    for i, b in enumerate(buckets):
        if not b:
            rows.append({"lo": round(i / bins, 2), "hi": round((i + 1) / bins, 2),
                         "n": 0, "pred": None, "actual": None})
            continue
        mp = sum(x[0] for x in b) / len(b)
        ar = sum(x[1] for x in b) / len(b)
        ece += len(b) / n * abs(mp - ar)
        rows.append({"lo": round(i / bins, 2), "hi": round((i + 1) / bins, 2),
                     "n": len(b), "pred": round(mp, 4), "actual": round(ar, 4)})
    return rows, ece


def score(records):
    """records: list of dicts with p05/p15/p25, lam, total."""
    n = len(records)
    out = {"n_matches": n, "markets": {}}

    # expected-goals accuracy
    errs = [r["lam"] - r["total"] for r in records]
    abse = [abs(e) for e in errs]
    mp = sum(r["lam"] for r in records) / n
    ma = sum(r["total"] for r in records) / n
    sp = sum((r["lam"] - mp) ** 2 for r in records)
    sa = sum((r["total"] - ma) ** 2 for r in records)
    cov = sum((r["lam"] - mp) * (r["total"] - ma) for r in records)
    out["lambda"] = {
        "mae": round(sum(abse) / n, 3),
        "rmse": round((sum(e * e for e in errs) / n) ** 0.5, 3),
        "bias": round(sum(errs) / n, 3),
        "within_1": round(sum(e <= 1.0 for e in abse) / n, 4),
        "within_1_5": round(sum(e <= 1.5 for e in abse) / n, 4),
        "mean_pred": round(mp, 3),
        "mean_actual": round(ma, 3),
        "corr": round(cov / math.sqrt(sp * sa), 3) if sp > 0 and sa > 0 else None,
    }

    for line in LINES:
        key = str(line)
        pk = {0.5: "p05", 1.5: "p15", 2.5: "p25"}[line]
        pairs = [(r[pk], 1 if r["total"] > line else 0) for r in records]
        base = sum(o for _, o in pairs) / n  # in-sample base rate of the outcome
        brier = sum((p - o) ** 2 for p, o in pairs) / n
        brier_base = sum((base - o) ** 2 for _, o in pairs) / n
        logloss = -sum(
            o * math.log(min(max(p, EPS), 1 - EPS)) + (1 - o) * math.log(min(max(1 - p, EPS), 1 - EPS))
            for p, o in pairs
        ) / n
        calib, ece = calibration(pairs)
        # "would you have been right" - the side the model leans to, vs actual.
        # Compare on the rounded percentage so a shown "50%" counts as an Over lean
        # (otherwise a p of 0.499 shows 50% but scores as a miss on an Over match).
        lean = lambda p: round(p * 100) >= 50
        pick_hits = sum(lean(p) == bool(o) for p, o in pairs)
        conf = [(p, o) for p, o in pairs if p >= 0.65 or p <= 0.35]
        conf_hits = sum(lean(p) == bool(o) for p, o in conf)
        out["markets"][key] = {
            "base_rate": round(base, 4),
            "brier": round(brier, 4),
            "brier_baseline": round(brier_base, 4),
            "skill": round(1 - brier / brier_base, 4) if brier_base else None,
            "logloss": round(logloss, 4),
            "ece": round(ece, 4),
            "calibrated_pct": round(100 * (1 - ece), 1),
            "pick_acc": round(pick_hits / n, 4),
            "conf_n": len(conf),
            "conf_acc": round(conf_hits / len(conf), 4) if conf else None,
            "calibration": calib,
        }
    return out


def main():
    divisions = {d: load_division(d) for d in DIVISIONS}
    for d, rows in divisions.items():
        print(f"{DIVISIONS[d]:15} {len(rows)} matches "
              f"({rows[0]['season']}..{rows[-1]['season']})")

    all_records = []
    by_league = {v: [] for v in DIVISIONS.values()}
    by_season = {s: [] for s in TARGET_SEASONS}
    samples = []

    for div, league in DIVISIONS.items():
        rows = divisions[div]
        by_code = {}
        for m in rows:
            by_code.setdefault(m["season"], []).append(m)
        for target in TARGET_SEASONS:
            ti = ALL_SEASONS.index(target)
            priors = ALL_SEASONS[max(0, ti - 4):ti][::-1]  # nearest first
            if len(priors) < 2:
                continue
            model = LeagueModel([(by_code.get(p, []), w)
                                 for p, w in zip(priors, PRIOR_WEIGHTS)])
            for m in by_code.get(target, []):
                lam = model.predict(m["home"], m["away"])
                rec = {
                    "league": league, "season": target, "date": m["date"],
                    "home": m["home"], "away": m["away"],
                    "lam": lam, "total": m["total"],
                    "score": f"{m['hg']}-{m['ag']}",
                    "p05": poisson_over(lam, 0),
                    "p15": poisson_over(lam, 1),
                    "p25": poisson_over(lam, 2),
                }
                all_records.append(rec)
                by_league[league].append(rec)
                by_season[target].append(rec)

    all_records.sort(key=lambda r: r["date"])
    step = max(1, len(all_records) // 40)
    for r in all_records[::step][:40]:
        samples.append({
            "season": season_label(r["season"]), "league": r["league"],
            "date": r["date"], "home": r["home"], "away": r["away"],
            "pred_lambda": round(r["lam"], 2),
            "p25": round(r["p25"], 3),
            "actual_total": r["total"], "actual_score": r["score"],
            "over25_hit": (round(r["p25"] * 100) >= 50) == (r["total"] > 2.5),
            "lambda_err": round(abs(r["lam"] - r["total"]), 2),
        })

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "data": "football-data.co.uk (exact results, 6 leagues)",
            "model": "BETAVUS goal model (LeagueModel) - identical to the live predictor",
            "target_seasons": [season_label(s) for s in TARGET_SEASONS],
            "training": "the 4 seasons before each target, weights 1.0/0.7/0.45/0.30, "
                        "no target-season data",
        },
        "overall": score(all_records),
        "by_league": {k: score(v) for k, v in by_league.items() if v},
        "by_season": {season_label(k): score(v) for k, v in by_season.items() if v},
        "samples": samples,
    }
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")
    o = payload["overall"]
    print(f"\nBacktested {o['n_matches']} matches across {len(TARGET_SEASONS)} seasons")
    print(f"  expected goals  MAE {o['lambda']['mae']}  bias {o['lambda']['bias']}  "
          f"within +/-1 goal {o['lambda']['within_1']*100:.1f}%")
    for line in LINES:
        mk = o["markets"][str(line)]
        print(f"  Over {line}:  calibrated {mk['calibrated_pct']}%  "
              f"Brier {mk['brier']} (base {mk['brier_baseline']}, skill {mk['skill']})  "
              f"pick-acc {mk['pick_acc']*100:.1f}%")
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
