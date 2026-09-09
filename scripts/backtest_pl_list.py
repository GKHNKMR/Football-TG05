"""Per-match backtest list for the Premier League 2025/26.

Model is built from the FOUR seasons before 2025/26 only (2021/22-2024/25,
weights 1.0/0.7/0.45/0.30) - no 2025/26 result touches it. Every 2025/26 fixture
is then predicted and lined up against what actually happened.

    python scripts/backtest_pl_list.py [line]      # line = 0.5 | 1.5 | 2.5 (default 2.5)

Writes data/backtest-pl-2526.csv (all three lines) and prints a table + season
summary focused on the chosen line.

Per-line "success %" = the probability the model placed on the ACTUAL outcome of
that line (p if the line went Over, else 1 - p). success_avg blends the three.
"""

import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import (  # noqa: E402  reuse the exact model + loader
    ALL_SEASONS, PRIOR_WEIGHTS, LeagueModel, load_division, poisson_over,
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DIV = "E0"
TARGET = "2526"
OUT = Path("data/backtest-pl-2526.csv")
LINE = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5
NBIN = {0.5: 0, 1.5: 1, 2.5: 2}[LINE]
PKEY = {0.5: "p_o05", 1.5: "p_o15", 2.5: "p_o25"}[LINE]
DKEY = {0.5: "o05_dir", 1.5: "o15_dir", 2.5: "o25_dir"}[LINE]
SKEY = {0.5: "success_o05", 1.5: "success_o15", 2.5: "success_o25"}[LINE]


def main():
    rows = load_division(DIV)
    by_code = {}
    for m in rows:
        by_code.setdefault(m["season"], []).append(m)

    ti = ALL_SEASONS.index(TARGET)
    priors = ALL_SEASONS[max(0, ti - 4):ti][::-1]
    model = LeagueModel([(by_code.get(p, []), w) for p, w in zip(priors, PRIOR_WEIGHTS)])
    train = ", ".join(f"20{p[:2]}/{p[2:]} ({w})" for p, w in zip(priors, PRIOR_WEIGHTS))
    print(f"Training seasons (weight): {train}")
    print(f"Target: Premier League 20{TARGET[:2]}/{TARGET[2:]}  -  "
          f"{len(by_code.get(TARGET, []))} matches   |   featured line: {LINE} Uest\n")

    out_rows = []
    for m in by_code.get(TARGET, []):
        lam = model.predict(m["home"], m["away"])
        p = {n: poisson_over(lam, n) for n in (0, 1, 2)}
        tot = m["total"]
        outc = {n: tot > (n + 0.5) for n in (0, 1, 2)}
        s = {n: (p[n] if outc[n] else 1 - p[n]) for n in (0, 1, 2)}
        out_rows.append({
            "date": m["date"], "home": m["home"], "away": m["away"],
            "pred_goals": round(lam, 2),
            "p_o05": round(p[0] * 100, 1),
            "p_o15": round(p[1] * 100, 1),
            "p_o25": round(p[2] * 100, 1),
            "score": f"{m['hg']}-{m['ag']}", "total": tot,
            "o05_dir": int((p[0] >= 0.5) == outc[0]),
            "o15_dir": int((p[1] >= 0.5) == outc[1]),
            "o25_dir": int((p[2] >= 0.5) == outc[2]),
            "success_o05": round(s[0] * 100, 1),
            "success_o15": round(s[1] * 100, 1),
            "success_o25": round(s[2] * 100, 1),
            "success_avg": round(sum(s.values()) / 3 * 100, 1),
            "goal_err": round(abs(lam - tot), 2),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    lname = f"{LINE}U"
    print(f"{'Tarih':10} {'Ev - Deplasman':32} {'Tahmin gol':>10} {lname+'%':>7}  "
          f"{'Skor':>6} {'Gol':>4} {'>'+str(LINE):>6}  {lname+' dogru':>9}  {'Basari%':>8}")
    print("-" * 96)
    for r in out_rows:
        mt = f"{r['home']} - {r['away']}"
        went = "Ust" if r["total"] > LINE else "Alt"
        print(f"{r['date']:10} {mt[:32]:32} {r['pred_goals']:>10} {r[PKEY]:>7}  "
              f"{r['score']:>6} {r['total']:>4} {went:>6}  "
              f"{('OK' if r[DKEY] else 'X'):>9}  {r[SKEY]:>7}%")

    n = len(out_rows)
    over = sum(r["total"] > LINE for r in out_rows)
    succ = sum(r[SKEY] for r in out_rows) / n
    d = sum(r[DKEY] for r in out_rows) / n
    mp = sum(r["pred_goals"] for r in out_rows) / n
    ma = sum(r["total"] for r in out_rows) / n
    mean_p = sum(r[PKEY] for r in out_rows) / n
    misses = [r for r in out_rows if not r[DKEY]]
    print("-" * 96)
    print(f"\nSEZON OZETI - Premier League 20{TARGET[:2]}/{TARGET[2:]} - {LINE} Uest - {n} mac")
    print(f"  {LINE} Ust gerceklesme                : {over}/{n}  ({over/n*100:.1f}%)")
    print(f"  Ortalama model olasiligi ({LINE}U)    : {mean_p:.1f}%   (gercek: {over/n*100:.1f}%  -> kalibrasyon farki {abs(mean_p-over/n*100):.1f} puan)")
    print(f"  Ortalama basari (dogruya konan olas.) : {succ:.1f}%")
    print(f"  Yon isabeti ({LINE}U Ust/Alt dogru)   : {d*100:.1f}%   ({n-len(misses)}/{n})")
    print(f"  Tahmini gol ort. / gercek gol ort.    : {mp:.2f} / {ma:.2f}")
    if misses:
        print(f"\n  {LINE}U yon hatasi olan {len(misses)} mac:")
        for r in misses:
            print(f"    {r['date']}  {r['home']} - {r['away']:22}  tahmin {r[PKEY]:>5}%  ->  {r['score']} ({r['total']} gol)")
    print(f"\nCSV: {OUT}")


if __name__ == "__main__":
    main()
