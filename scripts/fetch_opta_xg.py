"""Refresh data/opta-xg.json - team-match xG for our 9 leagues, pulled fresh
from peteowen1/pannadata's public Opta mirror (see scripts/opta_xg.py for
where it comes from and why the team-name resolution works the way it does).

NOT part of the live pipeline. scripts/tune_xg_weight.py's walk-forward
backtest found blending xG into the goal model only helps by ~0.0002-0.0016
Brier points out of sample, league to league - noise-level, not worth the
added dependency (this script needs `pip install duckdb`) and moving part
for the live model. Kept as a standalone tool in case that changes (more
seasons of data, or a more targeted use - e.g. only for a newly-promoted
side with too little top-flight history of its own to trust its goals
average yet).
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opta_xg import DIV_TO_OPTA, fetch_raw_matches, resolve_team_names  # noqa: E402

CSV_DIR = Path("data/football-data")
OUT_FILE = Path("data/opta-xg.json")


def fd_team_names(div):
    names = set()
    for csvf in (CSV_DIR / div).glob("*.csv"):
        with csvf.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row.get("HomeTeam"):
                    names.add(row["HomeTeam"].strip())
                if row.get("AwayTeam"):
                    names.add(row["AwayTeam"].strip())
    return names


def main():
    fd_names_by_div = {d: fd_team_names(d) for d in DIV_TO_OPTA}
    raw = fetch_raw_matches()
    resolved, unmatched = resolve_team_names(raw, fd_names_by_div)
    resolved.sort(key=lambda m: m["date"])

    out = [{
        "div": m["div"], "season": m["season"], "date": m["date"],
        "home": m["home"], "away": m["away"],
        "home_xg": round(m["home_xg"], 3), "away_xg": round(m["away_xg"], 3),
    } for m in resolved]
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")

    print(f"Opta xG: {len(out)} matches resolved across {len(DIV_TO_OPTA)} leagues, "
          f"{len(unmatched)} rows unmatched (older seasons our football-data.co.uk "
          f"cache doesn't cover, or a genuinely new/renamed club - see below)")
    seen = set()
    for m in unmatched:
        key = (m["div"], m["home_opta"], m["away_opta"])
        if key in seen:
            continue
        seen.add(key)
    if seen and len(seen) <= 30:
        for div, h, a in sorted(seen):
            print(f"  unmatched [{div}] {h} vs {a}")


if __name__ == "__main__":
    main()
