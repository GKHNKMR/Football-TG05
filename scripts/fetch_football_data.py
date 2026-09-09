"""Download historical match CSVs from football-data.co.uk (Main Leagues).

Only the six leagues used by BETAVUS are pulled, top division only:

    E0  England  - Premier League
    SP1 Spain    - LaLiga
    D1  Germany  - Bundesliga
    I1  Italy    - Serie A
    F1  France   - Ligue 1
    N1  Netherlands - Eredivisie

Files land in data/football-data/<DIV>/<SEASON>.csv (SEASON = e.g. 2425).
These raw CSVs are the local stats database; scripts/build_match_stats.py
folds them into data/match-stats.json for the dashboard.

Also grabs football-data.co.uk/fixtures.csv -> data/football-data/fixtures.csv:
the next few days' matches across many leagues WITH pre-match odds. Only the
imminent matches appear (it is repopulated a day or two before each round), so
scripts/update_predictions.py attaches those odds where it can.

The apex domain serves the files; the www host currently 503s.
"""

import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://football-data.co.uk/mmz4281"
FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"
OUT_DIR = Path("data/football-data")

DIVISIONS = {
    "E0": "Premier League",
    "SP1": "LaLiga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
}

# newest first; "2627" == season 2026/27
SEASONS = ["2627", "2526", "2425", "2324", "2223", "2122", "2021", "1920"]

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)


def fetch_csv(div, season):
    url = f"{BASE}/{season}/{div}.csv"
    req = Request(url, headers={"User-Agent": UA, "Referer": "https://football-data.co.uk/"})
    with urlopen(req, timeout=45) as resp:
        body = resp.read()
    text = body.decode("utf-8-sig", errors="replace")
    # a valid file starts with the "Div," header; anything else is an error page
    if not text.lstrip().startswith("Div,"):
        raise ValueError(f"unexpected body ({len(body)} bytes, starts {text[:40]!r})")
    return text


def fetch_fixtures():
    """Upcoming matches + odds (small, whatever is within a few days)."""
    req = Request(FIXTURES_URL, headers={"User-Agent": UA,
                                         "Referer": "https://football-data.co.uk/"})
    with urlopen(req, timeout=45) as resp:
        text = resp.read().decode("utf-8-sig", errors="replace")
    if not text.lstrip().startswith("Div,"):
        raise ValueError("unexpected fixtures.csv body")
    return text


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = skipped = failed = 0

    try:
        text = fetch_fixtures()
        (OUT_DIR / "fixtures.csv").write_text(text, encoding="utf-8")
        print(f"  fixtures.csv  {len(text)} bytes  ~{text.count(chr(10))} rows")
    except Exception as exc:
        print(f"  skip fixtures.csv: {exc}")

    for div in DIVISIONS:
        (OUT_DIR / div).mkdir(exist_ok=True)
        for season in SEASONS:
            dest = OUT_DIR / div / f"{season}.csv"
            is_current = season == SEASONS[0]
            # keep finished seasons once downloaded; always refresh the current one
            if dest.exists() and not is_current:
                skipped += 1
                continue
            try:
                text = fetch_csv(div, season)
            except Exception as exc:
                # a not-yet-started season 404s / errors - that is fine
                print(f"  skip {div} {season}: {exc}")
                failed += 1
                continue
            rows = text.count("\n")
            dest.write_text(text, encoding="utf-8")
            print(f"  {div}/{season}.csv  {len(text):>7} bytes  ~{rows} rows")
            ok += 1
            time.sleep(1.0)
    print(f"downloaded {ok}, kept {skipped}, missing {failed}")
    return 0 if ok or skipped else 1


if __name__ == "__main__":
    sys.exit(main())
