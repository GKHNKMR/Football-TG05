"""Fetch recent/live matches for our leagues from API-Football -> data/live-scores.json.

Needs the API_FOOTBALL_KEY GitHub secret. Without it, writes an empty list so
every consumer (update_predictions.py, build_results.py) degrades safely to
its slower always-available source (openfootball / football-data.co.uk).

Pulls one day at a time (date=YYYY-MM-DD, all leagues) for LOOKBACK_DAYS back
from today and keeps only fixtures in our 8 leagues that have a score - live
in-progress ones included, not just finished (FT) ones - so:
  - update_predictions.py can drop a fixture once it's actually finished, and
    show a live score while one is still being played;
  - build_results.py can grade a finished match before football-data.co.uk's
    CSV (which lags real matches by days) has it.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_scores import LIVE_FILE  # noqa: E402
from teams import DIVISIONS  # noqa: E402

API_URL = "https://v3.football.api-sports.io/fixtures"
LOOKBACK_DAYS = 8
LEAGUE_API_ID = {name: lid for _div, (name, lid) in DIVISIONS.items()}


def main():
    key = os.environ.get("API_FOOTBALL_KEY")
    LIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not key:
        print("API_FOOTBALL_KEY not set - writing an empty live-scores.json")
        LIVE_FILE.write_text("[]", encoding="utf-8")
        return

    id_to_league = {lid: name for name, lid in LEAGUE_API_ID.items()}
    out = []
    today = datetime.now(timezone.utc).date()
    for delta in range(LOOKBACK_DAYS):
        d = (today - timedelta(days=delta)).isoformat()
        try:
            req = Request(f"{API_URL}?date={d}", headers={"x-apisports-key": key})
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"  API-Football fetch failed for {d}: {exc}")
            continue
        for item in data.get("response", []):
            league = id_to_league.get((item.get("league") or {}).get("id"))
            if not league:
                continue
            goals = item.get("goals") or {}
            hg, ag = goals.get("home"), goals.get("away")
            if hg is None or ag is None:
                continue  # not started yet, or no data - nothing useful to show
            status = ((item.get("fixture") or {}).get("status") or {}).get("short")
            teams = item.get("teams") or {}
            out.append({
                "league": league, "date": d, "status": status,
                "finished": status == "FT",
                "home": (teams.get("home") or {}).get("name", ""),
                "away": (teams.get("away") or {}).get("name", ""),
                "score": f"{hg}-{ag}", "total": hg + ag,
            })
    LIVE_FILE.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")),
                         encoding="utf-8")
    n_fin = sum(1 for x in out if x["finished"])
    print(f"API-Football: {len(out)} matches with a score ({n_fin} finished), "
          f"last {LOOKBACK_DAYS} days")


if __name__ == "__main__":
    main()
