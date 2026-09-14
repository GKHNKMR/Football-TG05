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

Some API-Football plans only allow querying a few days back, so a single run
may only really cover "yesterday" - each run therefore MERGES its findings
into the existing file instead of overwriting it, and every entry from the
previous file is kept unless this run explicitly re-confirms that date+league
(so a transient failure never erases what an earlier run already found; a
day that DOES come back this run replaces yesterday's version of it, so a
match that went from in-progress to finished still updates).
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_scores import LIVE_FILE  # noqa: E402
from teams import DIVISIONS  # noqa: E402

API_URL = "https://v3.football.api-sports.io/fixtures"
LOOKBACK_DAYS = 8
KEEP_DAYS = 21  # prune anything older than this out of the merged file
LEAGUE_API_ID = {name: lid for _div, (name, lid) in DIVISIONS.items()}
# Committed to the repo each run so a per-day fetch failure is visible via a
# normal `git show`, without needing to pull GitHub Actions job logs.
DEBUG_FILE = Path("data/live-scores-debug.json")


def fetch_day(key, d):
    """One day's fixtures in our leagues with a score. Raises on a hard failure."""
    req = Request(f"{API_URL}?date={d}", headers={"x-apisports-key": key})
    try:
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    data = json.loads(body)
    if data.get("errors"):
        raise RuntimeError(f"API errors: {data['errors']}")
    id_to_league = {lid: name for name, lid in LEAGUE_API_ID.items()}
    out = []
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
    return out


def main():
    key = os.environ.get("API_FOOTBALL_KEY")
    LIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not key:
        print("API_FOOTBALL_KEY not set - writing an empty live-scores.json")
        LIVE_FILE.write_text("[]", encoding="utf-8")
        DEBUG_FILE.write_text(json.dumps(
            {"run_at": datetime.now(timezone.utc).isoformat(), "days": [],
             "error": "API_FOOTBALL_KEY not set"}, indent=1), encoding="utf-8")
        return

    try:
        previous = json.loads(LIVE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous = []

    today = datetime.now(timezone.utc).date()
    cutoff = (today - timedelta(days=KEEP_DAYS)).isoformat()
    fetched_dates = set()
    new_rows = []
    debug = {"run_at": datetime.now(timezone.utc).isoformat(), "days": []}
    for delta in range(LOOKBACK_DAYS):
        d = (today - timedelta(days=delta)).isoformat()
        try:
            rows = fetch_day(key, d)
        except Exception as exc:
            print(f"  {d}: fetch failed - {exc}")
            debug["days"].append({"date": d, "ok": False, "error": str(exc)})
            continue
        fetched_dates.add(d)
        new_rows.extend(rows)
        print(f"  {d}: {len(rows)} matches with a score")
        debug["days"].append({"date": d, "ok": True, "n_matches": len(rows)})
        if delta < LOOKBACK_DAYS - 1:
            time.sleep(1)  # be gentle with per-second rate limits
    DEBUG_FILE.write_text(json.dumps(debug, ensure_ascii=False, indent=1), encoding="utf-8")

    # keep prior entries for dates we couldn't re-fetch this run; drop the
    # rest of that date's old rows wherever we DID get a fresh answer, so a
    # finished match properly replaces its earlier in-progress version
    kept = [r for r in previous if r["date"] >= cutoff and r["date"] not in fetched_dates]
    merged = kept + new_rows

    LIVE_FILE.write_text(json.dumps(merged, ensure_ascii=False, separators=(",", ":")),
                         encoding="utf-8")
    n_fin = sum(1 for x in merged if x["finished"])
    print(f"API-Football: {len(merged)} matches with a score ({n_fin} finished) "
          f"across {len(set(x['date'] for x in merged))} days "
          f"({len(fetched_dates)}/{LOOKBACK_DAYS} fetched fresh this run)")


if __name__ == "__main__":
    main()
