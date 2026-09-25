"""Fetch recent/live matches for our leagues from ESPN's public scoreboard
API -> data/live-scores.json.

Free, no API key, no plan/date restriction (unlike API-Football's paid
tiers, whose free plan turned out to only allow a yesterday/today/tomorrow
window - see git history for that dead end). Pulls one (league, day) pair
at a time for LOOKBACK_DAYS back from today, so:

  - update_predictions.py can drop a fixture from Tahminler once it's
    actually finished (openfootball/football-data can take days to record
    a score, so without this a played match keeps showing as "upcoming"),
    and attach a live in-progress score onto ones still being played.
  - build_results.py can grade an archived (real pre-kickoff) prediction
    before football-data.co.uk's CSV (which lags real matches by days,
    sometimes over a week) has caught up with that result.

Each run MERGES its findings into the existing file rather than
overwriting it, so a transient per-request failure never erases what an
earlier run already found; a (league, date) that DOES come back this run
replaces its old entries, so a match that went from in-progress to
finished still updates.
"""

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_scores import LIVE_FILE  # noqa: E402

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
LOOKBACK_DAYS = 14
KEEP_DAYS = 21  # prune anything older than this out of the merged file
# Committed to the repo each run so a per-day fetch failure is visible via a
# normal `git show`, without needing to pull GitHub Actions job logs.
DEBUG_FILE = Path("data/live-scores-debug.json")

ESPN_SLUG = {
    "Premier League": "eng.1",
    "Championship": "eng.2",
    "LaLiga": "esp.1",
    "Bundesliga": "ger.1",
    "Serie A": "ita.1",
    "Ligue 1": "fra.1",
    "Eredivisie": "ned.1",
    "Turkish Süper Lig": "tur.1",
    "Primeira Liga": "por.1",
    "Belgian Pro League": "bel.1",
}


def fetch_league_day(league, slug, d):
    """One league's matches on day d (ISO date str) with a score. Raises on failure."""
    url = f"{SCOREBOARD_URL.format(slug=slug)}?dates={d.replace('-', '')}"
    # ESPN's WAF blocks a generic "Mozilla/5.0" UA (403) but is fine with no
    # override at all - leave the default urllib UA alone.
    req = Request(url)
    with urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = []
    for event in data.get("events", []):
        comp = (event.get("competitions") or [{}])[0]
        status = comp.get("status", {}).get("type", {})
        if status.get("state") == "pre":
            continue  # not started yet - nothing useful to show
        competitors = comp.get("competitors", [])
        by_side = {c.get("homeAway"): c for c in competitors}
        home, away = by_side.get("home"), by_side.get("away")
        if not home or not away:
            continue
        try:
            hg, ag = int(home["score"]), int(away["score"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({
            "league": league, "date": d,
            "status": status.get("shortDetail") or status.get("name"),
            "finished": bool(status.get("completed")),
            "home": home["team"]["displayName"], "away": away["team"]["displayName"],
            "score": f"{hg}-{ag}", "total": hg + ag,
        })
    return out


def main():
    LIVE_FILE.parent.mkdir(parents=True, exist_ok=True)

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
        day_ok, day_matches, day_error = True, 0, None
        for league, slug in ESPN_SLUG.items():
            try:
                rows = fetch_league_day(league, slug, d)
            except (HTTPError, URLError, json.JSONDecodeError) as exc:
                day_ok = False
                day_error = f"{league}: {exc}"
                print(f"  {d} {league}: fetch failed - {exc}")
                continue
            new_rows.extend(rows)
            day_matches += len(rows)
            time.sleep(0.2)  # be gentle with this public, unauthenticated endpoint
        fetched_dates.add(d)  # per-league failures still count the day as attempted
        print(f"  {d}: {day_matches} matches with a score")
        debug["days"].append({"date": d, "ok": day_ok, "n_matches": day_matches,
                               **({"error": day_error} if day_error else {})})
    DEBUG_FILE.write_text(json.dumps(debug, ensure_ascii=False, indent=1), encoding="utf-8")

    # keep prior entries for dates we couldn't (re)fetch this run; drop the
    # rest of that date's old rows wherever we DID get a fresh answer, so a
    # finished match properly replaces its earlier in-progress version
    kept = [r for r in previous if r["date"] >= cutoff and r["date"] not in fetched_dates]
    merged = kept + new_rows

    LIVE_FILE.write_text(json.dumps(merged, ensure_ascii=False, separators=(",", ":")),
                         encoding="utf-8")
    n_fin = sum(1 for x in merged if x["finished"])
    print(f"ESPN scoreboard: {len(merged)} matches with a score ({n_fin} finished) "
          f"across {len(set(x['date'] for x in merged))} days "
          f"({len(fetched_dates)}/{LOOKBACK_DAYS} days fetched this run)")


if __name__ == "__main__":
    main()
