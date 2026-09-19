"""Stadium coordinates for BETAVUS's clubs, from Wikidata (free, no key) -
so scripts/tune_weather.py can look up each match's historical weather
without hand-building a club -> city table (tried and rejected first - see
git history: a plain "club name" -> Nominatim geocode is unreliable,
"Manchester United FC" resolves to a street in Idaho, and hand-verifying a
city for all ~150-250 clubs that have appeared in these 9 leagues over
five-plus seasons was judged too error-prone to trust; a bulk SPARQL query
by "every club Wikidata links to this league via P118" was ALSO tried and
rejected - P118 turned out to record only a club's CURRENT league, not its
full history, so a club that has since been promoted/relegated out of the
league it played our target seasons in - Leicester City, West Bromwich
Albion, plenty of Championship-in-2026 sides that were Premier League
during our backtest window - was silently missing).

What actually works: per-club lookup, driven by the roster we already have
(football-data.co.uk's own team names via scripts/opta_xg.fd_team_names -
no dependency on Wikidata knowing league history at all):
  1. Wikidata's wbsearchentities API (fast, indexed - unlike a full-corpus
     SPARQL text scan, which timed out) to find candidate Wikidata items,
     searching the BETAVUS "pretty" display name (teams.to_pretty) when the
     crosswalk has one - "Nottingham Forest" searches far better than
     football-data's own "Nott'm Forest" - falling back to the raw
     football-data name otherwise.
  2. Keep the first candidate whose description mentions a football club
     (rejects e.g. "Inter" matching IMDb ahead of Inter Milan).
  3. A single targeted SPARQL query anchored to that specific Q-id (P115
     home venue -> P625 coordinate) - fast, since it's a one-entity lookup,
     not a scan.

Real coverage is well under 100% - some clubs (lower-table sides in
Championship/Eredivisie/Sueper Lig/Primeira Liga especially) don't resolve
cleanly by name at all and are left out rather than guessed at; a match
involving an unresolved club just has no weather signal for that side, the
same graceful-degradation idiom scripts/fetch_key_players.py already uses
for an unmatched team.

Cached to disk (data/cache/wikidata_stadiums/) - a fixed historical record,
a second run costs zero new requests.

Standard library only.
"""

import json
import re
import sys
import time
import unicodedata
import urllib.parse
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teams import to_pretty  # noqa: E402

CACHE_DIR = Path("data/cache/wikidata_stadiums")
SEARCH_URL = "https://www.wikidata.org/w/api.php"
SPARQL_URL = "https://query.wikidata.org/sparql"
_POINT_RE = re.compile(r"Point\(([-\d.]+)\s+([-\d.]+)\)")
_FOOTBALL_HINT = re.compile(r"football club|association football", re.I)
REQUEST_PAUSE = 1.5  # Wikidata's public endpoints throttle hard and for a
                      # sustained window - 0.5s wasn't nearly enough


def _safe(msg):
    """Exception text can carry non-ASCII (accented club/venue names) that
    crashes a Windows cp1252 console - never let a log line kill the run."""
    return msg.encode("ascii", errors="replace").decode("ascii")


def _get_json(url):
    """5 attempts total, backing off hard on a 429 - Wikidata's public
    endpoints throttle for a sustained window, not just a per-request beat,
    so a short backoff (or the earlier flat 0.5s pacing) wasn't enough."""
    last_exc = None
    for attempt in range(5):
        try:
            req = Request(url, headers={"User-Agent": "betavus-research/1.0 (one-off analysis script)"})
            with urlopen(req, timeout=25) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            last_exc = exc
            if exc.code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            raise
    raise last_exc


def _search_club_qid(term):
    url = (f"{SEARCH_URL}?action=wbsearchentities&search={urllib.parse.quote(term)}"
           f"&language=en&format=json&type=item&limit=5")
    try:
        data = _get_json(url)
    except Exception as exc:
        print(_safe(f"  wbsearchentities failed [{term}]: {exc}"))
        return None
    for cand in data.get("search", []):
        if _FOOTBALL_HINT.search(cand.get("description") or ""):
            return cand["id"]
    return None


def _venue_coord(qid):
    query = (f'SELECT ?coord WHERE {{ wd:{qid} wdt:P115 ?venue. '
             f'?venue wdt:P625 ?coord. }} LIMIT 1')
    url = f"{SPARQL_URL}?query={urllib.parse.quote(query)}&format=json"
    try:
        data = _get_json(url)
    except Exception as exc:
        print(_safe(f"  venue SPARQL failed [{qid}]: {exc}"))
        return None
    rows = data["results"]["bindings"]
    if not rows:
        return None
    m = _POINT_RE.match(rows[0]["coord"]["value"])
    if not m:
        return None
    lon, lat = float(m.group(1)), float(m.group(2))
    return (lat, lon)


def club_coord(league, fd_name):
    """(lat, lon) for this football-data.co.uk club name, or None if it
    can't be resolved. Cached per (league, fd_name) - a fixed historical
    fact once found (or not)."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", fd_name).strip("_")
    cache_path = CACHE_DIR / f"{re.sub(r'[^A-Za-z0-9]+', '_', league)}_{safe}.json"
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            return tuple(raw) if raw else None
        except json.JSONDecodeError:
            pass

    search_term = to_pretty(league, fd_name)
    if search_term == fd_name:
        search_term = fd_name  # no crosswalk entry - search the raw name as-is
    qid = _search_club_qid(search_term)
    time.sleep(REQUEST_PAUSE)
    coord = _venue_coord(qid) if qid else None
    if qid:
        time.sleep(REQUEST_PAUSE)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(list(coord) if coord else None), encoding="utf-8")
    return coord


def coords_for_league(league, fd_names):
    """{fd_name: (lat, lon)} for every name in fd_names this can resolve."""
    out = {}
    for fd_name in fd_names:
        coord = club_coord(league, fd_name)
        if coord:
            out[fd_name] = coord
    return out
