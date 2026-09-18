"""Historical UEFA club-competition match dates, from ESPN's public soccer
API (same free, no-key source scripts/fetch_live_scores.py already uses) -
so scripts/tune_rest_days.py can test the REAL "played a midweek Champions/
Europa/Conference League match" hypothesis instead of only the domestic-only
proxy (see that script's docstring for why the domestic-only version
understates the signal).

Two-step lookup, both via ESPN's site API:
  1. A domestic league's historical standings for a season lists every team
     that competed that season with ESPN's own persistent team id (stable
     across competitions and seasons for the same club) - cheap: one call
     per (league, season) instead of hunting day by day.
  2. That team id's schedule under each UEFA competition slug for the same
     season - most teams return zero events (never qualified); the ones
     that did are exactly the source of the rotation-risk hypothesis.

Everything is cached to disk (data/cache/espn_euro/) since this is a fixed
historical record that never changes once a season is over - a second run
(e.g. re-tuning with a different rest-day formula) costs zero new requests.

Standard library only.
"""

import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_scores import norm as _fold  # noqa: E402 - same ESPN-name-fold idiom, word aliases and all
from teams import to_fd  # noqa: E402

CACHE_DIR = Path("data/cache/espn_euro")

DOMESTIC_SLUG = {
    "Premier League": "eng.1", "LaLiga": "esp.1", "Bundesliga": "ger.1",
    "Serie A": "ita.1", "Ligue 1": "fra.1", "Eredivisie": "ned.1",
    "Turkish Süper Lig": "tur.1", "Primeira Liga": "por.1",
}
EURO_SLUGS = ["uefa.champions", "uefa.europa", "uefa.europa.conf"]

# BETAVUS season code ("2122") -> ESPN's own season query param (start year)
def season_code_to_espn_year(code):
    return f"20{code[:2]}"


def fetch_json(url):
    req = Request(url)  # ESPN's WAF blocks a custom UA - see fetch_live_scores.py
    with urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _cached(path, fetch_fn):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    data = fetch_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def season_team_ids(league, season_code):
    """{"espn display name": "espn team id"} for every club in that domestic
    league+season, or {} if ESPN has no standings for it (skip that league/
    season combo - rare, e.g. a league ESPN doesn't carry that far back)."""
    slug = DOMESTIC_SLUG.get(league)
    if not slug:
        return {}
    year = season_code_to_espn_year(season_code)
    cache_path = CACHE_DIR / f"teams_{slug}_{year}.json"

    def fetch():
        url = f"https://site.api.espn.com/apis/v2/sports/soccer/{slug}/standings?season={year}"
        try:
            data = fetch_json(url)
        except (HTTPError, URLError, json.JSONDecodeError) as exc:
            print(f"  standings fetch failed [{league} {year}]: {exc}")
            return {}
        out = {}
        for group in data.get("children", []):
            for entry in group.get("standings", {}).get("entries", []):
                t = entry.get("team", {})
                if t.get("id") and t.get("displayName"):
                    out[t["displayName"]] = t["id"]
        return out

    return _cached(cache_path, fetch)


def team_euro_dates(team_id, season_code):
    """Sorted list of ISO date strings this team_id played in ANY of the
    three UEFA club competitions during that season - empty if it never
    qualified for Europe that year (the overwhelmingly common case)."""
    year = season_code_to_espn_year(season_code)
    cache_path = CACHE_DIR / f"euro_{team_id}_{year}.json"

    def fetch():
        dates = set()
        for slug in EURO_SLUGS:
            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/teams/{team_id}/schedule?season={year}"
            try:
                data = fetch_json(url)
            except (HTTPError, URLError, json.JSONDecodeError) as exc:
                print(f"  euro schedule fetch failed [team {team_id} {slug} {year}]: {exc}")
                continue
            for ev in data.get("events", []):
                d = ev.get("date", "")
                if d:
                    dates.add(d[:10])
            time.sleep(0.15)  # be gentle with this public, unauthenticated endpoint
        return sorted(dates)

    return _cached(cache_path, fetch)


def build_matcher(league, espn_names_to_id, fd_names):
    """{fd_name: espn_id}, for every ESPN team that resolves to one of this
    division's football-data.co.uk names. ESPN's displayName ("Manchester
    City", "Wolverhampton Wanderers") is consistently the same long/official
    form openfootball uses for BETAVUS's own display names, so
    teams.to_fd() - built for exactly that openfootball -> football-data.co.uk
    gap - resolves almost everything directly; the fold-and-substring pass
    only mops up names identical on both sides (no crosswalk entry needed)."""
    fd_folded = {_fold(n): n for n in fd_names}

    def match(espn_name):
        via_crosswalk = to_fd(league, espn_name)
        if via_crosswalk in fd_names:
            return via_crosswalk
        f = _fold(espn_name)
        if f in fd_folded:
            return fd_folded[f]
        cands = [fdn for ff, fdn in fd_folded.items() if ff and (ff in f or f in ff)]
        return cands[0] if len(cands) == 1 else None

    out = {}
    for espn_name, espn_id in espn_names_to_id.items():
        fd_name = match(espn_name)
        if fd_name:
            out[fd_name] = espn_id
    return out


def euro_dates_for_league_season(league, season_code, fd_names):
    """{fd_name: [iso dates played in Europe that season]} for every team in
    this division+season that resolves to a football-data.co.uk name -
    teams that never qualified for Europe just get an empty (and cheap,
    cached) list back from team_euro_dates."""
    ids = season_team_ids(league, season_code)
    if not ids:
        return {}
    matched = build_matcher(league, ids, fd_names)
    return {fd_name: team_euro_dates(espn_id, season_code)
            for fd_name, espn_id in matched.items()}
