"""BETAVUS fixture + Over-goal prediction builder.

Fixture source: openfootball/football.json (open data, no API key required).
  https://github.com/openfootball/football.json  -- {season}/{code}.json

For each of the six leagues the script:
  1. downloads the current 2026-27 fixture list plus the last three completed
     seasons (cached under data/cache/openfootball/),
  2. builds a light home/away goals model per team + a head-to-head record,
  3. writes Over 0.5 / 1.5 / 2.5 Poisson probabilities for every not-yet-played
     fixture inside the upcoming Sunday-to-Sunday week into predictions.json.

Standard library only, so it runs on a bare `python` in GitHub Actions.
"""

import csv
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teams import DIV_BY_LEAGUE, to_fd  # noqa: E402

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - Python < 3.9
    ZoneInfo = None

REPO = "openfootball/football.json"
BRANCH = "master"
CACHE_DIR = Path("data/cache/openfootball")
OUTPUT_FILE = Path("predictions.json")

# league_id -> (openfootball file stem, display name, short code, stadium tz)
LEAGUES = {
    39: ("en.1", "Premier League", "PL", "Europe/London"),
    40: ("en.2", "Championship", "CH", "Europe/London"),
    140: ("es.1", "LaLiga", "LL", "Europe/Madrid"),
    78: ("de.1", "Bundesliga", "BL", "Europe/Berlin"),
    135: ("it.1", "Serie A", "SA", "Europe/Rome"),
    61: ("fr.1", "Ligue 1", "L1", "Europe/Paris"),
    88: ("nl.1", "Eredivisie", "ED", "Europe/Amsterdam"),
}

# current season first; older seasons contribute with a lower weight
SEASONS = [("2026-27", 1.0), ("2025-26", 0.7), ("2024-25", 0.45), ("2023-24", 0.30)]

H2H_MAX = 8
DEFAULT_TIME = "15:00"
# All six competitions sit on summer time through the mid-September window, so a
# fixed offset is exact and keeps the script working where tzdata is missing.
SUMMER_OFFSET_HOURS = {
    "Europe/London": 1,
    "Europe/Madrid": 2,
    "Europe/Berlin": 2,
    "Europe/Rome": 2,
    "Europe/Paris": 2,
    "Europe/Amsterdam": 2,
}

RAW_URL = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"
API_URL = "https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"


def fetch(path):
    """Return the text of a repo file, trying raw then the contents API."""
    attempts = (
        (RAW_URL.format(repo=REPO, branch=BRANCH, path=path), {}),
        (
            API_URL.format(repo=REPO, branch=BRANCH, path=path),
            {"Accept": "application/vnd.github.raw"},
        ),
    )
    for url, extra in attempts:
        try:
            req = Request(url, headers={"User-Agent": "betavus-bot", **extra})
            with urlopen(req, timeout=40) as resp:
                return resp.read().decode("utf-8")
        except Exception as exc:  # network / rate limit / 404
            print(f"  fetch failed ({url}): {exc}")
    return None


def load_season(stem, season):
    """Fixture list for one league-season, refreshing the local cache."""
    cache = CACHE_DIR / f"{season}_{stem}.json"
    text = fetch(f"{season}/{stem}.json")
    if text is not None:
        try:
            json.loads(text)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
        except json.JSONDecodeError:
            text = None
    if text is None and cache.exists():
        print(f"  using cached {cache}")
        text = cache.read_text(encoding="utf-8")
    if text is None:
        return []
    return json.loads(text).get("matches", [])


def ft_goals(match):
    score = match.get("score")
    if isinstance(score, dict) and isinstance(score.get("ft"), list) and len(score["ft"]) == 2:
        try:
            return int(score["ft"][0]), int(score["ft"][1])
        except (TypeError, ValueError):
            return None
    return None


def poisson_over(lam, n):
    """P(X > n) for X ~ Poisson(lam)."""
    term = math.exp(-lam)
    cdf = term
    for k in range(1, n + 1):
        term *= lam / k
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


def label(p):
    if p >= 0.95:
        return "ULTRA"
    if p >= 0.90:
        return "HIGH"
    if p >= 0.85:
        return "MEDIUM"
    return ""


# openfootball uses long canonical names; map the ones worth shortening for the
# dashboard. Keyed by the name *after* the generic-suffix trim below.
TEAM_NAMES = {
    # Bundesliga
    "1. FC Köln": "Köln", "1. FC Union Berlin": "Union Berlin",
    "1. FSV Mainz 05": "Mainz 05", "Bayer 04 Leverkusen": "Leverkusen",
    "Borussia Mönchengladbach": "Mönchengladbach", "FC Augsburg": "Augsburg",
    "FC Bayern München": "Bayern München", "FC Schalke 04": "Schalke 04",
    "SC Freiburg": "Freiburg", "SC Paderborn 07": "Paderborn",
    "SV 07 Elversberg": "Elversberg", "SV Werder Bremen": "Werder Bremen",
    "TSG 1899 Hoffenheim": "Hoffenheim", "VfB Stuttgart": "Stuttgart",
    # Premier League
    "AFC Bournemouth": "Bournemouth",
    # LaLiga
    "Athletic Club": "Athletic Bilbao", "CA Osasuna": "Osasuna",
    "Club Atlético de Madrid": "Atlético Madrid", "Deportivo Alavés": "Alavés",
    "FC Barcelona": "Barcelona", "Levante UD": "Levante",
    "RC Celta de Vigo": "Celta Vigo", "RC Deportivo La Coruña": "Deportivo La Coruña",
    "RCD Espanyol de Barcelona": "Espanyol", "Rayo Vallecano de Madrid": "Rayo Vallecano",
    "Real Betis Balompié": "Real Betis", "Real Racing Club de Santander": "Racing Santander",
    "Real Sociedad de Fútbol": "Real Sociedad",
    # Serie A
    "AC Milan": "Milan", "AC Monza": "Monza", "ACF Fiorentina": "Fiorentina",
    "AS Roma": "Roma", "Bologna FC 1909": "Bologna", "Cagliari Calcio": "Cagliari",
    "Como 1907": "Como", "FC Internazionale Milano": "Inter",
    "Frosinone Calcio": "Frosinone", "Genoa CFC": "Genoa",
    "Parma Calcio 1913": "Parma", "SS Lazio": "Lazio", "SSC Napoli": "Napoli",
    "US Lecce": "Lecce", "US Sassuolo Calcio": "Sassuolo", "Udinese Calcio": "Udinese",
    # Ligue 1
    "AJ Auxerre": "Auxerre", "AS Monaco": "Monaco", "Angers SCO": "Angers",
    "ES Troyes": "Troyes", "FC Lorient": "Lorient", "Lille OSC": "Lille",
    "OGC Nice": "Nice", "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille", "Paris": "Paris FC",
    "RC Strasbourg Alsace": "Strasbourg", "Racing Club de Lens": "Lens",
    "Stade Brestois 29": "Brest", "Stade Rennais FC 1901": "Rennes",
    # Eredivisie
    "AFC Ajax": "Ajax", "FC Groningen": "Groningen", "FC Twente '65": "Twente",
    "FC Utrecht": "Utrecht", "Feyenoord Rotterdam": "Feyenoord",
    "SBV Excelsior": "Excelsior", "SC Cambuur-Leeuwarden": "Cambuur",
    "SC Heerenveen": "Heerenveen", "Telstar 1963": "Telstar",
    "Willem II Tilburg": "Willem II",
}


def clean_name(name):
    n = name.strip()
    for suffix in (" FC", " CF", " AC", " BC", " SC", " AFC"):
        if n.endswith(suffix) and len(n) > len(suffix) + 3:
            n = n[: -len(suffix)]
    n = n.strip()
    return TEAM_NAMES.get(n, n)


def kickoff_utc(day, clock, tz_name):
    clock = clock or DEFAULT_TIME
    try:
        hh, mm = (int(x) for x in clock.split(":")[:2])
    except ValueError:
        hh, mm = 15, 0
    y, mo, da = (int(x) for x in day.split("-"))
    if ZoneInfo is not None:
        try:
            local = datetime(y, mo, da, hh, mm, tzinfo=ZoneInfo(tz_name))
            return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    offset = SUMMER_OFFSET_HOURS.get(tz_name, 1)
    naive = datetime(y, mo, da, hh, mm) - timedelta(hours=offset)
    return naive.strftime("%Y-%m-%dT%H:%M:%SZ")


FORECAST_DAYS = 10  # publish the next 10 days of fixtures (today + 10)


def sunday_to_sunday(today):
    """From today through FORECAST_DAYS ahead. (Name kept for import stability.)"""
    return today, today + timedelta(days=FORECAST_DAYS)


class LeagueModel:
    """Weighted home/away scoring rates and a head-to-head record for one league."""

    def __init__(self, seasons):
        self.home_gf, self.home_ga = {}, {}
        self.away_gf, self.away_ga = {}, {}
        self.h2h = {}
        hsum = [0.0, 0.0]
        asum = [0.0, 0.0]
        for matches, weight in seasons:
            for m in matches:
                goals = ft_goals(m)
                if not goals:
                    continue
                home, away = m["team1"], m["team2"]
                hg, ag = goals
                self._add(self.home_gf, home, hg, weight)
                self._add(self.home_ga, home, ag, weight)
                self._add(self.away_gf, away, ag, weight)
                self._add(self.away_ga, away, hg, weight)
                hsum[0] += hg * weight
                hsum[1] += weight
                asum[0] += ag * weight
                asum[1] += weight
                self.h2h.setdefault(frozenset((home, away)), []).append(
                    (m.get("date", ""), hg + ag)
                )
        self.base_home = hsum[0] / hsum[1] if hsum[1] else 1.5
        self.base_away = asum[0] / asum[1] if asum[1] else 1.1

    @staticmethod
    def _add(store, key, value, weight):
        entry = store.setdefault(key, [0.0, 0.0])
        entry[0] += value * weight
        entry[1] += weight

    @staticmethod
    def _avg(store, key, fallback):
        entry = store.get(key)
        return entry[0] / entry[1] if entry and entry[1] else fallback

    def predict(self, home, away):
        hgf = self._avg(self.home_gf, home, self.base_home)
        hga = self._avg(self.home_ga, home, self.base_away)
        agf = self._avg(self.away_gf, away, self.base_away)
        aga = self._avg(self.away_ga, away, self.base_home)
        known = (home in self.home_gf) + (away in self.away_gf)

        lam = (hgf + aga) / 2 + (agf + hga) / 2
        basis = "form" if known == 2 else "partial-form" if known == 1 else "league-avg"

        pair = sorted(self.h2h.get(frozenset((home, away)), []), reverse=True)[:H2H_MAX]
        h2h_used = len(pair)
        if h2h_used >= 2:
            h2h_avg = sum(tg for _, tg in pair) / h2h_used
            lam = 0.72 * lam + 0.28 * h2h_avg
            basis += "+h2h"

        lam = max(0.30, min(6.0, lam))
        p05 = poisson_over(lam, 0)
        return {
            "basis": basis,
            "h2h_matches_used": h2h_used,
            "exp_goals": round(lam, 3),
            "p_over_0_5": round(p05, 4),
            "p_over_1_5": round(poisson_over(lam, 1), 4),
            "p_over_2_5": round(poisson_over(lam, 2), 4),
            "label": label(p05) if basis.startswith("form") else "",
        }


FIXTURES_CSV = Path("data/football-data/fixtures.csv")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _implied(o, u):
    if not o or not u:
        return None
    io, iu = 1 / o, 1 / u
    return round(io / (io + iu), 4)


def load_market_odds():
    """(league, fd_home, fd_away) -> pre-match Over/Under 2.5 market-average odds
    from football-data.co.uk's fixtures.csv, when that round is listed there."""
    if not FIXTURES_CSV.exists():
        return {}
    league_by_div = {d: lg for lg, d in DIV_BY_LEAGUE.items()}
    out = {}
    with FIXTURES_CSV.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            league = league_by_div.get((r.get("Div") or "").strip())
            if not league:
                continue
            o, u = _f(r.get("Avg>2.5")), _f(r.get("Avg<2.5"))
            out[(league, (r.get("HomeTeam") or "").strip(), (r.get("AwayTeam") or "").strip())] = {
                "o25_odds": o, "u25_odds": u, "o25_implied": _implied(o, u),
                "h": _f(r.get("AvgH")), "d": _f(r.get("AvgD")), "a": _f(r.get("AvgA")),
            }
    return out


def main():
    now = datetime.now(timezone.utc)
    start, end = sunday_to_sunday(now.date())
    print(f"Window: {start} .. {end}")
    odds = load_market_odds()
    print(f"Market odds rows (our leagues): {len(odds)}")

    predictions = []
    for lid, (stem, name, code, tz_name) in LEAGUES.items():
        print(f"[{name}] {stem}")
        seasons = [(load_season(stem, s), w) for s, w in SEASONS]
        current = seasons[0][0]
        model = LeagueModel(seasons)

        count = 0
        for m in current:
            day = m.get("date")
            if not day:
                continue
            try:
                match_day = date.fromisoformat(day)
            except ValueError:
                continue
            if not (start <= match_day <= end) or ft_goals(m):
                continue
            count += 1
            pred = model.predict(m["team1"], m["team2"])
            home, away = clean_name(m["team1"]), clean_name(m["team2"])
            row = {
                "match_id": f"{code}-{match_day.isoformat()}-{count:02d}",
                "league_id": lid,
                "league": name,
                "kickoff_utc": kickoff_utc(day, m.get("time"), tz_name),
                "home": home,
                "away": away,
                "source": "openfootball/football.json",
                **pred,
                "updated_at": now.isoformat(),
            }
            mk = odds.get((name, to_fd(name, home), to_fd(name, away)))
            if mk:
                edge = None
                if mk["o25_implied"] is not None:
                    edge = round(pred["p_over_2_5"] - mk["o25_implied"], 4)
                row["market"] = {**mk, "edge25": edge}
            predictions.append(row)
        print(f"  {count} upcoming fixtures")

    predictions.sort(key=lambda x: x["kickoff_utc"])
    OUTPUT_FILE.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(predictions)} fixtures across {len(LEAGUES)} leagues")


if __name__ == "__main__":
    main()
