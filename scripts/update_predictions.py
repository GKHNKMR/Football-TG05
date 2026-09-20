"""BETAVUS fixture + Over-goal prediction builder.

Fixture source: openfootball/football.json (open data, no API key required).
  https://github.com/openfootball/football.json  -- {season}/{code}.json

For each openfootball-sourced league the script:
  1. downloads the current 2026-27 fixture list plus the last three completed
     seasons (cached under data/cache/openfootball/),
  2. builds a goals_model.LeagueModel (weighted, recency-decayed home/away
     scoring rates, Dixon-Coles-adjusted Over probabilities) + a head-to-head
     record,
  3. writes Over 0.5 / 1.5 / 2.5 probabilities for every not-yet-played
     fixture inside the upcoming Sunday-to-Sunday week into predictions.json.

Standard library only, so it runs on a bare `python` in GitHub Actions.
"""

import csv
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teams import DIV_BY_LEAGUE, to_fd, to_pretty  # noqa: E402
from goals_model import LeagueModel, DEFAULT_RHO  # noqa: E402
from live_scores import find_live_match, load_live_scores  # noqa: E402
from xg_blend import XG_WEIGHT_BY_LEAGUE, xg_seasons_for  # noqa: E402

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
    94: ("pt.1", "Primeira Liga", "PR", "Europe/Lisbon"),
}

# current season first; older seasons contribute with a lower weight
SEASONS = [("2026-27", 1.0), ("2025-26", 0.7), ("2024-25", 0.45), ("2023-24", 0.30)]

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
    "Europe/Lisbon": 1,  # Portugal runs on Western European Time, same clock as the UK
    "Europe/Istanbul": 3,
}

# Leagues openfootball does not publish current fixtures for: source both the
# history and the upcoming round from football-data.co.uk instead.
#   league_id -> (football-data DIV, display name, short code, stadium tz)
FD_LEAGUES = {203: ("T1", "Turkish Süper Lig", "TR", "Europe/Istanbul")}
FD_HIST_SEASONS = ["2223", "2324", "2425", "2526", "2627"]   # oldest -> newest
FD_MODEL_CODES = ["2627", "2526", "2425", "2324"]            # nearest -> ...
FD_MODEL_WEIGHTS = [1.0, 0.7, 0.45, 0.30]

# Turkish fixtures: openfootball has none and football-data's fixtures.csv only
# lists the imminent round, so read the schedule straight from the TFF site.
TFF_FIXTURE_URL = "https://www.tff.org/default.aspx?pageID=198"
TFF_MAP = {  # TFF display name -> football-data.co.uk name
    "BEŞİKTAŞ A.Ş.": "Besiktas", "GALATASARAY A.Ş.": "Galatasaray",
    "FENERBAHÇE A.Ş.": "Fenerbahce", "TRABZONSPOR A.Ş.": "Trabzonspor",
    "ÇAYKUR RİZESPOR A.Ş.": "Rizespor", "GÖZTEPE A.Ş.": "Goztep",
    "SAMSUNSPOR A.Ş.": "Samsunspor", "KASIMPAŞA A.Ş.": "Kasimpasa",
    "GAZİANTEP FUTBOL KULÜBÜ A.Ş.": "Gaziantep", "EYÜPSPOR": "Eyupspor",
    "ARCA ÇORUM FK": "Corum", "CORENDON ALANYASPOR": "Alanyaspor",
    "TÜMOSAN KONYASPOR": "Konyaspor", "GENÇLERBİRLİĞİ": "Genclerbirligi",
    "AMED SPORTİF FAALİYETLER": "Amedspor", "İSTANBUL BAŞAKŞEHİR FK": "Buyuksehyr",
    "KOCAELİSPOR": "Kocaelispor", "ERZURUMSPOR FK": "Erzurumspor",
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
    if isinstance(score, list) and len(score) == 2:
        try:
            return int(score[0]), int(score[1])
        except (TypeError, ValueError):
            return None
    return None


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
    # Primeira Liga
    "CD Nacional": "Nacional", "CD Santa Clara": "Santa Clara", "CD Tondela": "Tondela",
    "CF Estrela da Amadora": "Estrela da Amadora", "CS Marítimo": "Marítimo",
    "FC Alverca": "Alverca", "FC Arouca": "Arouca", "FC Famalicão": "Famalicão",
    "FC Porto": "Porto", "FC Vizela": "Vizela",
    "GD Chaves": "Chaves", "GD Estoril Praia": "Estoril",
    "SC Farense": "Farense", "Sport Lisboa e Benfica": "Benfica",
    "Sporting Clube de Braga": "Sporting Braga", "Sporting Clube de Portugal": "Sporting CP",
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


def _of_matches(raw_matches):
    """openfootball's own match shape (team1/team2/score.ft) -> the
    {home,away,hg,ag,date} shape goals_model.LeagueModel expects. Keeps
    team keys as the raw long openfootball names (not clean_name()'d) -
    the model has always been keyed that way; display cleanup happens only
    on the output row."""
    out = []
    for m in raw_matches:
        goals = ft_goals(m)
        if not goals:
            continue
        hg, ag = goals
        out.append({"home": m["team1"], "away": m["team2"], "hg": hg, "ag": ag,
                    "date": m.get("date", "")})
    return out


def of_predict(model, home, away):
    """model.predict() + the Vurgu label, for openfootball-sourced leagues."""
    pred = model.predict(home, away)
    pred["label"] = label(pred["p_over_0_5"]) if pred["basis"].startswith("form") else ""
    pred["base_lam_home"], pred["base_lam_away"], pred["base_rho"] = (
        pred["lam_home"], pred["lam_away"], pred["rho"])
    return pred


FD_DIR = Path("data/football-data")
FIXTURES_CSV = FD_DIR / "fixtures.csv"


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


def _fd_history(div):
    """Completed DIV matches from the local football-data CSVs, oldest first."""
    rows = []
    for s in FD_HIST_SEASONS:
        p = FD_DIR / div / f"{s}.csv"
        if not p.exists():
            continue
        with p.open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                try:
                    hg, ag = int(r["FTHG"]), int(r["FTAG"])
                    d = datetime.strptime(r["Date"].strip(), "%d/%m/%Y").date()
                except (KeyError, ValueError):
                    continue
                rows.append({"season": s, "date": d.isoformat(),
                             "home": r["HomeTeam"].strip(), "away": r["AwayTeam"].strip(),
                             "hg": hg, "ag": ag, "total": hg + ag})
    return rows


_FOLD = {0x130: "I", 0x131: "i", 0x15e: "S", 0x15f: "s", 0x218: "S", 0x219: "s",
         0x11e: "G", 0x11f: "g", 0xdc: "U", 0xfc: "u", 0xd6: "O", 0xf6: "o",
         0xc7: "C", 0xe7: "c", 0xa0: " "}


def _fold(s):
    return s.translate(_FOLD).upper().strip()


_TFF_FOLDED = {_fold(k): v for k, v in TFF_MAP.items()}


def _tff_name(raw, fd_teams):
    key = _fold(raw)
    if key in _TFF_FOLDED:
        return _TFF_FOLDED[key]
    base = re.sub(r"\b(A\.?S\.?|FK|SK|FUTBOL KULUBU)\b", "", key).strip()
    for t in fd_teams:
        tf = _fold(t)
        if tf and (tf in base or base in tf):
            return t
    return raw


def tff_upcoming(fd_teams, start, end):
    """Upcoming Süper Lig fixtures from tff.org (home/away as football-data names)."""
    try:
        req = Request(TFF_FIXTURE_URL, headers={"User-Agent": "Mozilla/5.0 (compatible; betavus-bot)"})
        with urlopen(req, timeout=30) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
        m = re.search(r"charset=([\w-]+)", ctype, re.I)
        enc = (m.group(1) if m else "cp1254")           # TFF serves windows-1254
        try:
            html = raw.decode(enc, errors="replace")
        except LookupError:
            html = raw.decode("cp1254", errors="replace")
    except Exception as exc:
        print(f"  TFF fetch failed: {exc}")
        return []
    rx = re.compile(
        r'lblTarih">([\d.]+)</span>\s*<span[^>]*lblSaat">([\d:]*)</span>'
        r'.*?haftaninMaclariEv">.*?<span[^>]*>([^<]+)</span>'
        r'.*?haftaninMaclariDeplasman">.*?<span[^>]*>([^<]+)</span>', re.S)
    out = []
    for d, tm, h, a in rx.findall(html):
        try:
            dd = datetime.strptime(d.strip(), "%d.%m.%Y").date()
        except ValueError:
            continue
        if start <= dd <= end:
            out.append({"date": dd, "time": tm.strip() or DEFAULT_TIME,
                        "home": _tff_name(h, fd_teams), "away": _tff_name(a, fd_teams)})
    return out


def _fd_upcoming(div, start, end):
    """Upcoming DIV matches from fixtures.csv inside [start, end]."""
    if not FIXTURES_CSV.exists():
        return []
    out = []
    with FIXTURES_CSV.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if (r.get("Div") or "").strip() != div:
                continue
            try:
                d = datetime.strptime(r["Date"].strip(), "%d/%m/%Y").date()
            except (KeyError, ValueError):
                continue
            if start <= d <= end:
                out.append({"date": d, "time": (r.get("Time") or "").strip(),
                            "home": (r.get("HomeTeam") or "").strip(),
                            "away": (r.get("AwayTeam") or "").strip()})
    return out


def _fd_pred_dict(model, home, away):
    """model.predict() + the Vurgu label, for football-data-sourced leagues."""
    pred = model.predict(home, away)
    pred["label"] = label(pred["p_over_0_5"]) if pred["basis"].startswith("form") else ""
    pred["base_lam_home"], pred["base_lam_away"], pred["base_rho"] = (
        pred["lam_home"], pred["lam_away"], pred["rho"])
    return pred


def fd_predictions(now, start, end, odds, live):
    """Predictions for leagues sourced entirely from football-data.co.uk
    (openfootball has no current fixtures for them). History trains the model;
    fixtures.csv supplies the upcoming round, so the horizon is short."""
    preds = []
    dropped_total = 0
    for lid, (div, name, code, tz_name) in FD_LEAGUES.items():
        hist = _fd_history(div)
        by_code = {}
        for m in hist:
            by_code.setdefault(m["season"], []).append(m)
        model = LeagueModel([(by_code.get(c, []), w)
                            for c, w in zip(FD_MODEL_CODES, FD_MODEL_WEIGHTS)])
        fd_teams = {m["home"] for m in hist} | {m["away"] for m in hist}
        up = tff_upcoming(fd_teams, start, end) if div == "T1" else []
        src = "tff.org"
        if not up:
            up, src = _fd_upcoming(div, start, end), "football-data.co.uk"
        n = dropped = 0
        for fx in up:
            if not fx["home"] or not fx["away"]:
                continue
            live_hit = find_live_match(live, name, fx["home"], fx["away"], fx["date"])
            if live_hit and live_hit.get("finished"):
                dropped += 1
                continue
            n += 1
            pred = _fd_pred_dict(model, fx["home"], fx["away"])
            home, away = to_pretty(name, fx["home"]), to_pretty(name, fx["away"])
            row = {
                "match_id": f"{code}-{fx['date'].isoformat()}-{n:02d}",
                "league_id": lid, "league": name,
                "kickoff_utc": kickoff_utc(fx["date"].isoformat(), fx["time"], tz_name),
                "home": home, "away": away,
                "source": src,
                **pred, "updated_at": now.isoformat(),
            }
            if live_hit:
                row["live"] = {"status": live_hit.get("status"), "score": live_hit.get("score")}
            mk = odds.get((name, fx["home"], fx["away"]))
            if mk:
                edge = None
                if mk["o25_implied"] is not None:
                    edge = round(pred["p_over_2_5"] - mk["o25_implied"], 4)
                row["market"] = {**mk, "edge25": edge}
            preds.append(row)
        dropped_total += dropped
        print(f"[{name}] {div} · {len(hist)} history rows · {n} upcoming ({src})"
              f"{f' · {dropped} dropped (already finished)' if dropped else ''}")
    return preds


def _league_rho(name):
    """Fit Dixon-Coles rho from the reliable football-data.co.uk CSVs, even for
    openfootball-sourced leagues - openfootball's own mirrored scores are known
    to under-report low-scoring results in some seasons (e.g. its 2025-26 PL
    cache is missing all 27 of that season's real 0-0 draws, along with ~7% of
    matches outright). That barely moves the mean-based lambda estimate the
    live rates use, but would badly bias a fit specifically about how often
    low scores happen, so rho always comes from the CSVs regardless of which
    source trains the rates themselves."""
    div = DIV_BY_LEAGUE.get(name)
    if not div:
        return DEFAULT_RHO
    hist = _fd_history(div)
    by_code = {}
    for m in hist:
        by_code.setdefault(m["season"], []).append(m)
    rho_model = LeagueModel([(by_code.get(c, []), w)
                             for c, w in zip(FD_MODEL_CODES, FD_MODEL_WEIGHTS)])
    return rho_model.rho


def main():
    now = datetime.now(timezone.utc)
    start, end = sunday_to_sunday(now.date())
    print(f"Window: {start} .. {end}")
    odds = load_market_odds()
    print(f"Market odds rows (our leagues): {len(odds)}")
    live = load_live_scores()
    print(f"Live-score entries (API-Football): {len(live)}")

    predictions = []
    for lid, (stem, name, code, tz_name) in LEAGUES.items():
        print(f"[{name}] {stem}")
        raw_seasons = [(load_season(stem, s), w) for s, w in SEASONS]
        current = raw_seasons[0][0]
        xg_weight = XG_WEIGHT_BY_LEAGUE.get(name, 0.0)
        div = DIV_BY_LEAGUE.get(name)
        xg_seasons = None
        if xg_weight and div:
            # openfootball's own raw names carry a club suffix clean_name()
            # strips ("Manchester City FC", "AFC Bournemouth") - LeagueModel
            # is trained and queried on those raw forms, so translate
            # football-data.co.uk's short names (via to_pretty, which lands
            # on the same clean display form clean_name() produces) back to
            # whichever raw spelling this league's own fixtures actually use.
            raw_names = {m[side] for matches, _ in raw_seasons for m in matches
                        for side in ("team1", "team2")}
            clean_to_raw = {clean_name(r): r for r in raw_names}
            xg_seasons = xg_seasons_for(
                div, [(s, w) for s, w in SEASONS],
                rename=lambda n: clean_to_raw.get(to_pretty(name, n), to_pretty(name, n)))
        model = LeagueModel([(_of_matches(matches), w) for matches, w in raw_seasons],
                            fit_rho_=False, default_rho=_league_rho(name),
                            xg_seasons=xg_seasons, xg_weight=xg_weight)

        count = dropped = 0
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
            home, away = clean_name(m["team1"]), clean_name(m["team2"])
            live_hit = find_live_match(live, name, home, away, match_day)
            if live_hit and live_hit.get("finished"):
                dropped += 1
                continue
            count += 1
            pred = of_predict(model, m["team1"], m["team2"])
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
            if live_hit:
                row["live"] = {"status": live_hit.get("status"), "score": live_hit.get("score")}
            mk = odds.get((name, to_fd(name, home), to_fd(name, away)))
            if mk:
                edge = None
                if mk["o25_implied"] is not None:
                    edge = round(pred["p_over_2_5"] - mk["o25_implied"], 4)
                row["market"] = {**mk, "edge25": edge}
            predictions.append(row)
        print(f"  {count} upcoming fixtures"
              f"{f' · {dropped} dropped (already finished)' if dropped else ''}")

    predictions += fd_predictions(now, start, end, odds, live)

    predictions.sort(key=lambda x: x["kickoff_utc"])
    OUTPUT_FILE.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(predictions)} fixtures across "
          f"{len(LEAGUES) + len(FD_LEAGUES)} leagues")


if __name__ == "__main__":
    main()
