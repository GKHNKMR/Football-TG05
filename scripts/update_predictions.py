import csv
import io
import json
import math
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUTPUT_FILE = Path("predictions.json")
CACHE_DIR = Path("data/cache/football_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Football-Data.co.uk publishes free, machine-readable results and current fixtures.
# Season codes are the starting year of the football season, e.g. 2526 = 2025/26.
LEAGUES = {
    "E0": "Premier League",
    "SP1": "LaLiga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
}
HIST_SEASONS = ["2122", "2223", "2324", "2425", "2526"]
CURRENT_SEASON = "2627"
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"


def fetch_csv(code, season):
    path = CACHE_DIR / f"{season}_{code}.csv"
    if path.exists():
        text = path.read_text(encoding="latin-1")
    else:
        url = BASE_URL.format(season=season, code=code)
        req = urllib.request.Request(url, headers={"User-Agent": "BETAVUS/1.0"})
        with urllib.request.urlopen(req, timeout=45) as response:
            text = response.read().decode("latin-1")
        path.write_text(text, encoding="latin-1")
    return list(csv.DictReader(io.StringIO(text)))


def parse_date(value):
    if not value:
        return None
    value = value.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def goals(row):
    try:
        return int(float(row["FTHG"])), int(float(row["FTAG"]))
    except (KeyError, TypeError, ValueError):
        return None


def completed(row):
    return goals(row) is not None


def weighted_mean(values):
    if not values:
        return None
    values = values[:5]
    weights = list(range(len(values), 0, -1))
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def simple_mean(values):
    return sum(values) / len(values) if values else None


def poisson_tail(lam, threshold):
    lam = max(0.05, float(lam))
    cutoff = int(threshold - 0.5)
    p = math.exp(-lam)
    cumulative = p
    for k in range(1, cutoff + 1):
        p *= lam / k
        cumulative += p
    return 1.0 - cumulative


def label(p05):
    if p05 >= 0.95:
        return "ULTRA"
    if p05 >= 0.90:
        return "HIGH"
    if p05 >= 0.85:
        return "MEDIUM"
    return ""


def normalize_team(name):
    aliases = {
        "Man United": "Manchester United",
        "Man City": "Manchester City",
        "Spurs": "Tottenham Hotspur",
        "Nott'm Forest": "Nottingham Forest",
        "Wolves": "Wolverhampton Wanderers",
        "Leicester": "Leicester City",
        "West Ham": "West Ham United",
        "Newcastle": "Newcastle United",
        "Brighton": "Brighton & Hove Albion",
        "QPR": "Queens Park Rangers",
        "PSG": "Paris Saint-Germain",
        "Paris SG": "Paris Saint-Germain",
        "AC Milan": "Milan",
        "Inter": "Internazionale",
        "Bayern Munich": "Bayern München",
        "M'gladbach": "Borussia Mönchengladbach",
        "FC Koln": "Cologne",
        "Köln": "Cologne",
    }
    return aliases.get(name.strip(), name.strip())


def build_stats(rows, history_from, now):
    team_all = defaultdict(list)
    team_home = defaultdict(list)
    team_away = defaultdict(list)
    league_goals = 0
    league_matches = 0
    h2h = defaultdict(list)

    for row in rows:
        if not completed(row):
            continue
        dt = parse_date(row.get("Date"))
        ga = goals(row)
        if not dt or not ga:
            continue
        hg, ag = ga
        total = hg + ag
        league_goals += total
        league_matches += 1
        home = normalize_team(row.get("HomeTeam", ""))
        away = normalize_team(row.get("AwayTeam", ""))

        if history_from <= dt <= now:
            team_all[home].append((dt, total))
            team_all[away].append((dt, total))
            team_home[home].append((dt, total))
            team_away[away].append((dt, total))

        pair = tuple(sorted((home, away)))
        h2h[pair].append((dt, total))

    for d in (team_all, team_home, team_away):
        for team in d:
            d[team].sort(key=lambda x: x[0], reverse=True)
            d[team] = [v for _, v in d[team]]
    for pair in h2h:
        h2h[pair].sort(key=lambda x: x[0], reverse=True)

    league_avg = league_goals / league_matches if league_matches else None
    return team_all, team_home, team_away, h2h, league_avg


def expected_total(home, away, stats):
    team_all, team_home, team_away, h2h, league_avg = stats

    candidates = []
    if team_all.get(home):
        candidates.append(weighted_mean(team_all[home]))
    if team_all.get(away):
        candidates.append(weighted_mean(team_all[away]))
    base = simple_mean(candidates)

    split = []
    if team_home.get(home):
        split.append(weighted_mean(team_home[home]))
    if team_away.get(away):
        split.append(weighted_mean(team_away[away]))
    if split:
        split_avg = simple_mean(split)
        base = split_avg if base is None else 0.70 * base + 0.30 * split_avg

    pair = tuple(sorted((home, away)))
    h2h_vals = [v for _, v in h2h.get(pair, [])[:10]]
    h2h_avg = simple_mean(h2h_vals)
    if h2h_avg is not None:
        base = h2h_avg if base is None else 0.55 * base + 0.45 * h2h_avg

    if league_avg is not None:
        base = league_avg if base is None else 0.65 * base + 0.35 * league_avg

    return max(0.25, min(5.5, base if base is not None else 2.5))


def main():
    now = datetime.now(timezone.utc)
    today = now.date()
    end_date = today + timedelta(days=7)
    history_from = now - timedelta(days=365)
    output = []

    for code, league_name in LEAGUES.items():
        historical = []
        for season in HIST_SEASONS:
            try:
                historical.extend(fetch_csv(code, season))
            except Exception as exc:
                print(f"Historical CSV unavailable {code} {season}: {exc}")

        stats = build_stats(historical, history_from, now)

        try:
            current_rows = fetch_csv(code, CURRENT_SEASON)
        except Exception as exc:
            print(f"Current CSV unavailable {code}: {exc}")
            current_rows = []

        fixtures = []
        for row in current_rows:
            dt = parse_date(row.get("Date"))
            if not dt or not (today <= dt.date() <= end_date):
                continue
            home = normalize_team(row.get("HomeTeam", ""))
            away = normalize_team(row.get("AwayTeam", ""))
            if not home or not away:
                continue
            fixtures.append((dt, home, away))

        print(f"{league_name}: {len(fixtures)} fixtures in dashboard window")
        for dt, home, away in fixtures:
            lam = expected_total(home, away, stats)
            p05 = poisson_tail(lam, 0.5)
            output.append({
                "match_id": f"{code}-{dt.strftime('%Y%m%d')}-{home}-{away}",
                "league": league_name,
                "kickoff_utc": dt.isoformat().replace("+00:00", "Z"),
                "home": home,
                "away": away,
                "p_over_0_5": round(p05, 4),
                "p_over_1_5": round(poisson_tail(lam, 1.5), 4),
                "p_over_2_5": round(poisson_tail(lam, 2.5), 4),
                "lambda_total": round(lam, 3),
                "label": label(p05),
                "updated_at": now.isoformat(),
            })

    output.sort(key=lambda x: x["kickoff_utc"])
    OUTPUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(output)} predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
