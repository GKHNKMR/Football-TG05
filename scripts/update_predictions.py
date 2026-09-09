import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BASE_URL = "https://v3.football.api-sports.io"
API_KEY = os.environ.get("API_FOOTBALL_KEY", "").strip()
CACHE_DIR = Path("data/cache")
HISTORY_DIR = CACHE_DIR / "history"
H2H_FILE = CACHE_DIR / "h2h.json"
OUTPUT_FILE = Path("predictions.json")

LEAGUES = {
    39: "Premier League",
    140: "LaLiga",
    78: "Bundesliga",
    135: "Serie A",
    61: "Ligue 1",
    88: "Eredivisie",
}
SEASONS = [2022, 2023, 2024, 2025, 2026]
MIN_SECONDS_BETWEEN_CALLS = 6.2  # Free plan: 10 requests/minute.
_last_request_at = 0.0


def api_get(path, params):
    global _last_request_at
    if not API_KEY:
        raise RuntimeError("API_FOOTBALL_KEY is missing")
    query = urlencode(params)
    req = Request(f"{BASE_URL}{path}?{query}", headers={"x-apisports-key": API_KEY})
    for attempt in range(3):
        wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        try:
            with urlopen(req, timeout=45) as response:
                _last_request_at = time.monotonic()
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("errors"):
                errors = payload["errors"]
                if "rateLimit" in str(errors):
                    time.sleep(65)
                    continue
                raise RuntimeError(str(errors))
            return payload.get("response", [])
        except HTTPError as exc:
            _last_request_at = time.monotonic()
            if exc.code == 429:
                print("API rate limit reached; waiting 65 seconds...")
                time.sleep(65)
                continue
            if attempt == 2:
                raise
            time.sleep(5)
        except Exception:
            _last_request_at = time.monotonic()
            if attempt == 2:
                raise
            time.sleep(5)
    raise RuntimeError("API request failed after retries")


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fixture_date(f):
    return datetime.fromisoformat(f["fixture"]["date"].replace("Z", "+00:00"))


def completed(f):
    return f.get("fixture", {}).get("status", {}).get("short") in {"FT", "AET", "PEN"}


def goals(f):
    h = f.get("goals", {}).get("home")
    a = f.get("goals", {}).get("away")
    if h is None or a is None:
        return None
    return int(h), int(a)


def weighted_mean(values):
    if not values:
        return None
    values = values[:5]
    weights = list(range(len(values), 0, -1))
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


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


def season_fixtures(league_id, season):
    path = HISTORY_DIR / f"{league_id}_{season}.json"
    cached = load_json(path, None)
    if cached is not None:
        return cached
    try:
        data = api_get("/fixtures", {"league": league_id, "season": season})
        save_json(path, data)
        return data
    except Exception as exc:
        print(f"History unavailable league={league_id} season={season}: {exc}")
        return []


def h2h_matches(home_id, away_id, cache):
    key = f"{min(home_id, away_id)}-{max(home_id, away_id)}"
    if key in cache:
        return cache[key]
    try:
        data = api_get("/fixtures/headtohead", {"h2h": f"{home_id}-{away_id}", "last": 10})
        result = [f for f in data if completed(f) and goals(f) is not None]
        cache[key] = result[:10]
        return cache[key]
    except Exception as exc:
        print(f"H2H unavailable {key}: {exc}")
        cache[key] = []
        return []


def build_stats(all_fixtures, start_dt, end_dt):
    team_all = defaultdict(list)
    team_home = defaultdict(list)
    team_away = defaultdict(list)
    league_totals = []
    for f in all_fixtures:
        if not completed(f):
            continue
        dt = fixture_date(f)
        ga = goals(f)
        if ga is None:
            continue
        hg, ag = ga
        league_totals.append(hg + ag)
        home_id = f["teams"]["home"]["id"]
        away_id = f["teams"]["away"]["id"]
        total = hg + ag
        if start_dt <= dt <= end_dt:
            team_all[home_id].append((dt, total))
            team_all[away_id].append((dt, total))
            team_home[home_id].append((dt, total))
            team_away[away_id].append((dt, total))
    for d in (team_all, team_home, team_away):
        for team_id in d:
            d[team_id].sort(key=lambda x: x[0], reverse=True)
            d[team_id] = [v for _, v in d[team_id]]
    return team_all, team_home, team_away, league_totals


def h2h_average(matches):
    vals = []
    for f in sorted(matches, key=fixture_date, reverse=True)[:10]:
        ga = goals(f)
        if ga:
            vals.append(sum(ga))
    return weighted_mean(vals)


def league_average(history_by_league):
    season_means = []
    for fixtures in history_by_league:
        vals = [sum(goals(f)) for f in fixtures if completed(f) and goals(f)]
        if vals:
            season_means.append(sum(vals) / len(vals))
    return sum(season_means) / len(season_means) if season_means else None


def expected_total(f, stats, h2h_avg, league_avg):
    team_all, team_home, team_away, _ = stats
    home_id = f["teams"]["home"]["id"]
    away_id = f["teams"]["away"]["id"]
    candidates = []
    if team_all.get(home_id):
        candidates.append(weighted_mean(team_all[home_id]))
    if team_all.get(away_id):
        candidates.append(weighted_mean(team_all[away_id]))
    base = sum(candidates) / len(candidates) if candidates else None
    split = []
    if team_home.get(home_id):
        split.append(weighted_mean(team_home[home_id]))
    if team_away.get(away_id):
        split.append(weighted_mean(team_away[away_id]))
    if split:
        split_avg = sum(split) / len(split)
        base = split_avg if base is None else 0.70 * base + 0.30 * split_avg
    if h2h_avg is not None:
        base = h2h_avg if base is None else 0.55 * base + 0.45 * h2h_avg
    if league_avg is not None:
        base = league_avg if base is None else 0.65 * base + 0.35 * league_avg
    return max(0.25, min(5.5, base if base is not None else 2.5))


def main():
    if not API_KEY:
        raise SystemExit("API_FOOTBALL_KEY is required")

    now = datetime.now(timezone.utc)
    today = now.date()
    end_date = today + timedelta(days=7)
    history_from = now - timedelta(days=365)

    stats_by_league = {}
    league_baselines = {}
    season_data = {}

    # One season call is enough to build both the historical statistics and
    # the current fixture list. This keeps the first daily run below the
    # Free-plan 100-request quota even when H2H is empty.
    for league_id in LEAGUES:
        seasons = []
        for season in SEASONS:
            data = season_fixtures(league_id, season)
            seasons.append(data)
        season_data[league_id] = seasons
        merged = [f for season in seasons for f in season]
        stats_by_league[league_id] = build_stats(merged, history_from, now)
        league_baselines[league_id] = league_average(seasons)

    h2h_cache = load_json(H2H_FILE, {})
    output = []
    for league_id, league_name in LEAGUES.items():
        # The 2026 season response already contains the future fixtures.
        # Do not spend another API call just to retrieve the same matches.
        fixtures = [
            f for f in season_data[league_id][-1]
            if today <= fixture_date(f).date() <= end_date
        ]
        print(f"{league_name}: {len(fixtures)} fixtures in dashboard window")

        stats = stats_by_league[league_id]
        for f in fixtures:
            status = f.get("fixture", {}).get("status", {}).get("short")
            if status in {"CANC", "PST", "ABD", "AWD", "WO"}:
                continue
            home = f["teams"]["home"]
            away = f["teams"]["away"]
            h2h = h2h_matches(home["id"], away["id"], h2h_cache)
            lam = expected_total(f, stats, h2h_average(h2h), league_baselines[league_id])
            p05 = poisson_tail(lam, 0.5)
            output.append({
                "match_id": str(f["fixture"]["id"]),
                "league_id": league_id,
                "league": league_name,
                "kickoff_utc": f["fixture"]["date"],
                "home": home["name"],
                "away": away["name"],
                "p_over_0_5": round(p05, 4),
                "p_over_1_5": round(poisson_tail(lam, 1.5), 4),
                "p_over_2_5": round(poisson_tail(lam, 2.5), 4),
                "lambda_total": round(lam, 3),
                "label": label(p05),
                "updated_at": now.isoformat(),
            })

    output.sort(key=lambda x: x["kickoff_utc"])
    save_json(H2H_FILE, h2h_cache)
    save_json(OUTPUT_FILE, output)
    print(f"Wrote {len(output)} predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
