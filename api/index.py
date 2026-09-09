from http.server import BaseHTTPRequestHandler
import json, math, os, urllib.parse, urllib.request

API_URL = "https://v3.football.api-sports.io"
LEAGUES = {
    39: "Premier League",
    140: "LaLiga",
    78: "Bundesliga",
    135: "Serie A",
    61: "Ligue 1",
    88: "Eredivisie",
}


def api_get(path, params):
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY is not configured")
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{API_URL}{path}?{query}",
        headers={"x-apisports-key": key, "User-Agent": "BETAVUS/1.0"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def poisson_tail(lam, threshold):
    if lam < 0:
        lam = 0
    cdf = 0.0
    for k in range(threshold):
        cdf += math.exp(-lam) * lam**k / math.factorial(k)
    return round((1.0 - cdf) * 100, 1)


def fixture_rows(date_from, date_to, league_id=None):
    ids = [int(league_id)] if league_id else list(LEAGUES)
    out = []
    for lid in ids:
        data = api_get("/fixtures", {
            "league": lid,
            "season": 2026,
            "from": date_from,
            "to": date_to,
            "timezone": "Europe/Amsterdam",
        })
        for x in data.get("response", []):
            out.append({
                "fixture_id": x["fixture"]["id"],
                "date": x["fixture"]["date"],
                "league_id": lid,
                "league": LEAGUES.get(lid, str(lid)),
                "home_id": x["teams"]["home"]["id"],
                "home": x["teams"]["home"]["name"],
                "away_id": x["teams"]["away"]["id"],
                "away": x["teams"]["away"]["name"],
            })
    return out


def completed_team_matches(team_id, date_from, date_to):
    data = api_get("/fixtures", {
        "team": team_id,
        "from": date_from,
        "to": date_to,
        "status": "FT",
        "timezone": "Europe/Amsterdam",
    })
    rows = []
    for x in data.get("response", []):
        h = x["teams"]["home"]
        a = x["teams"]["away"]
        hg = x["goals"]["home"]
        ag = x["goals"]["away"]
        if hg is None or ag is None:
            continue
        rows.append({
            "date": x["fixture"]["date"], "home_id": h["id"], "away_id": a["id"],
            "hg": int(hg), "ag": int(ag),
        })
    rows.sort(key=lambda r: r["date"])
    return rows


def weighted_avg(rows, mode=None):
    if not rows:
        return None
    vals = []
    for r in rows:
        if mode == "home" and r["home_id"] != rows[0].get("team_id"):
            pass
        vals.append(r["hg"] + r["ag"])
    n = len(vals)
    weights = list(range(1, n + 1))
    return sum(v*w for v, w in zip(vals, weights)) / sum(weights)


def team_window_avg(team_id, date_from, date_to, venue=None):
    rows = completed_team_matches(team_id, date_from, date_to)
    if venue == "home":
        rows = [r for r in rows if r["home_id"] == team_id]
    elif venue == "away":
        rows = [r for r in rows if r["away_id"] == team_id]
    if not rows:
        return None
    return weighted_avg(rows)


def h2h_avg(home_id, away_id):
    data = api_get("/fixtures/headtohead", {
        "h2h": f"{home_id}-{away_id}", "last": 10,
    })
    vals = []
    for x in data.get("response", []):
        hg = x["goals"]["home"]
        ag = x["goals"]["away"]
        if hg is not None and ag is not None:
            vals.append(int(hg) + int(ag))
    return sum(vals) / len(vals) if vals else None


def league_avg_5y(league_id, end_year=2026):
    season_avgs = []
    for season in range(end_year - 4, end_year + 1):
        data = api_get("/fixtures", {"league": league_id, "season": season, "status": "FT"})
        goals = []
        for x in data.get("response", []):
            hg = x["goals"]["home"]; ag = x["goals"]["away"]
            if hg is not None and ag is not None:
                goals.append(int(hg) + int(ag))
        if goals:
            season_avgs.append(sum(goals) / len(goals))
    return sum(season_avgs) / len(season_avgs) if season_avgs else None


def analyze(f):
    # The 365-day window is expressed by the caller; recent weighting is internal.
    y_from = os.environ.get("ANALYSIS_FROM", "2025-09-09")
    y_to = os.environ.get("ANALYSIS_TO", "2026-09-09")
    h2h = h2h_avg(f["home_id"], f["away_id"])
    home_all = team_window_avg(f["home_id"], y_from, y_to)
    away_all = team_window_avg(f["away_id"], y_from, y_to)
    home_split = team_window_avg(f["home_id"], y_from, y_to, "home")
    away_split = team_window_avg(f["away_id"], y_from, y_to, "away")
    league = league_avg_5y(f["league_id"])
    signals = [v for v in [h2h, home_all, away_all] if v is not None]
    base = sum(signals) / len(signals) if signals else None
    if base is None:
        return {**f, "tg05": None, "tg15": None, "tg25": None}
    splits = [v for v in [home_split, away_split] if v is not None]
    if splits:
        base = 0.70 * base + 0.30 * (sum(splits) / len(splits))
    if league is not None:
        base = 0.65 * base + 0.35 * league
    return {
        **f,
        "lambda": round(base, 3),
        "tg05": poisson_tail(base, 1),
        "tg15": poisson_tail(base, 2),
        "tg25": poisson_tail(base, 3),
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            url = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(url.query)
            if url.path == "/api/health":
                self._send(200, {"ok": True, "service": "BETAVUS backend"})
                return
            if url.path == "/api/fixtures":
                today = qs.get("from", ["2026-09-09"])[0]
                to = qs.get("to", ["2026-09-14"])[0]
                league = qs.get("league", [None])[0]
                rows = fixture_rows(today, to, league)
                self._send(200, {"count": len(rows), "fixtures": rows})
                return
            if url.path == "/api/analyze":
                fid = qs.get("fixture_id", [None])[0]
                if not fid:
                    self._send(400, {"error": "fixture_id is required"})
                    return
                all_rows = fixture_rows(qs.get("from", ["2026-09-09"])[0], qs.get("to", ["2026-09-14"])[0])
                match = next((x for x in all_rows if str(x["fixture_id"]) == fid), None)
                if not match:
                    self._send(404, {"error": "fixture not found"})
                    return
                self._send(200, analyze(match))
                return
            self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": str(e)})
