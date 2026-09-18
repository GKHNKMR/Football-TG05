"""Step 2 of the missing-key-player feature (see scripts/fetch_key_players.py
for step 1): shortly before kickoff, check whether a fixture's actual
starting XI (from ESPN, the same free source scripts/fetch_live_scores.py
already uses) is missing one of data/key-players.json's listed attacking
outlets for that team, and if so damp that team's lambda in predictions.json
and recompute its Over probabilities from the same (rho, Dixon-Coles) this
fixture's prediction was already built with.

Why this can't run at prediction-build time: an official starting XI is
only published roughly 60-75 minutes before kickoff, while
update_predictions.py's forecast window is the next 10 days - there is no
lineup yet for almost every fixture in predictions.json at build time. This
script instead runs every hour (same cron as the rest of the pipeline) and
only ever finds something to do for fixtures kicking off within
LOOKAHEAD_MINUTES - most hourly runs touch zero fixtures, same as
fetch_live_scores.py mostly doing nothing outside actual match windows.

Name matching across three independent providers (Opta's own player names in
key-players.json, ESPN's roster names, nothing standardized between them) is
inherently approximate - this compares normalized surnames, scoped to one
team's own starters, not a global player database. Treat a "missing" flag as
a best-effort signal, not a certainty.

Standard library only.
"""

import json
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teams import to_fd  # noqa: E402
from live_scores import norm as norm_team  # noqa: E402
from goals_model import dc_score_grid  # noqa: E402
from fetch_live_scores import ESPN_SLUG  # noqa: E402

PREDICTIONS_FILE = Path("predictions.json")
KEY_PLAYERS_FILE = Path("data/key-players.json")

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/summary"

LOOKAHEAD_MINUTES = 150   # a lineup can drop ~60-75 min pre-kickoff; give the
                          # hourly cron more than one chance to catch it
GRACE_MINUTES = 15        # keep checking a few minutes into the match too,
                          # in case a run lands just before lineups post
DAMPING_ALPHA = 0.5       # a missing player's output share only partly maps
                          # onto team lambda loss - teammates absorb some of it
MAX_DAMPING = 0.6         # cap combined damping so lambda never collapses to ~0

_JR_ALIASES = {"JR", "JUNIOR", "JNR"}


def _fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z]+", " ", s.upper()).split()


def _surname_key(name):
    toks = _fold(name)
    if not toks:
        return ""
    tok = toks[-1]
    return "JUNIOR" if tok in _JR_ALIASES else tok


def _starter_tokens(full_name):
    """All normalized tokens in an ESPN starter's name (JR-aliased), so a
    key player's surname just needs to appear somewhere in it."""
    toks = {("JUNIOR" if t in _JR_ALIASES else t) for t in _fold(full_name)}
    return toks


def fetch_json(url):
    # ESPN's WAF blocks an explicit UA override (403) but is fine with none at
    # all - same finding as fetch_live_scores.py's fetch_league_day.
    req = Request(url)
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def find_event_id(league, slug, day_iso, home, away):
    try:
        data = fetch_json(f"{SCOREBOARD_URL.format(slug=slug)}?dates={day_iso.replace('-', '')}")
    except (HTTPError, URLError, json.JSONDecodeError) as exc:
        print(f"  scoreboard fetch failed [{league} {day_iso}]: {exc}")
        return None
    h_n, a_n = norm_team(home), norm_team(away)
    for event in data.get("events", []):
        comp = (event.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", [])
        by_side = {c.get("homeAway"): c for c in competitors}
        eh, ea = by_side.get("home"), by_side.get("away")
        if not eh or not ea:
            continue
        eh_n = norm_team(eh["team"]["displayName"])
        ea_n = norm_team(ea["team"]["displayName"])
        if (eh_n in h_n or h_n in eh_n) and (ea_n in a_n or a_n in ea_n):
            return event.get("id")
    return None


def fetch_starting_xi(slug, event_id):
    """{'home': {surname tokens...}, 'away': {...}}, or None if not posted yet."""
    try:
        data = fetch_json(f"{SUMMARY_URL.format(slug=slug)}?event={event_id}")
    except (HTTPError, URLError, json.JSONDecodeError) as exc:
        print(f"  summary fetch failed [event {event_id}]: {exc}")
        return None
    rosters = data.get("rosters") or []
    if not rosters:
        return None
    out = {}
    for side in rosters:
        homeaway = side.get("homeAway")
        starters = set()
        for entry in side.get("roster", []):
            if not entry.get("starter"):
                continue
            full_name = (entry.get("athlete") or {}).get("fullName", "")
            starters |= _starter_tokens(full_name)
        if homeaway in ("home", "away"):
            out[homeaway] = starters
    return out if "home" in out and "away" in out else None


def missing_key_players(key_players, starter_tokens):
    """key_players: data/key-players.json[fd_team] list. Returns (missing, total_share)."""
    missing = []
    for kp in key_players:
        if _surname_key(kp["player"]) not in starter_tokens:
            missing.append(kp["player"])
    total_share = sum(kp["share"] for kp in key_players if kp["player"] in missing)
    return missing, total_share


def adjust_row(row, home_missing_share, away_missing_share):
    lam_home, lam_away, rho = row["lam_home"], row["lam_away"], row["rho"]
    damp_home = min(MAX_DAMPING, DAMPING_ALPHA * home_missing_share)
    damp_away = min(MAX_DAMPING, DAMPING_ALPHA * away_missing_share)
    if not damp_home and not damp_away:
        return False
    lam_home *= (1 - damp_home)
    lam_away *= (1 - damp_away)
    grid = dc_score_grid(lam_home, lam_away, rho)

    def over(n):
        return max(0.0, min(1.0, sum(p for (x, y), p in grid.items() if x + y > n)))

    row["pre_lineup"] = {"lam_home": row["lam_home"], "lam_away": row["lam_away"],
                          "exp_goals": row["exp_goals"], "p_over_0_5": row["p_over_0_5"],
                          "p_over_1_5": row["p_over_1_5"], "p_over_2_5": row["p_over_2_5"]}
    row["lam_home"], row["lam_away"] = round(lam_home, 3), round(lam_away, 3)
    row["exp_goals"] = round(lam_home + lam_away, 3)
    row["p_over_0_5"] = round(over(0), 4)
    row["p_over_1_5"] = round(over(1), 4)
    row["p_over_2_5"] = round(over(2), 4)
    return True


def main():
    if not PREDICTIONS_FILE.exists():
        print("predictions.json missing, nothing to do")
        return
    predictions = json.loads(PREDICTIONS_FILE.read_text(encoding="utf-8"))
    key_players = json.loads(KEY_PLAYERS_FILE.read_text(encoding="utf-8")) \
        if KEY_PLAYERS_FILE.exists() else {}

    now = datetime.now(timezone.utc)
    lo = now - timedelta(minutes=GRACE_MINUTES)
    hi = now + timedelta(minutes=LOOKAHEAD_MINUTES)

    # predictions.json is regenerated from scratch by update_predictions.py
    # every hourly run (no merge - see that script), so there is no state to
    # persist here across runs; this just re-derives the same adjustment
    # each hour a fixture stays inside the lookahead window, which is cheap
    # and idempotent (identical base lam_home/lam_away in, identical
    # adjustment out) rather than something that needs a "done already" flag.
    checked = adjusted = 0
    for row in predictions:
        slug = ESPN_SLUG.get(row["league"])
        if not slug:
            continue
        try:
            kickoff = datetime.strptime(row["kickoff_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if not (lo <= kickoff <= hi):
            continue

        fd_home = to_fd(row["league"], row["home"])
        fd_away = to_fd(row["league"], row["away"])
        home_kp = key_players.get(fd_home, [])
        away_kp = key_players.get(fd_away, [])
        if not home_kp and not away_kp:
            continue  # nothing this feature could ever flag for this pairing

        checked += 1
        day_iso = kickoff.date().isoformat()
        event_id = find_event_id(row["league"], slug, day_iso, row["home"], row["away"])
        if not event_id:
            continue  # try again next hourly run
        xi = fetch_starting_xi(slug, event_id)
        if not xi:
            continue  # lineup not posted yet - try again next run

        home_missing, home_share = missing_key_players(home_kp, xi["home"])
        away_missing, away_share = missing_key_players(away_kp, xi["away"])
        row["lineup"] = {"source": "ESPN starting XI",
                          "home_missing_key": home_missing, "away_missing_key": away_missing}
        if adjust_row(row, home_share, away_share):
            adjusted += 1

    PREDICTIONS_FILE.write_text(json.dumps(predictions, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(f"Lineup check: {checked} fixture(s) in window, {adjusted} adjusted for a missing key player")


if __name__ == "__main__":
    main()
