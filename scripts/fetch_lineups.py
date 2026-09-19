"""Final step of the missing-key-player feature (see scripts/
fetch_key_players.py for the key-player list and scripts/fetch_injuries.py
for the earlier, days-ahead injury/suspension estimate this supersedes):
shortly before kickoff, check the fixture's ACTUAL starting XI (ESPN, the
same free source scripts/fetch_live_scores.py already uses) against
data/key-players.json, and damp that team's lambda if a listed key player
isn't in it.

Always recomputed from base_lam_home/base_lam_away/base_rho - the pristine,
never-adjusted model output update_predictions.py stores on every row - not
from whatever fetch_injuries.py may have already written into lam_home/
lam_away. That's what keeps this a REPLACEMENT of the injury-based estimate
once the real lineup is known, not a second damping stacked on top of it:
an injured player who's actually back in the XI reverts fully, a healthy
player rested for rotation gets caught fresh, and a player who really is
still out just gets re-confirmed - all from the same starting point either
way.

Why this can't run at prediction-build time: an official starting XI is
only published roughly 60-75 minutes before kickoff, while
update_predictions.py's forecast window is the next 10 days - there is no
lineup yet for almost every fixture in predictions.json at build time. This
script instead runs every hour (same cron as the rest of the pipeline) and
only ever finds something to do for fixtures kicking off within
LOOKAHEAD_MINUTES - most hourly runs touch zero fixtures, same as
fetch_live_scores.py mostly doing nothing outside actual match windows.

Standard library only.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teams import to_fd  # noqa: E402
from live_scores import norm as norm_team  # noqa: E402
from goals_model import key_player_damping, dc_score_grid  # noqa: E402
from fetch_live_scores import ESPN_SLUG  # noqa: E402
from player_match import key_players_missing_from, name_tokens  # noqa: E402

PREDICTIONS_FILE = Path("predictions.json")
KEY_PLAYERS_FILE = Path("data/key-players.json")

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/summary"

LOOKAHEAD_MINUTES = 150   # a lineup can drop ~60-75 min pre-kickoff; give the
                          # hourly cron more than one chance to catch it
GRACE_MINUTES = 15        # keep checking a few minutes into the match too,
                          # in case a run lands just before lineups post


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
            starters |= name_tokens(full_name)
        if homeaway in ("home", "away"):
            out[homeaway] = starters
    return out if "home" in out and "away" in out else None


def adjust_row(row, home_missing_share, away_missing_share):
    """Always sets the row's FINAL lam_home/lam_away/exp_goals/p_over_* from
    base_lam_home/base_lam_away/base_rho - the confirmed real lineup is the
    authoritative source once known, so a player fetch_injuries.py flagged
    as out but who actually started must fully revert to the base numbers,
    not just leave the earlier (now stale) injury-based estimate in place.
    Returns whether the real lineup ended up applying any damping at all."""
    base_lh = row.get("base_lam_home", row["lam_home"])
    base_la = row.get("base_lam_away", row["lam_away"])
    base_rho = row.get("base_rho", row["rho"])
    adj = key_player_damping(base_lh, base_la, base_rho, home_missing_share, away_missing_share)
    if adj is None:
        row["lam_home"], row["lam_away"], row["rho"] = base_lh, base_la, base_rho
        row["exp_goals"] = round(base_lh + base_la, 3)
        grid = dc_score_grid(base_lh, base_la, base_rho)

        def over(n):
            return max(0.0, min(1.0, sum(p for (x, y), p in grid.items() if x + y > n)))
        row["p_over_0_5"], row["p_over_1_5"], row["p_over_2_5"] = (
            round(over(0), 4), round(over(1), 4), round(over(2), 4))
        return False
    row.update(adj)
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
    # and idempotent (identical base_lam_home/base_lam_away in, identical
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

        home_missing, home_share = key_players_missing_from(home_kp, xi["home"])
        away_missing, away_share = key_players_missing_from(away_kp, xi["away"])
        row["lineup"] = {"source": "ESPN starting XI",
                          "home_missing_key": home_missing, "away_missing_key": away_missing}
        if adjust_row(row, home_share, away_share):
            adjusted += 1

    PREDICTIONS_FILE.write_text(json.dumps(predictions, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(f"Lineup check: {checked} fixture(s) in window, {adjusted} adjusted for a missing key player")


if __name__ == "__main__":
    main()
