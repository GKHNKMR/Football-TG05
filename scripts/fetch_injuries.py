"""Earlier, days-ahead half of the missing-key-player feature (see
scripts/fetch_key_players.py for the key-player list and scripts/
fetch_lineups.py for the later, ~60-75-min-pre-kickoff confirmation this
gets superseded by): for every fixture in predictions.json whose home or
away side has a listed key player, check Transfermarkt's injury/suspension
list (scripts/injury_data.py) for that club and damp the fixture's lambda
if one is currently out.

Runs across the WHOLE 10-day prediction window, every hourly cycle - unlike
fetch_lineups.py, there's no "too early" here: a club's injury list is
already public knowledge days or weeks before a given fixture, which is the
whole reason this exists as a separate, earlier step. It's necessarily a
noisier signal than the real lineup (an "injured" listing doesn't always
mean out for THIS specific match, and Transfermarkt's return-date estimates
are exactly that - estimates), so fetch_lineups.py, once the real XI is
known, always overwrites it rather than stacking on top - see that script's
docstring.

Standard library only (injury_data.py's Wikidata calls are stdlib; only
scripts/fetch_key_players.py's own upstream data needs duckdb, already
built by the time this runs).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teams import to_fd  # noqa: E402
from goals_model import key_player_damping  # noqa: E402
from injury_data import transfermarkt_id, fetch_absences  # noqa: E402
from player_match import key_players_listed_in  # noqa: E402

PREDICTIONS_FILE = Path("predictions.json")
KEY_PLAYERS_FILE = Path("data/key-players.json")


def teams_with_a_key_player(predictions, key_players):
    """{(league, fd_team_name), ...} - only clubs worth spending a
    Transfermarkt request on, i.e. ones with something to flag at all."""
    needed = set()
    for row in predictions:
        league = row["league"]
        for side in ("home", "away"):
            fd = to_fd(league, row[side])
            if key_players.get(fd):
                needed.add((league, fd))
    return needed


def adjust_row(row, home_missing_share, away_missing_share):
    base_lh = row.get("base_lam_home", row["lam_home"])
    base_la = row.get("base_lam_away", row["lam_away"])
    base_rho = row.get("base_rho", row["rho"])
    adj = key_player_damping(base_lh, base_la, base_rho, home_missing_share, away_missing_share)
    if adj is None:
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

    needed = teams_with_a_key_player(predictions, key_players)
    absences_by_team = {}
    for league, fd_name in needed:
        tm_id = transfermarkt_id(league, fd_name)
        absences_by_team[(league, fd_name)] = fetch_absences(tm_id) if tm_id else []

    checked = adjusted = 0
    for row in predictions:
        league = row["league"]
        fd_home, fd_away = to_fd(league, row["home"]), to_fd(league, row["away"])
        home_kp, away_kp = key_players.get(fd_home, []), key_players.get(fd_away, [])
        if not home_kp and not away_kp:
            continue
        checked += 1

        home_absent = [a["player"] for a in absences_by_team.get((league, fd_home), [])]
        away_absent = [a["player"] for a in absences_by_team.get((league, fd_away), [])]
        home_missing, home_share = key_players_listed_in(home_kp, home_absent)
        away_missing, away_share = key_players_listed_in(away_kp, away_absent)
        if not home_missing and not away_missing:
            continue

        row["injury"] = {"source": "Transfermarkt",
                          "home_missing_key": home_missing, "away_missing_key": away_missing}
        if adjust_row(row, home_share, away_share):
            adjusted += 1

    PREDICTIONS_FILE.write_text(json.dumps(predictions, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(f"Injury check: {len(needed)} club(s) queried, {checked} fixture(s) with a listed key player, "
          f"{adjusted} adjusted for a missing key player")


if __name__ == "__main__":
    main()
