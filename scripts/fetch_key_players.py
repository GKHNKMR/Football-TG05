"""Refresh data/key-players.json - each team's top attacking contributors
(goal + chance-creation threat), for eventually weighting a missing-key-
player effect into a fixture's expected goals (see Football Goal Analyst
improvement item A: player-level impact).

Source: the same Opta player-match parquet fetch_opta_xg.py already reads
(see scripts/opta_xg.py's fetch_raw_player_matches), just kept at player
granularity instead of collapsed to a team total. NOT part of the live
pipeline yet - this is step 1 (identify who the key players are). Step 2
(scripts/fetch_lineups.py, not built yet) will fetch each imminent
fixture's actual starting XI from ESPN and cross-reference it against this
file to apply a lambda adjustment when a listed key player doesn't start.

Method: for each team, each player's (xg + xa) per appearance is weighted by
season recency (current season counts full, previous season half - a
transfer window can turn over a third of a squad, so older seasons say much
less about "who plays for this team now" than they do for the team-level
goals model) and by how long ago that specific appearance was (same
exponential match-recency decay idiom as goals_model.py, half-life
RECENCY_HALF_LIFE_MATCHES). A player's weighted output as a share of the
team's total is their "key player score"; the top 3 with enough of a
sample (MIN_MINUTES raw minutes across the window, so a single wondergoal
off the bench can't manufacture a 100% share) and a share above
MIN_SHARE are kept.

Needs `pip install duckdb`.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opta_xg import (  # noqa: E402
    DIV_TO_OPTA, build_matcher, fd_team_names, fetch_raw_player_matches,
)
from xg_blend import to_opta_season  # noqa: E402

OUT_FILE = Path("data/key-players.json")

RECENCY_HALF_LIFE_MATCHES = 6  # same idiom as goals_model.py
SEASON_WEIGHTS = [("2026-27", 1.0), ("2025-26", 0.6)]
MIN_MINUTES = 600     # ~6-7 full matches before a player counts as "known"
MIN_SHARE = 0.12
TOP_N = 3


def main():
    season_w = {to_opta_season(c): w for c, w in SEASON_WEIGHTS}
    rows = fetch_raw_player_matches()
    rows = [r for r in rows if r["season"] in season_w]

    fd_names_by_div = {d: fd_team_names(d) for d in DIV_TO_OPTA}
    matchers = {div: build_matcher(names) for div, names in fd_names_by_div.items()}

    # (div, team_opta, player_name) -> [(date, xg+xa, minutes, season_w), ...]
    by_player = {}
    for r in rows:
        key = (r["div"], r["team_opta"], r["player_name"])
        by_player.setdefault(key, []).append(
            (r["date"], r["xg"] + r["xa"], r["minutes"], season_w[r["season"]])
        )

    decay = math.log(2) / RECENCY_HALF_LIFE_MATCHES
    player_score = {}   # (div, team_opta, player_name) -> (weighted_output, raw_minutes)
    for key, apps in by_player.items():
        apps.sort(key=lambda a: a[0], reverse=True)
        weighted, raw_minutes = 0.0, 0.0
        for rank, (_date, out, minutes, sw) in enumerate(apps):
            weighted += out * sw * math.exp(-decay * rank)
            raw_minutes += minutes
        player_score[key] = (weighted, raw_minutes)

    team_total = {}  # (div, team_opta) -> sum of weighted output
    for (div, team_opta, _player), (weighted, _mins) in player_score.items():
        team_total[(div, team_opta)] = team_total.get((div, team_opta), 0.0) + weighted

    out = {}
    unmatched_teams = set()
    for (div, team_opta, player), (weighted, raw_minutes) in player_score.items():
        if raw_minutes < MIN_MINUTES:
            continue
        total = team_total.get((div, team_opta), 0.0)
        if total <= 0:
            continue
        share = weighted / total
        if share < MIN_SHARE:
            continue
        matcher = matchers.get(div)
        fd_name = matcher(team_opta) if matcher else None
        if not fd_name:
            unmatched_teams.add((div, team_opta))
            continue
        league = out.setdefault(fd_name, [])
        league.append({"player": player, "share": share, "minutes": raw_minutes})

    for fd_name, players in out.items():
        # Opta's own team_name sometimes has more than one string variant
        # for the same club across seasons/competitions (both resolving to
        # this fd_name via the matcher above) - a player who appears under
        # both gets computed as two independent, differently-denominated
        # fragments; merge same-name entries back into one before ranking
        # rather than showing a real player twice under two different shares.
        merged = {}
        for p in players:
            e = merged.setdefault(p["player"], {"share": 0.0, "minutes": 0})
            e["share"] += p["share"]
            e["minutes"] += p["minutes"]
        combined = [{"player": name, "share": round(e["share"], 3), "minutes": round(e["minutes"])}
                    for name, e in merged.items()]
        combined.sort(key=lambda p: p["share"], reverse=True)
        out[fd_name] = combined[:TOP_N]

    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True),
                        encoding="utf-8")
    n_flagged = sum(len(v) for v in out.values())
    print(f"Key players: {n_flagged} flagged across {len(out)} teams")
    if unmatched_teams:
        print(f"  {len(unmatched_teams)} teams unresolved (older/renamed club names): "
              f"{sorted(unmatched_teams)[:10]}")


if __name__ == "__main__":
    main()
