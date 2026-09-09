"""Fold the football-data.co.uk CSVs into data/match-stats.json.

For every fixture in predictions.json this writes, keyed by match_id:
  - each team's current-season aggregate (goals, over/BTTS/clean-sheet rates)
  - each team's last 8 completed matches (form)
  - each team's goal averages for the last 5 completed seasons (goals5, for the
    per-match chart in the drawer)
  - the head-to-head record between the two teams

The dashboard fetches this file and shows it when a match row is clicked.
Raw CSVs live under data/football-data/<DIV>/<SEASON>.csv (see
scripts/fetch_football_data.py) and stay the local stats database.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

CSV_DIR = Path("data/football-data")
PRED_FILE = Path("predictions.json")
OUT_FILE = Path("data/match-stats.json")

# newest last so a plain sort by (date) keeps seasons in order
SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
CURRENT_SEASON = "2627"
# last five completed seasons, oldest -> newest, for the goal-average chart
CHART_SEASONS = ["2122", "2223", "2324", "2425", "2526"]
FORM_N = 8
H2H_N = 10


def season_label(code):
    return f"{code[:2]}/{code[2:]}"

DIVISIONS = {
    "E0": ("Premier League", 39),
    "SP1": ("LaLiga", 140),
    "D1": ("Bundesliga", 78),
    "I1": ("Serie A", 135),
    "F1": ("Ligue 1", 61),
    "N1": ("Eredivisie", 88),
}

# BETAVUS display name (from predictions.json) -> football-data.co.uk name.
# Only the ones that differ; identical names fall through untouched.
CROSSWALK = {
    "Premier League": {
        "Brighton & Hove Albion": "Brighton", "Coventry City": "Coventry",
        "Hull City": "Hull", "Ipswich Town": "Ipswich", "Leeds United": "Leeds",
        "Manchester City": "Man City", "Manchester United": "Man United",
        "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
        "Tottenham Hotspur": "Tottenham",
    },
    "LaLiga": {
        "Athletic Bilbao": "Ath Bilbao", "Atlético Madrid": "Ath Madrid",
        "Alavés": "Alaves", "Celta Vigo": "Celta", "Deportivo La Coruña": "La Coruna",
        "Espanyol": "Espanol", "Málaga": "Malaga", "Racing Santander": "Santander",
        "Rayo Vallecano": "Vallecano", "Real Betis": "Betis", "Real Sociedad": "Sociedad",
        "Osasuna": "Osasuna",
    },
    "Bundesliga": {
        "Bayern München": "Bayern Munich", "Mönchengladbach": "M'gladbach",
        "Borussia Dortmund": "Dortmund", "Eintracht Frankfurt": "Ein Frankfurt",
        "Köln": "FC Koln", "Hamburger SV": "Hamburg", "Mainz 05": "Mainz",
    },
    "Serie A": {},
    "Ligue 1": {"Paris Saint-Germain": "Paris SG"},
    "Eredivisie": {
        "ADO Den Haag": "Den Haag", "AZ": "AZ Alkmaar", "Fortuna Sittard": "For Sittard",
        "NEC": "Nijmegen", "PEC Zwolle": "Zwolle", "PSV": "PSV Eindhoven",
    },
}


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_division(div):
    """Chronological list of completed matches for one division."""
    out = []
    for season in SEASONS:
        path = CSV_DIR / div / f"{season}.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                hg, ag = num(row.get("FTHG")), num(row.get("FTAG"))
                if hg is None or ag is None:
                    continue
                try:
                    date = datetime.strptime(row["Date"].strip(), "%d/%m/%Y").date()
                except (ValueError, KeyError):
                    continue
                hg, ag = int(hg), int(ag)
                hhg, hag = num(row.get("HTHG")), num(row.get("HTAG"))
                out.append(
                    {
                        "season": season,
                        "date": date.isoformat(),
                        "home": row["HomeTeam"].strip(),
                        "away": row["AwayTeam"].strip(),
                        "fthg": hg, "ftag": ag,
                        "hthg": int(hhg) if hhg is not None else None,
                        "htag": int(hag) if hag is not None else None,
                        "total": hg + ag,
                        "hs": num(row.get("HS")), "as": num(row.get("AS")),
                        "hst": num(row.get("HST")), "ast": num(row.get("AST")),
                        "hc": num(row.get("HC")), "ac": num(row.get("AC")),
                        "referee": (row.get("Referee") or "").strip() or None,
                        "odds": {
                            "h": num(row.get("AvgH")), "d": num(row.get("AvgD")),
                            "a": num(row.get("AvgA")),
                            "o25": num(row.get("Avg>2.5")), "u25": num(row.get("Avg<2.5")),
                            "ah_line": num(row.get("AHh")),
                            "ah_home": num(row.get("AvgAHH")), "ah_away": num(row.get("AvgAHA")),
                        },
                    }
                )
    out.sort(key=lambda m: m["date"])
    return out


def team_matches(matches, team):
    return [m for m in matches if m["home"] == team or m["away"] == team]


def perspective(m, team):
    home = m["home"] == team
    gf, ga = (m["fthg"], m["ftag"]) if home else (m["ftag"], m["fthg"])
    return {
        "date": m["date"],
        "season": m["season"],
        "venue": "H" if home else "A",
        "opp": m["away"] if home else m["home"],
        "gf": gf, "ga": ga,
        "res": "W" if gf > ga else "L" if gf < ga else "D",
        "total": m["total"],
        "over25": m["total"] > 2.5,
        "btts": m["fthg"] > 0 and m["ftag"] > 0,
        "ht": (
            f"{m['hthg']}-{m['htag']}"
            if m["hthg"] is not None and m["htag"] is not None
            else None
        ),
    }


def pct(count, total):
    return round(100 * count / total, 1) if total else None


def season_summary(matches, team, season):
    rows = [perspective(m, team) for m in matches
            if (m["home"] == team or m["away"] == team) and m["season"] == season]
    n = len(rows)
    if not n:
        return {"played": 0}
    gf = sum(r["gf"] for r in rows)
    ga = sum(r["ga"] for r in rows)
    return {
        "played": n,
        "w": sum(r["res"] == "W" for r in rows),
        "d": sum(r["res"] == "D" for r in rows),
        "l": sum(r["res"] == "L" for r in rows),
        "gf": gf, "ga": ga,
        "gf_avg": round(gf / n, 2), "ga_avg": round(ga / n, 2),
        "over05_pct": pct(sum(r["total"] > 0.5 for r in rows), n),
        "over15_pct": pct(sum(r["total"] > 1.5 for r in rows), n),
        "over25_pct": pct(sum(r["over25"] for r in rows), n),
        "btts_pct": pct(sum(r["btts"] for r in rows), n),
        "cs_pct": pct(sum(r["ga"] == 0 for r in rows), n),
        "fts_pct": pct(sum(r["gf"] == 0 for r in rows), n),
    }


def goals_by_season(matches, team):
    """Per-match goal averages for each of the last five completed seasons."""
    out = []
    for code in CHART_SEASONS:
        rows = [
            perspective(m, team)
            for m in matches
            if m["season"] == code and (m["home"] == team or m["away"] == team)
        ]
        n = len(rows)
        if n < 5:  # promoted/relegated - not a top-flight season, leave a gap
            out.append({"season": code, "label": season_label(code), "played": n})
            continue
        gf = sum(r["gf"] for r in rows)
        ga = sum(r["ga"] for r in rows)
        out.append(
            {
                "season": code,
                "label": season_label(code),
                "played": n,
                "gf_avg": round(gf / n, 2),
                "ga_avg": round(ga / n, 2),
                "total_avg": round((gf + ga) / n, 2),
            }
        )
    return out


def head_to_head(matches, home_fd, away_fd):
    pair = [m for m in matches if {m["home"], m["away"]} == {home_fd, away_fd}]
    pair.sort(key=lambda m: m["date"], reverse=True)
    recent = pair[:H2H_N]
    n = len(recent)
    if not n:
        return {"count": 0, "matches": []}
    hw = sum(
        (m["fthg"] > m["ftag"] and m["home"] == home_fd)
        or (m["ftag"] > m["fthg"] and m["away"] == home_fd)
        for m in recent
    )
    aw = sum(
        (m["fthg"] > m["ftag"] and m["home"] == away_fd)
        or (m["ftag"] > m["fthg"] and m["away"] == away_fd)
        for m in recent
    )
    return {
        "count": n,
        "home_wins": hw,
        "away_wins": aw,
        "draws": n - hw - aw,
        "avg_total": round(sum(m["total"] for m in recent) / n, 2),
        "over25_pct": pct(sum(m["total"] > 2.5 for m in recent), n),
        "btts_pct": pct(sum(m["fthg"] > 0 and m["ftag"] > 0 for m in recent), n),
        "matches": [
            {
                "date": m["date"], "home": m["home"], "away": m["away"],
                "score": f"{m['fthg']}-{m['ftag']}",
                "ht": (
                    f"{m['hthg']}-{m['htag']}"
                    if m["hthg"] is not None else None
                ),
                "total": m["total"],
            }
            for m in recent
        ],
    }


def main():
    predictions = json.loads(PRED_FILE.read_text(encoding="utf-8"))
    by_league = {}
    for div, (league, _lid) in DIVISIONS.items():
        by_league[league] = load_division(div)
        print(f"{league:15} {len(by_league[league])} completed matches")

    out_matches = {}
    unmatched = set()
    for p in predictions:
        league = p["league"]
        matches = by_league.get(league, [])
        cw = CROSSWALK.get(league, {})
        pretty = {v: k for k, v in cw.items()}  # fd name -> BETAVUS name
        team_names = {m["home"] for m in matches} | {m["away"] for m in matches}

        def resolve(name):
            fd = cw.get(name, name)
            if fd not in team_names:
                unmatched.add(f"{league}: {name!r} -> {fd!r}")
                return fd, False
            return fd, True

        home_fd, home_ok = resolve(p["home"])
        away_fd, away_ok = resolve(p["away"])

        def side(name, fd, ok):
            rows = team_matches(matches, fd) if ok else []
            form = [perspective(m, fd) for m in rows][-FORM_N:][::-1]
            for f in form:
                f["opp"] = pretty.get(f["opp"], f["opp"])
            return {
                "name": name,
                "fd": fd,
                "matched": ok,
                "season": season_summary(matches, fd, CURRENT_SEASON) if ok else {"played": 0},
                "goals5": goals_by_season(matches, fd) if ok else [],
                "form": form,
            }

        h2h = head_to_head(matches, home_fd, away_fd) if home_ok and away_ok else {"count": 0, "matches": []}
        for m in h2h.get("matches", []):
            m["home"] = pretty.get(m["home"], m["home"])
            m["away"] = pretty.get(m["away"], m["away"])

        out_matches[p["match_id"]] = {
            "league": league,
            "kickoff_utc": p["kickoff_utc"],
            "home": side(p["home"], home_fd, home_ok),
            "away": side(p["away"], away_fd, away_ok),
            "h2h": h2h,
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "football-data.co.uk",
        "seasons": SEASONS,
        "current_season": CURRENT_SEASON,
        "matches": out_matches,
    }
    OUT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(f"Wrote {OUT_FILE} for {len(out_matches)} fixtures")
    if unmatched:
        print("UNMATCHED team names (no CSV history):")
        for u in sorted(unmatched):
            print("  " + u)


if __name__ == "__main__":
    main()
