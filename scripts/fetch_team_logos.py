"""Team crest URLs for the match card (data/team-logos.json).

Source: ESPN's public site API team list per league (same keyless API as
fetch_live_scores.py) - each team carries a logo on a.espncdn.com. Our team
names come from football-data.co.uk (predictions.json + data/match-stats.json)
and are lined up with ESPN's names through live_scores.norm (same aliasing
the live-score matcher uses).

Crests barely change, so a league is only re-fetched when one of its current
teams has no crest yet (a promoted club, a new league) - on a normal hourly
run this makes no requests at all. Existing entries are never dropped, so a
failed fetch can't wipe crests the site already shows.

Output: {"leagues": {league: {our team name: logo url}}, "missing": {...}}
"""

import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_live_scores import ESPN_SLUG  # noqa: E402
from live_scores import norm  # noqa: E402

OUT = Path("data/team-logos.json")
TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/teams"
# Names the fuzzy match can't settle: too short to substring-match ("AZ"), or
# ambiguous ("Sporting Braga" also contains Sporting CP's "Sporting").
ESPN_NAME = {"AZ": "AZ Alkmaar", "Sporting Braga": "Braga"}
# Crests ESPN doesn't have, kept in the repo instead (always win over ESPN).
# Amedspor: tr.wikipedia "Dosya:Amed_SK.png", resized to 128 px.
MANUAL = {"Turkish Süper Lig": {"Amedspor": "brand/crests/amedspor.png"}}


def our_teams():
    """{league: set(team names)} from the fixtures and match stats we show."""
    out = {}

    def add(league, *names):
        for n in names:
            if league and n:
                out.setdefault(league, set()).add(n)

    try:
        preds = json.loads(Path("predictions.json").read_text(encoding="utf-8"))
        rows = preds if isinstance(preds, list) else preds.get("predictions") or preds.get("matches") or []
        for p in rows:
            add(p.get("league"), p.get("home"), p.get("away"))
    except (OSError, json.JSONDecodeError):
        pass
    try:
        ms = json.loads(Path("data/match-stats.json").read_text(encoding="utf-8"))
        for m in (ms.get("matches") or {}).values():
            add(m.get("league"), (m.get("home") or {}).get("name"), (m.get("away") or {}).get("name"))
    except (OSError, json.JSONDecodeError):
        pass
    return out


def fetch_espn_teams(slug):
    """[(name variants, logo url)] for one ESPN league. Raises on failure."""
    # ESPN's WAF blocks a custom UA (see fetch_live_scores.py) - keep urllib's default.
    with urlopen(Request(TEAMS_URL.format(slug=slug)), timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = []
    for item in data["sports"][0]["leagues"][0]["teams"]:
        t = item.get("team") or {}
        logos = t.get("logos") or []
        if not logos:
            continue
        names = {norm(t.get(k)) for k in ("displayName", "shortDisplayName", "name", "location") if t.get(k)}
        out.append((names - {""}, logos[0]["href"]))
    return out


def match(team, espn):
    """Logo url for our team name, or None. Exact normalized name first, then substring."""
    n = norm(ESPN_NAME.get(team, team))
    for names, url in espn:
        if n in names:
            return url
    hits = {url for names, url in espn for e in names if len(min(n, e, key=len)) >= 4 and (n in e or e in n)}
    return hits.pop() if len(hits) == 1 else None


def main():
    try:
        cur = json.loads(OUT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cur = {}
    logos = cur.get("leagues") or {}
    for league, teams in MANUAL.items():
        logos.setdefault(league, {}).update(teams)
    missing = {}
    for league, teams in sorted(our_teams().items()):
        have = logos.setdefault(league, {})
        todo = sorted(t for t in teams if t not in have)
        slug = ESPN_SLUG.get(league)
        if not todo or not slug:
            continue
        try:
            espn = fetch_espn_teams(slug)
        except Exception as e:  # noqa: BLE001 - one league failing must not stop the others
            print(f"  {league}: ESPN fetch failed ({e}) - keeping existing crests")
            missing[league] = todo
            continue
        for t in todo:
            url = match(t, espn)
            if url:
                have[t] = url
            else:
                missing.setdefault(league, []).append(t)
        print(f"  {league}: +{len(todo) - len(missing.get(league, []))} crests, {len(missing.get(league, []))} unmatched")
    out = {"source": "ESPN site API", "leagues": {k: dict(sorted(v.items())) for k, v in sorted(logos.items()) if v},
           "missing": missing}
    text = json.dumps(out, ensure_ascii=False, indent=1) + "\n"
    if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
        OUT.write_text(text, encoding="utf-8")
    total = sum(len(v) for v in out["leagues"].values())
    print(f"team-logos.json: {total} crests, unmatched: {sum(len(v) for v in missing.values())}")
    for league, teams in missing.items():
        print(f"  unmatched {league}: {', '.join(teams)}")


if __name__ == "__main__":
    main()
