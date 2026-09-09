import json, math, os, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://v3.football.api-sports.io"
API_KEY = os.environ.get("API_FOOTBALL_KEY", "").strip()
CACHE_DIR = Path("data/cache")
OUTPUT_FILE = Path("predictions.json")
LEAGUES = {39:"Premier League",140:"LaLiga",78:"Bundesliga",135:"Serie A",61:"Ligue 1",88:"Eredivisie"}
# Five-season historical window. Current 2025 season is restricted on the Free plan,
# so the calculation uses the accessible historical seasons 2021-2024.
SEASONS = [2021, 2022, 2023, 2024]
MIN_SECONDS_BETWEEN_CALLS = 6.2
_last_request_at = 0.0

SEED = [
("ED-50","Eredivisie","FC Twente","Telstar","2026-09-09T16:45:00Z"),
("ED-51","Eredivisie","AZ","Willem II","2026-09-11T18:00:00Z"),
("LL-12","LaLiga","Sevilla","Valencia","2026-09-11T19:00:00Z"),
("SA-23","Serie A","Venezia","Fiorentina","2026-09-11T18:45:00Z"),
("L1-34","Ligue 1","Rennes","Marseille","2026-09-11T18:45:00Z"),
("BL-13","Bundesliga","Union Berlin","Schalke 04","2026-09-11T18:30:00Z"),
("L1-35","Ligue 1","Strasbourg","Monaco","2026-09-12T15:15:00Z"),
("PL-1","Premier League","AFC Bournemouth","Brentford","2026-09-12T14:00:00Z"),
("PL-2","Premier League","Aston Villa","Nottingham Forest","2026-09-12T16:30:00Z"),
("PL-3","Premier League","Chelsea","Hull City","2026-09-12T16:30:00Z"),
("PL-4","Premier League","Crystal Palace","Ipswich Town","2026-09-12T16:30:00Z"),
("PL-5","Premier League","Liverpool","Fulham","2026-09-12T16:30:00Z"),
("SA-24","Serie A","Genoa","Frosinone","2026-09-12T16:00:00Z"),
("SA-25","Serie A","Lazio","AC Milan","2026-09-12T18:45:00Z"),
("SA-26","Serie A","Atalanta","Cagliari","2026-09-13T13:00:00Z"),
("L1-36","Ligue 1","Paris FC","Lyon","2026-09-13T13:00:00Z"),
("L1-37","Ligue 1","Lorient","Toulouse","2026-09-13T15:15:00Z"),
("L1-38","Ligue 1","Le Havre","Angers","2026-09-13T15:15:00Z"),
("L1-39","Ligue 1","Auxerre","Nice","2026-09-13T15:15:00Z"),
("PL-6","Premier League","Tottenham Hotspur","Everton","2026-09-13T15:30:00Z"),
("PL-7","Premier League","Sunderland","Arsenal","2026-09-13T18:00:00Z"),
("ED-52","Eredivisie","FC Twente","ADO Den Haag","2026-09-13T12:30:00Z"),
("ED-53","Eredivisie","Go Ahead Eagles","FC Groningen","2026-09-13T14:45:00Z"),
("ED-54","Eredivisie","Fortuna Sittard","Ajax","2026-09-13T14:45:00Z"),
("PL-8","Premier League","Coventry City","Brighton & Hove Albion","2026-09-14T19:00:00Z"),
("SA-27","Serie A","Lecce","Monza","2026-09-14T16:30:00Z"),
("ED-55","Eredivisie","Excelsior","FC Utrecht","2026-09-14T18:00:00Z"),
("ED-56","Eredivisie","sc Heerenveen","Telstar","2026-09-14T18:00:00Z"),
("ED-57","Eredivisie","SC Cambuur","N.E.C.","2026-09-14T19:00:00Z"),
("SA-28","Serie A","Napoli","Bologna","2026-09-14T18:45:00Z"),
("SA-29","Serie A","Sassuolo","Juventus","2026-09-14T18:45:00Z"),
]

def api_get(path, params):
    global _last_request_at
    if not API_KEY: return []
    req=Request(f"{BASE_URL}{path}?{urlencode(params)}",headers={"x-apisports-key":API_KEY})
    wait=MIN_SECONDS_BETWEEN_CALLS-(time.monotonic()-_last_request_at)
    if wait>0: time.sleep(wait)
    try:
        with urlopen(req,timeout=35) as r: data=json.loads(r.read().decode())
        _last_request_at=time.monotonic()
        if data.get("errors"): raise RuntimeError(str(data["errors"]))
        return data.get("response",[])
    except Exception as e:
        _last_request_at=time.monotonic(); print(f"API unavailable: {e}"); return []

def load(path, default):
    try: return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception: return default

def fixture_date(f): return datetime.fromisoformat(f["fixture"]["date"].replace("Z","+00:00"))
def goals(f):
    g=f.get("goals",{}); return None if g.get("home") is None or g.get("away") is None else (int(g["home"]),int(g["away"]))
def completed(f): return f.get("fixture",{}).get("status",{}).get("short") in {"FT","AET","PEN"}

def poisson(lam,n):
    p=math.exp(-lam); s=p
    for k in range(1,n+1): p*=lam/k; s+=p
    return 1-s

def label(p): return "ULTRA" if p>=.95 else "HIGH" if p>=.90 else "MEDIUM" if p>=.85 else ""

def build_history():
    by_league={}; name_id={}
    for lid in LEAGUES:
        allfx=[]
        for season in SEASONS:
            path=CACHE_DIR/"history"/f"{lid}_{season}.json"
            allfx += load(path,[])
        by_league[lid]=allfx
        for f in allfx:
            t=f.get("teams",{}); h=t.get("home",{}); a=t.get("away",{})
            if h.get("name") and h.get("id"): name_id[h["name"].lower()]=h["id"]
            if a.get("name") and a.get("id"): name_id[a["name"].lower()]=a["id"]
    return by_league,name_id

def h2h_values(fixtures,hid,aid):
    vals=[]
    for f in fixtures:
        if not completed(f): continue
        g=goals(f); t=f.get("teams",{})
        ids={t.get("home",{}).get("id"),t.get("away",{}).get("id")}
        if g and hid and aid and ids=={hid,aid}:
            vals.append((fixture_date(f),sum(g)))
    vals.sort(reverse=True)
    # Only the last 10 meetings within the historical five-season window.
    return [v for _,v in vals[:10]]

def main():
    today=datetime.now(timezone.utc).date(); end=today+timedelta(days=7)
    history,name_id=build_history()
    fixtures=[]
    # Use the simple current fixture seed when the API Free plan blocks 2026/27.
    # Prediction math below uses H2H only; no team-form or league-average component.
    for mid,league,home,away,kick in SEED:
        lid=next(k for k,v in LEAGUES.items() if v==league)
        fixtures.append({"id":mid,"league_id":lid,"league":league,"home":home,"away":away,"kickoff":kick,
                         "home_id":name_id.get(home.lower()),"away_id":name_id.get(away.lower())})

    output=[]
    for f in fixtures:
        dt=datetime.fromisoformat(f["kickoff"].replace("Z","+00:00")).date()
        if not today<=dt<=end: continue
        lid=f["league_id"]; hist=history.get(lid,[])
        hid=f.get("home_id"); aid=f.get("away_id")
        hh=h2h_values(hist,hid,aid) if hid and aid else []
        if not hh:
            # No H2H means no invented prediction. The UI can still show the fixture.
            continue
        lam=sum(hh)/len(hh)
        lam=max(.25,min(5.5,lam))
        p05=poisson(lam,0); p15=poisson(lam,1); p25=poisson(lam,2)
        output.append({"match_id":f["id"],"league_id":lid,"league":f["league"],"kickoff_utc":f["kickoff"],
                       "home":f["home"],"away":f["away"],"h2h_matches_used":len(hh),
                       "h2h_goals_avg":round(lam,3),"p_over_0_5":round(p05,4),"p_over_1_5":round(p15,4),
                       "p_over_2_5":round(p25,4),"label":label(p05),"updated_at":datetime.now(timezone.utc).isoformat()})
    output.sort(key=lambda x:x["kickoff_utc"])
    OUTPUT_FILE.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Wrote {len(output)} H2H-only predictions")

if __name__=="__main__": main()
