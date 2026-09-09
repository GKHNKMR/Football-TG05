import csv, io, json, math, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

OUTPUT_FILE = Path('predictions.json')
LEAGUES = {'E0':'Premier League','SP1':'LaLiga','D1':'Bundesliga','I1':'Serie A','F1':'Ligue 1','N1':'Eredivisie'}
HOSTS = ['https://www.football-data.co.uk/mmz4281','https://football-data.co.uk/mmz4281']


def fetch_csv(season, code):
    for host in HOSTS:
        url=f'{host}/{season}/{code}.csv'
        for attempt in range(4):
            try:
                req=Request(url,headers={'User-Agent':'Mozilla/5.0 BETAVUS','Accept':'text/csv,*/*'})
                with urlopen(req,timeout=30) as r:
                    raw=r.read()
                return list(csv.DictReader(io.TextIOWrapper(io.BytesIO(raw),encoding='latin1')))
            except Exception as e:
                print(f'CSV retry {url} #{attempt+1}: {e}')
                time.sleep(2*(attempt+1))
    return []


def parse_date(s):
    s=(s or '').strip()
    for fmt in ('%d/%m/%Y %H:%M','%d/%m/%Y'):
        try: return datetime.strptime(s,fmt).replace(tzinfo=timezone.utc)
        except ValueError: pass
    return None


def norm(s):
    return ''.join(c.lower() for c in (s or '') if c.isalnum())


def poisson_over(lam,n):
    p=math.exp(-lam); total=p
    for k in range(1,n+1):
        p*=lam/k; total+=p
    return 1-total


def main():
    now=datetime.now(timezone.utc); today=now.date(); end=today+timedelta(days=7)
    seasons=['2122','2223','2324','2425','2526']
    out=[]
    for code,league in LEAGUES.items():
        history=[]
        for season in seasons:
            rows=fetch_csv(season,code)
            print(f'{league} {season}: {len(rows)} historical rows')
            for r in rows:
                d=parse_date(r.get('Date'))
                if not d: continue
                try: hg=int(float(r.get('FTHG',''))); ag=int(float(r.get('FTAG','')))
                except (TypeError,ValueError): continue
                home=r.get('HomeTeam','').strip(); away=r.get('AwayTeam','').strip()
                if home and away: history.append((d,home,away,hg+ag))
        current=fetch_csv('2627',code)
        print(f'{league} 2627: {len(current)} current rows')
        for r in current:
            d=parse_date(r.get('Date')); home=r.get('HomeTeam','').strip(); away=r.get('AwayTeam','').strip()
            if not d or not home or not away or not (today<=d.date()<=end): continue
            h2h=[x for x in history if {norm(x[1]),norm(x[2])}=={norm(home),norm(away)}]
            h2h=sorted(h2h,key=lambda x:x[0],reverse=True)[:10]
            if not h2h: continue
            lam=sum(x[3] for x in h2h)/len(h2h)
            p05=poisson_over(lam,0); p15=poisson_over(lam,1); p25=poisson_over(lam,2)
            label='ULTRA' if p05>=.95 else 'HIGH' if p05>=.90 else 'MEDIUM' if p05>=.85 else ''
            out.append({'match_id':f'{code}-{d:%Y%m%d}-{norm(home)}-{norm(away)}','league':league,'kickoff_utc':d.isoformat().replace('+00:00','Z'),'home':home,'away':away,'h2h_matches':len(h2h),'h2h_total_goal_avg':round(lam,3),'lambda_total':round(lam,3),'p_over_0_5':round(p05,4),'p_over_1_5':round(p15,4),'p_over_2_5':round(p25,4),'label':label,'updated_at':now.isoformat()})
    out.sort(key=lambda x:x['kickoff_utc'])
    if not out: raise RuntimeError('No predictions generated from Football-Data CSV')
    OUTPUT_FILE.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Wrote {len(out)} predictions from Football-Data CSV; 5-year H2H only')

if __name__=='__main__': main()
