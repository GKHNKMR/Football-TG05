import csv, io, json, math, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

OUTPUT_FILE=Path('predictions.json')
LEAGUES={
  'E0':('Premier League',39), 'SP1':('LaLiga',140), 'D1':('Bundesliga',78),
  'I1':('Serie A',135), 'F1':('Ligue 1',61), 'N1':('Eredivisie',88)
}
SEASONS=['2122','2223','2324','2425','2526']
BASE='https://www.football-data.co.uk/mmz4281'

def get_csv(code,season):
    url=f'{BASE}/{season}/{code}.csv'
    try:
        req=Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urlopen(req,timeout=30) as r: return list(csv.DictReader(io.TextIOWrapper(r,encoding='latin1')))
    except Exception as e:
        print(f'CSV unavailable {code}/{season}: {e}'); return []

def parse_date(s):
    for fmt in ('%d/%m/%Y %H:%M','%d/%m/%Y'):
        try:return datetime.strptime(s.strip(),fmt).replace(tzinfo=timezone.utc)
        except:pass
    return None

def poisson_over(lam,n):
    p=math.exp(-lam); total=p
    for k in range(1,n+1): p*=lam/k; total+=p
    return 1-total

def main():
    now=datetime.now(timezone.utc); today=now.date(); end=today+timedelta(days=7)
    all_out=[]; total_h2h=0
    for code,(league,lid) in LEAGUES.items():
        history=[]
        for season in SEASONS:
            for r in get_csv(code,season):
                d=parse_date(r.get('Date',''))
                if not d or not r.get('HomeTeam') or not r.get('AwayTeam'): continue
                try: hg=int(float(r.get('FTHG',''))); ag=int(float(r.get('FTAG','')))
                except: continue
                history.append((d,r['HomeTeam'].strip(),r['AwayTeam'].strip(),hg,ag))
        current=get_csv(code,'2627')
        for r in current:
            d=parse_date(r.get('Date',''))
            if not d or not (today<=d.date()<=end): continue
            home=r.get('HomeTeam','').strip(); away=r.get('AwayTeam','').strip()
            if not home or not away: continue
            h2h=[x for x in history if {x[1].lower(),x[2].lower()}=={home.lower(),away.lower()}]
            h2h.sort(key=lambda x:x[0],reverse=True)
            h2h=h2h[:10]
            if not h2h:
                print(f'NO H2H: {home} - {away}')
                continue
            avg=sum(x[3]+x[4] for x in h2h)/len(h2h); total_h2h+=1
            p05=poisson_over(avg,0); p15=poisson_over(avg,1); p25=poisson_over(avg,2)
            label='ULTRA' if p05>=.95 else 'HIGH' if p05>=.90 else 'MEDIUM' if p05>=.85 else ''
            all_out.append({
              'match_id':f'{code}-{d.strftime("%Y%m%d")}-{home}-{away}',
              'league_id':lid,'league':league,'kickoff_utc':d.isoformat().replace('+00:00','Z'),
              'home':home,'away':away,'h2h_matches':len(h2h),'h2h_total_goal_avg':round(avg,3),
              'lambda_total':round(avg,3),'p_over_0_5':round(p05,4),'p_over_1_5':round(p15,4),
              'p_over_2_5':round(p25,4),'label':label,'updated_at':now.isoformat()
            })
    all_out.sort(key=lambda x:x['kickoff_utc'])
    OUTPUT_FILE.write_text(json.dumps(all_out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Wrote {len(all_out)} predictions from 5-year H2H only')

if __name__=='__main__':main()
