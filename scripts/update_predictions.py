import json, math, os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen
OUTPUT_FILE=Path('predictions.json')
BASE='https://v3.football.api-sports.io'
LEAGUES={39:'Premier League',140:'LaLiga',78:'Bundesliga',135:'Serie A',61:'Ligue 1',88:'Eredivisie'}
def api(path,params):
 k=os.environ.get('API_FOOTBALL_KEY')
 if not k: raise RuntimeError('API key missing')
 q='&'.join(f'{a}={b}' for a,b in params.items())
 req=Request(f'{BASE}/{path}?{q}',headers={'x-apisports-key':k})
 with urlopen(req,timeout=30) as r: d=json.loads(r.read())
 if d.get('errors'): raise RuntimeError(str(d['errors']))
 return d.get('response',[])
def over(lam,n):
 p=math.exp(-lam); s=p
 for k in range(1,n+1): p*=lam/k; s+=p
 return 1-s
def main():
 now=datetime.now(timezone.utc); today=now.date(); end=today+timedelta(days=7); out=[]; cache={}
 for lid,league in LEAGUES.items():
  fs=api('fixtures',{'league':lid,'season':2026,'from':today,'to':end}); print(league,len(fs))
  for f in fs:
   if f['fixture']['status']['short'] not in ('NS','TBD'): continue
   d=datetime.fromisoformat(f['fixture']['date'].replace('Z','+00:00')); h=f['teams']['home']; a=f['teams']['away']; pair='-'.join(map(str,sorted((h['id'],a['id']))))
   if pair not in cache: cache[pair]=api('fixtures/headtohead',{'h2h':f"{h['id']}-{a['id']}",'last':20})
   cut=now-timedelta(days=1827); games=[]
   for x in cache[pair]:
    xd=datetime.fromisoformat(x['fixture']['date'].replace('Z','+00:00')); gh=x.get('goals',{}).get('home'); ga=x.get('goals',{}).get('away')
    if xd>=cut and gh is not None and ga is not None: games.append((xd,int(gh)+int(ga)))
   games=sorted(games,reverse=True)[:10]
   if not games: continue
   lam=sum(g for _,g in games)/len(games); p05=over(lam,0); p15=over(lam,1); p25=over(lam,2)
   label='ULTRA' if p05>=.95 else 'HIGH' if p05>=.90 else 'MEDIUM' if p05>=.85 else ''
   out.append({'match_id':str(f['fixture']['id']),'league_id':lid,'league':league,'kickoff_utc':d.isoformat().replace('+00:00','Z'),'home':h['name'],'away':a['name'],'h2h_matches':len(games),'h2h_total_goal_avg':round(lam,3),'lambda_total':round(lam,3),'p_over_0_5':round(p05,4),'p_over_1_5':round(p15,4),'p_over_2_5':round(p25,4),'label':label,'updated_at':now.isoformat()})
 out.sort(key=lambda x:x['kickoff_utc'])
 if not out: raise RuntimeError('No predictions generated')
 OUTPUT_FILE.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print('Wrote',len(out))
if __name__=='__main__': main()
