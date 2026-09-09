import csv, io, json, math, re, time, unicodedata
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import Request, urlopen

OUTPUT_FILE = Path('predictions.json')
LEAGUES = {'E0':'Premier League','SP1':'LaLiga','D1':'Bundesliga','I1':'Serie A','F1':'Ligue 1','N1':'Eredivisie'}
HOSTS = ['https://www.football-data.co.uk/mmz4281','https://football-data.co.uk/mmz4281']

# Football-Data uses short names in many seasons; current fixtures can use longer names.
ALIASES = {
    # England
    'manchesterunited':'manchesterunited', 'manutd':'manchesterunited', 'manunited':'manchesterunited', 'manchesterutd':'manchesterunited',
    'manchestercity':'mancity', 'mancity':'mancity',
    'nottinghamforest':'nottinghamforest', 'nottmforest':'nottinghamforest', 'nottforest':'nottinghamforest',
    'newcastle':'newcastleunited', 'newcastleunited':'newcastleunited', 'newcastleutd':'newcastleunited',
    'wolves':'wolverhamptonwanderers', 'wolverhampton':'wolverhamptonwanderers', 'wolverhamptonwanderers':'wolverhamptonwanderers',
    'westham':'westhamunited', 'westhamunited':'westhamunited', 'brighton':'brightonandhovealbion', 'brightonandhove':'brightonandhovealbion',
    'tottenham':'tottenhamhotspur', 'tottenhamhotspur':'tottenhamhotspur', 'spurs':'tottenhamhotspur',
    'leicester':'leicestercity', 'leicestercity':'leicestercity', 'ipswich':'ipswichtown', 'ipswichtown':'ipswichtown',
    'norwich':'norwichcity', 'norwichcity':'norwichcity', 'leeds':'leedsunited', 'leedsunited':'leedsunited',
    # Spain
    'athbilbao':'athleticclub', 'athleticbilbao':'athleticclub', 'athleticclub':'athleticclub',
    'athmadrid':'atleticomadrid', 'atleticomadrid':'atleticomadrid', 'atletico':'atleticomadrid',
    'realbetis':'realbetis', 'betis':'realbetis', 'celta':'celtavigo', 'celtavigo':'celtavigo',
    'deportivoalaves':'deportivoalaves', 'alaves':'deportivoalaves',
    # Germany
    'bayernmunich':'bayernmunich', 'bayernmunchen':'bayernmunich', 'bayern':'bayernmunich',
    'borussiadortmund':'borussiadortmund', 'dortmund':'borussiadortmund',
    'borussiamgladbach':'borussiamgladbach', 'monchengladbach':'borussiamgladbach', 'mgladbach':'borussiamgladbach', 'gladbach':'borussiamgladbach',
    # Italy
    'inter':'inter', 'internazionale':'inter', 'intermilan':'inter', 'milan':'milan', 'acmilan':'milan',
    'roma':'roma', 'asroma':'roma', 'lazio':'lazio', 'juventus':'juventus', 'juve':'juventus',
    # France
    'parissg':'parissaintgermain', 'psg':'parissaintgermain', 'parissaintgermain':'parissaintgermain',
    'stetienne':'saintetienne', 'saintetienne':'saintetienne', 'monaco':'monaco',
    # Netherlands
    'ajax':'ajax', 'psv':'psveindhoven', 'psveindhoven':'psveindhoven', 'feyenoord':'feyenoord',
    'twente':'fctwente', 'fctwente':'fctwente', 'az':'azalkmaar', 'azalkmaar':'azalkmaar',
}


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
    s=unicodedata.normalize('NFKD',s or '')
    s=''.join(c for c in s if not unicodedata.combining(c)).lower()
    s=re.sub(r'[^a-z0-9]','',s)
    return ALIASES.get(s,s)


def same_team(a,b):
    na, nb = norm(a), norm(b)
    if na == nb:
        return True
    # Handles remaining harmless Football-Data abbreviations without making guesses across leagues.
    if len(na) >= 6 and len(nb) >= 6 and (na in nb or nb in na):
        return True
    return SequenceMatcher(None, na, nb).ratio() >= 0.82


def same_pair(h1,a1,h2,a2):
    return (same_team(h1,h2) and same_team(a1,a2)) or (same_team(h1,a2) and same_team(a1,h2))


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
            h2h=[x for x in history if same_pair(x[1],x[2],home,away)]
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
