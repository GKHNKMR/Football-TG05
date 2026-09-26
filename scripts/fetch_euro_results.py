"""Avrupa kupaları (Şampiyonlar Ligi, Avrupa Ligi, Konferans Ligi) geçmiş sonuçları — iş listesi #5.

Kaynak: ESPN'in herkese açık scoreboard API'si (scripts/fetch_live_scores.py ile aynı, anahtarsız).
Tarih aralığı sorgusu desteklenmiyor (400), bu yüzden UEFA maç günleri (salı / çarşamba / perşembe;
finaller için mayıs-haziran cumartesileri) tek tek sorgulanır. Ham yanıtlar data/cache/espn_euro/
altında önbelleklenir (git'e girmez); özet data/euro-results.json'a yazılır.

Uzatma / penaltıya giden maçlar (STATUS_FINAL_AET / _PEN) işaretlenir: bahis 90 dakikaya göre
sonuçlanır, ESPN skoru uzatmayı içerir → backtest bu maçları dışarıda bırakır.
Kullanım: python scripts/fetch_euro_results.py [--from 2021-07-01]
"""
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'data' / 'cache' / 'espn_euro'
OUT = ROOT / 'data' / 'euro-results.json'
COMPS = {'uefa.champions': 'UCL', 'uefa.europa': 'UEL', 'uefa.europa.conf': 'UECL'}
URL = 'https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={d}'


def match_days(start, end):
    d = start
    while d <= end:
        if d.weekday() in (1, 2, 3) or (d.month in (5, 6) and d.weekday() == 5):
            yield d
        d += timedelta(days=1)


def get(slug, d, today):
    p = CACHE / f'{slug}_{d:%Y%m%d}.json'
    if p.exists() and d < today - timedelta(days=3):
        return json.loads(p.read_text(encoding='utf-8'))
    for attempt in range(3):
        try:
            with urlopen(URL.format(slug=slug, d=f'{d:%Y%m%d}'), timeout=30) as r:
                j = json.loads(r.read().decode('utf-8'))
            p.write_text(json.dumps(j), encoding='utf-8')
            time.sleep(0.15)
            return j
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            print('  retry', slug, d, e)
            time.sleep(2 + attempt * 3)
    return {'events': []}


def main():
    start = date.fromisoformat(sys.argv[sys.argv.index('--from') + 1]) if '--from' in sys.argv else date(2021, 7, 1)
    today = date.today()
    CACHE.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    days = list(match_days(start, today))
    for i, d in enumerate(days):
        for slug, comp in COMPS.items():
            for e in get(slug, d, today).get('events', []):
                c = e['competitions'][0]
                st = c['status']['type']
                if not st.get('completed'):
                    continue
                side = {x['homeAway']: x for x in c['competitors']}
                if e['id'] in seen or 'home' not in side or 'away' not in side:
                    continue
                seen.add(e['id'])
                try:
                    hg, ag = int(side['home']['score']), int(side['away']['score'])
                except (TypeError, ValueError, KeyError):
                    continue
                rows.append({'id': e['id'], 'comp': comp, 'kickoff_utc': e['date'], 'home': side['home']['team']['displayName'],
                             'away': side['away']['team']['displayName'], 'home_id': side['home']['team']['id'],
                             'away_id': side['away']['team']['id'], 'hg': hg, 'ag': ag, 'status': st.get('name'),
                             'aet': st.get('name') in ('STATUS_FINAL_AET', 'STATUS_FINAL_PEN')})
        if i % 50 == 0:
            print(f'{d} ({i}/{len(days)}) — {len(rows)} maç')
    rows.sort(key=lambda r: r['kickoff_utc'])
    OUT.write_text(json.dumps({'source': 'ESPN scoreboard', 'from': str(start), 'matches': rows}, ensure_ascii=False, separators=(',', ':')) + '\n',
                   encoding='utf-8', newline='\n')
    by = {}
    for r in rows:
        by[r['comp']] = by.get(r['comp'], 0) + 1
    print('Wrote', OUT, len(rows), by, 'uzatma/penaltı:', sum(r['aet'] for r in rows))


if __name__ == '__main__':
    main()
