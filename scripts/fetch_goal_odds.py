"""Gerçek 0.5+/1.5+/2.5+ Üst oranları (iş listesi #20) → data/goal-odds.json

Kaynak: 5DollarFootballAPI (https://5dollarfootballapi.com), Bet365 `goal_line_fixed` pazarı
(0.5 … 9.5 sabit çizgiler, açılış + kapanış fiyatı). Ücretsiz plan: yalnız büyük 5 lig, saatte 60
istek (hesap başına), oranlar maç başına ayrı istek. Veri 26.09.2026'dan beri var. Ücretsiz planda
herkese açık sitede "Football data by 5DollarFootballAPI" bağlantısı zorunlu (index.html alt bilgisi).

Saatlik bot her çalıştırmada:
  1. Lig kimliklerini bir kez /v1/leagues ile bulur (belgelerdeki kimlikler birbirini tutmuyor).
  2. Her ligin önümüzdeki ~2 günlük fikstürünü 3 saatte bir listeler, predictions.json'daki maçlarla
     eşler (lig + başlama saati ± 3 sa + takım adı).
  3. Bütçe içinde oran çeker: maç yaklaştıkça daha sık (12 sa+ → 6 sa, 3–12 sa → 2 sa, <3 sa → 45 dk),
     maç bitince bir kez daha (kapanış oranı → ileride gerçek oranlı backtest için saklanır).

Anahtar: ortam değişkeni FIVEDOLLAR_API_KEY (GitHub Actions secret). Yoksa sessizce çıkar.
Kullanım: python scripts/fetch_goal_odds.py
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_scores import norm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data' / 'goal-odds.json'
PRED = ROOT / 'predictions.json'
API = 'https://api.5dollarfootballapi.com/v1'
UA = 'BETAVUS-bot/1.0 (+https://betavus.vercel.app)'
ATTRIBUTION = 'Football data by 5DollarFootballAPI'

# Bizim lig adı → (API ülke kodu, API'deki lig adının norm() hâli)
LEAGUES = {
    'Premier League': ('GB-ENG', 'PREMIER LEAGUE'),
    'LaLiga': ('ES', 'LA LIGA'),
    'Bundesliga': ('DE', 'BUNDESLIGA'),
    'Serie A': ('IT', 'SERIE A'),
    'Ligue 1': ('FR', 'LIGUE 1'),
}
LINES = {0.5: 'o05', 1.5: 'o15', 2.5: 'o25'}
LOOKAHEAD_H = 50           # bu kadar saat içinde başlayacak maçlar
LIST_EVERY_H = 3           # fikstür listesi yenileme aralığı
FINAL_AFTER_H = 2.5        # başlama + bu kadar saat sonra kapanış oranı
FINAL_GIVE_UP_D = 3
MAX_CALLS = 40             # bir çalıştırmada en fazla istek (saatlik kota 60)
MIN_REMAINING = 3          # X-RateLimit-Remaining bunun altına inerse dur
KEEP_DAYS = 400


class Budget:
    def __init__(self, key, max_calls=MAX_CALLS):
        self.key, self.left, self.remaining, self.calls, self.stopped = key, max_calls, None, 0, None

    def ok(self):
        return not self.stopped and self.left > 0 and (self.remaining is None or self.remaining > MIN_REMAINING)


def api_get(budget, path, params=None):
    """JSON gövdesini döndürür; hata olursa None (ve gerekirse bütçeyi durdurur)."""
    if not budget.ok():
        return None
    url = API + path + ('?' + urllib.parse.urlencode(params) if params else '')
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {budget.key}', 'Accept': 'application/json',
                                               'User-Agent': UA})
    budget.left -= 1
    budget.calls += 1
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            rem = r.headers.get('X-RateLimit-Remaining')
            if rem is not None and rem.isdigit():
                budget.remaining = int(rem)
            return json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', 'replace')[:300]
        print(f'  HTTP {e.code} {path}: {body}')
        if e.code in (401, 429):
            budget.stopped = f'HTTP {e.code}'
        return {'_error': e.code, '_body': body}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f'  ağ hatası {path}: {e}')
        return None


def rows(resp):
    if not resp or resp.get('_error'):
        return []
    d = resp.get('data')
    return d if isinstance(d, list) else []


def parse_iso(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00'))


def find_league_ids(budget, known, misses, now):
    """misses: bulunamayan lig → son deneme; kota harcamamak için günde bir kez yeniden denenir."""
    ids = dict(known)
    for ours, (country, want) in LEAGUES.items():
        if ours in ids or (misses.get(ours) and now - parse_iso(misses[ours]) < timedelta(hours=24)):
            continue
        resp = api_get(budget, '/leagues', {'country': country, 'per_page': 100})
        if resp is None:
            continue
        cands = [x for x in rows(resp) if norm(x.get('name', '')) == want]
        if cands:
            best = max(cands, key=lambda x: x.get('last_fixture_ts') or 0)
            ids[ours] = best['id']
            misses.pop(ours, None)
            print(f'  lig: {ours} → {best["id"]} ({best.get("name")})')
        else:
            misses[ours] = now.isoformat(timespec='seconds')
            print(f'  lig bulunamadı: {ours} ({country}): {[x.get("name") for x in rows(resp)][:15]}')
    return ids


def list_fixtures(budget, league_id, now):
    """Önümüzdeki LOOKAHEAD_H saatin (ve son 6 saatin) fikstürü. None = istek başarısız."""
    start, end = int((now - timedelta(hours=6)).timestamp()), int((now + timedelta(hours=LOOKAHEAD_H)).timestamp())
    resp = api_get(budget, f'/leagues/{league_id}/fixtures',
                   {'start_time': start, 'end_time': end, 'order': 'asc', 'per_page': 100})
    if resp and not resp.get('_error'):
        return rows(resp)
    if resp and resp.get('_error') == 403:     # lig listesi plana kapalıysa: tarih listesi (≤ 24 sa pencere)
        out, t = [], start
        while t < end:
            r = api_get(budget, '/fixtures', {'league': league_id, 'start_time': t, 'end_time': min(end, t + 86400),
                                              'per_page': 100})
            if r is None or r.get('_error'):
                return None
            out += rows(r)
            t += 86400
        return out
    return None


def match_prediction(fx, preds):
    """API fikstürünü predictions.json satırıyla eşler: aynı lig, başlama ± 3 sa, iki takım adı
    (norm, alt dize) ya da aynı dakikada başlayan ve bir takımı tutan tek maç."""
    ko = parse_iso(fx['kickoff_utc'])
    h, a = norm(fx['teams']['home']['name']), norm(fx['teams']['away']['name'])
    near = [p for p in preds if abs((parse_iso(p['kickoff_utc']) - ko).total_seconds()) <= 3 * 3600]

    def same(x, y):
        return bool(x and y) and (x in y or y in x)
    both = [p for p in near if same(h, norm(p['home'])) and same(a, norm(p['away']))]
    if len(both) == 1:
        return both[0]
    one = [p for p in near if parse_iso(p['kickoff_utc']) == ko and (same(h, norm(p['home'])) or same(a, norm(p['away'])))]
    return one[0] if len(one) == 1 else None


def parse_odds(resp):
    """{'o05': 1.05, 'o15': 1.25, 'o25': 1.85} — kapanış (maç öncesi son) fiyat, yoksa açılış."""
    if not resp or resp.get('_error'):
        return None
    out = {}
    for bm in (resp.get('data') or {}).get('bookmakers') or []:
        for e in (bm.get('odds') or {}).get('goal_line_fixed') or []:
            if not e or e.get('line') not in LINES:
                continue
            for k in ('closing', 'opening'):
                ov = (e.get(k) or {}).get('over')
                if isinstance(ov, (int, float)) and ov > 1:
                    out[LINES[e['line']]] = round(float(ov), 3)
                    break
        if out:
            break
    return out


def due(entry, now):
    """Bu maçın oranı bu çalıştırmada çekilmeli mi? (öncelik, neden) ya da None."""
    ko = parse_iso(entry['kickoff_utc'])
    hrs = (ko - now).total_seconds() / 3600
    last = parse_iso(entry['fetched_at']) if entry.get('fetched_at') else None
    age = (now - last).total_seconds() / 3600 if last else None
    if hrs > 0:
        if hrs > LOOKAHEAD_H:
            return None
        gap = 6 if hrs > 12 else 2 if hrs > 3 else 0.75
        if age is None or age >= gap:
            return (0 if hrs <= 24 else 2, hrs), 'pre'
        return None
    if entry.get('final') or entry.get('final_tries', 0) >= 3:
        return None
    if now - ko >= timedelta(hours=FINAL_AFTER_H) and now - ko <= timedelta(days=FINAL_GIVE_UP_D):
        return (1, -hrs), 'final'
    return None


def run(key, now=None):
    now = now or datetime.now(timezone.utc)
    budget = Budget(key)
    old = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
    fixtures = old.get('fixtures', {})
    listed_at = old.get('listed_at', {})
    preds = json.loads(PRED.read_text(encoding='utf-8'))
    misses = old.get('league_misses', {})
    ids = find_league_ids(budget, old.get('leagues', {}), misses, now)

    unmatched = []
    for ours, lid in ids.items():
        la = listed_at.get(ours)
        if la and now - parse_iso(la) < timedelta(hours=LIST_EVERY_H):
            continue
        fx = list_fixtures(budget, lid, now)
        if fx is None:
            continue
        listed_at[ours] = now.isoformat(timespec='seconds')
        lp = [p for p in preds if p['league'] == ours]
        for f in fx:
            if f.get('status') not in (None, 'scheduled', 'unknown') and str(f['id']) not in fixtures:
                continue
            p = match_prediction(f, lp)
            if not p:
                if f.get('status') in (None, 'scheduled'):
                    unmatched.append(f"{ours}: {f['teams']['home']['name']} - {f['teams']['away']['name']}")
                continue
            e = fixtures.setdefault(str(f['id']), {})
            e.update(league=ours, home=p['home'], away=p['away'], kickoff_utc=p['kickoff_utc'],
                     api_home=f['teams']['home']['name'], api_away=f['teams']['away']['name'])

    todo = sorted(((d[0], fid, d[1]) for fid, e in fixtures.items() if (d := due(e, now))), key=lambda t: t[0])
    fetched = 0
    for _, fid, why in todo:
        if not budget.ok():
            break
        odds = parse_odds(api_get(budget, f'/fixtures/{fid}/odds', {'market': 'goalline_fixed'}))
        if odds is None:
            continue
        e = fixtures[fid]
        e['fetched_at'] = now.isoformat(timespec='seconds')
        if odds:
            e.update(odds)
        fetched += 1
        if why == 'final':
            e['final_tries'] = e.get('final_tries', 0) + 1
            if odds:
                e['final'] = True

    cutoff = now - timedelta(days=KEEP_DAYS)
    fixtures = {k: v for k, v in fixtures.items() if 'kickoff_utc' in v and parse_iso(v['kickoff_utc']) >= cutoff
                and (v.get('o05') or v.get('o15') or parse_iso(v['kickoff_utc']) >= now - timedelta(days=FINAL_GIVE_UP_D))}
    out = {'generated_at': now.isoformat(timespec='seconds'), 'source': '5DollarFootballAPI · Bet365 goal_line_fixed',
           'attribution': ATTRIBUTION, 'leagues': ids, 'league_misses': misses, 'listed_at': listed_at,
           'last_run': {'calls': budget.calls, 'odds_fetched': fetched, 'rate_remaining': budget.remaining,
                        'stopped': budget.stopped, 'unmatched': unmatched[:30]},
           'fixtures': dict(sorted(fixtures.items(), key=lambda kv: kv[1]['kickoff_utc']))}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    with_odds = sum(1 for v in fixtures.values() if v.get('o05') or v.get('o15'))
    print(f'goal-odds.json: {len(fixtures)} maç ({with_odds} oranlı), bu çalıştırma {budget.calls} istek, '
          f'{fetched} oran, kalan kota {budget.remaining}, eşleşmeyen {len(unmatched)}'
          + (f', durdu: {budget.stopped}' if budget.stopped else ''))
    for u in unmatched[:10]:
        print('  eşleşmedi:', u)
    return out


def load_goal_odds():
    """(lig, ev, deplasman, kickoff_utc) → {'o05','o15','o25'} — build_coupon_rules.py kullanır."""
    if not OUT.exists():
        return {}
    try:
        fx = json.loads(OUT.read_text(encoding='utf-8')).get('fixtures', {})
    except json.JSONDecodeError:
        return {}
    return {(v['league'], v['home'], v['away'], v['kickoff_utc']): v for v in fx.values() if 'kickoff_utc' in v}


if __name__ == '__main__':
    k = os.environ.get('FIVEDOLLAR_API_KEY', '').strip()
    if not k:
        print('FIVEDOLLAR_API_KEY yok — gerçek gol oranı adımı atlandı.')
        sys.exit(0)
    t0 = time.time()
    run(k)
    print(f'{time.time() - t0:.1f} sn')
