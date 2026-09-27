"""fetch_goal_odds.py (iş listesi #20): sahte 5DollarFootballAPI ile lig bulma, fikstür eşleme,
oran ayrıştırma, çekme zamanlaması / kota ve kupon adayına gerçek oranın geçmesi (ağsız)."""
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_goal_odds as G  # noqa: E402
import build_coupon_rules as B  # noqa: E402

NOW = datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc)
iso = lambda dt: dt.strftime('%Y-%m-%dT%H:%M:%SZ')
KO1, KO2, KO3 = NOW + timedelta(hours=5), NOW + timedelta(hours=30), NOW + timedelta(hours=8)
preds = [
    dict(league='Premier League', home='Wolverhampton Wanderers', away='Manchester United', kickoff_utc=iso(KO1),
         p_over_0_5=0.96, p_over_1_5=0.86, p_over_2_5=0.6, basis='form+h2h', h2h_matches_used=4),
    dict(league='Premier League', home='Arsenal', away='Chelsea', kickoff_utc=iso(KO2),
         p_over_0_5=0.94, p_over_1_5=0.84, p_over_2_5=0.6, basis='form+h2h', h2h_matches_used=4),
    dict(league='Bundesliga', home='Bayern München', away='Köln', kickoff_utc=iso(KO3),
         p_over_0_5=0.97, p_over_1_5=0.9, p_over_2_5=0.8, basis='form+h2h', h2h_matches_used=4),
]
calls = []


def fx(i, lg, h, a, ko):
    return {'id': i, 'league': {'id': lg}, 'teams': {'home': {'name': h}, 'away': {'name': a}},
            'kickoff_utc': ko.isoformat(), 'status': 'scheduled'}


def fake(budget, path, params=None):
    if not budget.ok():
        return None
    budget.left -= 1
    budget.calls += 1
    calls.append(path)
    budget.remaining = 50
    if path == '/leagues':
        c = params.get('country')
        # gerçek API biçimi (27.09 canlı teşhis): ülke önekli adlar
        data = {'GB-ENG': [{'id': 11, 'name': 'England Premier League', 'country': {'code': 'GB-ENG', 'name': 'England'}},
                           {'id': 12, 'name': 'England Premier League 2', 'country': {'code': 'GB-ENG', 'name': 'England'}}],
                'DE': [{'id': 21, 'name': 'Germany Bundesliga I', 'country': {'code': 'DE', 'name': 'Germany'}},
                       {'id': 22, 'name': 'Germany Bundesliga II', 'country': {'code': 'DE', 'name': 'Germany'}}]}.get(c, [])
        return {'success': 1, 'data': data}
    if path == '/leagues/11/fixtures':
        return {'success': 1, 'data': [fx(101, 11, 'Wolves', 'Manchester Utd', KO1), fx(102, 11, 'Arsenal', 'Chelsea', KO2),
                                       fx(103, 11, 'Everton', 'Fulham', KO1)]}
    if path == '/leagues/21/fixtures':
        return {'_error': 403, '_body': 'insufficient_plan'}
    if path == '/fixtures':                      # 403 sonrası tarih listesi yedeği
        return {'success': 1, 'data': [fx(201, 21, 'Bayern Munich', 'FC Cologne', KO3)]}
    if path.endswith('/odds'):
        fid = int(path.split('/')[2])
        line = lambda l, o, c: {'line': l, 'opening': {'over': o, 'under': 5}, 'closing': {'over': c, 'under': 4} if c else None, 'inplay': None}
        g = {101: [line(0.5, 1.05, 1.06), line(1.5, 1.28, 1.3), None, line(2.5, 1.9, 1.85)],
             102: [line(0.5, 1.07, None), line(1.5, 1.33, None)], 201: []}[fid]
        return {'success': 1, 'data': {'fixture_id': fid, 'bookmakers': [{'slug': 'bet365', 'odds': {'goal_line_fixed': g}}]}}
    raise AssertionError(path)


with tempfile.TemporaryDirectory() as td:
    G.OUT = Path(td) / 'goal-odds.json'
    G.PRED = Path(td) / 'predictions.json'
    G.PRED.write_text(json.dumps(preds, ensure_ascii=False), encoding='utf-8')
    G.api_get = fake
    out = G.run('k', now=NOW)
    f = out['fixtures']
    assert out['leagues'] == {'Premier League': 11, 'Bundesliga': 21}, out['leagues']
    print('✓ Lig kimliği adla bulunuyor (Premier League 2 / 2. Bundesliga seçilmiyor).')
    assert f['101']['home'] == 'Wolverhampton Wanderers' and f['201']['away'] == 'Köln'
    assert '103' not in f and 'Everton - Fulham' in out['last_run']['unmatched'][0]
    print('✓ Eşleme: Wolves/Manchester Utd, Bayern Munich/FC Cologne bizim adlara oturuyor; fikstürümüzde olmayan maç alınmıyor.')
    assert f['101']['o05'] == 1.06 and f['101']['o15'] == 1.3 and f['101']['o25'] == 1.85
    assert f['102']['o05'] == 1.07 and f['102']['o15'] == 1.33 and 'o05' not in f['201']
    print('✓ Oran: kapanış fiyatı, yoksa açılış; geri çekilen çizgi (null) ve boş liste sorun çıkarmıyor.')
    assert '/fixtures' in calls, '403 sonrası tarih listesine düşülmeli'

    # Aynı saat tekrar: liste ve taze oran yeniden çekilmez
    calls.clear()
    G.run('k', now=NOW + timedelta(minutes=30))
    assert calls == [], calls
    # 4 saat sonra: KO1 (1 sa kaldı) yenilenir, KO2 (25 sa) 6 saat dolmadığı için yenilenmez
    calls.clear()
    G.run('k', now=NOW + timedelta(hours=4))
    assert '/fixtures/101/odds' in calls and '/fixtures/102/odds' not in calls, calls
    print('✓ Zamanlama: maç yaklaştıkça sık, uzaktaki maç 6 saatte bir; liste 3 saatte bir.')
    # Maç bittikten sonra kapanış oranı bir kez
    calls.clear()
    G.run('k', now=KO1 + timedelta(hours=3))
    calls2 = list(calls)
    calls.clear()
    G.run('k', now=KO1 + timedelta(hours=4))
    assert '/fixtures/101/odds' in calls2 and '/fixtures/101/odds' not in calls
    assert json.loads(G.OUT.read_text(encoding='utf-8'))['fixtures']['101'].get('final') is True
    print('✓ Maç bitince kapanış oranı bir kez çekilip saklanıyor.')

    # Kota: bütçe biterse durur
    b = G.Budget('k', max_calls=2)
    for _ in range(5):
        fake(b, '/fixtures/101/odds')
    assert b.calls == 2
    print('✓ Çalıştırma başına istek sınırı uygulanıyor.')

    # Kupon adayı: gerçek oran varsa real=True ve o fiyat
    go = G.load_goal_odds()
    x = dict(preds[0], lam_home=1.6, lam_away=1.3, rho=0.0)
    c = {d['market']: d for d in B.live_candidates(x, {}, go)}
    assert c['0.5+']['odds'] == 1.06 and c['0.5+']['real'] and c['1.5+']['odds'] == 1.3 and c['1.5+']['real']
    c2 = {d['market']: d for d in B.live_candidates(x, {}, {})}
    assert not c2['0.5+']['real']
    print('✓ Kupon: Bet365 gol oranı gelince 0.5+/1.5+ gerçek oranla (tahmini değil) kuruluyor.')
print('\nTüm gol oranı testleri geçti.')

# Lig adı esnekliği
ok = lambda n, w: G.league_name_ok({'name': n}, w)
for n, w in [('England Premier League', 'PREMIER LEAGUE'), ('Spain La Liga', 'LA LIGA'), ('Germany Bundesliga I', 'BUNDESLIGA'),
             ('Italy Serie A', 'SERIE A'), ('France Ligue 1', 'LIGUE 1'), ('LaLiga', 'LA LIGA')]:
    assert ok(n, w), n
for n, w in [('England Premier League 2', 'PREMIER LEAGUE'), ('Germany Bundesliga II', 'BUNDESLIGA'), ('Italy Serie B', 'SERIE A'),
             ('France Ligue 2', 'LIGUE 1')]:
    assert not ok(n, w), n
print('✓ Lig adı eşlemesi: API gerçek adları (England Premier League, Germany Bundesliga I vb.) kabul, alt ligler red.')
