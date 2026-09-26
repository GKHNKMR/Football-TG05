"""validate_data.py'nin bozuk veriyi gerçekten yakaladığını sentetik verilerle doğrular (tarayıcısız)."""
import datetime as dt
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_data as V  # noqa: E402

NOW = V.NOW
iso = lambda d: d.strftime('%Y-%m-%dT%H:%M:%SZ')


def match(i, league='Premier League', home=None, away=None, days=5, **kw):
    x = {'match_id': f'T-{i}', 'league': league, 'home': home or f'Ev{i}', 'away': away or f'Dep{i}',
         'kickoff_utc': iso(NOW + dt.timedelta(days=days)), 'lam_home': 1.4, 'lam_away': 1.1,
         'p_over_0_5': 0.93, 'p_over_1_5': 0.75, 'p_over_2_5': 0.5}
    x.update(kw)
    return x


def run(preds, archive=None, results=None):
    V.errors.clear(); V.warnings.clear()
    good = V.check_predictions(preds)
    V.check_same_team(good)
    V.check_league_gaps(good)
    if archive is not None:
        V.check_ungraded(archive, results)
    return list(V.errors), list(V.warnings)


def has(items, text):
    return any(text in s for s in items)


# 1. Temiz veri: hata ve uyarı yok (3 lig aynı hafta başlıyor)
e, w = run([match(1), match(2, 'LaLiga'), match(3, 'Serie A')])
assert not e and not w, (e, w)

# 2. Boş liste -> hata
e, _ = run([])
assert has(e, 'boş'), e

# 3. Eksik alan, bozuk saat, aralık dışı olasılık, bozuk sıra, geçersiz λ, aynı takım -> her biri hata
e, _ = run([match(1, p_over_2_5=None), match(2, kickoff_utc='2026-10-09 17:00'), match(3, p_over_0_5=1.2),
            match(4, p_over_1_5=0.95), match(5, lam_home=0), match(6, home='X', away='X')])
for t in ('eksik alan', 'Geçersiz başlama saati', '0-1 aralığında', 'sırası bozuk', 'Geçersiz lam_home', 'aynı takım'):
    assert has(e, t), (t, e)

# 4. Yinelenen match_id ve aynı maçın iki kez listelenmesi -> hata
a = match(1)
e, _ = run([a, dict(a), dict(a, match_id='T-99')])
assert has(e, 'Yinelenen match_id') and has(e, 'iki kez listelenmiş'), e

# 5. Uyarılar: başlamış ama düşmemiş maç, 48 saatte iki maç, geride kalan lig
e, w = run([match(1, days=-1), match(2, home='Ajax', days=3), match(3, away='Ajax', days=3.5),
            match(4, 'LaLiga', days=3), match(5, 'Serie A', days=20)])
assert not e, e
assert has(w, 'hâlâ bültende') and has(w, 'Ajax 12 saat arayla') and has(w, 'Serie A: sıradaki maç'), w

# 6. Milli ara: tüm ligler birlikte 2 hafta sonra başlıyor -> lig uyarısı yok
_, w = run([match(1, days=14), match(2, 'LaLiga', days=14), match(3, 'Serie A', days=15)])
assert not has(w, 'sıradaki maç'), w

# 7. Notlanmamış maç: 3 gün önceki arşiv maçı results'ta yok -> uyarı; notlanmış olan -> uyarı yok
old = match(7, days=-3); done = match(8, days=-3)
_, w = run([match(1)], {'a': old, 'b': done},
           {'matches': [{k: done[k] for k in ('league', 'home', 'away', 'kickoff_utc')}]})
assert has(w, 'Ev7 - Dep7') and not has(w, 'Ev8 - Dep8'), w

print('✓ validate_data: 7 senaryo (temiz, boş, 6 hata türü, yinelenen, 3 uyarı türü, milli ara, notlanmamış) geçti')
