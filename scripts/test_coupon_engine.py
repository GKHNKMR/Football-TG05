"""coupon_engine (iş listesi #21): kupon seçimi ve sonuçlandırma kuralları (tarayıcısız)."""
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coupon_engine import pick_coupon, hit, dc_probs  # noqa: E402

c = lambda m, p, o, tag: dict(market=m, p=p, odds=o, tag=tag)

# 1. Gereken orana ulaşan kombinasyonlar içinden tutma olasılığı en yüksek olanı seçer
day = [[c('0.5+', .97, 1.03, 'A05'), c('1X', .85, 1.15, 'A1X')],
       [c('1.5+', .88, 1.10, 'B15')],
       [c('2.5+', .76, 1.30, 'C25')],
       [c('0.5+', .96, 1.04, 'D05')]]
legs = pick_coupon(day, 1.25)
odds = math.prod(l['odds'] for l in legs)
assert odds >= 1.25, odds
# adaylar: A1X+B15 (1,265 · .748) ; C25 (1,30 · .76) ; A05+C25 ... C25 tek başına en iyisi
assert [l['tag'] for l in legs] == ['C25'], [l['tag'] for l in legs]

# 2. Her maçtan en fazla bir tahmin, en fazla 5 maç
legs = pick_coupon([[c('0.5+', .97, 1.03, f'M{i}a'), c('1.5+', .9, 1.08, f'M{i}b')] for i in range(8)], 1.40)
assert legs and len(legs) <= 5 and len({l['tag'][:2] for l in legs}) == len(legs), legs
assert math.prod(l['odds'] for l in legs) >= 1.40

# 3. Ulaşılamıyorsa kupon yok
assert pick_coupon([[c('0.5+', .97, 1.02, 'x')]] * 5, 1.30) is None

# 4. Sonuçlandırma
assert hit('0.5+', 1, 0) and not hit('0.5+', 0, 0)
assert hit('1.5+', 1, 1) and not hit('1.5+', 1, 0)
assert hit('2.5+', 2, 1) and not hit('2.5+', 1, 1)
assert hit('1X', 1, 1) and not hit('1X', 0, 1)
assert hit('12', 2, 0) and not hit('12', 1, 1)
assert hit('X2', 0, 0) and not hit('X2', 1, 0)

# 5. Çifte Şans olasılıkları js/cifte_engine.js ile aynı mantık: toplam 2 (1X + 12 + X2 = 2)
d = dc_probs(2.235, 1.248, -0.11)
assert abs(d['1X'] + d['12'] + d['X2'] - 2) < 0.005 and d['1X'] > d['X2'], d
print('✓ coupon_engine: seçim, 5 maç sınırı, ulaşılamayan gün, sonuçlandırma, Çifte Şans olasılıkları')
