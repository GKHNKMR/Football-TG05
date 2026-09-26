"""Kasa hedefine göre kupon (iş listesi #21) — tek kaynak: hem geçmiş testi hem canlı öneri buradan.

Kural: Fikstür'deki VURGULU tahminler (sitedeki eşiklerle: 0.5+ ≥ %93,5 · 1.5+ ≥ %83 · 2.5+ ≥ %75 ·
1X/12 ≥ %80 · X2 ≥ %78; kısıtlı veri hiç, kritik eksik oyunculu maç Çifte Şans'ta vurgulanmaz).
Her maçtan en fazla bir tahmin, en fazla 5 maç; oranların çarpımı gereken orana (R) ulaşan
kombinasyonlar içinden TUTMA OLASILIĞI (model olasılıklarının çarpımı) EN YÜKSEK olanı seçilir
(log-oran ızgarasında dinamik programlama). R'ye ulaşılamayan gün kupon yok.
"""
import math

MAX_LEGS = 5
HL_MIN = {'0.5+': 0.935, '1.5+': 0.83, '2.5+': 0.75, '1X': 0.80, '12': 0.80, 'X2': 0.78}
STEP = 0.004          # log-oran ızgarası (≈ %0,4 oran adımı)


def pick_coupon(matches, R, max_legs=MAX_LEGS):
    """matches: [[aday, ...], ...] — her maçın vurgulu adayları; aday = dict(p=, odds=, ...).
    Dönen: seçilen adaylar listesi (oran çarpımı ≥ R, olasılık çarpımı en büyük) ya da None."""
    need = int(math.ceil(math.log(R) / STEP - 1e-9))
    # dp[k][b] = (log p toplamı, seçimler)  — b: min(need, toplam log-oran ızgara adımı)
    dp = [dict() for _ in range(max_legs + 1)]
    dp[0][0] = (0.0, ())
    for cands in matches:
        if not cands:
            continue
        for k in range(max_legs - 1, -1, -1):
            for b, (lp, sel) in list(dp[k].items()):
                for c in cands:
                    if c['odds'] <= 1 or c['p'] <= 0:
                        continue
                    nb = min(need, b + int(math.log(c['odds']) / STEP))
                    nlp = lp + math.log(c['p'])
                    cur = dp[k + 1].get(nb)
                    if cur is None or nlp > cur[0]:
                        dp[k + 1][nb] = (nlp, sel + (c,))
    best = None
    for k in range(1, max_legs + 1):
        v = dp[k].get(need)
        if v and (best is None or v[0] > best[0]):
            best = v
    if not best:
        return None
    legs = list(best[1])
    if math.prod(c['odds'] for c in legs) < R - 1e-9:   # ızgara yuvarlaması: gerçek çarpım da R'yi geçmeli
        return None
    return legs


# ---- Çifte Şans olasılıkları: js/cifte_engine.js ile aynı (Dixon-Coles düzeltmeli 10×10 skor ızgarası) ----
def _pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def dc_probs(lam_h, lam_a, rho):
    lam_h, lam_a = max(lam_h, 1e-6), max(lam_a, 1e-6)
    hi = min(1.0 / (lam_h * lam_a), 1.0) - 1e-6
    lo = max(-1.0 / lam_h, -1.0 / lam_a) + 1e-6
    rho = min(max(rho, lo), hi)

    def tau(x, y):
        if x == 0 and y == 0: return 1 - lam_h * lam_a * rho
        if x == 0 and y == 1: return 1 + lam_h * rho
        if x == 1 and y == 0: return 1 + lam_a * rho
        if x == 1 and y == 1: return 1 - rho
        return 1.0
    ph, pd, pa, tot = 0.0, 0.0, 0.0, 0.0
    for x in range(10):
        for y in range(10):
            p = _pmf(x, lam_h) * _pmf(y, lam_a) * max(0.0, tau(x, y))
            tot += p
            if x > y: ph += p
            elif x == y: pd += p
            else: pa += p
    ph, pd, pa = ph / tot, pd / tot, pa / tot
    return {'1X': round(ph + pd, 3), '12': round(ph + pa, 3), 'X2': round(pd + pa, 3)}


def hit(market, hg, ag):
    t = hg + ag
    return {'0.5+': t >= 1, '1.5+': t >= 2, '2.5+': t >= 3, '1X': hg >= ag, '12': hg != ag, 'X2': ag >= hg}[market]
