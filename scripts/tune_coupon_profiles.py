"""Kasa risk temelli kupon önerisi (iş listesi #3) — önce öneri mantığının backtest'i.

Soru: Sanal Kasa'nın risk profilleri (Minimum / Orta / Yüksek) için "hangi pazar, kaç maç,
hangi olasılık eşiği" kuralı, GERÇEK oranlarla oynansaydı geçmişte ne yapardı?

Olasılıklar: data/stats-5season.json — canlı modelle, yalnızca maç öncesi veriyle (sızıntısız).
Oranlar (football-data.co.uk CSV, piyasa ortalaması "Avg"):
  2.5+          gerçek oran (Avg>2.5)
  1X / 12 / X2  1X2 ortalamasından türetilmiş: 1 / (1/ev + 1/beraberlik) vb. (1X2 marjı içinde kalır)
  0.5+ / 1.5+   GERÇEK ORAN YOK → TAHMİN: 2.5 Üst/Alt fiyatından piyasanın gol beklentisi (Poisson λ)
                çıkarılır, P(>0.5), P(>1.5) hesaplanır, 2.5 pazarındaki marj uygulanır. Rapor "tahmini" der.

Kupon: her gün (tarih), profil kuralına uyan en yüksek olasılıklı N maç; yeterli maç yoksa o gün kupon yok.
Ölçüler: kupon sayısı, tutma oranı, ortalama kupon oranı, düz bahiste ROI, kasa simülasyonu
(profilin kasa payı ile bileşik), en kötü düşüş. Ayar seçimi 2021–2024, doğrulama 2025/26–2026/27.

Kullanım: python scripts/tune_coupon_profiles.py
"""
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teams import to_pretty  # noqa: E402
from backtest import DIVISIONS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / 'data' / 'football-data'
CSV_SEASONS = ['2122', '2223', '2324', '2425', '2526', '2627']
TRAIN_UNTIL = 2024          # 2021/22–2024/25 ayar seçimi; 2025/26 + 2026/27 doğrulama

# stats-5season satır indeksi ve kazanma koşulu
MARKETS = {
    '0.5+': (6, lambda h, a: h + a >= 1),
    '1.5+': (7, lambda h, a: h + a >= 2),
    '2.5+': (8, lambda h, a: h + a >= 3),
    '1X': (9, lambda h, a: h >= a),
    '12': (10, lambda h, a: h != a),
    'X2': (11, lambda h, a: a >= h),
}

# Kodda bugün tanımlı profiller (js/paper_engine.js COUPON_CLASSES) — hedef oran = profilin varsaydığı
CURRENT = {
    'Minimum (bugünkü)': dict(markets=['0.5+'], thr=0.95, legs=5, stake=0.25, assumed_odds=1.28),
    'Orta (bugünkü)': dict(markets=['1.5+'], thr=0.85, legs=3, stake=0.50, assumed_odds=1.42),
    'Yüksek (bugünkü)': dict(markets=['0.5+', '1.5+'], thr=0.85, legs=5, stake=0.50, assumed_odds=1.35),
}


def season_of(date):
    y, m = int(date[:4]), int(date[5:7])
    return y if m >= 7 else y - 1


def poisson_over(lam, line):
    k = int(line)            # P(X > line) = 1 - P(X <= floor(line))
    cdf, term = 0.0, math.exp(-lam)
    for i in range(k + 1):
        cdf += term
        term *= lam / (i + 1)
    return 1 - cdf


def lam_from_p25(p):
    lo, hi = 0.2, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if poisson_over(mid, 2.5) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fnum(v):
    try:
        x = float(v)
        return x if x > 1 else None
    except (TypeError, ValueError):
        return None


def load_odds():
    """(lig, tarih, ev, dep) -> {pazar: (oran, gerçek mi)}"""
    out = {}
    for div, league in DIVISIONS.items():
        for sz in CSV_SEASONS:
            p = CSV_DIR / div / f'{sz}.csv'
            if not p.exists():
                continue
            with p.open(encoding='utf-8-sig') as fh:
                for r in csv.DictReader(fh):
                    try:
                        date = datetime.strptime(r['Date'].strip(), '%d/%m/%Y').date().isoformat()
                    except (KeyError, ValueError):
                        continue
                    H, D, A = (fnum(r.get(k)) or fnum(r.get(b)) for k, b in (('AvgH', 'B365H'), ('AvgD', 'B365D'), ('AvgA', 'B365A')))
                    O, U = fnum(r.get('Avg>2.5')) or fnum(r.get('B365>2.5')), fnum(r.get('Avg<2.5')) or fnum(r.get('B365<2.5'))
                    odds = {}
                    if H and D and A:
                        odds['1X'] = (1 / (1 / H + 1 / D), 'türetilmiş')
                        odds['12'] = (1 / (1 / H + 1 / A), 'türetilmiş')
                        odds['X2'] = (1 / (1 / D + 1 / A), 'türetilmiş')
                    if O and U:
                        odds['2.5+'] = (O, 'gerçek')
                        book = 1 / O + 1 / U                        # >1: marj
                        lam = lam_from_p25((1 / O) / book)
                        for mk, line in (('0.5+', 0.5), ('1.5+', 1.5)):
                            fair = poisson_over(lam, line)
                            odds[mk] = (max(1.01, 1 / min(0.999, fair * book)), 'tahmini')
                    key = (league, date, to_pretty(league, r['HomeTeam'].strip()), to_pretty(league, r['AwayTeam'].strip()))
                    out[key] = odds
    return out


def legs_by_day(rows, odds):
    """tarih -> [(pazar, olasılık, oran, oran türü, tuttu mu, maç anahtarı)] ; kısıtlı veri hariç"""
    days = defaultdict(list)
    miss = 0
    for r in rows:
        if r[12]:
            continue
        key = (r[1], r[0], r[2], r[3])
        o = odds.get(key)
        if not o:
            miss += 1
            continue
        for mk, (i, hit) in MARKETS.items():
            if mk in o:
                days[r[0]].append((mk, r[i] / 1000, o[mk][0], o[mk][1], hit(r[4], r[5]), key))
    return days, miss


def build_coupons(days, markets, thr, legs):
    """Her gün: kurala uyan en yüksek olasılıklı `legs` bacak (aynı maçtan tek bacak)."""
    coupons = []
    for date in sorted(days):
        cands = sorted((l for l in days[date] if l[0] in markets and l[1] >= thr), key=lambda l: -l[1])
        chosen, used = [], set()
        for l in cands:
            if l[5] in used:
                continue
            chosen.append(l); used.add(l[5])
            if len(chosen) == legs:
                break
        if len(chosen) < legs:
            continue
        odds = math.prod(l[2] for l in chosen)
        coupons.append(dict(date=date, season=season_of(date), won=all(l[4] for l in chosen), odds=odds,
                            p=math.prod(l[1] for l in chosen), kinds={l[3] for l in chosen}))
    return coupons


def summarize(coupons, stake_pct, assumed_odds=None):
    if not coupons:
        return None
    n = len(coupons)
    won = sum(c['won'] for c in coupons)
    avg_odds = sum(c['odds'] for c in coupons) / n
    roi = sum((c['odds'] if c['won'] else 0) for c in coupons) / n - 1
    model_p = sum(c['p'] for c in coupons) / n
    # kasa: her sezon 100'den başlar, her kupona kasanın stake_pct'i
    finals, dds = [], []
    for s in sorted({c['season'] for c in coupons}):
        bank, peak, dd = 100.0, 100.0, 0.0
        for c in (c for c in coupons if c['season'] == s):
            st = bank * stake_pct
            bank += st * (c['odds'] - 1) if c['won'] else -st
            peak = max(peak, bank); dd = max(dd, 1 - bank / peak)
        finals.append(bank); dds.append(dd)
    out = dict(n=n, win=won / n, model_p=model_p, avg_odds=avg_odds, roi=roi,
               bank_med=sorted(finals)[len(finals) // 2], bank_min=min(finals), dd_max=max(dds),
               kinds=set().union(*(c['kinds'] for c in coupons)))
    if assumed_odds:
        out['roi_assumed'] = sum((assumed_odds if c['won'] else 0) for c in coupons) / n - 1
    return out


def line(label, s):
    if not s:
        return f"{label:<34} (yeterli maç yok)"
    k = '/'.join(sorted(s['kinds']))
    extra = f" | varsayılan oranla ROI {s['roi_assumed']:+.1%}" if 'roi_assumed' in s else ''
    extra += f" | ADİL oran (1/tutma) {1 / s['win']:.2f}" if s['win'] else ''
    return (f"{label:<34} {s['n']:5d} kupon | tuttu {s['win']:.1%} (model {s['model_p']:.1%}) | ort. oran {s['avg_odds']:.2f} ({k})"
            f" | ROI {s['roi']:+.1%} | sezon sonu kasa (100'den) ortanca {s['bank_med']:.0f}, en kötü {s['bank_min']:.0f}"
            f" | en büyük düşüş {s['dd_max']:.0%}{extra}")


def main():
    d = json.loads((ROOT / 'data' / 'stats-5season.json').read_text(encoding='utf-8'))
    odds = load_odds()
    days, miss = legs_by_day(d['rows'], odds)
    print(f"{len(d['rows'])} maç; oranı bulunamayan {miss}; {len(days)} gün\n")

    print("=== 1) Kodda bugün tanımlı profiller, tüm sezonlar ===")
    for name, cfg in CURRENT.items():
        c = build_coupons(days, cfg['markets'], cfg['thr'], cfg['legs'])
        print(line(name, summarize(c, cfg['stake'], cfg['assumed_odds'])))

    print("\n=== 2) Ayar taraması (seçim 2021/22–2024/25) ===")
    grid = []
    for mk_set in (['0.5+'], ['1.5+'], ['2.5+'], ['1X'], ['12'], ['X2'], ['1X', '12', 'X2'], ['0.5+', '1.5+'],
                   ['0.5+', '1.5+', '1X', '12', 'X2']):
        for thr in (0.75, 0.80, 0.85, 0.90, 0.93, 0.95):
            for legs in (1, 2, 3, 4, 5):
                c = build_coupons(days, mk_set, thr, legs)
                tr = [x for x in c if x['season'] <= TRAIN_UNTIL]
                te = [x for x in c if x['season'] > TRAIN_UNTIL]
                s_tr, s_te = summarize(tr, 0.25), summarize(te, 0.25)
                if s_tr and s_tr['n'] >= 100 and s_te and s_te['n'] >= 20:
                    grid.append(('+'.join(mk_set), thr, legs, s_tr, s_te))
    grid.sort(key=lambda g: -g[3]['roi'])
    print("En iyi 12 ayar (eğitim ROI'sine göre) ve doğrulama dönemindeki sonuçları:")
    for mk, thr, legs, s_tr, s_te in grid[:12]:
        print(f"  {mk:<22} ≥{thr:.2f} {legs} maç | eğitim: {s_tr['n']:4d} kupon, tuttu {s_tr['win']:.1%}, oran {s_tr['avg_odds']:.2f}, ROI {s_tr['roi']:+.1%}"
              f" | doğrulama: {s_te['n']:4d} kupon, tuttu {s_te['win']:.1%}, ROI {s_te['roi']:+.1%}")
    pos_tr = sum(1 for g in grid if g[3]['roi'] > 0)
    pos_both = sum(1 for g in grid if g[3]['roi'] > 0 and g[4]['roi'] > 0)
    print(f"\n{len(grid)} ayar: eğitimde ROI>0 olan {pos_tr}, hem eğitimde hem doğrulamada ROI>0 olan {pos_both}")

    print("\n=== 3) Gerçek oranlı pazarlarda (2.5+, Çifte Şans) risk basamakları, tüm sezonlar ===")
    for label, mks, thr, legs in (('Düşük risk: 1 maç ÇŞ ≥%85', ['1X', '12', 'X2'], 0.85, 1),
                                  ('Düşük risk: 2 maç ÇŞ ≥%85', ['1X', '12', 'X2'], 0.85, 2),
                                  ('Orta risk: 2 maç ÇŞ ≥%80', ['1X', '12', 'X2'], 0.80, 2),
                                  ('Orta risk: 3 maç ÇŞ ≥%80', ['1X', '12', 'X2'], 0.80, 3),
                                  ('Yüksek risk: 1 maç 2.5+ ≥%75', ['2.5+'], 0.75, 1),
                                  ('Yüksek risk: 2 maç 2.5+ ≥%70', ['2.5+'], 0.70, 2)):
        print(line(label, summarize(build_coupons(days, mks, thr, legs), 0.25)))


if __name__ == '__main__':
    main()
