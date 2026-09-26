"""Takım kara listesi (iş listesi #2) — ileriye dönük (walk-forward) test.

Soru: vurgulu tahminleri geçmişte sistematik olarak tutmayan takımları vurgudan çıkarmak,
GELECEK sezonda isabeti artırıyor mu? Yoksa geçmişteki "kötü takımlar" yalnızca şans mı?

Yöntem (sızıntısız):
  - Her hedef sezon S için kara liste yalnızca S'den ÖNCEKİ sezonlardan çıkarılır.
  - Takım t, pazar m: t'nin oynadığı (ev/deplasman) vurgulu m tahminleri içinde isabet, aynı
    dönemin genel isabetinden anlamlı biçimde düşükse (tek yönlü binom p < ALPHA, en az N_MIN
    vurgu) kara listeye girer.
  - S sezonunda: kara listedeki takımların vurgulu tahminleri çıkarılsaydı isabet ne olurdu?
    Ayrıca kara listedeki takımların S'deki isabeti diğerlerinden gerçekten düşük mü?

Veri: data/stats-5season.json (canlı modelle, yalnızca maç öncesi veriyle üretilmiş olasılıklar).
Kullanım: python scripts/tune_team_blacklist.py
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent

# stats_ui.js ile aynı: [ad, satır indeksi, eşik (binde), olay gerçekleşti mi]
MARKETS = [
    ('0.5+', 6, 935, lambda h, a: h + a >= 1),
    ('1.5+', 7, 830, lambda h, a: h + a >= 2),
    ('2.5+', 8, 750, lambda h, a: h + a >= 3),
    ('1X', 9, 800, lambda h, a: h >= a),
    ('12', 10, 800, lambda h, a: h != a),
    ('X2', 11, 780, lambda h, a: a >= h),
]


def season_of(date):
    y, m = int(date[:4]), int(date[5:7])
    return y if m >= 7 else y - 1


def binom_cdf(k, n, p):
    """P(X <= k), X ~ Bin(n, p)."""
    if n == 0:
        return 1.0
    total, term = 0.0, (1 - p) ** n
    for i in range(k + 1):
        total += term
        term *= (n - i) / (i + 1) * p / (1 - p) if p < 1 else 0
    return min(total, 1.0)


def two_prop_z(h1, n1, h2, n2):
    if not n1 or not n2:
        return float('nan')
    p = (h1 + h2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (h1 / n1 - h2 / n2) / se if se else float('nan')


def picks(rows):
    """Her vurgulu tahmin: (sezon, pazar, ev, dep, tuttu mu)."""
    out = []
    for r in rows:
        if r[12]:           # kısıtlı veri asla vurgulanmaz
            continue
        s = season_of(r[0])
        for name, i, thr, hit in MARKETS:
            if r[i] >= thr:
                out.append((s, name, r[2], r[3], hit(r[4], r[5])))
    return out


def blacklist(train, n_min, alpha, pooled):
    """Eğitim dönemindeki vurgulardan kara liste: {(pazar, takım)} veya pooled ise {(None, takım)}."""
    base = defaultdict(lambda: [0, 0])
    team = defaultdict(lambda: [0, 0])
    for s, m, h, a, ok in train:
        key_m = None if pooled else m
        base[key_m][0] += ok; base[key_m][1] += 1
        for t in (h, a):
            team[(key_m, t)][0] += ok; team[(key_m, t)][1] += 1
    bl = {}
    for (m, t), (k, n) in team.items():
        p0 = base[m][0] / base[m][1]
        if n >= n_min and k / n < p0 and binom_cdf(k, n, p0) < alpha:
            bl[(m, t)] = (k, n, p0)
    return bl


def evaluate(all_picks, seasons, n_min, alpha, pooled, min_train_seasons, verbose=False):
    tot = {'kept': [0, 0], 'drop': [0, 0], 'all': [0, 0]}
    per_season = []
    for S in seasons:
        train = [p for p in all_picks if p[0] < S]
        if len({p[0] for p in train}) < min_train_seasons:
            continue
        bl = blacklist(train, n_min, alpha, pooled)
        kept, drop = [0, 0], [0, 0]
        for s, m, h, a, ok in all_picks:
            if s != S:
                continue
            key_m = None if pooled else m
            hit_bl = (key_m, h) in bl or (key_m, a) in bl
            b = drop if hit_bl else kept
            b[0] += ok; b[1] += 1
        allh, alln = kept[0] + drop[0], kept[1] + drop[1]
        per_season.append((S, len(bl), kept, drop, allh, alln))
        for k, v in (('kept', kept), ('drop', drop), ('all', [allh, alln])):
            tot[k][0] += v[0]; tot[k][1] += v[1]
        if verbose:
            teams = sorted({t for (_, t) in bl})
            print(f"    {S}/{str(S + 1)[2:]}: kara liste {len(bl)} (takım×pazar), {len(teams)} takım"
                  f" — ör. {', '.join(teams[:6])}{'…' if len(teams) > 6 else ''}")
    return tot, per_season


def pct(h, n):
    return f"{100 * h / n:5.1f}%" if n else '   — '


def main():
    d = json.loads((ROOT / 'data' / 'stats-5season.json').read_text(encoding='utf-8'))
    all_picks = picks(d['rows'])
    seasons = sorted({p[0] for p in all_picks})
    print(f"{len(d['rows'])} maç, {len(all_picks)} vurgulu tahmin, sezonlar {seasons[0]}/{seasons[0] + 1} – {seasons[-1]}/{seasons[-1] + 1}\n")

    print("Ayar                                  | kara listedeki takımların     | kalanların | hepsi  | fark (kalan − hepsi) | z (kara vs kalan)")
    print("                                      | S sezonundaki vurgu / isabet  | isabeti    |        |                      |")
    results = []
    for pooled in (False, True):
        for n_min in (15, 25, 40):
            for alpha in (0.05, 0.01):
                for mts in (1, 2):
                    tot, _ = evaluate(all_picks, seasons, n_min, alpha, pooled, mts)
                    k, dr, al = tot['kept'], tot['drop'], tot['all']
                    z = two_prop_z(dr[0], dr[1], k[0], k[1])
                    diff = (k[0] / k[1] - al[0] / al[1]) * 100 if k[1] and al[1] else 0
                    label = f"{'tüm pazarlar' if pooled else 'pazar bazında'}, n≥{n_min}, p<{alpha}, ≥{mts} sezon eğitim"
                    results.append((label, tot, z, diff, pooled, n_min, alpha, mts))
                    print(f"{label:<38}| {dr[1]:5d} vurgu, isabet {pct(*dr)}      | {pct(*k)}     | {pct(*al)} | {diff:+6.2f} puan          | {z:+.2f}")

    # Pazar karışımı yanılsamasını ayıkla: kara liste ile kalanları AYNI pazar içinde karşılaştır
    print("\nPazar içi karşılaştırma (pazar bazında, n≥25, p<0.05, ≥1 sezon eğitim):")
    print("  pazar | kara liste: vurgu, isabet | kalan: vurgu, isabet | fark (kara − kalan)")
    by_m = defaultdict(lambda: {'kept': [0, 0], 'drop': [0, 0]})
    for S in seasons:
        train = [p for p in all_picks if p[0] < S]
        if not train:
            continue
        bl = blacklist(train, 25, 0.05, False)
        for s, m, h, a, ok in all_picks:
            if s == S:
                b = by_m[m]['drop' if (m, h) in bl or (m, a) in bl else 'kept']
                b[0] += ok; b[1] += 1
    for name, *_ in MARKETS:
        k, dr = by_m[name]['kept'], by_m[name]['drop']
        diff = (dr[0] / dr[1] - k[0] / k[1]) * 100 if dr[1] and k[1] else float('nan')
        print(f"  {name:5} | {dr[1]:5d}, {pct(*dr)}            | {k[1]:5d}, {pct(*k)}      | {diff:+6.1f} puan  (z {two_prop_z(dr[0], dr[1], k[0], k[1]):+.2f})")

    best = max(results, key=lambda r: r[3])
    print(f"\nEn iyi ayar: {best[0]} → isabet {best[3]:+.2f} puan, z = {best[2]:+.2f}")
    print("Bu ayarda sezon sezon:")
    evaluate(all_picks, seasons, best[5], best[6], best[4], best[7], verbose=True)


if __name__ == '__main__':
    main()
