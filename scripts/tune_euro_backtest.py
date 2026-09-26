"""Avrupa kupaları için ayrı model + ileriye dönük backtest (iş listesi #5).

Sorun: lig modeli takım gücünü kendi ligine göre ölçer; Premier League'in ortalama takımı ile
Eredivisie'nin ortalama takımı aynı "1,0" puanı alır. Kupada karşılaşınca bu yanlış.

Model (maç: ev H, lig Lh — deplasman A, lig La):
  λ_H = μ_h · hücum_H(ev) · savunma_A(dep) · e^(q_Lh − q_La)
  λ_A = μ_a · hücum_A(dep) · savunma_H(ev) · e^(q_La − q_Lh)
  hücum / savunma: takımın kendi lig modelinde (scripts/goals_model.LeagueModel, hedef sezondan
  ÖNCEKİ 4 sezon) attığı / yediği golün lig ortalamasına oranı.
  μ_h, μ_a (kupa ev / dep gol ortalaması) ve lig katsayıları q_L (Premier League = 0) yalnızca
  hedef sezondan ÖNCEKİ kupa maçlarından Poisson olabilirliğiyle öğrenilir → sızıntı yok.
Yalnızca iki takımı da bizim 10 ligimizden olan, 90 dakikada biten (uzatmasız) maçlar.
Karşılaştırma: q = 0 (lig farkı yok) — kupalarda lig modelini olduğu gibi kullanmak.

Veri: data/euro-results.json (scripts/fetch_euro_results.py), data/football-data CSV'leri.
Kullanım: python scripts/tune_euro_backtest.py
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import load_division, DIVISIONS, PRIOR_WEIGHTS  # noqa: E402
from goals_model import LeagueModel  # noqa: E402
from live_scores import norm  # noqa: E402
from coupon_engine import HL_MIN, hit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CODES = ['1718', '1819', '1920', '2021', '2122', '2223', '2324', '2425', '2526', '2627']
DOMESTIC_HL = {'0.5+': 95.8, '1.5+': 87.3, '2.5+': 81.6, '1X': 87.6, '12': 86.7, 'X2': 84.7}   # sitedeki şerit (lig maçları)


def season_code(iso):
    y, m = int(iso[:4]), int(iso[5:7])
    s = y if m >= 7 else y - 1
    return f'{s % 100:02d}{(s + 1) % 100:02d}'


def pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def probs(lh, la, rho):
    lh, la = max(lh, 0.05), max(la, 0.05)
    hi = min(1 / (lh * la), 1) - 1e-6
    lo = max(-1 / lh, -1 / la) + 1e-6
    rho = min(max(rho, lo), hi)
    g, tot = {}, 0.0
    for x in range(11):
        for y in range(11):
            t = 1.0
            if x == 0 and y == 0: t = 1 - lh * la * rho
            elif x == 0 and y == 1: t = 1 + lh * rho
            elif x == 1 and y == 0: t = 1 + la * rho
            elif x == 1 and y == 1: t = 1 - rho
            p = pmf(x, lh) * pmf(y, la) * max(0, t)
            g[(x, y)] = p
            tot += p
    s = lambda f: sum(p for (x, y), p in g.items() if f(x, y)) / tot
    return {'0.5+': s(lambda x, y: x + y >= 1), '1.5+': s(lambda x, y: x + y >= 2), '2.5+': s(lambda x, y: x + y >= 3),
            '1X': s(lambda x, y: x >= y), '12': s(lambda x, y: x != y), 'X2': s(lambda x, y: y >= x)}


def main():
    euro = json.loads((ROOT / 'data' / 'euro-results.json').read_text(encoding='utf-8'))['matches']
    divs = {div: load_division(div, CODES) for div in DIVISIONS}
    # sezon → norm(takım) → (lig, football-data adı)
    teams = defaultdict(dict)
    for div, rows in divs.items():
        for r in rows:
            for t in (r['home'], r['away']):
                teams[r['season']][norm(t)] = (DIVISIONS[div], t)
    models = {}

    def model(league, code):
        k = (league, code)
        if k not in models:
            div = next(d for d, l in DIVISIONS.items() if l == league)
            i = CODES.index(code)
            priors = CODES[max(0, i - 4):i][::-1]
            by = defaultdict(list)
            for r in divs[div]:
                by[r['season']].append(r)
            seasons = [(by[p], w) for p, w in zip(priors, PRIOR_WEIGHTS) if by.get(p)]
            models[k] = LeagueModel(seasons) if len(seasons) >= 2 else None
        return models[k]

    def resolve(name, code):
        n = norm(name)
        for c in (code, CODES[CODES.index(code) - 1] if CODES.index(code) else code):
            t = teams.get(c, {})
            if n in t:
                return t[n]
            cand = [v for k, v in t.items() if (n in k or k in n) and min(len(n), len(k)) >= 4]
            if len(cand) == 1:
                return cand[0]
        return None

    rows, skipped = [], defaultdict(int)
    for m in euro:
        if m['aet']:
            skipped['uzatma/penaltı'] += 1
            continue
        code = season_code(m['kickoff_utc'])
        if code not in CODES[4:]:
            continue
        h, a = resolve(m['home'], code), resolve(m['away'], code)
        if not h or not a:
            skipped['takım 10 ligimizde değil'] += 1
            continue
        mh, ma = model(h[0], code), model(a[0], code)
        if not mh or not ma or h[1] not in mh.home_gf or a[1] not in ma.away_gf:
            skipped['takımın lig geçmişi yok (yeni çıkan)'] += 1
            continue
        rows.append(dict(code=code, comp=m['comp'], lh=h[0], la=a[0], hg=m['hg'], ag=m['ag'],
                         att_h=mh._avg(mh.home_gf, h[1], mh.base_home) / mh.base_home,
                         def_h=mh._avg(mh.home_ga, h[1], mh.base_away) / mh.base_away,
                         att_a=ma._avg(ma.away_gf, a[1], ma.base_away) / ma.base_away,
                         def_a=ma._avg(ma.away_ga, a[1], ma.base_home) / ma.base_home,
                         rho=(mh.rho + ma.rho) / 2, dom_h=mh.base_home, dom_a=ma.base_away))
    print(f"{len(euro)} kupa maçı; modellenebilen {len(rows)}; dışarıda: {dict(skipped)}")
    by_code = defaultdict(list)
    for r in rows:
        by_code[r['code']].append(r)
    print('sezon başına:', {c: len(v) for c, v in sorted(by_code.items())})

    leagues = sorted(set(DIVISIONS.values()))

    def lam(r, P):
        q = P['q']
        d = q.get(r['lh'], 0) - q.get(r['la'], 0)
        return (P['mh'] * r['att_h'] * r['def_a'] * math.exp(d), P['ma'] * r['att_a'] * r['def_h'] * math.exp(-d))

    def nll(train, P):
        s = 0.0
        for r in train:
            lh, la = lam(r, P)
            s += lh - r['hg'] * math.log(lh) + la - r['ag'] * math.log(la)
        return s

    def fit(train, with_q=True):
        P = {'mh': sum(r['hg'] for r in train) / len(train), 'ma': sum(r['ag'] for r in train) / len(train), 'q': {}}
        if not with_q:
            return P
        for _ in range(6):                          # koordinat inişi, adım küçülerek
            for step in (0.2, 0.08, 0.03, 0.01):
                for L in leagues:
                    if L == 'Premier League' or not any(r['lh'] == L or r['la'] == L for r in train):
                        continue
                    best, cur = nll(train, P), P['q'].get(L, 0)
                    for dv in (-step, step):
                        P['q'][L] = cur + dv
                        v = nll(train, P)
                        if v < best:
                            best, cur = v, cur + dv
                    P['q'][L] = cur
                for k in ('mh', 'ma'):
                    best, cur = nll(train, P), P[k]
                    for dv in (-step / 2, step / 2):
                        P[k] = cur + dv
                        v = nll(train, P)
                        if v < best:
                            best, cur = v, cur + dv
                    P[k] = cur
        return P

    res = {True: defaultdict(lambda: [0, 0, 0.0, 0]), False: defaultdict(lambda: [0, 0, 0.0, 0])}
    last_q = None
    for code in CODES[5:]:
        test = by_code.get(code, [])
        train = [r for c, v in by_code.items() if c < code for r in v]
        if len(train) < 60 or not test:
            continue
        for with_q in (True, False):
            P = fit(train, with_q)
            if with_q:
                last_q = (code, P)
            for r in test:
                lh, la = lam(r, P)
                pr = probs(lh, la, r['rho'])
                for mk, p in pr.items():
                    ok = hit(mk, r['hg'], r['ag'])
                    acc = res[with_q][mk]
                    acc[2] += p; acc[3] += ok        # kalibrasyon: ort. olasılık vs gerçekleşme
                    if p >= HL_MIN[mk]:
                        acc[0] += 1; acc[1] += ok
                res[with_q]['_n'][0] += 1
    n = res[True]['_n'][0]
    print(f"\nTest edilen kupa maçı (2022/23+): {n}")
    print("pazar | lig farkıyla: vurgu n, isabet | lig farkı yok: vurgu n, isabet | lig maçlarında isabet | ort. olasılık → gerçekleşme")
    for mk in HL_MIN:
        a, b = res[True][mk], res[False][mk]
        pa = f"{a[0]:4d}, %{a[1] / a[0] * 100:5.1f}" if a[0] else "   0,    —"
        pb = f"{b[0]:4d}, %{b[1] / b[0] * 100:5.1f}" if b[0] else "   0,    —"
        print(f"{mk:5} | {pa}             | {pb}             | %{DOMESTIC_HL[mk]}                | %{a[2] / n * 100:.1f} → %{a[3] / n * 100:.1f}")
    if last_q:
        code, P = last_q
        print(f"\nSon eğitilen lig katsayıları (≤{code}; PL = 0, pozitif = PL'den güçlü): μ_ev {P['mh']:.2f}, μ_dep {P['ma']:.2f}")
        print('  ' + ', '.join(f"{L} {v:+.2f}" for L, v in sorted(P['q'].items(), key=lambda kv: -kv[1])))


if __name__ == '__main__':
    main()
