"""Kupon önerisi (iş listesi #3, #21) → data/coupon-rules.json + data/coupons.json

coupon-rules.json — Sanal Kasa kartının kuralları ve ÖLÇÜLMÜŞ geçmişi (site kendi kafasına göre
    oran/başarı yazmaz):
    tiers   sabit Çifte Şans kuralları (1 / 2 / 3 maç), scripts/tune_coupon_profiles.py taramasından.
    target  "kasa hedefine göre kupon" kuralının (scripts/coupon_engine.py) gereken oran ızgarasında
            (1,05–2,00) geçmiş sonuçları + her kuponun (oran, tuttu mu) listesi — kart kişinin kendi
            kasasıyla tarayıcıda Monte Carlo yapar.
coupons.json — Minimum / Orta / Yüksek kasa için her maç gününün ÖNERİLEN kuponu (canlı fikstürden)
    ve geçmişi: kupondaki ilk maç başlayınca kupon dondurulur, maçlar bitince tuttu / tutmadı.

Geçmiş: data/stats-5season.json (canlı model, yalnızca maç öncesi veri) + football-data.co.uk piyasa
ortalaması oranları (2.5+ ve Çifte Şans gerçek; 0.5+/1.5+ gerçek oran verisi yok → 2.5 fiyatından
tahmini, bkz. iş listesi #20). Saatlik bot her gün çalıştırır.
Kullanım: python scripts/build_coupon_rules.py
"""
import json
import math
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_coupon_profiles import (ROOT, load_odds, legs_by_day, build_coupons, summarize, season_of,  # noqa: E402
                                  poisson_over, lam_from_p25)
from coupon_engine import MAX_LEGS, HL_MIN, pick_coupon, dc_probs, hit  # noqa: E402

DC = ['1X', '12', 'X2']
GOAL = {'0.5+': 'p_over_0_5', '1.5+': 'p_over_1_5', '2.5+': 'p_over_2_5'}
TIERS = [
    dict(id='low', profile='minimum', markets=DC, thr=0.85, legs=1),
    dict(id='mid', profile='medium', markets=DC, thr=0.80, legs=2),
    dict(id='high', profile='high', markets=DC, thr=0.80, legs=3),
]
# js/paper_engine.js RISK_PROFILES: kupon payı f = 1 − rezerv; gereken oran R = 1 + günlük büyüme / f
PROFILES = {'minimum': dict(g=0.10, f=0.25), 'medium': dict(g=0.15, f=0.50), 'high': dict(g=0.25, f=0.50)}
for v in PROFILES.values():
    v['R'] = round(1 + v['g'] / v['f'], 4)
TARGET_GRID = [round(1.05 + 0.05 * i, 2) for i in range(20)]   # 1,05 … 2,00
STAKE_PCT = 0.25
IST = timedelta(hours=3)          # Europe/Istanbul (2016'dan beri sabit UTC+3)
KEEP_DAYS = 180


def ist_day(iso):
    return (datetime.fromisoformat(iso.replace('Z', '+00:00')) + IST).date().isoformat()


# ---------------------------------------------------------------- geçmiş
def hist_candidates(days):
    """gün → [[aday, ...] maç başına]; yalnızca vurgu eşiğini geçen tahminler."""
    out = {}
    for d, legs in days.items():
        by = defaultdict(list)
        for mk, p, o, kind, won, key in legs:
            if p >= HL_MIN[mk] - 1e-9:
                by[key].append(dict(market=mk, p=p, odds=o, kind=kind, won=won))
        out[d] = list(by.values())
    return out


def calendar_days(days):
    by = defaultdict(list)
    for d in days:
        by[season_of(d)].append(d)
    return sum((date.fromisoformat(max(v)) - date.fromisoformat(min(v))).days + 1 for v in by.values())


def leg_odds_table(days):
    """pazar → [[olasılık %, ortanca oran, adet], ...] (canlıda oran yoksa tahmin için)."""
    b = defaultdict(lambda: defaultdict(list))
    for legs in days.values():
        for mk, p, o, *_ in legs:
            if p >= 0.70:
                b[mk][min(99, int(p * 100))].append(o)
    return {mk: [[k, round(sorted(v)[len(v) // 2], 3), len(v)] for k, v in sorted(t.items()) if len(v) >= 10]
            for mk, t in b.items()}


# ---------------------------------------------------------------- canlı
def missing_key(x):
    info = x.get('lineup') or x.get('injury') or {}
    return bool(info.get('home_missing_key') or info.get('away_missing_key'))


def limited(x):
    if x.get('h2h_tier') or (x.get('h2h_matches_used') or 0) >= 2:
        return False
    b = x.get('basis') or ''
    return b.startswith('partial-form') or b.startswith('league-avg')


def table_odds(tables, mk, p):
    t = tables.get(mk) or []
    if not t:
        return 1 / p
    k = int(p * 100)
    for row in t:
        if row[0] == k:
            return row[1]
    return t[-1][1] if k > t[-1][0] else t[0][1]


def live_candidates(x, tables):
    if limited(x):
        return []
    mk = x.get('market') or {}
    c = []
    o25, u25 = mk.get('o25_odds'), mk.get('u25_odds')
    lam_mkt = None
    if o25 and u25 and o25 > 1 and u25 > 1:
        book = 1 / o25 + 1 / u25
        lam_mkt = (lam_from_p25((1 / o25) / book), book)
    for m, key in GOAL.items():
        p = x.get(key)
        if p is None or p < HL_MIN[m]:
            continue
        if m == '2.5+' and o25 and o25 > 1:
            odds, real = o25, True
        elif m != '2.5+' and lam_mkt:
            lam, book = lam_mkt
            odds, real = max(1.01, 1 / min(0.999, poisson_over(lam, float(m[0]) + .5) * book)), False
        else:
            odds, real = table_odds(tables, m, p), False
        c.append(dict(market=m, p=round(p, 4), odds=round(odds, 3), real=real))
    if not missing_key(x):
        lh, la = x.get('lam_home') or x.get('base_lam_home'), x.get('lam_away') or x.get('base_lam_away')
        if lh and la:
            dc = dc_probs(float(lh), float(la), float(x.get('rho') or x.get('base_rho') or 0.02))
            h, d, a = (mk.get(k) for k in ('h', 'd', 'a'))
            for m in DC:
                if dc[m] < HL_MIN[m]:
                    continue
                if h and d and a and min(h, d, a) > 1:
                    inv = {'1X': 1 / h + 1 / d, '12': 1 / h + 1 / a, 'X2': 1 / d + 1 / a}[m]
                    odds, real = 1 / inv, True
                else:
                    odds, real = table_odds(tables, m, dc[m]), False
                c.append(dict(market=m, p=dc[m], odds=round(odds, 3), real=real))
    for cc in c:
        cc.update(league=x['league'], home=x['home'], away=x['away'], kickoff_utc=x['kickoff_utc'])
    return c


def result_index():
    p = ROOT / 'data' / 'results.json'
    if not p.exists():
        return {}
    ms = json.loads(p.read_text(encoding='utf-8')).get('matches', [])
    idx = {}
    for m in ms:
        s = m.get('score')
        if not s or '-' not in str(s):
            continue
        try:
            hg, ag = (int(v) for v in str(s).split('-')[:2])
        except ValueError:
            continue
        idx[(m['league'], m['home'], m['away'], ist_day(m['kickoff_utc']))] = (hg, ag)
    return idx


def update_coupons(tables, now):
    path = ROOT / 'data' / 'coupons.json'
    old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    coupons = old.get('coupons', {})
    preds = json.loads((ROOT / 'predictions.json').read_text(encoding='utf-8'))
    by_day = defaultdict(list)
    for x in preds:
        ko = datetime.fromisoformat(x['kickoff_utc'].replace('Z', '+00:00'))
        if ko <= now or x.get('live'):
            continue
        cands = live_candidates(x, tables)
        if cands:
            by_day[ist_day(x['kickoff_utc'])].append(cands)
    # Donmamış (henüz başlamamış) öneriler canlı fikstürden yeniden kurulur
    for k in [k for k, c in coupons.items() if not c.get('frozen')]:
        del coupons[k]
    for d, matches in by_day.items():
        for prof, cfg in PROFILES.items():
            key = f'{d}|{prof}'
            if key in coupons:
                continue
            legs = pick_coupon(matches, cfg['R'])
            if legs:
                coupons[key] = dict(date=d, profile=prof, R=cfg['R'], legs=legs, odds=round(math.prod(l['odds'] for l in legs), 3),
                                    p=round(math.prod(l['p'] for l in legs), 4), status='pending', frozen=False)
    idx = result_index()
    for c in coupons.values():
        first = min(datetime.fromisoformat(l['kickoff_utc'].replace('Z', '+00:00')) for l in c['legs'])
        if first <= now:
            c['frozen'] = True
        res = []
        for l in c['legs']:
            sc = idx.get((l['league'], l['home'], l['away'], ist_day(l['kickoff_utc'])))
            if sc:
                l['score'] = f'{sc[0]}-{sc[1]}'
                l['result'] = 'won' if hit(l['market'], *sc) else 'lost'
            res.append(l.get('result'))
        c['status'] = 'lost' if 'lost' in res else ('won' if all(r == 'won' for r in res) else 'pending')
    cutoff = (now + IST - timedelta(days=KEEP_DAYS)).date().isoformat()
    coupons = {k: v for k, v in sorted(coupons.items()) if v['date'] >= cutoff}
    out = {'generated_at': now.isoformat(timespec='seconds'), 'max_legs': MAX_LEGS, 'hl_min': HL_MIN,
           'profiles': PROFILES, 'coupons': coupons}
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')
    st = defaultdict(lambda: defaultdict(int))
    for c in coupons.values():
        st[c['profile']][c['status']] += 1
    print('coupons.json:', {k: dict(v) for k, v in st.items()})


def main():
    now = datetime.now(timezone.utc)
    d = json.loads((ROOT / 'data' / 'stats-5season.json').read_text(encoding='utf-8'))
    days, _ = legs_by_day(d['rows'], load_odds())
    cal = calendar_days(days)
    tiers = []
    for t in TIERS:
        s = summarize(build_coupons(days, t['markets'], t['thr'], t['legs']), STAKE_PCT)
        hist = None if not s else {
            'coupons': s['n'], 'win_pct': round(s['win'] * 100, 1), 'model_pct': round(s['model_p'] * 100, 1),
            'avg_odds': round(s['avg_odds'], 2), 'fair_odds': round(1 / s['win'], 2) if s['win'] else None,
            'roi_pct': round(s['roi'] * 100, 1), 'max_drawdown_pct': round(s['dd_max'] * 100),
        }
        tiers.append({**t, 'history': hist})
    cands = hist_candidates(days)
    target = []
    for R in TARGET_GRID:
        cs = []
        for dd in sorted(cands):
            legs = pick_coupon(cands[dd], R)
            if legs:
                cs.append(dict(won=all(l['won'] for l in legs), odds=math.prod(l['odds'] for l in legs), legs=len(legs),
                               est=any(l['kind'] == 'tahmini' for l in legs)))
        if len(cs) < 30:
            continue
        n = len(cs)
        target.append({
            'R': R, 'coupons': n, 'per_day': round(n / cal, 4),
            'win_pct': round(sum(c['won'] for c in cs) / n * 100, 1),
            'avg_odds': round(sum(c['odds'] for c in cs) / n, 3),
            'avg_legs': round(sum(c['legs'] for c in cs) / n, 2),
            'est_pct': round(sum(c['est'] for c in cs) / n * 100),
            'outcomes': [int(round(c['odds'] * 1000)) * (1 if c['won'] else -1) for c in cs],
        })
        t = target[-1]
        print(f"R={R}: {n} kupon, tuttu %{t['win_pct']}, oran {t['avg_odds']}, {t['avg_legs']} maç, tahmini oranlı %{t['est_pct']}")
    tables = leg_odds_table(days)
    res = {'generated_at': now.isoformat(timespec='seconds'), 'seasons': d.get('seasons'),
           'odds_source': 'football-data.co.uk piyasa ortalaması (2.5+ ve Çifte Şans gerçek; 0.5+/1.5+ 2.5 fiyatından tahmini)',
           'calendar_days': cal, 'tiers': tiers,
           'target': {'max_legs': MAX_LEGS, 'hl_min': HL_MIN, 'rules': target}}
    (ROOT / 'data' / 'coupon-rules.json').write_text(json.dumps(res, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')
    print('Wrote data/coupon-rules.json')
    update_coupons(tables, now)


if __name__ == '__main__':
    main()
