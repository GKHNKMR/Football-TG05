import json, math, sys, os
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

def poisson_pmf(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return (lam**k * math.exp(-lam)) / math.factorial(k)

def tau(x, y, lh, la, rho):
    if x == 0 and y == 0: return 1.0 - lh * la * rho
    if x == 0 and y == 1: return 1.0 + lh * rho
    if x == 1 and y == 0: return 1.0 + la * rho
    if x == 1 and y == 1: return 1.0 - rho
    return 1.0

def safe_rho(lh, la, rho):
    hi = min(1.0 / (lh * la), 1.0) - 1e-6
    lo = max(-1.0 / lh, -1.0 / la) + 1e-6
    return min(max(rho, lo), hi)

def match_season(m):
    d = m.get('kickoff_utc', '')[:10]
    if not d: return '2025/26'
    if d >= '2026-07-01': return '2026/27'
    if d >= '2025-07-01': return '2025/26'
    if d >= '2024-07-01': return '2024/25'
    if d >= '2023-07-01': return '2023/24'
    if d >= '2022-07-01': return '2022/23'
    return '2021/22'

with open('data/results.json', 'r', encoding='utf-8') as f:
    res = json.load(f)

matches = res.get('matches', [])
print(f"Loaded {len(matches)} matches from results.json")

def empty_stat():
    return {
        'total': 0,
        'dc_best_n': 0, 'dc_best_h': 0,
        'dc_1x_n': 0, 'dc_1x_h': 0,
        'dc_12_n': 0, 'dc_12_h': 0,
        'dc_x2_n': 0, 'dc_x2_h': 0,
        'sc_top1_h': 0, 'sc_top3_h': 0
    }

overall = empty_stat()
by_league = {}
by_season = {}
by_season_league = {}
samples = []

for idx, m in enumerate(matches):
    sc = m.get('score')
    lg = m.get('league') or 'Diğer'
    szn = match_season(m)
    if not sc or '-' not in sc: continue
    try:
        parts = sc.split('-')
        act_h = int(parts[0].strip())
        act_a = int(parts[1].strip())
    except: continue

    tot_lam = float(m.get('pred_lambda') or 2.65)
    mkt = m.get('market') or {}
    h_odds = float(mkt.get('h') or 0)
    a_odds = float(mkt.get('a') or 0)

    if h_odds > 1.0 and a_odds > 1.0:
        inv_h = 1.0 / h_odds
        inv_a = 1.0 / a_odds
        home_share = inv_h / (inv_h + inv_a)
        lh = tot_lam * max(0.25, min(0.75, home_share))
        la = tot_lam - lh
    else:
        lh = tot_lam * 0.55
        la = tot_lam * 0.45

    lh = max(0.3, lh)
    la = max(0.3, la)
    rho = safe_rho(lh, la, 0.02)

    grid = {}
    tot_p = 0.0
    for x in range(10):
        for y in range(10):
            p = poisson_pmf(x, lh) * poisson_pmf(y, la) * max(0, tau(x, y, lh, la, rho))
            grid[(x, y)] = p
            tot_p += p
    for k in grid: grid[k] /= tot_p

    p_home = sum(grid[(x, y)] for x in range(10) for y in range(10) if x > y)
    p_draw = sum(grid[(x, y)] for x in range(10) for y in range(10) if x == y)
    p_away = sum(grid[(x, y)] for x in range(10) for y in range(10) if x < y)

    p_1x = p_home + p_draw
    p_12 = p_home + p_away
    p_x2 = p_draw + p_away

    act_1x = (act_h >= act_a)
    act_12 = (act_h != act_a)
    act_x2 = (act_a >= act_h)

    best_p = max(p_1x, p_12, p_x2)
    if best_p == p_1x:
        best_pick = '1X'
        best_hit = act_1x
    elif best_p == p_12:
        best_pick = '12'
        best_hit = act_12
    else:
        best_pick = 'X2'
        best_hit = act_x2

    scores_sorted = sorted(grid.items(), key=lambda item: item[1], reverse=True)
    top1 = scores_sorted[0][0]
    top3 = [s[0] for s in scores_sorted[:3]]
    top1_hit = ((act_h, act_a) == top1)
    top3_hit = ((act_h, act_a) in top3)

    # Güncelleme fonksiyonu
    def record(target):
        target['total'] += 1
        if best_p >= 0.75:
            target['dc_best_n'] += 1
            if best_hit: target['dc_best_h'] += 1
        if p_1x >= 0.75:
            target['dc_1x_n'] += 1
            if act_1x: target['dc_1x_h'] += 1
        if p_12 >= 0.75:
            target['dc_12_n'] += 1
            if act_12: target['dc_12_h'] += 1
        if p_x2 >= 0.75:
            target['dc_x2_n'] += 1
            if act_x2: target['dc_x2_h'] += 1
        if top1_hit: target['sc_top1_h'] += 1
        if top3_hit: target['sc_top3_h'] += 1

    record(overall)
    if lg not in by_league: by_league[lg] = empty_stat()
    record(by_league[lg])
    if szn not in by_season: by_season[szn] = empty_stat()
    record(by_season[szn])

    if szn not in by_season_league: by_season_league[szn] = {}
    if lg not in by_season_league[szn]: by_season_league[szn][lg] = empty_stat()
    record(by_season_league[szn][lg])

    # Örnek maçlar (düzenli aralıklarla ~250 maç)
    if idx % 65 == 0 and len(samples) < 260:
        samples.append({
            'date': m.get('kickoff_utc', '')[:10],
            'league': lg,
            'home': m.get('home'),
            'away': m.get('away'),
            'pred_lambda': round(tot_lam, 2),
            'actual_score': sc,
            'best_dc': best_pick,
            'best_dc_pct': round(best_p * 100, 1),
            'dc_hit': bool(best_hit),
            'top_scores': [f"{s[0][0]}-{s[0][1]} (%{s[1]*100:.1f})" for s in scores_sorted[:3]],
            'top1_hit': bool(top1_hit),
            'top3_hit': bool(top3_hit)
        })

def finalize_stats(st):
    res = dict(st)
    res['dc_best_pct'] = round((st['dc_best_h'] / st['dc_best_n'] * 100), 1) if st['dc_best_n'] else 0
    res['dc_1x_pct'] = round((st['dc_1x_h'] / st['dc_1x_n'] * 100), 1) if st['dc_1x_n'] else 0
    res['dc_12_pct'] = round((st['dc_12_h'] / st['dc_12_n'] * 100), 1) if st['dc_12_n'] else 0
    res['dc_x2_pct'] = round((st['dc_x2_h'] / st['dc_x2_n'] * 100), 1) if st['dc_x2_n'] else 0
    res['sc_top1_pct'] = round((st['sc_top1_h'] / st['total'] * 100), 1) if st['total'] else 0
    res['sc_top3_pct'] = round((st['sc_top3_h'] / st['total'] * 100), 1) if st['total'] else 0
    return res

final_data = {
    'overall': finalize_stats(overall),
    'by_league': {k: finalize_stats(v) for k, v in by_league.items()},
    'by_season': {k: finalize_stats(v) for k, v in by_season.items()},
    'by_season_league': {sz: {lg: finalize_stats(v) for lg, v in lgdict.items()} for sz, lgdict in by_season_league.items()},
    'samples': samples
}

# Write JS bundle
js_content = f"window.CIFTE_BACKTEST_DATA = {json.dumps(final_data, ensure_ascii=False, indent=2)};\n"
with open('js/cifte_backtest_data.js', 'w', encoding='utf-8') as f:
    f.write(js_content)

print(f"Successfully generated js/cifte_backtest_data.js with {len(samples)} sample matches.")
print("Overall stats:")
print(f"  Total matches: {final_data['overall']['total']}")
print(f"  Best DC (>=75%): {final_data['overall']['dc_best_h']} / {final_data['overall']['dc_best_n']} (%{final_data['overall']['dc_best_pct']})")
print(f"  1X: {final_data['overall']['dc_1x_h']} / {final_data['overall']['dc_1x_n']} (%{final_data['overall']['dc_1x_pct']})")
print(f"  12: {final_data['overall']['dc_12_h']} / {final_data['overall']['dc_12_n']} (%{final_data['overall']['dc_12_pct']})")
print(f"  X2: {final_data['overall']['dc_x2_h']} / {final_data['overall']['dc_x2_n']} (%{final_data['overall']['dc_x2_pct']})")
print(f"  Score Top-1: {final_data['overall']['sc_top1_h']} / {final_data['overall']['total']} (%{final_data['overall']['sc_top1_pct']})")
print(f"  Score Top-3: {final_data['overall']['sc_top3_h']} / {final_data['overall']['total']} (%{final_data['overall']['sc_top3_pct']})")
