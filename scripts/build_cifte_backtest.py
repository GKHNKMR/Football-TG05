import json, math, sys, os
from pathlib import Path

# Add scripts directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding='utf-8')

from backtest import load_division, DIVISIONS, TARGET_SEASONS, ALL_SEASONS, PRIOR_WEIGHTS, season_label
from goals_model import LeagueModel
from xg_blend import XG_WEIGHT_BY_LEAGUE, xg_seasons_for
from teams import to_pretty

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

GOAL_RANGES = {
    '2-3': {'label': '2–3 Gol', 'hit': lambda total: 2 <= total <= 3},
    '3-4': {'label': '3–4 Gol', 'hit': lambda total: 3 <= total <= 4},
    '5+': {'label': '5+ Gol', 'hit': lambda total: total >= 5},
}

def goal_range_probabilities(grid):
    """Dixon-Coles skor matrisini istenen toplam gol bantlarına toplar."""
    totals = {n: 0.0 for n in range(19)}
    for (home_goals, away_goals), probability in grid.items():
        totals[home_goals + away_goals] += probability
    return {
        '2-3': totals[2] + totals[3],
        '3-4': totals[3] + totals[4],
        '5+': sum(totals[n] for n in range(5, 19)),
    }

def permille(p):
    # Aşağı yuvarla: %74,96 binde 750'ye yuvarlanıp ≥%75 vurgu eşiğini yanlışlıkla geçmesin
    return int(float(p) * 1000 + 1e-9)

def empty_stat():
    return {
        'total': 0,
        'dc_best_n': 0, 'dc_best_h': 0,
        'dc_1x_n': 0, 'dc_1x_h': 0,
        'dc_12_n': 0, 'dc_12_h': 0,
        'dc_x2_n': 0, 'dc_x2_h': 0,
        'gr_n': 0, 'gr_h': 0,
        'gr_23_n': 0, 'gr_23_h': 0,
        'gr_34_n': 0, 'gr_34_h': 0,
        'gr_5p_n': 0, 'gr_5p_h': 0
    }

overall = empty_stat()
by_league = {league: empty_stat() for league in DIVISIONS.values()}
by_season = {season_label(sz): empty_stat() for sz in TARGET_SEASONS}
by_season_league = {season_label(sz): {league: empty_stat() for league in DIVISIONS.values()} for sz in TARGET_SEASONS}
all_matches = []
# İstatistikler sekmesi: 5 sezonun + güncel sezonun oynanmış TÜM maçları (kısıtlı olanlar dahil, bayraklı)
stats_rows = []

# Güncel sezon yalnızca İstatistikler listesine / ana sayfa şeridine girer; Çifte Şans
# backtest'i (by_season, samples) 5 tamamlanmış sezonda sabit kalır. Güncel sezon da
# walk-forward: model yalnızca önceki sezonlarla kurulur, o sezonun sonuçlarını görmez.
CURRENT_SEASON = "2627"
SEASON_CHAIN = ALL_SEASONS + [CURRENT_SEASON]
STATS_SEASONS = TARGET_SEASONS + [CURRENT_SEASON]

divisions = {d: load_division(d, SEASON_CHAIN) for d in DIVISIONS}

for div, league in DIVISIONS.items():
    rows = divisions[div]
    by_code = {}
    for m in rows:
        by_code.setdefault(m["season"], []).append(m)

    for target in STATS_SEASONS:
        in_backtest = target in TARGET_SEASONS
        ti = SEASON_CHAIN.index(target)
        priors = SEASON_CHAIN[max(0, ti - 4):ti][::-1]
        if len(priors) < 2:
            continue

        xg_weight = XG_WEIGHT_BY_LEAGUE.get(league, 0.0)
        xg_seasons = (xg_seasons_for(div, list(zip(priors, PRIOR_WEIGHTS))) if xg_weight else None)
        model = LeagueModel(
            [(by_code.get(p, []), w) for p, w in zip(priors, PRIOR_WEIGHTS)],
            xg_seasons=xg_seasons,
            xg_weight=xg_weight
        )

        szn_name = season_label(target)

        for m in by_code.get(target, []):
            pred = model.predict(m["home"], m["away"])
            basis = pred.get("basis", "")
            h2h_used = pred.get("h2h_matches_used", 0)

            # Kısıtlı veri kontrolü: partial-form veya league-avg olup H2H < 2 ise kısıtlıdır
            is_limited = (basis.startswith("partial-form") or basis.startswith("league-avg")) and (h2h_used < 2)

            lh = max(0.2, float(pred["lam_home"]))
            la = max(0.2, float(pred["lam_away"]))
            rho = safe_rho(lh, la, float(pred.get("rho", 0.02)))

            grid = {}
            tot_p = 0.0
            for x in range(10):
                for y in range(10):
                    p = poisson_pmf(x, lh) * poisson_pmf(y, la) * max(0, tau(x, y, lh, la, rho))
                    grid[(x, y)] = p
                    tot_p += p
            for k in grid:
                grid[k] /= tot_p

            p_home = sum(grid[(x, y)] for x in range(10) for y in range(10) if x > y)
            p_draw = sum(grid[(x, y)] for x in range(10) for y in range(10) if x == y)
            p_away = sum(grid[(x, y)] for x in range(10) for y in range(10) if x < y)

            p_1x = p_home + p_draw
            p_12 = p_home + p_away
            p_x2 = p_draw + p_away

            act_h = int(m["hg"])
            act_a = int(m["ag"])
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

            range_probs = goal_range_probabilities(grid)
            best_range = max(range_probs, key=range_probs.get)
            actual_total = act_h + act_a
            range_hit = GOAL_RANGES[best_range]['hit'](actual_total)
            range_stat_key = {'2-3': 'gr_23', '3-4': 'gr_34', '5+': 'gr_5p'}[best_range]

            def record(target_stat):
                target_stat['total'] += 1
                # KISITLI VERİLERİ VURGULU OLARAK DAHİL ETME:
                if not is_limited:
                    if best_p >= 0.75:
                        target_stat['dc_best_n'] += 1
                        if best_hit: target_stat['dc_best_h'] += 1
                    if p_1x >= 0.75:
                        target_stat['dc_1x_n'] += 1
                        if act_1x: target_stat['dc_1x_h'] += 1
                    if p_12 >= 0.75:
                        target_stat['dc_12_n'] += 1
                        if act_12: target_stat['dc_12_h'] += 1
                    if p_x2 >= 0.75:
                        target_stat['dc_x2_n'] += 1
                        if act_x2: target_stat['dc_x2_h'] += 1
                    target_stat['gr_n'] += 1
                    target_stat[f'{range_stat_key}_n'] += 1
                    if range_hit:
                        target_stat['gr_h'] += 1
                        target_stat[f'{range_stat_key}_h'] += 1

            stats_rows.append([
                m.get('date', ''), league, to_pretty(league, m['home']), to_pretty(league, m['away']),
                act_h, act_a,
                permille(pred['p_over_0_5']), permille(pred['p_over_1_5']), permille(pred['p_over_2_5']),
                permille(p_1x), permille(p_12), permille(p_x2),
                1 if is_limited else 0,
                round(float(pred['lam_home']) + float(pred['lam_away']), 2),
            ])

            if not in_backtest:
                continue

            record(overall)
            record(by_league[league])
            record(by_season[szn_name])
            record(by_season_league[szn_name][league])

            all_matches.append({
                'date': m.get('date', ''),
                'season': szn_name,
                'league': league,
                'home': to_pretty(league, m['home']),
                'away': to_pretty(league, m['away']),
                'pred_lambda': round(lh + la, 2),
                'actual_score': f"{act_h}-{act_a}",
                'actual_total': actual_total,
                'best_dc': best_pick,
                'best_dc_pct': round(best_p * 100, 1),
                'dc_hit': bool(best_hit),
                'is_limited': bool(is_limited),
                'p_1x': round(p_1x * 100, 1),
                'p_12': round(p_12 * 100, 1),
                'p_x2': round(p_x2 * 100, 1),
                'goal_range': best_range,
                'goal_range_label': GOAL_RANGES[best_range]['label'],
                'goal_range_pct': round(range_probs[best_range] * 100, 1),
                'goal_range_probs': {key: round(value * 100, 1) for key, value in range_probs.items()},
                'range_hit': bool(range_hit)
            })

all_matches = [match for match in all_matches if not match['is_limited']]
all_matches.sort(key=lambda x: x['date'], reverse=True)
# ~250 dengeli örnek maç
step = max(1, len(all_matches) // 250)
samples = all_matches[::step][:250]

def finalize_stats(st):
    res = dict(st)
    res['dc_best_pct'] = round((st['dc_best_h'] / st['dc_best_n'] * 100), 1) if st['dc_best_n'] else 0.0
    res['dc_1x_pct'] = round((st['dc_1x_h'] / st['dc_1x_n'] * 100), 1) if st['dc_1x_n'] else 0.0
    res['dc_12_pct'] = round((st['dc_12_h'] / st['dc_12_n'] * 100), 1) if st['dc_12_n'] else 0.0
    res['dc_x2_pct'] = round((st['dc_x2_h'] / st['dc_x2_n'] * 100), 1) if st['dc_x2_n'] else 0.0
    res['gr_pct'] = round((st['gr_h'] / st['gr_n'] * 100), 1) if st['gr_n'] else 0.0
    for key in ('gr_23', 'gr_34', 'gr_5p'):
        res[f'{key}_pct'] = round((st[f'{key}_h'] / st[f'{key}_n'] * 100), 1) if st[f'{key}_n'] else 0.0
    return res

final_data = {
    'overall': finalize_stats(overall),
    'by_league': {k: finalize_stats(v) for k, v in by_league.items()},
    'by_season': {k: finalize_stats(v) for k, v in by_season.items()},
    'by_season_league': {sz: {lg: finalize_stats(v) for lg, v in lgdict.items()} for sz, lgdict in by_season_league.items()},
    'samples': samples
}

# İstatistikler sekmesi için kompakt maç listesi (eskiden yeniye).
# Olasılıklar binde birlik tamsayı: p05, p15, p25, p1x, p12, px2.
stats_rows.sort(key=lambda r: (r[0], r[1], r[2]))
stats_seasons = [season_label(sz) for sz in STATS_SEASONS]
generated_at = __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
# Workflow saatlik çalışır: yeni maç yoksa eski zaman damgasını koru ki dosyalar değişmesin, gereksiz commit olmasın
try:
    with open('data/stats-5season.json', encoding='utf-8') as f:
        prev = json.load(f)
    if prev.get('rows') == stats_rows and prev.get('seasons') == stats_seasons:
        generated_at = prev.get('generated_at', generated_at)
except (OSError, ValueError):
    pass
stats_payload = {
    'generated_at': generated_at,
    'seasons': stats_seasons,
    'fields': ['date', 'league', 'home', 'away', 'hg', 'ag', 'p05', 'p15', 'p25', 'p1x', 'p12', 'px2', 'limited', 'lambda'],
    'thresholds': {'p05': 950, 'p15': 850, 'p25': 800, 'dc': 800},
    'rows': stats_rows,
}
with open('data/stats-5season.json', 'w', encoding='utf-8') as f:
    json.dump(stats_payload, f, ensure_ascii=False, separators=(',', ':'))
print(f"Wrote data/stats-5season.json with {len(stats_rows)} matches.")

# Ana sayfa (Bülten) şeridi: yalnızca vurgulanan tahminlerin başarısı.
# Kısıtlı veri hariç; her pazar kendi eşiğiyle (0.5≥%95, 1.5≥%85, 2.5≥%80, 1X/12/X2≥%80).
HL_MARKETS = [
    ('0.5+', 6, 950, lambda hg, ag: hg + ag >= 1),
    ('1.5+', 7, 850, lambda hg, ag: hg + ag >= 2),
    ('2.5+', 8, 800, lambda hg, ag: hg + ag >= 3),
    ('1X', 9, 800, lambda hg, ag: hg >= ag),
    ('12', 10, 800, lambda hg, ag: hg != ag),
    ('X2', 11, 800, lambda hg, ag: ag >= hg),
]
hl_markets = {name: {'n': 0, 'h': 0} for name, *_ in HL_MARKETS}
hl_match_n = hl_match_h = 0
for r in stats_rows:
    if r[12]:
        continue
    outcomes = []
    for name, idx, thr, hit in HL_MARKETS:
        if r[idx] >= thr:
            ok = hit(r[4], r[5])
            outcomes.append(ok)
            hl_markets[name]['n'] += 1
            hl_markets[name]['h'] += int(ok)
    if outcomes:
        hl_match_n += 1
        hl_match_h += int(all(outcomes))
pick_n = sum(v['n'] for v in hl_markets.values())
pick_h = sum(v['h'] for v in hl_markets.values())
summary = {
    'generated_at': stats_payload['generated_at'],
    'seasons': stats_payload['seasons'],
    'total_matches': len(stats_rows),
    'picks': {'n': pick_n, 'h': pick_h, 'pct': round(pick_h / pick_n * 100, 1) if pick_n else 0.0},
    'matches': {'n': hl_match_n, 'h': hl_match_h, 'pct': round(hl_match_h / hl_match_n * 100, 1) if hl_match_n else 0.0},
    'markets': {k: dict(v, pct=round(v['h'] / v['n'] * 100, 1) if v['n'] else 0.0) for k, v in hl_markets.items()},
}
with open('data/stats-summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print(f"Wrote data/stats-summary.json: highlighted picks {pick_h}/{pick_n}, matches {hl_match_h}/{hl_match_n}")

js_content = f"window.CIFTE_BACKTEST_DATA = {json.dumps(final_data, ensure_ascii=False, indent=2)};\n"
with open('js/cifte_backtest_data.js', 'w', encoding='utf-8') as f:
    f.write(js_content)

print(f"Successfully generated js/cifte_backtest_data.js with {len(samples)} sample matches.")
print("Overall stats (16.478 matches - 5 completed seasons, excluding limited data from vurgu):")
print(f"  Total matches: {final_data['overall']['total']}")
print(f"  1X (>=75%): {final_data['overall']['dc_1x_h']} / {final_data['overall']['dc_1x_n']} (%{final_data['overall']['dc_1x_pct']})")
print(f"  12 (>=75%): {final_data['overall']['dc_12_h']} / {final_data['overall']['dc_12_n']} (%{final_data['overall']['dc_12_pct']})")
print(f"  X2 (>=75%): {final_data['overall']['dc_x2_h']} / {final_data['overall']['dc_x2_n']} (%{final_data['overall']['dc_x2_pct']})")
print(f"  Goal range: {final_data['overall']['gr_h']} / {final_data['overall']['gr_n']} (%{final_data['overall']['gr_pct']})")
print(f"  2-3 goals: {final_data['overall']['gr_23_h']} / {final_data['overall']['gr_23_n']} (%{final_data['overall']['gr_23_pct']})")
print(f"  3-4 goals: {final_data['overall']['gr_34_h']} / {final_data['overall']['gr_34_n']} (%{final_data['overall']['gr_34_pct']})")
print(f"  5+ goals: {final_data['overall']['gr_5p_h']} / {final_data['overall']['gr_5p_n']} (%{final_data['overall']['gr_5p_pct']})")
