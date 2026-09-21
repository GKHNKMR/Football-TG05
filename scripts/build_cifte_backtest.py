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
by_league = {league: empty_stat() for league in DIVISIONS.values()}
by_season = {season_label(sz): empty_stat() for sz in TARGET_SEASONS}
by_season_league = {season_label(sz): {league: empty_stat() for league in DIVISIONS.values()} for sz in TARGET_SEASONS}
all_matches = []

divisions = {d: load_division(d) for d in DIVISIONS}

for div, league in DIVISIONS.items():
    rows = divisions[div]
    by_code = {}
    for m in rows:
        by_code.setdefault(m["season"], []).append(m)

    for target in TARGET_SEASONS:
        ti = ALL_SEASONS.index(target)
        priors = ALL_SEASONS[max(0, ti - 4):ti][::-1]
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

            scores_sorted = sorted(grid.items(), key=lambda item: item[1], reverse=True)
            top1 = scores_sorted[0][0]
            top3 = [s[0] for s in scores_sorted[:3]]
            top1_hit = ((act_h, act_a) == top1)
            top3_hit = ((act_h, act_a) in top3)

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
                if top1_hit: target_stat['sc_top1_h'] += 1
                if top3_hit: target_stat['sc_top3_h'] += 1

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
                'best_dc': best_pick,
                'best_dc_pct': round(best_p * 100, 1),
                'dc_hit': bool(best_hit),
                'is_limited': bool(is_limited),
                'p_1x': round(p_1x * 100, 1),
                'p_12': round(p_12 * 100, 1),
                'p_x2': round(p_x2 * 100, 1),
                'top_scores': [f"{s[0][0]}-{s[0][1]} (%{s[1]*100:.1f})" for s in scores_sorted[:3]],
                'top1_hit': bool(top1_hit),
                'top3_hit': bool(top3_hit)
            })

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
    res['sc_top1_pct'] = round((st['sc_top1_h'] / st['total'] * 100), 1) if st['total'] else 0.0
    res['sc_top3_pct'] = round((st['sc_top3_h'] / st['total'] * 100), 1) if st['total'] else 0.0
    return res

final_data = {
    'overall': finalize_stats(overall),
    'by_league': {k: finalize_stats(v) for k, v in by_league.items()},
    'by_season': {k: finalize_stats(v) for k, v in by_season.items()},
    'by_season_league': {sz: {lg: finalize_stats(v) for lg, v in lgdict.items()} for sz, lgdict in by_season_league.items()},
    'samples': samples
}

js_content = f"window.CIFTE_BACKTEST_DATA = {json.dumps(final_data, ensure_ascii=False, indent=2)};\n"
with open('js/cifte_backtest_data.js', 'w', encoding='utf-8') as f:
    f.write(js_content)

print(f"Successfully generated js/cifte_backtest_data.js with {len(samples)} sample matches.")
print("Overall stats (16.478 matches - 5 completed seasons, excluding limited data from vurgu):")
print(f"  Total matches: {final_data['overall']['total']}")
print(f"  1X (>=75%): {final_data['overall']['dc_1x_h']} / {final_data['overall']['dc_1x_n']} (%{final_data['overall']['dc_1x_pct']})")
print(f"  12 (>=75%): {final_data['overall']['dc_12_h']} / {final_data['overall']['dc_12_n']} (%{final_data['overall']['dc_12_pct']})")
print(f"  X2 (>=75%): {final_data['overall']['dc_x2_h']} / {final_data['overall']['dc_x2_n']} (%{final_data['overall']['dc_x2_pct']})")
print(f"  Score Top-1: {final_data['overall']['sc_top1_h']} / {final_data['overall']['total']} (%{final_data['overall']['sc_top1_pct']})")
print(f"  Score Top-3: {final_data['overall']['sc_top3_h']} / {final_data['overall']['total']} (%{final_data['overall']['sc_top3_pct']})")
