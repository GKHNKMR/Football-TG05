import csv, math, json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

from backtest import DIVISIONS, ALL_SEASONS, TARGET_SEASONS, PRIOR_WEIGHTS, load_division, season_label
from goals_model import LeagueModel
from xg_blend import XG_WEIGHT_BY_LEAGUE, xg_seasons_for

divisions = {d: load_division(d) for d in DIVISIONS}
all_records = []
for div, league in DIVISIONS.items():
    rows = divisions[div]
    by_code = {}
    for m in rows:
        by_code.setdefault(m['season'], []).append(m)
    for target in TARGET_SEASONS:
        ti = ALL_SEASONS.index(target)
        priors = ALL_SEASONS[max(0, ti - 4):ti][::-1]
        if len(priors) < 2: continue
        xg_weight = XG_WEIGHT_BY_LEAGUE.get(league, 0.0)
        xg_seasons = xg_seasons_for(div, list(zip(priors, PRIOR_WEIGHTS))) if xg_weight else None
        model = LeagueModel([(by_code.get(p, []), w) for p, w in zip(priors, PRIOR_WEIGHTS)], xg_seasons=xg_seasons, xg_weight=xg_weight)
        for m in by_code.get(target, []):
            pred = model.predict(m['home'], m['away'], market_p25=m.get('mk_p25'))
            all_records.append({
                'league': league, 'season': target, 'date': m['date'],
                'home': m['home'], 'away': m['away'],
                'lam': pred['exp_goals'], 'total': m['total'],
                'score': f"{m['hg']}-{m['ag']}",
                'p05': pred['p_over_0_5'], 'p15': pred['p_over_1_5'], 'p25': pred['p_over_2_5'],
            })

print('Total records:', len(all_records))
# 0.5 general & vurgu
lean = lambda p: round(p * 100) >= 50
hit05 = sum(lean(r['p05']) == (r['total'] > 0.5) for r in all_records)
vurgu05 = [r for r in all_records if r['p05'] >= 0.95]
vurgu05_hit = sum((r['total'] > 0.5) for r in vurgu05)

# 1.5 general & vurgu
hit15 = sum(lean(r['p15']) == (r['total'] > 1.5) for r in all_records)
vurgu15 = [r for r in all_records if r['p15'] >= 0.85]
vurgu15_hit = sum((r['total'] > 1.5) for r in vurgu15)

# 2.5 general & vurgu
hit25 = sum(lean(r['p25']) == (r['total'] > 2.5) for r in all_records)
vurgu25 = [r for r in all_records if r['p25'] >= 0.75]
vurgu25_hit = sum((r['total'] > 2.5) for r in vurgu25)

print(f"0.5 Genel Yön: {hit05} / {len(all_records)} (%{hit05/len(all_records)*100:.1f})")
print(f"0.5 Vurgu (>=95%): {vurgu05_hit} / {len(vurgu05)} (%{vurgu05_hit/len(vurgu05)*100:.1f})")
print(f"1.5 Genel Yön: {hit15} / {len(all_records)} (%{hit15/len(all_records)*100:.1f})")
print(f"1.5 Vurgu (>=85%): {vurgu15_hit} / {len(vurgu15)} (%{vurgu15_hit/len(vurgu15)*100:.1f})")
print(f"2.5 Genel Yön: {hit25} / {len(all_records)} (%{hit25/len(all_records)*100:.1f})")
print(f"2.5 Vurgu (>=75%): {vurgu25_hit} / {len(vurgu25)} (%{vurgu25_hit/len(vurgu25)*100:.1f})")
