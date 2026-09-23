import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

from backtest import DIVISIONS, ALL_SEASONS, PRIOR_WEIGHTS, load_division
from goals_model import LeagueModel
from teams import to_pretty
from xg_blend import XG_WEIGHT_BY_LEAGUE, xg_seasons_for

RESULTS_FILE = Path("data/results.json")

def main():
    existing = json.load(RESULTS_FILE.open(encoding='utf-8'))
    existing_matches = existing.get('matches', [])
    already = {(m['league'], m['home'], m['away'], m['kickoff_utc'][:10]) for m in existing_matches}

    divisions = {d: load_division(d) for d in DIVISIONS}
    new_recs = []
    
    # 2122, 2223, 2324 are the earlier seasons of the 5-season window (2021/22 - 2025/26)
    target_seasons = ['2122', '2223', '2324']
    for div, league in DIVISIONS.items():
        rows = divisions[div]
        by_code = {}
        for m in rows:
            by_code.setdefault(m['season'], []).append(m)
        for target in target_seasons:
            ti = ALL_SEASONS.index(target)
            priors = ALL_SEASONS[max(0, ti - 4):ti][::-1]
            if len(priors) < 2:
                continue
            xg_weight = XG_WEIGHT_BY_LEAGUE.get(league, 0.0)
            xg_seasons = (xg_seasons_for(div, list(zip(priors, PRIOR_WEIGHTS)))
                          if xg_weight else None)
            model = LeagueModel([(by_code.get(p, []), w) for p, w in zip(priors, PRIOR_WEIGHTS)],
                                xg_seasons=xg_seasons, xg_weight=xg_weight)
            for m in by_code.get(target, []):
                home = to_pretty(league, m['home'])
                away = to_pretty(league, m['away'])
                d_str = m['date']
                if (league, home, away, d_str) in already:
                    continue
                pred = model.predict(m['home'], m['away'], market_p25=m.get('mk_p25'))
                p05 = pred['p_over_0_5']
                p15 = pred['p_over_1_5']
                p25 = pred['p_over_2_5']
                tot = m['total']
                hit = lambda p, line: int((round(p * 100) >= 50) == (tot > line))
                new_recs.append({
                    'match_id': f"{div}-{d_str}-{len(new_recs):05d}",
                    'league': league,
                    'kickoff_utc': f"{d_str}T12:00:00Z",
                    'home': home,
                    'away': away,
                    'pred_lambda': round(pred['exp_goals'], 3),
                    'p_over_0_5': round(p05, 4),
                    'p_over_1_5': round(p15, 4),
                    'p_over_2_5': round(p25, 4),
                    'score': f"{m['hg']}-{m['ag']}",
                    'total': tot,
                    'hits': {
                        'hit_05': hit(p05, 0.5),
                        'hit_15': hit(p15, 1.5),
                        'hit_25': hit(p25, 2.5)
                    },
                    'lambda_err': round(abs(pred['exp_goals'] - tot), 2),
                    'reconstructed': True
                })

    all_matches = existing_matches + new_recs
    all_matches.sort(key=lambda m: m['kickoff_utc'], reverse=True)

    existing['span'] = "5 Sezon (2021/22 - 2025/26) + Canlı 2026/27"
    existing['reconstructed_count'] = existing.get('reconstructed_count', 0) + len(new_recs)
    existing['matches'] = all_matches

    RESULTS_FILE.write_text(json.dumps(existing, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(f"Added {len(new_recs)} historical matches from 2021/22 - 2023/24.")
    print(f"Total matches in results.json now: {len(all_matches)}")

if __name__ == '__main__':
    main()
