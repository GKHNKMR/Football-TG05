"""Kupon önerisi kuralları + ölçülmüş geçmiş (iş listesi #3) → data/coupon-rules.json

Sanal Kasa'daki "Kupon önerisi" kartı kuralları ve geçmiş sayıları bu dosyadan okur; site kendi
kafasına göre oran/başarı yazmaz. Kurallar scripts/tune_coupon_profiles.py taramasından seçildi:
gerçek (1X2'den türetilmiş) oranı olan Çifte Şans pazarları; hepsi uzun vadede piyasa oranlarıyla
hafif zararda (ROI < 0) — kart bunu açıkça söyler.

Geçmiş: data/stats-5season.json (canlı modelle, yalnızca maç öncesi veriyle; güncel sezon dahil) +
football-data.co.uk piyasa ortalaması oranları. Saatlik bot her gün yeniden üretir.
Kullanım: python scripts/build_coupon_rules.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_coupon_profiles import ROOT, load_odds, legs_by_day, build_coupons, summarize  # noqa: E402

DC = ['1X', '12', 'X2']
# id → Sanal Kasa risk profili (js/paper_engine.js RISK_PROFILES) eşleşmesi
TIERS = [
    dict(id='low', profile='minimum', markets=DC, thr=0.85, legs=1),
    dict(id='mid', profile='medium', markets=DC, thr=0.80, legs=2),
    dict(id='high', profile='high', markets=DC, thr=0.80, legs=3),
]
STAKE_PCT = 0.25   # kasa simülasyonu yalnızca "en kötü sezon" göstergesi için


def main():
    d = json.loads((ROOT / 'data' / 'stats-5season.json').read_text(encoding='utf-8'))
    days, _ = legs_by_day(d['rows'], load_odds())
    out = []
    for t in TIERS:
        s = summarize(build_coupons(days, t['markets'], t['thr'], t['legs']), STAKE_PCT)
        hist = None if not s else {
            'coupons': s['n'], 'win_pct': round(s['win'] * 100, 1), 'model_pct': round(s['model_p'] * 100, 1),
            'avg_odds': round(s['avg_odds'], 2), 'fair_odds': round(1 / s['win'], 2) if s['win'] else None,
            'roi_pct': round(s['roi'] * 100, 1), 'max_drawdown_pct': round(s['dd_max'] * 100),
        }
        out.append({**t, 'history': hist})
        print(t['id'], hist)
    res = {'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
           'seasons': d.get('seasons'), 'odds_source': 'football-data.co.uk piyasa ortalaması (1X2 → Çifte Şans)',
           'tiers': out}
    (ROOT / 'data' / 'coupon-rules.json').write_text(json.dumps(res, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    print('Wrote data/coupon-rules.json')


if __name__ == '__main__':
    main()
