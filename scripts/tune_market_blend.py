"""Piyasa harmanı ağırlığı (goals_model.MARKET_WEIGHT) - walk-forward ölçüm.

Sorun: model yüksek lambda'larda iyimser (lambda >= 3.25 maçlarda gerçeği 0.3-0.5
gol fazla tahmin ediyor); 1.5+ vurgularında model ~%87.7 derken gerçek ~%84.6.
Tutmayan vurguların ortak noktası: piyasa (bahis oranları) o maçlara modelden
belirgin şekilde az gol veriyordu. Piyasa kadro, sakatlık, motivasyon gibi
modelin görmediği bilgileri taşıyor.

Düzeltme: maçın 2.5 üst/alt oranı varsa toplam lambda
    (1 - w) * model_toplam + w * piyasa_toplam
olur; piyasa_toplam, modelin ev/deplasman oranı ve rho'su korunarak Dixon-Coles
P(toplam>2.5)'i piyasanınkine eşitleyen toplamdır (goals_model.market_total).

Ölçüm canlı modeli taklit eder (update_predictions.SEASONS): hedef sezonun o
güne kadar oynanmış maçları ağırlık 1.0, önceki üç sezon 0.7 / 0.45 / 0.30;
model her REFIT_DAYS günde bir yeniden kurulur, hiçbir maç kendi sonucunu
görmez. Her sezonun w'si YALNIZCA önceki sezonlardan öğrenilir (örneklem
dışı), sonra o sezona uygulanır - sistem her sezon kendi hatalarından öğrenir.

Kullanım:  python scripts/tune_market_blend.py
"""

import csv
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
from backtest import load_division, DIVISIONS, ALL_SEASONS, TARGET_SEASONS  # noqa: E402
from goals_model import LeagueModel, low_total_cdf, market_total, MARKET_WEIGHT  # noqa: E402
from xg_blend import XG_WEIGHT_BY_LEAGUE, xg_seasons_for  # noqa: E402

CURRENT = "2627"
CHAIN = ALL_SEASONS + [CURRENT]
LIVE_WEIGHTS = [1.0, 0.7, 0.45, 0.30]   # güncel sezon, sonra 1-2-3 önceki sezon
REFIT_DAYS = 7
GRID = [round(0.1 * i, 1) for i in range(11)]   # 0.0 .. 1.0


def collect():
    """Her maç için ham model tahmini (piyasasız) + piyasa toplamı + sonuç."""
    out = []
    for div, league in DIVISIONS.items():
        by_code = {}
        for m in load_division(div, CHAIN):
            by_code.setdefault(m["season"], []).append(m)
        xg_weight = XG_WEIGHT_BY_LEAGUE.get(league, 0.0)
        for target in TARGET_SEASONS + [CURRENT]:
            ti = CHAIN.index(target)
            priors = CHAIN[max(0, ti - 3):ti][::-1]
            if len(priors) < 2:
                continue
            prior_seasons = [(by_code.get(p, []), w) for p, w in zip(priors, LIVE_WEIGHTS[1:])]
            # Güncel sezonun xG'si bütün sezonu kapsar (sızıntı) -> yalnızca önceki sezonlar
            xg_seasons = xg_seasons_for(div, list(zip(priors, LIVE_WEIGHTS[1:]))) if xg_weight else None
            matches = sorted(by_code.get(target, []), key=lambda m: m["date"])
            i = 0
            while i < len(matches):
                end = (date.fromisoformat(matches[i]["date"]) + timedelta(days=REFIT_DAYS)).isoformat()
                model = LeagueModel([(matches[:i], LIVE_WEIGHTS[0])] + prior_seasons,
                                    xg_seasons=xg_seasons, xg_weight=xg_weight)
                while i < len(matches) and matches[i]["date"] < end:
                    m = matches[i]
                    i += 1
                    p = model.predict(m["home"], m["away"])
                    b = p["basis"]
                    limited = (b.startswith("partial-form") or b.startswith("league-avg")) and p["h2h_matches_used"] < 2
                    lh, la, rho = p["lam_home"], p["lam_away"], p["rho"]
                    mk = m.get("mk_p25")
                    tm = market_total(lh, la, rho, mk) if mk is not None and 0.02 < mk < 0.98 else None
                    out.append((league, target, lh, la, rho, tm, m["total"], limited))
    return out


def score(rows, w):
    """(Brier ort. 0.5/1.5/2.5, 1.5+ vurgu [n, model p, tutan], 0.5+ vurgu [...])"""
    brier = 0.0
    p15, p05 = [0, 0.0, 0], [0, 0.0, 0]
    for _lg, _s, lh, la, rho, tm, tg, limited in rows:
        if tm is not None and w:
            tot = lh + la
            new = (1 - w) * tot + w * tm
            lh, la = lh * new / tot, la * new / tot
        c0, c1, c2 = low_total_cdf(lh, la, rho)
        o05, o15, o25 = 1 - c0, 1 - c1, 1 - c2
        brier += ((o05 - (tg > 0)) ** 2 + (o15 - (tg > 1)) ** 2 + (o25 - (tg > 2)) ** 2) / 3
        if not limited:
            if o15 >= 0.85:
                p15[0] += 1; p15[1] += o15; p15[2] += tg > 1
            if o05 >= 0.95:
                p05[0] += 1; p05[1] += o05; p05[2] += tg > 0
    return brier / len(rows), p15, p05


def fmt(p):
    return f"{p[0]:5d} vurgu, model %{100 * p[1] / p[0]:.1f} / gerçek %{100 * p[2] / p[0]:.1f}" if p[0] else "    0 vurgu"


def main():
    rows = collect()
    with_mk = [r for r in rows if r[5] is not None]
    print(f"{len(rows)} maç, {len(with_mk)} tanesinde piyasa oranı var")
    done = [r for r in with_mk if r[1] != CURRENT]
    print("Brier eğrisi (5 tamamlanmış sezon):", {w: round(score(done, w)[0], 4) for w in GRID})

    seasons = TARGET_SEASONS + [CURRENT]
    tot = {k: [0, 0.0, 0] for k in ("o15", "n15", "o05", "n05")}
    print("\nWalk-forward: her sezonun w'si yalnızca önceki sezonlardan öğrenilir")
    for i, s in enumerate(seasons):
        past = [r for r in with_mk if r[1] in seasons[:i]]
        w = min(GRID, key=lambda g: score(past, g)[0]) if past else 0.0
        cur = [r for r in rows if r[1] == s]
        _, o15, o05 = score(cur, 0.0)
        _, n15, n05 = score(cur, w)
        for k, v in (("o15", o15), ("n15", n15), ("o05", o05), ("n05", n05)):
            tot[k] = [a + b for a, b in zip(tot[k], v)]
        print(f"  {s}: w={w}  1.5+ eski {fmt(o15)}  ->  yeni {fmt(n15)}")
    print("\nTOPLAM (örneklem dışı):")
    print(f"  1.5+ eski {fmt(tot['o15'])}\n  1.5+ yeni {fmt(tot['n15'])}")
    print(f"  0.5+ eski {fmt(tot['o05'])}\n  0.5+ yeni {fmt(tot['n05'])}")
    best = min(GRID, key=lambda g: score(with_mk, g)[0])
    print(f"\nTüm veriyle en iyi w={best}  (goals_model.MARKET_WEIGHT şu an {MARKET_WEIGHT})")


if __name__ == "__main__":
    main()
