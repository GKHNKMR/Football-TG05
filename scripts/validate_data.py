"""Saatlik botun ürettiği verinin doğrulaması (iş listesi #6).

HATA  -> çıkış kodu 1: workflow commit adımına geçmez, bozuk veri siteye gitmez.
UYARI -> çıkış kodu 0: rapora yazılır (GitHub Actions özetinde görünür), yayını durdurmaz.

Kontroller
  predictions.json (HATA): boş liste, eksik alan, tarih biçimi, olasılık aralığı ve sırası
                           (0.5+ >= 1.5+ >= 2.5+), λ geçersiz, yinelenen maç (match_id / doğal anahtar)
  predictions.json (UYARI): başlamış ama listeden düşmemiş maç, aynı takımın 48 saat içinde iki maçı
                           (yanlış saat / çift fikstür), diğer liglerden 7+ gün geride başlayan lig
  arşiv + results (UYARI):  6 saatten eski ama notlanmamış (skoru gelmemiş) maçlar (son 30 gün)

Kullanım:  python scripts/validate_data.py           (repo kökünden)
"""
import datetime as dt
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent
NOW = dt.datetime.now(dt.timezone.utc)

REQUIRED = ('match_id', 'league', 'home', 'away', 'kickoff_utc', 'p_over_0_5', 'p_over_1_5', 'p_over_2_5')
STALE_AFTER = dt.timedelta(hours=3)        # başlamış maç bu süreden sonra listeden düşmüş olmalı
UNGRADED_AFTER = dt.timedelta(hours=6)     # skor en geç bu kadar sürede gelmeli
UNGRADED_WINDOW = dt.timedelta(days=30)
SAME_TEAM_GAP = dt.timedelta(hours=48)
LEAGUE_LAG = dt.timedelta(days=7)          # bir lig, diğerlerinin ortanca başlangıcından bu kadar geride kalmamalı

errors, warnings = [], []


def parse_ko(s):
    try:
        if not (isinstance(s, str) and s.endswith('Z')):
            return None
        return dt.datetime.fromisoformat(s[:-1] + '+00:00')
    except ValueError:
        return None


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def label(x):
    return f"{x.get('league')} · {x.get('home')} - {x.get('away')} ({x.get('kickoff_utc')})"


def load(rel):
    p = ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def check_predictions(preds):
    if not isinstance(preds, list):
        errors.append('predictions.json liste değil')
        return []
    if not preds:
        errors.append('predictions.json boş: 30 günlük pencerede hiç fikstür yok (kaynak hatası?)')
        return []
    good = []
    seen_id, seen_key = {}, {}
    for i, x in enumerate(preds):
        miss = [k for k in REQUIRED if x.get(k) in (None, '')]
        if miss:
            errors.append(f"#{i} eksik alan {miss}: {label(x)}")
            continue
        ko = parse_ko(x['kickoff_utc'])
        if not ko:
            errors.append(f"Geçersiz başlama saati: {label(x)}")
            continue
        p05, p15, p25 = x['p_over_0_5'], x['p_over_1_5'], x['p_over_2_5']
        if not all(num(v) and 0 <= v <= 1 for v in (p05, p15, p25)):
            errors.append(f"Olasılık 0-1 aralığında değil ({p05}, {p15}, {p25}): {label(x)}")
            continue
        if not (p05 + 1e-6 >= p15 >= p25 - 1e-6):
            errors.append(f"Olasılık sırası bozuk (0.5+ {p05} ≥ 1.5+ {p15} ≥ 2.5+ {p25} olmalı): {label(x)}")
        for k in ('lam_home', 'lam_away'):
            if k in x and not (num(x[k]) and x[k] > 0):
                errors.append(f"Geçersiz {k}={x[k]}: {label(x)}")
        if x['home'] == x['away']:
            errors.append(f"Ev sahibi ve deplasman aynı takım: {label(x)}")
        if x['match_id'] in seen_id:
            errors.append(f"Yinelenen match_id {x['match_id']}: {label(x)}")
        key = (x['league'], x['home'], x['away'], x['kickoff_utc'])
        if key in seen_key:
            errors.append(f"Aynı maç iki kez listelenmiş: {label(x)}")
        seen_id[x['match_id']] = seen_key[key] = True
        if ko < NOW - STALE_AFTER:
            warnings.append(f"Başlayalı {int((NOW - ko).total_seconds() // 3600)} saat oldu ama hâlâ bültende "
                            f"(canlı skor gelmemiş olabilir): {label(x)}")
        good.append((ko, x))
    return good


def check_same_team(good):
    by_team = defaultdict(list)
    for ko, x in good:
        by_team[x['home']].append((ko, x))
        by_team[x['away']].append((ko, x))
    reported = set()
    for team, games in by_team.items():
        games.sort(key=lambda g: g[0])
        for (k1, a), (k2, b) in zip(games, games[1:]):
            pair = (a['match_id'], b['match_id'])
            if k2 - k1 < SAME_TEAM_GAP and pair not in reported:
                reported.add(pair)
                h = (k2 - k1).total_seconds() / 3600
                warnings.append(f"{team} {h:.0f} saat arayla iki maçta (yanlış saat veya çift fikstür?): "
                                f"{label(a)} / {label(b)}")


def check_league_gaps(good):
    nxt = {}
    for ko, x in good:
        nxt[x['league']] = min(nxt.get(x['league'], ko), ko)
    # Milli aralarda tüm ligler birlikte durur; yalnızca diğer ligler oynarken geride kalan lig şüphelidir
    ordered = sorted(nxt.values())
    if len(ordered) >= 3:
        median = ordered[len(ordered) // 2]
        for lg, ko in nxt.items():
            if ko - median > LEAGUE_LAG:
                warnings.append(f"{lg}: sıradaki maç {ko:%d.%m.%Y}, diğer liglerin çoğu {median:%d.%m.%Y} civarı "
                                f"başlıyor — fikstür kaynağını kontrol et")
    return nxt


def check_ungraded(archive, results):
    if not isinstance(archive, dict) or not isinstance(results, dict):
        warnings.append('predictions-archive.json veya results.json okunamadı; notlanmamış maç kontrolü atlandı')
        return
    graded = {(m['league'], m['home'], m['away'], m['kickoff_utc'][:10]) for m in results.get('matches', [])}
    for x in archive.values():
        ko = parse_ko(x.get('kickoff_utc'))
        if not ko or not (NOW - UNGRADED_WINDOW <= ko <= NOW - UNGRADED_AFTER):
            continue
        if (x['league'], x['home'], x['away'], x['kickoff_utc'][:10]) not in graded:
            days = (NOW - ko).days
            warnings.append(f"Skoru gelmemiş / notlanmamış ({days} gün önce): {label(x)} — "
                            f"ertelenmiş olabilir ya da takım adı eşleşmiyor")


def main():
    preds = load('predictions.json')
    if preds is None:
        errors.append('predictions.json bulunamadı')
        good = []
    else:
        good = check_predictions(preds)
        check_same_team(good)
        nxt = check_league_gaps(good)
    check_ungraded(load('data/predictions-archive.json'), load('data/results.json'))

    n = len(preds) if isinstance(preds, list) else 0
    lines = [f"## BETAVUS veri doğrulaması — {NOW:%d.%m.%Y %H:%M} UTC",
             f"{n} fikstür · {len(errors)} hata · {len(warnings)} uyarı", '']
    if good:
        lines.append('Sıradaki maç (lig bazında): ' + ', '.join(f"{lg} {ko:%d.%m}" for lg, ko in sorted(nxt.items())))
        lines.append('')
    for title, items in (('Hatalar (yayın durduruldu)', errors), ('Uyarılar', warnings)):
        if items:
            lines.append(f"### {title}")
            lines += [f"- {s}" for s in items]
            lines.append('')
    report = '\n'.join(lines)
    print(report)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as f:
            f.write(report + '\n')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
