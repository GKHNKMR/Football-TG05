# BETAVUS ⚽

Football Goal Probability Engine.

BETAVUS is a mobile-friendly football dashboard for the following nine leagues:

- Premier League
- Championship
- Turkish Süper Lig
- LaLiga
- Bundesliga
- Serie A
- Ligue 1
- Eredivisie
- Primeira Liga

## Architecture

There is **no serverless backend** and **no API key ships to the browser** -
the static site only ever fetches JSON files out of this repo.

`openfootball/football.json + football-data.co.uk (+ tff.org, + ESPN scoreboard) → GitHub Actions (hourly) → predictions.json / data/*.json → static site`

Fixtures and historical results come from the open-data project
[`openfootball/football.json`](https://github.com/openfootball/football.json)
(`{season}/{code}.json`, e.g. `2026-27/en.1.json`). Season files are mirrored
under `data/cache/openfootball/` so the build is reproducible offline.

The job ([`scripts/update_predictions.py`](scripts/update_predictions.py), run
hourly) downloads the current season plus the last three completed ones for
each league, builds a weighted home/away goals model and a head-to-head record,
and commits the resulting `predictions.json`. The static site only reads that
JSON file (`index.html` fetches the repo-relative `predictions.json`).

### Live scores (ESPN scoreboard, no key needed)

openfootball and football-data.co.uk are community-maintained archives, not
live feeds - in practice they can lag real matches by up to a week.
[`scripts/fetch_live_scores.py`](scripts/fetch_live_scores.py) pulls the last
14 days of fixtures across our leagues from ESPN's public (undocumented, free,
keyless) soccer scoreboard API into `data/live-scores.json` before the other
scripts run. Each run merges its findings into the existing file rather than
overwriting it, so a transient per-request failure never erases a day an
earlier run already captured; a per-day fetch status is written to
`data/live-scores-debug.json` for troubleshooting.

An earlier version of this used API-Football instead, but its free tier only
allows querying a yesterday/today/tomorrow window - useless for backfilling
older stuck matches - and the paid tiers cost money for something ESPN already
gives away, so it was dropped.

- `update_predictions.py` drops a fixture from Tahminler the moment it's
  actually finished (instead of waiting on openfootball's score) and attaches
  a `live: {status, score}` field while a match is still being played.
- `build_results.py` grades an archived (real pre-kickoff) prediction from
  this feed when football-data.co.uk hasn't posted the result yet, tagging the
  row `live_source: true`; the pre-captured market odds are still used for
  `edge25`/`value_hit`, only the score itself comes from the live feed.

### Leagues and sources

| League | openfootball file |
| --- | --- |
| Premier League | `2026-27/en.1.json` |
| Championship | `2026-27/en.2.json` |
| Turkish Süper Lig | football-data only (openfootball has no current fixtures) |
| LaLiga | `2026-27/es.1.json` |
| Bundesliga | `2026-27/de.1.json` |
| Serie A | `2026-27/it.1.json` |
| Ligue 1 | `2026-27/fr.1.json` |
| Eredivisie | `2026-27/nl.1.json` |
| Primeira Liga | `2026-27/pt.1.json` |

## Model

[`scripts/goals_model.py`](scripts/goals_model.py) is the single shared
implementation the live predictor, the backtest and the leak-free
reconstruction all import (no duplicated model math). It combines:

1. Each team's home / away goals-for and goals-against rates
2. Four seasons of results, weighted toward the most recent (1.0 / 0.7 / 0.45 / 0.30)
   - **and**, within that, an exponential match-recency decay (half-life 6
     matches) so a team's last 5-10 games dominate its rate estimate instead
     of being diluted evenly across a whole season
3. League-average fallback for newly promoted teams with no top-flight history
4. Head-to-head record, last 8 meetings, blended in at 28% when 2+ meetings exist
5. A **Dixon-Coles** low-score correction on top of the independent-Poisson
   grid: plain Poisson(lam_home) x Poisson(lam_away) under-counts 0-0/1-0/0-1/1-1
   relative to what leagues actually produce, so those four cells get a
   `tau(x,y,rho)` adjustment before the grid is renormalized. rho is fit per
   league by a 1D grid-search MLE against that league's own historical
   low-score frequencies (Dixon & Coles 1997), not a fixed constant.

The dashboard displays Over 0.5, Over 1.5 and Over 2.5 goal probabilities,
read off the Dixon-Coles-adjusted scoreline grid. Each row in
`predictions.json` carries a `basis` field (`form`, `form+h2h`,
`partial-form`, `league-avg`) showing how much real data backs it, plus
`lam_home`/`lam_away`/`rho` for anyone who wants the underlying model state.

## Match stats (click a fixture)

A second data layer powers the drawer that opens when a fixture row is clicked:
head-to-head history, recent form, current-season aggregates, and a
**per-match goal-average chart** (both teams' total goals per game over the last
five completed seasons, drawn as inline SVG).

`football-data.co.uk → CSVs → data/match-stats.json → drawer`

- [`scripts/fetch_football_data.py`](scripts/fetch_football_data.py) downloads the
  *Main Leagues* CSVs (top division only) into `data/football-data/<DIV>/<SEASON>.csv`
  for eight seasons. These raw CSVs are the local stats database (FT/HT results,
  shots, corners, cards, 1X2 / Over-Under / Asian-handicap odds).
- [`scripts/build_match_stats.py`](scripts/build_match_stats.py) maps each fixture's
  teams onto the football-data naming, then writes per-`match_id` H2H + form +
  season summaries to `data/match-stats.json`, which `index.html` fetches once.

| League | football-data division | CSV |
| --- | --- | --- |
| Premier League | `E0` | `mmz4281/<season>/E0.csv` |
| Championship | `E1` | `mmz4281/<season>/E1.csv` |
| Turkish Süper Lig | `T1` | `mmz4281/<season>/T1.csv` |
| LaLiga | `SP1` | `mmz4281/<season>/SP1.csv` |
| Bundesliga | `D1` | `mmz4281/<season>/D1.csv` |
| Serie A | `I1` | `mmz4281/<season>/I1.csv` |
| Ligue 1 | `F1` | `mmz4281/<season>/F1.csv` |
| Eredivisie | `N1` | `mmz4281/<season>/N1.csv` |
| Primeira Liga | `P1` | `mmz4281/<season>/P1.csv` |

Use the apex domain `football-data.co.uk` (the `www` host currently 503s).

## Backtest (model validation — off-site)

[`scripts/backtest.py`](scripts/backtest.py) walk-forward tests the goal model:
each target season (2021/22 → 2025/26) is predicted using **only the seasons
before it** (up to four, no result leakage), then scored against what actually
happened, across all nine leagues. Output `data/backtest.json` is rendered both by the **Model doğruluğu** tab in the app and by the standalone
[`backtest.html`](backtest.html) (`/backtest.html`): matches tested plus
0.5/1.5/2.5 Üst direction accuracy, sliceable by league and season.

Headline: the model is **well calibrated** (a stated 60% comes in near 60%) and
**unbiased** on expected goals (bias ≈ −0.04), but single-match discrimination
over the base-rate baseline is modest (Over 2.5 Brier skill ≈ +1.8%). Marketing
copy should lean on *calibration*, never on a hit-rate guarantee.

Run it with `python scripts/backtest.py`; it is **not** part of the daily
workflow.

## Deployment

The repository is connected to the static deployment. Any push to `main` triggers a new static deployment.

## Paper-Betting & Sanal Kasa Yönetimi (Kasa Planı)

BETAVUS, futbol toplam gol pazarları (0.5 Üst, 1.5 Üst, 2.5 Üst) için yapay zekâ destekli bir **paper-betting, kupon planlama ve sanal kasa yönetimi platformudur**.

> **Yasal Uyarı & İlke:**
> BETAVUS bir bahis sitesi, bahis operatörü veya ödeme platformu değildir. Site bahis kabul etmez, para yatırma/çekme işlemi yapmaz ve kullanıcı adına harici bir bahis sitesinde kupon oynatmaz. Bütün bakiye, stake ve kazanç hesaplamaları tamamen **sanal (paper-betting)** simülasyondan ibarettir. Hiçbir tahmin garanti kazanç vaat etmez; kayıp kovalama (Martingale vb.) yöntemleri desteklenmez.

**Motto:** *"Önce simüle et. Riskini gör. Stratejini ölç. Sonra karar ver."*

### 5 Sekmeli Mimari

1. **⚽ Tahminler:** 9 lig için Poisson ve Dixon-Coles düzeltmeli maç bazlı 0.5/1.5/2.5 Üst olasılıkları ve detaylı H2H/Form istatistikleri.
2. **🎯 Tahmin vs Gerçekleşen:** Modelin 5 sezonluk walk-forward geçmişi, lig bazlı başarı oranları ve 12 Eylül 2026 sonrası canlı doğruluk analizi.
3. **💡 Kupon Önerileri:** Kullanıcının risk profiline (Temkinli, Dengeli, Atak) göre otomatik oluşturulan minimum, orta ve yüksek riskli kuponlar.
   - **Kupon Düzenleme Modülü:** Maç çıkarma, `+ Maç Ekle` ile uygun fikstürlerden seçim yapma, pazar değiştirme (0.5/1.5/2.5) ve harici oynanan gerçek oranı girebilme.
   - **Metrikler:** Birleşik olasılık, tahmini adil oran, başabaş olasılık ($1/\text{oran}$), beklenen değer ($EV = (P \times \text{oran}) - 1$).
4. **📋 Kuponlarım:**
   - **⏳ Bekleyenler:** Fikstürleri oynanmayı veya sonuçlanmayı bekleyen sanal kuponlar.
   - **✅ Sonuçlananlar:** Otomatik sonuçlandırılan kuponlar, toplam net kâr/zarar, ROI, kazanma oranı, seri istatistikleri ve max drawdown.
   - **📝 Taslaklar:** Hazırlanıp henüz plana dahil edilmemiş kuponlar.
   - **📊 12 Eylül Canlı Model Takibi:** 12.09.2026 tarihinden itibaren sistemin yüksek güvenle vurguladığı maçların canlı kümülatif takip istatistikleri.
5. **🏦 Kasa Planım:**
   - **Plan Tanımı:** Sanal başlangıç kasası ($S$), hedef kasa ($T$), plan süresi ($D$ gün) ve risk toleransı.
   - **Risk Profilleri ve Kasa Rezervleri:**
     - **Temkinli / Minimum Risk:** %75 Kasa Rezervi (%20 Minimum Risk kolu, %5 Orta Risk kolu, %0 Yüksek Risk).
     - **Dengeli / Medium:** %50 Kasa Rezervi (%30 Minimum Risk, %16 Orta Risk, %4 Yüksek Risk).
     - **Agresif:** %35 Kasa Rezervi (%40 Minimum Risk, %18 Orta Risk, %7 Yüksek Risk).
   - **Geometrik Büyüme Patikası:** $\text{hedef\_yolu}(d) = S \cdot (T/S)^{d/D}$ formülüyle günlük hedeflenen bakiye çizgisi.
   - **Monte Carlo Simülasyonu:** 5.000 iterasyonluk Mulberry32 PRNG motoru ile hedefe ulaşma olasılığı ($P(\text{hedef})$), beklenen medyan bakiye ve %5 VaR (Value at Risk) risk koridoru.
   - **Adaptif Öneri Motoru:** Planda sapma olduğunda kullanıcı onayıyla seçilebilecek 3 somut opsiyon: Süreyi uzatma, Hedefi revize etme, Risk profilini değiştirme.
   - **Veri Yedekleme:** `betavus.paper_v1` LocalStorage anahtarı üzerinden JSON dışa aktarma ve içe aktarma desteği.

### Otomatik Sonuçlandırma (Idempotent Settlement)

`js/paper_engine.js` içerisinde yer alan settlement motoru, `data/results.json` ve `data/live-scores.json` verilerini dinleyerek kuponları maçlar biter bitmez otomatik değerlendirir. Önceden sonuçlanan kuponlar tekrar hesaplanmaz (idempotent), sanal bakiye mükerrer güncellenmez.

### Otomasyon ve Testler

- `scripts/test_paper_betting.py`: Şartnamede tanımlanan Senaryo A–F, Monte Carlo simülasyonu, adaptif seçenekler, LocalStorage şeması ve DOM entegrasyonunu doğrulayan 10 adımlı test suite.
- `scripts/verify_e2e.py`: Tüm sekmelerin ve veri akışının geriye dönük uyumluluğunu doğrulayan 8 adımlı E2E regression testi.

<!-- BETAVUS data pipeline verified 2026-09-20 -->
<!-- Calculation run requested 2026-09-20 -->

