"""BETAVUS Paper-Betting, Kupon Planlama ve Sanal Kasa Yönetimi Kapsamlı Test Paketi.
Şartnamedeki tüm zorunlu senaryoları (A-F) ve kabul kriterlerini doğrular.
"""

import sys, os, time, threading, json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

PORT = 8898

def run_server():
    server = HTTPServer(('127.0.0.1', PORT), QuietHandler)
    server.serve_forever()

def main():
    print(f"Starting server at http://127.0.0.1:{PORT}...")
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(0.5)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Set bypass access gate
        page.add_init_script("""
            localStorage.setItem('betavus.access', '1f7b720c52ea3f6e8631a8eeaffaa7113fbed540ec0772108d39c52835d9855d');
            localStorage.setItem('betavus.tab', 'plan');
        """)

        print(f"Navigating to http://127.0.0.1:{PORT}/index.html...")
        page.goto(f"http://127.0.0.1:{PORT}/index.html", wait_until="networkidle")
        time.sleep(1)

        # ----------------------------------------------------------------------
        # TEST 1: Saf Hesaplama Motoru (Pure Functions) & Konfigürasyon
        # ----------------------------------------------------------------------
        print("\n--- TEST 1: Pure Engine & Configuration ---")
        cfg_res = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            if (!PE) return { error: 'BETAVUS_PAPER bulunamadı' };
            return {
                profiles: Object.keys(PE.RISK_PROFILES),
                minimum: PE.RISK_PROFILES.minimum,
                medium: PE.RISK_PROFILES.medium,
                high: PE.RISK_PROFILES.high,
                multi: PE.RISK_PROFILES.multi,
                cautious: PE.RISK_PROFILES.cautious,
                balanced: PE.RISK_PROFILES.balanced,
                aggressive: PE.RISK_PROFILES.aggressive,
                classes: Object.keys(PE.COUPON_CLASSES),
                minClass: PE.COUPON_CLASSES.minimum,
                medClass: PE.COUPON_CLASSES.medium,
                highClass: PE.COUPON_CLASSES.high
            };
        }""")
        assert 'error' not in cfg_res, f"Config error: {cfg_res.get('error')}"
        assert 'minimum' in cfg_res['profiles'] and 'medium' in cfg_res['profiles'] and 'high' in cfg_res['profiles'] and 'multi' in cfg_res['profiles']
        # Minimum Risk: Kasa Rezerv: %50, Günlük Büyüme Oranı: %15 (1.15x, Aktif Pay: %50)
        assert cfg_res['minimum']['reservePct'] == 0.50
        assert cfg_res['minimum']['dailyGrowthRate'] == 0.15
        assert cfg_res['minimum']['dailyFactor'] == 1.15
        assert cfg_res['minimum']['stakePct'] == 0.50

        # Orta Risk: Kasa Rezerv: %35, Günlük Büyüme Oranı: %20 (1.20x, Aktif Pay: %65)
        assert cfg_res['medium']['reservePct'] == 0.35
        assert cfg_res['medium']['dailyGrowthRate'] == 0.20
        assert cfg_res['medium']['dailyFactor'] == 1.20
        assert cfg_res['medium']['stakePct'] == 0.65

        # Yüksek Risk: Kasa Rezerv: %25, Günlük Büyüme Oranı: %25 (1.25x, Aktif Pay: %75)
        assert cfg_res['high']['reservePct'] == 0.25
        assert cfg_res['high']['dailyGrowthRate'] == 0.25
        assert cfg_res['high']['dailyFactor'] == 1.25
        assert cfg_res['high']['stakePct'] == 0.75

        # Geriye dönük uyumluluk takma adları
        assert cfg_res['cautious'] is not None and cfg_res['balanced'] is not None and cfg_res['aggressive'] is not None
        print("  ✓ Kasa risk profilleri (Minimum Risk %50 Rezerv / %15 Büyüme, Orta Risk %35 Rezerv / %20 Büyüme, Yüksek Risk %25 Rezerv / %25 Büyüme) doğrulandı.")

        # Günlük Büyüme Modeli Matematik Doğrulaması (minimum risk: %50 rezerv, %15 büyüme)
        excel_math = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            const m = PE.calculateExcelGrowthModel({ startingBank: 50, durationDays: 30 }, 'minimum');
            return {
                factor: m.dailyGrowthFactor,
                ratePct: m.dailyGrowthRatePct,
                finalBank: m.finalTheoreticalBank,
                multiplier: m.totalGrowthMultiplier,
                controlStatus: m.controlStatus,
                dayPointsLen: m.dayPoints.length,
                day0: m.dayPoints[0].theoreticalBank,
                day1: m.dayPoints[1].theoreticalBank,
                day30: m.dayPoints[30].theoreticalBank
            };
        }""")
        assert excel_math['factor'] == 1.15
        assert excel_math['ratePct'] == 15.0
        assert excel_math['finalBank'] == 3310.59
        assert excel_math['multiplier'] == 66.21
        assert excel_math['controlStatus'] == 'OK'
        assert excel_math['dayPointsLen'] == 31
        assert excel_math['day0'] == 50.0
        assert excel_math['day1'] == 57.50
        assert excel_math['day30'] == 3310.59
        print("  ✓ Günlük büyüme katsayısı ve bileşik kasa matematik projeksiyonu (1.15x, %15.0, 3.310,59€) doğrulandı.")

        # ----------------------------------------------------------------------
        # TEST 2: Senaryo A — Plan Oluşturma & Geometrik Hedef Yolu
        # ----------------------------------------------------------------------
        print("\n--- TEST 2: Senaryo A — Plan Oluşturma ---")
        # Başlangıç: 50 EUR, Hedef: 500 EUR, Süre: 30 gün, Temkinli
        plan_res = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            const state = PE.createInitialState(
                { currency: 'EUR', riskProfile: 'cautious' },
                { startingBank: 50, targetBank: 500, durationDays: 30 }
            );
            const dailyRate = PE.calculateRequiredDailyRate(50, 500, 30);
            const day0 = PE.calculateTargetPath(state.plan, 0);
            const day15 = PE.calculateTargetPath(state.plan, 15);
            const day30 = PE.calculateTargetPath(state.plan, 30);

            const statusOnTrack = PE.classifyPlanStatus(day15, day15);
            const statusAhead = PE.classifyPlanStatus(day15 * 1.10, day15);
            const statusBehind = PE.classifyPlanStatus(day15 * 0.90, day15);

            return {
                state,
                dailyRate,
                day0,
                day15,
                day30,
                statusOnTrack: statusOnTrack.code,
                statusAhead: statusAhead.code,
                statusBehind: statusBehind.code
            };
        }""")

        assert plan_res['state']['plan']['startingBank'] == 50
        assert plan_res['state']['plan']['targetBank'] == 500
        assert plan_res['state']['plan']['durationDays'] == 30
        assert plan_res['day0'] == 50.0
        assert plan_res['day30'] == 500.0
        # 50 * (10)^(15/30) = 50 * sqrt(10) ~ 158.11
        assert 155 <= plan_res['day15'] <= 162
        # (10)^(1/30) - 1 ~ 8.0%
        assert 7.8 <= plan_res['dailyRate'] <= 8.2
        assert plan_res['statusOnTrack'] == 'on_track'
        assert plan_res['statusAhead'] == 'ahead'
        assert plan_res['statusBehind'] == 'behind'

        # Trajectories hesaplama testi (her risk modeli için hedeflenen sürede kasa ulaşma eğrisi)
        traj_test = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            const plan = { startingBank: 50, targetBank: 500, durationDays: 30 };
            const traj = PE.calculatePlanTrajectories(plan, null, { iterations: 500, seed: 42 });
            return {
                startBank: traj.startBank,
                targetBank: traj.targetBank,
                durationDays: traj.durationDays,
                targetPointsLen: traj.targetPoints.length,
                target0: traj.targetPoints[0].targetBank,
                target30: traj.targetPoints[30].targetBank,
                hasMinimum: !!traj.trajectories.minimum,
                hasMedium: !!traj.trajectories.medium,
                hasHigh: !!traj.trajectories.high,
                hasMulti: !!traj.trajectories.multi,
                minPointsLen: traj.trajectories.minimum.dayPoints.length,
                min0: traj.trajectories.minimum.dayPoints[0].median,
                minHitPct: traj.trajectories.minimum.targetHitPct
            };
        }""")
        assert traj_test['targetPointsLen'] == 31  # Gün 0'dan 30'a 31 nokta
        assert traj_test['target0'] == 50.0
        assert traj_test['target30'] == 500.0
        assert traj_test['hasMinimum'] and traj_test['hasMedium'] and traj_test['hasHigh'] and traj_test['hasMulti']
        assert traj_test['minPointsLen'] == 31
        assert traj_test['min0'] == 50.0
        print("  ✓ Hedeflenen sürede kasa ulaşma trajektorisi ve 4 model hesaplama projeksiyonu doğrulandı.")
        print(f"  ✓ Günlük gerekli oran: %{plan_res['dailyRate']}, Gün 0: {plan_res['day0']}€, Gün 15: {plan_res['day15']}€, Gün 30: {plan_res['day30']}€")
        print("  ✓ Geometrik hedef yolu ve durum sınıflandırması doğrulandı.")

        # ----------------------------------------------------------------------
        # TEST 3: Senaryo B — Öneriyi Düzenleme (Maç Çıkar/Ekle & Canlı Güncelleme)
        # ----------------------------------------------------------------------
        print("\n--- TEST 3: Senaryo B — Öneriyi Düzenleme ---")
        edit_res = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            const leg1 = { matchId: 'M1', league: 'PL', home: 'A', away: 'B', market: 'over_0_5', probability: 0.96, estimatedLegOdds: 1.04 };
            const leg2 = { matchId: 'M2', league: 'LL', home: 'C', away: 'D', market: 'over_0_5', probability: 0.95, estimatedLegOdds: 1.05 };
            const leg3 = { matchId: 'M3', league: 'SA', home: 'E', away: 'F', market: 'over_0_5', probability: 0.97, estimatedLegOdds: 1.03 };

            // İlk 2 maçlı kupon
            const pInit = PE.calculateCombinedProbability([leg1, leg2]);
            const oInit = PE.calculateEstimatedOdds([leg1, leg2], 'minimum');

            // 1 maç çıkarıldı (leg2 çıkarıldı)
            const pAfterRemove = PE.calculateCombinedProbability([leg1]);
            const oAfterRemove = PE.calculateEstimatedOdds([leg1], 'minimum');

            // Başka maç eklendi (leg3 eklendi)
            const pAfterAdd = PE.calculateCombinedProbability([leg1, leg3]);
            const oAfterAdd = PE.calculateEstimatedOdds([leg1, leg3], 'minimum');

            return {
                pInit, oInit,
                pAfterRemove, oAfterRemove,
                pAfterAdd, oAfterAdd
            };
        }""")

        assert edit_res['pInit'] == round(0.96 * 0.95, 4)
        assert edit_res['pAfterRemove'] == 0.96
        assert edit_res['pAfterAdd'] == round(0.96 * 0.97, 4)
        assert edit_res['pAfterAdd'] > edit_res['pInit']
        print(f"  ✓ İlk olasılık: {edit_res['pInit']}, Maç çıkarılınca: {edit_res['pAfterRemove']}, Yeni maç eklenince: {edit_res['pAfterAdd']}")
        print("  ✓ Kupon düzenlemede canlı olasılık ve oran güncellemeleri doğrulandı.")

        # ----------------------------------------------------------------------
        # TEST 4: Senaryo C — Gerçek Oran Girişi, Başa Baş ve EV Hesabı
        # ----------------------------------------------------------------------
        print("\n--- TEST 4: Senaryo C — Gerçek Oran Girişi & EV ---")
        # Tahmini oran: 1.70, Kullanıcı girdiği gerçek oran: 1.55
        odds_res = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            const prob = 0.72; // model olasılığı
            const estOdds = 1.70;
            const actualOdds = 1.55;
            const stake = 10.0;

            const beProb = PE.calculateBreakEvenProbability(actualOdds);
            const ev = PE.calculateExpectedValue(prob, actualOdds);
            const ret = PE.calculatePotentialReturn(stake, actualOdds);
            const net = PE.calculatePotentialNet(stake, actualOdds);

            return { beProb, ev, ret, net };
        }""")

        # 1 / 1.55 ~ 0.6452
        assert 0.64 <= odds_res['beProb'] <= 0.65
        # 0.72 * 1.55 - 1 = 1.116 - 1 = 0.116
        assert 0.11 <= odds_res['ev'] <= 0.12
        assert odds_res['ret'] == 15.5
        assert odds_res['net'] == 5.5
        print(f"  ✓ Başa baş olasılık: %{round(odds_res['beProb']*100, 1)}, EV: %{round(odds_res['ev']*100, 1)}, Getiri: {odds_res['ret']}€, Net: {odds_res['net']}€")
        print("  ✓ Gerçek oran üzerinden formüller doğrulandı.")

        # ----------------------------------------------------------------------
        # TEST 5: Senaryo D — Otomatik Sonuçlandırma Kuralları
        # ----------------------------------------------------------------------
        print("\n--- TEST 5: Senaryo D — Otomatik Sonuçlandırma ---")
        # 2.5 Üst seçimi 2-1 biten maçta kazanmalı
        # 2.5 Üst seçimi 1-1 biten maçta kaybetmeli
        # Sonucu gelmeyen maç varsa kupon beklemede kalmalı
        settle_res = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            const sel1 = { matchId: 'M1', market: 'over_2_5', line: '2.5', probability: 0.76 };
            const sel2 = { matchId: 'M2', market: 'over_2_5', line: '2.5', probability: 0.77 };
            const selPending = { matchId: 'M3', market: 'over_2_5', line: '2.5', probability: 0.75 };

            const matchWon = { match_id: 'M1', score: '2-1', total: 3 };
            const matchLost = { match_id: 'M2', score: '1-1', total: 2 };

            const r1 = PE.settleSelection(sel1, matchWon);
            const r2 = PE.settleSelection(sel2, matchLost);
            const r3 = PE.settleSelection(selPending, null);

            // Kupon testleri
            const slipWon = {
                id: 'S-W', status: 'pending', couponClass: 'high', oddsUsed: 2.5, stake: 10,
                selections: [sel1]
            };
            const slipLost = {
                id: 'S-L', status: 'pending', couponClass: 'high', oddsUsed: 2.5, stake: 10,
                selections: [sel1, sel2]
            };
            const slipWait = {
                id: 'S-P', status: 'pending', couponClass: 'high', oddsUsed: 2.5, stake: 10,
                selections: [sel1, selPending]
            };

            const lookup = PE.buildResultsLookup([matchWon, matchLost]);
            const resWon = PE.settleSlip(slipWon, lookup);
            const resLost = PE.settleSlip(slipLost, lookup);
            const resWait = PE.settleSlip(slipWait, lookup);

            return {
                r1: r1.result,
                r2: r2.result,
                r3: r3.result,
                slipWonStatus: resWon.slip.status,
                slipLostStatus: resLost.slip.status,
                slipWaitStatus: resWait.slip.status,
                resWaitDebug: resWait
            };
        }""")

        print(f"    settle_res: {settle_res}")
        assert settle_res['r1'] == 'won', f"2-1 match failed to win 2.5 over: {settle_res['r1']}"
        assert settle_res['r2'] == 'lost', f"1-1 match failed to lose 2.5 over: {settle_res['r2']}"
        assert settle_res['r3'] == 'pending'
        assert settle_res['slipWonStatus'] == 'won'
        assert settle_res['slipLostStatus'] == 'lost'
        assert settle_res['slipWaitStatus'] == 'pending'
        print("  ✓ 2-1 maçta 2.5 Üst kazandı, 1-1 maçta kaybetti, eksik maçta beklemede kaldı.")

        # ----------------------------------------------------------------------
        # TEST 6: Senaryo E — Bakiye ve Sanal Kasa Akışı
        # ----------------------------------------------------------------------
        print("\n--- TEST 6: Senaryo E — Bakiye ve Sanal Kasa Akışı ---")
        # Kullanılabilir bakiye: 100 EUR, Sanal stake: 10 EUR, Gerçek oran: 2.00
        # Plana eklenince -> 90 EUR
        # Kazanırsa -> 110 EUR (+20 EUR ödeme)
        # Kaybederse -> 90 EUR
        bal_res = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            let state = PE.createInitialState(
                { currency: 'EUR', riskProfile: 'cautious' },
                { startingBank: 100, targetBank: 1000, durationDays: 30 }
            );

            const sel = { matchId: 'MATCH-X', market: 'over_1_5', line: '1.5', probability: 0.88 };
            const slip = PE.createSlipFromSelections([sel], {
                id: 'SLIP-E1',
                stake: 10.0,
                actualOdds: 2.00,
                couponClass: 'medium'
            });

            // 1. Plana ekle
            const addRes = PE.addSlipToPlan(state, slip);
            const balAfterAdd = addRes.state.plan.availableBalance;

            // 2. Kupon kazansın
            const lookupWin = PE.buildResultsLookup([{ match_id: 'MATCH-X', score: '2-0', total: 2 }]);
            const winRes = PE.settleAllSlips(addRes.state, lookupWin);
            const balAfterWin = winRes.state.plan.availableBalance;

            // 3. Kupon kaybetseydi ne olurdu simülasyonu
            const lookupLoss = PE.buildResultsLookup([{ match_id: 'MATCH-X', score: '0-0', total: 0 }]);
            const lossRes = PE.settleAllSlips(addRes.state, lookupLoss);
            const balAfterLoss = lossRes.state.plan.availableBalance;

            return {
                balStart: 100,
                balAfterAdd,
                balAfterWin,
                balAfterLoss,
                ledgerEntries: winRes.state.ledger.length
            };
        }""")

        assert bal_res['balAfterAdd'] == 90.0, f"Expected 90 after stake reservation, got {bal_res['balAfterAdd']}"
        assert bal_res['balAfterWin'] == 110.0, f"Expected 110 after win (90 + 20), got {bal_res['balAfterWin']}"
        assert bal_res['balAfterLoss'] == 90.0, f"Expected 90 after loss, got {bal_res['balAfterLoss']}"
        print(f"  ✓ Başlangıç: {bal_res['balStart']}€ -> Plana eklenince: {bal_res['balAfterAdd']}€ -> Kazanınca: {bal_res['balAfterWin']}€ -> Kaybedince: {bal_res['balAfterLoss']}€")

        # ----------------------------------------------------------------------
        # TEST 7: Senaryo F — İdempotent Settlement (Mükerrer Ödeme Önleme)
        # ----------------------------------------------------------------------
        print("\n--- TEST 7: Senaryo F — İdempotent Settlement ---")
        # Aynı sonuç dosyası tekrar işlendiğinde kupon ikinci kez ödeme almamalı
        idem_res = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            let state = PE.createInitialState(
                { currency: 'EUR', riskProfile: 'cautious' },
                { startingBank: 100, targetBank: 1000, durationDays: 30 }
            );

            const sel = { matchId: 'MATCH-F', market: 'over_0_5', line: '0.5', probability: 0.96 };
            const slip = PE.createSlipFromSelections([sel], {
                id: 'SLIP-IDEM',
                stake: 10.0,
                actualOdds: 1.50,
                couponClass: 'minimum'
            });

            const addRes = PE.addSlipToPlan(state, slip);
            const lookup = PE.buildResultsLookup([{ match_id: 'MATCH-F', score: '1-0', total: 1 }]);

            // 1. İlk settlement
            const run1 = PE.settleAllSlips(addRes.state, lookup);
            const bal1 = run1.state.plan.availableBalance;
            const ledgerCount1 = run1.state.ledger.length;

            // 2. İkinci settlement (aynı sonuçlar tekrar geldi)
            const run2 = PE.settleAllSlips(run1.state, lookup);
            const bal2 = run2.state.plan.availableBalance;
            const ledgerCount2 = run2.state.ledger.length;
            const settledCount2 = run2.settledCount;

            return {
                bal1,
                bal2,
                ledgerCount1,
                ledgerCount2,
                settledCount2
            };
        }""")

        assert idem_res['bal1'] == 105.0 # 90 + 15 = 105
        assert idem_res['bal2'] == 105.0 # İkinci çalıştırmada bakiye ASLA 120 olmamalı!
        assert idem_res['ledgerCount1'] == idem_res['ledgerCount2'] # Yeni ledger hareketi eklenmemeli
        assert idem_res['settledCount2'] == 0 # 0 kupon etkilendi
        print(f"  ✓ İlk çalıştırma bakiyesi: {idem_res['bal1']}€, İkinci çalıştırma bakiyesi: {idem_res['bal2']}€ (Mükerrer ödeme engellendi).")

        # ----------------------------------------------------------------------
        # TEST 8: Monte Carlo Simülasyonu (5.000 İterasyon)
        # ----------------------------------------------------------------------
        print("\n--- TEST 8: Monte Carlo Simülasyonu ---")
        sim_res = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            const plan = { startingBank: 50, targetBank: 500, durationDays: 30 };
            return PE.runPlanSimulation(plan, 'cautious', null, { seed: 42, iterations: 5000 });
        }""")

        assert sim_res['iterations'] == 5000
        assert sim_res['medianBank'] > 0
        assert sim_res['p10'] <= sim_res['medianBank'] <= sim_res['p90']
        assert 0 <= sim_res['targetHitProb'] <= 1.0
        assert 0 <= sim_res['halfBankLossProb'] <= 1.0
        print(f"  ✓ 5.000 İterasyon: Medyan={sim_res['medianBank']}€, P10={sim_res['p10']}€, P90={sim_res['p90']}€, Hedef Olasılığı=%{sim_res['targetHitPct']}, Yarı Kasa Riski=%{sim_res['halfBankLossPct']}")

        # ----------------------------------------------------------------------
        # TEST 9: Adaptif Plan Önerileri (Süreyi Uzat, Hedefi Ayarla, Riski Değiştir)
        # ----------------------------------------------------------------------
        print("\n--- TEST 9: Adaptif Plan Önerileri ---")
        adapt_res = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            let state = PE.createInitialState(
                { currency: 'EUR', riskProfile: 'cautious' },
                { startingBank: 50, targetBank: 500, durationDays: 30 }
            );

            // 5 adet sonuçlanmış kupon ekleyip kasanın geride kalmasını simüle edelim
            for (let i = 0; i < 5; i++) {
                state.slips.push({
                    id: `slip-lost-${i}`,
                    status: 'lost',
                    couponClass: 'minimum',
                    stake: 4,
                    oddsUsed: 1.25,
                    bankAfter: 30
                });
            }
            state.plan.availableBalance = 30; // 50'den 30'a geriledi

            const adapt = PE.buildAdaptiveOptions(state.plan, state, null, { force: true });
            return {
                show: adapt.showAdaptive,
                optionsCount: adapt.options.length,
                options: adapt.options.map(o => ({ id: o.id, title: o.title, tag: o.tag }))
            };
        }""")

        assert adapt_res['show'] is True
        assert adapt_res['optionsCount'] == 3
        ids = [o['id'] for o in adapt_res['options']]
        assert 'extend_duration' in ids
        assert 'adjust_target' in ids
        assert 'change_risk' in ids
        print("  ✓ Adaptif 3 alternatif (Süreyi Uzat, Hedefi Ayarla, Riski Değiştir) başarıyla üretildi.")

        # ----------------------------------------------------------------------
        # TEST 10: DOM & Arayüz Doğrulaması (5 Sekme, Banner, Kurulum Kartı)
        # ----------------------------------------------------------------------
        print("\n--- TEST 10: DOM & Arayüz Doğrulaması ---")
        # Sekmeler
        tabs = page.query_selector_all(".tabs .tab")
        tab_names = [t.inner_text() for t in tabs]
        print(f"  Bulunan Sekmeler: {tab_names}")
        assert "Tahminler" in tab_names
        assert "Tahmin vs Gerçekleşen" in tab_names
        assert "Kupon Önerileri" in tab_names
        assert "Kuponlarım" in tab_names
        assert "Kasa Planım" in tab_names

        # Global yasal sorumluluk reddi
        banner_text = page.inner_text(".global-paper-banner")
        print(f"  Yasal Uyarı Bannerı: {banner_text[:80]}...")
        assert "bahis kabul etmez" in banner_text

        # Kasa Planım ekranını test et
        page.click("#tab-plan")
        time.sleep(0.5)
        setup_title = page.inner_text("#pane-plan h2")
        print(f"  Kasa Planım Başlığı: {setup_title}")
        assert "Sanal Kasa Planı" in setup_title

        # Setup formundaki canlı önizleme grafiğini test et
        setup_chart = page.query_selector("#setupChartSvgContainer svg")
        assert setup_chart is not None, "Kurulum formu hedef grafiği önizlemesi bulunamadı"
        print("  ✓ Kasa planı oluşturma formunda canlı hedef grafiği önizlemesi doğrulandı.")

        # Plan formu doldurup oluşturma testi
        page.fill("#setupStartBank", "100")
        page.fill("#setupTargetBank", "1000")
        page.click("#btnCreatePlan")
        time.sleep(0.5)

        # Plan dashboard geldi mi?
        dash_title = page.inner_text("#pane-plan h2")
        print(f"  Plan Dashboard Başlığı: {dash_title}")
        assert "Sanal Kasa Planım" in dash_title
        plan_tiles = page.query_selector_all("#pane-plan .plan-tiles .tile")
        assert len(plan_tiles) == 4
        assert "100,00 €" in plan_tiles[0].inner_text()

        # Plan panosundaki interaktif hedef ve risk modelleri grafiğini test et
        assert page.query_selector("#planChartCard") is not None, "#planChartCard bulunamadı"
        assert page.query_selector("#planTrajectorySvg") is not None, "#planTrajectorySvg bulunamadı"
        chart_chips = page.query_selector_all("#chartViewChips .cchip")
        assert len(chart_chips) == 4, f"Beklenen 4 grafik çipi, bulunan {len(chart_chips)}"
        model_summaries = page.query_selector_all(".chart-models-summary .cms-card")
        assert len(model_summaries) == 3, f"Beklenen 3 model özeti, bulunan {len(model_summaries)}"

        # Yüksek risk profil kırmızı renk ve çip geçiş testi
        page.click("#chartViewChips button[data-view='high']")
        time.sleep(0.3)
        high_chip_class = page.get_attribute("#chartViewChips button[data-view='high']", "class")
        assert "active" in high_chip_class
        # SVG içinde kırmızı (#ef4444) çizgi kontrolü
        high_stroke = page.evaluate("""() => {
            const svg = document.querySelector('#planTrajectorySvg');
            return svg ? svg.innerHTML.includes('#ef4444') : false;
        }""")
        assert high_stroke is True, "Yüksek risk profil SVG içinde #ef4444 kırmızı rengi ile çizilmemiş!"
        print("  ✓ Yüksek risk profili kırmızı (#ef4444) renk ile grafikte doğrulandı.")

        # Günlük Büyüme Modeli Takip Çizelgesi kontrolü
        assert page.query_selector("#excelModelCard") is not None, "#excelModelCard bulunamadı"
        excel_rows = page.query_selector_all("#excelModelCard .excel-table tbody tr")
        assert len(excel_rows) == 31, f"Beklenen 31 satır (0..30 gün), bulunan {len(excel_rows)}"
        print("  ✓ Kasa planı panosunda interaktif hedef grafiği (4 çip, 3 kart) ve günlük takip çizelgesi doğrulandı.")

        # Kupon Önerileri sekmesine geç
        page.click("#tab-rec")
        time.sleep(0.5)
        rec_title = page.inner_text("#pane-rec h2")
        print(f"  Kupon Önerileri Başlığı: {rec_title}")
        assert "Kişiselleştirilmiş Kupon Önerileri" in rec_title
        rec_cards = page.query_selector_all("#pane-rec .rec-card")
        print(f"  Öneri Kartı Sayısı: {len(rec_cards)}")
        assert len(rec_cards) == 3

        # Kuponlarım sekmesine geç
        page.click("#tab-cpn")
        time.sleep(0.5)
        csub_tabs = page.query_selector_all(".subtabs-bar .subtab")
        csub_names = [b.inner_text() for b in csub_tabs]
        print(f"  Kuponlarım Alt Sekmeleri: {csub_names}")
        assert any("Bekleyenler" in s for s in csub_names)
        assert any("Sonuçlananlar" in s for s in csub_names)
        assert any("Taslaklar" in s for s in csub_names)
        assert any("12 Eylül Canlı Model Takibi" in s for s in csub_names)

        # 12 Eylül model takibi alt sekmesine tıkla
        page.click("#csub-model12")
        time.sleep(0.5)
        cpn_sum = page.inner_text("#cpnSummary")
        assert "12 Eylül 2026" in cpn_sum
        print("  ✓ 12 Eylül model izleme alt görünümü korundu.")

        # JSON Dışa Aktarma / İçe Aktarma testi
        json_roundtrip = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            const state = window.BETAVUS_PAPER_UI.getState();
            const exported = PE.exportPaperState(state);
            const val = PE.validateImportedJSON(exported);
            return {
                valid: val.valid,
                schemaVersion: val.data.schemaVersion,
                hasPlan: !!val.data.plan
            };
        }""")
        assert json_roundtrip['valid'] is True
        assert json_roundtrip['schemaVersion'] == 1
        assert json_roundtrip['hasPlan'] is True
        print("  ✓ JSON dışa ve içe aktarma şema doğrulaması başarılı.")

        # ----------------------------------------------------------------------
        # TEST 11: Gerçekçi Piyasa Oranları & 30 Günlük Geçmiş Simülasyon Motoru
        # ----------------------------------------------------------------------
        print("\n--- TEST 11: Realistic Market Odds & 30-Day Simulation Engine ---")
        sim_res = page.evaluate("""async () => {
            const PE = window.BETAVUS_PAPER;
            let resData = (window.RESULTS && window.RESULTS.matches) ? window.RESULTS.matches : [];
            if (!resData.length) {
                const fetched = await fetch('data/results.json').then(r => r.json()).catch(() => null);
                if (fetched && fetched.matches) {
                    resData = fetched.matches;
                    window.RESULTS = fetched;
                }
            }
            
            // Oran kalibrasyonu testi
            const dummyMatch = { p_over_0_5: 0.96, p_over_1_5: 0.88, p_over_2_5: 0.76 };
            const o05 = PE.calculateEstimatedLegOdds(dummyMatch, '0.5 Üst');
            const o15 = PE.calculateEstimatedLegOdds(dummyMatch, '1.5 Üst');
            const o25 = PE.calculateEstimatedLegOdds(dummyMatch, '2.5 Üst');

            const sim = PE.generate30DayHistoricalSimulation(resData, { startingBank: 50.0 });
            return {
                odds: { o05, o15, o25 },
                simWindow: sim.simulationWindow,
                startingBank: sim.startingBank,
                profiles: {
                    minimum: {
                        couponsCount: sim.profiles.minimum.coupons.length,
                        ledgerCount: sim.profiles.minimum.ledger.length,
                        firstStake: sim.profiles.minimum.coupons[0] ? sim.profiles.minimum.coupons[0].stake : 0,
                        firstOdds: sim.profiles.minimum.coupons[0] ? sim.profiles.minimum.coupons[0].totalOdds : 0,
                        finalBank: sim.profiles.minimum.stats.finalBank,
                        reservePct: sim.profiles.minimum.reservePct,
                        stakePct: sim.profiles.minimum.stakePct
                    },
                    medium: {
                        couponsCount: sim.profiles.medium.coupons.length,
                        ledgerCount: sim.profiles.medium.ledger.length,
                        firstStake: sim.profiles.medium.coupons[0] ? sim.profiles.medium.coupons[0].stake : 0,
                        firstOdds: sim.profiles.medium.coupons[0] ? sim.profiles.medium.coupons[0].totalOdds : 0,
                        finalBank: sim.profiles.medium.stats.finalBank,
                        reservePct: sim.profiles.medium.reservePct,
                        stakePct: sim.profiles.medium.stakePct
                    },
                    high: {
                        couponsCount: sim.profiles.high.coupons.length,
                        ledgerCount: sim.profiles.high.ledger.length,
                        firstStake: sim.profiles.high.coupons[0] ? sim.profiles.high.coupons[0].stake : 0,
                        firstOdds: sim.profiles.high.coupons[0] ? sim.profiles.high.coupons[0].totalOdds : 0,
                        finalBank: sim.profiles.high.stats.finalBank,
                        reservePct: sim.profiles.high.reservePct,
                        stakePct: sim.profiles.high.stakePct
                    }
                }
            };
        }""")

        odds = sim_res['odds']
        print(f"  Piyasa Kalibre Oranlar -> 0.5 Üst: {odds['o05']}, 1.5 Üst: {odds['o15']}, 2.5 Üst: {odds['o25']}")
        assert 1.03 <= odds['o05'] <= 1.07, f"0.5 Üst oranı gerçekçi aralıkta değil: {odds['o05']}"
        assert 1.16 <= odds['o15'] <= 1.25, f"1.5 Üst oranı gerçekçi aralıkta değil: {odds['o15']}"
        assert 1.42 <= odds['o25'] <= 1.68, f"2.5 Üst oranı gerçekçi aralıkta değil: {odds['o25']}"
        print("  ✓ Gerçekçi internet bahis oranları kalibrasyonu doğrulandı.")

        win = sim_res['simWindow']
        print(f"  Simülasyon Penceresi: {win['startDate']} - {win['endDate']} ({win['totalDays']} gün, {win['matchesInWindow']} maç)")
        assert win['totalDays'] == 31
        assert win['matchesInWindow'] >= 400
        assert sim_res['startingBank'] == 50.0

        for p_name in ['minimum', 'medium', 'high']:
            p_data = sim_res['profiles'][p_name]
            assert p_data['couponsCount'] == 31, f"{p_name} için 31 kupon bekleniyordu"
            assert p_data['ledgerCount'] == 31, f"{p_name} için 31 muhasebe satırı bekleniyordu"
            print(f"  ✓ {p_name.upper()} profili: 31 kupon, 31 çizelge satırı, Final Kasa: {p_data['finalBank']} EUR")

        # ----------------------------------------------------------------------
        # TEST 12: 30 Günlük Geçmiş Simülasyon UI & Alt Görünümler
        # ----------------------------------------------------------------------
        print("\n--- TEST 12: 30-Day Historical Simulation UI & Subviews ---")
        # Plan sekmesine geri dön
        page.click("#tab-plan")
        time.sleep(0.5)

        # Plan üst navigasyon butonlarını doğrula
        btn_active = page.query_selector("#btnPmnActive")
        btn_sim30 = page.query_selector("#btnPmnSim30")
        assert btn_active is not None, "#btnPmnActive butonu bulunamadı"
        assert btn_sim30 is not None, "#btnPmnSim30 butonu bulunamadı"

        # 30 Günlük Simülasyon butonuna tıkla
        page.click("#btnPmnSim30")
        page.wait_for_selector(".sim30-kpi-card", timeout=8000)

        assert "active" in page.get_attribute("#btnPmnSim30", "class")
        sim_kpi_cards = page.query_selector_all(".sim30-kpi-card")
        print(f"  Simülasyon KPI Kartları Sayısı: {len(sim_kpi_cards)}")
        assert len(sim_kpi_cards) == 3

        # SVG grafiğini kontrol et
        assert page.query_selector("#sim30Svg") is not None, "#sim30Svg grafiği render edilmedi"

        # Kuponlarım simülasyon kartlarını doğrula
        sim_cpn_cards = page.query_selector_all(".sim30-cpn-card")
        print(f"  Simülasyon Kupon Kartı Sayısı: {len(sim_cpn_cards)}")
        assert len(sim_cpn_cards) == 31

        # Kasa Muhasebe Çizelgesi sekmesine tıkla
        page.click("#btnSim30TabLedger")
        page.wait_for_selector("#sim30SubtabContent .sim30-table tbody tr", timeout=5000)
        ledger_rows = page.query_selector_all("#sim30SubtabContent .sim30-table tbody tr")
        print(f"  Simülasyon Muhasebe Çizelgesi Satır Sayısı: {len(ledger_rows)}")
        assert len(ledger_rows) == 31

        # Tekrar Aktif Plana dön
        page.click("#btnPmnActive")
        page.wait_for_selector("#planChartCard", timeout=5000)
        assert page.query_selector("#planChartCard") is not None
        print("  ✓ 30 Günlük Geçmiş Kasa ve Kuponlarım Simülasyon panosu, KPI kartları, kupon listesi ve muhasebe tablosu başarıyla doğrulandı.")

        browser.close()
        print("\n========================================================")
        print(">>> TÜM PAPER-BETTING VE ŞARTNAME TESTLERİ BAŞARIYLA GEÇTİ! <<<")
        print("========================================================")

if __name__ == "__main__":
    main()
