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
            localStorage.setItem('betavus.tab', 'sim-kasa');
            localStorage.setItem('betavus.lang', 'tr');
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
        # Paper_Betting_Kasa_Simulasyonu v01 "Risk Faktörleri" sayfası
        # Minimum Risk: Kasa Rezerv: %75, Günlük Büyüme Oranı: %10 (1.10x, Aktif Pay: %25)
        assert cfg_res['minimum']['reservePct'] == 0.75
        assert cfg_res['minimum']['dailyGrowthRate'] == 0.10
        assert cfg_res['minimum']['dailyFactor'] == 1.10
        assert cfg_res['minimum']['stakePct'] == 0.25

        # Orta Risk (Medium): Kasa Rezerv: %50, Günlük Büyüme Oranı: %15 (1.15x, Aktif Pay: %50)
        assert cfg_res['medium']['reservePct'] == 0.50
        assert cfg_res['medium']['dailyGrowthRate'] == 0.15
        assert cfg_res['medium']['dailyFactor'] == 1.15
        assert cfg_res['medium']['stakePct'] == 0.50

        # Yüksek Risk (High): Kasa Rezerv: %50, Günlük Büyüme Oranı: %25 (1.25x, Aktif Pay: %50)
        assert cfg_res['high']['reservePct'] == 0.50
        assert cfg_res['high']['dailyGrowthRate'] == 0.25
        assert cfg_res['high']['dailyFactor'] == 1.25
        assert cfg_res['high']['stakePct'] == 0.50

        # Geriye dönük uyumluluk takma adları
        assert cfg_res['cautious'] is not None and cfg_res['balanced'] is not None and cfg_res['aggressive'] is not None
        print("  ✓ Kasa risk profilleri (Minimum %75 Rezerv / %10 Büyüme, Medium %50 Rezerv / %15 Büyüme, High %50 Rezerv / %25 Büyüme) doğrulandı.")

        # Günlük Büyüme Modeli Matematik Doğrulaması (medium risk: %50 rezerv, %15 büyüme)
        excel_math = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            const m = PE.calculateExcelGrowthModel({ startingBank: 50, durationDays: 30 }, 'medium');
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
        # Excel: Hedefe Ulaşma Günü = ROUNDUP(LN(500/50)/LN(1.10)) = 25 (Minimum risk %10)
        assert plan_res['state']['plan']['durationDays'] == 25
        assert plan_res['day0'] == 50.0
        # Teorik Hedef Kasa = 50 * 1.10^gün
        assert plan_res['day15'] == 208.86
        assert plan_res['day30'] == 872.47
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
                targetLast: traj.targetPoints[traj.targetPoints.length - 1].targetBank,
                hasMinimum: !!traj.trajectories.minimum,
                hasMedium: !!traj.trajectories.medium,
                hasHigh: !!traj.trajectories.high,
                hasMulti: !!traj.trajectories.multi,
                minPointsLen: traj.trajectories.minimum.dayPoints.length,
                min0: traj.trajectories.minimum.dayPoints[0].median,
                minHitPct: traj.trajectories.minimum.targetHitPct
            };
        }""")
        assert traj_test['durationDays'] == 25  # Hedefe ulaşma günü (Minimum %10)
        assert traj_test['targetPointsLen'] == 26  # Gün 0'dan 25'e 26 nokta
        assert traj_test['target0'] == 50.0
        assert traj_test['targetLast'] == 541.74  # 50 * 1.10^25 >= 500 hedef
        assert traj_test['hasMinimum'] and traj_test['hasMedium'] and traj_test['hasHigh'] and traj_test['hasMulti']
        assert traj_test['minPointsLen'] == 26
        assert traj_test['min0'] == 50.0
        print("  ✓ Hedeflenen sürede kasa ulaşma trajektorisi ve 4 model hesaplama projeksiyonu doğrulandı.")
        print(f"  ✓ Günlük gerekli oran: %{plan_res['dailyRate']}, Gün 0: {plan_res['day0']}€, Gün 15: {plan_res['day15']}€, Gün 30: {plan_res['day30']}€")
        print("  ✓ Teorik hedef kasa yolu ve durum sınıflandırması doğrulandı.")

        # Paper_Betting_Kasa_Simulasyonu v01: Otomatik parametreler + GERÇEK/HEDEF tablosu
        kasa_res = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            const params = ['minimum', 'medium', 'high'].map(r =>
                PE.calculateKasaParams({ startingBank: 50, targetBank: 1000, riskProfile: r }));

            const state = PE.createInitialState({ riskProfile: 'medium' },
                { startingBank: 50, targetBank: 1000, riskProfile: 'medium' });
            const d = new Date(); d.setDate(d.getDate() - 2);
            state.plan.startDate = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
            PE.syncActivePlan(state);
            PE.setPlanDailyBank(state, 1, 68.66);
            PE.setPlanDailyBank(state, 2, 60);
            const sim = PE.buildKasaSimulation(state.plan, state);
            PE.ensurePlansArray(state);

            // Kupon eklenince plans[] kopyası da güncellenmeli (render sonrası bakiye kaybolmasın)
            const slip = PE.createSlipFromSelections(
                [{ matchId: 'K1', market: 'over_1_5', line: '1.5', probability: 0.9 }],
                { id: 'SLIP-K1', stake: 10, actualOdds: 1.5, couponClass: 'medium' });
            const added = PE.addSlipToPlan(state, slip).state;
            PE.ensurePlansArray(added);
            const won = PE.settleAllSlips(added, PE.buildResultsLookup([{ match_id: 'K1', score: '2-0', total: 2 }])).state;
            PE.ensurePlansArray(won);
            const simAfter = PE.buildKasaSimulation(won.plan, won);
            return { params, sim, planDays: state.plan.durationDays, slipPlanId: added.slips[0].planId,
                     planId: added.plan.id, balAfterAdd: added.plan.availableBalance, balAfterWin: won.plan.availableBalance,
                     todayAfter: simAfter.rows[simAfter.todayDay - 1] };
        }""")
        mn, md, hi = kasa_res['params']
        assert (mn['dailyGrowthRate'], mn['reservePct'], mn['daysToTarget'], mn['theoreticalAtTargetDay']) == (0.10, 0.75, 32, 1055.69)
        assert (md['dailyGrowthRate'], md['reservePct'], md['daysToTarget'], md['theoreticalAtTargetDay']) == (0.15, 0.50, 22, 1082.24)
        assert (hi['dailyGrowthRate'], hi['reservePct'], hi['daysToTarget'], hi['theoreticalAtTargetDay']) == (0.25, 0.50, 14, 1136.87)
        assert kasa_res['planDays'] == 22
        rows = kasa_res['sim']['rows']
        assert kasa_res['sim']['todayDay'] == 3 and len(rows) == 22  # kasa hedef gününde (Medium: 22) biter
        # Gün 1: Gerçek 68,66 → değişim 18,66 (başlangıca göre), büyüme %37,32; hedef 57,50, kazanç 7,50
        assert rows[0]['actualBank'] == 68.66 and rows[0]['isManual'] is True
        assert rows[0]['dailyChange'] == 18.66 and rows[0]['dailyGrowthPct'] == 37.32
        assert rows[0]['totalGrowthPct'] == 37.32  # 68,66 / 50 − 1
        assert rows[0]['targetBank'] == 57.5 and rows[0]['targetDailyGain'] == 7.5
        assert rows[0]['belowTarget'] is False
        # Gün 2: 60 < 66,13 hedef → kırmızı (Excel koşullu biçim)
        assert rows[1]['targetBank'] == 66.13 and rows[1]['belowTarget'] is True
        assert rows[1]['dailyChange'] == -8.66
        # Gün 3 (bugün): elle giriş ve sonuçlanan kupon yok → Excel'deki gibi boş
        assert rows[2]['actualBank'] is None and rows[2]['dailyChange'] is None
        assert rows[3]['actualBank'] is None
        assert rows[21]['targetReachPct'] == 100.0
        assert kasa_res['slipPlanId'] == kasa_res['planId']
        assert kasa_res['balAfterAdd'] == 40.0 and kasa_res['balAfterWin'] == 55.0
        assert kasa_res['todayAfter']['actualBank'] == 55.0  # 50 + 10 * (1.5 - 1)
        # Excel dosyasındaki örnek kasa: 1-12. gün gerçek kasa (gelecek günler de elle girilebilir)
        ex_res = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            const st = PE.createInitialState({ riskProfile: 'medium' }, PE.KASA_V01_EXAMPLE);
            const sim = PE.buildKasaSimulation(st.plan, st);
            PE.updatePlanInputs(st, { riskProfile: 'high', targetBank: 2000 });
            return { real: sim.rows.slice(0, 13).map(r => r.actualBank), below: sim.rows.slice(0, 12).map(r => r.belowTarget),
                     d12growth: sim.rows[11].dailyGrowthPct, afterDays: st.plan.durationDays, afterPlans: st.plans[0].riskProfile };
        }""")
        assert ex_res['real'] == [68.66, 75, 86, 103.72, 128, 255, 275, 278, 300, 320, 320, 300, None]
        # Excel koşullu biçim: 1-12. günlerin hepsi teorik hedefin üstünde (örn. 12. gün 300 > 267,51) → kırmızı yok
        assert ex_res['below'] == [False] * 12
        assert ex_res['d12growth'] == -6.25  # 300 / 320 - 1
        assert ex_res['afterDays'] == 17 and ex_res['afterPlans'] == 'high'  # ROUNDUP(LN(40)/LN(1.25)) = 17
        print("  ✓ Kasa Simülasyonu v01: hedef günü (32/22/14), teorik kasa, GERÇEK/HEDEF tablosu, Excel örnek verisi ve kırmızı işaretleme doğrulandı.")

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
        # Süre Excel modelinde risk faktöründen türetildiği için "Süreyi Uzat" alternatifi yoktur
        assert adapt_res['optionsCount'] == 2
        ids = [o['id'] for o in adapt_res['options']]
        assert 'adjust_target' in ids
        assert 'change_risk' in ids
        print("  ✓ Adaptif 2 alternatif (Hedefi Ayarla, Riski Değiştir) başarıyla üretildi.")

        # ----------------------------------------------------------------------
        # TEST 10: Menü mimarisi — görünen 4 sekme (Bülten · İstatistikler · Sanal Kasa · FAQ), eski sekmeler gizli
        # ----------------------------------------------------------------------
        print("\n--- TEST 10: Menü Mimarisi (4 görünen sekme, eski sekmeler gizli) ---")
        tab_state = page.evaluate("""() => [...document.querySelectorAll('.top .tabs .tab')].map(t => ({ id: t.id, hidden: t.hidden }))""")
        visible = [t['id'] for t in tab_state if not t['hidden']]
        hidden = [t['id'] for t in tab_state if t['hidden']]
        print(f"  Görünen: {visible} · Gizli: {hidden}")
        assert visible == ['tab-pred', 'tab-stats', 'tab-plan', 'tab-faq'], f"Görünen sekmeler beklenenden farklı: {visible}"
        for tid in ['tab-sim-kasa', 'tab-sim-kupon', 'tab-rec', 'tab-bt', 'tab-res', 'tab-cpn', 'tab-cifte']:
            assert tid in hidden, f"{tid} gizli olmalı"
        print("  ✓ Menüde 4 sekme görünüyor; eski sekmelerin kodu korunuyor ama menüde gizli.")

        # ----------------------------------------------------------------------
        # TEST 11: 30 Günlük Simülasyon Motoru & ~%4-5 Yanılma Oranı Kalibrasyonu
        # ----------------------------------------------------------------------
        print("\n--- TEST 11: 30-Day Historical Simulation Engine & %4-5 Error Rate ---")
        sim_eval = page.evaluate("""async () => {
            const PE = window.BETAVUS_PAPER;
            let resData = (window.RESULTS && window.RESULTS.matches) ? window.RESULTS.matches : [];
            if (!resData.length) {
                const fetched = await fetch('data/results.json').then(r => r.json()).catch(() => null);
                if (fetched && fetched.matches) {
                    resData = fetched.matches;
                    window.RESULTS = fetched;
                }
            }

            const sim = PE.generate30DayHistoricalSimulation(resData, { startingBank: 50.0 });
            const min = sim.profiles.minimum;
            return {
                startingBank: sim.startingBank,
                totalDays: sim.simulationWindow.totalDays,
                matchesInWindow: sim.simulationWindow.matchesInWindow,
                minStats: min.stats,
                minCouponsCount: min.coupons.length,
                minLedgerCount: min.ledger.length,
                wonCoupons: min.stats.wonCoupons,
                lostCoupons: min.stats.lostCoupons,
                totalLegs: min.stats.totalLegs,
                wonLegs: min.stats.wonLegs,
                lostLegs: min.stats.lostLegs,
                legSuccessRatePct: min.stats.legSuccessRatePct,
                legErrorRatePct: min.stats.legErrorRatePct,
                finalBank: min.stats.finalBank,
                reserveBank: min.stats.reserveBank,
                totalNetProfit: min.stats.totalNetProfit,
                totalRoiPct: min.stats.totalRoiPct
            };
        }""")

        assert sim_eval['startingBank'] == 50.0
        assert sim_eval['totalDays'] == 31
        assert sim_eval['matchesInWindow'] >= 400
        assert sim_eval['minCouponsCount'] == 31
        assert sim_eval['minLedgerCount'] == 31
        assert sim_eval['totalLegs'] == 155, f"155 bacak (31 gün x 5 bacak) bekleniyordu, bulunan {sim_eval['totalLegs']}"
        assert sim_eval['wonLegs'] == 148, f"148 kazanan maç bekleniyordu, bulunan {sim_eval['wonLegs']}"
        assert sim_eval['lostLegs'] == 7, f"7 kaybeden maç bekleniyordu, bulunan {sim_eval['lostLegs']}"
        assert 4.0 <= sim_eval['legErrorRatePct'] <= 5.0, f"Maç yanılma oranı %4-5 aralığında değil: %{sim_eval['legErrorRatePct']}"
        assert 95.0 <= sim_eval['legSuccessRatePct'] <= 96.0
        assert sim_eval['wonCoupons'] == 28
        assert sim_eval['lostCoupons'] == 3
        assert sim_eval['finalBank'] > 150.0, f"Kasa büyümesi beklentiyi karşılamadı: {sim_eval['finalBank']}€"
        assert sim_eval['reserveBank'] > 75.0
        print(f"  ✓ Minimum Risk Kalibrasyonu: 155 Maçta 148 İsabet (%{sim_eval['legSuccessRatePct']}), 7 Yanılma (%{sim_eval['legErrorRatePct']})")
        print(f"  ✓ Kasa Büyümesi: 50,00 € -> {sim_eval['finalBank']} € (+%{sim_eval['totalRoiPct']} ROI, %50 Rezerv Korumalı: {sim_eval['reserveBank']} €)")

        # ----------------------------------------------------------------------
        # TEST 12: Güven Payı motoru — güvenli sınır, kilitleme tutarı, geri alma, mola uyarısı, hedef
        # ----------------------------------------------------------------------
        print("\n--- TEST 12: Güven Payı Motoru ---")
        gp = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            const mk = (banks, prof = 'medium', S = 50, T = 10000) => {
                const st = PE.createInitialState({ currency: 'EUR', riskProfile: prof }, { startingBank: S, targetBank: T, riskProfile: prof, dailyBanks: banks });
                const d = new Date(); d.setDate(d.getDate() - (Object.keys(banks).length - 1));
                st.plan.startDate = d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
                st.plans = [st.plan];
                return st;
            };
            const coach = st => PE.getEICoach(st.plan, PE.buildKasaSimulation(st.plan, st));
            const out = {};
            // 50 € başlangıç, 4. gün kasa 105 €: Medium kupon = %50 → 52,50 € > sınır 50 €
            const st = mk({ 1: 60, 2: 75, 3: 90, 4: 105 });
            const c1 = coach(st);
            out.before = { level: c1.level, stake: c1.stake, limit: c1.limit, lock: c1.lock };
            PE.applyCashout(st, { day: c1.day, amount: c1.lock.amount, reason: 'lock' });
            const sim2 = PE.buildKasaSimulation(st.plan, st);
            const c2 = coach(st);
            out.after = { level: c2.level, stake: c2.stake, limit: c2.limit, secured: sim2.secured, working: sim2.currentBank,
                          total: sim2.currentTotal, lockToday: c2.lock, day5Target: sim2.rows[4].targetBank };
            // Kasaya geri alma (negatif tutar) ve son işlemi geri alma
            PE.applyCashout(st, { day: 4, amount: -20, reason: 'return' });
            const sim3 = PE.buildKasaSimulation(st.plan, st);
            out.afterReturn = { secured: sim3.secured, working: sim3.currentBank };
            out.overReturn = PE.applyCashout(st, { day: 4, amount: -1000 }) === null;
            PE.undoLastCashout(st);
            out.afterUndo = PE.buildKasaSimulation(st.plan, st).secured;
            // Minimum: kupon %25 → ilk kilit kasa 4 katını geçince
            out.min200 = coach(mk({ 1: 200 }, 'minimum')).level;
            out.min210 = coach(mk({ 1: 210 }, 'minimum')).level;
            // Zirveden %40 düşüş → mola uyarısı; hedefe ulaşıldı
            out.drop = coach(mk({ 1: 100, 2: 60 })).drop;
            out.reached = coach(mk({ 1: 10000 })).reached;
            // Yol haritası: 50 → 10.000 € Medium için durak sayısı
            out.stops = coach(mk({ 1: 55 })).roadmap.stops.length;
            return out;
        }""")
        print(f"  Önce: {gp['before']}")
        assert gp['before']['level'] == 'red' and gp['before']['stake'] == 52.5 and gp['before']['limit'] == 50
        assert gp['before']['lock']['amount'] == 53 and gp['before']['lock']['stakeAfter'] == 26
        print("  ✓ Sınır aşımı: kupon 52,50 € > güvenli sınır 50 € (başlangıç kasası) → 53 € kilitle, kupon 26 €'ya iner.")
        a = gp['after']
        assert a['level'] == 'green' and a['limit'] == 103 and a['secured'] == 53 and a['working'] == 52 and a['total'] == 105
        assert a['lockToday'] is None, "Aynı gün ikinci öneri çıkmamalı"
        assert a['day5Target'] == 112.8, f"5. gün hedefi kilitli + oyundaki büyümesi olmalı (53 + 52×1,15), bulunan {a['day5Target']}"
        print("  ✓ Kilit sonrası: sınır 103 € (başlangıç + kilitli), toplam 105 € korunur, hedef kilitli parayı sayar, aynı gün yeni öneri yok.")
        assert gp['afterReturn'] == {'secured': 33, 'working': 72} and gp['overReturn'] and gp['afterUndo'] == 53
        print("  ✓ Kasaya geri alma (20 €), kilitli paradan fazlasını geri almayı engelleme ve son işlemi geri alma doğrulandı.")
        assert gp['min200'] == 'yellow' and gp['min210'] == 'red'
        print("  ✓ Minimum risk: kasa başlangıcın 4 katını geçince (210 €) kilit önerisi.")
        assert gp['drop'] is not None and gp['drop']['pct'] == 40 and gp['reached'] is True
        assert 4 <= gp['stops'] <= 7, f"50 → 10.000 € Medium için 4-7 durak bekleniyordu, bulunan {gp['stops']}"
        print(f"  ✓ Zirveden %40 düşüşte mola uyarısı, hedefe ulaşma ve yol haritası ({gp['stops']} durak) doğrulandı.")

        # ----------------------------------------------------------------------
        # TEST 13: Sanal Kasa ekranı (#pane-plan) — Güven Payı kartı, kilitleme, tablo, Tab ile gün geçişi
        # ----------------------------------------------------------------------
        print("\n--- TEST 13: Sanal Kasa Ekranı & Güven Payı Kartı ---")
        page.on("dialog", lambda d: d.accept())
        page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            const d = new Date(); d.setDate(d.getDate() - 3);
            const st = PE.createInitialState({ currency: 'EUR', riskProfile: 'medium' }, { name: 'Test Kasa', startingBank: 50, targetBank: 10000, riskProfile: 'medium', dailyBanks: { 1: 60, 2: 75, 3: 90, 4: 105 } });
            st.plan.startDate = d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
            st.plans = [st.plan];
            localStorage.setItem('betavus.paper_v1', JSON.stringify(st));
        }""")
        page.reload(wait_until="networkidle")
        time.sleep(0.8)
        page.click("#tab-plan")
        time.sleep(0.5)
        assert page.is_visible("#pane-plan") is True
        assert page.query_selector("#pane-plan .kasa-sheet") is not None
        assert page.query_selector("#pane-plan .ks-chart svg") is not None
        assert page.query_selector("#ksComfort") is None, "Güvenli sınır otomatik olmalı; kullanıcıya sorulan alan olmamalı"
        light = page.inner_text("#eiCoachCard .ei-light")
        assert "52,50" in light and "50,00" in light and "aştın" in light, light
        assert "53,00" in page.inner_text("#btnEILock")
        print(f"  ✓ Kart: '{light}'")
        page.click("#btnEILock")
        time.sleep(0.4)
        light2 = page.inner_text("#eiCoachCard .ei-light")
        assert "26,00" in light2 and "103,00" in light2 and "Rahatsın" in light2, light2
        assert "Sonraki kilit" in page.inner_text("#eiCoachCard")
        heads = page.evaluate("""() => [...document.querySelectorAll('#kasaSimCard thead th')].map(t => t.textContent.trim())""")
        assert any(h.startswith('Oyundaki Kasa') for h in heads) and any(h.startswith('Kilitli') for h in heads), heads
        assert not any(h.startswith('Günlük Büyüme') for h in heads), "Günlük Büyüme sütunu kaldırılmış olmalı"
        print("  ✓ Kilitle butonu: kupon 26 €, sınır 103 €; tabloda 'Oyundaki Kasa' ve 'Kilitli' sütunları.")
        # Tab ile sonraki güne geçiş; tablo kartı ekranda yerinde kalır
        page.evaluate("scrollTo(0, document.querySelector('#kasaSimCard').getBoundingClientRect().top + scrollY - 40)")
        top_before = page.evaluate("Math.round(document.querySelector('#kasaSimCard').getBoundingClientRect().top)")
        page.click('.kasa-input[data-day="5"]')
        page.keyboard.type("62")
        page.keyboard.press("Tab")
        time.sleep(0.3)
        focus_day = page.evaluate("document.activeElement && document.activeElement.dataset.day")
        top_after = page.evaluate("Math.round(document.querySelector('#kasaSimCard').getBoundingClientRect().top)")
        saved = page.evaluate("JSON.parse(localStorage.getItem('betavus.paper_v1')).plan.dailyBanks['5']")
        assert focus_day == '6', f"Tab sonrası 6. güne geçilmeliydi, odak: {focus_day}"
        assert saved == 62 and abs(top_after - top_before) <= 1, (saved, top_before, top_after)
        print("  ✓ Değer girip Tab: kayıt yapıldı, imleç 6. güne geçti, tablo ekranda yerinde kaldı.")

        # ----------------------------------------------------------------------
        # TEST 14: Kapatılan kasayı yeniden aç / sil + senkronda geri gelmeme
        # ----------------------------------------------------------------------
        print("\n--- TEST 14: Kapatılan Kasa — Yeniden Aç / Sil ---")
        page.click("#btnCloseKasa")
        time.sleep(0.4)
        assert len(page.query_selector_all(".closed-kasa-table tbody tr")) == 1
        page.click(".ck-reopen")
        time.sleep(0.4)
        assert len(page.query_selector_all(".closed-kasa-table tbody tr")) == 0
        tabs_now = page.evaluate("[...document.querySelectorAll('.bankroll-tab .bt-name')].map(t => t.textContent.trim())")
        assert "Test Kasa" in tabs_now, tabs_now
        assert page.evaluate("JSON.parse(localStorage.getItem('betavus.paper_v1')).plan.cashouts.length") == 1, "Yeniden açılan kasa geçmişini korumalı"
        print("  ✓ Yeniden aç: kasa geçmişiyle (kilitleme dahil) aktif kasalara döndü.")
        page.click("#btnCloseKasa")
        time.sleep(0.4)
        assert len(page.query_selector_all(".closed-kasa-table tbody tr")) == 1, "Yeniden kapatılan kasa listede görünmeli"
        page.click(".ck-del")
        time.sleep(0.4)
        assert page.query_selector(".closed-kasa-table") is None or len(page.query_selector_all(".closed-kasa-table tbody tr")) == 0
        sync = page.evaluate("""() => {
            const PE = window.BETAVUS_PAPER;
            const st = JSON.parse(localStorage.getItem('betavus.paper_v1'));
            const tomb = st.closedTombstones || {};
            // Başka cihazdan gelen eski kopya (silinen kasa hâlâ closedPlans içinde) ilk senkronda birleştirilir
            const id = Object.keys(tomb)[0];
            const remote = JSON.parse(JSON.stringify(st));
            remote.closedTombstones = {};
            remote.closedPlans = [{ id, name: 'Eski kopya', closedAt: new Date(tomb[id] - 60000).toISOString(), plan: { id, name: 'Eski kopya' } }];
            const merged = window.BETAVUS_AUTH ? JSON.parse(window.BETAVUS_AUTH._mergePaper(JSON.stringify(st), JSON.stringify(remote), true)) : null;
            if (merged) PE.ensurePlansArray(merged);
            return { hasTomb: !!id, mergedClosed: merged ? merged.closedPlans.length : null };
        }""")
        assert sync['hasTomb'] is True
        assert sync['mergedClosed'] == 0, f"Silinen kasa senkronda geri gelmemeli: {sync}"
        print("  ✓ Sil: kasa listeden kalktı; başka cihazdan gelen eski kopya senkronda geri gelmiyor.")

        # ----------------------------------------------------------------------
        # TEST 15: Tüm DOM Genelinde Sıfır Argo ("Yatış") Doğrulaması
        # ----------------------------------------------------------------------
        print("\n--- TEST 15: Zero Slang ('Yatış') Global DOM Verification ---")
        page_body_text = page.inner_text("body")
        for bad_word in ['Yatış', 'yatış', 'Yatis', 'yatis']:
            assert bad_word not in page_body_text, f"Sayfa gövdesinde yasaklı kelime bulundu: {bad_word}"
        print("  ✓ Tüm sayfada sıfır 'Yatış' argo kelimesi teyit edildi.")

        # ----------------------------------------------------------------------
        # TEST 16: Üst menü her dilde sığar (yatay taşma yok)
        # ----------------------------------------------------------------------
        print("\n--- TEST 16: Üst Menü Sığdırma (TR / EN / NL) ---")
        for lang, width in [('tr', 1280), ('en', 1280), ('nl', 1280), ('nl', 1600), ('nl', 390)]:
            ctx = browser.new_context(viewport={'width': width, 'height': 800})
            ctx.add_init_script(f"localStorage.setItem('betavus.access', '1f7b720c52ea3f6e8631a8eeaffaa7113fbed540ec0772108d39c52835d9855d'); localStorage.setItem('betavus.lang', '{lang}');")
            pg2 = ctx.new_page()
            pg2.goto(f"http://127.0.0.1:{PORT}/index.html", wait_until="networkidle")
            time.sleep(0.5)
            sw = pg2.evaluate("document.documentElement.scrollWidth")
            mode = pg2.evaluate("document.querySelector('.top .topbar').className")
            assert sw <= width, f"{lang} {width}px: sayfa yatay taşıyor ({sw}px)"
            print(f"  ✓ {lang} {width}px: taşma yok ({mode})")
            ctx.close()

        browser.close()
        print("\n========================================================")
        print(">>> TÜM PAPER-BETTING VE ŞARTNAME TESTLERİ BAŞARIYLA GEÇTİ! <<<")
        print("========================================================")

if __name__ == "__main__":
    main()
