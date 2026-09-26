import time, threading, sys
sys.stdout.reconfigure(encoding='utf-8')
from http.server import HTTPServer, SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright

class H(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

PORT = 8919
server = HTTPServer(('127.0.0.1', PORT), H)
threading.Thread(target=server.serve_forever, daemon=True).start()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.on('console', lambda m: print('CONSOLE:', m.text))
    page.on('pageerror', lambda e: print('PAGE_ERROR:', e))
    page.on('dialog', lambda d: (print('DIALOG:', d.message), d.accept()))
    
    page.add_init_script("""
        localStorage.setItem('betavus.access', '1f7b720c52ea3f6e8631a8eeaffaa7113fbed540ec0772108d39c52835d9855d');localStorage.setItem('betavus.lang', 'tr');
    """)
    print(f"Navigating to http://127.0.0.1:{PORT}/index.html...")
    page.goto(f'http://127.0.0.1:{PORT}/index.html')
    time.sleep(1.5)
    
    # -------------------------------------------------------------------------
    # TEST 1: Model Doğruluğu (#tab-bt -> #pane-bt) Table Content
    # -------------------------------------------------------------------------
    print("\n--- TEST 1: Model Doğruluğu (#pane-bt) Tabloları ---")
    page.evaluate("setTab('bt')")
    time.sleep(0.5)
    assert page.is_visible('#pane-bt') is True, "pane-bt must be visible after clicking tab-bt"
    
    tbl_league = page.inner_html('#bt-byleague')
    tbl_season = page.inner_html('#bt-byseason')
    tbl_samples = page.inner_html('#bt-samples')
    foot_text = page.inner_text('#bt-foot')
    
    rows_league = len(page.query_selector_all('#bt-byleague tr'))
    rows_season = len(page.query_selector_all('#bt-byseason tr'))
    rows_samples = len(page.query_selector_all('#bt-samples tr'))
    
    print(f"  #bt-byleague rows count: {rows_league}")
    print(f"  #bt-byseason rows count: {rows_season}")
    print(f"  #bt-samples rows count: {rows_samples}")
    print(f"  #bt-foot text: {foot_text[:70]}...")
    
    assert rows_league >= 9, f"bt-byleague must have at least 9 league rows, found {rows_league}"
    assert rows_season >= 5, f"bt-byseason must have at least 5 season rows, found {rows_season}"
    assert rows_samples >= 5, f"bt-samples must have sample matches, found {rows_samples}"
    assert len(foot_text.strip()) > 0, "bt-foot must have text"
    print("  ✓ Model Doğruluğu altındaki tüm tablolar (Lig, Sezon, Örnek Maçlar) eksiksiz dolu ve çalışıyor!")

    # Lig satırına tıklama ve dinamik yüzde/KPI güncelleme testi
    # Toplamlar results.json'dan gelir ve lig eklendikçe büyür; sabit sayı yerine ilk açılışla karşılaştır
    kpi_total = lambda: page.inner_text('#btpane-kpi .card:first-child').split('\n')
    kpi_initial = kpi_total()
    pl_row = page.query_selector('#btpane-byleague tr[data-league="Premier League"]')
    assert pl_row is not None, "Premier League row must exist with data-league"
    pl_row.click()
    time.sleep(0.4)
    kpi_pl = page.inner_text('#btpane-kpi')
    print(f"  Premier League tıklandıktan sonra KPI özeti: {kpi_pl[:90].replace(chr(10), ' ')}...")
    assert "Premier League" in kpi_pl, "KPI cards must show Premier League"
    assert "1.900" in kpi_pl, "KPI cards must show 1.900 matches for Premier League (5 × 380)"
    print("  ✓ Lig satırına tıklandığında KPI kartları ve başarı oranları ilgili lige özel anında güncellendi!")

    # Tümü satırına tıklayarak sıfırlama testi
    page.click('#btpane-byleague tr[data-league="Tümü"]')
    time.sleep(0.3)
    assert kpi_total() == kpi_initial, f"KPI kartları Tümü'de ilk açılıştaki toplama dönmeli: {kpi_total()[:4]} != {kpi_initial[:4]}"
    print(f"  ✓ Tümü satırına tıklandığında genel toplam ({kpi_initial[3]} maç) başarıyla geri yüklendi!")

    # (Eski TEST 2 — eski kasa kurulum formu + gizli 'Gerçek Kuponlarım' — kaldırıldı: form 23.09'da
    #  Excel düzenine geçti; yeni Sanal Kasa test_paper_betting.py 12-14'te test ediliyor.)

    # -------------------------------------------------------------------------
    # TEST 3: 30 Günlük Kasa & Kuponlarım Simülasyonu Başarı Oranları (≥95%, ≥85%, ~1.35x)
    # -------------------------------------------------------------------------
    print("\n--- TEST 3: 30-Day Historical Simulation & Success Rates ---")
    page.evaluate("setTab('sim-kasa')")
    time.sleep(0.5)
    assert page.is_visible('#pane-sim-kasa') is True
    
    # async evaluate: fetch results.json inside browser context so RESULTS is always loaded
    sim_stats = page.evaluate("""async () => {
        const PE = window.BETAVUS_PAPER;
        let resData = (window.RESULTS && window.RESULTS.matches) ? window.RESULTS.matches : [];
        if (!resData.length) {
            try {
                const fetched = await fetch('data/results.json?ts=' + Date.now(), { cache: 'no-store' })
                    .then(r => r.ok ? r.json() : null);
                if (fetched && fetched.matches) {
                    resData = fetched.matches;
                    window.RESULTS = fetched;
                }
            } catch(e) {}
        }
        const sim = PE.generate30DayHistoricalSimulation(resData, { startingBank: 50.0 });
        return {
            minimum: sim.profiles.minimum.stats,
            medium: sim.profiles.medium.stats,
            high: sim.profiles.high.stats
        };
    }""")
    
    min_s = sim_stats['minimum']
    med_s = sim_stats['medium']
    high_s = sim_stats['high']
    
    print(f"  Minimum Risk Leg Success: %{min_s['legSuccessRatePct']} (Expected >= %95) | Final Bank: {min_s['finalBank']}€")
    print(f"  Orta Risk Leg Success: %{med_s['legSuccessRatePct']} (Expected >= %85) | Final Bank: {med_s['finalBank']}€")
    print(f"  Yüksek Risk Leg Success: %{high_s['legSuccessRatePct']} (Expected >= %90) | Final Bank: {high_s['finalBank']}€")
    
    assert min_s['legSuccessRatePct'] >= 95.0, f"Minimum risk 0.5Ü başarı oranı %95'in altında: %{min_s['legSuccessRatePct']}"
    assert med_s['legSuccessRatePct'] >= 85.0, f"Orta risk 1.5Ü başarı oranı %85'in altında: %{med_s['legSuccessRatePct']}"
    assert high_s['legSuccessRatePct'] >= 90.0, f"Yüksek risk başarı oranı %90'ın altında: %{high_s['legSuccessRatePct']}"
    
    assert min_s['finalBank'] > 50.0, "Minimum risk kasası büyümeli"
    assert med_s['finalBank'] > 50.0, "Orta risk kasası büyümeli"
    assert high_s['finalBank'] > 50.0, "Yüksek risk kasası büyümeli"
    print("  ✓ Kasa büyüme simülasyonunda maç başarı oranları (0.5Ü ≥ %95, 1.5Ü ≥ %85, Yüksek Risk ~1.35x combo ≥ %90) ve pozitif kasa büyümesi doğrulandı!")

    # -------------------------------------------------------------------------
    # TEST 4: Tüm Sekmelerin Çalışırlık Doğrulaması
    # -------------------------------------------------------------------------
    print("\n--- TEST 4: All Tabs Verification ---")
    for tab_id, pane_id in [
        ('#tab-sim-kasa', '#pane-sim-kasa'),
        ('#tab-sim-kupon', '#pane-sim-kupon'),
        ('#tab-plan', '#pane-plan'),
        ('#tab-rec', '#pane-rec'),
        ('#tab-bt', '#pane-bt'),
        ('#tab-res', '#pane-res'),
        ('#tab-cpn', '#pane-cpn'),
        ('#tab-pred', '#pane-pred'),
        ('#tab-stats', '#pane-stats')
    ]:
        page.evaluate(f"setTab('{tab_id[5:]}')")
        time.sleep(0.3)
        assert page.is_visible(pane_id) is True, f"{pane_id} must be visible when clicking {tab_id}"
        print(f"  ✓ {tab_id} -> {pane_id} başarıyla açıldı ve aktif.")

    print("\n>>> ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY! <<<")
