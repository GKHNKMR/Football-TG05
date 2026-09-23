import time, threading, sys, os
sys.stdout.reconfigure(encoding='utf-8')
from http.server import HTTPServer, SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright

class H(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

PORT = 8940
server = HTTPServer(('127.0.0.1', PORT), H)
threading.Thread(target=server.serve_forever, daemon=True).start()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1280, 'height': 900})
    page.on('console', lambda m: print('CONSOLE:', m.text))
    page.on('pageerror', lambda e: print('PAGE_ERROR:', e))
    
    page.add_init_script("localStorage.setItem('betavus.access', '1f7b720c52ea3f6e8631a8eeaffaa7113fbed540ec0772108d39c52835d9855d');")
    page.goto(f'http://127.0.0.1:{PORT}/index.html')
    time.sleep(1.5)
    
    # 1. Verify #tab-cifte exists next to #tab-pred
    # Menüde yalnızca 3 ana sekme görünür; Çifte Şans sekmesi gizli ama kodu duruyor
    tabs = [t.inner_text().strip() for t in page.query_selector_all('.tabs .tab:not([hidden])')]
    print(f"Mevcut Sekmeler ({len(tabs)}): {tabs}")
    assert [t.upper() for t in tabs] == ['BÜLTEN', 'İSTATİSTİKLER', 'SANAL KASA', 'FAQ'], f"Ana sekmeler hatalı: {tabs}"
    assert page.query_selector('#tab-cifte'), "tab-cifte DOM'da bulunamadı!"
    
    # 2. Click #tab-cifte: doğrudan Model Doğruluğu açılmalı
    print("\n--- 1. '🎲 Çifte Şans & Gol Aralığı' Model Doğruluğu Açılıyor ---")
    page.evaluate("setTab('cifte')")
    time.sleep(0.8)
    
    pane = page.query_selector('#pane-cifte')
    assert pane and pane.is_visible(), "pane-cifte görünür değil!"
    
    cifte_text = page.inner_text('#pane-cifte')
    assert "Çifte Şans & Gol Aralığı Model Doğruluğu" in cifte_text, "Model Doğruluğu doğrudan açılmadı"
    assert "Güncel Fikstür & Tahminler" not in cifte_text, "Güncel fikstür alt görünümü kaldırılmamış"
    assert not page.query_selector('.cifte-match-card'), "Güncel maç kartları bu sekmede görünmemeli"
    assert not page.query_selector('.cifte-chips'), "Güncel tahmin filtreleri bu sekmede görünmemeli"
    assert not page.query_selector('#qCifte'), "Güncel tahmin araması bu sekmede görünmemeli"
    assert not page.query_selector('.subtab-toggle'), "Güncel fikstür / doğruluk alt sekmeleri kaldırılmalı"

    # Gol aralığı motoru çalışmaya devam etmeli; güncel gösterim Tahminler bültenindedir.
    range_result = page.evaluate("""() => {
      const analysis = window.BETAVUS_CIFTE.analyzeMatch(window.__data[0]);
      return {keys:Object.keys(analysis.goalRanges), best:analysis.bestGoalRange.pick};
    }""")
    assert range_result['keys'] == ['2-3', '3-4', '5+'], f"Gol aralığı motoru eksik: {range_result}"
    assert range_result['best'] in range_result['keys'], "Model geçerli bir gol aralığı seçmedi"
    print("  ✓ Güncel fikstür kaldırıldı; sekme doğrudan Model Doğruluğu ekranını açıyor.")

    # 3. Capture screenshot
    screenshot_path = 'scratch/cifte_sans_screenshot.png'
    page.screenshot(path=screenshot_path, full_page=False)
    print(f"  ✓ Ekran görüntüsü kaydedildi: {screenshot_path}")
    
    # 4. Verify Tahminler (Bülten) tab still works
    print("\n--- 2. '⚽ Tahminler (Bülten)' Regresyon Doğrulaması ---")
    page.click('#tab-pred')
    time.sleep(0.6)
    pred_pane = page.query_selector('#pane-pred')
    assert pred_pane and pred_pane.is_visible(), "pane-pred açılmadı!"
    pred_rows = page.query_selector_all('#rows .row')
    print(f"  Tahminler bülteni satır sayısı: {len(pred_rows)}")
    assert len(pred_rows) > 0, "Tahminler sekmesi boş kaldı!"
    pred_headers = [h.inner_text().strip() for h in page.query_selector_all('#pane-pred .head .sortcol')]
    assert len(pred_headers) == 7, f"Tahminler tablosu 7 sütunlu olmalı, şu an: {len(pred_headers)}"
    assert pred_headers[1:] == ['0.5+', '1.5+', '2.5+', '1X', '12', 'X2'], f"Bülten sütunları hatalı: {pred_headers}"
    assert 'Gol' not in page.inner_text('#rows').replace('Gol ', ''), "Bültende gol aralığı olmamalı"

    # Seçili ligde maç yoksa başka ligden maç geri gelmemeli (eski fallback hatası)
    page.evaluate("""() => {
      window.__leagueFilterOriginal = window.__data;
      window.__data = [{...window.__data[0], match_id:'TEST-ONLY-PRIMEIRA', league:'Primeira Liga'}];
      renderPred();
    }""")
    page.select_option('#lgPred', 'Premier League')
    time.sleep(0.2)
    assert page.eval_on_selector('#lgPred', 'e=>e.value') == 'Premier League', "Premier League filtresi aktif olmadı"
    assert not page.query_selector('#rows .row'), "Premier League seçiliyken Primeira Liga maçı görünmemeli"
    assert "TEST-ONLY-PRIMEIRA" not in page.inner_text('#rows'), "Seçili lig dışında maç listelendi"
    page.evaluate("""() => { window.__data = window.__leagueFilterOriginal; delete window.__leagueFilterOriginal; }""")
    page.select_option('#lgPred', 'Tümü')
    time.sleep(0.2)
    print("  ✓ Lig filtresi, seçili ligde maç yokken başka ligleri göstermiyor.")

    # Veriden bağımsız: pencere içinde ev sahibinin açık favori olduğu sentetik, tam verili bir maç ekle
    page.evaluate("""() => {
      const ko = new Date(Date.now() + 2*86400000).toISOString().slice(0,19) + 'Z';
      window.__data.push({match_id:'TEST-DC-STRONG', league:'Premier League', home:'Test Güçlü', away:'Test Zayıf', kickoff_utc:ko,
        basis:'form+h2h', h2h_matches_used:6, lam_home:2.3, lam_away:0.5, exp_goals:2.8, rho:0.02,
        p_over_0_5:0.94, p_over_1_5:0.77, p_over_2_5:0.53});
      renderPred();
    }""")
    eligible_rows = page.query_selector_all('#rows .row[data-dc-eligible="1"]')
    assert len(eligible_rows) > 0, "Bültende yüksek güvenli Çifte Şans vurgusu bulunamadı!"

    # Vurgu kapalıyken Çifte Şans dahil hiçbir kutu yanıp sönmemeli
    page.select_option('#hlSel', 'off')
    time.sleep(0.2)
    assert not page.query_selector('#rows .dcp .pill.hot'), "Vurgu kapalıyken Çifte Şans kutusu yanıyor"
    assert not page.query_selector('#rows .row.dc-highlighted'), "Vurgu kapalıyken Çifte Şans satırı vurgulanıyor"

    # Tüm vurgular açıldığında uygun Çifte Şans kutuları yeniden yanmalı
    page.select_option('#hlSel', 'all')
    time.sleep(0.2)
    eligible_rows = page.query_selector_all('#rows .row[data-dc-eligible="1"]')
    assert all(r.query_selector('.dcp .pill.hot') for r in eligible_rows), "Uygun Çifte Şans satırında yanıp sönen vurgu eksik"
    dc_animation = page.eval_on_selector('.dcp .pill.hot', "el => getComputedStyle(el).animationName")
    assert dc_animation in ('pillhot', 'pillhot2'), f"Çifte Şans vurgu animasyonu çalışmıyor: {dc_animation}"
    for r in eligible_rows:   # vurgulanan her ÇŞ hücresi gerçekten ≥%80
        for v in r.eval_on_selector_all('.dcp .pill.hot', "e=>e.map(x=>parseFloat(x.textContent))"):
            assert v >= 80, f"%80 altı ÇŞ vurgulandı: {v}"
    print(f"  Çifte Şans yüksek güvenli maç sayısı: {len(eligible_rows)}")

    # Kısıtlı veri ve kritik eksik oyunculu iki sentetik maç asla ÇŞ vurgusu almamalı
    page.evaluate("""() => {
      const base = window.__data.find(m => m.match_id === 'TEST-DC-STRONG');
      window.__data.push({...base, match_id:'TEST-DC-LIMITED', home:'Test Limited', basis:'partial-form', h2h_tier:null, h2h_matches_used:0});
      window.__data.push({...base, match_id:'TEST-DC-CRITICAL', home:'Test Critical', lineup:{source:'ESPN starting XI',home_missing_key:['Kilit Oyuncu'],away_missing_key:[]}});
      renderPred();
    }""")
    limited_row = page.query_selector('#rows .row[data-mid="TEST-DC-LIMITED"]')
    critical_row = page.query_selector('#rows .row[data-mid="TEST-DC-CRITICAL"]')
    assert limited_row and limited_row.get_attribute('data-dc-eligible') == '0', "Kısıtlı veri ÇŞ vurgusu almamalı"
    assert critical_row and critical_row.get_attribute('data-dc-eligible') == '0', "Kritik eksik oyunculu maç ÇŞ vurgusu almamalı"

    # ÇŞ filtresi açılınca yalnızca uygun ve vurgulu maçlar listelenmeli
    page.select_option('#hlSel', 'dc')
    page.click('#hotToggle')
    time.sleep(0.4)
    filtered_rows = page.query_selector_all('#rows .row')
    assert len(filtered_rows) > 0, "Çifte Şans vurguları filtresi sonuç vermedi"
    assert all(r.get_attribute('data-dc-eligible') == '1' for r in filtered_rows), "ÇŞ filtresinde uygunsuz maç listelendi"
    assert not page.query_selector('#rows .row[data-mid="TEST-DC-LIMITED"]'), "Kısıtlı veri ÇŞ filtresine girdi"
    assert not page.query_selector('#rows .row[data-mid="TEST-DC-CRITICAL"]'), "Kritik maç ÇŞ filtresine girdi"
    assert 'ÇŞ' in page.inner_text('#hotToggle'), "ÇŞ filtre düğmesi etiketi güncellenmedi"

    page.screenshot(path='scratch/predictions_double_chance.png', full_page=False)
    print("  ✓ Tahminler bültenine Çifte Şans sütunu ve sıkı yüksek güven filtresi eklendi.")
    
    print("\n>>> ÇİFTE ŞANS & GOL ARALIĞI TAHMİNLERİ TÜM TESTLERİ BAŞARIYLA GEÇTİ! <<<")
    browser.close()
