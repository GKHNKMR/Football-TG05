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
    tabs = [t.inner_text().strip() for t in page.query_selector_all('.tabs .tab')]
    print(f"Mevcut Sekmeler ({len(tabs)}): {tabs}")
    assert '🎲 Çifte Şans & Gol Aralığı' in tabs, "tab-cifte bulunamadı!"
    
    # 2. Click #tab-cifte: doğrudan Model Doğruluğu açılmalı
    print("\n--- 1. '🎲 Çifte Şans & Gol Aralığı' Model Doğruluğu Açılıyor ---")
    page.click('#tab-cifte')
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
    assert len(pred_headers) == 5, f"Tahminler tablosu 5 sütunlu olmalı, şu an: {len(pred_headers)}"
    assert any('ŞANS' in h.upper() for h in pred_headers), "Çifte Şans sütunu bulunamadı!"

    eligible_rows = page.query_selector_all('#rows .row[data-dc-eligible="1"]')
    assert len(eligible_rows) > 0, "Bültende yüksek güvenli Çifte Şans vurgusu bulunamadı!"

    # Vurgu kapalıyken Çifte Şans dahil hiçbir kutu yanıp sönmemeli
    page.select_option('#hlSel', 'off')
    time.sleep(0.2)
    assert not page.query_selector('#rows .dc-pill.hot'), "Vurgu kapalıyken Çifte Şans kutusu yanıyor"
    assert not page.query_selector('#rows .row.dc-highlighted'), "Vurgu kapalıyken Çifte Şans satırı vurgulanıyor"

    # Tüm vurgular açıldığında uygun Çifte Şans kutuları yeniden yanmalı
    page.select_option('#hlSel', 'all')
    time.sleep(0.2)
    eligible_rows = page.query_selector_all('#rows .row[data-dc-eligible="1"]')
    assert all(r.query_selector('.dc-pill.is-highlight.hot') for r in eligible_rows), "Uygun Çifte Şans satırında yanıp sönen yeşil vurgu eksik"
    dc_animation = page.eval_on_selector('.dc-pill.is-highlight.hot', "el => getComputedStyle(el).animationName")
    assert dc_animation == 'pillhot', f"Çifte Şans vurgu animasyonu çalışmıyor: {dc_animation}"
    print(f"  Çifte Şans yüksek güvenli maç sayısı: {len(eligible_rows)}")

    # Kısıtlı veri ve kritik eksik oyunculu iki sentetik maç asla ÇŞ vurgusu almamalı
    page.evaluate("""() => {
      const base = window.__data[0];
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
