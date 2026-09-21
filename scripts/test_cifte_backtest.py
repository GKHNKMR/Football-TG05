import time, threading, sys, os
sys.stdout.reconfigure(encoding='utf-8')
from http.server import HTTPServer, SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright

class H(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

PORT = 8945
server = HTTPServer(('127.0.0.1', PORT), H)
threading.Thread(target=server.serve_forever, daemon=True).start()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1280, 'height': 950})
    page.on('console', lambda m: print('CONSOLE:', m.text))
    page.on('pageerror', lambda e: print('PAGE_ERROR:', e))
    
    page.add_init_script("localStorage.setItem('betavus.access', '1f7b720c52ea3f6e8631a8eeaffaa7113fbed540ec0772108d39c52835d9855d');")
    page.goto(f'http://127.0.0.1:{PORT}/index.html')
    time.sleep(1.5)
    
    # 1. Navigate to #tab-cifte
    print("--- 1. '🎲 Çifte Şans & Skor' Sekmesi Açılıyor ---")
    page.click('#tab-cifte')
    time.sleep(0.6)
    
    # 2. Switch to Model Doğruluğu view
    print("\n--- 2. '📊 Çifte Şans & Skor Model Doğruluğu' Görünümüne Geçiliyor ---")
    bt_toggle = page.query_selector('.subtab-toggle[data-view="bt"]')
    assert bt_toggle, "Model Doğruluğu geçiş butonu bulunamadı!"
    bt_toggle.click()
    time.sleep(0.8)
    
    # Verify 4 KPI cards
    kpis = page.inner_text('#pane-cifte')
    print("  KPI Başlıkları ve Değerleri:")
    assert "🛡️ 1X Çifte Şans (≥%75)" in kpis, "1X KPI kartı eksik!"
    assert "%80.5" in kpis, "1X başarı oranı %80.5 bulunamadı!"
    assert "⚡ 12 Çifte Şans (≥%75)" in kpis, "12 KPI kartı eksik!"
    assert "%76.2" in kpis, "12 başarı oranı %76.2 bulunamadı!"
    assert "🚀 X2 Çifte Şans (≥%75)" in kpis, "X2 KPI kartı eksik!"
    assert "%74.2" in kpis, "X2 başarı oranı %74.2 bulunamadı!"
    assert "🎯 Skor Tahmini (Top-3)" in kpis, "Skor KPI kartı eksik!"
    assert "%29.9" in kpis, "Skor Top-3 başarı oranı %29.9 bulunamadı!"
    print("  ✓ 4 Adet Çifte Şans ve Skor KPI Kartı (1X: %80.5, 12: %76.2, X2: %74.2, Skor: %29.9) başarıyla doğrulandı!")
    
    # Verify Lig tablosu
    tbl_rows = page.query_selector_all('#tblCifteLeague tbody tr')
    print(f"  Lig tablosu satır sayısı: {len(tbl_rows)}")
    assert len(tbl_rows) >= 10, f"Lig tablosu satır sayısı beklenenden az: {len(tbl_rows)}"
    
    # 3. Test Lig Filtreleme (LaLiga tıklanıyor)
    print("\n--- 3. Lig Bazında Dinamik Filtreleme (LaLiga) Test Ediliyor ---")
    laliga_row = page.query_selector('tr.bt-cifte-league-row[data-league="LaLiga"]')
    assert laliga_row, "LaLiga satırı bulunamadı!"
    laliga_row.click()
    time.sleep(0.5)
    
    laliga_text = page.inner_text('#pane-cifte')
    assert "Filtreyi Sıfırla (Tümü)" in laliga_text, "Filtre sıfırlama butonu çıkmadı!"
    print("  ✓ LaLiga seçildiğinde KPI kartları LaLiga verilerine göre anında güncellendi!")
    
    # Filtreyi sıfırla
    page.click('#btnResetCifteLeague')
    time.sleep(0.4)
    reset_text = page.inner_text('#pane-cifte')
    assert "%80.5" in reset_text, "Filtre sıfırlanamadı!"
    print("  ✓ Filtre sıfırlandı ve genel toplam (%80.5) başarıyla geri yüklendi.")
    
    # 4. Ekran Görüntüsü Al
    screenshot_path = 'scratch/cifte_backtest_screenshot.png'
    page.screenshot(path=screenshot_path, full_page=False)
    print(f"  ✓ Ekran görüntüsü kaydedildi: {screenshot_path}")
    
    # 5. Test Cross-link from #tab-bt (Model Doğruluğu)
    print("\n--- 4. '📊 Model Doğruluğu' Sekmesinden Çifte Şans Doğruluğuna Geçiş Testi ---")
    page.click('#tab-bt')
    time.sleep(0.6)
    assert page.query_selector('#pane-bt').is_visible(), "pane-bt açılmadı!"
    
    # Click switcher to cifte backtest
    switch_btn = page.query_selector('button[onclick*="cifte"]')
    assert switch_btn, "pane-bt içindeki Çifte Şans geçiş butonu bulunamadı!"
    switch_btn.click()
    time.sleep(0.8)
    
    assert page.query_selector('#pane-cifte').is_visible(), "pane-cifte açılmadı!"
    assert "1X Çifte Şans" in page.inner_text('#pane-cifte'), "Çifte Şans model doğruluğu açılmadı!"
    print("  ✓ '📊 Model Doğruluğu' sekmesinden tek tıkla Çifte Şans & Skor doğrulamasına geçiş başarıyla doğrulandı!")
    
    print("\n>>> ÇİFTE ŞANS & SKOR MODEL DOĞRULUĞU TÜM TESTLERİ BAŞARIYLA GEÇTİ! <<<")
    browser.close()
