import time, threading, sys, os
sys.stdout.reconfigure(encoding='utf-8')
from http.server import HTTPServer, SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright

class H(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

PORT = 8920
server = HTTPServer(('127.0.0.1', PORT), H)
threading.Thread(target=server.serve_forever, daemon=True).start()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.on('console', lambda m: print('CONSOLE:', m.text))
    page.on('pageerror', lambda e: print('PAGE_ERROR:', e))
    page.on('dialog', lambda d: (print('DIALOG:', d.message), d.accept()))
    
    page.add_init_script("""
        localStorage.setItem('betavus.access', '1f7b720c52ea3f6e8631a8eeaffaa7113fbed540ec0772108d39c52835d9855d');
    """)
    page.goto(f'http://127.0.0.1:{PORT}/index.html')
    time.sleep(1.5)
    
    # Navigate to #tab-plan
    page.click('#tab-plan')
    time.sleep(0.5)
    
    # If a plan exists, reset it so we start fresh
    if page.query_selector('#btnResetPlan'):
        page.click('#btnResetPlan')
        time.sleep(0.5)
        
    print("--- 1. İlk Kasa (Minimum Risk) Oluşturuluyor ---")
    page.fill('#setupPlanName', '1. Minimum Kasa (50€)')
    page.fill('#setupStartBank', '50')
    page.fill('#setupTargetBank', '500')
    page.click('.risk-card[data-risk="minimum"]')
    page.click('#btnCreatePlan')
    time.sleep(0.6)
    
    # Check tabs
    tabs = page.query_selector_all('.bankroll-tab')
    print(f"  Oluşturulan kasa sekme sayısı: {len(tabs)}")
    assert len(tabs) == 1, f"Beklenen 1 sekme, bulunan: {len(tabs)}"
    
    # Verify single-risk chart and summary card
    summary_cards = page.query_selector_all('#planChartCard .cms-card')
    print(f"  Grafik altındaki model kart sayısı: {len(summary_cards)} (Tekil risk bekleniyor)")
    assert len(summary_cards) == 1, f"Beklenen 1 model kartı, bulunan: {len(summary_cards)}"
    assert "Minimum Risk" in page.inner_text('#planChartCard .cms-card')
    assert "Sıfır Kayıp Potansiyeli" in page.inner_text('#planChartCard .cms-card')
    print("  ✓ Tekil risk grafiği ve sıfır kayıp potansiyeli başarıyla doğrulandı!")
    
    print("\n--- 2. İkinci Kasa (Orta Risk) Ekleniyor ---")
    page.click('#btnAddNewPlan')
    time.sleep(0.5)
    page.fill('#setupPlanName', '2. Orta Risk Kasa (100€)')
    page.fill('#setupStartBank', '100')
    page.fill('#setupTargetBank', '1000')
    page.click('.risk-card[data-risk="medium"]')
    page.click('#btnCreatePlan')
    time.sleep(0.6)
    
    tabs = page.query_selector_all('.bankroll-tab')
    print(f"  Toplam kasa sekme sayısı: {len(tabs)}")
    assert len(tabs) == 2, f"Beklenen 2 sekme, bulunan: {len(tabs)}"
    assert "Orta Risk" in page.inner_text('#planChartCard .cms-card')
    print("  ✓ İkinci kasa (Orta Risk) başarıyla açıldı ve aktif oldu!")
    
    print("\n--- 3. Üçüncü Kasa (Özel Risk) Ekleniyor ---")
    page.click('#btnAddNewPlan')
    time.sleep(0.5)
    page.fill('#setupPlanName', '3. Özel Kasa (%30 Rezerv)')
    page.fill('#setupStartBank', '75')
    page.fill('#setupTargetBank', '750')
    page.click('.risk-card[data-risk="custom"]')
    time.sleep(0.3)
    
    # Verify custom controls are visible
    assert page.is_visible('#customRiskControls') is True, "Özel risk kontrol paneli görünür olmalı"
    page.fill('#customReservePct', '30')
    page.fill('#customStakeRate', '60')
    page.fill('#customTargetOdds', '1.35')
    time.sleep(0.3)
    
    custom_growth_text = page.inner_text('#customDailyGrowthText')
    print(f"  Özel risk hesaplanan büyüme: {custom_growth_text}")
    
    page.click('#btnCreatePlan')
    time.sleep(0.6)
    
    tabs = page.query_selector_all('.bankroll-tab')
    print(f"  Toplam kasa sekme sayısı: {len(tabs)}")
    assert len(tabs) == 3, f"Beklenen 3 sekme, bulunan: {len(tabs)}"
    assert "Özel Risk" in page.inner_text('#planChartCard .cms-card') or "Özel Kasa" in page.inner_text('#planChartCard .cms-card')
    print("  ✓ Üçüncü kasa (Özel Risk) başarıyla açıldı!")
    
    # Take screenshot of the multi-bankroll screen
    os.makedirs('scratch', exist_ok=True)
    screenshot_path = 'scratch/multi_bankroll_screenshot.png'
    page.screenshot(path=screenshot_path, full_page=True)
    print(f"  ✓ Ekran görüntüsü kaydedildi: {screenshot_path}")
    
    print("\n--- 4. Kasalar Arası Geçiş (Tab Switching) Testi ---")
    # Click first tab (Minimum Risk)
    tabs[0].click()
    time.sleep(0.5)
    assert "Minimum Risk" in page.inner_text('#planChartCard .cms-card')
    print("  ✓ Minimum risk kasasına geçildi ve grafik anında güncellendi.")
    
    # Click second tab (Orta Risk)
    tabs = page.query_selector_all('.bankroll-tab')
    tabs[1].click()
    time.sleep(0.5)
    assert "Orta Risk" in page.inner_text('#planChartCard .cms-card')
    print("  ✓ Orta risk kasasına geçildi ve grafik anında güncellendi.")
    
    print("\n--- 5. Kasa Silme Testi ---")
    page.click('#btnDeleteCurrentPlan')
    time.sleep(0.5)
    tabs_after_del = page.query_selector_all('.bankroll-tab')
    print(f"  Silme sonrası kalan kasa sayısı: {len(tabs_after_del)}")
    assert len(tabs_after_del) == 2, f"Beklenen 2 kasa kalması, bulunan: {len(tabs_after_del)}"
    print("  ✓ Aktif kasa başarıyla silindi, kalan kasalar korundu!")
    
    print("\n>>> ÇOKLU KASA VE TEKİL RİSK GRAFİĞİ TESTİ BAŞARIYLA GEÇTİ! <<<")
