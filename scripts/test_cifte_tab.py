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
    assert '🎲 Çifte Şans & Skor' in tabs, "tab-cifte bulunamadı!"
    
    # 2. Click #tab-cifte
    print("\n--- 1. '🎲 Çifte Şans & Skor' Sekmesine Tıklanıyor ---")
    page.click('#tab-cifte')
    time.sleep(0.8)
    
    pane = page.query_selector('#pane-cifte')
    assert pane and pane.is_visible(), "pane-cifte görünür değil!"
    
    # Verify match cards rendered
    cards = page.query_selector_all('.cifte-match-card')
    print(f"  Listelenen maç kartı sayısı: {len(cards)}")
    assert len(cards) > 0, "Maç kartları render edilemedi!"
    
    # Check first card content
    first_card = cards[0]
    first_text = first_card.inner_text()
    print("  Örnek maç kartı başlığı:")
    print("   ", first_text.split('\n')[0], "|", first_text.split('\n')[1] if len(first_text.split('\n')) > 1 else "")
    assert "1X" in first_text and "12" in first_text and "X2" in first_text, "Çifte şans seçenekleri eksik!"
    assert "En Olası Skorlar" in first_text, "Skor tahminleri eksik!"
    print("  ✓ 1X, 12, X2 ve En Olası Skorlar kartta başarıyla görüntülendi.")
    
    # 3. Test Filters
    print("\n--- 2. Çifte Şans Filtreleri Test Ediliyor ---")
    # Click 1X filter
    page.click('.cifte-chips button[data-cf="dc_1x"]')
    time.sleep(0.4)
    cards_1x = page.query_selector_all('.cifte-match-card')
    print(f"  1X Ağırlıklı maç sayısı: {len(cards_1x)}")
    assert len(cards_1x) > 0, "1X filtresi sonuç vermedi!"
    
    # Click High Confidence filter
    page.click('.cifte-chips button[data-cf="high_conf"]')
    time.sleep(0.4)
    cards_high = page.query_selector_all('.cifte-match-card')
    print(f"  Yüksek Güven (≥%75) maç sayısı: {len(cards_high)}")
    assert len(cards_high) > 0, "Yüksek güven filtresi sonuç vermedi!"
    
    # Click All
    page.click('.cifte-chips button[data-cf="all"]')
    time.sleep(0.4)
    
    # 4. Test Search
    print("\n--- 3. Takım Arama Test Ediliyor ---")
    page.fill('#qCifte', 'Madrid')
    time.sleep(0.5)
    cards_search = page.query_selector_all('.cifte-match-card')
    print(f"  'Madrid' araması sonucu kart sayısı: {len(cards_search)}")
    assert len(cards_search) > 0, "Arama sonucu bulunamadı!"
    page.fill('#qCifte', '')
    time.sleep(0.4)
    
    # 5. Capture full screenshot
    screenshot_path = 'scratch/cifte_sans_screenshot.png'
    page.screenshot(path=screenshot_path, full_page=False)
    print(f"  ✓ Ekran görüntüsü kaydedildi: {screenshot_path}")
    
    # 6. Verify Tahminler (Bülten) tab still works
    print("\n--- 4. '⚽ Tahminler (Bülten)' Regresyon Doğrulaması ---")
    page.click('#tab-pred')
    time.sleep(0.6)
    pred_pane = page.query_selector('#pane-pred')
    assert pred_pane and pred_pane.is_visible(), "pane-pred açılmadı!"
    pred_rows = page.query_selector_all('#rows .row')
    print(f"  Tahminler bülteni satır sayısı: {len(pred_rows)}")
    assert len(pred_rows) > 0, "Tahminler sekmesi boş kaldı!"
    print("  ✓ '⚽ Tahminler' sekmesi bozulmadan eksiksiz çalışıyor.")
    
    print("\n>>> ÇİFTE ŞANS & SKOR TAHMİNLERİ TÜM TESTLERİ BAŞARIYLA GEÇTİ! <<<")
    browser.close()
