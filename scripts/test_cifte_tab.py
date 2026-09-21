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
    pred_headers = [h.inner_text().strip() for h in page.query_selector_all('#pane-pred .head .sortcol')]
    assert len(pred_headers) == 5, f"Tahminler tablosu 5 sütunlu olmalı, şu an: {len(pred_headers)}"
    assert any('ŞANS' in h.upper() for h in pred_headers), "Çifte Şans sütunu bulunamadı!"

    eligible_rows = page.query_selector_all('#rows .row[data-dc-eligible="1"]')
    assert len(eligible_rows) > 0, "Bültende yüksek güvenli Çifte Şans vurgusu bulunamadı!"
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
    
    print("\n>>> ÇİFTE ŞANS & SKOR TAHMİNLERİ TÜM TESTLERİ BAŞARIYLA GEÇTİ! <<<")
    browser.close()
