import time, threading, sys, os
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
from http.server import HTTPServer, SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright

class H(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

PORT = 8946
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
    
    print("\n--- TEST 1: Tahmin vs Gerçekleşen'de Kısıtlı Veri Kontrolü (Schalke 04 — Elversberg) ---")
    page.click('#tab-res')
    time.sleep(0.8)
    
    # Arama kutusuna Schalke yazıp maçı bulalım
    q_res = page.query_selector('#qRes')
    assert q_res, "Arama kutusu bulunamadı"
    q_res.fill('Schalke')
    time.sleep(0.5)
    
    res_text = page.inner_text('#resRows')
    assert "Schalke 04" in res_text, "Schalke maçı sonuçlarda bulunamadı"
    assert "Elversberg" in res_text, "Elversberg maçı sonuçlarda bulunamadı"
    
    # Bu maçın satırını bulup inceleyelim
    schalke_row = page.query_selector('.rrow[data-home="Schalke 04"]')
    assert schalke_row, "Schalke 04 satırı bulunamadı"
    schalke_row_text = schalke_row.inner_text()
    print("  Schalke 04 satır içeriği:\n   ", schalke_row_text.replace('\n', ' · '))
    
    # 1. Kısıtlı veri uyarısı görünmeli
    assert "Kısıtlı Veri" in schalke_row_text, "Schalke maçında 'Kısıtlı Veri' etiketi eksik!"
    # 2. Asla ★ yıldız vurgusu olmamalı
    assert "★" not in schalke_row_text, "Kısıtlı veri içeren maçta '★' yıldızı olmamalı!"
    # 3. Yön kutusunda yeşil halka (box-shadow) olmamalı
    hit_spans = schalke_row.query_selector_all('.hitset span')
    for sp in hit_spans:
        style = sp.get_attribute('style') or ''
        assert 'box-shadow' not in style, f"Kısıtlı veri yön kutusunda vurgu halkası olmamalı: {style}"
    
    print("  ✓ Schalke 04 — Elversberg maçında kısıtlı verinin vurgusuz olduğu (yıldızsız, halkasız ve 'Kısıtlı Veri' etiketli) doğrulandı!")

    # Sadece vurgulananlar filtresi testi
    print("\n--- TEST 2: '⚡ Sadece Vurgulanan' Filtresinde Kısıtlı Veri Hariç Tutma ---")
    # Arama kutusunu temizle
    q_res.fill('')
    time.sleep(0.5)
    page.click('#vurguMatchToggle')
    time.sleep(0.5)
    
    # Vurgulanan maçlar listesinde Schalke - Elversberg olmamalı!
    vurgu_text = page.inner_text('#resRows')
    assert "Schalke 04 — Elversberg" not in vurgu_text, "Kısıtlı verili Schalke maçı '⚡ sadece vurgulanan' listesinde ÇIKMAMALI!"
    print("  ✓ Kısıtlı verili maçların '⚡ sadece vurgulanan' listesine kesinlikle dahil edilmediği doğrulandı!")
    
    # Ekran görüntüsü al (Schalke maçı)
    q_res.fill('Schalke')
    page.click('#vurguMatchToggle') # kapat
    time.sleep(0.5)
    target_dir = Path(r"C:\Users\mtem01\.gemini\antigravity\brain\f33eb80d-72fc-483c-aef4-dc3ca74ebe48")
    page.screenshot(path=str(target_dir / "schalke_limited_data_clean.png"))
    print("  ✓ Schalke maçı temiz görünüm ekran görüntüsü kaydedildi: schalke_limited_data_clean.png")

    print("\n--- TEST 3: Çifte Şans & Skor Model Doğruluğu (16.478 Maç & Sıfır Iska) ---")
    page.click('#tab-cifte')
    time.sleep(0.6)
    
    bt_toggle = page.query_selector('.subtab-toggle[data-view="bt"]')
    assert bt_toggle, "Model Doğruluğu butonu bulunamadı!"
    bt_toggle.click()
    time.sleep(0.8)
    
    cifte_text = page.inner_text('#pane-cifte')
    
    # 1. Maç sayısı 16.478 olmalı (17.003 olmamalı!)
    assert "16.478" in cifte_text, "16.478 maç sayısı bulunamadı!"
    assert "17.003" not in cifte_text, "17.003 maç sayısı artık ÇIKMAMALI!"
    print("  ✓ Toplam maç sayısı tam 16.478 olarak doğrulandı (17.003 tamamen kaldırıldı).")
    
    # 2. Iska / Tutmadı kutuları olmamalı
    # Başlık ve kartlarda "Iska" veya "Tutmadı" kutusu olmamalı
    assert "Iska" not in cifte_text, "KPI kartlarında veya tablolarda 'Iska' kelimesi olmamalı!"
    assert "Tutmadı" not in cifte_text, "KPI kartlarında veya tablolarda 'Tutmadı' kelimesi olmamalı!"
    print("  ✓ 'Iska' ve 'Tutmadı' kutularının ve metinlerinin tamamen kaldırıldığı doğrulandı!")
    
    # 3. KPI kartları başarı oranları
    assert "1X Çifte Şans" in cifte_text, "1X kartı eksik"
    assert "12 Çifte Şans" in cifte_text, "12 kartı eksik"
    assert "X2 Çifte Şans" in cifte_text, "X2 kartı eksik"
    assert "Skor Tahmin Havuzu" in cifte_text, "Skor kartı eksik"
    print("  ✓ 4 Adet Model Doğruluğu KPI Kartı başarıyla doğrulandı!")
    
    # 4. Lig tablosu 6 kolonlu ve sade olmalı
    th_elements = page.query_selector_all('#tblCifteLeague thead th')
    th_texts = [th.inner_text().strip() for th in th_elements]
    print(f"  Lig tablosu başlıkları ({len(th_texts)} adet):", " | ".join(th_texts))
    assert len(th_texts) == 6, f"Lig tablosu tam 6 kolon olmalı (önceden 13 idi), şu an: {len(th_texts)}"
    
    # Ekran görüntüsü al (Çifte Şans Backtest)
    page.screenshot(path=str(target_dir / "cifte_backtest_clean.png"))
    print("  ✓ Çifte Şans Model Doğruluğu temiz görünüm ekran görüntüsü kaydedildi: cifte_backtest_clean.png")

    print("\n>>> TÜM DOĞRULAMA TESTLERİ BAŞARIYLA TAMAMLANDI! <<<")
