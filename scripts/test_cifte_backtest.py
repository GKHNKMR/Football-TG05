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
project_root = Path(__file__).resolve().parents[1]
target_dir = project_root / 'scratch'
target_dir.mkdir(exist_ok=True)

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
    
    # 4. Lig ve sezon tabloları Model Doğruluğu ile aynı iki katmanlı grup başlığına sahip olmalı
    league_top_headers = page.query_selector_all('#tblCifteLeague thead tr:first-child th')
    league_sub_headers = page.query_selector_all('#tblCifteLeague thead tr:nth-child(2) th')
    league_group_headers = page.query_selector_all('#tblCifteLeague thead .cifte-bt-group-head')
    league_data_cells = page.query_selector_all('#tblCifteLeague tbody tr:first-child td')
    assert len(league_top_headers) == 6, f"Lig tablosu üst başlık satırı 6 hücre olmalı, şu an: {len(league_top_headers)}"
    assert len(league_group_headers) == 4, f"Lig tablosunda 4 pazar grubu olmalı, şu an: {len(league_group_headers)}"
    assert all(h.get_attribute('colspan') == '3' for h in league_group_headers), "Her pazar grubu 3 alt sütun taşımalı"
    assert len(league_sub_headers) == 12, f"Lig tablosunda 12 alt başlık olmalı, şu an: {len(league_sub_headers)}"
    assert len(league_data_cells) == 14, f"Lig tablosu satırı 14 mantıksal sütun olmalı, şu an: {len(league_data_cells)}"

    season_group_headers = page.query_selector_all('#tblCifteSeason thead .cifte-bt-group-head')
    season_data_cells = page.query_selector_all('#tblCifteSeason tbody tr:first-child td')
    sample_headers = page.query_selector_all('#tblCifteSamples thead th')
    assert len(season_group_headers) == 4, f"Sezon tablosunda 4 pazar grubu olmalı, şu an: {len(season_group_headers)}"
    assert len(season_data_cells) == 14, f"Sezon tablosu satırı 14 mantıksal sütun olmalı, şu an: {len(season_data_cells)}"
    assert len(sample_headers) == 6, f"Örnek maç tablosu tam 6 kolon olmalı, şu an: {len(sample_headers)}"
    print("  ✓ Lig ve sezon tabloları Vurgu / Tuttu / % alt sütunlu Model Doğruluğu formatında.")

    # 5. Masaüstü okunabilirliği: KPI kartları 2 sütun, ana rakamlar ve tablo metni yeterince büyük
    kpi_cards = page.query_selector_all('.cifte-bt-kpi')
    assert len(kpi_cards) == 4, f"Tam 4 KPI kartı olmalı, şu an: {len(kpi_cards)}"
    kpi_columns = page.eval_on_selector(
        '.cifte-bt-kpi-grid',
        "el => getComputedStyle(el).gridTemplateColumns.split(' ').length"
    )
    metric_font = page.eval_on_selector(
        '.cifte-bt-metric-value',
        "el => parseFloat(getComputedStyle(el).fontSize)"
    )
    table_font = page.eval_on_selector(
        '#tblCifteLeague',
        "el => parseFloat(getComputedStyle(el).fontSize)"
    )
    assert kpi_columns == 2, f"Masaüstünde KPI kartları 2 sütun olmalı, şu an: {kpi_columns}"
    assert metric_font >= 20, f"KPI ana rakamları en az 20px olmalı, şu an: {metric_font}px"
    assert table_font >= 12.5, f"Tablo metni en az 12.5px olmalı, şu an: {table_font}px"
    sample_rows = page.query_selector_all('#tblCifteSamples tbody tr')
    assert len(sample_rows) == 12, f"Örnek maçlar ilk açılışta 12 satır göstermeli, şu an: {len(sample_rows)}"
    assert page.query_selector('#btnToggleCifteSamples'), "Daha fazla örnek göster düğmesi bulunamadı"
    print("  ✓ KPI yerleşimi, karakter büyüklükleri ve tablo okunabilirliği doğrulandı.")
    
    # Ekran görüntüsü al (Çifte Şans Backtest)
    page.screenshot(path=str(target_dir / "cifte_backtest_readability.png"), full_page=True)
    print("  ✓ Çifte Şans Model Doğruluğu okunabilirlik ekran görüntüsü kaydedildi: cifte_backtest_readability.png")

    # 6. Mobil yerleşim: KPI tek sütun, geniş tablolar kontrollü yatay kaydırma
    page.set_viewport_size({'width': 390, 'height': 844})
    time.sleep(0.3)
    mobile_kpi_columns = page.eval_on_selector(
        '.cifte-bt-kpi-grid',
        "el => getComputedStyle(el).gridTemplateColumns.split(' ').length"
    )
    table_scrolls = page.eval_on_selector(
        '#tblCifteLeague',
        "el => el.parentElement.scrollWidth > el.parentElement.clientWidth"
    )
    assert mobile_kpi_columns == 1, f"Mobilde KPI kartları tek sütun olmalı, şu an: {mobile_kpi_columns}"
    assert table_scrolls, "Mobilde geniş tablo kontrollü yatay kaydırılabilir olmalı"
    page.screenshot(path=str(target_dir / "cifte_backtest_mobile.png"), full_page=True)
    print("  ✓ Mobil KPI yerleşimi ve yatay kaydırılabilir tablo davranışı doğrulandı.")

    print("\n>>> TÜM DOĞRULAMA TESTLERİ BAŞARIYLA TAMAMLANDI! <<<")
