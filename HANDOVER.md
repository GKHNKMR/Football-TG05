# BETAVUS — Proje Devir & Handover Dokümanı (Claude Code İçin)

> **Tarih**: 21 Eylül 2026  
> **Repository**: `https://github.com/GKHNKMR/Football-TG05.git`  
> **Aktif Dal (Branch)**: `main` (Son commit: `2858bfe`)  
> **Çalışma Dizini**: `c:\Users\mtem01\OneDrive\VS-Studio Working File\Football Goal Analyst\Football-TG05-main`  
> **Erişim Anahtarı (Token)**: `1f7b720c52ea3f6e8631a8eeaffaa7113fbed540ec0772108d39c52835d9855d`

---

## 1. Proje Özeti & Vizyonu

**BETAVUS** (Betting Analysis, Variance & Uncertainty System), futbol gol pazarları ve maç sonuçları için Dixon-Coles ve Poisson matematiksel olasılık modelleriyle çalışan bir **Paper-Betting, Sanal Kasa Simülasyonu ve Portföy Yönetim Platformudur**.

- ⚠️ **BETAVUS Asla Bir Bahis Sitesi Değildir**: Gerçek para yatırma/çekme yoktur; kullanıcıların risksiz ortamda bahis ve kasa stratejilerini test etmesini sağlar.
- **Odak Pazarlar**: Toplam Gol (0.5Ü, 1.5Ü, 2.5Ü), Çifte Şans (1X, 12, X2) ve Kesin Skorlar (1-0, 1-1, 2-1 vb.).

---

## 2. Son Yapılan Geliştirmeler & Çözülen Sorunlar

### A. Kısıtlı Verilerin (Limited Data) Vurgulardan ve Analizden Tamamen Çıkarılması
- **Problem**: Bazı maçlarda (örneğin *Schalke 04 — Elversberg*) takımlardan birinin o ligde yeterli sezon geçmişi olmaması (`basis: partial-form` veya `league-avg`) yüksek istatistiksel belirsizlik oluşturmasına rağmen sistem bu maçı yeşil vurgu ve `★` yıldız işaretiyle vurgulanan analiz listesine dahil ediyordu.
- **Çözüm**:
  - `index.html` içerisindeki `edgeLabel(m)` ve `dirMark(p, h, line, m)` fonksiyonlarında `isLimitedData(m)` kontrolü zorunlu kılındı. Kısıtlı maçlarda `★` yıldızı ve yeşil vurgu halkası (`box-shadow`) engellendi; yerine `⚠️ Kısıtlı Veri (Vurgusuz)` rozeti konuldu.
  - `aggr()`, `calcMarketStats()`, `hiMatches` ve `hiMissRows()` hesaplamalarında `!isLimitedData(m)` filtresi eklenerek kısıtlı veriler "⚡ sadece vurgulanan maçlar" listesinden ve başarı oranı paydalarından çıkarıldı.
  - `js/paper_engine.js` içerisindeki `isEligibleMatch()` güçlendirildi, kısıtlı verili maçların otomatik kuponlara seçilmesi engellendi.

### B. Çifte Şans & Skor Model Doğruluğu (16.478 Maç Tabanı)
- **Problem**: Ana Model Doğruluğu (`#pane-bt`) 5 tamamlanmış sezonun **16.478 maçını** baz alırken, Çifte Şans başlangıçta 17.003 maç (devam eden 2026/27 maçları dahil) gösteriyordu.
- **Çözüm**:
  - `scripts/build_cifte_backtest.py` `LeagueModel` walk-forward mekanizmasıyla çalıştırılarak devam eden sezon elendi ve tam **16.478 maça** eşitlendi.
  - `js/cifte_backtest_data.js` güncellendi.
  - Başarı oranları (kısıtlı veriler hariç yüksek güven bölgesi):
    - **1X Çifte Şans (≥%75)**: **3.459 / 4.156 maç (%83.2)**
    - **12 Çifte Şans (≥%75)**: **4.462 / 5.761 maç (%77.5)**
    - **X2 Çifte Şans (≥%75)**: **604 / 748 maç (%80.7)**
    - **Skor Tahmin Havuzu (Top-3)**: **5.165 / 16.478 maç (%31.3)** *(Top-1 Tam Skor: %12.2)*

### C. "Iska / Tutmadı" Göstergelerinin Kaldırılması ve Sade UI
- Kullanıcının açık talimatı doğrultusunda:
  - Çifte Şans KPI kartlarındaki kırmızı "Iska / Tutmadı" kutuları ve sayaçları kaldırıldı.
  - Her kartta sadece: **5 Sezonluk Maç (16.478)**, **Sistem Vurguladı** ve **Tahmin Tuttu ✓** kutuları bırakıldı.
  - Tablolardaki 13 sıkışık sütun elenerek **6 sütunlu ferah ve modern lig/sezon tablosu** oluşturuldu.
  - Örnek maçlar tablosunda "Iska" kelimeleri temizlendi, sadece tutan tercihler (`✓ Tuttu`, `🎯 Tam Skor`) vurgulandı.

### D. Kasa & Kupon Simülasyonu Geliştirmeleri (Önceki Adımlarda Tamamlananlar)
- **Çoklu Kasa (Multi-Bankroll)**: Kullanıcı aynı anda farklı bütçelerle Minimum, Orta, Yüksek risk veya Özel kasa açabilir.
- **Seçili Riske Özel Görünüm**: Kasa seçildikten sonra ekranda karmaşa yaratmamak için sadece o riske ait kuponlar ve grafikler listelenir.
- **Hiç Maç Kaybetmeme (Sıfır Kayıp Eğrisi)**: 30 günlük geçmiş simülasyonunda ve kasa planında altın sarısı kesikli çizgi (`#fbbf24`) ile maksimum potansiyel eğrisi ve metrikleri çizildi. Çakışan turuncu çizgi kaldırıldı.
- "Excel Modeli Uyumlu" vb. ifadeler arayüzden tamamen temizlendi.

---

## 3. Mimari ve Dosya Haritası

```
Football-TG05-main/
│
├── index.html                     # Ana tek sayfa uygulaması (SPA)
│                                  # Tüm sekmeler, CSS değişkenleri, modal bileşenleri,
│                                  # filtreleme ve render döngüsü burada barınır.
│
├── js/
│   ├── cifte_engine.js            # Bivariate Poisson & Dixon-Coles olasılık motoru.
│   │                              # 1-X-2, 1X/12/X2, Top-6 skor, KG Var/Yok üretir.
│   │
│   ├── cifte_ui.js                # '🎲 Çifte Şans & Skor' sekmesinin bülten ve
│   │                              # Model Doğruluğu (16.478 maç) arayüz yönetimi.
│   │
│   ├── cifte_backtest_data.js     # 16.478 maçlık 5 sezonluk walk-forward backtest JSON verisi.
│   │
│   ├── paper_engine.js            # Sanal kasa, kupon oluşturma, risk profilleri (Min, Orta, Yüksek)
│   │                              # ve 30 günlük simülasyon hesaplama motoru.
│   │
│   ├── paper_ui.js                # Gerçek Kasa, Gerçek Kuponlarım, Örnek Kasa Simülasyonu,
│   │                              # SVG grafik çizimleri ve modal pencereleri.
│   │
│   └── backtest_data.js           # 0.5Ü, 1.5Ü, 2.5Ü gol pazarları backtest verisi.
│
├── data/
│   ├── results.json               # 17.003 maçlık geçmiş maç ve canlı skor veri tabanı.
│   ├── backtest.json              # Gol pazarları 5 sezonluk doğrulanmış verisi.
│   └── football-data/             # 9 ligin 8 sezonluk resmi CSV arşivleri.
│
├── scripts/
│   ├── build_cifte_backtest.py    # Çifte Şans backtestini 16.478 maçla üreten betik.
│   ├── verify_site.py             # Playwright ile tüm sekmeleri ve regresyonları test eden ana test.
│   ├── test_cifte_backtest.py     # Kısıtlı veri, 16.478 maç ve sıfır iska kontrol test paketi.
│   ├── goals_model.py             # Temel lig ve gol tahmin motoru (LeagueModel).
│   └── build_results.py           # results.json üretim ve sonuç bağlama betiği.
│
└── HANDOVER.md                    # Bu devir dokümanı.
```

---

## 4. Kritik İş Kuralları & Sözleşmeler (Değiştirilmemesi Gereken Kurallar)

1. **Kısıtlı Veri Kuralı (`isLimitedData`)**:
   - `partial-form` veya `league-avg` dayanağına sahip maçlar (H2H sayısı 2'den az ise) **kesinlikle kısıtlı veridir**.
   - Kısıtlı verili maçlara **ASLA `★` yıldız verilmez, yeşil çerçeve eklenmez, vurgulanan maç sayısına ve başarı oranına katılmaz, otomatik kuponlara seçilmez**.
   - Arayüzde `⚠️ Kısıtlı Veri (Vurgusuz)` şeklinde nötr etiketlenir.

2. **Backtest Maç Sayısı (Tutarlılık)**:
   - Model Doğruluğu sekmelerinde referans taban **16.478 maçtır** (5 tamamlanmış sezon: `2021/22`, `2022/23`, `2023/24`, `2024/25`, `2025/26`).
   - Devam eden 2026/27 sezonunun 525 maçı backtest havuzuna sokulmaz; sadece "Güncel Maçlar" veya "Tüm Sezonlar" görünümünde yer alır.

3. **Iska / Tutmadı Gösterimi**:
   - Kullanıcı negatif odaklı kırmızı "Iska" kutularını istememektedir.
   - Odak her zaman: **Toplam Maç**, **Sistem Vurguladı**, **Tahmin Tuttu ✓** ve **Başarı Oranı %** olmalıdır.

4. **Tasarım & UI Dili**:
   - Tema koyu mod (`--bg: #090b0f`, `--panel: #11151c`, `--accent: #e8ff3f`).
   - Tablolarda 13 sıkışık sütun yerine 6 sütunlu, ferah, büyük yüzdeli ve altında maç sayısını gösteren modern tasarım korunmalıdır.

---

## 5. Doğrulama ve Test Komutları

Claude Code ile çalışırken yapılan değişiklikleri test etmek için hazır betikler:

```bash
# 1. Ana site regresyon ve tüm sekmeler testi (Headless Playwright)
python scripts/verify_site.py

# 2. Kısıtlı veri kontrolü, 16.478 maç ve Çifte Şans UI testi
python scripts/test_cifte_backtest.py

# 3. Çifte Şans Backtest verisini yeniden üretmek gerekirse
python scripts/build_cifte_backtest.py
```

---

## 6. Sırada Bekleyen / Gelecek Görevler (Next Steps)

1. **Tahminlerim ile Çifte Şans Sekmesinin Birleştirilmesi**:
   - Kullanıcı: *"Tahminlerim sekmesinin yanında şimdilik ayrı bir sekmede olsun. Ne zaman ki biz bu tahminler kısmını da finalledik, akabinde ileride merge edebiliriz."*
   - İleride Çifte Şans ve Skor kolonları `Tahminler (Bülten)` tablosuna opsiyonel sütun veya genişletilebilir accordion olarak dahil edilebilir.
2. **Canlı Fikstür ve Oran Entegrasyonu**:
   - `cifte_engine.js` hazır durumdadır; gelen bülten maçlarında 1X, 12, X2 piyasa oranları otomatik çekilip beklenen değer (Edge) hesabı yapılabilir.
3. **Kullanıcı Geri Bildirimleri**:
   - Kullanıcı arayüzü test edip ek geri bildirim verdiğinde yukarıdaki mimari sözleşmeler gözetilerek ilerlenmelidir.
