# BETAVUS PROJE HANDOVER (DEVİR & TESLİM) RAPORU
**Tarih:** 23 Eylül 2026  
**Hazırlayan:** Antigravity AI  
**Devralan:** Claude Code / Geliştirici Ekibi  
**Depo (Git):** `https://github.com/GKHNKMR/Football-TG05.git` (Dal: `main`)

---

## 📌 1. PROJE GENEL BAKIŞI & MİMARİ

**BETAVUS**, Avrupa ve Türkiye liglerindeki binlerce maçı matematiksel ve istatistiksel modellerle (Poisson & Dixon-Coles) analiz eden; toplam gol, çifte şans ve skor tahminleri üreten, aynı zamanda kullanıcıya risksiz sanal kasa simülasyonu sunan gelişmiş bir **Paper-Betting & Karar Destek Platformudur**.

### Temel Teknolojiler & Prensipler:
- **Zero-Dependency Vanilla JS:** Harici ağır kütüphane (React/Vue vb.) olmadan, tarayıcıda ultra hızlı çalışan vanilla JavaScript mimarisi.
- **İnteraktif Görselleştirme:** D3.js v7 tabanlı interaktif zihin haritası, Canvas / SVG tabanlı kasa ve oran grafikleri.
- **Tarihsel Veri Seti:** `results.json` içerisinde **17.003 maçlık** kapsamlı tarihsel maç ve sonuç havuzu.
- **Gerçekçi Backtest:** 5 tamamlanmış sezonu kapsayan **16.478 maçlık** sızıntısız (walk-forward) model doğrulama altyapısı.
- **Veri Güvenliği & Saklama:** Kasa kayıtları, kuponlar ve kullanıcı tercihleri tarayıcı `localStorage` üzerinde izole tutulur.

---

## 🚀 2. SON GELİŞTİRMELER & YAPILAN İŞLER

Son dönemde kullanıcı talepleri doğrultusunda tamamlanan kritik geliştirmeler:

### 1. Çifte Şans ve Skor Tahminleri Sekmesi (`cifte-sans-tab`)
- **Konum:** `Tahminlerim` sekmesinin yanında, bağımsız yeni bir sekme olarak eklendi.
- **Modelleme:** Dixon-Coles 10x10 bivariate olasılık matrisi kullanılarak **1X (% Ev Sahibi + Beraberlik)**, **12 (% Ev Sahibi + Deplasman)** ve **X2 (% Beraberlik + Deplasman)** çifte şans ihtimalleri hesaplanır.
- **Top-3 Olası Skor:** Maçın en yüksek olasılıklı 3 kesin skoru (örneğin `1-1 (%13.2)`, `2-1 (%11.5)`) sunulur.
- **Kısıtlı Veri Koruması:** Sezonda 5 maçtan az verisi olan takımların maçları **`⚠️ Kısıtlı Veri`** rozetiyle işaretlenir ve yanıltıcı yüksek oran verilmesi engellenir.

### 2. Çifte Şans & Skor Model Doğruluğu Ekranı (`model-dogruluk-tab`)
- **Temiz & Ferah 6 Sütunlu Tasarım:** Kalabalık ve karmaşık kartlar yerine; *Lig*, *Toplam Maç*, *1X Doğruluk*, *12 Doğruluk*, *X2 Doğruluk* ve *Skor Top-3 Başarısı* sütunlarından oluşan sade tablo.
- **Sıfır Iska / Odaklanmış Başarı:** Kullanıcı odağını dağıtmamak adına "Iskalar" (tutmayanlar) gizlendi; yalnızca başarı oranları, tutan maç sayıları ve yüksek güvenli vurgular sergilendi.
- **16.478 Maçlık Filtreli Havuz:** Devam eden / kısıtlı verili mevcut sezon maçları filtrelenerek 5 tam sezonluk sızıntısız veri tabana oturtuldu (Başarı oranları: 1X %83.2, 12 %77.5, X2 %80.7, Skor Top-3 %31.3).

### 3. Çoklu Kasa (Multi-Bankroll) & Dinamik Kasa Yönetimi
- **Birden Fazla Kasa:** Kullanıcı istediği sayıda kasa açabilir (Örn: "50€ Minimum Risk", "100€ Yüksek Risk").
- **Seçilen Kasa Odaklı Ekran:** Kasa seçildikten sonra ekranda yalnızca o kasanın riski ve verileri görünür; karmaşa tamamen ortadan kaldırıldı.
- **Sıfır Kayıp / Hiç Maç Kaybetmeme Altın Eğrisi (`#fbbf24`):** İki kesik çizginin karışmaması için gerçek kasa simülasyonunda sıfır kayıp potansiyeli parlak altın sarısı renkle ayrıştırıldı.

### 4. BETAVUS İnteraktif Zihin Haritası (`BETAVUS_Dinamik_Zihin_Haritasi.html`)
- `Trading_Sistemi_Mindmap.html` standartlarında, D3.js v7 ile sıfırdan inşa edildi.
- 6 ana sütun, 18 alt modül, tıklanabilir katlanabilir düğümler, canlı terim arama (pulsing highlight), formülleri gösteren sağ detay paneli (drawer) ve PDF/Yazdır çıktısı eklendi.
- Playwright ile doğrulanarak repoya push edildi.

---

## 📂 3. DOSYA YAPISI & ÇALIŞMA ALANLARI

Proje çalışma dosyaları iki ana dizinde yer almaktadır:

```
c:\Users\mtem01\OneDrive\VS-Studio Working File\
├── BETAVUS_HANDOVER.md                    <-- Bu Devir Teslim Dokümanı
├── BETAVUS_Dinamik_Zihin_Haritasi.html    <-- İnteraktif D3.js Zihin Haritası
├── Trading_Sistemi_Mindmap.html           <-- Referans Mindmap Şablonu
└── Football Goal Analyst/
    └── Football-TG05-main/                <-- Ana Git Proje Deposu
        ├── index.html                     <-- Platform Ana Sayfası & UI
        ├── js/
        │   ├── app.js                     <-- Sekme yönetimi, modal ve genel entegrasyon
        │   ├── calculations.js            <-- Poisson, Dixon-Coles, Olasılık hesapları
        │   ├── predictions.js             <-- Tahmin motoru & Çifte şans üretimi
        │   ├── bankroll.js                <-- Çoklu kasa yönetimi & simülasyon eğrileri
        │   ├── modelAccuracy.js           <-- Model doğruluk analizleri ve tabloları
        │   └── couponEngine.js            <-- 1H / 1A akıllı kupon öneri motoru
        ├── data/
        │   └── results.json               <-- 17.003 maçlık tarihsel maç sonuçları
        ├── BETAVUS_Dinamik_Zihin_Haritasi.html
        └── BETAVUS_HANDOVER.md
```

---

## ⚙️ 4. KRİTİK İŞ KURALLARI & DİKKAT EDİLECEK HUSUSLAR

Claude Code ile çalışırken aşağıdaki kural ve felsefelere sadık kalınmalıdır:

1. **Kısıtlı Veri İzolasyonu:**
   - Yeni başlayan sezonlarda veya 5 maçtan az veriye sahip takımlarda matematiksel varyans yüksektir.
   - Bu maçlar ana güvenilir model analizlerine dahil edilmemeli, kullanıcıya sarı uyarı ikonu ile bildirilmelidir.
2. **Kasa / Risk Profili Ayrımı:**
   - Minimum Risk: %30 kasa payı (Çok yüksek güvenli maçlar).
   - Orta Risk: %15 kasa payı.
   - Yüksek Risk: %5 kasa payı (Daha yüksek oran/sürpriz).
   - Kullanıcı dilediğinde kendi risk yüzdesini belirleyebilir.
3. **Doğruluk Ekranında Ferahlık:**
   - İstatistikler sunulurken "tutmayan / iska" kutucukları ile negatif görsel karmaşa yaratılmamalıdır.
   - Başarı oranları, tutan maç sayıları ve yüksek güvenli filtreler ön planda olmalıdır.

---

## 🎯 5. CLAUDE CODE İÇİN SIRADAKİ ADIMLAR (YOL HARİTASI)

1. **Tahminlerim ile Çifte Şans Sekmelerinin Birleştirilmesi (Merge):**
   - Kullanıcı daha önce belirtti: *"Ne zamanki biz bu tahminler kismini da finalledik, akabinde ileride merge edebiliriz."*
   - Çifte Şans ve Skor tahminleri nihai onay aldığında, `Tahminlerim` kart yapısına entegre edilebilir veya yan yana kompakt görünüme kavuşturulabilir.
2. **Canlı Fikstür / Yeni Bülten Otomatik Çekici:**
   - API veya web scraping üzerinden güncel haftalık bültenin `results.json` formatıyla senkronize edilmesi.
3. **Mobil Uyumluluk & PWA:**
   - Çoklu kasa ve kupon düzenleme modalının küçük ekranlarda dokunmatik testlerinin optimize edilmesi.

---
*Bu doküman, BETAVUS projesinin tüm geçmişini, mimarisini ve hedeflerini Claude Code oturumuna eksiksiz aktarmak üzere hazırlanmıştır.*
