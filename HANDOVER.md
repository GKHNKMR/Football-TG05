# BETAVUS — Proje Devir Dokümanı (Claude Code İçin)

> **Güncelleme tarihi:** 23 Eylül 2026
> **Repository:** `https://github.com/GKHNKMR/Football-TG05.git`
> **Aktif dal:** `main`
> **Canlı site:** `https://betavus.vercel.app`
> **Çalışma dizini:** her geliştiricinin kendi klonu, `C:\Users\<kullanıcı>\source\Football-TG05` (OneDrive dışında)

Bu dosya projenin **kanonik handover kaynağıdır**. `BETAVUS_HANDOVER.md` yalnızca bu dosyaya yönlendirme amacıyla tutulur. Başlamadan önce `git status --short` ve `git log -5 --oneline` çalıştırılmalıdır.

## 1. Ürün ve mevcut kapsam

BETAVUS; Poisson ve Dixon-Coles modelleriyle futbol toplam gol ve çifte şans olasılıkları üreten, gerçek para kabul etmeyen bir paper-betting / sanal kasa / portföy yönetimi uygulamasıdır.

Güncel kullanıcı arayüzü kararları:

- `Tahminler (Bülten)` ekranı 0.5Ü, 1.5Ü, 2.5Ü ve Çifte Şans sonuçlarını birlikte gösterir.
- Çifte Şans yeşil yanıp-sönme vurgusu yalnızca genel `Vurgu: açık` filtresi açıkken çalışır.
- `Çifte Şans & Gol Aralığı` sekmesi artık yalnızca 5 sezonluk model doğrulama ekranıdır. Bu sekmedeki eski `Güncel Fikstür & Tahminler` alt görünümü kaldırılmıştır.
- Kesin skor tahmini kullanıcı arayüzünden çıkarılmış, yerine `2–3 Gol`, `3–4 Gol`, `5+ Gol` toplam gol aralıkları getirilmiştir.
- Gol aralığı yüzdesi **geçmiş başarı yüzdesi değil**, maç öncesi `Model Güveni`dir. Gerçekleşen başarı ayrı olarak `✓ Aralık tuttu` ile gösterilir.
- Çifte şans kodları açık yazılır: `1X (Ev/Beraberlik)`, `12 (Beraberlik Yok)`, `X2 (Beraberlik/Deplasman)`.
- Lig filtresinde seçilen ligde maç yoksa başka lig maçları geri getirilmez.

## 2. Değiştirilmemesi gereken mimari kurallar

### 2.1 Kısıtlı veri izolasyonu

Bir maç aşağıdaki koşulda kısıtlı veridir:

```text
(basis = partial-form veya league-avg) ve H2H maç sayısı < 2
```

Kısıtlı maçlar:

- yıldız, yeşil çerçeve veya yanıp-sönme vurgusu alamaz;
- başarı oranı pay/paydasına ve “sadece vurgulananlar” listesine giremez;
- otomatik kupona seçilemez;
- nötr `⚠️ Kısıtlı Veri (Vurgusuz)` etiketiyle gösterilir.

### 2.2 Backtest kapsamı

- Referans havuz 5 tamamlanmış sezondaki **16.478 maçtır**: `2021/22`–`2025/26`.
- Devam eden `2026/27` sezonu doğrulama havuzuna sokulmaz.
- Kısıtlı veriler gol aralığı/çifte şans başarı hesabından çıkarıldıktan sonra gol aralığı değerlendirme paydası **13.828 maçtır**.
- Walk-forward model hedef sezonu görmez; her sezon yalnızca önceki sezonlarla eğitilir.

### 2.3 Vurgu dili

- Çifte Şans yeşil vurgu eşiği **≥%75** ve tam veri şartıdır.
- Kullanıcı kırmızı `Iska` / `Tutmadı` kutuları istemiyor. Arayüzde olumlu başarı sayaçları ve nötr `—` kullanılır.
- “Vurgu” garanti veya kesin maç anlamına gelmez; kullanıcıya bu şekilde sunulmamalıdır.
- Gol aralığı için henüz yeşil yüksek-güven vurgusu uygulanmamıştır. Aşağıdaki `%47` kararı analiz/öneri aşamasındadır.

### 2.4 Gol aralığı anlamı

- `2–3 Gol` olasılığı `P(toplam=2) + P(toplam=3)` olarak hesaplanır.
- `3–4 Gol` olasılığı `P(toplam=3) + P(toplam=4)` olarak hesaplanır.
- `5+ Gol`, toplam golün en az 5 olmasıdır.
- Bantlar birbirini dışlamaz: 3 gol hem `2–3` hem `3–4` bandındadır. Model en yüksek olasılıklı bandı seçer.
- Dar `2–3` bandının Poisson altında teorik maksimum olasılığı yaklaşık `%47`dir. Bu nedenle gol aralığı yüzdesi Çifte Şans gibi `%75` seviyesine çıkamaz.

## 3. Doğrulanmış performans değerleri

Kısıtlı veri hariç, 5 sezonluk walk-forward sonuçları:

| Pazar | Tuttu / Vurgulanan | Başarı |
|---|---:|---:|
| 1X (≥%75) | 3.459 / 4.156 | %83,2 |
| 12 (≥%75) | 4.462 / 5.761 | %77,5 |
| X2 (≥%75) | 604 / 748 | %80,7 |
| Tüm gol aralığı tercihleri | 6.377 / 13.828 | %46,1 |
| 2–3 Gol | 6.113 / 13.182 | %46,4 |
| 3–4 Gol | 253 / 602 | %42,0 |
| 5+ Gol | 11 / 44 | %25,0 |

Örnek tabloda gösterilen ilk 12 maçta 9 isabet (`%75`) vardır. Bu sadece ekrandaki küçük örnektir; ana başarı oranı 13.828 maçtan hesaplanır. İlk 12 maç zaten ana havuzun içindedir.

### Gol aralığı vurgu eşiği analizi — henüz uygulanmadı

| Model güveni eşiği | Maç | Gerçek isabet | Kapsama |
|---|---:|---:|---:|
| ≥%45 | 10.012 | %46,61 | %72,40 |
| ≥%46 | 8.040 | %46,98 | %58,14 |
| ≥%47 | 4.008 | %47,48 | %28,98 |
| ≥%48 | 608 | %45,56 | %4,40 |
| ≥%50 | 0 | — | %0 |

Önerilen ürün kuralı:

```text
Sarı “Güçlü Gol Aralığı”:
  - yalnızca 2–3 gol tercihi
  - model güveni ≥ %47
  - tam veri
  - kritik eksik oyuncu/veri uyarısı yok

Yeşil yanıp-sönen “Yüksek Güven”:
  - gol aralığında şimdilik kapalı
```

Sebep: `%47+` grubunun gerçek isabeti yalnızca `%47,48`dir. Bunu yeşil “yüksek güven/kesin” olarak sunmak yanıltıcı olur. `%48` eşiği daha iyi değildir ve veri hacmi belirgin şekilde düşer. Claude bu kuralı uygulamadan önce bunun **konuşulmuş öneri, henüz kodlanmamış değişiklik** olduğunu bilmelidir.

## 4. Brighton — Manchester United örneği

`24.05.2026 Brighton & Hove Albion — Manchester United` doğrulama satırı kullanıcıyla ayrıntılı incelendi:

- Model: Brighton `λ=1,663`, Manchester United `λ=1,241`, toplam `λ=2,904`.
- Takım/xG toplamı H2H öncesi yaklaşık `2,914`.
- Son 8 H2H toplam gol ortalaması `2,875`; model `%72` takım bileşeni + `%28` H2H ile toplamı yaklaşık `2,903` yaptı.
- Toplam gol dağılımı: 0 `%5,48`, 1 `%15,91`, 2 `%23,11`, 3 `%22,37`, 4 `%16,24`, 5+ `%16,89`.
- `P(2 veya 3) = %23,11 + %22,37 = %45,48`, ekranda `%45,5`.
- Gerçek skor `0-3`, toplam 3 gol; bu nedenle `✓ Aralık tuttu`.
- `%45,5` maç öncesi olasılıktır; sonuç gerçekleştikten sonra geriye dönük `%100` yapılmaz.

## 5. Dosya haritası

```text
index.html
  Ana SPA, tema/CSS, sekmeler, lig filtresi ve Tahminler bülteni.

js/cifte_engine.js
  Dixon-Coles skor matrisi; 1X/12/X2 ve 2–3/3–4/5+ olasılıkları.

js/cifte_ui.js
  Çifte Şans & Gol Aralığı model doğrulama ekranı ve örnek tablolar.

js/cifte_backtest_data.js
  16.478 maçlık doğrulama özetleri ve 250 örnek maç.

scripts/build_cifte_backtest.py
  Walk-forward Çifte Şans/Gol Aralığı doğrulama üreticisi.

scripts/goals_model.py
  LeagueModel, xG/form/H2H harmanı ve Dixon-Coles düzeltmesi.

scripts/test_cifte_tab.py
  Tahminler bülteni, lig filtresi ve Çifte Şans vurgu regresyonu.

scripts/test_cifte_backtest.py
  16.478 maç, gol aralığı, örnek/genel kapsam ve mobil okunabilirlik testi.

scripts/verify_site.py
  Tüm ana sekmeler için genel Playwright regresyonu.

BETAVUS_Dinamik_Zihin_Haritasi.html
  D3.js mimari zihin haritası (commit c2df6ab).
```

`scratch/` test ekran görüntüleri içerir ve git'e eklenmemelidir.

## 6. Son önemli commitler

```text
c1a1e74  Çifte şans kodlarını ve model güvenini açıkla
75eeb19  Gol aralığı yüzdesini Model Güveni olarak etiketle
633d4a3  Lig filtresini ve gol aralığı kapsam açıklamasını düzelt
cdc0025  Çifte Şans sekmesini yalnızca doğrulama ekranına indirgeme
3e760ea  Kesin skor tahminini toplam gol aralığıyla değiştirme
b0e0f54  Çifte Şans vurgusunu genel vurgu filtresine bağlama
1658697  Tahminler bültenine Çifte Şans ekleme
```

Sonrasında `c2df6ab` ile zihin haritası, `b12a68d` ile ikinci handover dosyası eklenmiştir. Bu güncelleme iki handover dosyası arasındaki çelişkiyi kaldırır.

## 7. Test ve dağıtım

```powershell
python scripts/run_tests.py   # test_sim_math, test_paper_betting, test_cifte_tab, test_cifte_backtest, verify_site
git diff --check
```

Son doğrulanan durum:

- Çifte Şans/Gol Aralığı masaüstü ve mobil testleri geçti.
- Lig filtresi regresyon testi geçti.
- Genel site regresyonu geçti.
- Vercel dağıtımı GitHub `main` push'u sonrasında otomatik çalışır.
- Canlı dosya önbellek anahtarı: `cifte_ui.js?v=20260922-dc-confidence-label`.

## 8. Claude için başlangıç sırası

1. Bu `HANDOVER.md` dosyasını tamamen oku.
   Ardından `BETAVUS_Is_Listesi.xlsx` iş listesini oku ve `CLAUDE.md`'deki iş takibi kurallarını uygula (açık işleri kullanıcıya sor, biten işi kapat).
2. `git status --short` ile kullanıcıya ait değişiklikleri koru.
3. `git log -5 --oneline` ile başlangıç commit'ini doğrula.
4. Gol aralığı vurgusu istenirse Bölüm 3'teki `%47 sarı / yeşil kapalı` kararını temel al; bunu Çifte Şans `%75` kuralıyla karıştırma.
5. Değişiklikten sonra üç test paketini çalıştır; `scratch/` klasörünü commit etme.
6. Başarılı commit/push sonrası Vercel durumunu ve canlı `https://betavus.vercel.app` alan adını doğrula.

## 9. Güvenlik notu

- Repository veya handover belgelerine erişim anahtarı, token, parola ya da `.env` içeriği yazılmamalıdır.
- Önceki `HANDOVER.md` sürümünde düz metin bir erişim anahtarı bulunuyordu. Bu güncel dosyadan kaldırıldı; ancak git geçmişinde kalabileceği için ilgili anahtar **iptal edilmeli/döndürülmelidir**.
- Git geçmişi kullanıcı onayı olmadan yeniden yazılmamalıdır.

## 10. 23 Eylül 2026 — Yeniden tasarım + üyelik (Claude)

- **Menü:** yalnızca 3 ana sekme: `Bülten` (pred, varsayılan) · `İstatistikler` (stats, yeni) · `Sanal Kasa` (plan, eski "Gerçek Kasa"). Diğer sekmeler `hidden` butonlarla DOM'da duruyor, kodları silinmedi; `setTab('res')` vb. ile açılabilir.
- **Tasarım:** açık tema varsayılan (`betavus.theme2`), thepunterspage.com örnek alındı: lacivert üst bar `#203342`, turkuaz vurgu `#009f93`, Nunito Sans + Poppins. "BETAVUS nedir / ne değildir" kutuları sayfanın en altında.
- **Bülten:** 0.5+ · 1.5+ · 2.5+ · 1X · 12 · X2 olasılıkları ayrı sütunlar, gün başlıklı liste. Vurgu: 0.5≥%95, 1.5≥%85, 2.5≥%75, 1X/12/X2≥%75 (her ÇŞ pazarı ayrı; kısıtlı veri / kritik eksik oyuncu vurgulanmaz). Gol aralığı gösterilmez. Üstte `data/stats-summary.json`'dan vurgulu tahmin başarı şeridi.
- **İstatistikler:** `js/stats_ui.js` + `data/stats-5season.json` (16.478 maç, eskiden yeniye, 6 pazar olasılığı + skor). Vurgulanan başarı, genel yön isabeti, kalibrasyon (güven analizi), lig tablosu, sayfalı maç listesi. Her ikisi `scripts/build_cifte_backtest.py` ile üretilir (tamamlanmış sezonlar; workflow'da çalışmaz, gerek yok).
- **Üyelik:** Supabase (`js/auth_config.js` boşken devre dışı), `js/auth_sync.js`, `supabase/schema.sql`, kurulum: `supabase/KURULUM.md`. Tüm `betavus.*` localStorage anahtarları hesapla senkronlanır; ilk senkronda Sanal Kasa kimliklere göre birleştirilir. Erişim kodu kapısı korunuyor.
- **Önceki durum:** 21 Eylül'den beri workflow "Validate predictions" adımında düşüyor çünkü milli ara nedeniyle 10 günlük pencerede fikstür yok (ilk maçlar 9–10 Ekim); bu sırada sonuç notlama commit'leri de atlanıyor.

### 10.1 Aynı gün, kullanıcı geri bildirimleriyle ikinci tur
- **Vurgu eşikleri:** 0.5+ ≥%95 · 1.5+ ≥%85 · **2.5+ ≥%80 · 1X/12/X2 ≥%80** (önce %75'ti). Bülten vurgusu, İstatistikler ve ana sayfa şeridi aynı eşikleri kullanır. Gizli eski sekmeler (res/bt/cpn/cifte) hâlâ %75 kullanır.
- **İstatistikler listesi:** yalnızca vurgulu maçlar; Tüm / Kazanan / Kaybeden filtreleri; varsayılan sıra yeniden eskiye, Tarih başlığına tıklayınca tersine döner; hücre yeşil = vurgu tuttu, kırmızı = tutmadı; Vurgu sütunu "Tuttu" (yeşil) / "Kaybetti" (kırmızı). Kullanıcı kırmızıyı bu ekranda açıkça istedi (2.3'teki "kırmızı yok" kuralı bu liste için geçersiz). Maç hücresinde tahmini toplam gol (λ), `stats-5season.json` alanı `lambda`.
- **Bülten:** 1 aylık pencere (`FORECAST_DAYS = 30`, istemcide +30 gün). Lig filtresi arama kutusunun yanında açılır liste (`#lgPred`, İstatistikler'de `#stLeague`); `#filters` çipleri bu iki sekmede gizli. Şerit her pazar için tuttu/toplam gösterir.
- **FAQ sekmesi:** eski "?" butonunun içeriği ve "BETAVUS nedir / ne değildir" kutuları burada birleşti; "?" butonu ve alt bilgi kutuları kaldırıldı.
