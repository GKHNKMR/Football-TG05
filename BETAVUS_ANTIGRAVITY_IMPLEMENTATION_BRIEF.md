# BETAVUS — Antigravity Uygulama Şartnamesi

## 1. Görev

Mevcut BETAVUS sitesini aşağıdaki ürün tanımına göre güncelle.

BETAVUS bir bahis sitesi, bahis operatörü veya ödeme platformu değildir. Site bahis kabul etmez, para yatırma/çekme işlemi yapmaz ve kullanıcı adına harici bir bahis sitesinde kupon oynatmaz.

BETAVUS; futbol toplam gol pazarları için yapay zekâ destekli bir **paper-betting, kupon planlama ve sanal kasa yönetimi platformudur**.

Kullanıcı:

1. Sanal başlangıç kasasını, hedef kasasını, plan süresini ve risk toleransını tanımlar.
2. Risk profiline uygun kupon önerileri alır.
3. Önerilen kuponu aynen kullanabilir, maç çıkarabilir veya uygun maç ekleyebilir.
4. Kuponun tahmini oranını görür.
5. Kuponu harici bir bahis sitesinde oynadıysa, tahmini oranı aldığı gerçek toplam oranla değiştirebilir.
6. Maçlar tamamlanınca BETAVUS kuponu otomatik sonuçlandırır.
7. Sanal kasa hareketini, hedefe ulaşma olasılığını ve risk durumunu görür.
8. Hedef yolundan saparsa süreyi uzatma, hedefi değiştirme veya kontrollü biçimde risk profilini değiştirme alternatiflerini karşılaştırır.

Ana ürün mesajı:

> Önce simüle et. Riskini gör. Stratejini ölç. Sonra karar ver.

## 2. Değiştirilemez ürün ilkeleri

- Uygulamanın her yerinde bunun bir **paper-betting / sanal kasa simülasyonu** olduğu açıkça belirtilmelidir.
- Gerçek para bakiyesi, ödeme, para yatırma, para çekme veya BETAVUS içinde bahis oynama akışı eklenmemelidir.
- Kesin kazanç, garanti getiri veya “şu kadar kazanacaksınız” dili kullanılmamalıdır.
- Kasa tahminleri tek bir kesin sonuç olarak değil, olasılıklı senaryolar olarak sunulmalıdır.
- Kullanıcı bir kayıp yaşadığı için stake otomatik artırılmamalıdır.
- Martingale, kaybı kovalama veya otomatik bahis büyütme mantığı eklenmemelidir.
- Risk seviyesi sistem tarafından otomatik değiştirilmemeli; yalnızca seçenek olarak gösterilmeli ve kullanıcı onayı gerektirmelidir.
- Mevcut tahmin modeli, sonuç analizi, veri üretim hattı ve çalışan ekranlar korunmalıdır.
- Mevcut modelin kalibrasyon iddiası korunmalı; isabet veya kâr garantisi gibi sunulmamalıdır.

Uygulamada görünür biçimde şu açıklama bulunmalıdır:

> BETAVUS bahis kabul etmez, ödeme almaz ve kupon oynatmaz. Gösterilen kasa, stake ve getiriler sanaldır. Tahminler olasılıksaldır ve sonuç garantisi vermez.

## 3. Mevcut teknik yapı

Önce repoyu incele ve mevcut davranışı doğrula. Var olan mimariyi gereksiz yere yeniden yazma.

Mevcut yapının temel özellikleri:

- Uygulama statik ve mobil uyumlu bir tek sayfa uygulamasıdır.
- Ana arayüz `index.html` içindedir.
- Uygulamanın sunucu taraflı kullanıcı hesabı veya veritabanı yoktur.
- Tahminler `predictions.json` dosyasından okunur.
- Sonuçlar ve model performansı `data/results.json`, `data/backtest.json` ve ilgili mevcut JSON dosyalarından okunur.
- Canlı/sonuç verisi mevcut veri hattı tarafından üretilmektedir.
- Kullanıcı tercihleri mevcut `betavus.` önekli `localStorage` yardımcılarıyla saklanmaktadır.
- Mevcut ana sekmeler arasında Tahminler, Tahmin vs Gerçekleşen, Kuponlarım ve Günün Kuponu bulunur.

Bu görev kapsamında:

- Sunucu, kullanıcı hesabı veya harici veritabanı ekleme.
- API anahtarını tarayıcıya koyma.
- Mevcut Python tahmin modelini değiştirme.
- Mevcut JSON veri üretim hattını kırma.
- Sırf mimariyi değiştirmek amacıyla uygulamayı başka bir framework ile baştan yazma.

MVP kullanıcı verilerini mevcut mimariye uygun olarak `localStorage` içinde sakla. Kodun ileride sunucu tabanlı kullanıcı hesaplarına geçirilebilmesi için hesaplama mantığını DOM kodundan mümkün olduğunca ayrı, saf fonksiyonlar halinde tut.

## 4. İstenen bilgi mimarisi

Mevcut sekmeleri koru ve aşağıdaki yapıya getir:

1. **Tahminler** — mevcut ekran ve davranış korunur.
2. **Tahmin vs Gerçekleşen** — mevcut ekran ve davranış korunur.
3. **Kupon Önerileri** — mevcut Günün Kuponu mantığını kişiselleştirilmiş önerilere dönüştürür.
4. **Kuponlarım** — taslak, bekleyen ve sonuçlanan paper kuponlarını gösterir.
5. **Kasa Planım** — plan kurulumu, sanal kasa özeti, hedef ilerlemesi ve senaryo karşılaştırması.

Mevcut “Günün Kuponu” işlevi tamamen silinmemelidir. Uygun olduğu ölçüde “Kupon Önerileri” ekranının başlangıç veri/öneri mantığı olarak yeniden kullanılmalıdır.

## 5. Kasa Planım ekranı

### 5.1 İlk kurulum

Henüz planı olmayan kullanıcıya mobil uyumlu bir kurulum kartı göster.

Zorunlu alanlar:

- Sanal başlangıç kasası
- Hedef kasa
- Plan süresi: 7, 14, 30, 60 veya 90 gün; ayrıca geçerli özel gün sayısı
- Genel risk toleransı: Temkinli, Dengeli veya Agresif
- Para birimi: varsayılan EUR; en az EUR, TRY, USD ve GBP sun

Doğrulamalar:

- Başlangıç kasası sıfırdan büyük olmalı.
- Hedef kasa başlangıç kasasından büyük olmalı.
- Süre pozitif tam sayı olmalı.
- Sayısal alanlarda `NaN`, sonsuz değer veya negatif tutar kabul edilmemeli.

Plan oluşturulduğunda kullanıcıya hedefin garanti olmadığı açıkça gösterilmelidir.

### 5.2 Plan özeti

Plan kurulduktan sonra aşağıdaki kartları göster:

- Kullanılabilir sanal bakiye
- Bekleyen kuponlardaki sanal stake
- Toplam sanal kasa değeri
- Hedef kasa
- Hedefe ilerleme yüzdesi
- Geçen gün / kalan gün
- Başlangıca göre toplam büyüme veya düşüş
- Plan durumu: Hedefin Önünde / Hedef Yolunda / Hedefin Gerisinde
- Hedefe ulaşma tahmini olasılığı
- Simülasyon medyanı
- Kötü senaryo (P10) ve iyi senaryo (P90)
- Tahmini maksimum düşüş veya yarı kasa kaybı olasılığı

“Kasa” kavramının sanal olduğunu kart başlığında veya yakınında tekrar belirt.

### 5.3 Hedef yolunun hesabı

Başlangıç kasası `S`, hedef kasa `T`, toplam gün `D` ise hedef yolunu aşağıdaki geometrik yol ile hesapla:

```text
gerekli_gunluk_oran = (T / S)^(1 / D) - 1
hedef_yolu(gun) = S * (T / S)^(gun / D)
```

Kullanıcının güncel sanal kasasını o gün için hedef yoluyla karşılaştır:

- Hedef yolunun en az %5 üzerindeyse: Hedefin Önünde
- ±%5 içindeyse: Hedef Yolunda
- %5'ten fazla altındaysa: Hedefin Gerisinde

Bu eşikleri merkezi ve kolay değiştirilebilir bir ayar nesnesinde tut.

## 6. Risk yapısı

İki kavramı birbirinden ayır:

1. **Genel risk toleransı:** Kullanıcının toplam sanal kasasının ne kadarının günlük olarak riske açık olabileceğini belirler.
2. **Kupon sınıfı:** Kuponun hangi gol pazarı ve güven eşiğiyle oluşturulduğunu belirler.

### 6.1 Varsayılan genel risk profilleri

Tüm oranları tek bir merkezi `RISK_PROFILES` yapılandırmasında tut. Arayüz metinleri veya hesaplamalar içinde dağınık sihirli sayılar kullanma.

Başlangıç değerleri:

| Genel profil | Kenarda tutulan | Minimum risk kolu | Orta risk kolu | Yüksek risk kolu |
| --- | ---: | ---: | ---: | ---: |
| Temkinli | %90 | %8 | %2 | %0 |
| Dengeli | %75 | %15 | %8 | %2 |
| Agresif | %50 | %30 | %15 | %5 |

Agresif profil, mevcut Excel kasa modelindeki %50 rezerv + %30 / %15 / %5 dağılımını temsil eder. Bunu bütün kullanıcılar için varsayılan yapma. Varsayılan profil **Temkinli** olsun.

Bir gün için yeterli nitelikte kupon üretilemezse ayrılan tutarı zorla başka kola aktarma; sanal kasada rezerv olarak bırak.

### 6.2 Kupon sınıfları

Başlangıç eşikleri merkezi `COUPON_CLASSES` yapılandırmasında tanımlansın:

| Kupon sınıfı | Ana market | Minimum model olasılığı | Azami maç | Fallback tahmini toplam oran |
| --- | --- | ---: | ---: | ---: |
| Minimum Risk | 0,5 Üst | %95 | 5 | 1,25 |
| Orta Risk | 1,5 Üst | %85 | 3 | 1,70 |
| Yüksek Risk | 2,5 Üst | %75 | 3 | 3,25 |

Kurallar:

- Aynı maç bir kupon içinde yalnızca bir kez bulunabilir.
- `league-avg` veya `partial-form` gibi kısıtlı veri dayanakları otomatik öneriye alınmamalı; kullanıcı manuel eklerse uyarı gösterilmelidir.
- Öncelik sırası: `form+h2h`, ardından `form`.
- Eşiği geçen yeterli maç yoksa eşiği sessizce düşürme.
- Kullanıcıya “bugün bu risk sınıfında yeterli güvene sahip kupon yok” mesajı göster.
- Kupon bacakları varsayılan olarak en yüksek olasılıktan düşüğe sıralansın.
- Aynı maçın farklı gol marketlerini aynı kupona eklemeye izin verme.

## 7. Kupon Önerileri ekranı

Kullanıcının risk profiline göre oluşturulan önerileri ayrı kartlar halinde göster.

Her öneri kartında:

- Kupon sınıfı
- Maç sayısı
- Maçlar ve marketler
- Her maçın model olasılığı
- Veri dayanağı (`basis`)
- Tahmini birleşik tutma olasılığı
- Tahmini toplam oran
- Gerçek oran girildiyse gerçek toplam oran
- Önerilen sanal stake
- Kazanırsa tahmini sanal geri dönüş
- Beklenen değer göstergesi
- Açık risk uyarısı
- “Kupona ekle” düğmesi

Kullanıcı öneriyi açarak:

- Maç çıkarabilmeli.
- Tahminler listesinden uygun maç ekleyebilmeli.
- Marketi, mevcut BETAVUS olasılıkları bulunan 0,5 / 1,5 / 2,5 Üst seçenekleri arasında değiştirebilmeli.
- Sanal stake tutarını profil limitleri içinde değiştirebilmeli.
- Kuponu taslak olarak kaydedebilmeli.
- Kuponu plana ekleyebilmeli.

Her değişiklikte toplam olasılık, oran, beklenen geri dönüş ve risk göstergeleri anında yeniden hesaplanmalıdır.

## 8. Oran ve olasılık hesapları

### 8.1 Kupon tutma olasılığı

Farklı maçların bağımsız olduğu MVP varsayımıyla:

```text
kupon_olasiligi = p1 * p2 * ... * pn
```

Arayüzde bunun bağımsızlık varsayımına dayalı yaklaşık değer olduğunu belirt. Aynı maçın birden fazla marketini engellemek bu nedenle zorunludur.

### 8.2 Tahmini oran

Öncelik sırası:

1. İlgili market için mevcut veri içinde güvenilir piyasa oranı varsa onu kullan.
2. Piyasa oranı yoksa bacak bazında `adil_oran = 1 / model_olasiligi` hesapla ve kaynağı “Model adil oranı” olarak etiketle.
3. Bacak oranı üretilemeyen eski/özel durumlarda kupon sınıfının fallback toplam oranını kullan; bunu “yaklaşık oran” olarak işaretle.

Tahmini oranı gerçek bookmaker oranıymış gibi sunma. Kullanıcı oran kaynağını görebilmelidir.

### 8.3 Gerçek oran girişi

Kullanıcı, harici siteden aldığı **toplam ondalık kupon oranını** girebilsin.

- Değer 1,00'dan büyük olmalı.
- Gerçek oran girildiğinde hesaplamalarda tahmini toplam oran yerine bu değer kullanılmalı.
- Tahmini oran silinmemeli; karşılaştırma amacıyla saklanmalı.
- “Tahmini oran” ve “Girdiğiniz gerçek oran” açıkça yan yana gösterilmeli.

Hesaplar:

```text
basa_bas_olasilik = 1 / gercek_oran
beklenen_deger = (kupon_olasiligi * gercek_oran) - 1
potansiyel_geri_donus = sanal_stake * gercek_oran
potansiyel_net_degisim = sanal_stake * (gercek_oran - 1)
```

Gerçek oran girilmemişse aynı hesaplarda tahmini oranı kullan fakat sonucu “tahmini” olarak etiketle.

Pozitif beklenen değer, garanti kazanç şeklinde sunulmamalıdır.

## 9. Kupon yaşam döngüsü

Her kupon şu durumlardan birinde olmalıdır:

- `draft` — taslak; sanal kasayı etkilemez.
- `pending` — plana eklenmiş ve maçları bekliyor.
- `won` — bütün geçerli bacaklar tuttu.
- `lost` — en az bir bacak kaybetti.
- `void` — bütün bacaklar geçersiz/iade.

Bir kupon `pending` olduğunda sanal stake kullanılabilir bakiyeden düşülür ve bekleyen stake içinde gösterilir.

Sonuçlandırma:

```text
won  -> kullanılabilir bakiyeye stake * kullanılan_oran ekle
lost -> ekleme yapma; stake zaten kupon oluşturulurken düşülmüştür
void -> kullanılabilir bakiyeye stake iade et
```

Toplam sanal kasa değeri:

```text
toplam_sanal_kasa = kullanilabilir_bakiye + bekleyen_stake
```

Bir kupon yalnızca bütün bacaklar kesin sonuca ulaştığında `won` veya `lost` olarak sonuçlandırılmalı. Eksik sonuç varsa `pending` kalmalı.

Toplam gol sonucu:

- 0,5 Üst: toplam gol en az 1 ise kazanır.
- 1,5 Üst: toplam gol en az 2 ise kazanır.
- 2,5 Üst: toplam gol en az 3 ise kazanır.

Sonuç eşleştirmesinde mümkünse mevcut `match_id` kullanılmalı. Metin tabanlı takım adı eşleştirmesi yalnızca kontrollü fallback olmalı.

Aynı kupon ikinci kez sonuçlandırılmamalı. Tüm settlement işlemleri idempotent olmalıdır.

## 10. Kuponlarım ekranı

Üç alt görünüm oluştur:

1. Taslaklar
2. Bekleyenler
3. Sonuçlananlar

Her kupon kartında:

- Oluşturma tarihi
- Kaynak: BETAVUS önerisi / Kullanıcı kuponu / Karma
- Risk profili ve kupon sınıfı
- Bacaklar
- Sanal stake
- Tahmini oran
- Gerçek oran
- Kullanılan settlement oranı
- Tahmini olasılık
- Durum
- Kazanç/kayıp sonucu
- Kupon öncesi ve sonrası sanal kasa

Sonuçlanan kuponlarda filtreler:

- Tümü
- Kazanan
- Kaybeden
- Minimum / Orta / Yüksek risk
- Tarih aralığı

Özet metrikler:

- Toplam sonuçlanan kupon
- Kupon tutma oranı
- Sanal toplam stake
- Sanal net sonuç
- Sanal ROI
- En uzun kazanma ve kaybetme serisi
- Maksimum kasa düşüşü

Kullanıcı tüm paper-betting geçmişini JSON olarak dışa aktarabilmeli ve daha önce dışa aktarılan geçerli BETAVUS JSON dosyasını geri içe aktarabilmelidir. İçe aktarmadan önce şema ve sürüm doğrulaması yap; mevcut veriyi açık kullanıcı onayı olmadan ezme.

## 11. Plan simülasyonu

MVP için tarayıcı tarafında, deterministik seed kullanabilen bir Monte Carlo simülasyonu ekle. Hesaplamayı DOM kodundan ayrı saf bir fonksiyon olarak yaz.

Önerilen varsayılan iterasyon sayısı: 5.000. Mobil cihazlarda arayüzü kilitlememesi için gerekirse işi parçalara böl veya Web Worker kullan.

Her iterasyonda:

1. Kalan plan günlerini ilerlet.
2. Seçili risk profilinin izin verdiği kupon sınıflarını kullan.
3. Kupon sonucunu ilgili birleşik model olasılığına göre üret.
4. Kazanırsa sanal bakiyeyi kullanılan oranla güncelle.
5. Kaybederse stake kaybını uygula.
6. Kasa sıfırın altına inmesin.
7. Hedefe ilk ulaşılan günü kaydet.
8. Maksimum düşüşü kaydet.

Gösterilecek çıktılar:

- Plan sonu medyan kasa
- P10 / P90 aralığı
- Hedefe süre içinde ulaşma olasılığı
- Başlangıç kasasının yarısının altına düşme olasılığı
- Tahmini maksimum düşüş

Simülasyon sonucu “olasılıklı tahmin” olarak etiketlenmeli ve garanti gibi sunulmamalıdır.

Kupon önerisi bulunmayan günlerde bahis varmış gibi sonuç üretme. Veri yoksa rezervde kalma davranışını modelle.

## 12. Adaptif plan önerileri

Kullanıcının güncel kasası hedef yolunun gerisindeyse üç alternatifi yan yana göster:

1. **Süreyi uzat** — risk profilini değiştirmeden yeni tahmini sonuç.
2. **Hedefi ayarla** — süre ve risk profilini değiştirmeden ulaşılabilir alternatif hedef.
3. **Risk profilini değiştir** — bir üst risk seviyesinin potansiyel etkisi ve artan kayıp/düşüş riski.

Her alternatif için:

- Yeni hedefe ulaşma olasılığı
- Medyan plan sonu kasa
- P10 / P90 aralığı
- Yarı kasa kaybı olasılığı
- Maksimum düşüş tahmini

Risk profili değişikliği otomatik uygulanmamalı. Kullanıcı karşılaştırmayı görmeli ve ayrıca onaylamalıdır.

Tek bir kayıp kupon sonrası “riski artır” önerisi gösterme. En azından plan yolundan anlamlı sapma ve yeterli sonuçlanmış kupon geçmişi aranmalıdır. Minimum geçmiş sayısını merkezi ayarda tut; başlangıç değeri 5 sonuçlanmış kupon olsun.

## 13. Yerel veri modeli

Mevcut `betavus.` önekini koru. Yeni state için sürümlü tek bir anahtar kullan:

```text
betavus.paper_v1
```

Önerilen şema:

```json
{
  "schemaVersion": 1,
  "settings": {
    "currency": "EUR",
    "riskProfile": "cautious"
  },
  "plan": {
    "id": "plan-id",
    "createdAt": "ISO-8601",
    "startDate": "YYYY-MM-DD",
    "durationDays": 30,
    "startingBank": 50,
    "targetBank": 500,
    "availableBalance": 50,
    "status": "active"
  },
  "slips": [],
  "ledger": [],
  "simulation": {
    "lastRunAt": null,
    "seed": null,
    "result": null
  }
}
```

Kupon nesnesi en az şu alanları içermeli:

```json
{
  "id": "slip-id",
  "createdAt": "ISO-8601",
  "source": "recommended",
  "riskProfile": "cautious",
  "couponClass": "minimum",
  "status": "draft",
  "stake": 0,
  "estimatedOdds": 1.25,
  "actualOdds": null,
  "oddsUsed": 1.25,
  "oddsSource": "estimated",
  "combinedProbability": 0,
  "expectedValue": 0,
  "bankBefore": 0,
  "bankAfter": null,
  "settledAt": null,
  "selections": []
}
```

Seçim nesnesi en az şu alanları içermeli:

```json
{
  "matchId": "existing-match-id",
  "league": "Premier League",
  "kickoffUtc": "ISO-8601",
  "home": "Home Team",
  "away": "Away Team",
  "market": "over_1_5",
  "probability": 0.86,
  "basis": "form+h2h",
  "estimatedLegOdds": 1.16,
  "result": "pending",
  "score": null
}
```

Ledger girdileri değiştirilemez olay kayıtları gibi tutulmalı:

- `plan_created`
- `stake_reserved`
- `slip_won`
- `slip_lost`
- `slip_void`
- `manual_adjustment`

Kullanıcıya manuel sanal kasa düzeltmesi gerekiyorsa sebep alanı iste ve bunu ledger'a ayrı hareket olarak yaz. Geçmiş kupon kayıtlarını sessizce değiştirme.

## 14. Arayüz ve kullanıcı deneyimi

- Mevcut BETAVUS görsel dilini, renklerini ve mobil yaklaşımını koru.
- Yeni ekranlar masaüstü ve mobilde kullanılabilir olmalı.
- Para ve oran alanlarında Türkçe sayı girişini mümkün olduğunca tolere et; içeride sayısal değer sakla.
- Renk tek başına anlam taşımamalı; metin ve ikonla desteklenmeli.
- Pozitif değerleri aşırı teşvik edici animasyonlarla sunma.
- Kayıp ve risk bilgisini potansiyel getiriden daha az görünür yapma.
- Form hatalarını alanın yanında açık Türkçe mesajlarla göster.
- Bekleyen, kazanan, kaybeden ve geçersiz durumları erişilebilir rozetlerle göster.
- Kullanıcı kupon üzerinde yaptığı her değişikliğin sonucu nasıl etkilediğini anında görebilmeli.

Önerilen temel butonlar:

- Kasa Planı Oluştur
- Öneriyi İncele
- Kupona Ekle
- Maçı Çıkar
- Maç Ekle
- Taslak Olarak Kaydet
- Plana Ekle
- Gerçek Oranı Gir / Güncelle
- Senaryoları Karşılaştır
- Planı Güncelle
- Geçmişi Dışa Aktar

## 15. Hesaplama yardımcıları

En az aşağıdaki mantıkları bağımsız ve test edilebilir fonksiyonlar halinde uygula:

- `getMarketProbability(match, market)`
- `calculateCombinedProbability(selections)`
- `calculateEstimatedOdds(selections, couponClass)`
- `calculateExpectedValue(probability, odds)`
- `calculatePotentialReturn(stake, odds)`
- `getRiskAllocation(profile, currentBank)`
- `buildRecommendedCoupon(matches, couponClass)`
- `validateSlip(slip, state)`
- `settleSelection(selection, result)`
- `settleSlip(slip, results)`
- `calculateLedgerBalances(ledger)`
- `calculateTargetPath(plan, day)`
- `classifyPlanStatus(currentBank, targetPathValue)`
- `runPlanSimulation(plan, profile, couponInputs, options)`
- `buildAdaptiveOptions(plan, state, simulationInputs)`

Fonksiyon isimleri mevcut kod stiliyle uyumlu olacak şekilde değiştirilebilir; sorumluluk ayrımı korunmalıdır.

## 16. Kritik hata ve uç durumları

Aşağıdaki durumları güvenli şekilde ele al:

- Sonuç verisi henüz gelmeyen maç
- Ertelenen veya iptal edilen maç
- Aynı kuponun iki kez sonuçlandırılmaya çalışılması
- Aynı maçın iki kez eklenmesi
- Gerçek oranın boş, 1 veya geçersiz olması
- Sıfır bakiye
- Stake'in kullanılabilir bakiyeden yüksek olması
- Kupon düzenlenirken maçın başlamış olması
- `localStorage` verisinin bozuk veya eski sürüm olması
- JSON içe aktarma dosyasının hatalı olması
- Sonuç dosyasında maç bulunamaması
- Eksik `basis`, olasılık veya market oranı
- Bölme-sıfır hataları
- Simülasyon girişlerinin eksik olması
- Kullanıcının birden fazla bekleyen kupona toplam bakiyesinden fazla stake ayırmaya çalışması

Bozuk state bulunursa kullanıcı verisini sessizce silme. Güvenli yedek/dışa aktarma seçeneği sun ve varsayılan boş state'e geçmeden önce uyar.

## 17. Kabul kriterleri

Görev aşağıdaki şartların tamamı sağlanmadan bitmiş sayılmaz:

1. Mevcut Tahminler ve Tahmin vs Gerçekleşen ekranları çalışmaya devam eder.
2. Kullanıcı sanal başlangıç kasası, hedef, süre ve risk profiliyle plan oluşturabilir.
3. Kullanıcı risk profiline göre otomatik kupon önerisi alabilir.
4. Kullanıcı öneriden maç çıkarabilir ve uygun maç ekleyebilir.
5. Kupon olasılığı ve tahmini oran düzenleme sırasında anında güncellenir.
6. Kullanıcı gerçek toplam oranı girebilir; beklenen değer ve potansiyel geri dönüş yeniden hesaplanır.
7. Taslak kupon sanal kasayı etkilemez.
8. Plana eklenen kuponun stake'i kullanılabilir sanal bakiyeden ayrılır.
9. Tamamlanan maçlar mevcut sonuç verisinden okunarak kupon yalnızca bir kez sonuçlandırılır.
10. Kazanan, kaybeden ve void kuponlarda sanal kasa doğru güncellenir.
11. Kasa Planım ekranı hedef ilerlemesini ve olasılıklı simülasyon sonuçlarını gösterir.
12. Hedefin gerisinde kalan kullanıcıya süre, hedef ve risk alternatifleri karşılaştırmalı gösterilir.
13. Risk hiçbir zaman otomatik yükseltilmez.
14. Sayfa yenilendiğinde plan ve kupon geçmişi `localStorage` üzerinden korunur.
15. Mobil görünümde taşma, üst üste binme veya kullanılamayan kontrol bulunmaz.
16. Konsolda yeni kritik hata bulunmaz.
17. `NaN`, `Infinity`, `#DIV/0!` benzeri kullanıcıya yansıyan hesap hatası bulunmaz.
18. Paper-betting açıklaması ve garanti verilmeyen olasılıklı dil görünürdür.

## 18. Zorunlu doğrulama senaryoları

Uygulamayı teslim etmeden önce en az şu senaryoları doğrula:

### Senaryo A — Plan oluşturma

- Başlangıç sanal kasası: 50 EUR
- Hedef: 500 EUR
- Süre: 30 gün
- Profil: Temkinli

Plan oluşturulmalı, ilk gün hedef yolu ve gerekli günlük büyüme gösterilmelidir.

### Senaryo B — Öneriyi düzenleme

- Önerilen kupondan bir maç çıkar.
- Başka uygun maç ekle.

Birleşik olasılık, tahmini oran ve geri dönüş canlı olarak değişmelidir.

### Senaryo C — Gerçek oran

- Tahmini toplam oran: 1,70
- Kullanıcının girdiği gerçek oran: 1,55

Başa baş olasılık, beklenen değer ve potansiyel geri dönüş 1,55 üzerinden yeniden hesaplanmalı; iki oran da geçmişte saklanmalıdır.

### Senaryo D — Otomatik sonuçlandırma

- 2,5 Üst seçimi 2-1 biten maçta kazanmalı.
- 2,5 Üst seçimi 1-1 biten maçta kaybetmeli.
- Sonucu gelmeyen diğer bacak varsa kupon beklemede kalmalıdır.

### Senaryo E — Bakiye

- Kullanılabilir bakiye: 100 EUR
- Sanal stake: 10 EUR
- Gerçek oran: 2,00

Kupon plana eklenince kullanılabilir bakiye 90 EUR olmalı. Kazanırsa 20 EUR kredi eklenip bakiye 110 EUR olmalı. Kaybederse 90 EUR olarak kalmalıdır.

### Senaryo F — İdempotent settlement

Aynı sonuç dosyası tekrar işlendiğinde kupon ikinci kez ödeme almamalı ve ledger'a ikinci settlement kaydı eklenmemelidir.

## 19. Uygulama sırası

1. Mevcut kodu ve veri şemalarını incele.
2. Mevcut davranışın kısa teknik özetini çıkar.
3. Paper-betting state şemasını ve saf hesaplama fonksiyonlarını ekle.
4. Kasa Planım ekranını ekle.
5. Kupon Önerileri ve kupon düzenleyiciyi ekle.
6. Kuponlarım yaşam döngüsünü ekle.
7. Sonuç eşleştirme ve idempotent settlement mantığını ekle.
8. Simülasyon ve adaptif plan seçeneklerini ekle.
9. Mobil/erişilebilirlik kontrollerini yap.
10. Kabul kriterleri ve zorunlu senaryoları test et.
11. README'ye yeni paper-betting özelliklerini, localStorage anahtarını ve hesaplama yaklaşımını belgele.

## 20. Teslimat beklentisi

Uygulama sonunda:

- Değiştirilen dosyaları listele.
- Her dosyada yapılan değişikliği kısa şekilde açıkla.
- Kullanılan hesaplama varsayımlarını belirt.
- Kabul kriterlerinin sonuçlarını tek tek raporla.
- Çalıştırılan testleri ve sonuçlarını yaz.
- Bilinen sınırlamaları açıkça belirt.
- Mevcut deployment düzenini değiştirmeden çalışır durumda bırak.

Eksik ürün kararıyla karşılaşırsan güvenli, şeffaf ve paper-betting kimliğini koruyan seçeneği tercih et. Gerçek para kullanımını veya garanti getiri algısını artıracak varsayımlarda bulunma.
