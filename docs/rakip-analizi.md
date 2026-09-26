# Rakip site analizi (iş listesi #8)

26 Eylül 2026. Siteler gerçek tarayıcıyla masaüstünde (1366 px) ve telefonda (390 px) açılıp
ekran görüntüleri üzerinden incelendi. Sofascore ve FootyStats masaüstü bot korumasına takıldı;
FootyStats telefon görünümü incelendi.

| Site | Ne yapıyor | Güçlü yanı | Zayıf yanı |
|---|---|---|---|
| **Forebet** | Her maç için olasılık %, tahmin, doğru skor, ort. gol, hava durumu | Pazar sekmeleri (1X2 · Alt/Üst · KG · Çifte Şans …) tek satırda; **Yaklaşan / Biten** ayrımı; biten maçta skor tahminin yanında; telefonda **alt menü çubuğu**; takvim; favori yıldızı | Kalabalık; reklam tablonun ortasında |
| **ThePuntersPage** | Tahmin + istatistik + bahis şirketi rehberi | BETAVUS'un tasarım örneği. Büyük kart butonlar; maç kartında **1-X-2 olasılık çubuğu**; lacivert/turkuaz çapraz şekillerle güçlü bir hero | Asıl gelir bonus/bahis şirketi yönlendirmesi |
| **PredictZ** | Günlük tahmin + son 5 maç formu | **Gün düğmeleri** (Bugün, Yarın, Pzt …); lig bazında gruplu liste; form rozetleri (G/B/M) | Koyu, eski görünüm; çerez bandı ekranı kaplıyor |
| **WinDrawWin** | Tahmin, KG, 2.5 Üst, doğru skor | **Takvim şeridi** (gün kutuları); "**Lige atla**" listesi, her ligde kaç tahmin olduğu yazıyor; filtreler tek satırda | Fotoğraflı hero içeriği aşağı itiyor |
| **Over25Tips** | 2.5 Üst tahminleri | **Dün · Bugün · Yarın** tek dokunuşla; lig sekmeleri | "Sure Win Prediction", "Sure Six Straight Win" gibi **garanti dili**; casino bağlantıları |
| **Adam Choi** | Takım bazında Üst/KG istatistikleri | Takım satırında **yüzde çubuğu** (yeşil/kırmızı) ve maç maç liste | İlk ekran casino bonus reklamları |
| **SoccerSTATS** | Lig tabloları, form, istatistik | Üstte **bayrak şeridi**: tüm ligler tek bakışta, bir dokunuşla | Dev reklamlar; 2010 görünümü |
| **FootyStats** | İstatistik + tahmin | Telefonda sade liste: pazar · maç · oran; **"Profit tracked"** (kâr takipli) ibaresi; koyu tema düğmesi | Önemli kısımlar ücretli |
| **Flashscore** | Canlı skor | Hepsi / Canlı / Biten / Planlı sekmeleri; tarih oku; favori yıldızı | İlk açılışta yaş doğrulama + kumar reklamı penceresi |

## BETAVUS'un zaten öne çıktığı yerler — korunmalı

- **Doğrulanmış geçmiş performans şeridi** (%91,9 · 12.065/13.133). Rakiplerde böyle açık,
  sayılı bir geçmiş başarı gösterimi yok; yalnız FootyStats "kâr takipli" diyor. En güçlü farkımız bu.
- **Reklam, casino, bonus yok.** İncelenen sitelerin neredeyse hepsinde ilk ekranı reklam kaplıyor.
- **Garanti dili yok.** Over25Tips "Sure Win" diyor; BETAVUS "garanti değildir" diyor. Böyle kalmalı.
- Altı pazarın olasılığı tek satırda, eşik geçince vurgulu: Forebet'teki sekme değiştirme gerekmiyor.

## Uygulanabilecekler (öncelik sırasıyla)

1. **Lig filtresini bayraklı düğme şeridine çevirmek** — SoccerSTATS, WinDrawWin, Over25Tips.
   Bugün düz bir açılır liste; hangi liglerin olduğu görünmüyor. Bayrak + lig adı + maç sayısı,
   telefonda yatay kaydırma. → **#9**
2. **Gün şeridi** (Bugün · Yarın · Cmt · Paz …, her günün maç sayısıyla) — PredictZ, WinDrawWin,
   Over25Tips, Forebet. Bugün "Tüm günler" butonu bir takvim açıyor, bir dokunuş fazla. → yeni öneri
3. **Biten maçı tahminin yanında göstermek** — Forebet. Fikstür'de oynanmış maçın skoru ve
   vurgunun tutup tutmadığı aynı satırda. Geçmiş performans şeridini her maçta somutlaştırır;
   #4 (canlı isabet takibi) ile birlikte düşünülmeli. → yeni öneri
4. **Telefonda alt menü çubuğu** — Forebet. Sekmeler başparmak altında, üst başlık küçülür;
   #16 (mobilde üst başlık alanı) ile aynı sorunu çözer. → #16'ya not
5. **Olasılık çubuğu** — ThePuntersPage, Adam Choi. Maç detayında sayıların yanında ince bir çubuk,
   yüzdeleri bir bakışta okunur kılar. → yeni öneri (düşük öncelik)
6. **Arka plan / hero** — ThePuntersPage'in lacivert-turkuaz çapraz şekilleri, WinDrawWin'in
   bulanık saha fotoğrafı. BETAVUS'un sayfa zemini düz gri-beyaz; hero'daki saha çizgileri çok
   silik. Renk paleti zaten ThePuntersPage'den; zemin de aynı dile çekilebilir. → **#10**

## Yapılmamalı

- **Hava durumu sütunu** (Forebet): BETAVUS'ta test edildi, gol sayısına etkisi ölçülemedi.
- **Doğru skor tahmini**: tek skorun tutma olasılığı çok düşük (~%10–15); "garanti" izlenimi
  yaratmadan göstermek zor.
- **Reklam, bonus, "sure win" dili**: güveni ve paper-betting konumlandırmasını bozar.
