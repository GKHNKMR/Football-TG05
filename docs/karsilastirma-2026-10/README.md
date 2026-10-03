# BETAVUS ↔ SistemGol karşılaştırması (Ekim 2026)

Murat'ın arkadaşı Ufuk Özbek kendi sistemi SistemGol ile BETAVUS'u karşılaştırdı. Bu klasörde o karşılaştırma ve BETAVUS tarafının yanıtı var. Okuma sırası:

| Dosya | Ne | Kimden |
|---|---|---|
| `SISTEM_RAPORU.md` | BETAVUS'un bağımsız sistem raporu (commit `371a24b`, 02.10.2026). Karşılaştırmanın kaynağı. | Claude oturumu (Murat) |
| `BETAVUS_SistemGol_Karsilastirma.pdf` | Karşılaştırmanın ilk sürümü | Ufuk Özbek |
| `BETAVUS_SistemGol_KarsilastirmaR01.pdf` | **Güncel sürüm (R01):** site ekranının incelemesi ve kasa bölümü eklendi | Ufuk Özbek |
| `BETAVUS_SistemGol_Savunma_Raporu.pdf` | **BETAVUS'un yanıtı:** tez tez savunma, R01'deki 41 iddianın her biri için doğru mu / kabul mü ve neden, iyileştirme listeleri | Claude oturumu (Murat), 03.10.2026 |
| `BETAVUS_SistemGol_Savunma_Raporu.html` | PDF'in kaynağı (düzenlemek için) | |

SistemGol'ün kendi sistem raporu (`SISTEMGOL_RAPORU.md`) bilerek bu repoda **yok**, çünkü repo herkese açık. Gerekirse Murat ayrıca iletir.

## Öne çıkanlar

- R01'in büyük çerçevesi doğru: iki sistemde de ölçülebilir bir avantaj yok, kasa planı sürdürülemez, eşikler örneklem-içi seçilmiş.
- **Bir tez bugün için yanlış:** openfootball'da 0-0'lar eksik değil. `"score": [0, 0]` biçiminde duruyorlar ve `ft_goals` bunları `db2f1d0`'dan (20.09.2026) beri okuyor. Ancak 09–20.09 canlı tahminleri 0-0'suz modelle üretildi.
- Piyasa harmanı canlıya 23.09'da girdi (`864da10`); o tarihten beri maç oynanmadı. Arşiv, maçtan medyan 4,2 gün önceki **ilk** tahmini sakladığı için oranlı sürüm kayda geçmiyor.

## Karar bekleyenler

Savunma raporundaki BETAVUS iyileştirmeleri (B1–B14), `BETAVUS_Is_Listesi.xlsx`'e **#53–#66** olarak `Öneri` durumuyla eklendi. Hiçbiri uygulanmadı; Murat ve Gökhan karar verecek.
