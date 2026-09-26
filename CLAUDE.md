# BETAVUS — Claude Code talimatları

Bu repo iki kişi (Murat, Gökhan) tarafından, her biri kendi bilgisayarındaki
klonda ve kendi Claude oturumuyla geliştiriliyor. Ortak tek kaynak GitHub'daki
`main` dalıdır. Proje bilgisi için önce [`HANDOVER.md`](HANDOVER.md) okunur.

## İş takibi: `BETAVUS_Is_Listesi.xlsx`

Tüm işler ve fikirler bu dosyada tutulur (`İş Listesi` sayfası, satırlar 5'ten
başlar; `Özet` sayfası formüllerle kendini günceller, elle yazılmaz).

**Her oturumun başında**
1. `git pull --rebase origin main`
2. İş listesini oku. Kullanıcıya kısaca şunları göster:
   - `Öneri` durumundaki satırlar: karar bekliyor. Her biri için "şimdi
     yapayım mı, plana mı alayım (Açık), yoksa iptal mi?" diye sor.
   - `Devam ediyor` satırları: yarım kalmış iş, devam edilsin mi?
   - `Açık` satırlar: öncelik ve faza göre sıradaki 1–3 iş, "bunu yapayım mı?"
   Kullanıcı başka bir işle geldiyse bu özeti tek kısa paragrafla ver ve
   kullanıcının istediği işe geç.

**Kullanıcı yeni bir fikir/istek söylediğinde**
- Listede zaten varsa o satırı kullan; yoksa yeni satır ekle: sıradaki `No`,
  uygun `Segment` (sığmıyorsa `3 - Genel / Diğer`), `Başlık`, `Açıklama`,
  `Kaynak` (ör. "Murat", "Gökhan"), `Öncelik`, `Faz`, `Durum = Öneri`,
  `Eklenme Tarihi = bugün`.
- Sonra sor: "Şimdi mi yapayım, plana mı alayım?" Cevaba göre `Devam ediyor`
  veya `Açık` yap. Kullanıcı açıkça "hemen yap" dediyse sormadan
  `Devam ediyor` yap ve başla.

**Bir iş bittiğinde**
- `Durum = Tamamlandı`, `Kapanış Tarihi = bugün`, `Commit / Yapılan` = kısa
  commit hash'i + bir cümle ne yapıldı. Test bekleyen iş için `Test`.
- Commit hash'ini listeye **push'tan sonra** yaz: bot sık commit attığı için `pull --rebase`
  hash'i değiştirir. Sıra: iş commit + push → hash'i listeye yaz → liste commit + push.
- Canlıda etkin olması başkasının adımına bağlıysa (ör. Supabase'de SQL çalıştırmak)
  `Tamamlandı` değil `Test` yap ve `Not`'a kimin ne yapması gerektiğini yaz.

**Dosyayı düzenleme kuralları**
- Yalnızca `openpyxl` ile yaz; mevcut biçim, açılır listeler, koşullu renkler
  ve `Özet` formülleri korunmalı. Kaydederken
  `wb.calculation.fullCalcOnLoad = True` bırak (Özet açılışta hesaplansın).
- Durum değerleri yalnızca: `Öneri, Açık, Devam ediyor, Test, Tamamlandı, İptal`.
- Dosya Excel'de açıksa Windows kilitler ve yazma başarısız olur; kullanıcıdan
  Excel'i kapatmasını iste, zorlama.
- `.xlsx` ikili dosyadır, git satır birleştirmesi yapamaz. Düzenlemeden hemen
  önce pull, hemen sonra commit + push. Rebase'de bu dosyada çakışma olursa
  birinin satırlarını ezme: iki sürümü (`git show :2:` / `:3:`) okuyup
  satırları birleştir, sonra devam et.
- İnsanların Excel'de elle eklediği satırlar da aynı kurallarla işlenir;
  eksik alan varsa (ör. tarih) tamamla, içeriğini değiştirme.

## Git

- Değişiklikler yerelde bırakılmaz: her iş commit + push ile biter.
- Bot sık sık `Update BETAVUS predictions` commit'i atar; push reddedilirse
  `git pull --rebase origin main` ve tekrar push.
- Değişiklikten sonra `python scripts/run_tests.py` (tüm testler) yeşil olmalı.
- `scratch/` commit edilmez.
