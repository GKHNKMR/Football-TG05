// BETAVUS dil desteği: Türkçe (tr) · English (en) · Nederlands (nl)
//
// - Dil: localStorage 'betavus.lang' (hesapla giriş yapılmışsa cihazlar arasında senkronlanır),
//   yoksa tarayıcı dili (tr / nl, diğer her şey en).
// - _t('Türkçe metin', {değişken}) çeviriyi döndürür; sözlükte yoksa Türkçe metnin kendisi döner.
//   Uzun HTML blokları için 'faq.q1' gibi açık anahtarlar kullanılır ({tr, en, nl}).
// - Statik HTML: data-i18n (metin), data-i18n-html (HTML), data-i18n-title / -placeholder / -aria.
// - Dil değişince sayfa yeniden yüklenir; her ekran seçili dille baştan çizilir.
(function (root) {
  'use strict';

  const SUPPORTED = ['tr', 'en', 'nl'];
  const LOCALES = { tr: 'tr-TR', en: 'en-GB', nl: 'nl-NL' };
  const KEY = 'betavus.lang';

  function detect() {
    try { const s = localStorage.getItem(KEY); if (SUPPORTED.includes(s)) return s; } catch (e) {}
    const prefs = (navigator.languages && navigator.languages.length ? navigator.languages : [navigator.language || ''])
      .map(x => String(x).slice(0, 2).toLowerCase());
    for (const l of prefs) if (SUPPORTED.includes(l)) return l;
    return 'en';
  }

  const lang = detect();
  const locale = LOCALES[lang];
  const D = {};

  function t(key, vars) {
    const e = D[key];
    let s = e ? (e[lang] != null ? e[lang] : (e.tr != null ? e.tr : key)) : key;
    if (vars) s = s.replace(/\{(\w+)\}/g, (m, k) => (vars[k] != null ? vars[k] : m));
    return s;
  }

  // Ondalık ayırıcı: en "12.5", tr/nl "12,5"
  function dec(v) { const s = String(v); return lang === 'en' ? s : s.replace('.', ','); }
  // pct: Türkçede başa yazılan yüzde ("%95") → en "95%", nl "95%". Türkçe görünüm değişmez.
  function pct(v) { const s = String(v); return lang === 'tr' ? '%' + s : (lang === 'nl' ? dec(s) : s) + '%'; }
  // pctS: zaten sona yazılan yüzde ("12.3%") → yalnızca nl'de virgül. Türkçe görünüm değişmez.
  function pctS(v) { const s = String(v); return (lang === 'nl' ? dec(s) : s) + '%'; }
  function num(v, opts) { return Number(v).toLocaleString(locale, opts); }

  function apply(scope) {
    const r = scope || document;
    r.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
    r.querySelectorAll('[data-i18n-html]').forEach(el => { el.innerHTML = t(el.dataset.i18nHtml); });
    r.querySelectorAll('[data-i18n-title]').forEach(el => { el.title = t(el.dataset.i18nTitle); });
    r.querySelectorAll('[data-i18n-placeholder]').forEach(el => { el.placeholder = t(el.dataset.i18nPlaceholder); });
    r.querySelectorAll('[data-i18n-aria]').forEach(el => { el.setAttribute('aria-label', t(el.dataset.i18nAria)); });
  }

  function setLang(l) {
    if (!SUPPORTED.includes(l) || l === lang) return;
    try { localStorage.setItem(KEY, l); } catch (e) {}
    location.reload();
  }

  // Üst menüdeki TR / EN / NL seçici
  function mountSwitcher() {
    const host = document.querySelector('.topbtns');
    if (!host || document.getElementById('langSel')) return;
    const sel = document.createElement('select');
    sel.id = 'langSel'; sel.className = 'langsel';
    sel.setAttribute('aria-label', t('Dil'));
    sel.innerHTML = SUPPORTED.map(l => `<option value="${l}"${l === lang ? ' selected' : ''}>${l.toUpperCase()}</option>`).join('');
    sel.onchange = () => setLang(sel.value);
    host.insertBefore(sel, host.firstChild);
  }

  function add(dict) { Object.assign(D, dict); }

  document.documentElement.lang = lang;
  root.I18N = { lang, locale, t, add, apply, setLang, pct, pctS, dec, num, mountSwitcher, SUPPORTED };
  root._t = t;

  // ---------------------------------------------------------------------------
  // Sözlük — anahtar Türkçe metnin kendisi: { en, nl }
  // ---------------------------------------------------------------------------
  add({
    // Genel / üst menü
    'Dil': { en: 'Language', nl: 'Taal' },
    'Fikstür': { en: 'Fixture', nl: 'Wedstrijden' },
    'İstatistikler': { en: 'Statistics', nl: 'Statistieken' },
    'Sanal Kasa': { en: 'Virtual Bankroll', nl: 'Virtuele bankroll' },
    'FAQ': { en: 'FAQ', nl: 'FAQ' },
    'Maç tahminleri': { en: 'Match predictions', nl: 'Wedstrijdvoorspellingen' },
    'Model performansı': { en: 'Model performance', nl: 'Modelprestaties' },
    'Risksiz simülasyon': { en: 'Risk-free simulation', nl: 'Risicoloze simulatie' },
    'Tema': { en: 'Theme', nl: 'Thema' },
    'Yenile': { en: 'Refresh', nl: 'Vernieuwen' },
    'Giriş / Kayıt Ol': { en: 'Log in / Sign up', nl: 'Inloggen / Registreren' },
    'BETAVUS ana sayfa': { en: 'BETAVUS home', nl: 'BETAVUS home' },
    'Kapat': { en: 'Close', nl: 'Sluiten' },
    'Temizle': { en: 'Clear', nl: 'Wissen' },
    'Takım ara…': { en: 'Search team…', nl: 'Zoek team…' },
    'Lig filtresi': { en: 'League filter', nl: 'Competitiefilter' },
    'Tüm ligler': { en: 'All leagues', nl: 'Alle competities' },
    'Tümü': { en: 'All', nl: 'Alle' },
    'Veriler yükleniyor…': { en: 'Loading data…', nl: 'Gegevens laden…' },
    'İstatistikler yükleniyor…': { en: 'Loading statistics…', nl: 'Statistieken laden…' },
    'Analizler hazırlanıyor': { en: 'Preparing analyses', nl: 'Analyses worden voorbereid' },

    // Erişim kodu ekranı
    'Bu site davetlidir. Devam etmek için erişim kodunu gir.': { en: 'This site is invite-only. Enter your access code to continue.', nl: 'Deze site is alleen op uitnodiging. Voer je toegangscode in om verder te gaan.' },
    'Erişim kodu': { en: 'Access code', nl: 'Toegangscode' },
    'Gir': { en: 'Enter', nl: 'Verder' },
    'Kod hatalı.': { en: 'Wrong code.', nl: 'Onjuiste code.' },

    // Bülten
    'Vurgu': { en: 'Highlight', nl: 'Markering' },
    'Vurgulu': { en: 'Highlighted', nl: 'Gemarkeerd' },
    'Çifte Şans': { en: 'Double chance', nl: 'Dubbele kans' },
    'pred.col.match': { tr: 'Maç <span class="sarr">↑</span>', en: 'Match <span class="sarr">↑</span>', nl: 'Wedstrijd <span class="sarr">↑</span>' },
    'Ev sahibi kazanır veya berabere': { en: 'Home win or draw', nl: 'Thuisoverwinning of gelijkspel' },
    'Beraberlik olmaz': { en: 'No draw', nl: 'Geen gelijkspel' },
    'Deplasman kazanır veya berabere': { en: 'Away win or draw', nl: 'Uitoverwinning of gelijkspel' },
    '«{q}» araması': { en: 'Search: «{q}»', nl: 'Zoekopdracht: «{q}»' },
    '{from} – {to} · yaklaşan maçlar': { en: '{from} – {to} · upcoming matches', nl: '{from} – {to} · komende wedstrijden' },
    '{n} / {total} maç': { en: '{n} / {total} matches', nl: '{n} / {total} wedstrijden' },
    '{n} maç': { en: '{n} matches', nl: '{n} wedstrijden' },
    '«{q}» için maç yok': { en: 'No matches for «{q}»', nl: 'Geen wedstrijden voor «{q}»' },
    '{label} tahmin yok': { en: 'No {label} predictions', nl: 'Geen {label} voorspellingen' },
    'Bu aralıkta maç bulunamadı': { en: 'No matches found in this period', nl: 'Geen wedstrijden gevonden in deze periode' },
    'Takım adını kontrol et.': { en: 'Check the team name.', nl: 'Controleer de teamnaam.' },
    'Eşiği gevşetmeyi dene.': { en: 'Try a lower threshold.', nl: 'Probeer een lagere drempel.' },
    'Veri akışı kontrol ediliyor.': { en: 'Checking the data feed.', nl: 'De datastroom wordt gecontroleerd.' },
    'Aralarındaki son {n} maç{tier}: {w}G-{d}B-{l}M · ortalama {avg} gol · 2.5Ü {o25}': { en: 'Last {n} meetings{tier}: {w}W-{d}D-{l}L · average {avg} goals · Over 2.5 {o25}', nl: 'Laatste {n} onderlinge duels{tier}: {w}W-{d}G-{l}V · gemiddeld {avg} doelpunten · Over 2.5 {o25}' },
    ' ({t} dahil)': { en: ' (incl. {t})', nl: ' (incl. {t})' },
    '{n} H2H (ort. {avg} gol)': { en: '{n} H2H (avg. {avg} goals)', nl: '{n} H2H (gem. {avg} doelpunten)' },
    '● CANLI': { en: '● LIVE', nl: '● LIVE' },
    'İlk yarı': { en: '1st half', nl: '1e helft' },
    'Devre arası': { en: 'Half-time', nl: 'Rust' },
    'İkinci yarı': { en: '2nd half', nl: '2e helft' },
    'Uzatma': { en: 'Extra time', nl: 'Verlenging' },
    'Mola': { en: 'Break', nl: 'Pauze' },
    'Penaltılar': { en: 'Penalties', nl: 'Strafschoppen' },
    'Askıda': { en: 'Suspended', nl: 'Onderbroken' },
    'Yarıda kaldı': { en: 'Abandoned', nl: 'Gestaakt' },
    'Eksik kilit oyuncu: {text} ({src})': { en: 'Missing key player: {text} ({src})', nl: 'Ontbrekende sleutelspeler: {text} ({src})' },
    '⚠️ Eksik: {text}': { en: '⚠️ Missing: {text}', nl: '⚠️ Ontbreekt: {text}' },
    '{line}+ tahmini: {v} · ⚠️ Kısıtlı Veri ({basis}) — yüksek belirsizlik nedeniyle vurgulanmaz': { en: '{line}+ prediction: {v} · ⚠️ Limited data ({basis}) — not highlighted because of high uncertainty', nl: '{line}+ voorspelling: {v} · ⚠️ Beperkte data ({basis}) — niet gemarkeerd vanwege hoge onzekerheid' },
    'lig ortalaması': { en: 'league average', nl: 'competitiegemiddelde' },
    'kısmi form': { en: 'partial form', nl: 'gedeeltelijke vorm' },
    'Kısıtlı veri': { en: 'Limited data', nl: 'Beperkte data' },
    '{line} model olasılığı': { en: '{line} model probability', nl: '{line} modelkans' },
    '{line} model olasılığı {v}': { en: '{line} model probability {v}', nl: '{line} modelkans {v}' },
    ' · vurgulanan': { en: ' · highlighted', nl: ' · gemarkeerd' },
    ' (vurgulanmaz)': { en: ' (not highlighted)', nl: ' (niet gemarkeerd)' },
    'Alt lig ({t}) dahil {n} H2H maçı + form verisine dayanıyor': { en: 'Based on {n} H2H matches incl. lower league ({t}) + form data', nl: 'Gebaseerd op {n} H2H-duels incl. lagere divisie ({t}) + vormdata' },
    "H2H + form'a dayanıyor (en sağlam)": { en: 'Based on H2H + form (most robust)', nl: 'Gebaseerd op H2H + vorm (meest betrouwbaar)' },
    "Form'a dayanıyor": { en: 'Based on form', nl: 'Gebaseerd op vorm' },
    'Kısıtlı form verisi (tek takım veya az maç) — yüksek belirsizlik': { en: 'Limited form data (one team or few matches) — high uncertainty', nl: 'Beperkte vormdata (één team of weinig wedstrijden) — hoge onzekerheid' },
    '⚠️ Kısıtlı Veri': { en: '⚠️ Limited data', nl: '⚠️ Beperkte data' },
    'Yetersiz veri (sadece lig ortalaması) — yüksek belirsizlik': { en: 'Insufficient data (league average only) — high uncertainty', nl: 'Onvoldoende data (alleen competitiegemiddelde) — hoge onzekerheid' },
    '⚠️ Yetersiz Veri': { en: '⚠️ Insufficient data', nl: '⚠️ Onvoldoende data' },
    'Veri yok': { en: 'No data', nl: 'Geen data' },
    'Çifte Şans model girdileri eksik': { en: 'Double-chance model inputs missing', nl: 'Invoer voor dubbele kans ontbreekt' },
    'Çifte Şans motoru hazır değil': { en: 'Double-chance engine not ready', nl: 'Dubbele-kansmodel niet gereed' },
    'Çifte Şans analizi hesaplanamadı': { en: 'Double-chance analysis could not be calculated', nl: 'Analyse dubbele kans kon niet worden berekend' },
    'Kısıtlı veya yetersiz veri': { en: 'Limited or insufficient data', nl: 'Beperkte of onvoldoende data' },
    'Kritik eksik oyuncu uyarısı': { en: 'Critical missing-player warning', nl: 'Waarschuwing: belangrijke speler ontbreekt' },
    'Güven eşiğinin altında': { en: 'Below the confidence threshold', nl: 'Onder de betrouwbaarheidsdrempel' },
    'Yüksek güven eşiğini geçti': { en: 'Passed the high-confidence threshold', nl: 'Boven de hoge betrouwbaarheidsdrempel' },
    'Vurgulanmadı': { en: 'Not highlighted', nl: 'Niet gemarkeerd' },
    'Model olasılığı {v} · yüksek güven vurgusu, garanti değildir': { en: 'Model probability {v} · high-confidence highlight, not a guarantee', nl: 'Modelkans {v} · markering met hoge betrouwbaarheid, geen garantie' },
    '★ Yüksek güven': { en: '★ High confidence', nl: '★ Hoge betrouwbaarheid' },

    // Takvim
    'cal.weekdays': { tr: 'Pt,Sa,Ça,Pe,Cu,Ct,Pz', en: 'Mo,Tu,We,Th,Fr,Sa,Su', nl: 'Ma,Di,Wo,Do,Vr,Za,Zo' },
    'Tarih yaz:': { en: 'Enter date:', nl: 'Datum invoeren:' },
    'Bugün': { en: 'Today', nl: 'Vandaag' },
    'Bugün maç yok': { en: 'No matches today', nl: 'Vandaag geen wedstrijden' },
    'en yakın:': { en: 'nearest:', nl: 'dichtstbij:' },
    'Tüm günler': { en: 'All days', nl: 'Alle dagen' },

    // Yükleme / güncelleme satırı
    'Fikstürler yükleniyor…': { en: 'Loading fixtures…', nl: 'Wedstrijden laden…' },
    'Kaynak: openfootball + football-data.co.uk · Poisson + Dixon-Coles motoru': { en: 'Source: openfootball + football-data.co.uk · Poisson + Dixon-Coles engine', nl: 'Bron: openfootball + football-data.co.uk · Poisson + Dixon-Coles-model' },
    ' · güncelleme {ago}': { en: ' · updated {ago}', nl: ' · bijgewerkt {ago}' },
    'İstatistikler →': { en: 'Statistics →', nl: 'Statistieken →' },
    'Canlı veri bekleniyor · Geçici fikstür': { en: 'Waiting for live data · Temporary fixtures', nl: 'Wachten op live data · Tijdelijk programma' },
    '{n} dk önce': { en: '{n} min ago', nl: '{n} min geleden' },
    '{n} saat önce': { en: '{n} h ago', nl: '{n} uur geleden' },
    '{n} gün önce': { en: '{n} days ago', nl: '{n} dagen geleden' },

    // Maç detay penceresi
    'İstatistik yok': { en: 'No statistics', nl: 'Geen statistieken' },
    'Bu maç için CSV verisi bulunamadı.': { en: 'No CSV data found for this match.', nl: 'Geen CSV-data gevonden voor deze wedstrijd.' },
    'Son {n} maç{tier} · {home} {hw} – {d} B – {aw} {away} · ort. {avg} gol · 0.5Ü {o05} · 1.5Ü {o15} · 2.5Ü {o25}': { en: 'Last {n} matches{tier} · {home} {hw} – {d} D – {aw} {away} · avg. {avg} goals · O0.5 {o05} · O1.5 {o15} · O2.5 {o25}', nl: 'Laatste {n} duels{tier} · {home} {hw} – {d} G – {aw} {away} · gem. {avg} doelpunten · O0.5 {o05} · O1.5 {o15} · O2.5 {o25}' },
    'Kayıtlı karşılaşma yok.': { en: 'No recorded meetings.', nl: 'Geen geregistreerde onderlinge duels.' },
    '{n} gol': { en: '{n} goals', nl: '{n} doelpunten' },
    'Doğrulanmış İlk 11 (ESPN)': { en: 'Confirmed starting XI (ESPN)', nl: 'Bevestigde basiself (ESPN)' },
    'Sakat / Cezalı Durumu (Transfermarkt)': { en: 'Injuries / suspensions (Transfermarkt)', nl: 'Blessures / schorsingen (Transfermarkt)' },
    'Kilit Oyuncu Durumu · {s}': { en: 'Key player status · {s}', nl: 'Status sleutelspelers · {s}' },
    "(İlk 11'de yok)": { en: '(not in starting XI)', nl: '(niet in basiself)' },
    '✓ Kilit oyuncu eksikliği yok': { en: '✓ No key players missing', nl: '✓ Geen sleutelspelers afwezig' },
    'Opta xG/xA verilerine göre hücum katkısı en yüksek kilit oyuncular oynamadığında, model o tarafın gol beklentisini (λ) otomatik törpüler.': { en: 'When the key players with the highest attacking contribution according to Opta xG/xA data do not play, the model automatically trims that side\'s expected goals (λ).', nl: 'Als de sleutelspelers met de grootste aanvallende bijdrage volgens Opta xG/xA-data niet spelen, verlaagt het model automatisch de doelpuntverwachting (λ) van die ploeg.' },
    '✓ Alt Lig Karşılaşma ve Form Verisi Aktif ({t})': { en: '✓ Lower-league meetings and form data active ({t})', nl: '✓ Duels en vormdata uit lagere divisie actief ({t})' },
    'sheet.lowerTier': {
      tr: 'Bu iki takım daha önce <b>{t}</b> liginde {n} kez karşı karşıya gelmiştir (ort. <b>{avg}</b> gol · 2.5Ü <b>{o25}</b>). Alt ligdeki bu resmi karşılaşmalar ve form geçmişi modele dahil edilmiş, 0.5 / 1.5 / 2.5 olasılıkları yüksek veri güvenirliğiyle hesaplanmıştır.',
      en: 'These two teams have met {n} times before in the <b>{t}</b> (avg. <b>{avg}</b> goals · Over 2.5 <b>{o25}</b>). These official lower-league meetings and the form history are included in the model, so the 0.5 / 1.5 / 2.5 probabilities are calculated with high data reliability.',
      nl: 'Deze twee teams zijn elkaar eerder {n} keer tegengekomen in de <b>{t}</b> (gem. <b>{avg}</b> doelpunten · Over 2.5 <b>{o25}</b>). Deze officiële duels uit de lagere divisie en de vormhistorie zijn in het model opgenomen, zodat de 0.5 / 1.5 / 2.5-kansen met hoge databetrouwbaarheid zijn berekend.'
    },
    '⚠️ Kısıtlı Veri / Yetersiz Sezon Uyarısı': { en: '⚠️ Limited data / insufficient season warning', nl: '⚠️ Waarschuwing: beperkte data / onvoldoende seizoenen' },
    'sheet.lim.avg': {
      tr: 'Bu maçtaki takımların bu ligde yeterli maç geçmişi bulunmadığı için model <b>sadece lig ortalamasını</b> temel almıştır.',
      en: 'Because the teams in this match lack enough match history in this league, the model is based <b>only on the league average</b>.',
      nl: 'Omdat de teams in deze wedstrijd onvoldoende wedstrijdhistorie in deze competitie hebben, is het model <b>alleen op het competitiegemiddelde</b> gebaseerd.'
    },
    'sheet.lim.partial': {
      tr: 'Takımlardan birinin bu ligdeki geçmiş maç verisi sınırlı olduğu için model <b>kısmi form</b> üzerinden hesaplanmıştır.',
      en: 'Because one of the teams has limited match history in this league, the model is calculated from <b>partial form</b>.',
      nl: 'Omdat een van de teams beperkte wedstrijdhistorie in deze competitie heeft, is het model berekend op basis van <b>gedeeltelijke vorm</b>.'
    },
    'Küçük örneklem nedeniyle hesaplanan 0.5 / 1.5 / 2.5 olasılıkları yüksek istatistiksel varyans içerir ve güvenli vurgu sayılmaz.': { en: 'Because of the small sample, the calculated 0.5 / 1.5 / 2.5 probabilities carry high statistical variance and do not count as safe highlights.', nl: 'Door de kleine steekproef hebben de berekende 0.5 / 1.5 / 2.5-kansen een hoge statistische variantie en gelden ze niet als veilige markering.' },
    ' · kaynak: football-data.co.uk': { en: ' · source: football-data.co.uk', nl: ' · bron: football-data.co.uk' },
    'Aralarındaki maçlar (H2H)': { en: 'Head-to-head matches (H2H)', nl: 'Onderlinge duels (H2H)' },
    'Maç başına gol ortalaması · son 5 sezon': { en: 'Goals per match · last 5 seasons', nl: 'Doelpunten per wedstrijd · laatste 5 seizoenen' },
    'Form &amp; sezon · bu sezon': { en: 'Form &amp; season · this season', nl: 'Vorm &amp; seizoen · dit seizoen' },
    'Puan durumu': { en: 'Standings', nl: 'Stand' },
    ' · {n} takımlık lig': { en: ' · {n}-team league', nl: ' · competitie met {n} teams' },
    'std.cols': { tr: '#,Takım,O,G,B,M,A,Y,AV,P', en: '#,Team,P,W,D,L,GF,GA,GD,Pts', nl: '#,Team,G,W,G,V,DV,DT,DS,P' },
    'Bu sezon CSV verisi yok.': { en: 'No CSV data this season.', nl: 'Geen CSV-data dit seizoen.' },
    '(E)': { en: '(H)', nl: '(T)' },
    '(D)': { en: '(A)', nl: '(U)' },
    ' · son 3 maç': { en: ' · last 3 matches', nl: ' · laatste 3 wedstrijden' },
    'Attığı / yediği (ort.)': { en: 'Scored / conceded (avg.)', nl: 'Voor / tegen (gem.)' },
    '0.5 Üst': { en: 'Over 0.5', nl: 'Over 0.5' },
    '1.5 Üst': { en: 'Over 1.5', nl: 'Over 1.5' },
    '2.5 Üst': { en: 'Over 2.5', nl: 'Over 2.5' },
    'form verisi yok': { en: 'no form data', nl: 'geen vormdata' },
    'Sezon verisi yok.': { en: 'No season data.', nl: 'Geen seizoensdata.' },
    'Yeterli sezon verisi yok.': { en: 'Not enough season data.', nl: 'Onvoldoende seizoensdata.' },
    '{v} gol/maç': { en: '{v} goals/match', nl: '{v} doelpunten/wedstrijd' },
    ' ({gf} attı · {ga} yedi)': { en: ' ({gf} scored · {ga} conceded)', nl: ' ({gf} voor · {ga} tegen)' },
    'Son 5 sezon maç başına toplam gol ortalaması': { en: 'Average total goals per match over the last 5 seasons', nl: 'Gemiddeld totaal aantal doelpunten per wedstrijd, laatste 5 seizoenen' },
    'sheet.goalNote': {
      tr: 'Her takımın o sezondaki <b>tüm lig maçlarında</b> maç başına toplam gol (attığı + yediği) — bu iki takımın aralarındaki maçlar <b>değil</b>. Noktaya gelince attığı/yediği ayrımı görünür. Kaynak: football-data.co.uk',
      en: 'Total goals per match (scored + conceded) in <b>all league matches</b> of each team that season — <b>not</b> the matches between these two teams. Hover a point to see scored/conceded. Source: football-data.co.uk',
      nl: 'Totaal aantal doelpunten per wedstrijd (voor + tegen) in <b>alle competitiewedstrijden</b> van elk team dat seizoen — <b>niet</b> de duels tussen deze twee teams. Beweeg over een punt om voor/tegen te zien. Bron: football-data.co.uk'
    },

    // FAQ
    'Sık Sorulan Sorular': { en: 'Frequently Asked Questions', nl: 'Veelgestelde vragen' },
    'faq.what.tag': { tr: '🛡️ BETAVUS NEDİR?', en: '🛡️ WHAT IS BETAVUS?', nl: '🛡️ WAT IS BETAVUS?' },
    'faq.what.h': { tr: 'Yapay Zekâ Destekli Paper-Betting &amp; Kasa Yönetimi', en: 'AI-powered Paper Betting &amp; Bankroll Management', nl: 'AI-gestuurde Paper Betting &amp; Bankrollbeheer' },
    'faq.what.p': {
      tr: 'Futbol toplam gol (0.5Ü, 1.5Ü, 2.5Ü) ve çifte şans (1X, 12, X2) pazarları için Poisson ve Dixon-Coles matematiksel olasılık modelleriyle çalışan, <b>risksiz sanal kasa simülasyonu, akıllı kupon planlama ve disiplinli portföy yönetimi</b> platformudur.',
      en: 'A platform for <b>risk-free virtual bankroll simulation, smart bet planning and disciplined portfolio management</b>, powered by Poisson and Dixon-Coles probability models for football total-goals (Over 0.5, 1.5, 2.5) and double-chance (1X, 12, X2) markets.',
      nl: 'Een platform voor <b>risicovrije virtuele bankrollsimulatie, slimme weddenplanning en gedisciplineerd portefeuillebeheer</b>, gebaseerd op de wiskundige kansmodellen van Poisson en Dixon-Coles voor totaal-doelpunten- (Over 0.5, 1.5, 2.5) en dubbele-kansmarkten (1X, 12, X2) in het voetbal.'
    },
    'faq.not.tag': { tr: '🚫 BETAVUS NE DEĞİLDİR?', en: '🚫 WHAT BETAVUS IS NOT', nl: '🚫 WAT BETAVUS NIET IS' },
    'faq.not.h': { tr: 'Bahis Operatörü, Kumarhane veya Ödeme Aracı Değildir!', en: 'Not a Bookmaker, Casino or Payment Service!', nl: 'Geen bookmaker, casino of betaaldienst!' },
    'faq.not.p': {
      tr: 'BETAVUS <b>asla bahis kabul etmez, para yatırma/çekme yapmaz</b> ve kullanıcı adına kupon oynatmaz. Gösterilen tüm bakiyeler ve getiriler sanaldır; kesin kazanç vaat etmez. Kullanıcının para kaybetmeden stratejisini ölçmesini sağlar.',
      en: 'BETAVUS <b>never accepts bets or handles deposits/withdrawals</b> and never places bets on your behalf. All balances and returns shown are virtual; no winnings are promised. It lets you measure your strategy without losing money.',
      nl: 'BETAVUS <b>accepteert nooit weddenschappen en verwerkt geen stortingen/opnames</b>, en plaatst nooit weddenschappen namens jou. Alle getoonde saldi en rendementen zijn virtueel; er wordt geen winst beloofd. Je kunt je strategie meten zonder geld te verliezen.'
    },
    'faq.q1': { tr: 'Olasılıklar nasıl hesaplanıyor?', en: 'How are the probabilities calculated?', nl: 'Hoe worden de kansen berekend?' },
    'faq.a1': {
      tr: 'Her takımın <b>ev / deplasman</b> maçlarında attığı ve yediği gol oranları, son 4 sezondan yakın sezona daha çok ağırlık verilerek (1.0 / 0.7 / 0.45 / 0.30) çıkarılır; yakın tarihli maçlar ayrıca daha ağır sayılır. İki taraf birleştirilip ev ve deplasmanın beklenen golleri (λ) bulunur. İki takım 2+ kez karşılaşmışsa son karşılaşmaların gol ortalaması %28 ağırlıkla karıştırılır. Bu λ değerleri düşük skorlar için Dixon-Coles düzeltmesiyle bir skor matrisine dönüştürülür; 0.5+ / 1.5+ / 2.5+ ve 1X / 12 / X2 olasılıklarının hepsi bu matristen gelir.',
      en: 'Each team\'s goals scored and conceded in <b>home / away</b> matches are taken from the last 4 seasons, giving recent seasons more weight (1.0 / 0.7 / 0.45 / 0.30); recent matches also count more. Combining both sides gives the expected goals (λ) for the home and away team. If the two teams have met 2+ times, the goal average of their recent meetings is blended in at 28% weight. These λ values are turned into a score matrix with the Dixon-Coles correction for low scores; all 0.5+ / 1.5+ / 2.5+ and 1X / 12 / X2 probabilities come from this matrix.',
      nl: 'Voor elk team worden de gescoorde en tegengekregen doelpunten in <b>thuis- / uitwedstrijden</b> uit de laatste 4 seizoenen gehaald, waarbij recentere seizoenen zwaarder wegen (1,0 / 0,7 / 0,45 / 0,30); recente wedstrijden tellen bovendien zwaarder. Door beide kanten te combineren ontstaan de verwachte doelpunten (λ) voor thuis en uit. Als de twee teams elkaar 2+ keer hebben ontmoet, wordt het doelpuntengemiddelde van hun recente duels met 28% gewicht meegewogen. Deze λ-waarden worden met de Dixon-Coles-correctie voor lage scores omgezet in een scorematrix; alle kansen voor 0.5+ / 1.5+ / 2.5+ en 1X / 12 / X2 komen uit deze matrix.'
    },
    'faq.q2': { tr: '"Tahmini toplam gol" nedir?', en: 'What are "predicted total goals"?', nl: 'Wat zijn "verwachte totale doelpunten"?' },
    'faq.a2': {
      tr: 'Modelin maçtan önce beklediği toplam gol sayısıdır (ev λ + deplasman λ). Örneğin 2,74 demek, bu maçın uzun vadede ortalama 2,74 gol üretmesinin beklendiği anlamına gelir; kesin skor tahmini değildir.',
      en: 'It is the total number of goals the model expects before the match (home λ + away λ). For example, 2.74 means the match is expected to produce 2.74 goals on average in the long run; it is not an exact score prediction.',
      nl: 'Het is het totale aantal doelpunten dat het model vóór de wedstrijd verwacht (thuis-λ + uit-λ). Zo betekent 2,74 dat de wedstrijd op lange termijn gemiddeld 2,74 doelpunten zou opleveren; het is geen voorspelling van de exacte uitslag.'
    },
    'faq.q3': { tr: 'Vurgu eşikleri nelerdir?', en: 'What are the highlight thresholds?', nl: 'Wat zijn de markeringsdrempels?' },
    'faq.a3': {
      tr: 'Bir tahmin şu eşiği geçerse vurgulanır: <b>0.5+ ≥ %93,5 · 1.5+ ≥ %83 · 2.5+ ≥ %75 · 1X / 12 ≥ %80 · X2 ≥ %78</b>. Vurgu, modelin en emin olduğu tahminlerdir; garanti değildir.',
      en: 'A prediction is highlighted when it passes this threshold: <b>0.5+ ≥ 93.5% · 1.5+ ≥ 83% · 2.5+ ≥ 75% · 1X / 12 ≥ 80% · X2 ≥ 78%</b>. Highlights are the predictions the model is most confident about; they are not guarantees.',
      nl: 'Een voorspelling wordt gemarkeerd als ze deze drempel haalt: <b>0.5+ ≥ 93,5% · 1.5+ ≥ 83% · 2.5+ ≥ 75% · 1X / 12 ≥ 80% · X2 ≥ 78%</b>. Markeringen zijn de voorspellingen waar het model het zekerst van is; ze zijn geen garantie.'
    },
    'faq.q4': { tr: 'Hangi maçlar vurgulanmaz?', en: 'Which matches are not highlighted?', nl: 'Welke wedstrijden worden niet gemarkeerd?' },
    'faq.a4': {
      tr: '<span class="badge b-lim">⚠️ Kısıtlı Veri</span> etiketli maçlar (yeni yükselen takım, çok az maç geçmişi, yalnızca lig ortalamasına dayanan tahmin) ve kilit bir oyuncusunun eksik olduğu doğrulanan maçlar, olasılığı eşiği geçse bile vurgulanmaz; başarı oranlarına da dahil edilmez.',
      en: 'Matches labelled <span class="badge b-lim">⚠️ Limited data</span> (newly promoted team, very little match history, a prediction based only on the league average) and matches where a key player is confirmed missing are not highlighted, even if the probability passes the threshold; they are also excluded from the success rates.',
      nl: 'Wedstrijden met het label <span class="badge b-lim">⚠️ Beperkte data</span> (net gepromoveerd team, heel weinig wedstrijdhistorie, een voorspelling alleen op basis van het competitiegemiddelde) en wedstrijden waarin een sleutelspeler bevestigd ontbreekt, worden niet gemarkeerd, ook als de kans boven de drempel ligt; ze tellen ook niet mee in de succespercentages.'
    },
    'faq.q5': { tr: 'Rozetler ne anlama geliyor?', en: 'What do the badges mean?', nl: 'Wat betekenen de badges?' },
    'faq.a5': {
      tr: '<span class="badge b3">●●●</span> H2H + form verisine dayanıyor (en sağlam) · <span class="badge b2">●●</span> forma dayanıyor · <span class="badge b-lim">⚠️ Kısıtlı Veri</span> tek takım / yetersiz maç geçmişi.',
      en: '<span class="badge b3">●●●</span> based on H2H + form data (most robust) · <span class="badge b2">●●</span> based on form · <span class="badge b-lim">⚠️ Limited data</span> one team only / insufficient match history.',
      nl: '<span class="badge b3">●●●</span> gebaseerd op H2H + vormdata (meest betrouwbaar) · <span class="badge b2">●●</span> gebaseerd op vorm · <span class="badge b-lim">⚠️ Beperkte data</span> slechts één team / onvoldoende wedstrijdhistorie.'
    },
    'faq.q6': { tr: 'Kilit oyuncu eksikse ne oluyor?', en: 'What happens if a key player is missing?', nl: 'Wat gebeurt er als een sleutelspeler ontbreekt?' },
    'faq.a6': {
      tr: "Opta xG ve xA verileriyle takımların hücum yükünü çeken kilit oyuncular belirlenir. Maç günü ilk 11'de (ESPN) veya sakatlık / ceza listelerinde (Transfermarkt) kilit isimlerin olmadığı doğrulanırsa o takımın gol beklentisi otomatik olarak düşürülür.",
      en: 'Opta xG and xA data identify the key players who carry each team\'s attacking load. If it is confirmed on match day that key names are missing from the starting XI (ESPN) or appear on the injury / suspension lists (Transfermarkt), that team\'s expected goals are lowered automatically.',
      nl: 'Met Opta xG- en xA-data worden de sleutelspelers bepaald die de aanvallende last van een team dragen. Als op de wedstrijddag bevestigd wordt dat sleutelspelers niet in de basiself staan (ESPN) of op de blessure- / schorsingslijst staan (Transfermarkt), wordt de doelpuntverwachting van dat team automatisch verlaagd.'
    },
    'faq.q7': { tr: 'Başarı oranları nasıl hesaplanıyor?', en: 'How are the success rates calculated?', nl: 'Hoe worden de succespercentages berekend?' },
    'faq.a7': {
      tr: '2021/22 – 2025/26 arasındaki 5 tamamlanmış sezonun 16.478 maçı ve 2026/27 sezonunun bugüne kadar oynanan maçları, canlı sitedeki modelle <b>yalnızca o maçtan önceki verilerle</b> tahmin edildi ve gerçek skorlarla karşılaştırıldı; model hiçbir zaman tahmin ettiği sezonun sonuçlarını görmedi. Liste her gün yeni oynanan maçlarla güncellenir. Ana sayfadaki oran, bu maçlarda eşiği geçen vurgulu tahminlerin kaçının tuttuğudur. Tüm liste <a href="#" onclick="setTab(\'stats\');return false">İstatistikler</a> sayfasında.',
      en: 'The 16,478 matches of the 5 completed seasons 2021/22 – 2025/26 and the 2026/27 matches played so far were predicted with the live site\'s model <b>using only the data available before each match</b> and compared with the real scores; the model never saw the results of the season it was predicting. The list is updated every day with newly played matches. The rate on the home page is the share of highlighted predictions above the threshold that came true. The full list is on the <a href="#" onclick="setTab(\'stats\');return false">Statistics</a> page.',
      nl: 'De 16.478 wedstrijden van de 5 afgeronde seizoenen 2021/22 – 2025/26 en de tot nu toe gespeelde wedstrijden van 2026/27 zijn met het model van de live site voorspeld <b>met alleen de gegevens van vóór elke wedstrijd</b> en vergeleken met de echte uitslagen; het model heeft de uitslagen van het voorspelde seizoen nooit gezien. De lijst wordt elke dag bijgewerkt met nieuw gespeelde wedstrijden. Het percentage op de startpagina is het aandeel gemarkeerde voorspellingen boven de drempel dat is uitgekomen. De volledige lijst staat op de pagina <a href="#" onclick="setTab(\'stats\');return false">Statistieken</a>.'
    },
    'faq.q8': { tr: 'Model ne kadar güvenilir?', en: 'How reliable is the model?', nl: 'Hoe betrouwbaar is het model?' },
    'faq.a8': {
      tr: 'Model <b>kalibre</b>dir: "%60" dediğinde uzun vadede yaklaşık %60 gerçekleşir. Tüm tahminlere bakıldığında tek maç ayrıştırma gücü piyasanın çok az üstündedir; ama yüksek güven eşiğini geçen daha az sayıda tahmin belirgin biçimde daha iyi tutar. Kayıp garantisi yoktur.',
      en: 'The model is <b>calibrated</b>: when it says "60%", it happens about 60% of the time in the long run. Across all predictions, its single-match discriminating power is only slightly better than the market; but the smaller number of predictions above the high-confidence threshold come true noticeably more often. There is no guarantee against losses.',
      nl: 'Het model is <b>gekalibreerd</b>: als het "60%" zegt, gebeurt het op lange termijn ongeveer 60% van de tijd. Over alle voorspellingen is het onderscheidend vermogen per wedstrijd maar iets beter dan de markt; het kleinere aantal voorspellingen boven de hoge betrouwbaarheidsdrempel komt echter duidelijk vaker uit. Er is geen garantie tegen verlies.'
    },
    'faq.q9': { tr: "Fikstür'de hangi maçlar var?", en: 'Which matches are in the Fixture?', nl: 'Welke wedstrijden staan in het Wedstrijdschema?' },
    'faq.a9': {
      tr: '10 ligin (Premier League, Championship, LaLiga, Bundesliga, Serie A, Ligue 1, Eredivisie, Süper Lig, Primeira Liga, Belçika Pro League) önümüzdeki 1 aydaki maçları. Veriler saatte bir güncellenir; milli aralarda liste kısalabilir.',
      en: 'The next month\'s matches from 10 leagues (Premier League, Championship, LaLiga, Bundesliga, Serie A, Ligue 1, Eredivisie, Süper Lig, Primeira Liga, Belgian Pro League). Data is updated every hour; the list may be shorter during international breaks.',
      nl: 'De wedstrijden van de komende maand uit 10 competities (Premier League, Championship, LaLiga, Bundesliga, Serie A, Ligue 1, Eredivisie, Süper Lig, Primeira Liga, Belgische Pro League). De gegevens worden elk uur bijgewerkt; tijdens interlandperiodes kan de lijst korter zijn.'
    },
    'faq.q10': { tr: 'Sanal Kasa verilerim nerede saklanıyor?', en: 'Where is my Virtual Bankroll data stored?', nl: 'Waar worden mijn Virtuele bankroll-gegevens opgeslagen?' },
    'faq.a10': {
      tr: 'Giriş yapmadan kullanırsan veriler yalnızca bu tarayıcıda durur. Üye olup giriş yaparsan Sanal Kasa ve tercihlerin hesabına kaydedilir; başka bir cihazda giriş yaptığında aynı veriler orada da görünür.',
      en: 'If you use the site without logging in, the data stays only in this browser. If you sign up and log in, your Virtual Bankroll and preferences are saved to your account; when you log in on another device, the same data appears there too.',
      nl: 'Als je de site gebruikt zonder in te loggen, blijven de gegevens alleen in deze browser. Als je je registreert en inlogt, worden je Virtuele bankroll en voorkeuren in je account opgeslagen; als je op een ander apparaat inlogt, zie je daar dezelfde gegevens.'
    },
    'faq.q11': { tr: 'Gizlilik', en: 'Privacy', nl: 'Privacy' },
    'faq.a11': {
      tr: 'Hangi bilgileri neden tuttuğumuzu <a href="gizlilik.html">Gizlilik Politikası</a> sayfasında bulabilirsin.',
      en: 'You can read which data we keep and why on the <a href="gizlilik.html">Privacy Policy</a> page.',
      nl: 'Welke gegevens we bewaren en waarom, lees je op de pagina <a href="gizlilik.html">Privacybeleid</a>.'
    },

    // İstatistikler
    'Vurgulanan tahmin başarısı': { en: 'Highlighted prediction success', nl: 'Succes gemarkeerde voorspellingen' },
    '{h} / {n} tahmin tuttu': { en: '{h} / {n} predictions correct', nl: '{h} / {n} voorspellingen uitgekomen' },
    'Tam isabetli vurgulu maç': { en: 'Fully correct highlighted matches', nl: 'Volledig juiste gemarkeerde wedstrijden' },
    '{h} / {n} maçta tüm vurgular tuttu': { en: 'all highlights correct in {h} / {n} matches', nl: 'alle markeringen juist in {h} / {n} wedstrijden' },
    'Analiz edilen maç': { en: 'Matches analysed', nl: 'Geanalyseerde wedstrijden' },
    '{n} maç kısıtlı veri, vurgu dışı': { en: '{n} matches with limited data, not highlighted', nl: '{n} wedstrijden met beperkte data, niet gemarkeerd' },
    'Lig bazında doğruluk': { en: 'Accuracy by market', nl: 'Nauwkeurigheid per markt' },
    'st.note.market': {
      tr: '<b>Vurgulanan</b>: modelin güven eşiğini geçtiği tahminler. <b>Genel yön isabeti</b>: tüm maçlarda modelin eğildiği taraf (olur / olmaz) doğru mu? <b>Ort. model olasılığı</b> ile <b>gerçekleşme</b> birbirine yakınsa model iyi kalibre demektir.',
      en: '<b>Highlighted</b>: predictions where the model passed its confidence threshold. <b>Overall direction accuracy</b>: across all matches, was the side the model leaned to (happens / does not happen) right? If the <b>avg. model probability</b> and the <b>occurrence</b> are close, the model is well calibrated.',
      nl: '<b>Gemarkeerd</b>: voorspellingen waarbij het model zijn betrouwbaarheidsdrempel haalde. <b>Algemene richtingsnauwkeurigheid</b>: klopte over alle wedstrijden de kant waar het model naar neigde (gebeurt / gebeurt niet)? Als de <b>gem. modelkans</b> en het <b>uitkomstpercentage</b> dicht bij elkaar liggen, is het model goed gekalibreerd.'
    },
    'st.col.market': { tr: 'Lig', en: 'Market', nl: 'Markt' },
    'Eşik': { en: 'Threshold', nl: 'Drempel' },
    'Vurgulanan başarı': { en: 'Highlighted success', nl: 'Succes gemarkeerd' },
    'Tuttu / Vurgu': { en: 'Correct / Highlighted', nl: 'Juist / Gemarkeerd' },
    'Genel yön isabeti': { en: 'Overall direction accuracy', nl: 'Algemene richtingsnauwkeurigheid' },
    'Ort. model olasılığı': { en: 'Avg. model probability', nl: 'Gem. modelkans' },
    'Gerçekleşme': { en: 'Occurrence', nl: 'Uitgekomen' },
    'Lig bazında vurgulanan başarı': { en: 'Highlighted success by league', nl: 'Succes gemarkeerd per competitie' },
    'Lig': { en: 'League', nl: 'Competitie' },
    'Maç': { en: 'Match', nl: 'Wedstrijd' },
    'Maçlar': { en: 'Matches', nl: 'Wedstrijden' },
    'Model {p}': { en: 'Model {p}', nl: 'Model {p}' },
    ' · vurgulandı · ': { en: ' · highlighted · ', nl: ' · gemarkeerd · ' },
    'tuttu': { en: 'correct', nl: 'uitgekomen' },
    'tutmadı': { en: 'missed', nl: 'niet uitgekomen' },
    ' · gerçekleşti': { en: ' · happened', nl: ' · uitgekomen' },
    'Modelin maç öncesi tahmin ettiği toplam gol (λ)': { en: 'Total goals the model predicted before the match (λ)', nl: 'Totaal aantal doelpunten dat het model vóór de wedstrijd voorspelde (λ)' },
    'Tahmini toplam gol:': { en: 'Predicted total goals:', nl: 'Verwachte totale doelpunten:' },
    'Tuttu': { en: 'Won', nl: 'Gewonnen' },
    'Kaybetti': { en: 'Lost', nl: 'Verloren' },
    '‹ Önceki': { en: '‹ Previous', nl: '‹ Vorige' },
    'Sonraki ›': { en: 'Next ›', nl: 'Volgende ›' },
    'Sayfa {p} / {n} · {m} maç': { en: 'Page {p} / {n} · {m} matches', nl: 'Pagina {p} / {n} · {m} wedstrijden' },
    'Vurgulanan maçlar — tahmin vs gerçekleşen': { en: 'Highlighted matches — predicted vs actual', nl: 'Gemarkeerde wedstrijden — voorspeld vs werkelijk' },
    'Tüm vurgulular': { en: 'All highlighted', nl: 'Alle gemarkeerde' },
    '✓ Tutan': { en: '✓ Won', nl: '✓ Uitgekomen' },
    '✗ Tutmayan': { en: '✗ Lost', nl: '✗ Niet uitgekomen' },
    '{n} tahmin': { en: '{n} predictions', nl: '{n} voorspellingen' },
    'st.note.list': {
      tr: 'Her hücre maç öncesi model olasılığıdır. <span class="st-legend win">yeşil</span> = vurgulanan tahmin tuttu, <span class="st-legend lose">kırmızı</span> = vurgulanan tahmin tutmadı, <b>✓</b> = vurgusuz ama gerçekleşti. Bir maçta birden fazla vurgu olabilir; sayaçlar vurgulu tahmin sayısını gösterir. Tarih başlığına tıklayarak sıralamayı değiştir.',
      en: 'Each cell is the pre-match model probability. <span class="st-legend win">green</span> = highlighted prediction came true, <span class="st-legend lose">red</span> = highlighted prediction missed, <b>✓</b> = not highlighted but happened. A match can carry several highlights; the counters show highlighted predictions. Click the date header to change the order.',
      nl: 'Elke cel is de modelkans van vóór de wedstrijd. <span class="st-legend win">groen</span> = gemarkeerde voorspelling uitgekomen, <span class="st-legend lose">rood</span> = gemarkeerde voorspelling niet uitgekomen, <b>✓</b> = niet gemarkeerd maar wel gebeurd. Een wedstrijd kan meerdere markeringen hebben; de tellers tonen gemarkeerde voorspellingen. Klik op de datumkop om de volgorde te wijzigen.'
    },
    'Tıkla: {x} sırala': { en: 'Click: sort {x}', nl: 'Klik: sorteer {x}' },
    'eskiden yeniye': { en: 'oldest first', nl: 'oudste eerst' },
    'yeniden eskiye': { en: 'newest first', nl: 'nieuwste eerst' },
    'Tarih': { en: 'Date', nl: 'Datum' },
    'Skor': { en: 'Score', nl: 'Uitslag' },
    'Bu filtrede maç yok': { en: 'No matches for this filter', nl: 'Geen wedstrijden voor dit filter' },
    'Tüm sezonlar': { en: 'All seasons', nl: 'Alle seizoenen' },
    'Maç verisi yükleniyor…': { en: 'Loading match data…', nl: 'Wedstrijddata laden…' },
    'İstatistik verisi yüklenemedi': { en: 'Statistics could not be loaded', nl: 'Statistieken konden niet worden geladen' },
    'Sayfayı yenilemeyi dene.': { en: 'Try refreshing the page.', nl: 'Probeer de pagina te vernieuwen.' },
    'st.intro': {
      tr: '{seasons} sezonlarının {n} maçı, canlı sitedeki modelle <b>yalnızca maçtan önceki verilerle</b> tahmin edildi ve gerçek skorlarla karşılaştırıldı.',
      en: '{n} matches from the {seasons} seasons were predicted with the live site\'s model <b>using only pre-match data</b> and compared with the real scores.',
      nl: '{n} wedstrijden uit de seizoenen {seasons} zijn met het model van de live site voorspeld <b>met alleen gegevens van vóór de wedstrijd</b> en vergeleken met de echte uitslagen.'
    },
    ' Filtre: <b>{lg}</b>.': { en: ' Filter: <b>{lg}</b>.', nl: ' Filter: <b>{lg}</b>.' },
    'isabet oranı': { en: 'hit rate', nl: 'trefkans' },
    'Doğrulanmış geçmiş performans': { en: 'Verified past performance', nl: 'Geverifieerde prestaties' },
    'Vurguladığımız tahminlerin başarı oranı': { en: 'Success rate of our highlighted predictions', nl: 'Succespercentage van onze gemarkeerde voorspellingen' },
    '{season} sezonundan bugüne · <strong>{h}</strong> / {n} vurgulu tahmin tuttu': { en: 'From the {season} season to today · <strong>{h}</strong> / {n} highlighted predictions correct', nl: 'Van seizoen {season} tot vandaag · <strong>{h}</strong> / {n} gemarkeerde voorspellingen uitgekomen' },
    '{h} tahmin tuttu / {n} vurgulu tahmin': { en: '{h} correct / {n} highlighted predictions', nl: '{h} uitgekomen / {n} gemarkeerde voorspellingen' },
    '{h}/{n} maç': { en: '{h}/{n} matches', nl: '{h}/{n} wedstrijden' },
    'Tüm istatistikler →': { en: 'All statistics →', nl: 'Alle statistieken →' },

    // Üyelik
    'Giriş Yap': { en: 'Log in', nl: 'Inloggen' },
    'Kayıt Ol': { en: 'Sign up', nl: 'Registreren' },
    'Üyelik': { en: 'Membership', nl: 'Account' },
    'Üyelik çok yakında': { en: 'Accounts coming soon', nl: 'Accounts binnenkort beschikbaar' },
    'Kayıt ve giriş sistemi henüz etkinleştirilmedi. Şimdilik Sanal Kasa verilerin yalnızca bu cihazda saklanıyor.': { en: 'Sign-up and login are not enabled yet. For now your Virtual Bankroll data is stored only on this device.', nl: 'Registreren en inloggen zijn nog niet ingeschakeld. Voorlopig worden je Virtuele bankroll-gegevens alleen op dit apparaat opgeslagen.' },
    'Tamam': { en: 'OK', nl: 'OK' },
    'Google ile devam et': { en: 'Continue with Google', nl: 'Doorgaan met Google' },
    'veya': { en: 'or', nl: 'of' },
    'Google ile giriş başlatılamadı: ': { en: 'Could not start Google sign-in: ', nl: 'Inloggen met Google kon niet worden gestart: ' },
    'Tekrar hoş geldin': { en: 'Welcome back', nl: 'Welkom terug' },
    'Sanal Kasa ve tercihlerin her cihazda seninle olsun.': { en: 'Keep your Virtual Bankroll and preferences with you on every device.', nl: 'Neem je Virtuele bankroll en voorkeuren mee naar elk apparaat.' },
    'Kullanıcı adı veya e-posta': { en: 'Username or email', nl: 'Gebruikersnaam of e-mail' },
    'Şifre': { en: 'Password', nl: 'Wachtwoord' },
    'Şifremi unuttum': { en: 'Forgot password', nl: 'Wachtwoord vergeten' },
    'Kullanıcı adı/e-posta ve şifre gerekli.': { en: 'Username/email and password are required.', nl: 'Gebruikersnaam/e-mail en wachtwoord zijn verplicht.' },
    'Giriş yapılıyor…': { en: 'Logging in…', nl: 'Bezig met inloggen…' },
    'Çok fazla hatalı deneme. 15 dakika sonra tekrar dene.': { en: 'Too many failed attempts. Try again in 15 minutes.', nl: 'Te veel mislukte pogingen. Probeer het over 15 minuten opnieuw.' },
    'Kullanıcı adı veya şifre hatalı.': { en: 'Wrong username or password.', nl: 'Onjuiste gebruikersnaam of wachtwoord.' },
    'E-posta adresin henüz doğrulanmadı. Gelen kutundaki bağlantıya tıkla.': { en: 'Your email address is not verified yet. Click the link in your inbox.', nl: 'Je e-mailadres is nog niet bevestigd. Klik op de link in je inbox.' },
    'Kullanıcı adı / e-posta veya şifre hatalı.': { en: 'Wrong username / email or password.', nl: 'Onjuiste gebruikersnaam / e-mail of wachtwoord.' },
    'Şifre sıfırlama': { en: 'Reset password', nl: 'Wachtwoord resetten' },
    'Kayıtlı e-posta adresine bir sıfırlama bağlantısı gönderelim.': { en: 'We will send a reset link to your registered email address.', nl: 'We sturen een resetlink naar je geregistreerde e-mailadres.' },
    'E-posta': { en: 'Email', nl: 'E-mail' },
    'Bağlantı gönder': { en: 'Send link', nl: 'Link versturen' },
    'Geçerli bir e-posta gir.': { en: 'Enter a valid email address.', nl: 'Voer een geldig e-mailadres in.' },
    'Bağlantı gönderildi. E-postanı kontrol et.': { en: 'Link sent. Check your email.', nl: 'Link verstuurd. Controleer je e-mail.' },
    'Yeni şifre': { en: 'New password', nl: 'Nieuw wachtwoord' },
    'Yeni şifreni belirle': { en: 'Set your new password', nl: 'Stel je nieuwe wachtwoord in' },
    'Yeni şifre (en az 8 karakter)': { en: 'New password (at least 8 characters)', nl: 'Nieuw wachtwoord (minstens 8 tekens)' },
    'Kaydet': { en: 'Save', nl: 'Opslaan' },
    'Şifre en az 8 karakter olmalı.': { en: 'The password must be at least 8 characters.', nl: 'Het wachtwoord moet minstens 8 tekens hebben.' },
    'Şifren güncellendi.': { en: 'Your password has been updated.', nl: 'Je wachtwoord is bijgewerkt.' },
    'Kullanıcı adı': { en: 'Username', nl: 'Gebruikersnaam' },
    '3–20 karakter: harf, rakam, _ .': { en: '3–20 characters: letters, digits, _ .', nl: '3–20 tekens: letters, cijfers, _ .' },
    'Yaş': { en: 'Age', nl: 'Leeftijd' },
    'Cinsiyet': { en: 'Gender', nl: 'Geslacht' },
    'Ülke': { en: 'Country', nl: 'Land' },
    'Seç…': { en: 'Select…', nl: 'Kies…' },
    'Diğer': { en: 'Other', nl: 'Anders' },
    'Erkek': { en: 'Male', nl: 'Man' },
    'Kadın': { en: 'Female', nl: 'Vrouw' },
    'Belirtmek istemiyorum': { en: 'Prefer not to say', nl: 'Zeg ik liever niet' },
    'Kullanıcı adı 3–20 karakter olmalı; yalnızca harf, rakam, _ ve . kullanılabilir.': { en: 'The username must be 3–20 characters; only letters, digits, _ and . are allowed.', nl: 'De gebruikersnaam moet 3–20 tekens hebben; alleen letters, cijfers, _ en . zijn toegestaan.' },
    "BETAVUS'u kullanmak için 18 yaşından büyük olmalısın.": { en: 'You must be over 18 to use BETAVUS.', nl: 'Je moet ouder dan 18 zijn om BETAVUS te gebruiken.' },
    'Cinsiyet seç.': { en: 'Select a gender.', nl: 'Kies een geslacht.' },
    'Ülke seç.': { en: 'Select a country.', nl: 'Kies een land.' },
    'Bu kullanıcı adı alınmış.': { en: 'This username is already taken.', nl: 'Deze gebruikersnaam is al in gebruik.' },
    'Hesap oluştur': { en: 'Create account', nl: 'Account aanmaken' },
    'Sadece birkaç basit bilgi. Verilerin hiçbir cihazda kaybolmaz.': { en: 'Just a few simple details. Your data is never lost on any device.', nl: 'Slechts een paar eenvoudige gegevens. Je data gaat op geen enkel apparaat verloren.' },
    'Şifre (en az 8 karakter)': { en: 'Password (at least 8 characters)', nl: 'Wachtwoord (minstens 8 tekens)' },
    'Kaydediliyor…': { en: 'Saving…', nl: 'Opslaan…' },
    'Bu e-posta ile zaten bir hesap var. Giriş yapmayı dene.': { en: 'An account with this email already exists. Try logging in.', nl: 'Er bestaat al een account met dit e-mailadres. Probeer in te loggen.' },
    'Kayıt alındı! {email} adresine bir doğrulama bağlantısı gönderdik. Bağlantıya tıkladıktan sonra giriş yapabilirsin.': { en: 'Registration received! We sent a verification link to {email}. You can log in after clicking the link.', nl: 'Registratie ontvangen! We hebben een bevestigingslink naar {email} gestuurd. Na het klikken op de link kun je inloggen.' },
    'Profilini tamamla': { en: 'Complete your profile', nl: 'Maak je profiel compleet' },
    'Son bir adım': { en: 'One last step', nl: 'Nog één stap' },
    'Hesabın açıldı. Birkaç basit bilgiyle profilini tamamla.': { en: 'Your account has been created. Complete your profile with a few simple details.', nl: 'Je account is aangemaakt. Maak je profiel compleet met een paar eenvoudige gegevens.' },
    'Çıkış yap': { en: 'Log out', nl: 'Uitloggen' },
    'Hesabım': { en: 'My account', nl: 'Mijn account' },
    'Sanal Kasa ve tercihlerin hesabına kaydedilir; başka bir cihazda giriş yaptığında aynı veriler orada da görünür.': { en: 'Your Virtual Bankroll and preferences are saved to your account; when you log in on another device, the same data appears there too.', nl: 'Je Virtuele bankroll en voorkeuren worden in je account opgeslagen; als je op een ander apparaat inlogt, zie je daar dezelfde gegevens.' },
    'Şimdi senkronla': { en: 'Sync now', nl: 'Nu synchroniseren' },
    'Senkronlanıyor…': { en: 'Syncing…', nl: 'Synchroniseren…' },
    'Profili düzenle': { en: 'Edit profile', nl: 'Profiel bewerken' },
    'Son senkron: {time}': { en: 'Last sync: {time}', nl: 'Laatste synchronisatie: {time}' },
    'Senkron hatası — değişiklikler bu cihazda saklandı, tekrar denenecek.': { en: 'Sync error — changes are kept on this device and will be retried.', nl: 'Synchronisatiefout — wijzigingen zijn op dit apparaat bewaard en worden opnieuw geprobeerd.' },
    'Başka bir cihazdaki değişiklikler alındı.': { en: 'Changes from another device were received.', nl: 'Wijzigingen van een ander apparaat ontvangen.' },

    // Sanal Kasa
    'PAPER BETTING – KASA SİMÜLASYONU': { en: 'PAPER BETTING – BANKROLL SIMULATION', nl: 'PAPER BETTING – BANKROLLSIMULATIE' },
    '« Önce simüle et. Riskini gör. Stratejini ölç. Sonra karar ver. »': { en: '« Simulate first. See your risk. Measure your strategy. Then decide. »', nl: '« Eerst simuleren. Zie je risico. Meet je strategie. Dan pas beslissen. »' },
    'KASALARIM:': { en: 'MY BANKROLLS:', nl: 'MIJN BANKROLLS:' },
    '➕ Yeni Kasa Aç': { en: '➕ New bankroll', nl: '➕ Nieuwe bankroll' },
    '🗑️ Kasayı Sil': { en: '🗑️ Delete bankroll', nl: '🗑️ Bankroll verwijderen' },
    'Aktif Kasayı Sil': { en: 'Delete active bankroll', nl: 'Actieve bankroll verwijderen' },
    'KULLANICI GİRİŞLERİ': { en: 'USER INPUTS', nl: 'INVOER' },
    'OTOMATİK PARAMETRELER': { en: 'AUTOMATIC PARAMETERS', nl: 'AUTOMATISCHE PARAMETERS' },
    'Kasa Adı': { en: 'Bankroll name', nl: 'Naam bankroll' },
    'Örn: Hafta sonu kasam': { en: 'e.g. My weekend bankroll', nl: 'bijv. Mijn weekendbankroll' },
    'Başlangıç Kasası ({sym})': { en: 'Starting bankroll ({sym})', nl: 'Startbankroll ({sym})' },
    'Hedef Kasa ({sym})': { en: 'Target bankroll ({sym})', nl: 'Doelbankroll ({sym})' },
    'Risk Faktörü': { en: 'Risk factor', nl: 'Risicofactor' },
    'Günlük Büyüme Oranı': { en: 'Daily growth rate', nl: 'Dagelijkse groei' },
    'Kasa Rezerv Oranı': { en: 'Bankroll reserve rate', nl: 'Reservepercentage' },
    'Hedefe Ulaşma Günü': { en: 'Day target is reached', nl: 'Dag waarop doel wordt bereikt' },
    'Hedef Günündeki Teorik Kasa': { en: 'Theoretical bankroll on target day', nl: 'Theoretische bankroll op doeldag' },
    'SEÇİLEN RİSK': { en: 'SELECTED RISK', nl: 'GEKOZEN RISICO' },
    'GÜNLÜK ARTIŞ': { en: 'DAILY GROWTH', nl: 'DAGELIJKSE GROEI' },
    'REZERV': { en: 'RESERVE', nl: 'RESERVE' },
    'Kasa Gelişim Grafiği': { en: 'Bankroll Growth Chart', nl: 'Groeigrafiek bankroll' },
    'Gerçek Kasa ({sym})': { en: 'Actual bankroll ({sym})', nl: 'Werkelijke bankroll ({sym})' },
    'Teorik Hedef Kasa ({sym})': { en: 'Theoretical target bankroll ({sym})', nl: 'Theoretische doelbankroll ({sym})' },
    'Gün': { en: 'Day', nl: 'Dag' },
    'Günlük Değişim ({sym})': { en: 'Daily change ({sym})', nl: 'Dagelijkse wijziging ({sym})' },
    'Toplam Büyüme (%)': { en: 'Total growth (%)', nl: 'Totale groei (%)' },
    'Toplam Büyüme': { en: 'Total growth', nl: 'Totale groei' },
    '🔒 Kasayı Kapat': { en: '🔒 Close bankroll', nl: '🔒 Bankroll sluiten' },
    '📊 Excel olarak dışa aktar': { en: '📊 Export to Excel', nl: '📊 Exporteren naar Excel' },
    'Excel olarak dışa aktar': { en: 'Export to Excel', nl: 'Exporteren naar Excel' },
    '📊 Excel': { en: '📊 Excel', nl: '📊 Excel' },
    'kasa.closeConfirm': {
      tr: '"{name}" kasası kapatılsın mı?\n\nBaşlangıç: {start}\nKapanış: {final}\n\nKasa, geçmişiyle birlikte "Kapatılan Kasalar" listesine taşınır.',
      en: 'Close the "{name}" bankroll?\n\nStart: {start}\nClose: {final}\n\nThe bankroll and its history move to "Closed bankrolls".',
      nl: 'Bankroll "{name}" sluiten?\n\nStart: {start}\nSluiting: {final}\n\nDe bankroll en de geschiedenis gaan naar "Gesloten bankrolls".'
    },
    'Kapatılan Kasalar': { en: 'Closed bankrolls', nl: 'Gesloten bankrolls' },
    'Kapatılan Kasa': { en: 'Closed bankrolls', nl: 'Gesloten bankrolls' },
    'Toplam Başlangıç': { en: 'Total start', nl: 'Totale start' },
    'Toplam Kapanış': { en: 'Total close', nl: 'Totale sluiting' },
    'Bugüne Kadar Toplam Kazanç': { en: 'Total profit to date', nl: 'Totale winst tot nu' },
    'Toplam Kazanç': { en: 'Total profit', nl: 'Totale winst' },
    'Toplam Kazanç ({sym})': { en: 'Total profit ({sym})', nl: 'Totale winst ({sym})' },
    'Kapanış Kasası ({sym})': { en: 'Closing bankroll ({sym})', nl: 'Slotbankroll ({sym})' },
    'Kapanış Kasası': { en: 'Closing bankroll', nl: 'Slotbankroll' },
    'Kapanış Tarihi': { en: 'Close date', nl: 'Sluitingsdatum' },
    'Başlangıç Tarihi': { en: 'Start date', nl: 'Startdatum' },
    'Başlangıç Kasası': { en: 'Starting bankroll', nl: 'Startbankroll' },
    'Hedef Kasa': { en: 'Target bankroll', nl: 'Doelbankroll' },
    'Gerçek Kasa': { en: 'Actual bankroll', nl: 'Werkelijke bankroll' },
    'Günlük Değişim': { en: 'Daily change', nl: 'Dagelijkse wijziging' },
    'Günlük Büyüme': { en: 'Daily growth', nl: 'Dagelijkse groei' },
    'Risk': { en: 'Risk', nl: 'Risico' },
    'Kasa': { en: 'Bankroll', nl: 'Bankroll' },
    'Günlük Büyüme (%)': { en: 'Daily growth (%)', nl: 'Dagelijkse groei (%)' },
    '{day}. gün gerçek kasa': { en: 'Actual bankroll on day {day}', nl: 'Werkelijke bankroll op dag {day}' },
    'Elle girildi — silerseniz boş/otomatik değere döner': { en: 'Entered manually — clear it to go back to empty/automatic', nl: 'Handmatig ingevoerd — wis het om terug te gaan naar leeg/automatisch' },
    'Sonuçlanan kuponlardan otomatik hesaplandı': { en: 'Calculated automatically from settled bets', nl: 'Automatisch berekend uit afgehandelde weddenschappen' },
    'kasa.note': {
      tr: 'Oyundaki Kasa sütununa her günün kasasını yazabilirsiniz; yazılmayan geçmiş günler sonuçlanan kuponlardan otomatik dolar. Teorik hedef kasanın altında kalan günler kırmızı gösterilir. Tab veya Enter ile sonraki güne geçebilirsiniz.',
      en: 'You can enter each day\'s bankroll in the Bankroll in play column; past days you leave empty are filled automatically from settled bets. Days below the theoretical target bankroll are shown in red. Press Tab or Enter to move to the next day.',
      nl: 'Je kunt de bankroll van elke dag invullen in de kolom Bankroll in het spel; lege dagen in het verleden worden automatisch aangevuld uit afgehandelde weddenschappen. Dagen onder de theoretische doelbankroll worden rood weergegeven. Met Tab of Enter ga je naar de volgende dag.'
    },
    'Kasa {n}': { en: 'Bankroll {n}', nl: 'Bankroll {n}' },
    'Özel': { en: 'Custom', nl: 'Aangepast' },
    'Kasa Gelişim Grafiği: gerçek kasa ve teorik hedef kasa, 1-{n}. gün': { en: 'Bankroll growth chart: actual and theoretical target bankroll, days 1–{n}', nl: 'Groeigrafiek bankroll: werkelijke en theoretische doelbankroll, dag 1–{n}' },
    '{day}. gün · Gerçek Kasa: {real} · Teorik Hedef Kasa: {target}': { en: 'Day {day} · Actual bankroll: {real} · Theoretical target: {target}', nl: 'Dag {day} · Werkelijke bankroll: {real} · Theoretisch doel: {target}' },
    'Yeni kasanın adı:': { en: 'Name of the new bankroll:', nl: 'Naam van de nieuwe bankroll:' },
    '"{name}" kasasını silmek istediğinize emin misiniz? Diğer kasalarınız korunacaktır.': { en: 'Are you sure you want to delete the bankroll "{name}"? Your other bankrolls will be kept.', nl: 'Weet je zeker dat je de bankroll "{name}" wilt verwijderen? Je andere bankrolls blijven behouden.' },
    "{label} 0'dan büyük olmalıdır.": { en: '{label} must be greater than 0.', nl: '{label} moet groter zijn dan 0.' },
    'Başlangıç kasası': { en: 'The starting bankroll', nl: 'De startbankroll' },
    'Hedef kasa': { en: 'The target bankroll', nl: 'De doelbankroll' },
    'Geçerli bir kasa tutarı girin (örn: 68,66). Boş bırakırsanız gün boş kalır.': { en: 'Enter a valid amount (e.g. 68.66). If you leave it empty, the day stays empty.', nl: 'Voer een geldig bedrag in (bijv. 68,66). Als je het leeg laat, blijft de dag leeg.' },
    // Güven Payı (Sanal Kasa kilitleme)
    '🔒 Güven Payı': { en: '🔒 Safety margin', nl: '🔒 Veiligheidsbuffer' },
    'Kasan büyüdükçe bir kısmını kilitle. Kilitli para bir daha riske girmez ama hedefine sayılır.': { en: 'As your bankroll grows, lock part of it. Locked money is never at risk again, but it still counts towards your target.', nl: 'Zet een deel van je bankroll vast naarmate die groeit. Vastgezet geld loopt nooit meer risico, maar telt wel mee voor je doel.' },
    '🔴 Bugünkü kuponun {stake}, güvenli sınırın {limit}. Sınırı aştın.': { en: "🔴 Today's bet is {stake}, your safe limit is {limit}. You are over the limit.", nl: '🔴 Je inzet vandaag is {stake}, je veilige grens is {limit}. Je zit erboven.' },
    '🟡 Bugünkü kuponun {stake}, güvenli sınırın {limit}. Sınıra yaklaşıyorsun.': { en: "🟡 Today's bet is {stake}, your safe limit is {limit}. You are getting close.", nl: '🟡 Je inzet vandaag is {stake}, je veilige grens is {limit}. Je nadert de grens.' },
    '🟢 Bugünkü kuponun {stake}, güvenli sınırın {limit}. Rahatsın.': { en: "🟢 Today's bet is {stake}, your safe limit is {limit}. You're fine.", nl: '🟢 Je inzet vandaag is {stake}, je veilige grens is {limit}. Je zit goed.' },
    'Kuponun {stake} olur.': { en: 'Your bet becomes {stake}.', nl: 'Je inzet wordt {stake}.' },
    'Sonraki kilit: kasan {bank} olunca.': { en: 'Next lock: when your bankroll reaches {bank}.', nl: 'Volgende vastzetting: als je bankroll {bank} is.' },
    'Güvenli sınır nedir?': { en: 'What is the safe limit?', nl: 'Wat is de veilige grens?' },
    'ei.why': {
      tr: 'Güvenli sınır otomatik hesaplanır: başlangıç kasan ({start}) + kilitli paran. Başladığın tutar, kaybetmeyi baştan göze aldığın paradır; tek kupona bundan fazlasını koymak stresi artırır ve kararları bozar. Kazançtan sonra insanlar riski fark etmeden büyütür, bu yüzden sınır kazançla değil yalnızca kilitlediğin parayla büyür.',
      en: 'The safe limit is calculated automatically: your starting bankroll ({start}) + your locked money. The amount you started with is what you accepted you could lose; putting more than that on a single bet raises stress and hurts decisions. After a win people raise their risk without noticing, so the limit grows only with the money you lock, not with your winnings.',
      nl: 'De veilige grens wordt automatisch berekend: je startbankroll ({start}) + je vastgezette geld. Het bedrag waarmee je begon, is wat je bereid was te verliezen; meer dan dat op één weddenschap zetten verhoogt de stress en schaadt je beslissingen. Na winst verhogen mensen hun risico zonder het te merken, daarom groeit de grens alleen met het geld dat je vastzet, niet met je winst.'
    },
    'Oyundaki': { en: 'In play', nl: 'In het spel' },
    'Kilitli': { en: 'Locked', nl: 'Vastgezet' },
    'Toplam': { en: 'Total', nl: 'Totaal' },
    'İlk günün kasasını tabloya girince başlar.': { en: "Starts once you enter the first day's bankroll in the table.", nl: 'Begint zodra je de bankroll van de eerste dag in de tabel invult.' },
    '🏁 Hedefe ulaştın! Kasayı kapatıp kazancını koruyabilirsin.': { en: '🏁 You reached your target! You can close the bankroll and keep your profit.', nl: '🏁 Je hebt je doel bereikt! Je kunt de bankroll sluiten en je winst veiligstellen.' },
    '🔒 {amount} kilitle': { en: '🔒 Lock {amount}', nl: '🔒 {amount} vastzetten' },
    '📉 Zirveden %{pct} düştün. Bugün mola vermeyi düşün; kaybı hemen geri kazanmaya çalışma.': { en: "📉 You're down {pct}% from your peak. Consider taking a break today; don't try to win it back right away.", nl: '📉 Je staat {pct}% onder je piek. Overweeg vandaag een pauze; probeer het verlies niet meteen terug te winnen.' },
    'Kilitle': { en: 'Lock', nl: 'Vastzetten' },
    'Geri al': { en: 'Take back', nl: 'Terugnemen' },
    'Son işlemi geri al': { en: 'Undo last action', nl: 'Laatste actie ongedaan maken' },
    '0 ile {max} arasında bir tutar girin.': { en: 'Enter an amount between 0 and {max}.', nl: 'Voer een bedrag in tussen 0 en {max}.' },
    'Oyundaki kasadan ne kadar kilitlemek istiyorsun?': { en: 'How much of your bankroll in play do you want to lock?', nl: 'Hoeveel van je bankroll in het spel wil je vastzetten?' },
    'Kilitli paradan ne kadarını oyundaki kasaya geri almak istiyorsun?': { en: 'How much of the locked money do you want to put back into play?', nl: 'Hoeveel van het vastgezette geld wil je terugzetten in het spel?' },
    'Oyundaki Kasa': { en: 'Bankroll in play', nl: 'Bankroll in het spel' },
    'Oyundaki Kasa ({sym})': { en: 'Bankroll in play ({sym})', nl: 'Bankroll in het spel ({sym})' },
    'Kilitli ({sym})': { en: 'Locked ({sym})', nl: 'Vastgezet ({sym})' },
    'Kilitli: {v}': { en: 'Locked: {v}', nl: 'Vastgezet: {v}' },
    '{day}. gün · Oyundaki Kasa: {real} · Teorik Hedef Kasa: {target}': { en: 'Day {day} · Bankroll in play: {real} · Theoretical target: {target}', nl: 'Dag {day} · Bankroll in het spel: {real} · Theoretisch doel: {target}' },
    'Bu gün {amount} kilitlendi': { en: 'On this day {amount} was locked', nl: 'Op deze dag is {amount} vastgezet' },
    'Bu gün {amount} kasaya geri alındı': { en: 'On this day {amount} was put back into play', nl: 'Op deze dag is {amount} teruggezet in het spel' },
    'kasa.note.cashout': {
      tr: 'Oyundaki Kasa sütununa yalnızca oyundaki parayı yaz; kilitlediğin para Kilitli sütunundadır. Hedef ve toplam büyüme, oyundaki + kilitli paraya göre hesaplanır; grafikte kilitli para yeşil gösterilir.',
      en: 'Enter only the money in play in the Bankroll in play column; locked money is in the Locked column. Target and total growth are based on in play + locked; the chart shows locked money in green.',
      nl: 'Vul in de kolom Bankroll in het spel alleen het geld in het spel in; vastgezet geld staat in de kolom Vastgezet. Doel en totale groei zijn gebaseerd op in het spel + vastgezet; de grafiek toont vastgezet geld in groen.'
    },
    'Kasayı yeniden aç: girdileri düzeltip tekrar kapatabilirsin': { en: 'Reopen the bankroll: fix the entries and close it again', nl: 'Heropen de bankroll: corrigeer de invoer en sluit hem opnieuw' },
    '✏️ Yeniden aç': { en: '✏️ Reopen', nl: '✏️ Heropenen' },
    'Kapatılan kasayı sil': { en: 'Delete the closed bankroll', nl: 'Gesloten bankroll verwijderen' },
    '🗑️ Sil': { en: '🗑️ Delete', nl: '🗑️ Verwijderen' },
    '"{name}" kasası yeniden açılsın mı? Girdileri düzeltip tekrar kapatabilirsin.': { en: 'Reopen the "{name}" bankroll? You can fix the entries and close it again.', nl: 'Bankroll "{name}" heropenen? Je kunt de invoer corrigeren en hem opnieuw sluiten.' },
    '"{name}" kapatılan kasası kalıcı olarak silinsin mi? Bu işlem geri alınamaz.': { en: 'Permanently delete the closed bankroll "{name}"? This cannot be undone.', nl: 'Gesloten bankroll "{name}" definitief verwijderen? Dit kan niet ongedaan worden gemaakt.' },
  });
})(window);
