// BETAVUS arayüz kolaylıkları (iş listesi #26): A Aurora teması + 1 mobil alt menü · 2 günün öne
// çıkanları · 3 yükleniyor iskeleti ve dolan göstergeler · 4 ilk giriş rehberi.
// index.html'deki genel fonksiyonları (setTab, openSheet, dcAll, isLimitedData, HL, dayLabel, _t)
// yalnızca çalışma anında kullanır; sayfanın ana betiğinden sonra yüklenmesi gerekmez.
(function () {
  'use strict';
  const T = (s, v) => (window._t ? window._t(s, v) : s);
  const RM = () => window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $ = id => document.getElementById(id);
  const get = k => { try { return localStorage.getItem('betavus.' + k); } catch (e) { return null; } };
  const put = (k, v) => { try { localStorage.setItem('betavus.' + k, v); } catch (e) { /* özel pencere */ } };
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // ---------------------------------------------------------------- 3 · iskelet
  function skeleton(n) {
    const row = '<div class="row sk-row" aria-hidden="true"><div class="match"><i class="sk sk-l"></i><i class="sk sk-t"></i><i class="sk sk-k"></i></div>' +
      '<div class="prob"><i class="sk sk-p"></i></div>'.repeat(6) + '</div>';
    return `<div class="sk-wrap" data-skeleton><span class="sr-only">${T('Fikstürler yükleniyor…')}</span>${row.repeat(n || 6)}</div>`;
  }
  function skeletonBlocks() {
    return `<div class="sk-wrap" data-skeleton><span class="sr-only">${T('İstatistikler yükleniyor…')}</span>` +
      '<div class="sk-kpis"><i class="sk sk-kpi"></i><i class="sk sk-kpi"></i><i class="sk sk-kpi"></i></div>' +
      '<i class="sk sk-block"></i><i class="sk sk-block short"></i></div>';
  }

  // ---------------------------------------------------------------- dolan göstergeler
  // Sekme açılınca (ve veri ilk geldiğinde) kısa bir pencere açılır: o sırada görünen / eklenen
  // çubuklar ve halkalar CSS ile sıfırdan dolar, büyük sayılar sıfırdan sayarak yükselir.
  const COUNT_SEL = '.hlb-v,.st-kpi .v,.cs-sum b,.cs-pick em,.ei-sum b,.tdy-ring b';
  const windows = new WeakMap();
  let counted = new WeakSet();

  function parseNum(txt) {
    const m = /\d[\d.,]*/.exec(txt);
    if (!m) return null;
    const tok = m[0].replace(/[.,]$/, '');
    const seps = tok.replace(/\d/g, '');
    let dec = '', th = '', decimals = 0;
    if (seps) {
      const last = Math.max(tok.lastIndexOf('.'), tok.lastIndexOf(',')), ch = tok[last], tail = tok.length - last - 1;
      if (tail === 3 && seps.split('').every(c => c === ch)) th = ch;          // 12.660 · 25,000 → binlik
      else { dec = ch; decimals = tail; th = seps.split('').find(c => c !== ch) || ''; }   // 89,5 · 25,000.00
    }
    const clean = tok.split(th || '\u0000').join('').replace(dec || '\u0000', '.');
    const v = parseFloat(clean);
    return isFinite(v) ? { v, start: m.index, len: tok.length, dec, th, decimals } : null;
  }
  function fmtNum(v, p) {
    const s = v.toFixed(p.decimals);
    let [i, d] = s.split('.');
    if (p.th) i = i.replace(/\B(?=(\d{3})+(?!\d))/g, p.th);
    return d ? i + p.dec + d : i;
  }
  function countUp(el) {
    if (counted.has(el) || el.children.length) return;
    counted.add(el);
    const txt = el.textContent, p = parseNum(txt);
    if (!p || !(p.v > 0) || RM()) return;
    const pre = txt.slice(0, p.start), post = txt.slice(p.start + p.len), t0 = performance.now(), dur = 950;
    const step = now => {
      if (!el.isConnected || el.textContent !== pre + fmtNum(cur, p) + post) return;   // yeniden çizildi: bırak
      const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      cur = k >= 1 ? p.v : p.v * e;
      el.textContent = pre + fmtNum(cur, p) + post;
      if (k < 1) requestAnimationFrame(step);
    };
    let cur = 0;
    el.textContent = pre + fmtNum(0, p) + post;
    requestAnimationFrame(step);
  }
  function openWindow(pane, ms) {
    if (!pane) return;
    counted = new WeakSet();
    pane.classList.remove('fillin');
    void pane.offsetWidth;                     // animasyonu yeniden başlat
    pane.classList.add('fillin');
    clearTimeout(windows.get(pane));
    windows.set(pane, setTimeout(() => pane.classList.remove('fillin'), ms || 2600));
    pane.querySelectorAll(COUNT_SEL).forEach(countUp);
  }
  function watchPane(pane) {
    new MutationObserver(recs => {
      if (!pane.classList.contains('fillin')) return;
      recs.forEach(r => r.addedNodes.forEach(n => {
        if (n.nodeType !== 1) return;
        if (n.matches && n.matches(COUNT_SEL)) countUp(n);
        n.querySelectorAll && n.querySelectorAll(COUNT_SEL).forEach(countUp);
      }));
    }).observe(pane, { childList: true, subtree: true });
  }
  const activePane = () => $('pane-' + (document.documentElement.dataset.page || 'pred'));

  // ---------------------------------------------------------------- 2 · günün öne çıkanları
  function istDay(iso) { return new Date(iso).toLocaleDateString('sv-SE', { timeZone: 'Europe/Istanbul' }); }
  function bestPick(x) {
    if (typeof isLimitedData === 'function' && isLimitedData(x)) return null;
    if (x.live) return null;
    const c = [];
    try {
      [['0.5+', HL.o05], ['1.5+', HL.o15], ['2.5+', HL.o25]].forEach(([m, h]) => {
        const v = Number(x[h.key]);
        if (x[h.key] != null && v >= h.min) c.push({ m, p: v });
      });
      if (typeof dcAll === 'function' && typeof dcMin === 'function') {
        const d = dcAll(x);
        // Çifte Şans olasılıkları yüzde ölçeğinde (86,9), gol çizgileri 0–1 (0,96): 0–1'e çevrilir
        if (d && d.ok && d.v) ['1X', '12', 'X2'].forEach(k => { if (d.v[k] >= dcMin(k)) c.push({ m: k, p: d.v[k] > 1 ? d.v[k] / 100 : d.v[k] }); });
      }
    } catch (e) { return null; }
    if (!c.length) return null;
    return c.sort((a, b) => b.p - a.p)[0];
  }
  function renderToday() {
    const host = $('todayCard');
    if (!host) return;
    const data = window.__data || [], now = Date.now();
    const up = data.filter(x => x && x.kickoff_utc && new Date(x.kickoff_utc).getTime() > now)
      .map(x => ({ x, b: bestPick(x) })).filter(o => o.b);
    if (!up.length) { host.hidden = true; host.innerHTML = ''; return; }
    const day = up.map(o => istDay(o.x.kickoff_utc)).sort()[0];
    const top = up.filter(o => istDay(o.x.kickoff_utc) === day).sort((a, b) => b.b.p - a.b.p).slice(0, 3);
    const today = istDay(new Date().toISOString()) === day;
    const dl = typeof dayLabel === 'function' ? dayLabel(day) : day;
    const pctTxt = p => (window.I18N && I18N.pctS ? I18N.pctS((Math.floor(p * 1000 + 1e-9) / 10).toFixed(1)) : Math.round(p * 100) + '%');
    const time = iso => new Date(iso).toLocaleTimeString(window.I18N && I18N.locale || 'tr-TR', { hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Istanbul' });
    const sig = day + top.map(o => o.x.match_id + o.b.m + o.b.p).join();
    if (host.dataset.sig === sig && !host.hidden) return;
    host.dataset.sig = sig;
    host.hidden = false;
    host.innerHTML = `<div class="tdy-h"><h2>${T('Günün öne çıkanları')}</h2><span>${today ? T('Bugün · en yüksek olasılıklı {n} vurgu', { n: top.length }) : T('{day} · en yüksek olasılıklı {n} vurgu', { day: dl, n: top.length })}</span></div>
      <div class="tdy-picks">${top.map(o => `<button type="button" class="tdy-pick" data-mid="${esc(o.x.match_id || '')}" aria-label="${esc(T('Maç detayını aç'))}: ${esc(o.x.home)} — ${esc(o.x.away)}">
        <span class="tdy-ring" style="--p:${Math.round(o.b.p * 1000) / 10}"><b>${pctTxt(o.b.p)}</b></span>
        <span class="tdy-txt"><b>${esc(o.x.home)} — ${esc(o.x.away)}</b><small>${esc(o.x.league)} · ${time(o.x.kickoff_utc)} · <em>${esc(o.b.m)}</em></small></span></button>`).join('')}</div>`;
  }

  // ---------------------------------------------------------------- 1 · mobil alt menü
  const NAV = [
    ['pred', 'Fikstür', '<rect x="4" y="3" width="16" height="18" rx="3"/><path d="M8 8h8M8 12h8M8 16h5"/>'],
    ['stats', 'İstatistikler', '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>'],
    ['plan', 'Sanal Kasa', '<path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H18v4"/><rect x="3" y="7" width="18" height="13" rx="2.5"/><path d="M16 13.5h2"/>'],
    ['faq', 'FAQ', '<circle cx="12" cy="12" r="9"/><path d="M9.6 9.3a2.5 2.5 0 1 1 3.4 2.4c-.7.3-1 .8-1 1.5M12 17h.01"/>']
  ];
  function buildNav() {
    if ($('botNav')) return;
    const nav = document.createElement('nav');
    nav.id = 'botNav';
    nav.className = 'botnav';
    nav.setAttribute('aria-label', T('Sayfalar'));
    nav.innerHTML = NAV.map(([k, l, svg]) => `<button type="button" data-pg="${k}"><svg viewBox="0 0 24 24" aria-hidden="true">${svg}</svg><span>${T(l)}</span></button>`).join('');
    nav.addEventListener('click', e => {
      const b = e.target.closest('button[data-pg]');
      if (b && typeof setTab === 'function') { setTab(b.dataset.pg); window.scrollTo({ top: 0, behavior: RM() ? 'auto' : 'smooth' }); }
    });
    document.body.appendChild(nav);
    document.body.classList.add('has-botnav');
    syncNav();
  }
  function syncNav() {
    const pg = document.documentElement.dataset.page;
    document.querySelectorAll('#botNav button').forEach(b => {
      const on = b.dataset.pg === pg;
      b.classList.toggle('on', on);
      if (on) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current');
    });
  }

  // ---------------------------------------------------------------- 4 · sekme rehberleri
  // Her sekmenin kendi kısa turu: sekme ilk açıldığında bir kez kendiliğinden başlar; köşedeki
  // "Rehber" düğmesi ve SSS'teki düğmeler o sekmenin turunu yeniden açar.
  const q = s => document.querySelector(s);
  const vis = el => (el && el.offsetParent !== null ? el : null);
  const TOURS = {
    pred: [
      { sel: () => vis(q('#rows .pill.hot')) || vis($('rows')), t: 'Vurgu nedir?', p: 'tour.p1' },
      { sel: () => vis($('todayCard')), t: 'Günün öne çıkanları', p: 'tour.p2' },
      { sel: () => vis($('dayStrip')), t: 'Gün seçimi', p: 'tour.pred3' },
      { sel: () => vis(q('#pane-pred .datectl')), t: 'Arama ve filtreler', p: 'tour.pred4' }
    ],
    stats: [
      { sel: () => vis(q('#pane-stats .st-kpis')), t: 'Geçmiş başarı', p: 'tour.st1' },
      { sel: () => (vis($('stLeague')) || vis($('stSeason'))) && ($('stSeason') || $('stLeague')).parentElement, t: 'Lig, sezon ve takım', p: 'tour.st2' },
      { sel: () => vis($('stCal')), t: 'Model ne kadar güvenilir?', p: 'tour.st3' },
      { sel: () => vis(q('#pane-stats .st-list')), t: 'Maç maç sonuçlar', p: 'tour.st4' }
    ],
    plan: [
      { sel: () => vis(q('#pane-plan .kasa-sheet')), t: 'Kasanı kur', p: 'tour.pl1' },
      { sel: () => vis($('cpnSuggest')), t: 'Kupon önerisi', p: 'tour.pl2' },
      { sel: () => vis($('eiCoachCard')), t: 'Güven Payı', p: 'tour.pl3' },
      { sel: () => vis($('kasaSimCard')), t: 'Günlük kasa tablosu', p: 'tour.pl4' }
    ]
  };
  const tourKeyOf = tab => 'tour_' + tab + '_v1';
  const seen = tab => !!(get(tourKeyOf(tab)) || (tab === 'pred' && get('tour_v1')));
  let tourEl = null, tourI = 0, tourSteps = [], tourTab = null;
  function endTour() {
    if (tourTab) put(tourKeyOf(tourTab), '1');
    if (tourEl) { tourEl.remove(); tourEl = null; }
    tourTab = null;
    window.removeEventListener('resize', placeTour);
    window.removeEventListener('scroll', placeTour, true);
    document.removeEventListener('keydown', tourKey);
  }
  function tourKey(e) { if (e.key === 'Escape') endTour(); }
  function placeTour() {
    if (!tourEl) return;
    const s = tourSteps[tourI], el = s && s.sel();
    const ring = tourEl.querySelector('.tour-ring'), tip = tourEl.querySelector('.tour-tip');
    if (!el) return;
    const r = el.getBoundingClientRect(), pad = 6, vh = window.innerHeight;
    const top0 = Math.max(r.top, 4), bot0 = Math.min(r.bottom, vh - 4);        // çok uzun öğede görünen kısmı
    Object.assign(ring.style, { left: r.left - pad + 'px', top: top0 - pad + 'px', width: r.width + pad * 2 + 'px', height: Math.max(0, bot0 - top0) + pad * 2 + 'px' });
    const vw = document.documentElement.clientWidth, tw = Math.min(340, vw - 24), th = tip.offsetHeight || 160;
    const bottomSafe = document.body.classList.contains('has-botnav') && vw <= 650 ? 70 : 0;
    let top = bot0 + 14;
    if (top + th > vh - 12 - bottomSafe) top = top0 - th - 14;
    if (top < 12) top = Math.max(12, vh - th - 12 - bottomSafe);
    tip.style.width = tw + 'px';
    tip.style.left = Math.max(12, Math.min(vw - tw - 12, r.left + r.width / 2 - tw / 2)) + 'px';
    tip.style.top = top + 'px';
  }
  function showStep(i) {
    tourI = i;
    const s = tourSteps[i], el = s.sel();
    if (el) {
      const r = el.getBoundingClientRect();
      el.scrollIntoView({ block: r.height > window.innerHeight * 0.55 ? 'start' : 'center', behavior: 'auto' });
    }
    const tip = tourEl.querySelector('.tour-tip');
    tip.querySelector('.tour-n').textContent = `${i + 1}/${tourSteps.length}`;
    tip.querySelector('b').textContent = T(s.t);
    tip.querySelector('p').innerHTML = T(s.p);
    tip.querySelector('.tour-prev').disabled = i === 0;
    tip.querySelector('.tour-next').textContent = i === tourSteps.length - 1 ? T('Bitir') : T('İleri');
    tip.querySelectorAll('.tour-dots i').forEach((d, k) => d.classList.toggle('on', k === i));
    requestAnimationFrame(placeTour);
    tip.querySelector('.tour-next').focus({ preventScroll: true });
  }
  function startTour(tab) {
    tab = tab || document.documentElement.dataset.page || 'pred';
    if (!TOURS[tab]) tab = 'pred';
    if (tourEl) endTour();
    if (document.documentElement.dataset.page !== tab && typeof setTab === 'function') setTab(tab);
    tourSteps = TOURS[tab].filter(s => s.sel());
    if (!tourSteps.length) return false;
    tourTab = tab;
    tourEl = document.createElement('div');
    tourEl.className = 'tour';
    tourEl.dataset.tab = tab;
    tourEl.innerHTML = `<div class="tour-ring"></div><div class="tour-tip" role="dialog" aria-modal="true" aria-labelledby="tourT">
      <div class="tour-top"><span class="tour-n"></span><button type="button" class="tour-skip">${T('Atla')}</button></div>
      <b id="tourT"></b><p></p>
      <div class="tour-nav"><span class="tour-dots">${tourSteps.map(() => '<i></i>').join('')}</span>
        <button type="button" class="tour-prev">${T('Geri')}</button><button type="button" class="tour-next">${T('İleri')}</button></div></div>`;
    document.body.appendChild(tourEl);
    tourEl.querySelector('.tour-skip').onclick = endTour;
    tourEl.querySelector('.tour-prev').onclick = () => tourI > 0 && showStep(tourI - 1);
    tourEl.querySelector('.tour-next').onclick = () => (tourI < tourSteps.length - 1 ? showStep(tourI + 1) : endTour());
    window.addEventListener('resize', placeTour);
    window.addEventListener('scroll', placeTour, true);
    document.addEventListener('keydown', tourKey);
    showStep(0);
    return true;
  }
  // Sekme ilk açıldığında: ilk adımın hedefi çizilene kadar (en çok ~6 sn) bekler, sonra bir kez başlar
  let autoTimer = null;
  function maybeTour() {
    clearTimeout(autoTimer);
    const tab = document.documentElement.dataset.page;
    if (!TOURS[tab] || seen(tab) || tourEl) return;
    if (navigator.webdriver && !get('tour_force')) return;   // otomasyon testleri: rehber tıklamaları engellemesin
    let n = 0;
    const still = () => document.documentElement.dataset.page === tab && !seen(tab) && !tourEl;
    const tick = () => {
      if (!still()) return;
      if ($('gate') && !$('gate').hidden) { autoTimer = setTimeout(tick, 1000); return; }
      const ready = tab === 'pred' ? !!q('#rows .row[data-mid]') : !!TOURS[tab][0].sel();
      if (ready) { autoTimer = setTimeout(() => { if (still()) startTour(tab); }, 700); return; }
      if (++n < 20) autoTimer = setTimeout(tick, 300);
    };
    autoTimer = setTimeout(tick, 300);
  }
  function buildHelpBtn() {
    if ($('tourBtn')) return;
    const b = document.createElement('button');
    b.type = 'button';
    b.id = 'tourBtn';
    b.className = 'tour-fab';
    b.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M9.6 9.3a2.5 2.5 0 1 1 3.4 2.4c-.7.3-1 .8-1 1.5M12 17h.01"/></svg><span>${T('Rehber')}</span>`;
    b.title = T('Bu sayfanın rehberi');
    b.addEventListener('click', () => startTour(document.documentElement.dataset.page));
    document.body.appendChild(b);
    syncHelpBtn();
  }
  function syncHelpBtn() {
    const b = $('tourBtn');
    if (b) b.hidden = !TOURS[document.documentElement.dataset.page];
  }

  // ---------------------------------------------------------------- kurulum
  function init() {
    buildNav();
    buildHelpBtn();
    const sl = document.querySelector('#statsBody > .loading');
    if (sl) sl.outerHTML = skeletonBlocks();
    ['pred', 'stats', 'plan', 'faq'].forEach(k => { const p = $('pane-' + k); if (p) watchPane(p); });
    new MutationObserver(() => {
      syncNav(); syncHelpBtn(); openWindow(activePane());
      if (tourEl && tourEl.dataset.tab !== document.documentElement.dataset.page) endTour();
      maybeTour();
    }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-page'] });
    const rows = $('rows');
    if (rows) {
      let wasSkeleton = !!rows.querySelector('[data-skeleton]');
      new MutationObserver(() => {
        const sk = !!rows.querySelector('[data-skeleton]');
        if (wasSkeleton && !sk) { openWindow($('pane-pred')); maybeTour(); }
        wasSkeleton = sk;
        renderToday();
      }).observe(rows, { childList: true });
    }
    const card = $('todayCard');
    if (card) card.addEventListener('click', e => {
      const b = e.target.closest('.tdy-pick');
      if (b && b.dataset.mid && typeof openSheet === 'function') openSheet(b.dataset.mid);
    });
    document.querySelectorAll('[data-tour]').forEach(b => b.addEventListener('click', () => startTour(b.dataset.tour)));
    if (document.documentElement.dataset.page) { openWindow(activePane()); maybeTour(); }
  }

  window.BV_UX = { skeleton, skeletonBlocks, renderToday, startTour, openWindow, _parseNum: parseNum, _fmtNum: fmtNum };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
