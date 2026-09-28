// Fikstür başlığında "olasılık çekirdeği" + lig yörüngesi (iş listesi #27, #30).
// Çekirdek: iç içe üç halka vurgu isabet oranlarıyla dolar (genel, 1X, 2.5+), ortada genel isabet yazar;
// değerler başarı bandıyla aynı kaynaktan (data/stats-summary.json). Tıklayınca İstatistikler açılır.
// Bayraklar sitedeki FLAGS çizimleri; tıklayınca setLeague() ile Fikstür o lige filtrelenir.
// Sayfa görünmüyorsa ya da başlık ekran dışındaysa çizim durur; "hareketi azalt" açıksa halkalar sabit.
(function () {
  'use strict';
  const host = document.getElementById('orb');
  if (!host) return;
  const T = (s, v) => (window._t ? window._t(s, v) : s);
  const I = () => window.I18N || { pct: x => '%' + x, dec: x => String(x).replace('.', ',') };
  const RM = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ORBIT = 0.0014;         // yörünge: rad / kare
  const FILL_MS = 1600;         // halkaların dolma süresi

  // ---------------- DOM
  host.innerHTML = `<div class="orb-ring"></div><div class="orb-ring r2"></div>
    <canvas class="orb-core" role="img" tabindex="0"></canvas>`;
  const cv = host.querySelector('canvas'), ctx = cv.getContext('2d');
  let DPR = 1, S = 200, R = 90;
  function resize() {
    DPR = Math.min(2, window.devicePixelRatio || 1);
    S = cv.clientWidth || 100;
    cv.width = Math.round(S * DPR); cv.height = Math.round(S * DPR);
    R = S * 0.42;
  }

  // ---------------- veri
  let rings = null, fillT0 = 0;
  const RING_DEF = [['all', 1], ['1X', -1.4], ['2.5+', 1.9]];   // [pazar, dönüş yönü/hızı]
  // [renk başı, renk sonu] halka başına; açık temada limon beyaz zeminde kaybolur, koyu tonlar
  const PAL = {
    dark: { rings: [['#2ed3c3', '#d9ff4a'], ['#66e3a1', '#2ed3c3'], ['#d9ff4a', '#66e3a1']], tick: '46,211,195', glow: 0.14 },
    light: { rings: [['#009f93', '#65a30d'], ['#0f9d58', '#009f93'], ['#65a30d', '#0f9d58']], tick: '0,159,147', glow: 0.05 },
  };
  const fmt = v => I().pct(I().dec(Number(v).toFixed(1)));
  fetch('data/stats-summary.json', { cache: 'no-cache' }).then(r => r.ok ? r.json() : null).then(s => {
    if (!s || !s.picks) return;
    const mk = s.markets || {};
    rings = RING_DEF.map(([k, sp]) => ({ k, sp, v: +(k === 'all' ? s.picks.pct : (mk[k] || {}).pct) / 100 }))
      .filter(r => r.v > 0);
    const all = rings.find(r => r.k === 'all');
    cv.setAttribute('aria-label', T('Vurgulanan tahminlerin isabet oranı {p}; istatistikleri aç', { p: all ? fmt(all.v * 100) : '—' }));
    cv.title = rings.map(r => `${r.k === 'all' ? T('Genel') : r.k} ${fmt(r.v * 100)}`).join(' · ');
    fillT0 = performance.now();
    draw(fillT0); kick();
  }).catch(() => {});

  // ---------------- çizim
  function draw(t) {
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, S, S);
    const c0 = S / 2, cs = getComputedStyle(host);
    const txt = cs.getPropertyValue('--text').trim() || '#f4f6f8';
    const P = /^#[0-3]/.test(txt) ? PAL.light : PAL.dark;          // koyu yazı = açık tema
    const halo = ctx.createRadialGradient(c0, c0, R * 0.2, c0, c0, R * 1.12);
    halo.addColorStop(0, `rgba(${P.tick},.18)`); halo.addColorStop(1, `rgba(${P.tick},0)`);
    ctx.beginPath(); ctx.arc(c0, c0, R * 1.12, 0, Math.PI * 2); ctx.fillStyle = halo; ctx.fill();
    const k = RM ? 1 : Math.min(1, (t - fillT0) / FILL_MS), e = 1 - Math.pow(1 - k, 3);
    const spin = RM ? 0 : t / 1000;
    // dış ölçek çizgileri
    ctx.save(); ctx.translate(c0, c0); ctx.rotate(-spin * 0.12);
    ctx.lineWidth = Math.max(1, R * 0.015);
    for (let i = 0; i < 48; i++) {
      const a = i / 48 * Math.PI * 2, big = i % 4 === 0, r2 = R * (big ? 1.1 : 1.06);
      ctx.beginPath(); ctx.moveTo(Math.cos(a) * R * 1.02, Math.sin(a) * R * 1.02); ctx.lineTo(Math.cos(a) * r2, Math.sin(a) * r2);
      ctx.strokeStyle = `rgba(${P.tick},${big ? 0.7 : 0.28})`; ctx.stroke();
    }
    ctx.restore();
    const lw = R * 0.12, radii = [0.93, 0.74, 0.56];
    ctx.lineWidth = lw; ctx.lineCap = 'round';
    radii.forEach((rr, i) => {
      const r = R * rr - lw / 2;
      ctx.beginPath(); ctx.arc(c0, c0, r, 0, Math.PI * 2); ctx.strokeStyle = 'rgba(128,140,150,.16)'; ctx.stroke();
      const g = rings && rings[i];
      if (!g || !e) return;
      const a0 = -Math.PI / 2 + spin * 0.35 * g.sp, a1 = a0 + Math.PI * 2 * g.v * e;
      const gr = ctx.createLinearGradient(c0 + Math.cos(a0) * r, c0 + Math.sin(a0) * r, c0 + Math.cos(a1) * r, c0 + Math.sin(a1) * r);
      const [c1, c2] = P.rings[i];
      gr.addColorStop(0, c1); gr.addColorStop(1, c2);
      ctx.save(); ctx.shadowColor = c2; ctx.shadowBlur = R * P.glow;
      ctx.beginPath(); ctx.arc(c0, c0, r, a0, a1); ctx.strokeStyle = gr; ctx.stroke();
      ctx.beginPath(); ctx.arc(c0 + Math.cos(a1) * r, c0 + Math.sin(a1) * r, lw * 0.32, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill();
      ctx.restore();
    });
    // orta: genel isabet, sayarak yükselir (renkler temadan: açık temada koyu yazı)
    const all = rings && rings.find(r => r.k === 'all');
    if (all) {
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillStyle = txt; ctx.font = `italic 900 ${R * 0.3}px Poppins, sans-serif`;
      ctx.fillText(fmt(all.v * 100 * e), c0, c0 - R * 0.02);
      ctx.fillStyle = cs.getPropertyValue('--muted').trim() || '#8e98a7'; ctx.font = `800 ${R * 0.105}px "Nunito Sans", sans-serif`;
      ctx.fillText(T('İSABET'), c0, c0 + R * 0.2);
    }
  }
  const openStats = () => { if (typeof setTab === 'function') setTab('stats'); };
  cv.addEventListener('click', openStats);
  cv.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openStats(); } });

  // ---------------- lig yörüngesi
  const ORDER = ['Premier League', 'LaLiga', 'Bundesliga', 'Serie A', 'Ligue 1', 'Eredivisie', 'Turkish Süper Lig', 'Primeira Liga', 'Belgian Pro League', 'Championship'];
  const SHORT = { 'Premier League': 'PREM', 'LaLiga': 'LALIGA', 'Bundesliga': 'BUNDES', 'Serie A': 'SERIE A', 'Ligue 1': 'LIGUE 1', 'Eredivisie': 'EREDIV',
    'Turkish Süper Lig': 'SÜPER LİG', 'Primeira Liga': 'PRIMEIRA', 'Belgian Pro League': 'PRO LEAGUE', 'Championship': 'CHAMP' };
  let flags = [];
  function buildFlags(leagues) {
    flags.forEach(b => b.remove());
    flags = leagues.map(lg => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'orb-flag';
      b.innerHTML = (typeof flag === 'function' ? flag(lg, 22) : '') + `<span>${SHORT[lg] || lg}</span>`;
      b.setAttribute('aria-label', T('Fikstürü {lg} ile filtrele', { lg }));
      b.title = lg;
      b.addEventListener('click', () => {
        if (typeof setTab === 'function' && document.documentElement.dataset.page !== 'pred') setTab('pred');
        if (typeof setLeague === 'function') setLeague(lg);
        const t = document.querySelector('#pane-pred .datectl');
        if (t) t.scrollIntoView({ block: 'start', behavior: RM ? 'auto' : 'smooth' });
      });
      host.appendChild(b);
      return b;
    });
  }
  let orbitA = 0, hover = false;
  host.addEventListener('pointerenter', e => { if (e.pointerType === 'mouse') hover = true; });
  host.addEventListener('pointerleave', () => { hover = false; });
  host.addEventListener('focusin', () => { hover = true; });
  host.addEventListener('focusout', () => { hover = false; });
  function placeFlags() {
    const w = host.clientWidth, h = host.clientHeight, rx = w * 0.44, ry = h * 0.3, cx = w / 2, cy = h / 2, n = flags.length || 1;
    flags.forEach((b, i) => {
      const a = orbitA + i * (Math.PI * 2 / n);
      const x = cx + Math.cos(a) * rx, y = cy + Math.sin(a) * ry, depth = Math.sin(a);
      const sc = 0.78 + 0.22 * (depth + 1) / 2;
      b.style.transform = `translate(${(x - b.offsetWidth / 2).toFixed(1)}px,${(y - b.offsetHeight / 2).toFixed(1)}px) scale(${sc.toFixed(3)})`;
      b.style.opacity = (0.45 + 0.55 * (depth + 1) / 2).toFixed(2);
      b.style.zIndex = depth > 0 ? 8 : 3;
      const behind = depth < 0 && Math.abs(x - cx) < w * 0.18;      // çekirdeğin arkasında: tıklanmasın
      b.style.pointerEvents = behind ? 'none' : 'auto';
      b.tabIndex = behind ? -1 : 0;
    });
  }

  // ---------------- döngü: yalnız görünürken
  let onScreen = true, raf = 0;
  if ('IntersectionObserver' in window) new IntersectionObserver(es => { onScreen = es[0].isIntersecting; kick(); }).observe(host);
  document.addEventListener('visibilitychange', kick);
  function running() { return !RM && onScreen && !document.hidden && host.offsetParent !== null; }
  function frame(t) {
    raf = 0;
    if (!running()) return;
    if (!hover) orbitA += ORBIT;
    draw(t); placeFlags();
    raf = requestAnimationFrame(frame);
  }
  function kick() { if (!raf && running()) raf = requestAnimationFrame(frame); }
  new MutationObserver(kick).observe(document.documentElement, { attributes: true, attributeFilter: ['data-page'] });
  // Poppins geç yüklenirse ortadaki yazı yedek fontla kalmasın
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => draw(performance.now()));

  // ---------------- bayraklar veriden (fikstürde olan ligler)
  let shown = '';
  function refresh() {
    const data = (window.__data || []).filter(x => x && x.kickoff_utc);
    if (!data.length) return;
    const leagues = ORDER.filter(lg => data.some(x => x.league === lg));
    const sig = leagues.join('|');
    if (sig === shown) return;
    shown = sig;
    buildFlags(leagues);
    placeFlags(); kick();
  }

  resize(); draw(performance.now());
  window.addEventListener('resize', () => { resize(); draw(performance.now()); placeFlags(); });
  const rows = document.getElementById('rows');
  if (rows) new MutationObserver(refresh).observe(rows, { childList: true });
  refresh(); kick();
})();
