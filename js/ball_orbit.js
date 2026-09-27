// Fikstür başlığında dönen futbol topu + lig yörüngesi (iş listesi #27).
// Top: kesik ikosahedron (12 beşgen, 20 altıgen) küreye izdüşülerek canvas'a çizilir, kütüphane yok.
// Bayraklar sitedeki FLAGS çizimleri; tıklayınca setLeague() ile Fikstür o lige filtrelenir.
// Kutulardaki sayılar canlı veriden: bu ayki maç, vurgu isabeti (data/stats-summary.json), sıradaki gün.
// Sayfa görünmüyorsa ya da başlık ekran dışındaysa çizim durur; "hareketi azalt" açıksa top sabit.
(function () {
  'use strict';
  const host = document.getElementById('orb');
  if (!host) return;
  const T = (s, v) => (window._t ? window._t(s, v) : s);
  const RM = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  const SPIN = 0.0032;          // top: rad / kare (yavaş)
  const ORBIT = 0.0014;         // yörünge: rad / kare

  // ---------------- geometri
  const phi = (1 + Math.sqrt(5)) / 2;
  const norm = v => { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; };
  const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const arc = (a, b, n) => { const o = []; for (let k = 0; k <= n; k++) o.push(norm(lerp(a, b, k / n))); return o; };
  const ico = [];                                   // (0,±1,±φ) (±1,±φ,0) (±φ,0,±1)
  for (const s1 of [1, -1]) for (const s2 of [1, -1]) { ico.push([0, s1, s2 * phi]); ico.push([s1, s2 * phi, 0]); ico.push([s2 * phi, 0, s1]); }
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
  const nb = ico.map((v, i) => ico.map((_, j) => j).filter(j => j !== i && Math.abs(d2(v, ico[j]) - 4) < 1e-6));
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  // beşgen köşeleri: köşeden komşuya giden kenarın 1/3'ü; açıya göre sıralı
  const pents = ico.map((v, i) => {
    const c = norm(v), u = norm(cross(c, Math.abs(c[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0])), w = cross(c, u);
    const cs = nb[i].map(j => { const p = norm(lerp(v, ico[j], 1 / 3)); return { p, a: Math.atan2(dot(p, w), dot(p, u)) }; })
      .sort((x, y) => x.a - y.a).map(o => o.p);
    const poly = [];
    for (let k = 0; k < 5; k++) poly.push(...arc(cs[k], cs[(k + 1) % 5], 6).slice(0, -1));
    return { c, poly };
  });
  // dikişler: her ikosahedron kenarının orta üçte biri (iki beşgeni bağlayan altıgen kenarı)
  const seams = [];
  ico.forEach((v, i) => nb[i].forEach(j => { if (j > i) seams.push(arc(norm(lerp(v, ico[j], 1 / 3)), norm(lerp(v, ico[j], 2 / 3)), 6)); }));

  // ---------------- DOM
  host.innerHTML = `<div class="orb-ring"></div><div class="orb-ring r2"></div><div class="orb-shadow"></div>
    <canvas class="orb-ball" role="img" aria-label="${T('Dönen futbol topu; sürükleyerek çevirebilirsin')}"></canvas>
    <div class="orb-stat s1"><small>${T('Bu ay')}</small><b id="orbN1">—</b><em>${T('maç')}</em></div>
    <div class="orb-stat s2"><small>${T('Vurgu isabeti')}</small><b id="orbN2">—</b></div>
    <div class="orb-stat s3"><small>${T('Sıradaki gün')}</small><b id="orbN3">—</b><em id="orbN3e"></em></div>`;
  const cv = host.querySelector('canvas'), ctx = cv.getContext('2d');
  let DPR = 1, S = 200, R = 90;
  function resize() {
    DPR = Math.min(2, window.devicePixelRatio || 1);
    S = cv.clientWidth || 180;
    cv.width = Math.round(S * DPR); cv.height = Math.round(S * DPR);
    R = S * 0.46;
  }
  let ax = -0.35, ay = 0, vx = 0, vy = SPIN;
  const L = norm([-0.5, -0.65, 0.75]);
  function rot(p) {
    const cy = Math.cos(ay), sy = Math.sin(ay), cx = Math.cos(ax), sx = Math.sin(ax);
    const x = p[0] * cy + p[2] * sy, z = -p[0] * sy + p[2] * cy, y = p[1];
    return [x, y * cx - z * sx, y * sx + z * cx];
  }
  function proj(p) {                     // arkadaki nokta ufka itilir: yarım küredeki kenarlar düzgün kırpılır
    let [x, y, z] = p;
    if (z < 0) { const l = Math.hypot(x, y) || 1; x /= l; y /= l; }
    return [S / 2 + x * R, S / 2 + y * R];
  }
  function path(pts) { pts.forEach((p, k) => { const q = proj(p); k ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }); }
  function draw() {
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, S, S);
    const c0 = S / 2;
    const g = ctx.createRadialGradient(c0 - R * 0.38, c0 - R * 0.42, R * 0.08, c0, c0, R);
    g.addColorStop(0, '#ffffff'); g.addColorStop(0.55, '#eef1f4'); g.addColorStop(1, '#b9c3cc');
    ctx.beginPath(); ctx.arc(c0, c0, R, 0, Math.PI * 2); ctx.fillStyle = g; ctx.fill();
    ctx.save(); ctx.beginPath(); ctx.arc(c0, c0, R, 0, Math.PI * 2); ctx.clip();
    ctx.lineWidth = Math.max(1, R * 0.012); ctx.lineCap = 'round';
    seams.forEach(pts => {
      const rp = pts.map(rot);
      if (rp[3][2] < 0) return;
      ctx.beginPath(); path(rp);
      ctx.strokeStyle = `rgba(40,56,70,${0.25 + 0.35 * rp[3][2]})`; ctx.stroke();
    });
    pents.map(pt => ({ pt, c: rot(pt.c) })).filter(o => o.c[2] > -0.35).sort((a, b) => a.c[2] - b.c[2]).forEach(({ pt, c }) => {
      ctx.beginPath(); path(pt.poly.map(rot)); ctx.closePath();
      const sh = Math.round(18 + 46 * Math.max(0, dot(c, L)));
      ctx.fillStyle = `rgb(${sh},${sh + 6},${sh + 14})`; ctx.fill();
      ctx.strokeStyle = 'rgba(20,30,40,.55)'; ctx.lineWidth = Math.max(1, R * 0.01); ctx.stroke();
    });
    ctx.restore();
    const hl = ctx.createRadialGradient(c0 - R * 0.42, c0 - R * 0.46, 0, c0 - R * 0.42, c0 - R * 0.46, R * 0.55);
    hl.addColorStop(0, 'rgba(255,255,255,.55)'); hl.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.beginPath(); ctx.arc(c0, c0, R, 0, Math.PI * 2); ctx.fillStyle = hl; ctx.fill();
    const rim = ctx.createRadialGradient(c0, c0, R * 0.7, c0, c0, R);
    rim.addColorStop(0, 'rgba(19,41,61,0)'); rim.addColorStop(1, 'rgba(19,41,61,.28)');
    ctx.fillStyle = rim; ctx.fill();
  }

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
    const w = host.clientWidth, h = host.clientHeight, rx = w * 0.42, ry = h * 0.2, cx = w / 2, cy = h / 2, n = flags.length || 1;
    flags.forEach((b, i) => {
      const a = orbitA + i * (Math.PI * 2 / n);
      const x = cx + Math.cos(a) * rx, y = cy + Math.sin(a) * ry, depth = Math.sin(a);
      const sc = 0.78 + 0.22 * (depth + 1) / 2;
      b.style.transform = `translate(${(x - b.offsetWidth / 2).toFixed(1)}px,${(y - b.offsetHeight / 2).toFixed(1)}px) scale(${sc.toFixed(3)})`;
      b.style.opacity = (0.45 + 0.55 * (depth + 1) / 2).toFixed(2);
      b.style.zIndex = depth > 0 ? 8 : 3;
      const behind = depth < 0 && Math.abs(x - cx) < w * 0.2;      // topun arkasında: tıklanmasın
      b.style.pointerEvents = behind ? 'none' : 'auto';
      b.tabIndex = behind ? -1 : 0;
    });
  }

  // ---------------- sürükleyerek çevirme
  let drag = null;
  cv.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY }; cv.setPointerCapture(e.pointerId); vx = vy = 0; });
  cv.addEventListener('pointermove', e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag = { x: e.clientX, y: e.clientY };
    vy = dx * 0.006; vx = -dy * 0.006; ay += vy; ax += vx;
    if (RM) draw();
  });
  const endDrag = () => { drag = null; };
  cv.addEventListener('pointerup', endDrag);
  cv.addEventListener('pointercancel', endDrag);

  // ---------------- döngü: yalnız görünürken
  let onScreen = true, raf = 0;
  if ('IntersectionObserver' in window) new IntersectionObserver(es => { onScreen = es[0].isIntersecting; kick(); }).observe(host);
  document.addEventListener('visibilitychange', kick);
  function running() { return !RM && onScreen && !document.hidden && host.offsetParent !== null; }
  function frame() {
    raf = 0;
    if (!running()) return;
    if (!drag) { vy += (SPIN - vy) * 0.02; vx += (0 - vx) * 0.03; ay += vy; ax += vx; ax += (-0.35 - ax) * 0.01; }
    if (!hover) orbitA += ORBIT;
    draw(); placeFlags();
    raf = requestAnimationFrame(frame);
  }
  function kick() { if (!raf && running()) raf = requestAnimationFrame(frame); }
  new MutationObserver(kick).observe(document.documentElement, { attributes: true, attributeFilter: ['data-page'] });

  // ---------------- canlı sayılar
  const istDay = iso => new Date(iso).toLocaleDateString('sv-SE', { timeZone: 'Europe/Istanbul' });
  function countTo(el, v, fmt) {
    if (RM) { el.textContent = fmt(v); return; }
    const t0 = performance.now();
    (function step(now) { const k = Math.min(1, (now - t0) / 1100), e = 1 - Math.pow(1 - k, 3); el.textContent = fmt(v * e); if (k < 1) requestAnimationFrame(step); })(t0);
  }
  let shown = '';
  function refresh() {
    const data = (window.__data || []).filter(x => x && x.kickoff_utc);
    if (!data.length) return;
    const leagues = ORDER.filter(lg => data.some(x => x.league === lg));
    const sig = leagues.join('|') + data.length;
    if (sig === shown) return;
    shown = sig;
    buildFlags(leagues);
    const now = Date.now();
    const up = data.filter(x => new Date(x.kickoff_utc).getTime() > now);
    countTo(document.getElementById('orbN1'), up.length, v => Math.round(v).toLocaleString((window.I18N && I18N.locale) || 'tr-TR'));
    if (up.length) {
      const day = up.map(x => istDay(x.kickoff_utc)).sort()[0];
      const n = up.filter(x => istDay(x.kickoff_utc) === day).length;
      const d = new Date(day + 'T12:00:00Z');
      countTo(document.getElementById('orbN3'), d.getUTCDate(), v => String(Math.round(v)));
      document.getElementById('orbN3e').textContent =
        d.toLocaleDateString((window.I18N && I18N.locale) || 'tr-TR', { month: 'long', timeZone: 'UTC' }) + ' · ' + T('{n} maç', { n });
    }
    placeFlags(); kick();
  }
  fetch('data/stats-summary.json?ts=' + Date.now(), { cache: 'no-store' }).then(r => (r.ok ? r.json() : null)).then(s => {
    const pct = s && s.picks && +s.picks.pct;
    if (!(pct > 0)) return;
    const I = window.I18N, lang = (I && I.lang) || 'tr';
    const f = v => { const t = I && I.dec ? I.dec(v.toFixed(1)) : v.toFixed(1); return lang === 'tr' ? '%' + t : t + '%'; };
    countTo(document.getElementById('orbN2'), pct, f);
  }).catch(() => {});

  resize(); draw();
  window.addEventListener('resize', () => { resize(); draw(); placeFlags(); });
  const rows = document.getElementById('rows');
  if (rows) new MutationObserver(refresh).observe(rows, { childList: true });
  refresh(); kick();
})();
