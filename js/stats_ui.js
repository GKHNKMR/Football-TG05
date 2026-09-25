// İstatistikler sekmesi (#pane-stats)
// Son 5 tamamlanmış sezonun + güncel sezonun oynanmış (walk-forward, sızıntısız) TÜM maçları: her maç için
// 0.5+ / 1.5+ / 2.5+ / 1X / 12 / X2 model olasılığı ve gerçekleşen skor.
// Veri: data/stats-5season.json (scripts/build_cifte_backtest.py üretir).
// Ana sayfa şeridi: data/stats-summary.json (yalnızca vurgulanan tahminler).
(function (root) {
  'use strict';

  const DATA_URL = 'data/stats-5season.json';
  const SUMMARY_URL = 'data/stats-summary.json';
  const PAGE = 50;
  // [ad, satır indeksi, vurgu eşiği (binde), olay gerçekleşti mi]
  const MARKETS = [
    ['0.5+', 6, 935, (h, a) => h + a >= 1],
    ['1.5+', 7, 830, (h, a) => h + a >= 2],
    ['2.5+', 8, 750, (h, a) => h + a >= 3],
    ['1X', 9, 800, (h, a) => h >= a],
    ['12', 10, 800, (h, a) => h !== a],
    ['X2', 11, 780, (h, a) => a >= h],
  ];

  let DATA = null, loading = null;
  const st = { season: '', res: 'all', q: '', order: 'desc', page: 0 };   // res: all | won | lost (vurgulu maçlar)
  try {
    st.season = localStorage.getItem('betavus.stats_season') || '';
    st.order = localStorage.getItem('betavus.stats_order2') || 'desc';   // varsayılan: yeniden eskiye
  } catch (e) {}

  const escH = s => String(s).replace(/[&<>'"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
  const T = (k, v) => (root._t ? root._t(k, v) : k);
  const I = root.I18N || { pct: x => '%' + x, pctS: x => x + '%', dec: x => String(x).replace('.', ','), locale: 'tr-TR' };
  const pc = (h, n) => n ? I.pctS((h / n * 100).toFixed(1)) : '—';
  const fmtN = n => n.toLocaleString(I.locale);
  const dmyS = d => d.slice(8, 10) + '.' + d.slice(5, 7) + '.' + d.slice(0, 4);
  const seasonOfRow = d => { let y = +d.slice(0, 4); if (+d.slice(5, 7) < 7) y -= 1; return `${y}/${String(y + 1).slice(2)}`; };
  const curLeague = () => (typeof selected !== 'undefined' ? selected : 'Tümü');
  const flagOf = (lg, h) => (typeof flag === 'function' ? flag(lg, h) : '');

  function load() {
    if (DATA) return Promise.resolve(DATA);
    if (!loading) {
      loading = fetch(DATA_URL, { cache: 'no-cache' }).then(r => {
        if (!r.ok) throw new Error('stats ' + r.status);
        return r.json();
      }).then(d => {
        d.rows.forEach(r => { r[14] = seasonOfRow(r[0]); });   // r[13] = tahmini toplam gol (λ), r[14] = sezon
        DATA = d; return d;
      }).catch(e => { loading = null; throw e; });
    }
    return loading;
  }

  function hlOutcomes(r) {           // vurgulanan pazarların sonuçları (kısıtlı veri hiç vurgulanmaz)
    if (r[12]) return [];
    const out = [];
    for (const [, i, thr, hit] of MARKETS) if (r[i] >= thr) out.push(hit(r[4], r[5]));
    return out;
  }

  function scoped() {
    const lg = curLeague();
    const q = st.q.toLocaleLowerCase('tr-TR').split(/\s+/).filter(Boolean);
    return DATA.rows.filter(r =>
      (lg === 'Tümü' || r[1] === lg) &&
      (!st.season || r[14] === st.season) &&
      (!q.length || q.every(w => (r[2] + ' ' + r[3]).toLocaleLowerCase('tr-TR').includes(w))));
  }

  function aggregate(rows) {
    const m = {}; MARKETS.forEach(([k]) => { m[k] = { hn: 0, hh: 0, dn: 0, dh: 0, psum: 0, occ: 0 }; });
    let hlMatchN = 0, hlMatchH = 0, limited = 0;
    for (const r of rows) {
      if (r[12]) limited++;
      const outs = [];
      for (const [k, i, thr, hit] of MARKETS) {
        const p = r[i], ok = hit(r[4], r[5]), a = m[k];
        // Genel yön isabeti: model olaya ≥%50 veriyorsa "olur", altındaysa "olmaz" demiş sayılır
        const saysYes = p >= 500, right = saysYes ? ok : !ok;
        a.dn++; if (right) a.dh++;
        a.psum += p; if (ok) a.occ++;
        if (!r[12] && p >= thr) { a.hn++; if (ok) a.hh++; outs.push(ok); }
      }
      if (outs.length) { hlMatchN++; if (outs.every(Boolean)) hlMatchH++; }
    }
    const pickN = MARKETS.reduce((s, [k]) => s + m[k].hn, 0), pickH = MARKETS.reduce((s, [k]) => s + m[k].hh, 0);
    return { m, hlMatchN, hlMatchH, pickN, pickH, limited, total: rows.length };
  }

  function kpis(a) {
    return `<div class="st-kpis">
      <div class="st-kpi main"><div class="v">${pc(a.pickH, a.pickN)}</div><div class="k">${T('Vurgulanan tahmin başarısı')}</div><div class="n">${T('{h} / {n} tahmin tuttu', { h: fmtN(a.pickH), n: fmtN(a.pickN) })}</div></div>
      <div class="st-kpi"><div class="v">${pc(a.hlMatchH, a.hlMatchN)}</div><div class="k">${T('Tam isabetli vurgulu maç')}</div><div class="n">${T('{h} / {n} maçta tüm vurgular tuttu', { h: fmtN(a.hlMatchH), n: fmtN(a.hlMatchN) })}</div></div>
      <div class="st-kpi"><div class="v">${fmtN(a.total)}</div><div class="k">${T('Analiz edilen maç')}</div><div class="n">${T('{n} maç kısıtlı veri, vurgu dışı', { n: fmtN(a.limited) })}</div></div>
    </div>`;
  }

  function marketTable(a) {
    const thrTxt = { '0.5+': '≥' + I.pct(I.dec('93.5')), '1.5+': '≥' + I.pct(83), '2.5+': '≥' + I.pct(75), '1X': '≥' + I.pct(80), '12': '≥' + I.pct(80), 'X2': '≥' + I.pct(78) };
    return `<div class="card st-card"><h2>${T('Lig bazında doğruluk')}</h2>
      <p class="st-note">${T('st.note.market')}</p>
      <div class="tbl-scroll"><table class="bt-table st-table"><thead><tr>
        <th>${T('st.col.market')}</th><th>${T('Eşik')}</th><th>${T('Vurgulanan başarı')}</th><th>${T('Tuttu / Vurgu')}</th><th>${T('Genel yön isabeti')}</th><th>${T('Ort. model olasılığı')}</th><th>${T('Gerçekleşme')}</th>
      </tr></thead><tbody>${MARKETS.map(([k]) => {
        const x = a.m[k];
        return `<tr><td><b>${k}</b></td><td>${thrTxt[k]}</td><td class="st-strong">${pc(x.hh, x.hn)}</td><td>${fmtN(x.hh)} / ${fmtN(x.hn)}</td>
          <td>${pc(x.dh, x.dn)}</td><td>${x.dn ? I.pctS((x.psum / x.dn / 10).toFixed(1)) : '—'}</td><td>${pc(x.occ, x.dn)}</td></tr>`;
      }).join('')}</tbody></table></div></div>`;
  }

  function leagueTable() {
    if (curLeague() !== 'Tümü') return '';
    const by = {};
    for (const r of DATA.rows) {
      if (st.season && r[14] !== st.season) continue;
      (by[r[1]] = by[r[1]] || []).push(r);
    }
    const order = (typeof BT_ORDER !== 'undefined' ? BT_ORDER : Object.keys(by)).filter(l => by[l]);
    return `<div class="card st-card"><h2>${T('Lig bazında vurgulanan başarı')}</h2>
      <div class="tbl-scroll"><table class="bt-table st-table"><thead><tr><th>${T('Lig')}</th><th>${T('Maçlar')}</th><th>${T('Tümü')}</th>${MARKETS.map(([k]) => `<th>${k}</th>`).join('')}</tr></thead><tbody>
      ${order.map(l => {
        const a = aggregate(by[l]);
        return `<tr><td class="st-lg">${flagOf(l, 11)} ${escH(l)}</td><td>${fmtN(a.total)}</td><td class="st-strong">${pc(a.pickH, a.pickN)}</td>
          ${MARKETS.map(([k]) => `<td title="${a.m[k].hh} / ${a.m[k].hn}">${pc(a.m[k].hh, a.m[k].hn)}</td>`).join('')}</tr>`;
      }).join('')}</tbody></table></div></div>`;
  }

  function cell(r, i, thr, hit) {
    const p = r[i], ok = hit(r[4], r[5]), hl = !r[12] && p >= thr;
    const cls = hl ? (ok ? ' hl win' : ' hl lose') : (ok ? ' ok' : '');
    return `<td class="st-p${cls}" title="${T('Model {p}', { p: I.pct((p / 10).toFixed(1)) })}${hl ? T(' · vurgulandı · ') + T(ok ? 'tuttu' : 'tutmadı') : ''}${!hl && ok ? T(' · gerçekleşti') : ''}">${I.pctS((p / 10).toFixed(1))}${!hl && ok ? '<i>✓</i>' : ''}</td>`;
  }

  // Yalnızca vurgulu maçlar. Sayaçlar TAHMİN bazlıdır (bir maçta birden çok vurgu olabilir),
  // böylece ana sayfa şeridiyle aynı sayılar görünür: tutan = en az bir vurgusu tutan maçlar,
  // tutmayan = en az bir vurgusu tutmayan maçlar (karışık maç ikisinde de listelenir).
  function hlRows(rows) {
    const out = { all: [], won: [], lost: [], n: { all: 0, won: 0, lost: 0 } };
    for (const r of rows) {
      const o = hlOutcomes(r);
      if (!o.length) continue;
      const h = o.filter(Boolean).length;
      out.all.push(r); out.n.all += o.length;
      if (h) { out.won.push(r); out.n.won += h; }
      if (h < o.length) { out.lost.push(r); out.n.lost += o.length - h; }
    }
    return out;
  }

  function matchList(rows) {
    const groups = hlRows(rows);
    let list = groups[st.res] || groups.all;
    if (st.order === 'desc') list = list.slice().reverse();
    const pages = Math.max(1, Math.ceil(list.length / PAGE));
    if (st.page >= pages) st.page = pages - 1;
    const slice = list.slice(st.page * PAGE, st.page * PAGE + PAGE);
    const body = slice.map(r => {
      const outs = hlOutcomes(r), h = outs.filter(Boolean).length, won = h === outs.length;
      return `<tr>
        <td class="st-d">${dmyS(r[0])}</td>
        <td class="st-m"><span class="st-lgs">${flagOf(r[1], 10)} ${escH(r[1])}</span>${escH(r[2])} — ${escH(r[3])}
          <span class="st-lam" title="${T('Modelin maç öncesi tahmin ettiği toplam gol (λ)')}">${T('Tahmini toplam gol:')} <b>${r[13] != null ? I.dec(Number(r[13]).toFixed(2)) : '—'}</b></span></td>
        <td class="st-s">${r[4]}-${r[5]}</td>
        ${MARKETS.map(([, i, thr, hit]) => cell(r, i, thr, hit)).join('')}
        <td class="st-v"><span class="st-res ${won ? 'won' : 'lost'}">${T(won ? 'Tuttu' : 'Kaybetti')}</span><small>${h}/${outs.length}</small></td></tr>`;
    }).join('');
    const pager = `<div class="st-pager">
      <button class="hotbtn" data-pg="first" ${st.page ? '' : 'disabled'}>«</button>
      <button class="hotbtn" data-pg="prev" ${st.page ? '' : 'disabled'}>${T('‹ Önceki')}</button>
      <span>${T('Sayfa {p} / {n} · {m} maç', { p: st.page + 1, n: pages, m: fmtN(list.length) })}</span>
      <button class="hotbtn" data-pg="next" ${st.page < pages - 1 ? '' : 'disabled'}>${T('Sonraki ›')}</button>
      <button class="hotbtn" data-pg="last" ${st.page < pages - 1 ? '' : 'disabled'}>»</button></div>`;
    const fbtn = k => `<button class="hotbtn st-rf${st.res === k ? ' on' : ''} st-rf-${k}" type="button" data-res="${k}">${T({ all: 'Tüm vurgulular', won: '✓ Tutan', lost: '✗ Tutmayan' }[k])} <b>${T('{n} tahmin', { n: fmtN(groups.n[k]) })}</b> <small>· ${T('{n} maç', { n: fmtN(groups[k].length) })}</small></button>`;
    return `<div class="card st-card st-list"><h2>${T('Vurgulanan maçlar — tahmin vs gerçekleşen')}</h2>
      <div class="st-rfs">${fbtn('all')}${fbtn('won')}${fbtn('lost')}</div>
      <p class="st-note">${T('st.note.list')}</p>
      ${pager}
      <div class="tbl-scroll"><table class="bt-table st-table st-matches"><thead><tr><th class="st-sort" id="stDateSort" title="${T('Tıkla: {x} sırala', { x: T(st.order === 'desc' ? 'eskiden yeniye' : 'yeniden eskiye') })}">${T('Tarih')} ${st.order === 'desc' ? '▼' : '▲'}</th><th>${T('Maç')}</th><th>${T('Skor')}</th>${MARKETS.map(([k]) => `<th>${k}</th>`).join('')}<th>${T('Vurgu')}</th></tr></thead>
      <tbody>${body || `<tr><td colspan="10" class="st-mut" style="text-align:center;padding:24px">${T('Bu filtrede maç yok')}</td></tr>`}</tbody></table></div>
      ${pager}</div>`;
  }

  function leagueSelect() {
    const lgs = (typeof leagues !== 'undefined' ? leagues : ['Tümü']);
    return `<select class="sel" id="stLeague" title="${T('Lig filtresi')}">${lgs.map(l => `<option value="${escH(l)}"${l === curLeague() ? ' selected' : ''}>${l === 'Tümü' ? T('Tüm ligler') : escH(l)}</option>`).join('')}</select>`;
  }

  function controls() {
    const seasons = DATA.seasons || [];
    return `<div class="st-ctl">
      ${leagueSelect()}
      <select class="sel" id="stSeason"><option value="">${T('Tüm sezonlar')}</option>${seasons.map(s => `<option value="${s}"${s === st.season ? ' selected' : ''}>${s}</option>`).join('')}</select>
      <span class="srch"><input id="stQ" type="search" placeholder="${T('Takım ara…')}" value="${escH(st.q)}" autocomplete="off" spellcheck="false"></span>
    </div>`;
  }

  function render() {
    const host = document.getElementById('statsBody');
    if (!host) return;
    if (!DATA) {
      host.innerHTML = `<div class="loading">${T('Maç verisi yükleniyor…')}</div>`;
      load().then(render).catch(() => { host.innerHTML = `<div class="empty"><strong>${T('İstatistik verisi yüklenemedi')}</strong>${T('Sayfayı yenilemeyi dene.')}</div>`; });
      return;
    }
    const focusQ = document.activeElement && document.activeElement.id === 'stQ';
    const rows = scoped(), a = aggregate(rows);
    const lg = curLeague();
    host.innerHTML = `<div class="st-intro"><h1 class="st-h1">${T('İstatistikler')}</h1>
        <p>${T('st.intro', { seasons: (DATA.seasons || []).join(', '), n: fmtN(DATA.rows.length) })}${lg !== 'Tümü' ? T(' Filtre: <b>{lg}</b>.', { lg: escH(lg) }) : ''}</p></div>
      ${controls()}${kpis(a)}${marketTable(a)}${leagueTable()}${matchList(rows)}`;
    const $ = id => document.getElementById(id);
    $('stSeason').onchange = e => { st.season = e.target.value; st.page = 0; try { localStorage.setItem('betavus.stats_season', st.season); } catch (x) {} render(); };
    $('stDateSort').onclick = () => { st.order = st.order === 'asc' ? 'desc' : 'asc'; st.page = 0; try { localStorage.setItem('betavus.stats_order2', st.order); } catch (x) {} render(); };
    $('stLeague').onchange = e => { if (typeof setLeague === 'function') setLeague(e.target.value); st.page = 0; render(); };
    host.querySelectorAll('[data-res]').forEach(b => b.onclick = () => { st.res = b.dataset.res; st.page = 0; render(); });
    let t = null;
    $('stQ').oninput = e => { clearTimeout(t); t = setTimeout(() => { st.q = e.target.value.trim(); st.page = 0; render(); }, 250); };
    if (focusQ) { const q = $('stQ'); q.focus(); q.setSelectionRange(q.value.length, q.value.length); }
    host.querySelectorAll('[data-pg]').forEach(b => b.onclick = () => {
      const pages = Math.max(1, Math.ceil((hlRows(rows)[st.res] || []).length / PAGE));
      st.page = { first: 0, prev: st.page - 1, next: st.page + 1, last: pages - 1 }[b.dataset.pg];
      render();
      const lst = host.querySelector('.st-list'); if (lst) lst.scrollIntoView({ block: 'start' });
    });
  }

  // Ana sayfa (Bülten) şeridi — insanlar en yüksek güvenle vurguladığımız maçlardaki başarıyı görsün
  function renderBanner() {
    const el = document.getElementById('hlBanner');
    if (!el) return;
    fetch(SUMMARY_URL, { cache: 'no-cache' }).then(r => r.ok ? r.json() : null).then(s => {
      if (!s || !s.picks) { el.hidden = true; return; }
      const mk = s.markets || {};
      const pct = +s.picks.pct, fmtP = v => I.dec(v), pctP = v => (I.lang === 'tr' || !I.lang ? '%' + fmtP(v) : fmtP(v) + '%');
      const R = 34, C = +(2 * Math.PI * R).toFixed(1);
      el.innerHTML = `<div class="hlb-main">
          <div class="hlb-ring" aria-hidden="true"><svg viewBox="0 0 80 80"><circle class="hlb-rbg" cx="40" cy="40" r="${R}"/><circle class="hlb-rfg" cx="40" cy="40" r="${R}" stroke-dasharray="${C}" stroke-dashoffset="${C}"/></svg><span>✓</span></div>
          <div class="hlb-num"><div class="hlb-v">${pctP(pct)}</div><div class="hlb-cap">${T('isabet oranı')}</div></div>
          <div class="hlb-t"><span class="hlb-badge">${T('Doğrulanmış geçmiş performans')}</span>
            <b>${T('Vurguladığımız tahminlerin başarı oranı')}</b>
            <span class="hlb-sub">${T('{season} sezonundan bugüne · <strong>{h}</strong> / {n} vurgulu tahmin tuttu', { season: s.seasons[0], h: fmtN(s.picks.h), n: fmtN(s.picks.n) })}</span></div>
        </div>
        <div class="hlb-mk">${MARKETS.map(([k]) => [k, mk[k]]).filter(([, v]) => v).map(([k, v]) => `<span class="hlb-chip" title="${T('{h} tahmin tuttu / {n} vurgulu tahmin', { h: fmtN(v.h), n: fmtN(v.n) })}"><span class="hlb-ct"><b>${k}</b><i>${pctP(v.pct)}</i></span><span class="hlb-bar"><span style="width:${v.pct}%"></span></span><em>${T('{h}/{n} maç', { h: fmtN(v.h), n: fmtN(v.n) })}</em></span>`).join('')}
          <a href="#" class="hlb-link" onclick="setTab('stats');return false">${T('Tüm istatistikler →')}</a></div>`;
      el.hidden = false;
      // Giriş animasyonu: halka dolar, yüzde sayarak yükselir
      const ring = el.querySelector('.hlb-rfg'), num = el.querySelector('.hlb-v');
      requestAnimationFrame(() => requestAnimationFrame(() => { ring.style.strokeDashoffset = (C * (1 - pct / 100)).toFixed(1); }));
      if (!(root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches)) {
        const t0 = performance.now(), D = 1400;
        const step = now => {
          const k = Math.min(1, (now - t0) / D), e = 1 - Math.pow(1 - k, 3);
          num.textContent = pctP(k < 1 ? (pct * e).toFixed(1) : pct);
          if (k < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
      }
    }).catch(() => { el.hidden = true; });
  }

  root.BETAVUS_STATS = { render, renderBanner, load };
})(window);
