// İstatistikler sekmesi (#pane-stats)
// Son 5 tamamlanmış sezonun (walk-forward, sızıntısız) TÜM maçları: her maç için
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
    ['0.5+', 6, 950, (h, a) => h + a >= 1],
    ['1.5+', 7, 850, (h, a) => h + a >= 2],
    ['2.5+', 8, 800, (h, a) => h + a >= 3],
    ['1X', 9, 800, (h, a) => h >= a],
    ['12', 10, 800, (h, a) => h !== a],
    ['X2', 11, 800, (h, a) => a >= h],
  ];
  const BUCKETS = [[500, 600], [600, 700], [700, 800], [800, 900], [900, 1001]];

  let DATA = null, loading = null;
  const st = { season: '', res: 'all', q: '', order: 'desc', page: 0 };   // res: all | won | lost (vurgulu maçlar)
  try {
    st.season = localStorage.getItem('betavus.stats_season') || '';
    st.order = localStorage.getItem('betavus.stats_order2') || 'desc';   // varsayılan: yeniden eskiye
  } catch (e) {}

  const escH = s => String(s).replace(/[&<>'"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
  const pc = (h, n) => n ? (h / n * 100).toFixed(1) + '%' : '—';
  const fmtN = n => n.toLocaleString('tr-TR');
  const dmyS = d => d.slice(8, 10) + '.' + d.slice(5, 7) + '.' + d.slice(0, 4);
  const seasonOfRow = d => { let y = +d.slice(0, 4); if (+d.slice(5, 7) < 7) y -= 1; return `${y}/${String(y + 1).slice(2)}`; };
  const curLeague = () => (typeof selected !== 'undefined' ? selected : 'Tümü');
  const flagOf = (lg, h) => (typeof flag === 'function' ? flag(lg, h) : '');

  function load() {
    if (DATA) return Promise.resolve(DATA);
    if (!loading) {
      loading = fetch(DATA_URL, { cache: 'force-cache' }).then(r => {
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
    const buckets = BUCKETS.map(() => ({ n: 0, h: 0, psum: 0 }));
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
        const conf = saysYes ? p : 1000 - p;          // modelin eğildiği tarafın güveni
        const bi = BUCKETS.findIndex(([lo, hi]) => conf >= lo && conf < hi);
        if (bi >= 0) { const b = buckets[bi]; b.n++; b.psum += conf; if (right) b.h++; }
        if (!r[12] && p >= thr) { a.hn++; if (ok) a.hh++; outs.push(ok); }
      }
      if (outs.length) { hlMatchN++; if (outs.every(Boolean)) hlMatchH++; }
    }
    const pickN = MARKETS.reduce((s, [k]) => s + m[k].hn, 0), pickH = MARKETS.reduce((s, [k]) => s + m[k].hh, 0);
    return { m, buckets, hlMatchN, hlMatchH, pickN, pickH, limited, total: rows.length };
  }

  function kpis(a) {
    return `<div class="st-kpis">
      <div class="st-kpi main"><div class="v">${pc(a.pickH, a.pickN)}</div><div class="k">Vurgulanan tahmin başarısı</div><div class="n">${fmtN(a.pickH)} / ${fmtN(a.pickN)} tahmin tuttu</div></div>
      <div class="st-kpi"><div class="v">${pc(a.hlMatchH, a.hlMatchN)}</div><div class="k">Tam isabetli vurgulu maç</div><div class="n">${fmtN(a.hlMatchH)} / ${fmtN(a.hlMatchN)} maçta tüm vurgular tuttu</div></div>
      <div class="st-kpi"><div class="v">${fmtN(a.total)}</div><div class="k">Analiz edilen maç</div><div class="n">${fmtN(a.limited)} maç kısıtlı veri, vurgu dışı</div></div>
    </div>`;
  }

  function marketTable(a) {
    const thrTxt = { '0.5+': '≥%95', '1.5+': '≥%85', '2.5+': '≥%80', '1X': '≥%80', '12': '≥%80', 'X2': '≥%80' };
    return `<div class="card st-card"><h2>Pazar bazında doğruluk</h2>
      <p class="st-note"><b>Vurgulanan</b>: modelin güven eşiğini geçtiği tahminler. <b>Genel yön isabeti</b>: tüm maçlarda modelin eğildiği taraf (olur / olmaz) doğru mu? <b>Ort. model olasılığı</b> ile <b>gerçekleşme</b> birbirine yakınsa model iyi kalibre demektir.</p>
      <div class="tbl-scroll"><table class="bt-table st-table"><thead><tr>
        <th>Pazar</th><th>Eşik</th><th>Vurgulanan başarı</th><th>Tuttu / Vurgu</th><th>Genel yön isabeti</th><th>Ort. model olasılığı</th><th>Gerçekleşme</th>
      </tr></thead><tbody>${MARKETS.map(([k]) => {
        const x = a.m[k];
        return `<tr><td><b>${k}</b></td><td>${thrTxt[k]}</td><td class="st-strong">${pc(x.hh, x.hn)}</td><td>${fmtN(x.hh)} / ${fmtN(x.hn)}</td>
          <td>${pc(x.dh, x.dn)}</td><td>${x.dn ? (x.psum / x.dn / 10).toFixed(1) + '%' : '—'}</td><td>${pc(x.occ, x.dn)}</td></tr>`;
      }).join('')}</tbody></table></div></div>`;
  }

  function confTable(a) {
    return `<div class="card st-card"><h2>Güven analizi</h2>
      <p class="st-note">Altı pazardaki tüm tahminler, modelin eğildiği tarafa verdiği olasılığa göre gruplandı. İyi bir modelde "ortalama güven" ile "gerçek isabet" birbirine yakın olur.</p>
      <div class="tbl-scroll"><table class="bt-table st-table"><thead><tr><th>Model güveni</th><th>Tahmin</th><th>Ortalama güven</th><th>Gerçek isabet</th><th></th></tr></thead><tbody>
      ${a.buckets.map((b, i) => {
        const [lo, hi] = BUCKETS[i], acc = b.n ? b.h / b.n * 100 : 0;
        return `<tr><td>%${lo / 10}–${hi > 1000 ? 100 : hi / 10}</td><td>${fmtN(b.n)}</td><td>${b.n ? (b.psum / b.n / 10).toFixed(1) + '%' : '—'}</td>
          <td class="st-strong">${b.n ? acc.toFixed(1) + '%' : '—'}</td><td class="st-barc"><span class="st-bar" style="width:${acc.toFixed(0)}%"></span></td></tr>`;
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
    return `<div class="card st-card"><h2>Lig bazında vurgulanan başarı</h2>
      <div class="tbl-scroll"><table class="bt-table st-table"><thead><tr><th>Lig</th><th>Maç</th><th>Tümü</th>${MARKETS.map(([k]) => `<th>${k}</th>`).join('')}</tr></thead><tbody>
      ${order.map(l => {
        const a = aggregate(by[l]);
        return `<tr><td class="st-lg">${flagOf(l, 11)} ${escH(l)}</td><td>${fmtN(a.total)}</td><td class="st-strong">${pc(a.pickH, a.pickN)}</td>
          ${MARKETS.map(([k]) => `<td title="${a.m[k].hh} / ${a.m[k].hn}">${pc(a.m[k].hh, a.m[k].hn)}</td>`).join('')}</tr>`;
      }).join('')}</tbody></table></div></div>`;
  }

  function cell(r, i, thr, hit) {
    const p = r[i], ok = hit(r[4], r[5]), hl = !r[12] && p >= thr;
    const cls = hl ? (ok ? ' hl win' : ' hl lose') : (ok ? ' ok' : '');
    return `<td class="st-p${cls}" title="Model %${(p / 10).toFixed(1)}${hl ? ' · vurgulandı · ' + (ok ? 'tuttu' : 'tutmadı') : ''}${!hl && ok ? ' · gerçekleşti' : ''}">${(p / 10).toFixed(1)}%${!hl && ok ? '<i>✓</i>' : ''}</td>`;
  }

  function hlRows(rows) {          // yalnızca vurgulu maçlar + tutan / kaybeden filtresi
    const out = { all: [], won: [], lost: [] };
    for (const r of rows) {
      const o = hlOutcomes(r);
      if (!o.length) continue;
      out.all.push(r);
      (o.every(Boolean) ? out.won : out.lost).push(r);
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
          <span class="st-lam" title="Modelin maç öncesi tahmin ettiği toplam gol (λ)">Tahmini toplam gol: <b>${r[13] != null ? Number(r[13]).toFixed(2).replace('.', ',') : '—'}</b></span></td>
        <td class="st-s">${r[4]}-${r[5]}</td>
        ${MARKETS.map(([, i, thr, hit]) => cell(r, i, thr, hit)).join('')}
        <td class="st-v"><span class="st-res ${won ? 'won' : 'lost'}">${won ? 'Tuttu' : 'Kaybetti'}</span><small>${h}/${outs.length}</small></td></tr>`;
    }).join('');
    const pager = `<div class="st-pager">
      <button class="hotbtn" data-pg="first" ${st.page ? '' : 'disabled'}>«</button>
      <button class="hotbtn" data-pg="prev" ${st.page ? '' : 'disabled'}>‹ Önceki</button>
      <span>Sayfa ${st.page + 1} / ${pages} · ${fmtN(list.length)} maç</span>
      <button class="hotbtn" data-pg="next" ${st.page < pages - 1 ? '' : 'disabled'}>Sonraki ›</button>
      <button class="hotbtn" data-pg="last" ${st.page < pages - 1 ? '' : 'disabled'}>»</button></div>`;
    const fbtn = (k, label, n) => `<button class="hotbtn st-rf${st.res === k ? ' on' : ''} st-rf-${k}" type="button" data-res="${k}">${label} <b>${fmtN(n)}</b></button>`;
    return `<div class="card st-card st-list"><h2>Vurgulanan maçlar — tahmin vs gerçekleşen</h2>
      <div class="st-rfs">${fbtn('all', 'Tüm vurgulular', groups.all.length)}${fbtn('won', '✓ Kazanan vurgulular', groups.won.length)}${fbtn('lost', '✗ Kaybeden vurgulular', groups.lost.length)}</div>
      <p class="st-note">Her hücre maç öncesi model olasılığıdır. <span class="st-legend win">yeşil</span> = vurgulanan tahmin tuttu, <span class="st-legend lose">kırmızı</span> = vurgulanan tahmin tutmadı, <b>✓</b> = vurgusuz ama gerçekleşti. Tarih başlığına tıklayarak sıralamayı değiştir.</p>
      ${pager}
      <div class="tbl-scroll"><table class="bt-table st-table st-matches"><thead><tr><th class="st-sort" id="stDateSort" title="Tıkla: ${st.order === 'desc' ? 'eskiden yeniye' : 'yeniden eskiye'} sırala">Tarih ${st.order === 'desc' ? '▼' : '▲'}</th><th>Maç</th><th>Skor</th>${MARKETS.map(([k]) => `<th>${k}</th>`).join('')}<th>Vurgu</th></tr></thead>
      <tbody>${body || `<tr><td colspan="10" class="st-mut" style="text-align:center;padding:24px">Bu filtrede maç yok</td></tr>`}</tbody></table></div>
      ${pager}</div>`;
  }

  function leagueSelect() {
    const lgs = (typeof leagues !== 'undefined' ? leagues : ['Tümü']);
    return `<select class="sel" id="stLeague" title="Lig filtresi">${lgs.map(l => `<option value="${escH(l)}"${l === curLeague() ? ' selected' : ''}>${l === 'Tümü' ? 'Tüm ligler' : escH(l)}</option>`).join('')}</select>`;
  }

  function controls() {
    const seasons = DATA.seasons || [];
    return `<div class="st-ctl">
      ${leagueSelect()}
      <select class="sel" id="stSeason"><option value="">Tüm sezonlar (5 sezon)</option>${seasons.map(s => `<option value="${s}"${s === st.season ? ' selected' : ''}>${s}</option>`).join('')}</select>
      <span class="srch"><input id="stQ" type="search" placeholder="Takım ara…" value="${escH(st.q)}" autocomplete="off" spellcheck="false"></span>
    </div>`;
  }

  function render() {
    const host = document.getElementById('statsBody');
    if (!host) return;
    if (!DATA) {
      host.innerHTML = '<div class="loading">5 sezonluk maç verisi yükleniyor…</div>';
      load().then(render).catch(() => { host.innerHTML = '<div class="empty"><strong>İstatistik verisi yüklenemedi</strong>Sayfayı yenilemeyi dene.</div>'; });
      return;
    }
    const focusQ = document.activeElement && document.activeElement.id === 'stQ';
    const rows = scoped(), a = aggregate(rows);
    const lg = curLeague();
    host.innerHTML = `<div class="st-intro"><h1 class="st-h1">İstatistikler</h1>
        <p>${(DATA.seasons || []).join(', ')} sezonlarının ${fmtN(DATA.rows.length)} maçı, canlı sitedeki modelle <b>yalnızca maçtan önceki verilerle</b> tahmin edildi ve gerçek skorlarla karşılaştırıldı.${lg !== 'Tümü' ? ` Filtre: <b>${escH(lg)}</b>.` : ''}</p></div>
      ${controls()}${kpis(a)}${marketTable(a)}${confTable(a)}${leagueTable()}${matchList(rows)}`;
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
      el.innerHTML = `<div class="hlb-main">
          <div class="hlb-v">%${String(s.picks.pct).replace('.', ',')}</div>
          <div class="hlb-t"><b>Vurguladığımız tahminlerin başarı oranı</b>
            <span>Son 5 sezon (${s.seasons[0]} – ${s.seasons[s.seasons.length - 1]}) · ${fmtN(s.picks.h)} / ${fmtN(s.picks.n)} vurgulu tahmin tuttu</span></div>
        </div>
        <div class="hlb-mk">${MARKETS.map(([k]) => [k, mk[k]]).filter(([, v]) => v).map(([k, v]) => `<span class="hlb-chip" title="${fmtN(v.h)} tahmin tuttu / ${fmtN(v.n)} vurgulu tahmin"><b>${k}</b> %${String(v.pct).replace('.', ',')} <em>${fmtN(v.h)}/${fmtN(v.n)} maç</em></span>`).join('')}
          <a href="#" class="hlb-link" onclick="setTab('stats');return false">Tüm istatistikler →</a></div>`;
      el.hidden = false;
    }).catch(() => { el.hidden = true; });
  }

  root.BETAVUS_STATS = { render, renderBanner, load };
})(window);
