// Sanal Kasa → "Kupon önerisi" kartı (iş listesi #3)
// Kurallar ve geçmiş başarı data/coupon-rules.json'dan gelir (scripts/build_coupon_rules.py, günlük).
// Maç seçimi: Fikstür'deki Çifte Şans olasılıkları (index.html → dcAll: kısıtlı veri / kritik eksik
// oyuncu hariç). Her maçtan en yüksek olasılıklı tek Çifte Şans; kurala uyan en yüksek N maç.
// Büyüme vaadi yok: geçmiş tutma oranı ve piyasa oranlarıyla getiri açıkça yazılır.
(function (root) {
  'use strict';

  const URL = 'data/coupon-rules.json';
  const DC = ['1X', '12', 'X2'];
  const DC_MEANING = { '1X': 'Ev sahibi kazanır veya berabere', '12': 'Beraberlik olmaz', 'X2': 'Deplasman kazanır veya berabere' };
  const TIER_NAME = { low: 'Düşük risk', mid: 'Orta risk', high: 'Yüksek risk' };
  const T = (k, v) => (root._t ? root._t(k, v) : k);
  const I = () => root.I18N || { pct: x => '%' + x, dec: x => String(x).replace('.', ','), locale: 'tr-TR' };
  const esc = s => String(s == null ? '' : s).replace(/[&<>'"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
  const pc = v => I().pct(I().dec((v * 100).toFixed(1)));
  const odd = v => I().dec(v.toFixed(2));
  const dkey = iso => new Date(iso).toLocaleDateString('sv-SE', { timeZone: 'Europe/Istanbul' });

  let RULES = null, loading = null, dayIdx = 0, lastProfile = null;

  function load() {
    if (RULES || loading) return loading;
    loading = fetch(URL + '?ts=' + Date.now(), { cache: 'no-store' }).then(r => r.ok ? r.json() : null)
      .then(j => { RULES = j; loading = null; }).catch(() => { loading = null; });
    return loading;
  }

  // Başlamamış maçlardan kurala aday bacaklar: gün → [{x, code, p}]
  function legsByDay() {
    const now = Date.now(), out = {};
    const dcAll = root.dcAll;
    if (typeof dcAll !== 'function') return out;
    for (const x of (root.__data || [])) {
      if (!x || new Date(x.kickoff_utc).getTime() <= now || x.live) continue;
      const d = dcAll(x);
      if (!d || !d.ok || !d.v) continue;
      const code = DC.reduce((a, b) => (d.v[b] > d.v[a] ? b : a), DC[0]);
      (out[dkey(x.kickoff_utc)] = out[dkey(x.kickoff_utc)] || []).push({ x, code, p: d.v[code] / 100 });
    }
    Object.values(out).forEach(l => l.sort((a, b) => b.p - a.p));
    return out;
  }

  function couponFor(legs, tier) {
    const pick = (legs || []).filter(l => l.p >= tier.thr).slice(0, tier.legs);
    return pick.length === tier.legs ? pick : null;
  }

  function dayLabel(k) {
    const d = new Date(k + 'T12:00:00');
    return d.toLocaleDateString(I().locale, { weekday: 'long', day: '2-digit', month: 'long' });
  }

  function legHtml(l) {
    const t = new Date(l.x.kickoff_utc).toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Istanbul' });
    const fl = typeof root.flag === 'function' ? root.flag(l.x.league, 11) : '';
    return `<li class="cs-leg"><div class="cs-m"><span class="cs-lg">${fl} ${esc(l.x.league)} · ${t}</span><b>${esc(l.x.home)} — ${esc(l.x.away)}</b></div>
      <span class="cs-pick" title="${esc(T(DC_MEANING[l.code]))}">${l.code} <em>${pc(l.p)}</em></span></li>`;
  }

  function tierHtml(tier, legs, mine) {
    const h = tier.history, c = couponFor(legs, tier);
    const rule = T('{n} maç · Çifte Şans ≥ {t}', { n: tier.legs, t: I().pct(Math.round(tier.thr * 100)) });
    let body;
    if (c) {
      const p = c.reduce((a, l) => a * l.p, 1);
      body = `<ul class="cs-legs">${c.map(legHtml).join('')}</ul>
        <div class="cs-sum"><span>${T('Model: kuponun tutma olasılığı')} <b>${pc(p)}</b></span><span>${T('Adil oran')} <b>${odd(1 / p)}</b></span></div>`;
    } else {
      body = `<p class="cs-empty">${T('Bu gün kurala uyan yeterli maç yok.')}</p>`;
    }
    const hist = h ? `<div class="cs-hist">${T('cs.hist', { n: h.coupons.toLocaleString(I().locale), w: pc(h.win_pct / 100), o: odd(h.avg_odds) })}</div>` : '';
    return `<div class="cs-tier${mine ? ' mine' : ''}">
      <div class="cs-th"><b>${T(TIER_NAME[tier.id])}</b>${mine ? `<span class="cs-badge">${T('Kasana uygun')}</span>` : ''}</div>
      <div class="cs-rule">${rule}</div>${body}${hist}</div>`;
  }

  function html(profile) {
    lastProfile = profile;
    if (!RULES) {
      const pr = load(); if (pr) pr.then(() => mount(lastProfile));
      return `<div class="card cs-card" id="cpnSuggest"><div class="loading">${T('Kupon önerileri yükleniyor…')}</div></div>`;
    }
    const byDay = legsByDay(), tiers = RULES.tiers || [];
    const days = Object.keys(byDay).sort().filter(k => tiers.some(t => couponFor(byDay[k], t)));
    if (dayIdx >= days.length) dayIdx = Math.max(0, days.length - 1);
    const k = days[dayIdx];
    const mineTier = tiers.find(t => t.profile === profile);
    const roi = tiers.map(t => t.history ? `${T(TIER_NAME[t.id])} ${I().dec(t.history.roi_pct.toFixed(1))}%` : '').filter(Boolean).join(' · ');
    const head = `<div class="cs-head"><div><h2>🎯 ${T('Kupon önerisi')}</h2>
        <p class="cs-sub">${mineTier ? T('Kasanın risk seviyesine uygun kupon işaretli.') : T('Kasan özel risk seviyesinde; üç öneri de gösteriliyor.')}</p></div>
      ${k ? `<div class="cs-nav"><button type="button" class="cs-nb" data-cs="-1" ${dayIdx ? '' : 'disabled'} aria-label="${T('Önceki gün')}">‹</button><span>${esc(dayLabel(k))}</span><button type="button" class="cs-nb" data-cs="1" ${dayIdx < days.length - 1 ? '' : 'disabled'} aria-label="${T('Sonraki gün')}">›</button></div>` : ''}</div>`;
    const grid = k ? `<div class="cs-grid">${tiers.map(t => tierHtml(t, byDay[k], mineTier && t.id === mineTier.id)).join('')}</div>`
      : `<p class="cs-empty">${T('Önümüzdeki günlerde kurala uyan maç yok.')}</p>`;
    const note = `<p class="cs-note">${T('cs.note', { roi: esc(roi) })}</p>`;
    return `<div class="card cs-card" id="cpnSuggest">${head}${grid}${note}</div>`;
  }

  function mount(profile = lastProfile) {
    const box = document.getElementById('cpnSuggest');
    if (!box) return;
    box.outerHTML = html(profile);
    wire();
  }

  // Olay yetkilendirme: Sanal Kasa ekranı her yeniden çizildiğinde ayrıca bağlamaya gerek yok
  function wire() {}
  document.addEventListener('click', e => {
    const b = e.target.closest && e.target.closest('#cpnSuggest [data-cs]');
    if (b && !b.disabled) { dayIdx += +b.dataset.cs; mount(); }
  });

  load();
  root.BETAVUS_COUPON = { html, wire, mount, _rules: () => RULES };
})(window);
