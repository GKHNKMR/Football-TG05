// Sanal Kasa → "Kupon önerisi" kartı (iş listesi #3, #21)
// Kurallar ve geçmiş başarı data/coupon-rules.json'dan gelir (scripts/build_coupon_rules.py, günlük).
// Maç seçimi: Fikstür'deki Çifte Şans olasılıkları (index.html → dcAll: kısıtlı veri / kritik eksik
// oyuncu hariç). Her maçtan en yüksek olasılıklı tek Çifte Şans; kurala uyan en yüksek N maç.
// Büyüme vaadi yok: geçmiş tutma oranı ve piyasa oranlarıyla getiri açıkça yazılır.
(function (root) {
  'use strict';

  const URL = 'data/coupon-rules.json', CPN_URL = 'data/coupons.json';
  const DC = ['1X', '12', 'X2'];
  const DC_MEANING = { '1X': 'Ev sahibi kazanır veya berabere', '12': 'Beraberlik olmaz', 'X2': 'Deplasman kazanır veya berabere' };
  const TIER_NAME = { low: 'Düşük risk', mid: 'Orta risk', high: 'Yüksek risk' };
  const T = (k, v) => (root._t ? root._t(k, v) : k);
  const I = () => root.I18N || { pct: x => '%' + x, dec: x => String(x).replace('.', ','), locale: 'tr-TR' };
  const esc = s => String(s == null ? '' : s).replace(/[&<>'"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
  const pc = v => I().pct(I().dec((v * 100).toFixed(1)));
  const odd = v => I().dec(v.toFixed(2));
  const dkey = iso => new Date(iso).toLocaleDateString('sv-SE', { timeZone: 'Europe/Istanbul' });

  let RULES = null, CPN = null, loading = null, dayIdx = 0, lastProfile = null;

  function load() {
    if (RULES || loading) return loading;
    const get = u => fetch(u + '?ts=' + Date.now(), { cache: 'no-store' }).then(r => r.ok ? r.json() : null).catch(() => null);
    loading = Promise.all([get(URL), get(CPN_URL)]).then(([a, b]) => { RULES = a; CPN = b || { coupons: {}, profiles: {} }; loading = null; });
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

  // ---- #21 Kasa hedefine göre kupon: kuponları saatlik bot kurar (scripts/coupon_engine.py → data/coupons.json);
  //      kartta görülen kupon ile sonradan tuttu / tutmadı diye değerlendirilen kupon birebir aynıdır. ----
  const MK_MEANING = { '0.5+': 'Maçta en az 1 gol', '1.5+': 'Maçta en az 2 gol', '2.5+': 'Maçta en az 3 gol',
    '1X': DC_MEANING['1X'], '12': DC_MEANING['12'], 'X2': DC_MEANING['X2'] };
  function profileFor(ctx) {
    const P = (CPN && CPN.profiles) || {};
    if (P[ctx.profile]) return ctx.profile;
    const f = 1 - (+ctx.reserve || 0), Rq = f > 0 ? 1 + (+ctx.g || 0) / f : 0;   // özel risk: en yakın standart profil
    return Object.keys(P).sort((a, b) => Math.abs(P[a].R - Rq) - Math.abs(P[b].R - Rq))[0] || null;
  }
  function ruleFor(Rq) {
    const rs = (RULES && RULES.target && RULES.target.rules) || [];
    return rs.find(r => r.R >= Rq - 1e-9) || rs[rs.length - 1] || null;
  }
  // Geçmişteki gerçek kuponlarla (oran, tuttu mu) 1 yıllık Monte Carlo: bugünkü kasa → hedef
  function simulate(rule, bank, goal, f) {
    const N = 2000, H = 365, out = rule.outcomes, n = out.length, pd = rule.per_day, floor = bank * 0.1;
    let reach = 0, bust = 0; const days = [];
    let seed = 12345; const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let i = 0; i < N; i++) {
      let b = bank;
      for (let d = 1; d <= H; d++) {
        if (rnd() > pd) continue;
        const v = out[Math.floor(rnd() * n)], st = b * f;
        b += v > 0 ? st * (v / 1000 - 1) : -st;
        if (b >= goal) { reach++; days.push(d); break; }
        if (b < floor) { bust++; break; }
      }
    }
    days.sort((a, b) => a - b);
    return { reach: reach / N, med: days.length ? days[Math.floor(days.length / 2)] : null, bust: bust / N };
  }
  function cLegHtml(l) {
    const t = new Date(l.kickoff_utc).toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Istanbul' });
    const fl = typeof root.flag === 'function' ? root.flag(l.league, 11) : '';
    const res = l.result ? `<span class="cs-res ${l.result}">${l.result === 'won' ? '✓' : '✗'} ${esc(l.score || '')}</span>` : '';
    return `<li class="cs-leg"><div class="cs-m"><span class="cs-lg">${fl} ${esc(l.league)} · ${t}</span><b>${esc(l.home)} — ${esc(l.away)}</b></div>
      ${res}<span class="cs-pick" title="${esc(T(MK_MEANING[l.market] || ''))} · ${T('oran')} ${odd(l.odds)}${l.real ? '' : ' ' + T('(tahmini)')}">${l.market} <em>${pc(l.p)}</em></span></li>`;
  }
  function historyHtml(prof) {
    const all = Object.values((CPN && CPN.coupons) || {}).filter(c => c.profile === prof && c.status !== 'pending').sort((a, b) => (a.date < b.date ? 1 : -1));
    if (!all.length) return `<div class="cs-past"><b>${T('Geçmiş öneriler')}</b> <span class="cs-muted">${T('Henüz sonuçlanan önerilen kupon yok; maçlar oynandıkça burada tuttu / tutmadı olarak listelenir.')}</span></div>`;
    const w = all.filter(c => c.status === 'won').length;
    const rows = all.slice(0, 10).map(c => `<li class="cs-pr ${c.status}"><span>${esc(dayLabel(c.date))}</span><span>${c.legs.length} ${T('maç')} · ${T('oran')} ${odd(c.odds)}</span><b>${c.status === 'won' ? '✓ ' + T('Tuttu') : '✗ ' + T('Tutmadı')}</b></li>`).join('');
    return `<details class="cs-past"><summary><b>${T('Geçmiş öneriler')}</b> — ${T('{n} kupondan {w} tuttu', { n: all.length, w })}</summary><ul>${rows}</ul></details>`;
  }
  function targetHtml(ctx, day) {
    const prof = profileFor(ctx); if (!prof) return '';
    const cfg = CPN.profiles[prof], f = cfg.f, cur = I().locale;
    const eur = v => Number(v).toLocaleString(cur, { style: 'currency', currency: 'EUR' });
    const c = day && CPN.coupons[`${day}|${prof}`];
    const bank = +ctx.bank || +ctx.start || 0, goal = Math.max(0, (+ctx.target || 0) - (+ctx.secured || 0));
    let body;
    if (c) {
      const est = c.legs.some(l => !l.real);
      body = `<ul class="cs-legs">${c.legs.map(cLegHtml).join('')}</ul>
        <div class="cs-sum"><span>${T('Kupon oranı')} <b>${odd(c.odds)}</b>${est ? ` <em class="cs-est">${T('(tahmini — gerçek oran maç haftası gelir)')}</em>` : ''}</span><span>${T('Model: kuponun tutma olasılığı')} <b>${pc(c.p)}</b></span>${bank > 0 ? `<span>${T('Kupona yatacak')} <b>${eur(bank * f)}</b></span>` : ''}</div>`;
    } else {
      body = `<p class="cs-empty">${T('Bu gün en fazla 5 vurgulu maçla gereken orana ulaşılamıyor.')}</p>`;
    }
    const rule = ruleFor(cfg.R);
    let proj = '';
    if (rule && bank > 0 && goal > bank) {
      const m = simulate(rule, bank, goal, f);
      proj = `<div class="cs-proj"><b>${T('Gerçekçi beklenti')}</b> — ${T('cs.proj', { bank: eur(bank), goal: eur(goal), p: pc(m.reach), med: m.med ? T('ortanca {d} gün', { d: m.med }) : '—', bust: pc(m.bust), site: ctx.days || '—' })}
        <div class="cs-projm">${T('cs.projm', { n: rule.coupons.toLocaleString(cur), w: pc(rule.win_pct / 100), o: odd(rule.avg_odds), l: I().dec(String(rule.avg_legs)), per: I().dec((1 / rule.per_day).toFixed(1)) })}</div></div>`;
    }
    const note = ctx.profile !== prof ? ` ${T('(özel risk: en yakın standart profil)')}` : '';
    return `<div class="cs-target">
      <div class="cs-th"><b>${T('Kasa hedefine göre kupon')}</b><span class="cs-badge">${T('Kasana göre')}</span></div>
      <div class="cs-rule">${T('cs.need', { g: I().pct(I().dec(String(Math.round(cfg.g * 1000) / 10))), f: I().pct(Math.round(f * 100)), r: odd(cfg.R), max: CPN.max_legs || 5 })}${note}</div>
      ${body}${proj}${historyHtml(prof)}</div>`;
  }

  function html(ctxIn) {
    const ctx = typeof ctxIn === 'object' && ctxIn ? ctxIn : { profile: ctxIn };
    const profile = ctx.profile;
    lastProfile = ctx;
    if (!RULES) {
      const pr = load(); if (pr) pr.then(() => mount(lastProfile));
      return `<div class="card cs-card" id="cpnSuggest"><div class="loading">${T('Kupon önerileri yükleniyor…')}</div></div>`;
    }
    const byDay = legsByDay(), tiers = RULES.tiers || [];
    const prof = profileFor(ctx), today = dkey(new Date().toISOString());
    const cDays = Object.values(CPN.coupons || {}).filter(c => c.profile === prof && c.date >= today).map(c => c.date);
    const days = [...new Set(Object.keys(byDay).filter(k => tiers.some(t => couponFor(byDay[k], t))).concat(cDays))].sort();
    if (dayIdx >= days.length) dayIdx = Math.max(0, days.length - 1);
    const k = days[dayIdx];
    const mineTier = tiers.find(t => t.profile === profile);
    const roi = tiers.map(t => t.history ? `${T(TIER_NAME[t.id])} ${I().dec(t.history.roi_pct.toFixed(1))}%` : '').filter(Boolean).join(' · ');
    const head = `<div class="cs-head"><div><h2>🎯 ${T('Kupon önerisi')}</h2>
        <p class="cs-sub">${T('Önce kasanın hedefine göre kupon, altında daha güvenli sabit kurallar.')}</p></div>
      ${k ? `<div class="cs-nav"><button type="button" class="cs-nb" data-cs="-1" ${dayIdx ? '' : 'disabled'} aria-label="${T('Önceki gün')}">‹</button><span>${esc(dayLabel(k))}</span><button type="button" class="cs-nb" data-cs="1" ${dayIdx < days.length - 1 ? '' : 'disabled'} aria-label="${T('Sonraki gün')}">›</button></div>` : ''}</div>`;
    const grid = k ? `${targetHtml(ctx, k)}<h3 class="cs-h3">${T('Daha güvenli sabit kurallar')}</h3><div class="cs-grid">${tiers.map(t => tierHtml(t, byDay[k], mineTier && t.id === mineTier.id)).join('')}</div>`
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
