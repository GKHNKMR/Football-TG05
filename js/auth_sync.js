// Üyelik (Supabase Auth) + kullanıcı verisinin cihazlar arası senkronu.
//
// Senkron modeli:
//  - Tarayıcıdaki tüm "betavus.*" localStorage anahtarları (Sanal Kasa = betavus.paper_v1,
//    tercihler, filtreler…) kullanıcının user_state satırında {anahtar: {v, t}} olarak tutulur.
//  - Her yerel yazmanın zamanı (t) ayrıca kaydedilir; birleştirmede yeni olan kazanır.
//  - Bir cihazın ilk senkronunda Sanal Kasa verisi ezilmez: iki taraftaki kasalar,
//    kuponlar ve hesap hareketleri kimliklerine göre birleştirilir.
//  - Yerelde üzerine yazılan her değer önce betavus.__backup.* altına yedeklenir.
(function (root) {
  'use strict';

  const cfg = root.BETAVUS_AUTH_CONFIG || {};
  const configured = !!(cfg.supabaseUrl && cfg.supabaseAnonKey && root.supabase && root.supabase.createClient);
  const PREFIX = 'betavus.';
  const META = 'betavus.__sync_meta';
  const BACKUP = 'betavus.__backup.';
  const PAPER = 'betavus.paper_v1';
  const NO_SYNC = new Set(['betavus.access', META]);

  const LS = root.localStorage;
  const origSet = Storage.prototype.setItem, origRemove = Storage.prototype.removeItem, origGet = Storage.prototype.getItem;
  let applying = false, client = null, user = null, profile = null, pushTimer = null, pushing = null, lastSync = null;

  const tracked = k => typeof k === 'string' && k.startsWith(PREFIX) && !NO_SYNC.has(k) && !k.startsWith(BACKUP);
  function metaGet() { try { const m = JSON.parse(origGet.call(LS, META) || '{}'); m.t = m.t || {}; m.synced = m.synced || {}; return m; } catch (e) { return { t: {}, synced: {} }; } }
  function metaSet(m) { try { origSet.call(LS, META, JSON.stringify(m)); } catch (e) {} }

  // Yerel yazmaları zaman damgasıyla işaretle (uygulamanın geri kalanı değişmeden kalır)
  function touched(k) {
    if (applying || !tracked(k)) return;
    const m = metaGet(); m.t[k] = Date.now(); metaSet(m);
    schedulePush();
  }
  Storage.prototype.setItem = function (k, v) { origSet.call(this, k, v); if (this === LS) touched(k); };
  Storage.prototype.removeItem = function (k) { origRemove.call(this, k); if (this === LS) touched(k); };

  function localKeys() {
    const out = [];
    for (let i = 0; i < LS.length; i++) { const k = LS.key(i); if (tracked(k)) out.push(k); }
    return out;
  }
  function backup(k, v) {
    if (v == null) return;
    try {
      origSet.call(LS, BACKUP + k.slice(PREFIX.length) + '.' + Date.now(), v);
      // anahtar başına en fazla 3 yedek
      const pre = BACKUP + k.slice(PREFIX.length) + '.';
      const olds = [];
      for (let i = 0; i < LS.length; i++) { const x = LS.key(i); if (x && x.startsWith(pre)) olds.push(x); }
      olds.sort().slice(0, Math.max(0, olds.length - 3)).forEach(x => origRemove.call(LS, x));
    } catch (e) {}
  }

  // İlk senkronda iki cihazın Sanal Kasa verisini kimliklere göre birleştir
  function mergePaper(localStr, remoteStr, localNewer) {
    let L, R;
    try { L = JSON.parse(localStr); R = JSON.parse(remoteStr); } catch (e) { return localNewer ? localStr : remoteStr; }
    if (!L || !R || L.schemaVersion !== R.schemaVersion) return localNewer ? localStr : remoteStr;
    const base = JSON.parse(JSON.stringify(localNewer ? L : R)), other = localNewer ? R : L;
    const union = (a, b) => {
      const seen = new Set((a || []).map(x => x && x.id));
      return (a || []).concat((b || []).filter(x => x && x.id && !seen.has(x.id)));
    };
    base.plans = union(base.plans, other.plans);
    base.slips = union(base.slips, other.slips);
    base.ledger = union(base.ledger, other.ledger).sort((x, y) => String(x.timestamp || '').localeCompare(String(y.timestamp || '')));
    return JSON.stringify(base);
  }

  // Yerel + uzak durumu birleştir. Dönüş: {data (uzağa yazılacak), changed (yerel değişti mi)}
  function merge(remote, firstSync) {
    const m = metaGet();
    const data = Object.assign({}, remote || {});
    let changed = false;
    const keys = new Set(localKeys().concat(Object.keys(m.t)).concat(Object.keys(data)));
    applying = true;
    try {
      for (const k of keys) {
        if (!tracked(k)) continue;
        const lv = origGet.call(LS, k), lt = m.t[k] || 0, r = data[k];
        if (!r) { if (lv != null || lt) data[k] = { v: lv, t: lt }; continue; }
        if (r.v === lv) { m.t[k] = Math.max(lt, r.t || 0); continue; }
        let winner;
        if (firstSync && k === PAPER && lv != null && r.v != null) {
          winner = { v: mergePaper(lv, r.v, lt > (r.t || 0)), t: Date.now() };
        } else if (lv == null && !lt) {
          winner = r;                                   // yerelde hiç yoktu
        } else {
          winner = lt > (r.t || 0) ? { v: lv, t: lt } : r;   // eşitlikte uzak kazanır (yerel yedeklenir)
        }
        data[k] = winner;
        m.t[k] = winner.t || 0;
        if (winner.v !== lv) {
          backup(k, lv);
          if (winner.v == null) origRemove.call(LS, k); else origSet.call(LS, k, winner.v);
          changed = true;
        }
      }
    } finally { applying = false; }
    metaSet(m);
    return { data, changed };
  }

  async function fetchRemote() {
    const { data, error } = await client.from('user_state').select('data').eq('user_id', user.id).maybeSingle();
    if (error) throw error;
    return (data && data.data) || {};
  }
  async function writeRemote(data) {
    const { error } = await client.from('user_state').upsert({ user_id: user.id, data, updated_at: new Date().toISOString() });
    if (error) throw error;
    lastSync = new Date();
    setStatus('');
  }

  async function syncNow(opts) {
    if (!client || !user) return;
    if (pushing) { await pushing.catch(() => {}); }
    pushing = (async () => {
      const m = metaGet();
      // Bu cihazda başka bir hesap senkronlanmışsa, onun verisini bu hesaba karıştırma
      if (m.owner && m.owner !== user.id) {
        applying = true;
        try { localKeys().forEach(k => { backup(k, origGet.call(LS, k)); origRemove.call(LS, k); }); } finally { applying = false; }
        metaSet({ t: {}, synced: {}, owner: user.id });
      }
      const firstSync = !metaGet().synced[user.id];
      const remote = await fetchRemote();
      const res = merge(remote, firstSync);
      await writeRemote(res.data);
      const m2 = metaGet(); m2.synced[user.id] = true; m2.owner = user.id; metaSet(m2);
      if (res.changed) onRemoteChange(opts && opts.initial);
      else if (opts && opts.initial) { try { sessionStorage.removeItem('betavus_sync_reload'); } catch (e) {} }
    })();
    try { await pushing; } catch (e) { setStatus('Senkron hatası — değişiklikler bu cihazda saklandı, tekrar denenecek.'); console.warn('sync', e); }
    finally { pushing = null; }
  }
  function schedulePush() {
    if (!client || !user) return;
    clearTimeout(pushTimer);
    pushTimer = setTimeout(() => syncNow(), 1500);
  }
  function onRemoteChange(initial) {
    if (initial) {
      // İlk girişte veriler geldi: arayüz eski durumu göstermesin diye bir kez yenile
      try { if (sessionStorage.getItem('betavus_sync_reload') !== '1') { sessionStorage.setItem('betavus_sync_reload', '1'); location.reload(); return; } } catch (e) {}
    }
    toast('Başka bir cihazdaki değişiklikler alındı.', 'Yenile', () => location.reload());
  }

  // ---------------------------------------------------------------- UI
  const $ = id => document.getElementById(id);
  const escH = s => String(s == null ? '' : s).replace(/[&<>'"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
  const COUNTRY_CODES = 'TR NL DE BE GB FR ES IT PT AT CH AZ CY GR BG RO RS BA MK AL XK HR SI HU CZ SK PL UA RU GE SE NO DK FI IE LU US CA AU NZ ZA EG MA TN DZ SA AE QA KW BH OM JO LB IQ IR IL KZ UZ TM KG TJ CN JP KR IN PK BR AR MX CL CO PE'.split(' ');
  function countryOptions(sel) {
    let dn = null; try { dn = new Intl.DisplayNames(['tr'], { type: 'region' }); } catch (e) {}
    const list = COUNTRY_CODES.map(c => [c, dn ? dn.of(c) : c]);
    const top = list.slice(0, 2), rest = list.slice(2).sort((a, b) => a[1].localeCompare(b[1], 'tr'));
    return `<option value="">Seç…</option>` + top.concat(rest).concat([['OTHER', 'Diğer']])
      .map(([c, n]) => `<option value="${c}"${c === sel ? ' selected' : ''}>${escH(n)}</option>`).join('');
  }
  const GENDERS = [['erkek', 'Erkek'], ['kadin', 'Kadın'], ['belirtmek_istemiyorum', 'Belirtmek istemiyorum']];
  const genderOptions = sel => `<option value="">Seç…</option>` + GENDERS.map(([v, n]) => `<option value="${v}"${v === sel ? ' selected' : ''}>${n}</option>`).join('');
  const countryName = c => { if (c === 'OTHER') return 'Diğer'; try { return new Intl.DisplayNames(['tr'], { type: 'region' }).of(c); } catch (e) { return c; } };
  const USER_RE = /^[A-Za-z0-9_.]{3,20}$/;

  function modal(html) {
    let bg = $('authBg');
    if (!bg) {
      bg = document.createElement('div'); bg.id = 'authBg'; bg.className = 'auth-bg';
      bg.addEventListener('click', e => { if (e.target === bg) close(); });
      document.body.appendChild(bg);
    }
    bg.innerHTML = `<div class="auth" role="dialog" aria-modal="true">${html}</div>`;
    bg.hidden = false;
    const x = bg.querySelector('.auth-x'); if (x) x.onclick = close;
    const f = bg.querySelector('input,select'); if (f) setTimeout(() => f.focus(), 30);
    return bg;
  }
  function close() { const bg = $('authBg'); if (bg) { bg.hidden = true; bg.innerHTML = ''; } }
  function msg(text, kind) { const el = $('aMsg'); if (!el) return; el.textContent = text; el.className = 'a-msg ' + (kind || 'err'); el.hidden = !text; }
  function busy(btn, on, label) { if (!btn) return; btn.disabled = on; if (label) btn.textContent = label; }
  function setStatus(t) { const el = $('aSync'); if (el) el.textContent = t || (lastSync ? 'Son senkron: ' + lastSync.toLocaleTimeString('tr-TR') : ''); }
  function toast(text, action, fn) {
    let t = $('syncToast');
    if (!t) { t = document.createElement('div'); t.id = 'syncToast'; t.className = 'sync-toast'; document.body.appendChild(t); }
    t.innerHTML = `<span>${escH(text)}</span>${action ? `<button type="button">${escH(action)}</button>` : ''}`;
    t.hidden = false;
    if (action) t.querySelector('button').onclick = () => { t.hidden = true; fn(); };
    else setTimeout(() => { t.hidden = true; }, 4000);
  }

  const GOOGLE_SVG = '<svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true"><path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3C33.7 32.7 29.2 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.3-.1-2.4-.4-3.5z"/><path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.7 15.1 19 12 24 12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 16.3 4 9.7 8.3 6.3 14.7z"/><path fill="#4CAF50" d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2C29.2 35.1 26.7 36 24 36c-5.2 0-9.6-3.3-11.3-7.9l-6.5 5C9.5 39.6 16.2 44 24 44z"/><path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.2-2.2 4.2-4.1 5.6l6.2 5.2C37 39.2 44 34 44 24c0-1.3-.1-2.4-.4-3.5z"/></svg>';

  function tabsHtml(active) {
    return `<div class="auth-tabs"><button type="button" data-v="login" class="${active === 'login' ? 'on' : ''}">Giriş Yap</button><button type="button" data-v="register" class="${active === 'register' ? 'on' : ''}">Kayıt Ol</button><button type="button" class="auth-x" aria-label="Kapat">×</button></div>`;
  }
  function wireTabs(bg) { bg.querySelectorAll('.auth-tabs [data-v]').forEach(b => b.onclick = () => (b.dataset.v === 'login' ? showLogin() : showRegister())); }

  function showNotConfigured() {
    modal(`<div class="auth-tabs"><button type="button" class="on">Üyelik</button><button type="button" class="auth-x" aria-label="Kapat">×</button></div>
      <div class="auth-body"><h3>Üyelik çok yakında</h3><p class="a-sub">Kayıt ve giriş sistemi henüz etkinleştirilmedi. Şimdilik Sanal Kasa verilerin yalnızca bu cihazda saklanıyor.</p>
      <button class="a-btn" type="button" id="aOk">Tamam</button></div>`);
    $('aOk').onclick = close;
  }

  // Google ile giriş, Supabase'de Google sağlayıcısı açılana kadar gizli (auth_config.js → googleEnabled)
  const GOOGLE_ENABLED = cfg.googleEnabled === true;
  function googleBtn() { return GOOGLE_ENABLED ? `<button class="a-google" type="button" id="aGoogle">${GOOGLE_SVG} Google ile devam et</button><div class="a-or">veya</div>` : ''; }
  function wireGoogle() {
    if (!$('aGoogle')) return;
    $('aGoogle').onclick = async () => {
      busy($('aGoogle'), true);
      const { error } = await client.auth.signInWithOAuth({ provider: 'google', options: { redirectTo: location.origin + location.pathname } });
      if (error) { busy($('aGoogle'), false); msg('Google ile giriş başlatılamadı: ' + error.message); }
    };
  }

  function showLogin() {
    const bg = modal(`${tabsHtml('login')}<div class="auth-body">
      <h3>Tekrar hoş geldin</h3><p class="a-sub">Sanal Kasa ve tercihlerin her cihazda seninle olsun.</p>
      ${googleBtn()}
      <form id="aForm" novalidate>
        <label class="a-field"><span>Kullanıcı adı veya e-posta</span><input id="aId" autocomplete="username" required></label>
        <label class="a-field"><span>Şifre</span><input id="aPw" type="password" autocomplete="current-password" required></label>
        <button class="a-btn" type="submit" id="aSubmit">Giriş Yap</button>
      </form>
      <p style="margin:10px 0 0;font-size:13px"><button type="button" class="a-link" id="aForgot">Şifremi unuttum</button></p>
      <div id="aMsg" class="a-msg" hidden></div></div>`);
    wireTabs(bg); wireGoogle();
    $('aForgot').onclick = showForgot;
    $('aForm').onsubmit = async e => {
      e.preventDefault();
      const id = $('aId').value.trim(), pw = $('aPw').value;
      if (!id || !pw) return msg('Kullanıcı adı/e-posta ve şifre gerekli.');
      const btn = $('aSubmit'); busy(btn, true, 'Giriş yapılıyor…'); msg('');
      try {
        let email = id;
        if (!id.includes('@')) {
          const { data, error } = await client.rpc('email_for_login', { p_username: id, p_password: pw });
          if (error) throw new Error(/too_many/.test(error.message) ? 'Çok fazla hatalı deneme. 15 dakika sonra tekrar dene.' : error.message);
          if (!data) throw new Error('Kullanıcı adı veya şifre hatalı.');
          email = data;
        }
        const { error } = await client.auth.signInWithPassword({ email, password: pw });
        if (error) throw new Error(/confirm/i.test(error.message) ? 'E-posta adresin henüz doğrulanmadı. Gelen kutundaki bağlantıya tıkla.' : 'Kullanıcı adı / e-posta veya şifre hatalı.');
        close();
      } catch (err) { msg(err.message); busy(btn, false, 'Giriş Yap'); }
    };
  }

  function showForgot() {
    modal(`${tabsHtml('login')}<div class="auth-body"><h3>Şifre sıfırlama</h3><p class="a-sub">Kayıtlı e-posta adresine bir sıfırlama bağlantısı gönderelim.</p>
      <form id="aForm" novalidate><label class="a-field"><span>E-posta</span><input id="aEmail" type="email" autocomplete="email"></label>
      <button class="a-btn" type="submit" id="aSubmit">Bağlantı gönder</button></form><div id="aMsg" class="a-msg" hidden></div></div>`);
    wireTabs($('authBg'));
    $('aForm').onsubmit = async e => {
      e.preventDefault();
      const email = $('aEmail').value.trim(); if (!/.+@.+\..+/.test(email)) return msg('Geçerli bir e-posta gir.');
      busy($('aSubmit'), true);
      const { error } = await client.auth.resetPasswordForEmail(email, { redirectTo: location.origin + location.pathname });
      busy($('aSubmit'), false);
      if (error) msg(error.message); else msg('Bağlantı gönderildi. E-postanı kontrol et.', 'ok');
    };
  }

  function showNewPassword() {
    modal(`<div class="auth-tabs"><button type="button" class="on">Yeni şifre</button><button type="button" class="auth-x" aria-label="Kapat">×</button></div>
      <div class="auth-body"><h3>Yeni şifreni belirle</h3><form id="aForm" novalidate>
      <label class="a-field"><span>Yeni şifre (en az 8 karakter)</span><input id="aPw" type="password" autocomplete="new-password"></label>
      <button class="a-btn" type="submit" id="aSubmit">Kaydet</button></form><div id="aMsg" class="a-msg" hidden></div></div>`);
    $('aForm').onsubmit = async e => {
      e.preventDefault();
      const pw = $('aPw').value; if (pw.length < 8) return msg('Şifre en az 8 karakter olmalı.');
      busy($('aSubmit'), true);
      const { error } = await client.auth.updateUser({ password: pw });
      busy($('aSubmit'), false);
      if (error) msg(error.message); else { msg('Şifren güncellendi.', 'ok'); setTimeout(close, 1200); }
    };
  }

  function profileFields(p) {
    p = p || {};
    return `<label class="a-field"><span>Kullanıcı adı</span><input id="aUser" autocomplete="nickname" maxlength="20" value="${escH(p.username || '')}" placeholder="3–20 karakter: harf, rakam, _ ."></label>
      <div class="a-row">
        <label class="a-field"><span>Yaş</span><input id="aAge" type="number" inputmode="numeric" min="18" max="100" value="${escH(p.age || '')}"></label>
        <label class="a-field"><span>Cinsiyet</span><select id="aGender">${genderOptions(p.gender)}</select></label>
      </div>
      <label class="a-field"><span>Ülke</span><select id="aCountry">${countryOptions(p.country)}</select></label>`;
  }
  async function readProfileFields() {
    const username = $('aUser').value.trim(), age = parseInt($('aAge').value, 10), gender = $('aGender').value, country = $('aCountry').value;
    if (!USER_RE.test(username)) throw new Error('Kullanıcı adı 3–20 karakter olmalı; yalnızca harf, rakam, _ ve . kullanılabilir.');
    if (!Number.isFinite(age) || age < 18 || age > 100) throw new Error('BETAVUS\'u kullanmak için 18 yaşından büyük olmalısın.');
    if (!gender) throw new Error('Cinsiyet seç.');
    if (!country) throw new Error('Ülke seç.');
    return { username, age, gender, country };
  }
  async function checkUsername(username, ownId) {
    const { data, error } = await client.rpc('username_available', { p_username: username });
    if (error) throw new Error(error.message);
    if (!data && !(profile && ownId && profile.username && profile.username.toLowerCase() === username.toLowerCase())) throw new Error('Bu kullanıcı adı alınmış.');
  }

  function showRegister() {
    const bg = modal(`${tabsHtml('register')}<div class="auth-body">
      <h3>Hesap oluştur</h3><p class="a-sub">Sadece birkaç basit bilgi. Verilerin hiçbir cihazda kaybolmaz.</p>
      ${googleBtn()}
      <form id="aForm" novalidate>
        ${profileFields()}
        <label class="a-field"><span>E-posta</span><input id="aEmail" type="email" autocomplete="email"></label>
        <label class="a-field"><span>Şifre (en az 8 karakter)</span><input id="aPw" type="password" autocomplete="new-password"></label>
        <button class="a-btn" type="submit" id="aSubmit">Kayıt Ol</button>
      </form><div id="aMsg" class="a-msg" hidden></div></div>`);
    wireTabs(bg); wireGoogle();
    $('aForm').onsubmit = async e => {
      e.preventDefault(); msg('');
      const btn = $('aSubmit');
      try {
        const p = await readProfileFields();
        const email = $('aEmail').value.trim(), pw = $('aPw').value;
        if (!/.+@.+\..+/.test(email)) throw new Error('Geçerli bir e-posta gir.');
        if (pw.length < 8) throw new Error('Şifre en az 8 karakter olmalı.');
        busy(btn, true, 'Kaydediliyor…');
        await checkUsername(p.username);
        const { data, error } = await client.auth.signUp({ email, password: pw, options: { data: p, emailRedirectTo: location.origin + location.pathname } });
        if (error) throw new Error(/registered|exists/i.test(error.message) ? 'Bu e-posta ile zaten bir hesap var. Giriş yapmayı dene.' : error.message);
        if (!data.session) {
          $('aForm').hidden = true;
          bg.querySelectorAll('.a-google, .a-or').forEach(el => { el.hidden = true; });
          msg('Kayıt alındı! ' + email + ' adresine bir doğrulama bağlantısı gönderdik. Bağlantıya tıkladıktan sonra giriş yapabilirsin.', 'ok');
        } else close();
      } catch (err) { msg(err.message); busy(btn, false, 'Kayıt Ol'); }
    };
  }

  function showCompleteProfile() {
    modal(`<div class="auth-tabs"><button type="button" class="on">Profilini tamamla</button></div>
      <div class="auth-body"><h3>Son bir adım</h3><p class="a-sub">Hesabın açıldı. Birkaç basit bilgiyle profilini tamamla.</p>
      <form id="aForm" novalidate>${profileFields(profile)}<button class="a-btn" type="submit" id="aSubmit">Kaydet</button></form>
      <p style="margin:10px 0 0;font-size:13px"><button type="button" class="a-link" id="aOut">Çıkış yap</button></p>
      <div id="aMsg" class="a-msg" hidden></div></div>`);
    $('aOut').onclick = signOut;
    $('aForm').onsubmit = async e => {
      e.preventDefault(); msg('');
      const btn = $('aSubmit');
      try {
        const p = await readProfileFields();
        busy(btn, true, 'Kaydediliyor…');
        await checkUsername(p.username, user.id);
        const { error } = await client.from('profiles').upsert(Object.assign({ id: user.id, updated_at: new Date().toISOString() }, p));
        if (error) throw new Error(/duplicate|unique/i.test(error.message) ? 'Bu kullanıcı adı alınmış.' : error.message);
        profile = Object.assign({}, profile, p);
        renderButton(); close();
      } catch (err) { msg(err.message); busy(btn, false, 'Kaydet'); }
    };
  }

  function showAccount() {
    const p = profile || {};
    const g = (GENDERS.find(x => x[0] === p.gender) || [, '—'])[1];
    modal(`<div class="auth-tabs"><button type="button" class="on">Hesabım</button><button type="button" class="auth-x" aria-label="Kapat">×</button></div>
      <div class="auth-body"><h3>${escH(p.username || 'Hesabım')}</h3>
      <dl class="a-prof"><dt>E-posta</dt><dd>${escH(user.email || '—')}</dd><dt>Yaş</dt><dd>${escH(p.age || '—')}</dd>
        <dt>Cinsiyet</dt><dd>${escH(g)}</dd><dt>Ülke</dt><dd>${escH(p.country ? countryName(p.country) : '—')}</dd></dl>
      <p class="a-sub">Sanal Kasa ve tercihlerin hesabına kaydedilir; başka bir cihazda giriş yaptığında aynı veriler orada da görünür. <span id="aSync"></span></p>
      <button class="a-btn" type="button" id="aSyncNow">Şimdi senkronla</button>
      <p style="margin:12px 0 0;font-size:13px;display:flex;gap:16px"><button type="button" class="a-link" id="aEdit">Profili düzenle</button><button type="button" class="a-link" id="aOut">Çıkış yap</button></p>
      <div id="aMsg" class="a-msg" hidden></div></div>`);
    setStatus('');
    $('aSyncNow').onclick = async () => { busy($('aSyncNow'), true, 'Senkronlanıyor…'); await syncNow(); busy($('aSyncNow'), false, 'Şimdi senkronla'); setStatus(''); };
    $('aEdit').onclick = showCompleteProfile;
    $('aOut').onclick = signOut;
  }

  async function signOut() {
    clearTimeout(pushTimer);
    try { await syncNow(); } catch (e) {}
    await client.auth.signOut();
    close();
  }

  function profileComplete(p) { return !!(p && p.username && p.age && p.gender && p.country); }
  function renderButton() {
    const b = $('acctBtn'); if (!b) return;
    if (user) { b.textContent = '👤 ' + ((profile && profile.username) || 'Hesabım'); b.classList.add('in'); }
    else { b.textContent = 'Giriş / Kayıt Ol'; b.classList.remove('in'); }
  }
  async function loadProfile() {
    const { data } = await client.from('profiles').select('username,age,gender,country').eq('id', user.id).maybeSingle();
    profile = data || null;
  }

  let started = false;
  async function onSession(session, event) {
    const prev = user && user.id;
    user = session ? session.user : null;
    if (!user) { profile = null; renderButton(); return; }
    if (prev === user.id && event !== 'USER_UPDATED') return;
    await loadProfile().catch(() => {});
    renderButton();
    if (!profileComplete(profile)) showCompleteProfile();
    await syncNow({ initial: true });
  }

  function init() {
    const b = $('acctBtn');
    if (!configured) { if (b) b.onclick = showNotConfigured; return; }
    if (started) return; started = true;
    client = root.supabase.createClient(cfg.supabaseUrl, cfg.supabaseAnonKey, { auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true } });
    if (b) b.onclick = () => (user ? (profileComplete(profile) ? showAccount() : showCompleteProfile()) : showLogin());
    client.auth.onAuthStateChange((event, session) => {
      if (event === 'PASSWORD_RECOVERY') { user = session && session.user; showNewPassword(); return; }
      // Supabase çağrılarını bu callback'in içinde beklemek kilitlenmeye yol açabilir; bir sonraki tura bırak
      setTimeout(() => onSession(session, event), 0);
    });
    // Diğer cihazdaki değişiklikleri al: sekmeye dönüldüğünde ve 60 sn'de bir
    document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') syncNow(); });
    setInterval(() => { if (document.visibilityState === 'visible') syncNow(); }, 60000);
  }

  root.BETAVUS_AUTH = { init, syncNow, isConfigured: () => configured, _merge: merge, _mergePaper: mergePaper };
})(window);
