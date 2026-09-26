"""'Hesabımı sil' akışı (iş listesi #12) — sahte Supabase istemcisiyle uçtan uca.

1. Onay: kullanıcı adı yazılmadan buton kapalı (büyük/küçük harf fark etmez)
2. Supabase'de fonksiyon henüz yoksa (PGRST202): anlaşılır mesaj, hiçbir şey silinmez, oturum açık kalır
3. Başarılı silme + "bu cihazdaki verileri de sil": Sanal Kasa ve yedekler silinir, dil/tema kalır,
   yalnızca yerel oturum kapanır, silmeden sonra buluta yazma olmaz
4. Başarılı silme, kutu işaretsiz: Sanal Kasa bu cihazda kalır, senkron meta temizlenir
"""
import sys, time, threading
sys.stdout.reconfigure(encoding='utf-8')
from http.server import HTTPServer, SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright

class H(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

PORT = 8944
threading.Thread(target=HTTPServer(('127.0.0.1', PORT), H).serve_forever, daemon=True).start()

FAKE_SUPABASE = """
window.__sb = window.__sb || { calls: [], rpcMode: 'missing' };
window.supabase = { createClient: () => {
  const listeners = [];
  const session = { user: { id: 'u1', email: 'tester@example.com' } };
  const q = table => { const b = {
    select() { return b; }, eq() { return b; },
    maybeSingle: async () => { __sb.calls.push('select:' + table);
      return table === 'profiles' ? { data: { username: 'Tester', age: 30, gender: 'erkek', country: 'NL' } } : { data: null }; },
    upsert: async () => { __sb.calls.push('upsert:' + table); return { error: null }; } }; return b; };
  return {
    auth: {
      onAuthStateChange(cb) { listeners.push(cb); setTimeout(() => cb('SIGNED_IN', session), 10); return { data: { subscription: { unsubscribe() {} } } }; },
      signOut: async o => { __sb.calls.push('signOut:' + (o && o.scope)); listeners.forEach(cb => cb('SIGNED_OUT', null)); return { error: null }; },
      signInWithOAuth: async () => ({}) },
    from: q,
    rpc: async name => { __sb.calls.push('rpc:' + name);
      if (name !== 'delete_my_account') return { data: true, error: null };
      return __sb.rpcMode === 'missing'
        ? { data: null, error: { code: 'PGRST202', message: 'Could not find the function public.delete_my_account without parameters in the schema cache' } }
        : { data: null, error: null }; } };
} };
"""
SEED = """localStorage.setItem('betavus.access','1f7b720c52ea3f6e8631a8eeaffaa7113fbed540ec0772108d39c52835d9855d');
localStorage.setItem('betavus.lang','tr'); localStorage.setItem('betavus.theme2','dark');
if (!sessionStorage.getItem('seeded')) { sessionStorage.setItem('seeded','1'); sessionStorage.setItem('betavus_sync_reload','1');
  localStorage.setItem('betavus.paper_v1', JSON.stringify({schemaVersion: 1, plans: [{id: 'p1'}]}));
  localStorage.setItem('betavus.__backup.paper_v1.1', '{}'); }"""


def open_delete(page):
    page.goto(f'http://127.0.0.1:{PORT}/index.html')
    page.wait_for_function("document.getElementById('acctBtn') && document.getElementById('acctBtn').classList.contains('in')", timeout=10000)
    page.click('#acctBtn')
    page.click('#aDel')
    page.wait_for_selector('#aDelConfirm')


def run_case(browser, mode, wipe):
    ctx = browser.new_context()
    page = ctx.new_page()
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.route('**/supabase.min.js', lambda r: r.fulfill(body=FAKE_SUPABASE, content_type='application/javascript'))
    page.add_init_script(SEED)
    open_delete(page)
    page.evaluate(f"__sb.rpcMode = '{mode}'")
    # 1. Onay
    assert page.is_disabled('#aSubmit'), "Kullanıcı adı yazılmadan silme butonu açık olmamalı"
    page.fill('#aDelConfirm', 'yanlis')
    assert page.is_disabled('#aSubmit'), "Yanlış kullanıcı adıyla buton açılmamalı"
    page.fill('#aDelConfirm', 'tester')
    assert page.is_enabled('#aSubmit'), "Doğru kullanıcı adıyla (küçük harf) buton açılmalı"
    if not wipe:
        page.uncheck('#aDelLocal')
    calls_before = page.evaluate("__sb.calls.length")
    page.click('#aSubmit')
    time.sleep(0.6)
    calls = page.evaluate(f"__sb.calls.slice({calls_before})")
    ls = page.evaluate("Object.fromEntries(Object.keys(localStorage).map(k => [k, localStorage.getItem(k)]))")
    assert not errors, errors
    return page, ctx, calls, ls


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)

    # 2. Fonksiyon kurulmamış
    page, ctx, calls, ls = run_case(browser, 'missing', True)
    assert calls == ['rpc:delete_my_account'], calls
    assert 'henüz etkinleştirilmedi' in page.inner_text('#aMsg'), page.inner_text('#aMsg')
    assert 'betavus.paper_v1' in ls, "Silme başarısızken yerel veri silinmemeli"
    assert page.is_enabled('#aSubmit'), "Hata sonrası buton yeniden kullanılabilir olmalı"
    assert page.evaluate("document.getElementById('acctBtn').classList.contains('in')"), "Oturum açık kalmalı"
    ctx.close()
    print("  ✓ Fonksiyon kurulmamışken: anlaşılır mesaj, veri ve oturum yerinde")

    # 3. Başarılı silme + bu cihazdaki verileri de sil
    page, ctx, calls, ls = run_case(browser, 'ok', True)
    assert calls[0] == 'rpc:delete_my_account' and 'signOut:local' in calls, calls
    assert not any(c.startswith('upsert') for c in calls), f"Silmeden sonra buluta yazılmamalı: {calls}"
    assert 'betavus.paper_v1' not in ls and not any(k.startswith('betavus.__backup.') for k in ls), ls.keys()
    assert ls.get('betavus.lang') == 'tr' and ls.get('betavus.theme2') == 'dark', "Dil ve tema korunmalı"
    assert 'betavus.__sync_meta' not in ls, "Senkron meta temizlenmeli"
    assert 'Hesabın silindi' in page.inner_text('.auth-body')
    assert not page.evaluate("document.getElementById('acctBtn').classList.contains('in')"), "Hesap butonu çıkış yapmış görünmeli"
    n_after = page.evaluate("__sb.calls.length")
    page.evaluate("localStorage.setItem('betavus.paper_v1', '{}')")   # yeni yerel yazma senkron tetiklemeye çalışır
    time.sleep(2.5)   # zamanlanmış senkron (1,5 sn) bu sürede çalışırdı
    later = page.evaluate(f"__sb.calls.slice({n_after})")
    assert not any(c.startswith('upsert') for c in later), f"Silmeden sonra geç senkron yazdı: {later}"
    ctx.close()
    print("  ✓ Silme + cihaz verisi: Sanal Kasa ve yedekler silindi, dil/tema kaldı, yerel oturum kapandı")

    # 4. Başarılı silme, kutu işaretsiz
    page, ctx, calls, ls = run_case(browser, 'ok', False)
    assert 'betavus.paper_v1' in ls, "Kutu işaretsizken Sanal Kasa cihazda kalmalı"
    assert 'betavus.__sync_meta' not in ls, "Senkron meta temizlenmeli (yeni hesaba eski sahip bilgisi taşınmasın)"
    assert 'yerinde duruyor' in page.inner_text('.auth-body')
    ctx.close()
    print("  ✓ Silme, cihaz verisi korunarak: Sanal Kasa bu cihazda kaldı")

    browser.close()
print("\n>>> HESABIMI SİL TESTLERİ GEÇTİ <<<")
