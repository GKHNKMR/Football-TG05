"""#26 arayüz kolaylıkları (js/ux_extras.js): Aurora arka planı, günün öne çıkanları, iskelet +
dolan göstergeler, sayarak yükselen sayılar, mobil alt menü, ilk giriş rehberi."""
import os
import sys
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get('UX_SHOTS', '')) if os.environ.get('UX_SHOTS') else None
ACCESS = '1f7b720c52ea3f6e8631a8eeaffaa7113fbed540ec0772108d39c52835d9855d'


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


srv = HTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(ROOT)))
PORT = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f'http://127.0.0.1:{PORT}/index.html'

with sync_playwright() as p:
    b = p.chromium.launch()
    errs = []

    # ---- masaüstü, ilk ziyaret: rehber açılır
    pg = b.new_page(viewport={'width': 1280, 'height': 900})
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.add_init_script(f"localStorage.setItem('betavus.access','{ACCESS}');localStorage.setItem('betavus.lang','tr');localStorage.setItem('betavus.tour_force','1');")
    pg.goto(URL)
    pg.wait_for_selector('#rows .row[data-mid]', timeout=15000)
    assert pg.locator('.aurora').count() == 1
    bg = pg.evaluate("getComputedStyle(document.querySelector('.table')).backdropFilter")
    assert 'blur' in bg, bg
    print('✓ A · Aurora: arka plan ışığı ve buzlu cam kutular var.')

    pg.wait_for_selector('#todayCard:not([hidden]) .tdy-pick', timeout=5000)
    picks = pg.eval_on_selector_all('#todayCard .tdy-pick', 'els=>els.map(e=>e.innerText.replace(/\\s+/g," "))')
    assert 1 <= len(picks) <= 3, picks
    ps = pg.eval_on_selector_all('#todayCard .tdy-ring', 'els=>els.map(e=>+getComputedStyle(e).getPropertyValue("--p"))')
    assert ps == sorted(ps, reverse=True) and all(75 <= v <= 100 for v in ps), ps
    print(f'✓ 2 · Günün öne çıkanları: {len(picks)} vurgu, olasılığa göre sıralı ({ps}).')

    pv = pg.eval_on_selector_all('#rows .pill.pf', 'els=>els.slice(0,6).map(e=>e.style.getPropertyValue("--pv"))')
    assert pv and all(v.isdigit() for v in pv), pv
    allpv = pg.eval_on_selector_all('#rows .pill.pf', 'els=>els.map(e=>+e.style.getPropertyValue("--pv"))')
    assert max(allpv) <= 100, f'çubuk 100 değerini aşıyor: {max(allpv)}'
    print('✓ 3 · Olasılık kutularında dolan çubuk (--pv) var.')

    pg.wait_for_selector('.tour .tour-tip', timeout=5000)
    assert '1/' in pg.inner_text('.tour-n')
    if OUT:
        pg.screenshot(path=str(OUT / 'ux-tour-desktop.png'))
    steps = int(pg.inner_text('.tour-n').split('/')[1])
    for _ in range(steps):
        pg.click('.tour-next')
    assert pg.locator('.tour').count() == 0
    assert pg.evaluate("localStorage.getItem('betavus.tour_pred_v1')") == '1'
    print(f'✓ 4 · Fikstür rehberi ilk açılışta {steps} adımda açılıp kapanıyor, bir daha gösterilmiyor.')
    assert pg.is_visible('#tourBtn')

    # her sekmenin kendi rehberi: ilk açılışta bir kez
    for tab in ('stats', 'plan'):
        pg.evaluate(f"setTab('{tab}')")
        pg.wait_for_selector(f'.tour[data-tab="{tab}"] .tour-tip', timeout=8000)
        n = int(pg.inner_text('.tour-n').split('/')[1])
        titles = []
        for _ in range(n):
            titles.append(pg.inner_text('#tourT'))
            pg.click('.tour-next')
        assert pg.locator('.tour').count() == 0 and n >= 3, (tab, n)
        print(f'✓ 4 · {tab} rehberi ilk açılışta: {n} adım ({" / ".join(titles)}).')
    pg.evaluate("setTab('stats')")
    pg.wait_for_timeout(1500)
    assert pg.locator('.tour').count() == 0, 'görülen rehber tekrar açılmamalı'
    pg.click('#tourBtn')
    pg.wait_for_selector('.tour[data-tab="stats"] .tour-tip', timeout=5000)
    pg.keyboard.press('Escape')
    assert pg.locator('.tour').count() == 0
    print("✓ 4 · Köşedeki 'Rehber' düğmesi o sekmenin turunu yeniden açıyor, Esc kapatıyor.")
    if OUT:
        pg.screenshot(path=str(OUT / 'ux-desktop.png'))

    # sayarak yükselme: İstatistikler sekmesine geçince büyük sayı önce 0'dan başlar
    pg.evaluate("setTab('stats')")
    pg.wait_for_selector('#pane-stats .st-kpi .v', timeout=15000)
    assert pg.evaluate("document.getElementById('pane-stats').classList.contains('fillin')")
    pg.wait_for_timeout(1300)
    final = pg.inner_text('#pane-stats .st-kpi .v')
    assert any(c.isdigit() for c in final) and not final.startswith('%0,0'), final
    print(f'✓ 3 · Sekme açılınca göstergeler doluyor, sayı sonunda gerçek değerde: {final}')
    nums = pg.evaluate("""[['%89,5','89,5'],['12.660','12660'],['1,57','1,57'],['€25,000.00','25,000.00'],['83.2%','83.2']].map(([t])=>{const p=BV_UX._parseNum(t);return BV_UX._fmtNum(p.v,p)})""")
    assert nums == ['89,5', '12.660', '1,57', '25,000.00', '83.2'], nums
    print('✓ Sayı biçimi korunuyor (89,5 · 12.660 · 1,57 · 25,000.00 · 83.2).')

    # rehber yeniden: SSS'teki düğme
    pg.evaluate("setTab('faq')")
    assert not pg.is_visible('#tourBtn')
    pg.click('[data-tour="plan"]')
    pg.wait_for_selector('.tour[data-tab="plan"] .tour-tip', timeout=5000)
    assert pg.evaluate("document.documentElement.dataset.page") == 'plan'
    pg.click('.tour-skip')
    assert pg.locator('.tour').count() == 0
    print("✓ 4 · SSS'teki sayfa rehberi düğmeleri doğru sekmeye geçip turu açıyor, 'Atla' kapatıyor.")

    # ---- telefon: alt menü
    m = b.new_page(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
    m.on('pageerror', lambda e: errs.append(str(e)))
    m.add_init_script(f"localStorage.setItem('betavus.access','{ACCESS}');localStorage.setItem('betavus.lang','tr');localStorage.setItem('betavus.tour_v1','1');")
    m.goto(URL)
    m.wait_for_selector('#rows .row[data-mid]', timeout=15000)
    assert m.is_visible('#botNav') and not m.is_visible('.top .tabs')
    assert m.locator('.tour').count() == 0, 'rehber bir kez görüldüyse açılmamalı'
    if OUT:
        m.screenshot(path=str(OUT / 'ux-mobile.png'))
    m.click('#botNav [data-pg="plan"]')
    assert m.evaluate("document.documentElement.dataset.page") == 'plan'
    assert m.get_attribute('#botNav [data-pg="plan"]', 'aria-current') == 'page'
    sw = m.evaluate('document.documentElement.scrollWidth')
    assert sw <= 390, f'yatay kaydırma var: {sw}'
    print('✓ 1 · Telefonda alt menü görünüyor, üst sekmeler gizli, sekme değişiyor, yatay taşma yok.')
    d = b.new_page(viewport={'width': 1280, 'height': 900})
    d.add_init_script(f"localStorage.setItem('betavus.access','{ACCESS}');localStorage.setItem('betavus.tour_v1','1');")
    d.goto(URL)
    d.wait_for_selector('#rows .row[data-mid]', timeout=15000)
    assert not d.is_visible('#botNav')
    print('✓ 1 · Masaüstünde alt menü gizli.')

    assert not errs, errs
    b.close()
print('\nTüm arayüz kolaylığı testleri geçti.')

# ---- #27 dönen top + lig yörüngesi
with sync_playwright() as p:
    b = p.chromium.launch()
    errs = []
    pg = b.new_page(viewport={'width': 1280, 'height': 900})
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.add_init_script(f"localStorage.setItem('betavus.access','{ACCESS}');localStorage.setItem('betavus.lang','tr');localStorage.setItem('betavus.tour_pred_v1','1');")
    pg.goto(URL)
    pg.wait_for_selector('#rows .row[data-mid]', timeout=15000)
    pg.wait_for_selector('#orb .orb-flag', timeout=5000)
    nflags = pg.locator('#orb .orb-flag').count()
    nlg = pg.evaluate("new Set(window.__data.map(x=>x.league)).size")
    assert nflags == nlg, (nflags, nlg)
    a1 = pg.evaluate("document.querySelector('#orb .orb-flag').style.transform")
    pg.wait_for_timeout(600)
    a2 = pg.evaluate("document.querySelector('#orb .orb-flag').style.transform")
    assert a1 != a2, 'yörünge dönmüyor'
    assert pg.locator('#orb .orb-stat').count() == 0, 'bandı tekrar eden sayı kutuları olmamalı'
    pg.evaluate("setTab('stats')")
    assert not pg.is_visible('#orb'), 'top yalnız Fikstür sekmesinde'
    pg.evaluate("setTab('pred')")
    assert pg.is_visible('#orb')
    print(f'✓ #27 Top: {nflags} lig bayrağı dönüyor, yalnız Fikstür sekmesinde, tekrar eden sayı kutusu yok.')
    # bayrağa tıklayınca Fikstür o lige filtrelenir (öndeki, tıklanabilir bir bayrak)
    pg.mouse.move(5, 5)
    i = pg.evaluate("[...document.querySelectorAll('#orb .orb-flag')].findIndex(b=>b.style.pointerEvents!=='none' && +b.style.opacity>0.8)")
    lg = pg.evaluate(f"document.querySelectorAll('#orb .orb-flag')[{i}].title")
    pg.locator('#orb .orb-flag').nth(i).click(force=True)
    pg.wait_for_timeout(300)
    leagues = set(pg.eval_on_selector_all('#rows .row[data-league]', 'els=>els.map(e=>e.dataset.league)'))
    assert leagues == {lg}, (lg, leagues)
    print(f'✓ #27 Bayrağa tıklama Fikstür\'ü {lg} ile filtreliyor.')
    m = b.new_page(viewport={'width': 390, 'height': 844}, is_mobile=True)
    m.on('pageerror', lambda e: errs.append(str(e)))
    m.add_init_script(f"localStorage.setItem('betavus.access','{ACCESS}');")
    m.goto(URL)
    m.wait_for_selector('#orb .orb-flag', timeout=15000)
    assert m.evaluate('document.documentElement.scrollWidth') <= 390
    ow = m.evaluate("document.getElementById('orb').getBoundingClientRect().width")
    assert 200 <= ow <= 330, ow
    print(f'✓ #27 Telefonda top başlığın altında ({ow:.0f}px), yatay taşma yok.')
    assert not errs, errs
    b.close()
print('Dönen top testleri geçti.')

# ---- İstatistikler listesi: 0.5+ … X2 sütunlarına göre sıralama
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1280, 'height': 900})
    pg.add_init_script(f"localStorage.setItem('betavus.access','{ACCESS}');localStorage.setItem('betavus.lang','tr');")
    pg.goto(URL)
    pg.wait_for_selector('#rows .row[data-mid]', timeout=15000)
    pg.evaluate("setTab('stats')")
    pg.wait_for_selector('.st-list .st-row', timeout=15000)
    col = lambda i: pg.eval_on_selector_all(f'.st-matches tbody tr.st-row td:nth-child({i})', 'els=>els.map(e=>parseFloat(e.textContent.replace(",",".")))')
    for k, i in [('1.5+', 5), ('X2', 9)]:
        pg.click(f'.st-matches th[data-sort="{k}"]')
        v = col(i); assert v == sorted(v, reverse=True), (k, v[:6])
        pg.click(f'.st-matches th[data-sort="{k}"]')
        v = col(i); assert v == sorted(v), (k, v[:6])
    pg.click('#stDateSort')
    d = pg.eval_on_selector_all('.st-matches tbody tr.st-row td:nth-child(1)', 'els=>els.map(e=>e.textContent.split(".").reverse().join(""))')
    assert d == sorted(d, reverse=True), d[:5]
    print('✓ İstatistikler listesi: 0.5+ … X2 başlıkları yüksekten düşüğe / düşükten yükseğe sıralıyor, Tarih geri alıyor.')
    b.close()
