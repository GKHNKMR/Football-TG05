import time, threading, sys
sys.stdout.reconfigure(encoding='utf-8')
from http.server import HTTPServer, SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright

class H(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

server = HTTPServer(('127.0.0.1', 8908), H)
threading.Thread(target=server.serve_forever, daemon=True).start()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.on('console', lambda m: print('CONSOLE:', m.text))
    page.on('pageerror', lambda e: print('PAGE_ERROR:', e))
    page.on('dialog', lambda d: (print('DIALOG:', d.message), d.accept()))
    page.add_init_script("""
        localStorage.setItem('betavus.access', '1f7b720c52ea3f6e8631a8eeaffaa7113fbed540ec0772108d39c52835d9855d');
        localStorage.setItem('betavus.tab', 'sim-kasa');
    """)
    page.goto('http://127.0.0.1:8908/index.html')
    time.sleep(1)
    
    print('\n=== TEST 1: tab-sim-kasa chart buttons ===')
    for prof in ['minimum', 'medium', 'high', 'all']:
        btn = page.query_selector(f'#sim30ChartPills button[data-prof="{prof}"]')
        if btn:
            txt = btn.inner_text()
            btn.click()
            time.sleep(0.3)
            svg_html = page.inner_html('#sim30Svg')
            has_min = 'stroke="#10b981"' in svg_html
            has_med = 'stroke="#38bdf8"' in svg_html
            has_high = 'stroke="#ef4444"' in svg_html
            print(f"sim-kasa clicked {prof} ({txt}):")
            print(f"  has minimum curve (stroke #10b981): {has_min}")
            print(f"  has medium curve (stroke #38bdf8): {has_med}")
            print(f"  has high curve (stroke #ef4444): {has_high}")
            if prof == 'minimum':
                assert has_min and not has_med and not has_high, f"Failed: only minimum should be visible for {prof}"
            elif prof == 'medium':
                assert not has_min and has_med and not has_high, f"Failed: only medium should be visible for {prof}"
            elif prof == 'high':
                assert not has_min and not has_med and has_high, f"Failed: only high should be visible for {prof}"
            elif prof == 'all':
                assert has_min and has_med and has_high, f"Failed: all curves should be visible for {prof}"

    print('\n=== TEST 2: tab-plan chart buttons ===')
    page.click('#tab-plan')
    time.sleep(0.5)
    # create plan if needed
    if page.query_selector('#setupStartBank'):
        page.fill('#setupStartBank', '100')
        page.fill('#setupTargetBank', '1000')
        page.click('#btnCreatePlan')
        time.sleep(0.5)

    for view in ['all', 'minimum', 'medium', 'high']:
        chip = page.query_selector(f'#chartViewChips button[data-view="{view}"]')
        if chip:
            txt = chip.inner_text()
            chip.click()
            time.sleep(0.3)
            svg_html = page.inner_html('#planTrajectorySvg')
            has_min = 'stroke="#10b981"' in svg_html
            has_med = 'stroke="#3b82f6"' in svg_html
            has_high = 'stroke="#ef4444"' in svg_html
            print(f"plan clicked {view} ({txt}):")
            print(f"  has minimum (#10b981): {has_min}")
            print(f"  has medium (#3b82f6): {has_med}")
            print(f"  has high (#ef4444): {has_high}")
            if view == 'minimum':
                assert has_min and not has_med and not has_high, f"Failed: only minimum should be visible in plan for {view}"
            elif view == 'medium':
                assert not has_min and has_med and not has_high, f"Failed: only medium should be visible in plan for {view}"
            elif view == 'high':
                assert not has_min and not has_med and has_high, f"Failed: only high should be visible in plan for {view}"
            elif view == 'all':
                assert has_min and has_med and has_high, f"Failed: all curves should be visible in plan for {view}"

    print('\n=== TEST 3: Reset Plan button ===')
    btn_reset = page.query_selector('#btnResetPlan')
    assert btn_reset is not None, "btnResetPlan must exist"
    btn_reset.click()
    time.sleep(0.5)
    has_setup = page.query_selector('#setupStartBank') is not None
    print('After reset, pane-plan has setupStartBank:', has_setup)
    assert has_setup is True, "setupStartBank must be visible after reset"

    print('\n=== TEST 4: Admin Kuponlarım tile and filter clicks ===')
    page.click('#tab-cpn')
    time.sleep(0.5)
    # click 12 Eylül Canlı Model Takibi
    btn_m12 = page.query_selector('#csub-model12')
    assert btn_m12 is not None, "csub-model12 must exist"
    btn_m12.click()
    time.sleep(0.5)
    
    # Check summary tiles
    t_05 = page.query_selector('#cpnSummary .tile[data-f="line05"]')
    assert t_05 is not None, "line05 tile must exist"
    print("Found line05 tile. Clicking...")
    t_05.click()
    time.sleep(0.5)
    
    # Check active class on tile
    t_05_reloaded = page.query_selector('#cpnSummary .tile[data-f="line05"]')
    t_05_classes = t_05_reloaded.get_attribute('class') or ''
    print("line05 tile classes after click:", t_05_classes)
    assert 'active' in t_05_classes, "line05 tile should have active class"
    
    # Check table content in cpnList
    cpn_list_html = page.inner_html('#cpnList')
    print("cpnList has table:", '<table class="cpn-tbl">' in cpn_list_html)
    assert '<table class="cpn-tbl">' in cpn_list_html, "cpnList should show match table"

    # Click 'all' tile
    t_all = page.query_selector('#cpnSummary .tile[data-f="all"]')
    assert t_all is not None, "all tile must exist"
    t_all.click()
    time.sleep(0.5)
    t_all_reloaded = page.query_selector('#cpnSummary .tile[data-f="all"]')
    t_all_classes = t_all_reloaded.get_attribute('class') or ''
    print("all tile classes after click:", t_all_classes)
    assert 'active' in t_all_classes, "all tile should have active class"

    print('\n=== TEST 5: Model Doğruluğu (tab-bt) ===')
    page.click('#tab-bt')
    time.sleep(0.5)
    pane_bt = page.query_selector('#pane-bt')
    assert pane_bt is not None and pane_bt.is_visible(), "pane-bt must be visible"
    tbl_league = page.query_selector('#bt-byleague')
    tbl_season = page.query_selector('#bt-byseason')
    tbl_samples = page.query_selector('#bt-samples')
    assert tbl_league and len(tbl_league.inner_text().strip()) > 0, "bt-byleague must have content"
    assert tbl_season and len(tbl_season.inner_text().strip()) > 0, "bt-byseason must have content"
    assert tbl_samples and len(tbl_samples.inner_text().strip()) > 0, "bt-samples must have content"
    print("pane-bt loaded successfully with league, season, and sample match tables!")

    print('\n=== TEST 6: Tahminler (Bülten) tab-pred ===')
    page.click('#tab-pred')
    time.sleep(0.5)
    pane_pred = page.query_selector('#pane-pred')
    assert pane_pred is not None and pane_pred.is_visible(), "pane-pred must be visible"
    rows_el = page.query_selector('#rows')
    row_count = len(page.query_selector_all('#rows .row'))
    print(f"tab-pred match rows count: {row_count}")
    assert row_count > 0, "tab-pred must have match rows and not be empty"

    print('\nALL DIAGNOSTIC AND INTEGRATION TESTS PASSED!')
    browser.close()
