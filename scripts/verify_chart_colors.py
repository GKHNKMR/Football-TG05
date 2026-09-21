import time, threading, sys, os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright

class H(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

PORT = 8931
server = HTTPServer(('127.0.0.1', PORT), H)
threading.Thread(target=server.serve_forever, daemon=True).start()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1280, 'height': 900})
    page.on('dialog', lambda d: d.accept())
    page.add_init_script("localStorage.setItem('betavus.access', '1f7b720c52ea3f6e8631a8eeaffaa7113fbed540ec0772108d39c52835d9855d');")
    page.goto(f'http://127.0.0.1:{PORT}/index.html')
    time.sleep(1.2)
    page.click('#tab-plan')
    time.sleep(0.5)
    if page.query_selector('#btnResetPlan'):
        page.click('#btnResetPlan')
        time.sleep(0.5)
    page.fill('#setupPlanName', 'Minimum Risk Kasa')
    page.fill('#setupStartBank', '50')
    page.fill('#setupTargetBank', '500')
    page.click('.risk-card[data-risk="minimum"]')
    page.click('#btnCreatePlan')
    time.sleep(0.8)
    card = page.query_selector('#planChartCard')
    if card:
        card.screenshot(path='scratch/distinguishable_curves.png')
        print('SUCCESS: Captured scratch/distinguishable_curves.png')
    browser.close()
