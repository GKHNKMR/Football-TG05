import sys, os, time, threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

PORT = 8899

def run_server():
    server = HTTPServer(('127.0.0.1', PORT), QuietHandler)
    server.serve_forever()

def main():
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(0.5)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Unlock gate
        page.add_init_script("""
            localStorage.setItem('betavus.access', '1f7b720c52ea3f6e8631a8eeaffaa7113fbed540ec0772108d39c52835d9855d');
            localStorage.setItem('betavus.tab', 'res');
            localStorage.setItem('betavus.res_mode', 'vurgu');
            localStorage.setItem('betavus.resseason', '');
        """)

        print(f"Navigating to http://127.0.0.1:{PORT}/index.html...")
        page.goto(f"http://127.0.0.1:{PORT}/index.html", wait_until="networkidle")

        # Wait for data to load
        page.wait_for_selector("#resSummary .tiles", timeout=10000)
        time.sleep(1)

        # 1. Check Default Vurgu Mode
        print("\n--- 1. Testing Default Vurgu Mode ---")
        tiles = page.query_selector_all("#resSummary .tile")
        print(f"Number of tiles: {len(tiles)}")
        tile_texts = [t.inner_text().replace('\n', ' | ') for t in tiles]
        for i, txt in enumerate(tile_texts, 1):
            print(f"  Tile {i}: {txt}")

        sub_matches = page.inner_text("#subtab-matches")
        print(f"  Subtab Matches: {sub_matches}")

        assert "95.4%" in tile_texts[0] and "3.043 / 3.190" in tile_texts[0], f"Tile 1 unexpected: {tile_texts[0]}"
        assert "83.8%" in tile_texts[1] and "1.296 / 1.547" in tile_texts[1], f"Tile 2 unexpected: {tile_texts[1]}"
        assert "73.9%" in tile_texts[2] and "88 / 119" in tile_texts[2], f"Tile 3 unexpected: {tile_texts[2]}"
        assert "3.220" in tile_texts[3], f"Tile 4 unexpected: {tile_texts[3]}"
        assert "16.478" in sub_matches, f"Subtab count unexpected: {sub_matches}"

        # 2. Check Genel Mode Switch
        print("\n--- 2. Testing Genel Mode Switch ---")
        page.click("#modeAll")
        time.sleep(0.5)
        tiles_all = page.query_selector_all("#resSummary .tile")
        tile_all_texts = [t.inner_text().replace('\n', ' | ') for t in tiles_all]
        for i, txt in enumerate(tile_all_texts, 1):
            print(f"  Tile {i}: {txt}")

        assert "93.5%" in tile_all_texts[0] and "15.414 / 16.478" in tile_all_texts[0], f"Tile 1 unexpected: {tile_all_texts[0]}"
        assert "76.6%" in tile_all_texts[1] and "12.614 / 16.478" in tile_all_texts[1], f"Tile 2 unexpected: {tile_all_texts[1]}"
        assert "54.8%" in tile_all_texts[2] and "9.022 / 16.478" in tile_all_texts[2], f"Tile 3 unexpected: {tile_all_texts[2]}"
        assert "16.478" in tile_all_texts[3], f"Tile 4 unexpected: {tile_all_texts[3]}"

        # Switch back to Vurgu
        page.click("#modeVurgu")
        time.sleep(0.3)

        # 3. Check Season Filter: "Tüm Sezonlar + Güncel (16.988)"
        print("\n--- 3. Testing Tüm Sezonlar + Güncel (16.988) ---")
        page.click("button.chip[data-s='all']")
        time.sleep(0.5)
        tiles_all_s = page.query_selector_all("#resSummary .tile")
        tile_all_s_texts = [t.inner_text().replace('\n', ' | ') for t in tiles_all_s]
        for i, txt in enumerate(tile_all_s_texts, 1):
            print(f"  Tile {i}: {txt}")
        sub_matches_all = page.inner_text("#subtab-matches")
        print(f"  Subtab Matches: {sub_matches_all}")

        assert "95.3%" in tile_all_s_texts[0] and "3.184 / 3.340" in tile_all_s_texts[0], f"Tile 1 unexpected: {tile_all_s_texts[0]}"
        assert "84.0%" in tile_all_s_texts[1] and "1.357 / 1.615" in tile_all_s_texts[1], f"Tile 2 unexpected: {tile_all_s_texts[1]}"
        assert "75.0%" in tile_all_s_texts[2] and "96 / 128" in tile_all_s_texts[2], f"Tile 3 unexpected: {tile_all_s_texts[2]}"
        assert "16.988" in sub_matches_all, f"Subtab count unexpected: {sub_matches_all}"

        # 4. Check Vurgu Matches Toggle Filter
        print("\n--- 4. Testing Vurgu Matches Toggle Filter ---")
        # Go back to 5 seasons
        page.click("button.chip[data-s='']")
        time.sleep(0.3)
        vurgu_btn = page.inner_text("#vurguMatchToggle")
        print(f"  vurguMatchToggle button: {vurgu_btn}")
        assert "3.220" in vurgu_btn, f"Vurgu button count mismatch: {vurgu_btn}"

        page.click("#vurguMatchToggle")
        time.sleep(0.3)
        sub_matches_vurgu = page.inner_text("#subtab-matches")
        print(f"  Subtab Matches after toggle: {sub_matches_vurgu}")
        assert "3.220 vurgulanan" in sub_matches_vurgu

        # Toggle off
        page.click("#vurguMatchToggle")
        time.sleep(0.3)

        # 5. Check High Confidence Misses Toggle Filter
        print("\n--- 5. Testing High Confidence Misses Toggle Filter ---")
        miss_btn = page.inner_text("#hiMissToggle")
        print(f"  hiMissToggle button: {miss_btn}")
        assert "352" in miss_btn, f"Miss button count mismatch: {miss_btn}"

        page.click("#hiMissToggle")
        time.sleep(0.3)
        sub_matches_miss = page.inner_text("#subtab-matches")
        print(f"  Subtab Matches after miss toggle: {sub_matches_miss}")
        assert "352 ıska" in sub_matches_miss

        # Toggle off
        page.click("#hiMissToggle")
        time.sleep(0.3)

        # 6. Check View 2 (Lig & Sezon Dağılımı)
        print("\n--- 6. Testing View 2 (Lig & Sezon Dağılımı) ---")
        page.click("#subtab-tables")
        time.sleep(0.5)
        s_table = page.inner_text("#bt-byseason")
        print("  Season table snippet:")
        for line in s_table.split('\n')[:8]:
            print(f"    {line}")
        assert "2026/27 (Güncel)" in s_table
        assert "510" in s_table
        assert "2025/26" in s_table
        assert "2021/22" in s_table

        # 7. Check Kuponlarım Tab Integrity (>= 2026-09-12)
        print("\n--- 7. Testing Kuponlarım Tab Integrity (>= 2026-09-12) ---")
        page.click("#tab-cpn")
        time.sleep(0.5)
        cpn_summary = page.inner_text("#cpnSummary")
        print(f"  Kuponlarım summary: {cpn_summary[:100]}...")
        assert "12.09.2026" in cpn_summary or "Kupon" in cpn_summary

        browser.close()
        print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    main()
