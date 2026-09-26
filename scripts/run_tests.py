"""Tüm regresyon testlerini sırayla çalıştırır ve özet verir.

Kullanım (repo kökünden):  python scripts/run_tests.py
Herhangi bir test düşerse çıkış kodu 1 olur; düşen testin son satırları yazdırılır.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent
TESTS = [
    'test_sim_math.py',          # saf hesap (tarayıcısız)
    'test_validate_data.py',     # saatlik bot veri doğrulaması bozuk veriyi yakalıyor mu (tarayıcısız)
    'test_paper_betting.py',     # Sanal Kasa motoru + ekranı, menü, üst menü taşması
    'test_cifte_tab.py',         # Fikstür bülteni, Çifte Şans vurgusu, lig filtresi
    'test_cifte_backtest.py',    # Çifte Şans model doğruluğu (gizli sekme)
    'verify_site.py',            # Model doğruluğu tabloları, 30 günlük simülasyon, tüm sekmeler açılıyor
]

env = dict(os.environ, PYTHONIOENCODING='utf-8')
failed = []
for name in TESTS:
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / name)], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    ok = r.returncode == 0
    print(f"{'✓' if ok else '✗'} {name:<24} {time.time() - t0:5.1f} sn")
    if not ok:
        failed.append(name)
        tail = [l for l in (r.stdout + r.stderr).splitlines() if not l.startswith('CONSOLE:')][-12:]
        print('    ' + '\n    '.join(tail))

print(f"\n{len(TESTS) - len(failed)}/{len(TESTS)} test geçti" + (f" — düşenler: {', '.join(failed)}" if failed else ''))
sys.exit(1 if failed else 0)
