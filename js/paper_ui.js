/**
 * BETAVUS — Paper-Betting, Kupon Planlama ve Sanal Kasa Arayüz Modülü
 * (UI & DOM Controller)
 */

(function (root) {
  'use strict';

  const PE = root.BETAVUS_PAPER;
  if (!PE) {
    console.error('BETAVUS_PAPER motoru bulunamadı!');
    return;
  }

  // ---------------------------------------------------------------------------
  // State & Yerel Değişkenler
  // ---------------------------------------------------------------------------

  let paperState = null;
  let activeCpnSubtab = 'pending'; // 'drafts' | 'pending' | 'settled' | 'model12'
  let settledFilter = 'all'; // 'all' | 'won' | 'lost' | 'minimum' | 'medium' | 'high'
  let editingSlip = null; // Aktif düzenlenen kupon nesnesi
  let modalMatchPickerCallback = null;

  let planSubView = 'active'; // 'active' | 'sim30'
  let sim30ActiveProfile = 'minimum'; // 'minimum' | 'medium' | 'high' | 'all'
  let sim30ActiveSubtab = 'coupons'; // 'coupons' | 'ledger'
  let sim30Data = null;
  let fetchingResultsPromise = null;

  // ---------------------------------------------------------------------------
  // Yardımcı Biçimlendirme Fonksiyonları
  // ---------------------------------------------------------------------------

  function parseNumber(val) {
    if (val == null) return null;
    if (typeof val === 'number') return isNaN(val) ? null : val;
    let s = String(val).trim().replace(/\s+/g, '').replace(',', '.');
    const n = parseFloat(s);
    return isNaN(n) || !isFinite(n) ? null : n;
  }

  function formatCurrency(amount, currencyCode = 'EUR') {
    const curr = PE.CURRENCIES[currencyCode] || PE.CURRENCIES.EUR;
    const num = Number(amount) || 0;
    const I = root.I18N || { lang: 'tr', locale: 'tr-TR' };
    const formatted = num.toLocaleString(I.locale, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
    if (I.lang === 'en') return num < 0 ? `-${curr.symbol}${formatted.slice(1)}` : `${curr.symbol}${formatted}`;
    if (I.lang === 'nl') return `${curr.symbol} ${formatted}`;
    return `${formatted} ${curr.symbol}`;
  }

  function formatPct(val, dec = 1) {
    const num = Number(val);
    if (isNaN(num)) return '—';
    const I = root.I18N || { lang: 'tr', locale: 'tr-TR' };
    const s = num.toLocaleString(I.locale, { minimumFractionDigits: dec, maximumFractionDigits: dec });
    return I.lang === 'tr' ? '%' + s : s + '%';
  }

  function formatOdds(val) {
    const num = Number(val);
    if (isNaN(num) || num <= 0) return '—';
    return num.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function esc(s) {
    return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function flag(l, size = 12) {
    if (typeof root.flag === 'function') return root.flag(l, size);
    return '⚽';
  }

  function dmy(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return String(iso).slice(0, 10);
    return `${String(d.getDate()).padStart(2, '0')}.${String(d.getMonth() + 1).padStart(2, '0')}.${d.getFullYear()}`;
  }

  function timeStr(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  }

  // ---------------------------------------------------------------------------
  // Kalıcılık (Persistence)
  // ---------------------------------------------------------------------------

  function loadState() {
    try {
      const raw = localStorage.getItem(PE.STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (parsed && parsed.schemaVersion === PE.SCHEMA_VERSION) {
        return parsed;
      }
    } catch (e) {
      console.warn('Paper-betting state okunamadı:', e);
    }
    return null;
  }

  function saveState() {
    if (!paperState) return;
    try {
      localStorage.setItem(PE.STORAGE_KEY, JSON.stringify(paperState));
    } catch (e) {
      console.error('Paper-betting state kaydedilemedi:', e);
    }
  }

  function initState() {
    paperState = loadState();
  }

  // ---------------------------------------------------------------------------
  // 30 Günlük Geçmiş Kasa & Kuponlarım Simülasyonu (50 € Örnek Model)
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // 30 Günlük Geçmiş Kasa & Kuponlarım Simülasyonu (50 € Örnek Model)
  // ---------------------------------------------------------------------------

  function getOrGenerateSim30Data(callback) {
    if (sim30Data) {
      callback(sim30Data);
      return;
    }

    let matchesList = null;
    if (typeof RESULTS !== 'undefined' && RESULTS && Array.isArray(RESULTS.matches) && RESULTS.matches.length > 0) {
      matchesList = RESULTS.matches;
    } else if (typeof window !== 'undefined' && window.RESULTS && Array.isArray(window.RESULTS.matches) && window.RESULTS.matches.length > 0) {
      matchesList = window.RESULTS.matches;
    }

    if (matchesList) {
      sim30Data = PE.generate30DayHistoricalSimulation(matchesList, { startingBank: 50.0 });
      callback(sim30Data);
      return;
    }

    if (!fetchingResultsPromise) {
      fetchingResultsPromise = fetch('data/results.json?ts=' + Date.now(), { cache: 'no-store' })
        .then(r => r.ok ? r.json() : null)
        .then(data => {
          const list = (data && Array.isArray(data.matches)) ? data.matches : [];
          sim30Data = PE.generate30DayHistoricalSimulation(list, { startingBank: 50.0 });
          return sim30Data;
        })
        .catch(err => {
          console.error('Failed to load results.json for 30-day simulation:', err);
          sim30Data = PE.generate30DayHistoricalSimulation([], { startingBank: 50.0 });
          return sim30Data;
        });
    }

    fetchingResultsPromise.then(res => {
      callback(res);
    });
  }

  // ---- 1. Örnek Kasa Simülasyonu Ekranı (#pane-sim-kasa) ----
  function renderSimKasaPane() {
    const pane = document.getElementById('pane-sim-kasa');
    if (!pane) return;

    pane.innerHTML = `
      <div class="sim30-loading" style="padding:48px 24px;text-align:center;color:var(--muted);">
        <div style="font-size:24px;margin-bottom:8px;">⏳</div>
        <div>30 Günlük Geçmiş Kasa Simülasyonu Hazırlanıyor...</div>
      </div>
    `;

    getOrGenerateSim30Data((data) => {
      pane.innerHTML = renderSimKasaHtml(data);
      wireSimKasaEvents(data);
    });
  }

  function renderSimKasaHtml(simData) {
    if (!simData || !simData.profiles) {
      return `
        <div class="card" style="padding:24px;text-align:center;color:var(--muted);">
          Simülasyon verisi yüklenemedi. Lütfen sayfayı yenileyip tekrar deneyin.
        </div>
      `;
    }

    const minProf = simData.profiles.minimum;
    const medProf = simData.profiles.medium;
    const highProf = simData.profiles.high;
    const win = simData.simulationWindow;
    const activeProfData = simData.profiles[sim30ActiveProfile] || minProf;

    return `
      <div class="paper-disclaimer">
        <span class="p-badge">30 GÜNLÜK GEÇMİŞ KASA SİMÜLASYONU</span>
        <p><b>BETAVUS</b> gerçek bahis sitesi değildir; para kabul etmez veya kupon oynatmaz. Aşağıdaki model, <b>${dmy(win.startDate)} – ${dmy(win.endDate)}</b> tarihleri arasında oynanmış <b>${win.matchesInWindow} adet gerçek resmi maç</b> ve gerçek skorlar ile Avrupa bahis piyasası oranları üzerinden çalıştırılmış 30 günlük disiplinli kasa simülasyonudur. Modelimiz <b>0.5 Üstü için %95 ve yukarısı</b>, <b>1.5 Üstü için %85 ve üstü</b>, <b>2.5 Üstü için %75 ve üstü</b> başarı kriterleriyle çalışır. Yüksek risk demek garanti olmayan maçları oynamak demek değildir; hedeflenen ~1.35x oranına %95 ve %85 üzeri yüksek güvenli maçların disiplinli birleşimiyle ulaşılır.</p>
        <div class="p-quote">« 50 € Başlangıç Sermayesi · 3 Risk Modeli · Gerçek Maçlar · Disiplinli Kasa Rezervi »</div>
      </div>

      <div class="sim30-hero-card">
        <div class="sim30-hero-title">
          <div>
            <h3>📈 30 Günlük Örnek Kasa Büyümesi Simülasyonu (50 € Başlangıç Kasası)</h3>
            <div class="sim30-desc">
              Başlangıç Kasası: <b>50,00 €</b> · Dönem: <b>${dmy(win.startDate)} – ${dmy(win.endDate)} (${win.totalDays} Gün)</b> · Maç Havuzu: <b>${win.matchesInWindow} Resmi Maç</b>
            </div>
          </div>
          <div class="sim30-badges">
            <span class="sim30-badge">🇪🇺 Avrupa Gerçek Piyasa Oranları</span>
            <span class="sim30-badge">🛡️ Korumalı Kasa Rezervi</span>
          </div>
        </div>
      </div>

      <!-- Risk Profili Seçim & KPI Grid Kartları -->
      <div class="sim30-kpi-grid">
        <!-- Minimum Risk Kartı -->
        <div class="sim30-kpi-card minimum ${sim30ActiveProfile === 'minimum' ? 'active-card' : ''}" data-prof="minimum">
          <div class="sim30-kpi-head">
            <b style="color:#10b981;font-size:13px;">🟢 Minimum Risk (%50 Rezerv)</b>
            <span class="r-badge b-min">5x 0.5 Üst · ~1.26x</span>
          </div>
          <div class="sim30-kpi-rows">
            <div class="sim30-kpi-row"><span>Başlangıç Kasası:</span><b>50,00 €</b></div>
            <div class="sim30-kpi-row"><span>30. Gün Kasa:</span><b style="color:#10b981;font-size:13px;">${formatCurrency(minProf.stats.finalBank, 'EUR')}</b></div>
            <div class="sim30-kpi-row"><span>Toplam Kâr / ROI:</span><b class="${minProf.stats.totalNetProfit >= 0 ? 'good' : 'bad'}">${minProf.stats.totalNetProfit >= 0 ? '+' : ''}${formatCurrency(minProf.stats.totalNetProfit, 'EUR')} (${minProf.stats.totalRoiPct >= 0 ? '+' : ''}%${minProf.stats.totalRoiPct})</b></div>
            <div class="sim30-kpi-row"><span>Kupon Başarısı:</span><b>${minProf.stats.wonCoupons} / ${minProf.stats.totalCoupons} (%${minProf.stats.winRatePct})</b></div>
            <div class="sim30-kpi-row"><span>Maç Yanılma Oranı:</span><b style="color:#10b981;">%${minProf.stats.legErrorRatePct} (İsabet: %${minProf.stats.legSuccessRatePct})</b></div>
            <div class="sim30-kpi-row"><span>Rezerv / Aktif:</span><span>${formatCurrency(minProf.stats.reserveBank, 'EUR')} / ${formatCurrency(minProf.stats.activeBank, 'EUR')}</span></div>
            ${minProf.maxPotential ? `
            <div class="sim30-kpi-row" style="margin-top:6px;padding-top:6px;border-top:1px dashed rgba(255,255,255,0.12);">
              <span style="color:#fbbf24;font-weight:700;">★ Sıfır Kayıp Potansiyeli:</span>
              <b style="color:#fbbf24;font-size:12.5px;">${formatCurrency(minProf.maxPotential.finalBank, 'EUR')} (+%${minProf.maxPotential.roiPct})</b>
            </div>` : ''}
          </div>
        </div>

        <!-- Orta Risk Kartı -->
        <div class="sim30-kpi-card medium ${sim30ActiveProfile === 'medium' ? 'active-card' : ''}" data-prof="medium">
          <div class="sim30-kpi-head">
            <b style="color:#38bdf8;font-size:13px;">🔵 Orta Risk (%35 Rezerv)</b>
            <span class="r-badge b-med">3x 1.5 Üst (≥ %85 Güven) · ~1.42x</span>
          </div>
          <div class="sim30-kpi-rows">
            <div class="sim30-kpi-row"><span>Başlangıç Kasası:</span><b>50,00 €</b></div>
            <div class="sim30-kpi-row"><span>30. Gün Kasa:</span><b style="color:#38bdf8;font-size:13px;">${formatCurrency(medProf.stats.finalBank, 'EUR')}</b></div>
            <div class="sim30-kpi-row"><span>Toplam Kâr / ROI:</span><b class="${medProf.stats.totalNetProfit >= 0 ? 'good' : 'bad'}">${medProf.stats.totalNetProfit >= 0 ? '+' : ''}${formatCurrency(medProf.stats.totalNetProfit, 'EUR')} (${medProf.stats.totalRoiPct >= 0 ? '+' : ''}%${medProf.stats.totalRoiPct})</b></div>
            <div class="sim30-kpi-row"><span>Kupon Başarısı:</span><b>${medProf.stats.wonCoupons} / ${medProf.stats.totalCoupons} (%${medProf.stats.winRatePct})</b></div>
            <div class="sim30-kpi-row"><span>Maç Yanılma Oranı:</span><b>%${medProf.stats.legErrorRatePct} (İsabet: %${medProf.stats.legSuccessRatePct})</b></div>
            <div class="sim30-kpi-row"><span>Rezerv / Aktif:</span><span>${formatCurrency(medProf.stats.reserveBank, 'EUR')} / ${formatCurrency(medProf.stats.activeBank, 'EUR')}</span></div>
            ${medProf.maxPotential ? `
            <div class="sim30-kpi-row" style="margin-top:6px;padding-top:6px;border-top:1px dashed rgba(255,255,255,0.12);">
              <span style="color:#fbbf24;font-weight:700;">★ Sıfır Kayıp Potansiyeli:</span>
              <b style="color:#fbbf24;font-size:12.5px;">${formatCurrency(medProf.maxPotential.finalBank, 'EUR')} (+%${medProf.maxPotential.roiPct})</b>
            </div>` : ''}
          </div>
        </div>

        <!-- Yüksek Risk Kartı -->
        <div class="sim30-kpi-card high ${sim30ActiveProfile === 'high' ? 'active-card' : ''}" data-prof="high">
          <div class="sim30-kpi-head">
            <b style="color:#ef4444;font-size:13px;">🔴 Yüksek Risk (%25 Rezerv)</b>
            <span class="r-badge b-high">Yüksek Güven Kombinasyon · ~1.35x</span>
          </div>
          <div class="sim30-kpi-rows">
            <div class="sim30-kpi-row"><span>Başlangıç Kasası:</span><b>50,00 €</b></div>
            <div class="sim30-kpi-row"><span>30. Gün Kasa:</span><b style="color:#ef4444;font-size:13px;">${formatCurrency(highProf.stats.finalBank, 'EUR')}</b></div>
            <div class="sim30-kpi-row"><span>Toplam Kâr / ROI:</span><b class="${highProf.stats.totalNetProfit >= 0 ? 'good' : 'bad'}">${highProf.stats.totalNetProfit >= 0 ? '+' : ''}${formatCurrency(highProf.stats.totalNetProfit, 'EUR')} (${highProf.stats.totalRoiPct >= 0 ? '+' : ''}%${highProf.stats.totalRoiPct})</b></div>
            <div class="sim30-kpi-row"><span>Kupon Başarısı:</span><b>${highProf.stats.wonCoupons} / ${highProf.stats.totalCoupons} (%${highProf.stats.winRatePct})</b></div>
            <div class="sim30-kpi-row"><span>Maç Yanılma Oranı:</span><b>%${highProf.stats.legErrorRatePct} (İsabet: %${highProf.stats.legSuccessRatePct})</b></div>
            <div class="sim30-kpi-row"><span>Rezerv / Aktif:</span><span>${formatCurrency(highProf.stats.reserveBank, 'EUR')} / ${formatCurrency(highProf.stats.activeBank, 'EUR')}</span></div>
            ${highProf.maxPotential ? `
            <div class="sim30-kpi-row" style="margin-top:6px;padding-top:6px;border-top:1px dashed rgba(255,255,255,0.12);">
              <span style="color:#fbbf24;font-weight:700;">★ Sıfır Kayıp Potansiyeli:</span>
              <b style="color:#fbbf24;font-size:12.5px;">${formatCurrency(highProf.maxPotential.finalBank, 'EUR')} (+%${highProf.maxPotential.roiPct})</b>
            </div>` : ''}
          </div>
        </div>
      </div>

      <!-- 30 Günlük Kasa Gelişim Grafiği (SVG) -->
      <div class="card" style="margin-bottom:16px;">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:gap:10px;margin-bottom:12px;">
          <div>
            <h4 style="margin:0;font-size:14px;color:var(--text);">📊 30 Günlük Kasa Büyüme Patikası &amp; Gerçekleşen Eğri</h4>
            <div style="font-size:11.5px;color:var(--muted);margin-top:2px;">
              50 € başlangıç sermayesiyle 31 günlük gerçek kupon neticelerine göre oluşan kasanın günlük hareketi.
            </div>
          </div>
          <div class="chart-mode-pills" id="sim30ChartPills">
            <button type="button" class="cmp-btn ${sim30ActiveProfile === 'minimum' ? 'active' : ''}" data-prof="minimum">🟢 Minimum</button>
            <button type="button" class="cmp-btn ${sim30ActiveProfile === 'medium' ? 'active' : ''}" data-prof="medium">🔵 Orta</button>
            <button type="button" class="cmp-btn ${sim30ActiveProfile === 'high' ? 'active' : ''}" data-prof="high">🔴 Yüksek</button>
            <button type="button" class="cmp-btn ${sim30ActiveProfile === 'all' ? 'active' : ''}" data-prof="all">🌈 Hepsini Karşılaştır</button>
          </div>
        </div>
        <div id="sim30ChartContainer" class="chart-svg-box">
          ${render30DaySimulationSvg(simData, sim30ActiveProfile)}
        </div>
      </div>

      <!-- Doğrudan Eylem Kartları (CTA) -->
      <div class="card" style="padding:16px 20px;margin-bottom:16px;background:var(--panel2);display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
        <div>
          <b style="font-size:13.5px;color:var(--text);">Sonraki Adım: Bu simülasyondaki tüm kuponları ve maç tahminlerini inceleyin</b>
          <div style="font-size:11.5px;color:var(--muted);margin-top:3px;">
            Hangi maçlar oynandı? Modelin λ (beklenen gol) tahmini neydi, sahada kaç gol bitti?
          </div>
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
          <button type="button" class="btn-primary" id="btnGoToSimKupon" style="padding:8px 16px;font-size:12px;font-weight:700;">
            🎫 Örnek Kuponlarım ve Model Analizine Git →
          </button>
          <button type="button" class="btn-sec" id="btnGoToPlan" style="padding:8px 16px;font-size:12px;font-weight:700;">
            💼 Kendi Sanal Kasamı Oluştur →
          </button>
        </div>
      </div>

      <!-- 31 Günlük Kasa Muhasebe Çizelgesi -->
      <div id="sim30LedgerContainer">
        ${render30DayLedgerTableHtml(simData, sim30ActiveProfile)}
      </div>
    `;
  }

  // ---- 2. Örnek Kuponlarım Simülasyonu Ekranı (#pane-sim-kupon) ----
  function renderSimKuponPane() {
    const pane = document.getElementById('pane-sim-kupon');
    if (!pane) return;

    pane.innerHTML = `
      <div class="sim30-loading" style="padding:48px 24px;text-align:center;color:var(--muted);">
        <div style="font-size:24px;margin-bottom:8px;">⏳</div>
        <div>Örnek Kuponlarım ve Model Analizi Hazırlanıyor...</div>
      </div>
    `;

    getOrGenerateSim30Data((data) => {
      pane.innerHTML = renderSimKuponHtml(data);
      wireSimKuponEvents(data);
    });
  }

  function renderSimKuponHtml(simData) {
    if (!simData || !simData.profiles) {
      return `
        <div class="card" style="padding:24px;text-align:center;color:var(--muted);">
          Simülasyon kupon verisi yüklenemedi. Lütfen sayfayı yenileyip tekrar deneyin.
        </div>
      `;
    }

    const profKey = (sim30ActiveProfile === 'all' || !simData.profiles[sim30ActiveProfile]) ? 'minimum' : sim30ActiveProfile;
    const prof = simData.profiles[profKey];
    if (!prof) return '';

    return `
      <!-- Model İsabet ve Yanılma Analiz Özeti Kartı -->
      <div class="sim-analysis-summary-card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:14px;border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:12px;">
          <div>
            <div style="display:flex;align-items:center;gap:8px;">
              <h3 style="margin:0;font-size:16px;font-weight:900;color:var(--text);">🎯 30 Günlük Kupon &amp; Model Başarı Analizi</h3>
              <span style="background:rgba(16,185,129,0.12);color:#10b981;border:1px solid rgba(16,185,129,0.3);font-size:10.5px;font-weight:800;padding:2px 8px;border-radius:99px;">Tahmin vs Gerçekleşen</span>
            </div>
            <div style="font-size:11.5px;color:var(--muted);margin-top:4px;">
              Modelin hesapladığı beklenen gol (λ) ve olasılıklar ile 31 günde sonuçlanan gerçek maçların performans dökümü:
            </div>
          </div>
          <div class="chart-mode-pills" id="simKuponProfilePills" style="padding:5px;gap:8px;">
            <button type="button" class="cmp-btn ${profKey === 'minimum' ? 'active' : ''}" data-prof="minimum">
              <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#10b981;box-shadow:0 0 8px #10b981;"></span>
              <span>Minimum Risk</span>
              <span style="font-size:10px;opacity:.75;font-weight:600;">(5x 0.5Ü)</span>
            </button>
            <button type="button" class="cmp-btn ${profKey === 'medium' ? 'active' : ''}" data-prof="medium">
              <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#38bdf8;box-shadow:0 0 8px #38bdf8;"></span>
              <span>Orta Risk</span>
              <span style="font-size:10px;opacity:.75;font-weight:600;">(3x 1.5Ü)</span>
            </button>
            <button type="button" class="cmp-btn ${profKey === 'high' ? 'active' : ''}" data-prof="high">
              <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ef4444;box-shadow:0 0 8px #ef4444;"></span>
              <span>Yüksek Risk</span>
              <span style="font-size:10px;opacity:.75;font-weight:600;">(~1.35x Kombo)</span>
            </button>
          </div>
        </div>

        <div class="sim-analysis-grid">
          <div class="sim-analysis-box" style="border-top:3px solid #10b981;">
            <div class="sav" style="color:#10b981;">%${prof.stats.legSuccessRatePct || 95.5}</div>
            <div class="sal">Maç Başına Başarı</div>
            <div style="font-size:11px;color:#94a3b8;margin-top:3px;font-weight:600;">${prof.stats.wonLegs} / ${prof.stats.totalLegs} Maç Geldi</div>
          </div>
          <div class="sim-analysis-box" style="border-top:3px solid #f59e0b;">
            <div class="sav" style="color:#f59e0b;">%${prof.stats.legErrorRatePct || 4.5}</div>
            <div class="sal">Maç Başına Yanılma</div>
            <div style="font-size:11px;color:#94a3b8;margin-top:3px;font-weight:600;">Yalnızca ${prof.stats.lostLegs} Maçta Iska</div>
          </div>
          <div class="sim-analysis-box" style="border-top:3px solid #38bdf8;">
            <div class="sav" style="color:#38bdf8;">${prof.stats.wonCoupons} / ${prof.stats.totalCoupons}</div>
            <div class="sal">Kupon Tutma Oranı</div>
            <div style="font-size:11px;color:#94a3b8;margin-top:3px;font-weight:600;">%${prof.stats.winRatePct} Başarılı Gün</div>
          </div>
          <div class="sim-analysis-box" style="border-top:3px solid #34d399;">
            <div class="sav" style="color:#34d399;">+${formatCurrency(prof.stats.totalNetProfit, 'EUR')}</div>
            <div class="sal">Net Kasa Kârı</div>
            <div style="font-size:11px;color:#94a3b8;margin-top:3px;font-weight:600;">ROI: +%${prof.stats.totalRoiPct}</div>
          </div>
          <div class="sim-analysis-box" style="border-top:3px solid #a855f7;">
            <div class="sav" style="color:#c084fc;">${formatCurrency(prof.stats.reserveBank, 'EUR')}</div>
            <div class="sal">Kasa Rezervi</div>
            <div style="font-size:11px;color:#94a3b8;margin-top:3px;font-weight:600;">%${prof.reservePct} Korumada Kaldı</div>
          </div>
        </div>

        <!-- Hiç Maç Kaybetmeme Durumu (Sıfır Kayıp / %100 İsabet İterasyonu) -->
        ${prof.maxPotential ? `
          <div class="sim-noloss-banner" style="margin-top:14px;background:linear-gradient(135deg, rgba(245,158,11,0.08) 0%, rgba(16,185,129,0.08) 100%);border:1px solid rgba(245,158,11,0.25);border-radius:12px;padding:14px 16px;">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px;">
              <div style="display:flex;align-items:center;gap:8px;">
                <span style="font-size:16px;">⭐</span>
                <b style="color:#fbbf24;font-size:13px;">Hiç Maç Kaybetmeme Durumu (Sıfır Kayıp / %100 İsabet İterasyonu)</b>
              </div>
              <span style="background:rgba(245,158,11,0.2);color:#fbbf24;border:1px solid rgba(245,158,11,0.4);font-size:10.5px;font-weight:800;padding:2px 8px;border-radius:99px;">
                ${prof.maxPotential.wonCoupons}/${prof.maxPotential.wonCoupons} Kupon Kazandı · %100 Başarı
              </span>
            </div>
            <div style="font-size:11.5px;color:var(--muted);line-height:1.5;margin-bottom:12px;">
              31 günlük kupon serisinde <b>hiçbir maç veya kuponun kaybedilmemesi</b> (her gün kuponun tutması ve kasanın dokunulmaz rezervi ayrı tutularak aktif payın bileşik katlanması) halinde modelin ulaştığı teorik maksimum:
            </div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(130px, 1fr));gap:10px;">
              <div style="background:rgba(11,17,32,0.65);border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:10.5px;color:var(--muted);font-weight:600;">Başlangıç Kasa</div>
                <div style="font-size:15px;font-weight:800;color:var(--text);margin-top:2px;">${formatCurrency(prof.maxPotential.startingBank, 'EUR')}</div>
              </div>
              <div style="background:rgba(11,17,32,0.65);border:1px solid rgba(245,158,11,0.35);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:10.5px;color:#fbbf24;font-weight:700;">30 Gün Sonu Kasa</div>
                <div style="font-size:16px;font-weight:900;color:#fbbf24;margin-top:2px;">${formatCurrency(prof.maxPotential.finalBank, 'EUR')}</div>
              </div>
              <div style="background:rgba(11,17,32,0.65);border:1px solid rgba(16,185,129,0.35);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:10.5px;color:#10b981;font-weight:700;">Maksimum Net Kâr</div>
                <div style="font-size:16px;font-weight:900;color:#10b981;margin-top:2px;">+${formatCurrency(prof.maxPotential.totalNetProfit, 'EUR')}</div>
              </div>
              <div style="background:rgba(11,17,32,0.65);border:1px solid rgba(56,189,248,0.35);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:10.5px;color:#38bdf8;font-weight:700;">Maksimum ROI</div>
                <div style="font-size:16px;font-weight:900;color:#38bdf8;margin-top:2px;">+%${prof.maxPotential.roiPct}</div>
              </div>
              <div style="background:rgba(11,17,32,0.65);border:1px solid rgba(168,85,247,0.35);border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:10.5px;color:#c084fc;font-weight:700;">Kasa Katlama</div>
                <div style="font-size:16px;font-weight:900;color:#c084fc;margin-top:2px;">${(prof.maxPotential.finalBank / prof.maxPotential.startingBank).toFixed(1)}x</div>
              </div>
            </div>
          </div>
        ` : ''}
      </div>

      <!-- Kuponlar Listesi -->
      <div style="margin-bottom:12px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <h4 style="margin:0;font-size:14px;color:var(--text);">🎫 ${esc(prof.name)} · 31 Günlük Oynanan Tüm Kuponlar ve Maç Dökümü</h4>
        <button type="button" class="btn-sec" id="btnKuponGoToPlan" style="padding:6px 14px;font-size:11.5px;font-weight:700;">
          💼 Kendi Sanal Kasamı Başlat →
        </button>
      </div>

      ${render30DayCouponsListHtml(simData, profKey)}
    `;
  }

  function render30DayCouponsListHtml(simData, activeProfile) {
    const profKey = (activeProfile === 'all' || !simData.profiles[activeProfile]) ? 'minimum' : activeProfile;
    const prof = simData.profiles[profKey];
    if (!prof || !prof.coupons) return '';

    return `
      <div class="sim30-coupon-list">
        ${prof.coupons.map(cpn => `
          <div class="sim30-cpn-card">
            <div class="sim30-cpn-head">
              <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                <span style="font-weight:800;font-size:13px;color:var(--text);">Gün ${cpn.day} · ${dmy(cpn.date)}</span>
                <span class="r-badge ${profKey === 'minimum' ? 'b-min' : profKey === 'medium' ? 'b-med' : 'b-high'}">${esc(prof.name)}</span>
                <span style="font-size:11px;color:var(--muted);">Hedef Pazar: <b>${esc(cpn.targetMarket)}</b> (${prof.legCount} Maç)</span>
              </div>
              <div>
                <span class="c-badge ${cpn.status === 'won' ? 'won' : 'lost'}" style="font-size:12px;padding:3px 10px;font-weight:700;">
                  ${cpn.status === 'won' ? '✅ KAZANDI' : '❌ KAYBETTİ'}
                </span>
              </div>
            </div>

            <div class="sim30-cpn-stats">
              <div>Güne Başlangıç: <b>${formatCurrency(cpn.startBank, 'EUR')}</b></div>
              <div>Rezerv Kasa: <b>${formatCurrency(cpn.reserveBank, 'EUR')}</b></div>
              <div>Kullanılabilir Aktif: <b>${formatCurrency(cpn.activeBank, 'EUR')}</b></div>
              <div>Kupon Stake: <b>${formatCurrency(cpn.stake, 'EUR')}</b></div>
              <div>Kupon Oranı: <b style="color:#38bdf8;">${cpn.totalOdds.toFixed(2)}x</b></div>
              <div>Net Getiri: <b class="${cpn.netProfit >= 0 ? 'good' : 'bad'}">${cpn.netProfit >= 0 ? '+' : ''}${formatCurrency(cpn.netProfit, 'EUR')}</b></div>
              <div>Gün Sonu Kasa: <b style="color:var(--accent);font-size:12px;">${formatCurrency(cpn.endBank, 'EUR')}</b></div>
              <div>Günlük Değişim: <b class="${cpn.dailyChangePct >= 0 ? 'good' : 'bad'}">${cpn.dailyChangePct >= 0 ? '+' : ''}%${cpn.dailyChangePct}</b></div>
            </div>

            <div class="tbl-scroll" style="overflow-x:auto;">
              <table class="sim30-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Lig</th>
                    <th>Karşılaşma</th>
                    <th>Başlama</th>
                    <th>Pazar</th>
                    <th>Model Tahmini (λ &amp; %)</th>
                    <th>Piyasa Oranı</th>
                    <th>Biten Skor / Gol</th>
                    <th>Model Sonucu</th>
                  </tr>
                </thead>
                <tbody>
                  ${cpn.legs.map((leg, lIdx) => {
                    const isLegWon = leg.isWon != null ? leg.isWon : leg.hit;
                    const legScore = leg.score || '—';
                    const legTot = leg.totalGoals != null ? `${leg.totalGoals} Gol` : '';
                    const lamStr = leg.pred_lambda != null ? `λ ${Number(leg.pred_lambda).toFixed(2)}` : 'λ —';
                    const probStr = leg.probability != null ? `%${Math.round(leg.probability * 100)}` : '';
                    return `
                    <tr>
                      <td>${lIdx + 1}</td>
                      <td>${flag(leg.league)} ${esc(leg.league)}</td>
                      <td><b>${esc(leg.home)}</b> vs <b>${esc(leg.away)}</b></td>
                      <td style="color:var(--muted);">${timeStr(leg.kickoff_utc)}</td>
                      <td><span class="line-badge">${esc(leg.marketLabel || leg.market)}</span></td>
                      <td><b>${lamStr}</b> <span style="color:var(--muted);font-size:10px;">(${probStr})</span></td>
                      <td><b style="color:#38bdf8;">${leg.odds ? leg.odds.toFixed(2) : '—'}</b></td>
                      <td><b>${esc(legScore)}</b> <span style="color:var(--muted);font-size:10px;">${esc(legTot)}</span></td>
                      <td>
                        <span class="${isLegWon ? 'good' : 'bad'}" style="font-weight:700;">
                          ${isLegWon ? `✅ Geldi (${esc(legScore)})` : `❌ Gelmedi (${esc(legScore)})`}
                        </span>
                      </td>
                    </tr>
                    `;
                  }).join('')}
                </tbody>
              </table>
            </div>
          </div>
        `).join('')}
      </div>
    `;
  }

  function render30DayLedgerTableHtml(simData, activeProfile) {
    const profKey = (activeProfile === 'all' || !simData.profiles[activeProfile]) ? 'minimum' : activeProfile;
    const prof = simData.profiles[profKey];
    if (!prof) return '';

    return `
      <div class="card" style="padding:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px;">
          <div>
            <h4 style="margin:0;font-size:14px;">📋 ${esc(prof.name)} · 31 Günlük Kasa Muhasebe Çizelgesi</h4>
            <div style="font-size:11.5px;color:var(--muted);margin-top:2px;">
              Rezerv Oranı: <b>%${prof.reservePct}</b> · Kupon Stake Oranı: <b>%${prof.stakePct}</b> · Pazar: <b>${esc(prof.market)} (${prof.legCount} Maç)</b>
            </div>
          </div>
          <span class="r-badge ${profKey === 'minimum' ? 'b-min' : profKey === 'medium' ? 'b-med' : 'b-high'}">
            Kapanış Kasası: ${formatCurrency(prof.stats.finalBank, 'EUR')} (${prof.stats.totalRoiPct >= 0 ? '+' : ''}%${prof.stats.totalRoiPct})
          </span>
        </div>

        <div class="tbl-scroll" style="max-height:480px;overflow-y:auto;border:1px solid var(--line);border-radius:8px;">
          <table class="sim30-table" style="font-size:11.5px;">
            <thead>
              <tr>
                <th>Gün</th>
                <th>Tarih</th>
                <th>Başlangıç Kasa</th>
                <th>Rezerv Kasa</th>
                <th>Aktif Kasa</th>
                <th>Stake</th>
                <th>Kupon Oranı</th>
                <th>Kupon Durumu</th>
                <th>Net Kâr/Zarar</th>
                <th>Gün Sonu Kasa</th>
                <th>Günlük Değişim</th>
                <th>Teorik Hedef Kasa</th>
              </tr>
            </thead>
            <tbody>
              ${prof.ledger.map(row => `
                <tr style="${row.status === 'won' ? 'background:rgba(16,185,129,0.03);' : 'background:rgba(239,68,68,0.03);'}">
                  <td><b>${row.day}</b></td>
                  <td>${dmy(row.date)}</td>
                  <td>${formatCurrency(row.startBank, 'EUR')}</td>
                  <td><span style="color:var(--muted);">${formatCurrency(row.reserveBank, 'EUR')}</span></td>
                  <td>${formatCurrency(row.activeBank, 'EUR')}</td>
                  <td><b>${formatCurrency(row.stake, 'EUR')}</b></td>
                  <td><b style="color:#38bdf8;">${row.odds.toFixed(2)}x</b></td>
                  <td>
                    <span class="c-badge ${row.status === 'won' ? 'won' : 'lost'}" style="font-size:10px;padding:2px 6px;">
                      ${row.status === 'won' ? '✅ Kazandı' : '❌ Kaybetti'}
                    </span>
                  </td>
                  <td>
                    <b class="${row.netProfit >= 0 ? 'good' : 'bad'}">
                      ${row.netProfit >= 0 ? '+' : ''}${formatCurrency(row.netProfit, 'EUR')}
                    </b>
                  </td>
                  <td>
                    <b style="color:var(--text);">${formatCurrency(row.endBank, 'EUR')}</b>
                  </td>
                  <td>
                    <span class="${row.dailyChangePct >= 0 ? 'good' : 'bad'}" style="font-weight:700;">
                      ${row.dailyChangePct >= 0 ? '+' : ''}%${row.dailyChangePct}
                    </span>
                  </td>
                  <td>
                    <span style="color:#f59e0b;font-weight:600;">${formatCurrency(row.theoreticalTarget, 'EUR')}</span>
                  </td>
                </tr>
              `).join('')}
            </tbody>
            <tfoot>
              <tr style="background:var(--panel2);font-weight:800;border-top:2px solid var(--line);">
                <td colspan="2">TOPLAM ÖZET</td>
                <td>${formatCurrency(prof.stats.startingBank, 'EUR')}</td>
                <td>${formatCurrency(prof.stats.reserveBank, 'EUR')}</td>
                <td>${formatCurrency(prof.stats.activeBank, 'EUR')}</td>
                <td>—</td>
                <td>—</td>
                <td>${prof.stats.wonCoupons} K / ${prof.stats.lostCoupons} Y (%${prof.stats.winRatePct})</td>
                <td>
                  <b class="${prof.stats.totalNetProfit >= 0 ? 'good' : 'bad'}">
                    ${prof.stats.totalNetProfit >= 0 ? '+' : ''}${formatCurrency(prof.stats.totalNetProfit, 'EUR')}
                  </b>
                </td>
                <td style="color:var(--accent);font-size:13px;">
                  ${formatCurrency(prof.stats.finalBank, 'EUR')}
                </td>
                <td>
                  <span class="${prof.stats.totalRoiPct >= 0 ? 'good' : 'bad'}">
                    ${prof.stats.totalRoiPct >= 0 ? '+' : ''}%${prof.stats.totalRoiPct}
                  </span>
                </td>
                <td>—</td>
              </tr>
            </tfoot>
          </table>
        </div>

        <div style="margin-top:12px;padding:10px 14px;background:rgba(255,255,255,0.02);border:1px solid var(--line);border-radius:8px;font-size:11px;color:var(--muted);line-height:1.5;">
          🛡️ <b>Kasa Yönetim Notu:</b> ${esc(prof.name)} modelinde her gün kasanın %${prof.reservePct}'si (dokunulmaz sermaye tabanı) korunmuştur. Kupon kaybetmesi halinde bile kasa sıfırlanmaz, sonraki günün stake tutarı geriye kalan toplam kasanın %${prof.stakePct}'si olarak otomatik küçülür.
        </div>
      </div>
    `;
  }

  function render30DaySimulationSvg(simData, activeProfile) {
    if (!simData || !simData.profiles) return '';

    const W = 820;
    const H = 340;
    const L = 65;
    const R = 30;
    const T = 30;
    const B = 40;
    const pw = W - L - R;
    const ph = H - T - B;

    const totalDays = (simData.simulationWindow && simData.simulationWindow.totalDays) || 31;
    const getX = (d) => L + (d / totalDays) * pw;

    // Find maxY
    let maxVal = 70;
    const profilesToCheck = (activeProfile === 'all' || !activeProfile)
      ? ['minimum', 'medium', 'high']
      : [activeProfile];

    profilesToCheck.forEach(k => {
      const prof = simData.profiles[k];
      if (!prof) return;
      (prof.ledger || []).forEach(pt => {
        if (pt.endBank) maxVal = Math.max(maxVal, pt.endBank);
      });
      // Sıfır kayıp potansiyel kasasını hem tek profil hem de all modunda ölçeklemeye dahil et
      if (prof.maxPotential && prof.maxPotential.ledger) {
        prof.maxPotential.ledger.forEach(pt => {
          if (pt.endBank) maxVal = Math.max(maxVal, pt.endBank);
        });
      }
    });

    const maxY = Math.ceil(Math.max(70, maxVal * 1.12) / 10) * 10;
    const getY = (val) => T + ph - (Math.max(0, val) / maxY) * ph;

    // Grid lines
    let gridLines = '';
    const numYSteps = 5;
    const yStepVal = maxY / numYSteps;
    for (let i = 0; i <= numYSteps; i++) {
      const v = Math.round(i * yStepVal);
      const yPos = getY(v);
      gridLines += `
        <line x1="${L}" y1="${yPos.toFixed(1)}" x2="${W - R}" y2="${yPos.toFixed(1)}" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
        <text x="${(L - 8).toFixed(1)}" y="${(yPos + 3.5).toFixed(1)}" fill="var(--muted)" font-size="10" text-anchor="end" font-family="inherit">${formatCurrency(v, 'EUR')}</text>
      `;
    }

    // X axis day guides
    let xGuides = '';
    const xSteps = [0, 7, 14, 21, totalDays];
    for (const d of xSteps) {
      const xPos = getX(d);
      xGuides += `
        <line x1="${xPos.toFixed(1)}" y1="${T}" x2="${xPos.toFixed(1)}" y2="${(T + ph).toFixed(1)}" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>
        <text x="${xPos.toFixed(1)}" y="${(T + ph + 16).toFixed(1)}" fill="var(--muted)" font-size="10.5" text-anchor="middle" font-family="inherit">${d === 0 ? '0. Gün' : `${d}. Gün`}</text>
      `;
    }

    // Curves
    const profStyles = {
      minimum: { stroke: '#10b981', fill: 'rgba(16,185,129,0.12)', gradId: 'gradSimMin', name: 'Minimum Risk' },
      medium: { stroke: '#38bdf8', fill: 'rgba(56,189,248,0.12)', gradId: 'gradSimMed', name: 'Orta Risk' },
      high: { stroke: '#ef4444', fill: 'rgba(239,68,68,0.14)', gradId: 'gradSimHigh', name: 'Yüksek Risk' }
    };

    let curvesSvg = '';
    let dotsSvg = '';
    let legendSvg = '';

    profilesToCheck.forEach(profKey => {
      const prof = simData.profiles[profKey];
      if (!prof || !prof.ledger) return;
      const st = profStyles[profKey] || profStyles.minimum;

      // 1. Gerçekleşen Eğri
      let pathD = `M ${getX(0).toFixed(1)},${getY(50.0).toFixed(1)}`;
      let areaD = `M ${getX(0).toFixed(1)},${(T + ph).toFixed(1)} L ${getX(0).toFixed(1)},${getY(50.0).toFixed(1)}`;

      prof.ledger.forEach(pt => {
        const px = getX(pt.day).toFixed(1);
        const py = getY(pt.endBank).toFixed(1);
        pathD += ` L ${px},${py}`;
        areaD += ` L ${px},${py}`;

        dotsSvg += `
          <circle class="sim30-dot" cx="${px}" cy="${py}" r="3.5" fill="${st.stroke}" stroke="#0b1120" stroke-width="1.5"
            data-day="${pt.day}" data-date="${pt.date}" data-prof="${prof.name}" data-bank="${pt.endBank}" data-change="${pt.dailyChangePct}" data-status="${pt.status}"
            style="cursor:pointer;transition:r .15s;" />
        `;
      });

      areaD += ` L ${getX(totalDays).toFixed(1)},${(T + ph).toFixed(1)} Z`;

      curvesSvg += `
        <path d="${areaD}" fill="url(#${st.gradId})" opacity="${activeProfile === 'all' ? 0.30 : 0.45}"/>
        <path d="${pathD}" fill="none" stroke="${st.stroke}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
      `;

      // 2. Sıfır Kayıp / Maksimum Potansiyel Eğrisi
      if (prof.maxPotential && prof.maxPotential.ledger) {
        let noLossPathD = `M ${getX(0).toFixed(1)},${getY(50.0).toFixed(1)}`;
        const isAll = (activeProfile === 'all');
        const curveColor = isAll ? st.stroke : '#fbbf24';

        prof.maxPotential.ledger.forEach(pt => {
          const px = getX(pt.day).toFixed(1);
          const py = getY(pt.endBank).toFixed(1);
          noLossPathD += ` L ${px},${py}`;

          dotsSvg += `
            <circle class="sim30-dot sim30-dot-noloss" cx="${px}" cy="${py}" r="${isAll ? 2.8 : 3.2}" fill="${curveColor}" stroke="#0b1120" stroke-width="1.2"
              data-day="${pt.day}" data-date="${pt.date}" data-prof="${prof.name} (Sıfır Kayıp Potansiyeli)" data-bank="${pt.endBank}" data-change="${pt.dailyChangePct}" data-status="won"
              style="cursor:pointer;transition:r .15s;" />
          `;
        });

        curvesSvg += `
          <path d="${noLossPathD}" fill="none" stroke="${curveColor}" stroke-width="${isAll ? '1.8' : '2.2'}" ${isAll ? 'stroke-dasharray="5,4"' : ''} stroke-linecap="round" stroke-linejoin="round"/>
        `;

        if (!isAll) {
          legendSvg = `
            <g transform="translate(${L + 10}, ${T - 12})">
              <line x1="0" y1="0" x2="16" y2="0" stroke="${st.stroke}" stroke-width="2.4"/>
              <circle cx="8" cy="0" r="3.5" fill="${st.stroke}"/>
              <text x="22" y="3.5" fill="var(--text)" font-size="11" font-weight="700" font-family="inherit">Gerçekleşen Kasa (${formatCurrency(prof.stats.finalBank, 'EUR')})</text>

              <line x1="190" y1="0" x2="206" y2="0" stroke="#fbbf24" stroke-width="2.2"/>
              <circle cx="198" cy="0" r="3" fill="#fbbf24"/>
              <text x="212" y="3.5" fill="#fbbf24" font-size="11" font-weight="700" font-family="inherit">★ Sıfır Kayıp Maksimum Potansiyel (${formatCurrency(prof.maxPotential.finalBank, 'EUR')})</text>
            </g>
          `;
        }
      }
    });

    if (activeProfile === 'all') {
      legendSvg = `
        <g transform="translate(${L + 10}, ${T - 12})">
          <line x1="0" y1="0" x2="16" y2="0" stroke="#94a3b8" stroke-width="2.4"/>
          <text x="22" y="3.5" fill="var(--text)" font-size="10.5" font-weight="700" font-family="inherit">Dolu Çizgiler: Gerçekleşen</text>

          <line x1="170" y1="0" x2="190" y2="0" stroke="#fbbf24" stroke-width="1.8" stroke-dasharray="5,4"/>
          <text x="196" y="3.5" fill="#fbbf24" font-size="10.5" font-weight="700" font-family="inherit">Kesikli Çizgiler: Sıfır Kayıp Potansiyelleri</text>
        </g>
      `;
    }

    let defsSvg = '<defs>';
    if (profilesToCheck.includes('minimum')) {
      defsSvg += `
        <linearGradient id="gradSimMin" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#10b981" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="#10b981" stop-opacity="0.0"/>
        </linearGradient>
      `;
    }
    if (profilesToCheck.includes('medium')) {
      defsSvg += `
        <linearGradient id="gradSimMed" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.0"/>
        </linearGradient>
      `;
    }
    if (profilesToCheck.includes('high')) {
      defsSvg += `
        <linearGradient id="gradSimHigh" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#ef4444" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="#ef4444" stop-opacity="0.0"/>
        </linearGradient>
      `;
    }
    defsSvg += '</defs>';

    return `
      <svg viewBox="0 0 ${W} ${H}" class="trajectory-svg" id="sim30Svg" style="width:100%;height:auto;display:block;user-select:none;">
        ${defsSvg}

        <!-- Arka Plan Kılavuz Çizgileri -->
        ${gridLines}
        ${xGuides}

        <!-- 50 Euro Başlangıç Kılavuzu -->
        <line x1="${L}" y1="${getY(50.0).toFixed(1)}" x2="${W - R}" y2="${getY(50.0).toFixed(1)}" stroke="rgba(255,255,255,0.2)" stroke-dasharray="2,2"/>
        <text x="${(W - R).toFixed(1)}" y="${(getY(50.0) - 5).toFixed(1)}" fill="var(--muted)" font-size="9.5" text-anchor="end" font-family="inherit">Başlangıç: 50,00 €</text>

        <!-- Lejant (Tek profil modunda) -->
        ${legendSvg}

        <!-- Eğriler ve Alanlar -->
        ${curvesSvg}
        ${dotsSvg}

        <!-- İnteraktif Tooltip Kutusu -->
        <g id="sim30ChartTooltip" opacity="0" pointer-events="none">
          <rect id="sim30TooltipBg" x="0" y="0" width="160" height="62" rx="6" fill="#0f172a" stroke="#334155" stroke-width="1" filter="drop-shadow(0 4px 8px rgba(0,0,0,0.5))"/>
          <text id="sim30TtTitle" x="0" y="0" fill="#94a3b8" font-size="10" font-weight="700" font-family="inherit"></text>
          <text id="sim30TtVal" x="0" y="0" fill="#38bdf8" font-size="12" font-weight="800" font-family="inherit"></text>
          <text id="sim30TtSub" x="0" y="0" fill="#94a3b8" font-size="9.5" font-family="inherit"></text>
        </g>
      </svg>
    `;
  }

  function wireSimKasaEvents(simData) {
    function updateSimKasaSelection(prof) {
      if (!prof) return;
      sim30ActiveProfile = prof;
      // Update pills active class
      document.querySelectorAll('#pane-sim-kasa #sim30ChartPills .cmp-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.prof === prof);
      });
      // Update KPI cards active class
      document.querySelectorAll('#pane-sim-kasa .sim30-kpi-card').forEach(c => {
        c.classList.toggle('active-card', c.dataset.prof === prof);
      });
      // Re-render chart SVG smoothly
      const chartBox = document.getElementById('sim30ChartContainer');
      if (chartBox) {
        chartBox.innerHTML = render30DaySimulationSvg(simData, prof);
        const newSvg = chartBox.querySelector('#sim30Svg');
        if (newSvg) wire30DaySimulationChartEvents(newSvg);
      }
      // Re-render ledger table smoothly
      const ledgerBox = document.getElementById('sim30LedgerContainer');
      if (ledgerBox) {
        ledgerBox.innerHTML = render30DayLedgerTableHtml(simData, prof);
      }
    }

    document.querySelectorAll('#pane-sim-kasa .sim30-kpi-card').forEach(card => {
      card.onclick = () => {
        const prof = card.dataset.prof;
        if (prof) updateSimKasaSelection(prof);
      };
    });

    document.querySelectorAll('#pane-sim-kasa #sim30ChartPills .cmp-btn').forEach(btn => {
      btn.onclick = () => {
        const prof = btn.dataset.prof;
        if (prof) updateSimKasaSelection(prof);
      };
    });

    const btnGoKupon = document.getElementById('btnGoToSimKupon');
    if (btnGoKupon) {
      btnGoKupon.onclick = () => {
        if (typeof root.setTab === 'function') root.setTab('sim-kupon');
      };
    }

    const btnGoPlan = document.getElementById('btnGoToPlan');
    if (btnGoPlan) {
      btnGoPlan.onclick = () => {
        if (typeof root.setTab === 'function') root.setTab('plan');
      };
    }

    const svgEl = document.querySelector('#pane-sim-kasa #sim30Svg');
    if (svgEl) {
      wire30DaySimulationChartEvents(svgEl);
    }
  }

  function wireSimKuponEvents(simData) {
    document.querySelectorAll('#pane-sim-kupon #simKuponProfilePills .cmp-btn').forEach(btn => {
      btn.onclick = () => {
        const prof = btn.dataset.prof;
        if (prof) {
          sim30ActiveProfile = prof;
          renderSimKuponPane();
        }
      };
    });

    const btnKuponGoPlan = document.getElementById('btnKuponGoToPlan');
    if (btnKuponGoPlan) {
      btnKuponGoPlan.onclick = () => {
        if (typeof root.setTab === 'function') root.setTab('plan');
      };
    }
  }

  function wire30DaySimulationChartEvents(svgEl) {
    const tooltip = svgEl.querySelector('#sim30ChartTooltip');
    const ttBg = svgEl.querySelector('#sim30TooltipBg');
    const ttTitle = svgEl.querySelector('#sim30TtTitle');
    const ttVal = svgEl.querySelector('#sim30TtVal');
    const ttSub = svgEl.querySelector('#sim30TtSub');
    if (!tooltip || !ttBg || !ttTitle || !ttVal || !ttSub) return;

    const dots = svgEl.querySelectorAll('.sim30-dot, .sim30-dot-noloss');
    dots.forEach(dot => {
      dot.onmouseenter = () => {
        const cx = parseFloat(dot.getAttribute('cx'));
        const cy = parseFloat(dot.getAttribute('cy'));
        const day = dot.dataset.day;
        const date = dmy(dot.dataset.date);
        const prof = dot.dataset.prof;
        const bank = formatCurrency(dot.dataset.bank, 'EUR');
        const change = parseFloat(dot.dataset.change);
        const isNoLoss = dot.classList.contains('sim30-dot-noloss');
        const status = isNoLoss ? '⭐ Sıfır Kayıp Potansiyeli' : (dot.dataset.status === 'won' ? '✅ Kupon Kazandı' : '❌ Kupon Kaybetti');

        dot.setAttribute('r', '6');

        ttTitle.textContent = `Gün ${day} · ${date} (${prof})`;
        ttVal.textContent = `Kasa: ${bank}`;
        ttVal.setAttribute('fill', isNoLoss ? '#fbbf24' : '#38bdf8');
        ttSub.textContent = `${status} (${change >= 0 ? '+' : ''}%${change})`;

        let boxX = cx + 12;
        let boxY = cy - 35;
        if (boxX + 170 > 820) boxX = cx - 175;
        if (boxY < 10) boxY = 10;

        ttBg.setAttribute('x', boxX);
        ttBg.setAttribute('y', boxY);
        ttTitle.setAttribute('x', boxX + 10);
        ttTitle.setAttribute('y', boxY + 16);
        ttVal.setAttribute('x', boxX + 10);
        ttVal.setAttribute('y', boxY + 34);
        ttSub.setAttribute('x', boxX + 10);
        ttSub.setAttribute('y', boxY + 50);

        tooltip.setAttribute('opacity', '1');
      };

      dot.onmouseleave = () => {
        dot.setAttribute('r', dot.classList.contains('sim30-dot-noloss') ? '3' : '3.5');
        tooltip.setAttribute('opacity', '0');
      };
    });
  }

  // ---------------------------------------------------------------------------
  // Sanal Kasa Ekranı (#pane-plan) — Paper_Betting_Kasa_Simulasyonu v01 birebir
  // KULLANICI GİRİŞLERİ · OTOMATİK PARAMETRELER · Kasa Gelişim Grafiği ·
  // Kasa tablosu: Gün · Hedef Kasa · Gerçek Kasa · Günlük Değişim · Günlük Büyüme · Toplam Büyüme
  // ---------------------------------------------------------------------------

  const KASA_RISK_LABELS = { minimum: 'Minimum', medium: 'Medium', high: 'High' };
  const KASA_COLOR_REAL = '#4F81BD';
  const KASA_COLOR_TARGET = '#C0504D';
  const KASA_COLOR_SECURED = '#9BBB59';   // kenara konan (cash-out) kısım

  function kasaRiskLabel(params) {
    return KASA_RISK_LABELS[params.riskProfile] || params.riskName;
  }

  function formatAmount(val) {
    return (Number(val) || 0).toLocaleString((root.I18N && root.I18N.locale) || 'tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  // Excel 0.0% biçimi; Türkçede eksi işareti yüzde simgesinin önünde (-%6,3), en/nl'de -6.3% / -6,3%
  function signedPct(val) {
    return (val < 0 ? '-' : '') + formatPct(Math.abs(val), 1);
  }

  function kasaDisplayName(plan, idx) {
    return (!plan.name || / Risk Kasası$/.test(plan.name)) ? _t('Kasa {n}', { n: idx + 1 }) : plan.name;
  }

  // Kasa yoksa Excel dosyasındaki örnek kasa ile başlar (50 € → 1000 €, Medium, 1-12. gün gerçek kasa).
  // Kişi bir değer değiştirene kadar kaydedilmez: yeni bir cihaz hesaba bağlandığında bu örnek kasa
  // buluttaki kasalarla birleşip fazladan bir "Kasa 1" oluşturmasın.
  function ensureDefaultKasa() {
    if (paperState) PE.ensurePlansArray(paperState);
    if (paperState && paperState.plans && paperState.plans.length) return;
    if (paperState && paperState.closedPlans && paperState.closedPlans.length) {
      openBlankKasaAfter(paperState.closedPlans[paperState.closedPlans.length - 1]);
      return;
    }
    const example = Object.assign({}, PE.KASA_V01_EXAMPLE, { name: _t('Kasa {n}', { n: 1 }) });
    if (paperState) {
      PE.createNewPlan(paperState, example);
    } else {
      paperState = PE.createInitialState({ currency: 'EUR', riskProfile: example.riskProfile }, example);
    }
  }

  // Kapatılan kasanın ayarlarıyla (başlangıç/hedef/risk) boş yeni bir kasa açar
  function openBlankKasaAfter(prev) {
    const n = paperState.plans.length + paperState.closedPlans.length + 1;
    PE.createNewPlan(paperState, {
      name: _t('Kasa {n}', { n }),
      startingBank: (prev && prev.startingBank) || PE.KASA_V01_EXAMPLE.startingBank,
      targetBank: (prev && prev.targetBank) || PE.KASA_V01_EXAMPLE.targetBank,
      riskProfile: (prev && prev.riskProfile) || PE.KASA_V01_EXAMPLE.riskProfile
    });
  }

  // ---- Excel (.xlsx) dışa aktarma: SheetJS ilk tıklamada CDN'den yüklenir; yüklenemezse CSV iner ----
  let xlsxLoading = null;
  function loadXlsxLib() {
    if (root.XLSX) return Promise.resolve(root.XLSX);
    if (!xlsxLoading) {
      xlsxLoading = new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = 'https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js';
        s.onload = () => (root.XLSX ? resolve(root.XLSX) : reject(new Error('XLSX')));
        s.onerror = () => { xlsxLoading = null; reject(new Error('XLSX')); };
        document.head.appendChild(s);
      });
    }
    return xlsxLoading;
  }

  function safeFileName(s) {
    return String(s || 'kasa').replace(/[\\/:*?"<>|]+/g, '').replace(/\s+/g, '_').slice(0, 40) || 'kasa';
  }

  // Kasa sayfası: özet bilgiler + günlük tablo (sayılar Excel'de sayı olarak kalır)
  function kasaSheetRows(plan, sim, closed) {
    const p = sim.params;
    const pct = v => (v == null ? '' : Math.round(v * 100) / 10000);   // Excel yüzde hücresi (0,3732 = %37,3)
    const head = [
      [_t('Kasa Adı'), plan.name || ''],
      [_t('Risk Faktörü'), KASA_RISK_LABELS[p.riskProfile] || p.riskName],
      [_t('Başlangıç Tarihi'), dmy(plan.startDate)],
      [_t('Başlangıç Kasası'), p.startingBank],
      [_t('Hedef Kasa'), p.targetBank],
      [_t('Günlük Büyüme Oranı'), p.dailyGrowthRate],
      [_t('Hedefe Ulaşma Günü'), p.daysToTarget != null ? p.daysToTarget : '']
    ];
    const moneyRows = [3, 4], pctRows = [5];
    if (closed) {
      head.push([_t('Kapanış Tarihi'), dmy(closed.closedAt)]);
      moneyRows.push(head.push([_t('Kapanış Kasası'), closed.finalBank]) - 1);
      moneyRows.push(head.push([_t('Toplam Kazanç'), closed.profit]) - 1);
      pctRows.push(head.push([_t('Toplam Büyüme'), pct(closed.growthPct)]) - 1);
    }
    if (sim.secured > 0) moneyRows.push(head.push([_t('Kilitli'), sim.secured]) - 1);
    const cols = [_t('Gün'), _t('Tarih'), _t('Hedef Kasa'), _t('Oyundaki Kasa'), _t('Kilitli'), _t('Günlük Değişim'), _t('Günlük Büyüme'), _t('Toplam Büyüme')];
    const body = sim.rows.map(r => [r.day, dmy(r.date), r.targetBank, r.actualBank != null ? r.actualBank : '',
      r.actualBank != null && r.secured > 0 ? r.secured : '',
      r.dailyChange != null ? r.dailyChange : '', pct(r.dailyGrowthPct), pct(r.totalGrowthPct)]);
    return { aoa: head.concat([[]], [cols], body), moneyRows, pctRows, dataStart: head.length + 2 };
  }

  function exportKasaExcel(plan, sim, closed) {
    const { aoa, moneyRows, pctRows, dataStart } = kasaSheetRows(plan, sim, closed);
    const base = `betavus-${safeFileName(plan.name)}-${new Date().toISOString().slice(0, 10)}`;
    loadXlsxLib().then(XLSX => {
      const ws = XLSX.utils.aoa_to_sheet(aoa);
      const fmt = (r, c, z) => { const a = XLSX.utils.encode_cell({ r, c }); if (ws[a] && typeof ws[a].v === 'number') ws[a].z = z; };
      moneyRows.forEach(r => fmt(r, 1, '#,##0.00'));
      pctRows.forEach(r => fmt(r, 1, '0.0%'));
      for (let r = dataStart; r < aoa.length; r++) {
        [2, 3, 4, 5].forEach(c => fmt(r, c, '#,##0.00'));
        [6, 7].forEach(c => fmt(r, c, '0.0%'));
      }
      ws['!cols'] = [{ wch: 26 }, { wch: 14 }, { wch: 14 }, { wch: 14 }, { wch: 14 }, { wch: 15 }, { wch: 14 }, { wch: 14 }];
      const wb = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(wb, ws, 'Kasa');
      XLSX.writeFile(wb, base + '.xlsx');
    }).catch(() => {
      // Kütüphane yüklenemezse Excel'in açabildiği CSV (; ayraçlı, UTF-8 BOM)
      const csv = aoa.map(row => row.map(v => {
        const s = typeof v === 'number' ? String(v).replace('.', ',') : String(v == null ? '' : v);
        return /[;"\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
      }).join(';')).join('\r\n');
      const url = URL.createObjectURL(new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8' }));
      const a = document.createElement('a');
      a.href = url; a.download = base + '.csv'; a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
  }

  // Kapatılan Kasalar: bugüne kadarki toplam + kasa bazında liste
  function renderClosedKasasHtml(curr) {
    const list = (paperState.closedPlans || []).slice().reverse();   // en son kapatılan üstte
    if (!list.length) return '';
    const sum = PE.getClosedPlansSummary(paperState);
    const sym = (PE.CURRENCIES[curr] || PE.CURRENCIES.EUR).symbol;
    const cls = v => (v == null ? '' : v >= 0 ? 'good' : 'bad');
    const sign = v => (v > 0 ? '+' : '');
    return `
      <div class="card closed-kasa-card">
        <h3 class="ck-title">${_t('Kapatılan Kasalar')}</h3>
        <div class="ck-summary">
          <div><span>${_t('Kapatılan Kasa')}</span><b>${sum.count}</b></div>
          <div><span>${_t('Toplam Başlangıç')}</span><b>${formatCurrency(sum.totalStart, curr)}</b></div>
          <div><span>${_t('Toplam Kapanış')}</span><b>${formatCurrency(sum.totalFinal, curr)}</b></div>
          <div><span>${_t('Bugüne Kadar Toplam Kazanç')}</span><b class="${cls(sum.totalProfit)}">${sign(sum.totalProfit)}${formatCurrency(sum.totalProfit, curr)}</b></div>
          <div><span>${_t('Toplam Büyüme')}</span><b class="${cls(sum.totalGrowthPct)}">${sum.totalGrowthPct != null ? signedPct(sum.totalGrowthPct) : '—'}</b></div>
        </div>
        <div class="tbl-scroll">
          <table class="excel-table kasa-sim-table closed-kasa-table">
            <thead><tr>
              <th>${_t('Kasa')}</th><th>${_t('Risk')}</th><th>${_t('Tarih')}</th><th>${_t('Gün')}</th>
              <th>${_t('Başlangıç Kasası ({sym})', { sym })}</th>
              <th>${_t('Kapanış Kasası ({sym})', { sym })}</th>
              <th>${_t('Toplam Kazanç ({sym})', { sym })}</th><th>${_t('Toplam Büyüme (%)')}</th><th></th>
            </tr></thead>
            <tbody>
              ${list.map(c => `
                <tr>
                  <td class="ck-name">${esc(c.name)}</td>
                  <td>${esc(KASA_RISK_LABELS[c.riskProfile] || _t('Özel'))}</td>
                  <td>${dmy(c.startDate)} – ${dmy(c.closedAt)}</td>
                  <td>${c.daysPlayed}</td>
                  <td>${formatCurrency(c.startingBank, curr)}</td>
                  <td>${formatCurrency(c.finalBank, curr)}</td>
                  <td class="${cls(c.profit)}">${sign(c.profit)}${formatCurrency(c.profit, curr)}</td>
                  <td class="${cls(c.growthPct)}">${c.growthPct != null ? signedPct(c.growthPct) : ''}</td>
                  <td class="ck-acts"><button type="button" class="ck-xls" data-closed-id="${esc(c.id)}" title="${_t('Excel olarak dışa aktar')}">${_t('📊 Excel')}</button>
                    <button type="button" class="ck-xls ck-reopen" data-closed-id="${esc(c.id)}" title="${_t('Kasayı yeniden aç: girdileri düzeltip tekrar kapatabilirsin')}">${_t('✏️ Yeniden aç')}</button>
                    <button type="button" class="ck-xls ck-del" data-closed-id="${esc(c.id)}" title="${_t('Kapatılan kasayı sil')}">${_t('🗑️ Sil')}</button></td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>`;
  }

  // Excel "Kasa Gelisim Grafigi": çizgi grafik, 1-30. gün, Gerçek Kasa (€) ve Teorik Hedef Kasa (€),
  // işaretçisiz 2,25 pt çizgiler, boş günler boşluk, açıklama altta
  // Eksen adımı 1-2-2,5-5 × 10^n dizisinden seçilir; değer ne kadar büyük olursa olsun 5-8 etiket kalır
  function niceAxisStep(rough) {
    const mag = Math.pow(10, Math.floor(Math.log10(Math.max(rough, 1e-9))));
    const f = rough / mag;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * mag;
  }

  function renderKasaChartSvg(sim, curr) {
    const W = 820, H = 330, L = 64, R = 18, T = 18, B = 34;
    const pw = W - L - R, ph = H - T - B;
    const rows = sim.rows;                 // 1. günden hedef gününe kadar
    const n = rows.length;
    const peak = Math.max(1, ...rows.map(r => r.targetBank), ...rows.map(r => r.totalBank || 0));
    const step = niceAxisStep(peak / 6);
    const maxY = Math.ceil(peak / step) * step;
    const x = i => L + (i + 0.5) * (pw / n);
    const y = v => T + ph - (Math.max(0, v) / maxY) * ph;

    let grid = '';
    for (let v = 0; v <= maxY; v += step) {
      grid += `<line x1="${L}" y1="${y(v).toFixed(1)}" x2="${W - R}" y2="${y(v).toFixed(1)}" stroke="var(--line)" stroke-width="1"/>
        <text x="${L - 8}" y="${(y(v) + 3.5).toFixed(1)}" fill="var(--muted)" font-size="11" text-anchor="end">${v.toLocaleString((root.I18N && root.I18N.locale) || 'tr-TR')}</text>`;
    }
    let xLabels = '';
    const labelEvery = Math.max(1, Math.ceil(n / 15));  // en fazla ~15 gün etiketi
    rows.forEach((r, i) => {
      if (i % labelEvery === 0 || i === n - 1) xLabels += `<text x="${x(i).toFixed(1)}" y="${(T + ph + 18).toFixed(1)}" fill="var(--muted)" font-size="11" text-anchor="middle">${r.day}</text>`;
    });

    // Kümelenmiş sütunlar: her gün için yan yana Gerçek Kasa + Teorik Hedef Kasa.
    // Sütunlar tabandan yükselir, üst köşeleri yuvarlaktır; iki sütun arasında 2px boşluk.
    // Gerçek kasası girilmemiş günde yalnızca hedef sütunu çizilir (dispBlanksAs = gap).
    const slot = pw / n;
    const groupW = Math.min(slot * 0.72, 30);
    const barW = Math.max(1, (groupW - 2) / 2);
    const base = T + ph;
    const bar = (bx, v, color) => {
      if (v == null || !(v > 0)) return '';
      const top = y(v), h = base - top;
      const rad = Math.min(4, barW / 2, h);
      return `<path d="M${bx.toFixed(1)},${base.toFixed(1)} V${(top + rad).toFixed(1)} Q${bx.toFixed(1)},${top.toFixed(1)} ${(bx + rad).toFixed(1)},${top.toFixed(1)} H${(bx + barW - rad).toFixed(1)} Q${(bx + barW).toFixed(1)},${top.toFixed(1)} ${(bx + barW).toFixed(1)},${(top + rad).toFixed(1)} V${base.toFixed(1)} Z" fill="${color}"/>`;
    };
    const bars = rows.map((r, i) => {
      const gx = x(i) - groupW / 2;
      // Güven payı (yeşil) oyundaki kasanın (mavi) üstünde durur; mavi yükseklik hedefle kıyaslanır
      const sec = r.actualBank != null && r.secured - r.cashout > 0 ? r.secured - r.cashout : 0;
      const real = sec > 0
        ? bar(gx, r.actualBank + sec, KASA_COLOR_SECURED) + (r.actualBank > 0 ? `<rect x="${gx.toFixed(1)}" y="${y(r.actualBank).toFixed(1)}" width="${barW.toFixed(1)}" height="${(base - y(r.actualBank)).toFixed(1)}" fill="${KASA_COLOR_REAL}"/>` : '')
        : bar(gx, r.actualBank, KASA_COLOR_REAL);
      return real + bar(gx + barW + 2, r.targetBank, KASA_COLOR_TARGET);
    }).join('');

    // Üzerine gelince günün iki değeri (sütundan geniş, tüm gün dilimi)
    const hover = rows.map((r, i) => `<rect class="ks-hit" x="${(L + i * slot).toFixed(1)}" y="${T}" width="${slot.toFixed(1)}" height="${ph}"><title>${_t('{day}. gün · Oyundaki Kasa: {real} · Teorik Hedef Kasa: {target}', { day: r.day, real: r.actualBank != null ? formatCurrency(r.actualBank, curr) : '—', target: formatCurrency(r.targetBank, curr) })}${r.secured > 0 && r.actualBank != null ? ' · ' + _t('Kilitli: {v}', { v: formatCurrency(r.secured, curr) }) : ''}</title></rect>`).join('');

    return `
      <svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${_t('Kasa Gelişim Grafiği: gerçek kasa ve teorik hedef kasa, 1-{n}. gün', { n })}" style="display:block">
        ${grid}
        ${hover}
        <g pointer-events="none">${bars}</g>
        <line x1="${L}" y1="${T + ph}" x2="${W - R}" y2="${T + ph}" stroke="var(--muted)" stroke-width="1" opacity="0.5"/>
        ${xLabels}
      </svg>`;
  }

  // ---------------------------------------------------------------------------
  // Güven Payı kartı: kupon / rahat sınır ışığı, tek butonla kilitleme, duraklı yol çizgisi
  // ---------------------------------------------------------------------------

  function eiRiskName(k) {
    return KASA_RISK_LABELS[k] || _t('Özel');
  }

  function renderEICoachHtml(plan, sim, curr) {
    const c = PE.getEICoach(plan, sim);
    const m = v => formatCurrency(v, curr);
    // Durak kasası yaklaşık gösterilir (263,37 € → ~260 €): kişi kuruşla değil, seviyeyle ilgilenir
    const approx = v => '~' + formatCurrency(v >= 100 ? Math.round(v / 10) * 10 : Math.round(v), curr).replace(/[.,]00(?=\D*$)/, '');
    const vars = { stake: `<b>${m(c.stake)}</b>`, limit: `<b>${m(c.limit)}</b>` };

    let status;
    if (!c.day) {
      status = `<p class="ei-msg">${_t('İlk günün kasasını tabloya girince başlar.')}</p>`;
    } else if (c.reached) {
      status = `<p class="ei-msg good">${_t('🏁 Hedefe ulaştın! Kasayı kapatıp kazancını koruyabilirsin.')}</p>`;
    } else {
      const line = {
        red: _t('🔴 Bugünkü kuponun {stake}, güvenli sınırın {limit}. Sınırı aştın.', vars),
        yellow: _t('🟡 Bugünkü kuponun {stake}, güvenli sınırın {limit}. Sınıra yaklaşıyorsun.', vars),
        green: _t('🟢 Bugünkü kuponun {stake}, güvenli sınırın {limit}. Rahatsın.', vars)
      }[c.level];
      status = `
        <div class="ei-light ei-${c.level}">${line}</div>
        ${c.lock ? `
          <div class="ei-act">
            <button type="button" class="ei-lock-btn" id="btnEILock" data-day="${c.day}" data-amount="${c.lock.amount}">${_t('🔒 {amount} kilitle', { amount: m(c.lock.amount) })}</button>
            <span class="ei-muted">${_t('Kuponun {stake} olur.', { stake: m(c.lock.stakeAfter) })}</span>
          </div>` : ''}
        ${c.drop ? `<p class="ei-msg bad">${_t('📉 Zirveden %{pct} düştün. Bugün mola vermeyi düşün; kaybı hemen geri kazanmaya çalışma.', { pct: c.drop.pct })}</p>` : ''}
        ${!c.lock && c.next ? `<p class="ei-next">${_t('Sonraki kilit: kasan {bank} olunca.', { bank: approx(c.next.total) })}</p>` : ''}`;
    }

    return `
      <div class="card ei-card" id="eiCoachCard">
        <h3 class="ei-title">${_t('🔒 Güven Payı')}</h3>
        <p class="ei-lead">${_t('Kasan büyüdükçe bir kısmını kilitle. Kilitli para bir daha riske girmez ama hedefine sayılır.')}</p>
        <div class="ei-sum">
          <span>${_t('Oyundaki')} <b>${m(c.working)}</b></span>
          <span>${_t('Kilitli')} <b class="good">${m(c.secured)}</b></span>
          <span>${_t('Toplam')} <b>${m(c.total)}</b></span>
        </div>
        ${status}
        <details class="ei-why">
          <summary>${_t('Güvenli sınır nedir?')}</summary>
          <p>${_t('ei.why', { start: m(Number(plan.startingBank) || 0) })}</p>
        </details>
        <div class="ei-links">
          <button type="button" class="ei-link" id="btnEIManual" data-day="${c.day}" ${c.day ? '' : 'disabled'}>${_t('Kilitle')}</button>
          <button type="button" class="ei-link" id="btnEIReturn" data-day="${c.day}" ${c.day && c.secured > 0 ? '' : 'disabled'}>${_t('Geri al')}</button>
          ${(sim.cashouts || []).length ? `<button type="button" class="ei-link" id="btnEIUndo">${_t('Son işlemi geri al')}</button>` : ''}
        </div>
      </div>`;
  }

  function wireEICoachEvents() {
    const card = document.getElementById('eiCoachCard');
    if (!card) return;
    const curr = (paperState.settings && paperState.settings.currency) || 'EUR';
    const lock = document.getElementById('btnEILock');
    if (lock) {
      lock.onclick = () => {
        PE.applyCashout(paperState, { day: Number(lock.dataset.day), amount: Number(lock.dataset.amount), reason: 'lock' });
        rerenderAllPanes();
      };
    }
    const ask = (msg, def, max) => {
      const raw = prompt(msg, formatAmount(def));
      if (raw === null) return null;
      const v = parseKasaAmount(raw);
      if (v === '' || v == null || v <= 0 || v > max) {
        alert(_t('0 ile {max} arasında bir tutar girin.', { max: formatCurrency(max, curr) }));
        return null;
      }
      return v;
    };
    const man = document.getElementById('btnEIManual');
    if (man) {
      man.onclick = () => {
        const sim = PE.buildKasaSimulation(paperState.plan, paperState);
        const v = ask(_t('Oyundaki kasadan ne kadar kilitlemek istiyorsun?'), Math.floor(sim.currentBank / 2), sim.currentBank);
        if (v == null) return;
        PE.applyCashout(paperState, { day: Number(man.dataset.day), amount: v, reason: 'manual' });
        rerenderAllPanes();
      };
    }
    const ret = document.getElementById('btnEIReturn');
    if (ret) {
      ret.onclick = () => {
        const sim = PE.buildKasaSimulation(paperState.plan, paperState);
        const v = ask(_t('Kilitli paradan ne kadarını oyundaki kasaya geri almak istiyorsun?'), sim.secured, sim.secured);
        if (v == null) return;
        PE.applyCashout(paperState, { day: Number(ret.dataset.day), amount: -v, reason: 'return' });
        rerenderAllPanes();
      };
    }
    const undo = document.getElementById('btnEIUndo');
    if (undo) undo.onclick = () => { PE.undoLastCashout(paperState); rerenderAllPanes(); };
  }

  function renderPlanPane(opts = {}) {
    const pane = document.getElementById('pane-plan');
    if (!pane) return;

    ensureDefaultKasa();
    const plan = PE.getActivePlan(paperState);
    const curr = (paperState.settings && paperState.settings.currency) || 'EUR';
    const sym = (PE.CURRENCIES[curr] || PE.CURRENCIES.EUR).symbol;
    const sim = PE.buildKasaSimulation(plan, paperState);
    const p = sim.params;
    const riskOptions = Object.keys(KASA_RISK_LABELS).concat(p.riskProfile === 'custom' ? ['custom'] : []);
    const hasCash = sim.cashouts.length > 0;
    const lastCashDay = hasCash ? sim.cashouts[sim.cashouts.length - 1].day : 0;

    // Yeniden çizimde tablo/sayfa kaydırması ve odaktaki gerçek kasa hücresi korunur
    // (aksi halde giriş sonrası tablo başa sarar, sayfa zıplar)
    // Üstteki kartların yüksekliği değişse bile (koç önerisi açılıp kapanınca) tablo kartı ekranda aynı yerde kalır.
    const oldWrap = pane.querySelector('.ks-table-wrap');
    const anchorSel = opts.anchor || (document.activeElement && document.activeElement.closest && document.activeElement.closest('#kasaSimCard') ? '#kasaSimCard' : null);
    const oldAnchor = anchorSel ? pane.querySelector(anchorSel) : null;
    const keep = {
      wrapTop: oldWrap ? oldWrap.scrollTop : null,
      winY: root.scrollY,
      anchorTop: oldAnchor ? oldAnchor.getBoundingClientRect().top : null,
      focusDay: opts.focusDay != null ? String(opts.focusDay)
        : document.activeElement && pane.contains(document.activeElement) && document.activeElement.classList.contains('kasa-input')
          ? document.activeElement.dataset.day : null
    };

    pane.innerHTML = `
      <div class="bankroll-switcher-bar">
        <div class="bankroll-tabs-scroll">
          <span class="bs-label">${_t('KASALARIM:')}</span>
          ${paperState.plans.map((pl, idx) => `
            <button type="button" class="bankroll-tab ${pl.id === paperState.activePlanId ? 'active' : ''}" data-plan-id="${pl.id}">
              <span class="bt-name">${esc(kasaDisplayName(pl, idx))}</span>
              <span class="bt-bank">${esc(KASA_RISK_LABELS[pl.riskProfile] || _t('Özel'))}</span>
            </button>
          `).join('')}
        </div>
        <div class="bankroll-actions">
          <button type="button" class="btn-new-bankroll" id="btnAddNewPlan">${_t('➕ Yeni Kasa Aç')}</button>
          ${paperState.plans.length > 1 ? `<button type="button" class="btn-delete-bankroll" id="btnDeleteCurrentPlan" title="${_t('Aktif Kasayı Sil')}">${_t('🗑️ Kasayı Sil')}</button>` : ''}
        </div>
      </div>

      <div class="card kasa-sheet">
        <h2 class="ks-title">${_t('PAPER BETTING – KASA SİMÜLASYONU')}</h2>
        <div class="p-quote ks-motto">${_t('« Önce simüle et. Riskini gör. Stratejini ölç. Sonra karar ver. »')}</div>
        <div class="ks-top">
          <div class="ks-left">
            <div class="ks-block">
              <div class="ks-block-h">${_t('KULLANICI GİRİŞLERİ')}</div>
              <label class="ks-row"><span>${_t('Kasa Adı')}</span>
                <input type="text" id="ksName" class="ks-input ks-name" maxlength="40" value="${esc(kasaDisplayName(plan, paperState.plans.findIndex(pl => pl.id === plan.id)))}" placeholder="${_t('Örn: Hafta sonu kasam')}"></label>
              <label class="ks-row"><span>${_t('Başlangıç Kasası ({sym})', { sym })}</span>
                <input type="text" inputmode="decimal" id="ksStart" class="ks-input" value="${formatAmount(p.startingBank)}"></label>
              <label class="ks-row"><span>${_t('Hedef Kasa ({sym})', { sym })}</span>
                <input type="text" inputmode="decimal" id="ksTarget" class="ks-input" value="${formatAmount(p.targetBank)}"></label>
              <label class="ks-row"><span>${_t('Risk Faktörü')}</span>
                <select id="ksRisk" class="ks-input">
                  ${riskOptions.map(k => `<option value="${k}" ${k === p.riskProfile ? 'selected' : ''}>${esc(KASA_RISK_LABELS[k] || p.riskName)}</option>`).join('')}
                </select></label>
            </div>
            <div class="ks-block">
              <div class="ks-block-h">${_t('OTOMATİK PARAMETRELER')}</div>
              <div class="ks-row"><span>${_t('Günlük Büyüme Oranı')}</span><b>${formatPct(p.dailyGrowthRate * 100, 0)}</b></div>
              <div class="ks-row"><span>${_t('Kasa Rezerv Oranı')}</span><b>${formatPct(p.reservePct * 100, 0)}</b></div>
              <div class="ks-row"><span>${_t('Hedefe Ulaşma Günü')}</span><b>${p.daysToTarget != null ? p.daysToTarget : ''}</b></div>
              <div class="ks-row"><span>${_t('Hedef Günündeki Teorik Kasa')}</span><b>${p.theoreticalAtTargetDay != null ? formatCurrency(p.theoreticalAtTargetDay, curr) : ''}</b></div>
            </div>
            <div class="ks-summary">
              <div><span>${_t('SEÇİLEN RİSK')}</span><b>${esc(kasaRiskLabel(p))}</b></div>
              <div><span>${_t('GÜNLÜK ARTIŞ')}</span><b>${formatPct(p.dailyGrowthRate * 100, 1)}</b></div>
              <div><span>${_t('REZERV')}</span><b>${formatPct(p.reservePct * 100, 0)}</b></div>
            </div>
          </div>
          <div class="ks-right">
            <div class="ks-chart">
              <div class="ks-chart-title">${_t('Kasa Gelişim Grafiği')}</div>
              ${renderKasaChartSvg(sim, curr)}
              <div class="ks-legend">
                <span><i style="background:${KASA_COLOR_REAL}"></i>${_t('Oyundaki Kasa ({sym})', { sym })}</span>
                <span><i style="background:${KASA_COLOR_TARGET}"></i>${_t('Teorik Hedef Kasa ({sym})', { sym })}</span>
                ${hasCash ? `<span><i style="background:${KASA_COLOR_SECURED}"></i>${_t('Kilitli ({sym})', { sym })}</span>` : ''}
              </div>
            </div>
            <div class="ks-actions">
              <button type="button" class="ks-act ks-act-close" id="btnCloseKasa">${_t('🔒 Kasayı Kapat')}</button>
              <button type="button" class="ks-act ks-act-xls" id="btnExportKasaXlsx">${_t('📊 Excel olarak dışa aktar')}</button>
            </div>
          </div>
        </div>
      </div>

      ${renderEICoachHtml(plan, sim, curr)}

      ${window.BETAVUS_COUPON ? window.BETAVUS_COUPON.html(p.riskProfile) : ''}

      <div class="card excel-model-card" id="kasaSimCard">
        <div class="tbl-scroll ks-table-wrap">
          <table class="excel-table kasa-sim-table">
            <thead>
              <tr>
                <th>${_t('Gün')}</th>
                <th>${_t('Hedef Kasa ({sym})', { sym })}</th>
                <th>${_t('Oyundaki Kasa ({sym})', { sym })}</th>
                ${hasCash ? `<th>${_t('Kilitli ({sym})', { sym })}</th>` : ''}
                <th>${_t('Günlük Değişim ({sym})', { sym })}</th>
                <th>${_t('Toplam Büyüme (%)')}</th>
              </tr>
            </thead>
            <tbody>
              ${sim.rows.map(r => `
                <tr class="${r.isToday ? 'row-today' : ''}${r.cashout ? ' row-cashout' : ''}">
                  <td title="${dmy(r.date)}">${r.day}</td>
                  <td>${formatCurrency(r.targetBank, curr)}</td>
                  <td class="real ${r.belowTarget ? 'below-target' : ''}"><input type="text" inputmode="decimal" class="kasa-input${r.isManual ? ' manual' : r.actualBank != null ? ' auto' : ''}" data-day="${r.day}" value="${r.actualBank != null ? formatAmount(r.actualBank) : ''}" title="${r.isManual ? _t('Elle girildi — silerseniz boş/otomatik değere döner') : r.actualBank != null ? _t('Sonuçlanan kuponlardan otomatik hesaplandı') : ''}" aria-label="${_t('{day}. gün gerçek kasa', { day: r.day })}"></td>
                  ${hasCash ? `<td class="ks-secured" ${r.cashout ? `title="${r.cashout > 0 ? _t('Bu gün {amount} kilitlendi', { amount: formatCurrency(r.cashout, curr) }) : _t('Bu gün {amount} kasaya geri alındı', { amount: formatCurrency(-r.cashout, curr) })}"` : ''}>${(r.secured > 0 || r.cashout) && (r.actualBank != null || r.day <= lastCashDay) ? formatCurrency(r.secured, curr) + (r.cashout ? ` <span class="ks-co">${r.cashout > 0 ? '+' : '−'}${formatAmount(Math.abs(r.cashout))}</span>` : '') : ''}</td>` : ''}
                  <td>${r.dailyChange != null ? formatCurrency(r.dailyChange, curr) : ''}</td>
                  <td class="${r.totalGrowthPct == null ? '' : r.totalGrowthPct >= 0 ? 'good' : 'bad'}">${r.totalGrowthPct != null ? signedPct(r.totalGrowthPct) : ''}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
        <div class="ks-note">${_t('kasa.note')}${hasCash ? ' ' + _t('kasa.note.cashout') : ''}</div>
      </div>
      ${renderClosedKasasHtml(curr)}
    `;

    wirePlanDashboardEvents();

    const newWrap = pane.querySelector('.ks-table-wrap');
    if (newWrap && keep.wrapTop != null) newWrap.scrollTop = keep.wrapTop;
    if (root.scrollY !== keep.winY) root.scrollTo(root.scrollX, keep.winY);
    const newAnchor = anchorSel ? pane.querySelector(anchorSel) : null;
    if (newAnchor && keep.anchorTop != null) {
      const delta = newAnchor.getBoundingClientRect().top - keep.anchorTop;
      if (Math.abs(delta) > 0.5) root.scrollTo(root.scrollX, root.scrollY + delta);
    }
    if (keep.focusDay) focusKasaInput(keep.focusDay);
  }

  // Gerçek kasa hücresine odaklanır; hücre tablo kutusunun dışındaysa yalnızca tablo kayar, sayfa kaymaz
  function focusKasaInput(day) {
    const f = document.querySelector(`#kasaSimCard .kasa-input[data-day="${day}"]`);
    if (!f) return false;
    f.focus({ preventScroll: true });
    const wrap = f.closest('.ks-table-wrap');
    if (wrap) {
      const fr = f.getBoundingClientRect(), wr = wrap.getBoundingClientRect();
      const head = (wrap.querySelector('thead') || { offsetHeight: 0 }).offsetHeight;
      if (fr.bottom > wr.bottom - 4) wrap.scrollTop += fr.bottom - wr.bottom + 8;
      else if (fr.top < wr.top + head + 4) wrap.scrollTop -= wr.top + head - fr.top + 8;
    }
    try { f.select(); } catch (e) {}
    return true;
  }

  // Dile göre binlik/ondalık: en "1,234.56" · tr/nl "1.234,56". Tek ayırıcı ve ≤2 hane ("68.66" / "68,66") her dilde ondalıktır.
  function parseKasaAmount(text) {
    let raw = String(text || '').trim().replace(/[\s€₺$£]/g, '');
    if (raw === '') return '';
    const en = root.I18N && root.I18N.lang === 'en';
    const [thou, decSep] = en ? [',', '.'] : ['.', ','];
    if (/^-?\d*[.,]\d{1,2}$/.test(raw)) raw = raw.replace(',', '.');
    else raw = raw.split(thou).join('').replace(decSep, '.');
    const val = parseNumber(raw);
    return val == null || val < 0 ? null : val;
  }

  function rerenderAllPanes() {
    saveState();
    renderPlanPane();
    renderRecPane();
    renderCouponsPane();
  }

  function wirePlanDashboardEvents() {
    document.querySelectorAll('.bankroll-tab').forEach(tab => {
      tab.onclick = () => {
        const pId = tab.dataset.planId;
        if (pId && pId !== paperState.activePlanId) {
          PE.switchActivePlan(paperState, pId);
          rerenderAllPanes();
        }
      };
    });

    const btnNewBank = document.getElementById('btnAddNewPlan');
    if (btnNewBank) {
      btnNewBank.onclick = () => {
        const defaultName = _t('Kasa {n}', { n: paperState.plans.length + 1 });
        const entered = prompt(_t('Yeni kasanın adı:'), defaultName);
        if (entered === null) return; // Vazgeç
        PE.createNewPlan(paperState, {
          name: entered.trim().slice(0, 40) || defaultName,
          startingBank: PE.KASA_V01_EXAMPLE.startingBank,
          targetBank: PE.KASA_V01_EXAMPLE.targetBank,
          riskProfile: PE.KASA_V01_EXAMPLE.riskProfile
        });
        rerenderAllPanes();
      };
    }

    const btnDelBank = document.getElementById('btnDeleteCurrentPlan');
    if (btnDelBank) {
      btnDelBank.onclick = () => {
        const idx = paperState.plans.findIndex(pl => pl.id === paperState.activePlanId);
        const pName = kasaDisplayName(paperState.plan, idx);
        if (confirm(_t('"{name}" kasasını silmek istediğinize emin misiniz? Diğer kasalarınız korunacaktır.', { name: pName }))) {
          PE.deletePlan(paperState, paperState.activePlanId);
          rerenderAllPanes();
        }
      };
    }

    const btnClose = document.getElementById('btnCloseKasa');
    if (btnClose) {
      btnClose.onclick = () => {
        const plan = paperState.plan;
        if (!plan) return;
        const curr = (paperState.settings && paperState.settings.currency) || 'EUR';
        const sim = PE.buildKasaSimulation(plan, paperState);
        const idx = paperState.plans.findIndex(pl => pl.id === plan.id);
        const name = kasaDisplayName(plan, idx);
        if (!confirm(_t('kasa.closeConfirm', { name, start: formatCurrency(plan.startingBank, curr), final: formatCurrency(sim.currentBank, curr) }))) return;
        if (name !== plan.name) PE.updatePlanInputs(paperState, { name });   // listede ekranda görünen ad kalsın
        const rec = PE.closePlan(paperState, plan.id);
        if (rec && !paperState.plans.length) openBlankKasaAfter(rec);
        rerenderAllPanes();
      };
    }

    const btnXls = document.getElementById('btnExportKasaXlsx');
    if (btnXls) {
      btnXls.onclick = () => {
        const plan = paperState.plan;
        if (!plan) return;
        const idx = paperState.plans.findIndex(pl => pl.id === plan.id);
        exportKasaExcel({ ...plan, name: kasaDisplayName(plan, idx) }, PE.buildKasaSimulation(plan, paperState), null);
      };
    }

    document.querySelectorAll('.ck-reopen').forEach(b => {
      b.onclick = () => {
        const c = (paperState.closedPlans || []).find(x => x.id === b.dataset.closedId);
        if (!c || !confirm(_t('"{name}" kasası yeniden açılsın mı? Girdileri düzeltip tekrar kapatabilirsin.', { name: c.name }))) return;
        PE.reopenClosedPlan(paperState, c.id);
        rerenderAllPanes();
        root.scrollTo(0, 0);
      };
    });
    document.querySelectorAll('.ck-del').forEach(b => {
      b.onclick = () => {
        const c = (paperState.closedPlans || []).find(x => x.id === b.dataset.closedId);
        if (!c || !confirm(_t('"{name}" kapatılan kasası kalıcı olarak silinsin mi? Bu işlem geri alınamaz.', { name: c.name }))) return;
        PE.deleteClosedPlan(paperState, c.id);
        rerenderAllPanes();
      };
    });

    document.querySelectorAll('.ck-xls:not(.ck-reopen):not(.ck-del)').forEach(b => {
      b.onclick = () => {
        const c = (paperState.closedPlans || []).find(x => x.id === b.dataset.closedId);
        if (!c || !c.plan) return;
        exportKasaExcel({ ...c.plan, name: c.name }, PE.buildKasaSimulation(c.plan, paperState, new Date(c.closedAt)), c);
      };
    });

    wireEICoachEvents();

    // KULLANICI GİRİŞLERİ
    const nameInp = document.getElementById('ksName');
    if (nameInp) {
      nameInp.onchange = () => {
        // Boş bırakılırsa "Kasa N" varsayılan adına döner
        PE.updatePlanInputs(paperState, { name: nameInp.value });
        rerenderAllPanes();
      };
    }

    const bindAmount = (id, key, label) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.onchange = () => {
        const val = parseKasaAmount(el.value);
        if (val === '' || val == null || val <= 0) {
          alert(_t("{label} 0'dan büyük olmalıdır.", { label }));
          renderPlanPane();
          return;
        }
        PE.updatePlanInputs(paperState, { [key]: val });
        rerenderAllPanes();
      };
    };
    bindAmount('ksStart', 'startingBank', _t('Başlangıç kasası'));
    bindAmount('ksTarget', 'targetBank', _t('Hedef kasa'));
    const riskSel = document.getElementById('ksRisk');
    if (riskSel) {
      riskSel.onchange = () => {
        PE.updatePlanInputs(paperState, { riskProfile: riskSel.value });
        rerenderAllPanes();
      };
    }

    // GERÇEK KASA girişleri: değişince kaydedilir; Tab / Enter sonraki güne, Shift+Tab önceki güne geçer
    const commitKasaInput = (inp, focusDay) => {
      const val = parseKasaAmount(inp.value);
      if (val == null) {
        alert(_t('Geçerli bir kasa tutarı girin (örn: 68,66). Boş bırakırsanız gün boş kalır.'));
        renderPlanPane({ anchor: '#kasaSimCard', focusDay: inp.dataset.day });
        return;
      }
      PE.setPlanDailyBank(paperState, inp.dataset.day, val === '' ? null : val);
      saveState();
      renderPlanPane({ anchor: '#kasaSimCard', focusDay });
    };
    document.querySelectorAll('#kasaSimCard .kasa-input').forEach(inp => {
      inp.onchange = () => {
        if (inp._committed) return;
        commitKasaInput(inp, null);
      };
      inp.onkeydown = e => {
        if (e.key !== 'Tab' && e.key !== 'Enter') return;
        const day = Number(inp.dataset.day);
        const next = e.shiftKey ? day - 1 : day + 1;
        const hasNext = !!document.querySelector(`#kasaSimCard .kasa-input[data-day="${next}"]`);
        if (!hasNext && e.key === 'Tab') return;          // tablo sonu: normal Tab davranışı
        e.preventDefault();
        const target = hasNext ? next : day;
        if (inp.value !== inp.defaultValue) {
          inp._committed = true;                          // yeniden çizimde change ikinci kez çalışmasın
          commitKasaInput(inp, target);
        } else {
          focusKasaInput(target);
        }
      };
    });
  }

  // ---------------------------------------------------------------------------
  // Kupon Önerileri Ekranı (#pane-rec)
  // ---------------------------------------------------------------------------

  function renderRecPane() {
    const pane = document.getElementById('pane-rec');
    if (!pane) return;

    const matches = window.__data || [];
    const rawProfKey = (paperState && paperState.settings && paperState.settings.riskProfile) || 'minimum';
    const profKey = (rawProfKey === 'cautious' ? 'minimum' : rawProfKey === 'balanced' ? 'medium' : rawProfKey === 'aggressive' ? 'high' : rawProfKey);
    const prof = PE.RISK_PROFILES[profKey] || PE.RISK_PROFILES.minimum;
    const available = (paperState && paperState.plan && paperState.plan.availableBalance) || 50;
    const curr = (paperState && paperState.settings && paperState.settings.currency) || 'EUR';

    const recs = PE.buildAllRecommendations(matches, profKey, available);

    pane.innerHTML = `
      <div class="paper-disclaimer">
        <span class="p-badge">SANAL KASA SİMÜLASYONU</span>
        <p><b>BETAVUS</b> bahis kabul etmez ve kupon oynatmaz. Önerilen kuponlar model olasılıklarına dayalı sanal simülasyonlardır. Sonuç garantisi vermez.</p>
        <div class="p-quote">« Önce simüle et. Riskini gör. Stratejini ölç. Sonra karar ver. »</div>
      </div>

      <div class="rec-header">
        <div>
          <h2>Kişiselleştirilmiş Kupon Önerileri</h2>
          <p>Seçili Kasa Modeliniz: <b>${esc(prof.name)}</b> (%${Math.round(prof.reservePct * 100)} Rezervde · Kullanılabilir: ${formatCurrency(available, curr)})</p>
          <div style="margin-top:6px;font-size:11.5px;color:var(--muted);line-height:1.5;">
            💡 Kupon maçları öncelikle <b>önümüzdeki 1 haftalık</b> fikstürlerden seçilir. 1 haftalık programda yeterli güvene sahip maç bulunamadığında sistem otomatik olarak <b>önümüzdeki 1 aylık</b> analiz havuzuna genişletir.
          </div>
        </div>
        <div class="rec-actions">
          <button class="btn-sec" id="btnRecChangeRisk" type="button">⚙️ Kasa Modelini Değiştir</button>
        </div>
      </div>

      <div class="rec-cards-grid">
        ${Object.keys(PE.COUPON_CLASSES).sort((a, b) => (a === prof.id ? -1 : b === prof.id ? 1 : 0)).map(k => renderRecCardHtml(recs[k], k, curr, prof)).join('')}
      </div>
    `;

    wireRecEvents(recs);
  }

  function renderRecCardHtml(rec, classKey, curr, prof) {
    const cls = PE.COUPON_CLASSES[classKey];
    const armPct = Math.round((prof[cls.armKey] || 0) * 100);
    const isPlanProfile = (classKey === prof.id);

    const planBadgeHtml = isPlanProfile ? `
      <div style="display:inline-flex;align-items:center;gap:6px;background:rgba(232,255,63,0.12);border:1px solid rgba(232,255,63,0.3);color:var(--accent);font-weight:800;font-size:11px;padding:4px 10px;border-radius:6px;margin-bottom:8px;">
        🎯 KASA PLANINIZA ÖZEL SEÇİLEN KUPON
      </div>
    ` : '';

    const aiBadgeHtml = classKey === 'minimum' ? `
      <div class="ai-badge safe" style="margin:10px 0;padding:8px 12px;background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);border-radius:8px;font-size:11.5px;color:#10b981;line-height:1.4;">
        🛡️ <b>Model Analizi: Yüksek Güven / Garanti Profil</b> — Tarihsel maç isabet oranı <b>%95.5</b> (Model yanılma oranı sadece <b>%4.5</b>). Minimum riskli 0.5 Üst maçlarından oluşur.
      </div>
    ` : classKey === 'medium' ? `
      <div class="ai-badge med" style="margin:10px 0;padding:8px 12px;background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.25);border-radius:8px;font-size:11.5px;color:#38bdf8;line-height:1.4;">
        ⚖️ <b>Model Analizi: Dengeli / Orta Risk Profil</b> — Tarihsel maç isabet oranı <b>%88.2</b> (Model yanılma oranı <b>%11.8</b>). 1.5 Üst odaklıdır.
      </div>
    ` : `
      <div class="ai-badge risky" style="margin:10px 0;padding:8px 12px;background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.25);border-radius:8px;font-size:11.5px;color:#f87171;line-height:1.4;">
        ⚠️ <b>Model Analizi: Yüksek Risk / Düşük Başarı Oranı Uyarısı</b> — 2.5 Üst maçlarda model yanılma oranı <b>%23.7</b>'ye kadar çıkar. Başarı oranı düşüktür; kupon tercihi, maç ekleme/çıkarma ve oluşacak kâr/zarar durumları tamamen sizin sorumluluğunuzdadır.
      </div>
    `;

    if (!rec || !rec.available) {
      return `
        <div class="card rec-card unavailable ${cls.id}" style="${isPlanProfile ? 'border:1.5px solid var(--accent);box-shadow:0 0 16px rgba(232,255,63,0.1);' : ''}">
          ${planBadgeHtml}
          <div class="rc-head">
            <span class="rc-badge ${cls.badgeClass}">${esc(cls.name)}</span>
            <span class="rc-arm-note">Kasadan ayrılan: %${armPct}</span>
          </div>
          ${aiBadgeHtml}
          <p class="rc-desc">${esc(cls.desc)}</p>
          <div class="empty-rec">
            <strong>${rec ? esc(rec.message) : 'Bugün uygun maç bulunamadı.'}</strong>
            <p>Eşikler model güvenini korumak için otomatik düşürülmez. Yeni fikstürler oluştukça güncellenir.</p>
          </div>
        </div>
      `;
    }

    const evPct = Math.round(rec.expectedValue * 100);
    const evClass = evPct > 0 ? 'good' : 'warn';

    return `
      <div class="card rec-card ${cls.id}" style="${isPlanProfile ? 'border:1.5px solid var(--accent);box-shadow:0 0 16px rgba(232,255,63,0.1);' : ''}">
        ${planBadgeHtml}
        <div class="rc-head">
          <div class="rc-title-box">
            <span class="rc-badge ${cls.badgeClass}">${esc(cls.name)}</span>
            <span class="badge" style="background:rgba(56,189,248,0.15);color:#38bdf8;font-size:10.5px;font-weight:700;padding:2px 7px;border-radius:6px;">🗓️ ${esc(rec.windowLabel || 'Önümüzdeki 1 Hafta')}</span>
            <span class="rc-legs-count">${rec.selections.length} Maç</span>
          </div>
          <span class="rc-arm-note">Risk Kolu: %${armPct} (${formatCurrency(rec.recommendedStake, curr)})</span>
        </div>

        ${aiBadgeHtml}

        <p class="rc-desc">${esc(cls.desc)}</p>

        <div class="tbl-scroll">
          <table class="rc-table">
            <thead>
              <tr><th>Tarih</th><th>Lig</th><th>Karşılaşma</th><th>Seçim</th><th>Model</th><th>Dayanak</th></tr>
            </thead>
            <tbody>
              ${rec.selections.map(s => `
                <tr>
                  <td>${dmy(s.kickoffUtc)}<div class="sub">${timeStr(s.kickoffUtc)}</div></td>
                  <td>${flag(s.league, 10)} ${esc(s.league)}</td>
                  <td class="mL">${esc(s.home)} — ${esc(s.away)}</td>
                  <td><b>${esc(s.line)} Üst</b></td>
                  <td><b>%${Math.round(s.probability * 100)}</b></td>
                  <td><span class="badge ${s.basis === 'form+h2h' ? 'b3' : 'b2'}">${esc(s.basis)}</span></td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>

        <div class="rc-summary-bar">
          <div class="s-item">
            <span class="lbl">Birleşik Tutma İhtimali</span>
            <span class="val">%${Math.round(rec.combinedProbability * 100)}</span>
          </div>
          <div class="s-item">
            <span class="lbl">Tahmini Toplam Oran</span>
            <span class="val">${formatOdds(rec.estimatedOdds)}</span>
          </div>
          <div class="s-item">
            <span class="lbl">Beklenen Değer (EV)</span>
            <span class="val ${evClass}">%+${evPct}%</span>
          </div>
          <div class="s-item">
            <span class="lbl">Tahmini Geri Dönüş</span>
            <span class="val good">${formatCurrency(rec.potentialReturn, curr)}</span>
          </div>
        </div>

        <div class="rc-footer-actions">
          <button class="btn-sec btn-edit-rec" data-key="${cls.id}" type="button">✏️ İncele &amp; Düzenle</button>
          <button class="btn-primary btn-add-plan-rec" data-key="${cls.id}" type="button">🚀 Plana Ekle (${formatCurrency(rec.recommendedStake, curr)})</button>
          <button class="btn-subtle btn-draft-rec" data-key="${cls.id}" type="button">📋 Taslak Kaydet</button>
        </div>
      </div>
    `;
  }

  function wireRecEvents(recs) {
    document.querySelectorAll('.btn-edit-rec').forEach(b => {
      b.onclick = () => {
        const key = b.dataset.key;
        const rec = recs[key];
        if (!rec) return;
        openCouponEditor(rec);
      };
    });

    document.querySelectorAll('.btn-add-plan-rec').forEach(b => {
      b.onclick = () => {
        if (!paperState || !paperState.plan) {
          alert('Önce "Kasa Planım" sekmesinden sanal kasa planınızı oluşturmalısınız.');
          if (typeof root.setTab === 'function') root.setTab('plan');
          return;
        }
        const key = b.dataset.key;
        const rec = recs[key];
        if (!rec) return;

        const slip = PE.createSlipFromSelections(rec.selections, {
          source: 'recommended',
          riskProfile: paperState.settings.riskProfile,
          couponClass: rec.couponClass,
          stake: rec.recommendedStake
        });

        const res = PE.addSlipToPlan(paperState, slip);
        if (!res.success) {
          alert('Hata: ' + res.error);
          return;
        }

        paperState = res.state;
        saveState();
        alert(`Kupon başarıyla plana eklendi! ${formatCurrency(slip.stake, paperState.settings.currency)} bakiye ayrıldı.`);
        renderPlanPane();
        renderRecPane();
        renderCouponsPane();
        if (typeof root.setTab === 'function') root.setTab('cpn');
      };
    });

    document.querySelectorAll('.btn-draft-rec').forEach(b => {
      b.onclick = () => {
        if (!paperState) {
          paperState = PE.createInitialState();
        }
        const key = b.dataset.key;
        const rec = recs[key];
        if (!rec) return;

        const slip = PE.createSlipFromSelections(rec.selections, {
          source: 'recommended',
          riskProfile: paperState.settings.riskProfile,
          couponClass: rec.couponClass,
          stake: rec.recommendedStake,
          status: 'draft'
        });

        paperState.slips = [slip, ...paperState.slips];
        saveState();
        alert('Kupon taslak olarak Kuponlarım sekmesine kaydedildi.');
        renderCouponsPane();
      };
    });

    const btnChangeRisk = document.getElementById('btnRecChangeRisk');
    if (btnChangeRisk) {
      btnChangeRisk.onclick = () => {
        const rawCur = (paperState && paperState.settings.riskProfile) || 'minimum';
        const cur = (rawCur === 'cautious' ? 'minimum' : rawCur === 'balanced' ? 'medium' : rawCur === 'aggressive' ? 'high' : rawCur);
        const next = cur === 'minimum' ? 'medium' : cur === 'medium' ? 'high' : cur === 'high' ? 'multi' : 'minimum';
        if (confirm(`Kasa modelinizi "${PE.RISK_PROFILES[next].name}" olarak değiştirmek istiyor musunuz?`)) {
          if (!paperState) paperState = PE.createInitialState();
          paperState.settings.riskProfile = next;
          saveState();
          renderRecPane();
          renderPlanPane();
        }
      };
    }
  }

  // ---------------------------------------------------------------------------
  // İnteraktif Kupon Düzenleyici (Interactive Coupon Editor Modal)
  // ---------------------------------------------------------------------------

  function openCouponEditor(slipOrRec) {
    const isNew = !slipOrRec.id || slipOrRec.id.startsWith('rec-');
    editingSlip = {
      id: isNew ? `slip-${Date.now().toString(36)}` : slipOrRec.id,
      source: slipOrRec.source || (isNew ? 'recommended' : 'user'),
      riskProfile: (paperState && paperState.settings && paperState.settings.riskProfile) || slipOrRec.riskProfile || 'minimum',
      couponClass: slipOrRec.couponClass || 'medium',
      stake: slipOrRec.stake != null ? Number(slipOrRec.stake) : (slipOrRec.recommendedStake || 10),
      actualOdds: slipOrRec.actualOdds || null,
      status: slipOrRec.status || 'draft',
      selections: (slipOrRec.selections || []).map(s => ({ ...s }))
    };

    renderCouponEditorModal();
  }

  function renderCouponEditorModal() {
    let modal = document.getElementById('couponEditorModal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'couponEditorModal';
      modal.className = 'modal-bg';
      document.body.appendChild(modal);
    }
    modal.hidden = false;

    const s = editingSlip;
    const combinedProb = PE.calculateCombinedProbability(s.selections);
    const estOdds = PE.calculateEstimatedOdds(s.selections, s.couponClass);
    const actualOdds = s.actualOdds ? Number(s.actualOdds) : null;
    const oddsUsed = actualOdds && actualOdds > 1.0 ? actualOdds : estOdds;
    const breakEvenProb = PE.calculateBreakEvenProbability(oddsUsed);
    const ev = PE.calculateExpectedValue(combinedProb, oddsUsed);
    const potReturn = PE.calculatePotentialReturn(s.stake, oddsUsed);
    const potNet = PE.calculatePotentialNet(s.stake, oddsUsed);

    const curr = (paperState && paperState.settings && paperState.settings.currency) || 'EUR';
    const avail = (paperState && paperState.plan && paperState.plan.availableBalance) || 0;

    modal.innerHTML = `
      <div class="modal-box editor-box">
        <div class="modal-head">
          <h3>✏️ Kuponu İncele &amp; Düzenle</h3>
          <button class="modal-close" id="btnCloseEditor">×</button>
        </div>

        <div class="editor-body">
          <div class="editor-selections">
            <div class="es-head">
              <b>Kupon Bacakları (${s.selections.length})</b>
              <button class="btn-subtle" id="btnAddMatchToEditor" type="button">+ Maç Ekle</button>
            </div>

            ${!s.selections.length ? '<div class="empty-msg">Kuponda maç kalmadı. Lütfen "+ Maç Ekle" ile maç ekleyin.</div>' : ''}

            <div class="es-list">
              ${s.selections.map((sel, idx) => `
                <div class="es-item">
                  <div class="es-info">
                    <div class="es-teams">${flag(sel.league, 10)} <b>${esc(sel.home)} — ${esc(sel.away)}</b></div>
                    <div class="es-sub">${esc(sel.league)} · ${dmy(sel.kickoffUtc)} ${timeStr(sel.kickoffUtc)} · Dayanak: <span class="badge ${sel.basis === 'form+h2h' ? 'b3' : 'b2'}">${esc(sel.basis)}</span></div>
                  </div>
                  <div class="es-market-ctrl">
                    <select class="market-select" data-idx="${idx}">
                      <option value="over_0_5" ${sel.market === 'over_0_5' ? 'selected' : ''}>0.5 Üst (%${Math.round((sel.probability || 0.95) * 100)})</option>
                      <option value="over_1_5" ${sel.market === 'over_1_5' ? 'selected' : ''}>1.5 Üst</option>
                      <option value="over_2_5" ${sel.market === 'over_2_5' ? 'selected' : ''}>2.5 Üst</option>
                    </select>
                    <button class="btn-remove-leg" data-idx="${idx}" type="button" title="Maçı Çıkar">🗑️</button>
                  </div>
                </div>
              `).join('')}
            </div>
          </div>

          <div class="editor-sidebar">
            <div class="form-group">
              <label>Kupon Sınıfı</label>
              <select id="edCouponClass" class="form-select">
                <option value="minimum" ${s.couponClass === 'minimum' ? 'selected' : ''}>Minimum Risk (0.5 Üst)</option>
                <option value="medium" ${s.couponClass === 'medium' ? 'selected' : ''}>Orta Risk (1.5 Üst)</option>
                <option value="high" ${s.couponClass === 'high' ? 'selected' : ''}>Yüksek Risk (2.5 Üst)</option>
              </select>
            </div>

            <div class="form-group" style="margin-top:10px;">
              <label>Sanal Stake Tutarı (${curr})</label>
              <input type="text" id="edStake" class="form-input" value="${s.stake}" autocomplete="off">
              <span class="form-hint">Kullanılabilir Sanal Bakiye: ${formatCurrency(avail, curr)}</span>
            </div>

            <div class="form-group" style="margin-top:10px;">
              <label>Gerçek Toplam Oran (Opsiyonel)</label>
              <input type="text" id="edActualOdds" class="form-input" placeholder="Örn: 1.65" value="${s.actualOdds || ''}" autocomplete="off">
              <span class="form-hint">Harici siteden aldığınız toplam kupon oranı varsa girin.</span>
            </div>

            <div class="ai-editor-feedback" style="margin-top:10px;margin-bottom:10px;">
              ${s.couponClass === 'minimum' ? `
                <div class="ai-badge safe" style="padding:8px 10px;background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);border-radius:6px;font-size:11px;color:#10b981;line-height:1.35;">
                  🛡️ <b>Model Güveni: Yüksek / Garanti Profil</b><br>
                  Model bu kupondaki maçları %${Math.round(combinedProb * 100)} birleşik tutma olasılığıyla değerlendirdi (Tarihsel hata oranı sadece %4.5).
                </div>
              ` : s.couponClass === 'medium' ? `
                <div class="ai-badge med" style="padding:8px 10px;background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.25);border-radius:6px;font-size:11px;color:#38bdf8;line-height:1.35;">
                  ⚖️ <b>Model Güveni: Dengeli Risk Profili</b><br>
                  Model birleşik tutma olasılığı: %${Math.round(combinedProb * 100)} (Tarihsel hata oranı %11.8).
                </div>
              ` : `
                <div class="ai-badge risky" style="padding:8px 10px;background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.25);border-radius:6px;font-size:11px;color:#f87171;line-height:1.35;">
                  ⚠️ <b>Model Uyarısı: Düşük Başarı / Yüksek Yanılma Riski</b><br>
                  Model birleşik tutma olasılığı %${Math.round(combinedProb * 100)}. 2.5 Üst seçimlerinde başarı oranı düşüktür; kupondaki maç tercihleri ve risk tamamen size aittir.
                </div>
              `}
            </div>

            <div class="ed-calc-box">
              <div class="ed-calc-row"><span>Birleşik Olasılık:</span><b>%${Math.round(combinedProb * 100)}</b></div>
              <div class="ed-calc-row"><span>Tahmini Oran:</span><b>${formatOdds(estOdds)}</b></div>
              ${actualOdds ? `<div class="ed-calc-row"><span>Girdiğiniz Gerçek Oran:</span><b class="good">${formatOdds(actualOdds)}</b></div>` : ''}
              <div class="ed-calc-row"><span>Başa Baş Olasılık:</span><b>%${Math.round(breakEvenProb * 100)}</b></div>
              <div class="ed-calc-row"><span>Beklenen Değer (EV):</span><b class="${ev > 0 ? 'good' : 'warn'}">%${Math.round(ev * 100)}</b></div>
              <div class="ed-calc-row total"><span>Potansiyel Geri Dönüş:</span><b class="good">${formatCurrency(potReturn, curr)}</b></div>
              <div class="ed-calc-row"><span>Potansiyel Net Büyüme:</span><b>+${formatCurrency(potNet, curr)}</b></div>
            </div>

            <div id="edError" class="form-error" hidden></div>

            <div class="ed-buttons">
              <button class="btn-primary" id="btnEdAddToPlan" type="button" style="width:100%">🚀 Plana Ekle</button>
              <button class="btn-sec" id="btnEdSaveDraft" type="button" style="width:100%;margin-top:6px;">📋 Taslak Olarak Kaydet</button>
            </div>
          </div>
        </div>
      </div>
    `;

    wireCouponEditorEvents();
  }

  function wireCouponEditorEvents() {
    const modal = document.getElementById('couponEditorModal');
    const closeBtn = document.getElementById('btnCloseEditor');
    if (closeBtn && modal) {
      closeBtn.onclick = () => { modal.hidden = true; };
    }

    // Market değiştirme
    document.querySelectorAll('.market-select').forEach(sel => {
      sel.onchange = (e) => {
        const idx = parseInt(sel.dataset.idx, 10);
        const mkt = e.target.value;
        const line = mkt === 'over_0_5' ? '0.5' : mkt === 'over_1_5' ? '1.5' : '2.5';
        if (editingSlip.selections[idx]) {
          const matchOrig = (window.__data || []).find(m => m.match_id === editingSlip.selections[idx].matchId);
          const prob = matchOrig ? PE.getMarketProbability(matchOrig, mkt) : (mkt === 'over_0_5' ? 0.95 : mkt === 'over_1_5' ? 0.85 : 0.75);
          editingSlip.selections[idx].market = mkt;
          editingSlip.selections[idx].line = line;
          editingSlip.selections[idx].probability = prob;
          editingSlip.selections[idx].estimatedLegOdds = PE.round(1.0 / (prob || 0.95), 2);
          renderCouponEditorModal();
        }
      };
    });

    // Maçı çıkar
    document.querySelectorAll('.btn-remove-leg').forEach(btn => {
      btn.onclick = () => {
        const idx = parseInt(btn.dataset.idx, 10);
        editingSlip.selections.splice(idx, 1);
        renderCouponEditorModal();
      };
    });

    // Maç Ekle
    const btnAddMatch = document.getElementById('btnAddMatchToEditor');
    if (btnAddMatch) {
      btnAddMatch.onclick = () => {
        openMatchPickerModal((selectedMatch) => {
          const exists = editingSlip.selections.some(s => s.matchId === selectedMatch.match_id);
          if (exists) {
            alert('Bu maç zaten kuponda yer alıyor.');
            return;
          }
          const mkt = editingSlip.couponClass === 'minimum' ? 'over_0_5' : editingSlip.couponClass === 'medium' ? 'over_1_5' : 'over_2_5';
          const line = mkt === 'over_0_5' ? '0.5' : mkt === 'over_1_5' ? '1.5' : '2.5';
          const prob = PE.getMarketProbability(selectedMatch, mkt) || 0.85;
          editingSlip.selections.push({
            matchId: selectedMatch.match_id,
            league: selectedMatch.league,
            kickoffUtc: selectedMatch.kickoff_utc,
            home: selectedMatch.home,
            away: selectedMatch.away,
            market: mkt,
            line,
            probability: PE.round(prob, 4),
            basis: selectedMatch.basis || 'form',
            marketOdds: null,
            estimatedLegOdds: PE.round(1.0 / (prob || 0.95), 2),
            result: 'pending',
            score: null
          });
          renderCouponEditorModal();
        });
      };
    }

    // Dinamik inputlar (Stake & Gerçek Oran)
    const stakeInput = document.getElementById('edStake');
    if (stakeInput) {
      stakeInput.oninput = () => {
        const val = parseNumber(stakeInput.value);
        if (val != null && val >= 0) editingSlip.stake = val;
      };
    }

    const actualOddsInput = document.getElementById('edActualOdds');
    if (actualOddsInput) {
      actualOddsInput.oninput = () => {
        const val = parseNumber(actualOddsInput.value);
        editingSlip.actualOdds = val;
      };
    }

    const classSelect = document.getElementById('edCouponClass');
    if (classSelect) {
      classSelect.onchange = (e) => {
        editingSlip.couponClass = e.target.value;
        renderCouponEditorModal();
      };
    }

    // Plana Ekle
    const btnAddPlan = document.getElementById('btnEdAddToPlan');
    if (btnAddPlan) {
      btnAddPlan.onclick = () => {
        if (!paperState || !paperState.plan) {
          alert('Önce Kasa Planım sekmesinden bir plan oluşturmalısınız.');
          return;
        }
        editingSlip.stake = parseNumber(document.getElementById('edStake')?.value) || 0;
        editingSlip.actualOdds = parseNumber(document.getElementById('edActualOdds')?.value);

        const slip = PE.createSlipFromSelections(editingSlip.selections, {
          id: editingSlip.id,
          source: editingSlip.source || 'user',
          riskProfile: paperState.settings.riskProfile,
          couponClass: editingSlip.couponClass,
          stake: editingSlip.stake,
          actualOdds: editingSlip.actualOdds
        });

        const res = PE.addSlipToPlan(paperState, slip);
        if (!res.success) {
          const errEl = document.getElementById('edError');
          if (errEl) { errEl.textContent = res.error; errEl.hidden = false; }
          return;
        }

        paperState = res.state;
        saveState();
        if (modal) modal.hidden = true;
        alert('Kupon başarıyla plana eklendi!');
        renderPlanPane();
        renderRecPane();
        renderCouponsPane();
      };
    }

    // Taslak Kaydet
    const btnDraft = document.getElementById('btnEdSaveDraft');
    if (btnDraft) {
      btnDraft.onclick = () => {
        if (!paperState) paperState = PE.createInitialState();
        editingSlip.stake = parseNumber(document.getElementById('edStake')?.value) || 0;
        editingSlip.actualOdds = parseNumber(document.getElementById('edActualOdds')?.value);

        const slip = PE.createSlipFromSelections(editingSlip.selections, {
          id: editingSlip.id,
          source: editingSlip.source || 'user',
          riskProfile: paperState.settings.riskProfile,
          couponClass: editingSlip.couponClass,
          stake: editingSlip.stake,
          actualOdds: editingSlip.actualOdds,
          status: 'draft'
        });

        // Varsa güncelle, yoksa ekle
        const idx = paperState.slips.findIndex(s => s.id === slip.id);
        if (idx >= 0) {
          paperState.slips[idx] = slip;
        } else {
          paperState.slips = [slip, ...paperState.slips];
        }

        saveState();
        if (modal) modal.hidden = true;
        alert('Kupon taslak olarak kaydedildi.');
        renderCouponsPane();
      };
    }
  }

  // ---------------------------------------------------------------------------
  // Maç Seçici Modalı (Match Picker Modal)
  // ---------------------------------------------------------------------------

  function openMatchPickerModal(onSelect) {
    modalMatchPickerCallback = onSelect;
    let modal = document.getElementById('matchPickerModal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'matchPickerModal';
      modal.className = 'modal-bg';
      document.body.appendChild(modal);
    }
    modal.hidden = false;

    const matches = (window.__data || []).filter(m => {
      return new Date(m.kickoff_utc).getTime() > Date.now() - 30 * 60 * 1000;
    }).slice(0, 40);

    modal.innerHTML = `
      <div class="modal-box picker-box">
        <div class="modal-head">
          <h3>⚽ Listeden Maç Seç</h3>
          <button class="modal-close" id="btnClosePicker">×</button>
        </div>
        <div class="picker-search-bar">
          <input type="search" id="pickerSearch" class="form-input" placeholder="Takım veya lig ara…">
        </div>
        <div class="picker-list" id="pickerList">
          ${matches.map(m => `
            <div class="picker-item" data-mid="${esc(m.match_id)}">
              <div class="pi-info">
                <div class="pi-teams">${flag(m.league, 12)} <b>${esc(m.home)} — ${esc(m.away)}</b></div>
                <div class="pi-sub">${esc(m.league)} · ${dmy(m.kickoff_utc)} ${timeStr(m.kickoff_utc)}</div>
              </div>
              <div class="pi-probs">
                <span>0.5Ü: %${Math.round(m.p_over_0_5 * 100)}</span>
                <span>1.5Ü: %${Math.round(m.p_over_1_5 * 100)}</span>
                <span>2.5Ü: %${Math.round(m.p_over_2_5 * 100)}</span>
              </div>
              <button class="btn-sec btn-pick-match" data-mid="${esc(m.match_id)}" type="button">Seç</button>
            </div>
          `).join('')}
        </div>
      </div>
    `;

    const closeBtn = document.getElementById('btnClosePicker');
    if (closeBtn) closeBtn.onclick = () => { modal.hidden = true; };

    const searchInput = document.getElementById('pickerSearch');
    if (searchInput) {
      searchInput.oninput = (e) => {
        const q = String(e.target.value || '').toLowerCase();
        document.querySelectorAll('.picker-item').forEach(el => {
          const txt = el.textContent.toLowerCase();
          el.style.display = txt.includes(q) ? 'flex' : 'none';
        });
      };
    }

    document.querySelectorAll('.btn-pick-match').forEach(b => {
      b.onclick = () => {
        const mid = b.dataset.mid;
        const match = (window.__data || []).find(m => m.match_id === mid);
        if (match && typeof modalMatchPickerCallback === 'function') {
          modalMatchPickerCallback(match);
          modal.hidden = true;
        }
      };
    });
  }

  // ---------------------------------------------------------------------------
  // Kuponlarım Ekranı (#pane-cpn)
  // ---------------------------------------------------------------------------

  function renderCouponsPane() {
    const pane = document.getElementById('pane-cpn');
    if (!pane) return;

    const slips = (paperState && paperState.slips) || [];
    const drafts = slips.filter(s => s.status === 'draft');
    const pending = slips.filter(s => s.status === 'pending');
    const settled = slips.filter(s => s.status === 'won' || s.status === 'lost' || s.status === 'void');

    const curr = (paperState && paperState.settings && paperState.settings.currency) || 'EUR';
    const metrics = (paperState && PE.getPlanMetrics(paperState)) || {
      winRate: 0, wonCount: 0, settledCount: 0, totalStake: 0, netProfit: 0, roi: 0,
      longestWinStreak: 0, longestLossStreak: 0, maxDrawdownPct: 0
    };

    pane.innerHTML = `
      <div class="paper-disclaimer">
        <span class="p-badge">SANAL KASA SİMÜLASYONU</span>
        <p><b>BETAVUS</b> kupon oynatmaz veya ödeme almaz. Bu ekranda planınıza eklediğiniz sanal kuponların (paper slips) durumunu, oranlarını ve kasa hareketlerini takip edersiniz.</p>
      </div>

      <!-- Admin Kupon Takip Uyarısı -->
      <div class="admin-notice-banner" style="margin-bottom:14px;padding:12px 16px;background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);border-radius:10px;font-size:12px;color:#f59e0b;display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;">
        <div>
          <b>🛠️ Admin &amp; Canlı Model Takip Ekranı:</b>
          <span style="color:var(--text);margin-left:4px;">Bu sekme yöneticiler ve sistem denetimi için 12 Eylül model doğrulama verilerini ve aktif sanal kuponların anlık canlı skor mutabakatını sunar.</span>
        </div>
        <span class="badge" style="background:rgba(245,158,11,0.2);color:#f59e0b;font-weight:700;">Admin Modu</span>
      </div>

      <!-- Kuponlarım Alt Sekmeleri -->
      <div class="subtabs-bar" style="margin-bottom:14px;">
        <button class="subtab ${activeCpnSubtab === 'pending' ? 'active' : ''}" id="csub-pending" type="button">
          ⏳ Bekleyenler (${pending.length})
        </button>
        <button class="subtab ${activeCpnSubtab === 'settled' ? 'active' : ''}" id="csub-settled" type="button">
          ✅ Sonuçlananlar (${settled.length})
        </button>
        <button class="subtab ${activeCpnSubtab === 'drafts' ? 'active' : ''}" id="csub-drafts" type="button">
          📋 Taslaklar (${drafts.length})
        </button>
        <button class="subtab ${activeCpnSubtab === 'model12' ? 'active' : ''}" id="csub-model12" type="button">
          📊 12 Eylül Canlı Model Takibi
        </button>
      </div>

      <div id="cpnSubContent">
        ${activeCpnSubtab === 'pending' ? renderPendingSlipsHtml(pending, curr) :
          activeCpnSubtab === 'settled' ? renderSettledSlipsHtml(settled, metrics, curr) :
          activeCpnSubtab === 'drafts' ? renderDraftSlipsHtml(drafts, curr) :
          renderModel12Html()}
      </div>
    `;

    wireCouponsEvents();
  }

  function renderPendingSlipsHtml(slips, curr) {
    if (!slips.length) {
      return `<div class="empty"><strong>Bekleyen sanal kuponunuz bulunmuyor.</strong>Kupon Önerileri sekmesinden önerilen kuponları veya kendi kuponunuzu plana ekleyebilirsiniz.</div>`;
    }
    return `
      <div class="slips-list">
        ${slips.map(s => renderSlipCardHtml(s, curr)).join('')}
      </div>
    `;
  }

  function renderSettledSlipsHtml(slips, metrics, curr) {
    let filtered = slips;
    if (settledFilter === 'won') filtered = slips.filter(s => s.status === 'won');
    else if (settledFilter === 'lost') filtered = slips.filter(s => s.status === 'lost');
    else if (settledFilter === 'minimum') filtered = slips.filter(s => s.couponClass === 'minimum');
    else if (settledFilter === 'medium') filtered = slips.filter(s => s.couponClass === 'medium');
    else if (settledFilter === 'high') filtered = slips.filter(s => s.couponClass === 'high');

    return `
      <!-- Özet Metrikler -->
      <div class="tiles" style="grid-template-columns:repeat(4,1fr);margin-bottom:14px;">
        <div class="tile">
          <div class="v">%${metrics.winRate}</div>
          <div class="k">Kupon Tutma Oranı</div>
          <div class="n">${metrics.wonCount} Kazandı / ${metrics.settledCount} Kupon</div>
        </div>
        <div class="tile">
          <div class="v">${formatCurrency(metrics.totalStake, curr)}</div>
          <div class="k">Toplam Sanal Stake</div>
          <div class="n">Sonuçlanan kuponların toplamı</div>
        </div>
        <div class="tile">
          <div class="v ${metrics.netProfit >= 0 ? 'good' : 'bad'}">${metrics.netProfit >= 0 ? '+' : ''}${formatCurrency(metrics.netProfit, curr)}</div>
          <div class="k">Sanal Net Sonuç</div>
          <div class="n">Kâr/Zarar (ROI: ${metrics.roi >= 0 ? '+' : ''}%${metrics.roi})</div>
        </div>
        <div class="tile">
          <div class="v">${metrics.longestWinStreak} G / ${metrics.longestLossStreak} M</div>
          <div class="k">En Uzun Seriler</div>
          <div class="n">Maksimum Kasa Düşüşü: %${metrics.maxDrawdownPct}</div>
        </div>
      </div>

      <!-- Filtre Çipleri -->
      <nav class="filters" style="margin:0 0 14px 0;">
        <button class="chip ${settledFilter === 'all' ? 'active' : ''}" data-sf="all">Tümü (${slips.length})</button>
        <button class="chip ${settledFilter === 'won' ? 'active' : ''}" data-sf="won">Kazananlar (${slips.filter(s => s.status === 'won').length})</button>
        <button class="chip ${settledFilter === 'lost' ? 'active' : ''}" data-sf="lost">Kaybedenler (${slips.filter(s => s.status === 'lost').length})</button>
        <button class="chip ${settledFilter === 'minimum' ? 'active' : ''}" data-sf="minimum">Minimum Risk</button>
        <button class="chip ${settledFilter === 'medium' ? 'active' : ''}" data-sf="medium">Orta Risk</button>
        <button class="chip ${settledFilter === 'high' ? 'active' : ''}" data-sf="high">Yüksek Risk</button>
      </nav>

      ${!filtered.length ? `<div class="empty"><strong>Bu filtrede sonuçlanan kupon bulunamadı.</strong></div>` : `
        <div class="slips-list">
          ${filtered.map(s => renderSlipCardHtml(s, curr)).join('')}
        </div>
      `}
    `;
  }

  function renderDraftSlipsHtml(slips, curr) {
    if (!slips.length) {
      return `<div class="empty"><strong>Kayıtlı taslak kuponunuz yok.</strong>Kupon Önerileri sekmesinden kupon düzenleyip taslak olarak kaydedebilirsiniz.</div>`;
    }
    return `
      <div class="slips-list">
        ${slips.map(s => renderSlipCardHtml(s, curr, true)).join('')}
      </div>
    `;
  }

  function renderModel12Html() {
    // 12 Eylül canlı model takibi — önceki oturumdaki otomatik bacak analizi
    return `
      <div class="model12-box">
        <div id="cpnSummary"></div>
        <div class="datectl">
          <span class="srch">
            <svg width="13" height="13" viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="4.5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M11 11l3.5 3.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
            <input id="qCpn" type="search" placeholder="Takım ara…" autocomplete="off" spellcheck="false">
            <button class="sx" id="qCpnX" type="button" hidden aria-label="Temizle">×</button>
          </span>
        </div>
        <div id="cpnList"></div>
      </div>
    `;
  }

  function renderSlipCardHtml(slip, curr, isDraft = false) {
    const cls = PE.COUPON_CLASSES[slip.couponClass] || PE.COUPON_CLASSES.medium;
    const statusText = slip.status === 'won' ? 'KAZANDI' : slip.status === 'lost' ? 'KAYBETTİ' : slip.status === 'void' ? 'İADE' : isDraft ? 'TASLAK' : 'BEKLİYOR';
    const statusBadge = slip.status === 'won' ? 'won' : slip.status === 'lost' ? 'lost' : slip.status === 'void' ? 'void' : isDraft ? 'draft' : 'pending';

    const potReturn = PE.calculatePotentialReturn(slip.stake, slip.oddsUsed);
    const net = slip.status === 'won' ? PE.round(potReturn - slip.stake, 2) : slip.status === 'lost' ? -slip.stake : 0;

    return `
      <div class="cpn slip-card ${slip.status}">
        <div class="cpn-head">
          <div class="sc-title">
            <span class="cbadge ${statusBadge}">${statusText}</span>
            <span class="rc-badge ${cls.badgeClass}">${esc(cls.name)}</span>
            <span class="sc-date">${dmy(slip.createdAt)}</span>
            <span class="sc-source">${slip.source === 'recommended' ? '🤖 BETAVUS Önerisi' : '👤 Kullanıcı Kuponu'}</span>
          </div>
          <div class="sc-odds-info">
            <span>Tahmini: <b>${formatOdds(slip.estimatedOdds)}</b></span>
            ${slip.actualOdds ? `<span>Gerçek: <b class="good">${formatOdds(slip.actualOdds)}</b></span>` : ''}
            <span>Kullanılan: <b>${formatOdds(slip.oddsUsed)}</b></span>
          </div>
        </div>

        <div class="tbl-scroll">
          <table class="cpn-tbl">
            <thead>
              <tr><th>Tarih</th><th>Lig</th><th>Karşılaşma</th><th>Seçim</th><th>Model</th><th>Skor</th><th>Durum</th></tr>
            </thead>
            <tbody>
              ${slip.selections.map(sel => `
                <tr>
                  <td>${dmy(sel.kickoffUtc)}</td>
                  <td>${flag(sel.league, 10)} ${esc(sel.league)}</td>
                  <td class="mL">${esc(sel.home)} — ${esc(sel.away)}</td>
                  <td><b>${esc(sel.line)} Üst</b></td>
                  <td>%${Math.round(sel.probability * 100)}</td>
                  <td><b>${sel.score || '—'}</b></td>
                  <td>
                    <span class="cbadge ${sel.result === 'won' ? 'won' : sel.result === 'lost' ? 'lost' : 'pending'}">
                      ${sel.result === 'won' ? '✓ TUTTU' : sel.result === 'lost' ? '✗ ISKA' : 'BEKLİYOR'}
                    </span>
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>

        <div class="sc-foot-bar">
          <div class="sf-financials">
            <span>Sanal Stake: <b>${formatCurrency(slip.stake, curr)}</b></span>
            <span>Potansiyel Dönüş: <b>${formatCurrency(potReturn, curr)}</b></span>
            ${slip.bankBefore != null ? `<span>Önceki Kasa: ${formatCurrency(slip.bankBefore, curr)}</span>` : ''}
            ${slip.bankAfter != null ? `<span>Sonraki Kasa: <b>${formatCurrency(slip.bankAfter, curr)}</b></span>` : ''}
            ${slip.status === 'won' || slip.status === 'lost' ? `
              <span class="net-res ${net >= 0 ? 'good' : 'bad'}">Net: ${net >= 0 ? '+' : ''}${formatCurrency(net, curr)}</span>
            ` : ''}
          </div>

          <div class="sf-actions">
            ${isDraft ? `
              <button class="btn-sec btn-slip-edit" data-id="${slip.id}" type="button">✏️ Düzenle</button>
              <button class="btn-primary btn-slip-add-plan" data-id="${slip.id}" type="button">🚀 Plana Ekle</button>
              <button class="btn-danger-subtle btn-slip-del" data-id="${slip.id}" type="button">🗑️ Sil</button>
            ` : ''}
          </div>
        </div>
      </div>
    `;
  }

  function wireModel12SubtabEvents() {
    if (typeof root.renderCoupons === 'function') {
      root.renderCoupons();
    }

    const sumEl = document.getElementById('cpnSummary');
    if (sumEl) {
      sumEl.onclick = (e) => {
        const tile = e.target.closest('.tile[data-f]');
        if (tile) {
          const f = tile.dataset.f;
          if (typeof root.setCouponsFilter === 'function') {
            root.setCouponsFilter(f);
          } else if (typeof root.renderCoupons === 'function') {
            root.cpnFilter = f;
            root.renderCoupons();
          }
        }
      };
    }

    const listEl = document.getElementById('cpnList');
    if (listEl) {
      listEl.onclick = (e) => {
        const chip = e.target.closest('.chip[data-f]');
        if (chip) {
          const f = chip.dataset.f;
          if (typeof root.setCouponsFilter === 'function') {
            root.setCouponsFilter(f);
          } else if (typeof root.renderCoupons === 'function') {
            root.cpnFilter = f;
            root.renderCoupons();
          }
          return;
        }
        const row = e.target.closest('.cpn-row');
        if (row && typeof root.openH2HSheet === 'function') {
          root.openH2HSheet(row.dataset.league, row.dataset.home, row.dataset.away, row.dataset.ko);
        }
      };
    }

    const qInput = document.getElementById('qCpn');
    const qClear = document.getElementById('qCpnX');
    if (qInput) {
      qInput.oninput = () => {
        if (typeof root.setCouponsSearch === 'function') {
          root.setCouponsSearch(qInput.value.trim());
        } else {
          root.qCpn = qInput.value.trim();
          if (typeof root.renderCoupons === 'function') root.renderCoupons();
        }
        if (qClear) qClear.hidden = !qInput.value;
      };
    }
    if (qClear && qInput) {
      qClear.onclick = () => {
        qInput.value = '';
        if (typeof root.setCouponsSearch === 'function') {
          root.setCouponsSearch('');
        } else {
          root.qCpn = '';
          if (typeof root.renderCoupons === 'function') root.renderCoupons();
        }
        qClear.hidden = true;
        qInput.focus();
      };
    }
  }

  function wireCouponsEvents() {
    // Alt sekmeler
    ['pending', 'settled', 'drafts', 'model12'].forEach(key => {
      const btn = document.getElementById('csub-' + key);
      if (btn) {
        btn.onclick = () => {
          activeCpnSubtab = key;
          renderCouponsPane();
        };
      }
    });

    if (activeCpnSubtab === 'model12') {
      wireModel12SubtabEvents();
    }

    // Filtre çipleri
    document.querySelectorAll('.filters .chip[data-sf]').forEach(c => {
      c.onclick = () => {
        settledFilter = c.dataset.sf;
        renderCouponsPane();
      };
    });

    // Taslak butonları
    document.querySelectorAll('.btn-slip-edit').forEach(b => {
      b.onclick = () => {
        const id = b.dataset.id;
        const slip = paperState.slips.find(s => s.id === id);
        if (slip) openCouponEditor(slip);
      };
    });

    document.querySelectorAll('.btn-slip-del').forEach(b => {
      b.onclick = () => {
        const id = b.dataset.id;
        if (confirm('Bu taslak kuponu silmek istediğinize emin misiniz?')) {
          paperState.slips = paperState.slips.filter(s => s.id !== id);
          saveState();
          renderCouponsPane();
        }
      };
    });

    document.querySelectorAll('.btn-slip-add-plan').forEach(b => {
      b.onclick = () => {
        const id = b.dataset.id;
        const slip = paperState.slips.find(s => s.id === id);
        if (!slip) return;

        // Taslaktan çıkarıp plana ekle
        paperState.slips = paperState.slips.filter(s => s.id !== id);
        const res = PE.addSlipToPlan(paperState, slip);
        if (!res.success) {
          alert('Hata: ' + res.error);
          paperState.slips.push(slip); // geri koy
          return;
        }

        paperState = res.state;
        saveState();
        alert('Taslak kupon başarıyla plana eklendi!');
        renderPlanPane();
        renderCouponsPane();
      };
    });
  }

  // ---------------------------------------------------------------------------
  // Otomatik Settlement Tetikleyicisi
  // ---------------------------------------------------------------------------

  function triggerSettlementCheck() {
    if (!paperState || !paperState.slips || !paperState.slips.length) return;
    const results = (root.RESULTS && root.RESULTS.matches) || [];
    const live = root.LIVE_SCORES || [];
    const lookup = PE.buildResultsLookup(results, live);

    const res = PE.settleAllSlips(paperState, lookup);
    if (res.changed) {
      paperState = res.state;
      saveState();
      console.log(`[BETAVUS Paper] ${res.settledCount} kupon sonuçlandırıldı.`);
      renderPlanPane();
      renderCouponsPane();
    }
  }

  // ---------------------------------------------------------------------------
  // Başlatma ve Dışa Aktarılan Arayüz
  // ---------------------------------------------------------------------------

  function init() {
    initState();
  }

  root.BETAVUS_PAPER_UI = {
    init,
    getState: () => paperState,
    setState: (st) => { paperState = st; saveState(); },
    renderSimKasaPane,
    renderSimKuponPane,
    renderPlanPane,
    renderRecPane,
    renderCouponsPane,
    triggerSettlementCheck,
    openCouponEditor,
    getSim30Data: () => sim30Data,
    getPlanSubView: () => planSubView,
    setPlanSubView: (v) => { planSubView = v; renderPlanPane(); }
  };

  // Sayfa yüklendiğinde otomatik başlat
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})(typeof window !== 'undefined' ? window : this);
