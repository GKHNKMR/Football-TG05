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
  let currentChartMode = 'all'; // 'all' | 'cautious' | 'balanced' | 'aggressive'
  let cachedTrajData = null;

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
    const formatted = num.toLocaleString('tr-TR', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
    return `${formatted} ${curr.symbol}`;
  }

  function formatPct(val, dec = 1) {
    const num = Number(val);
    if (isNaN(num)) return '—';
    return '%' + num.toLocaleString('tr-TR', { minimumFractionDigits: dec, maximumFractionDigits: dec });
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
  // Hedef Kasa Ulaşma Grafiği & Risk Modelleri Projeksiyonları
  // ---------------------------------------------------------------------------

  function generateTrajectoryChartSvg(trajData, plan, curr, viewMode, history) {
    const W = 820;
    const H = 360;
    const L = 70;
    const R = 35;
    const T = 35;
    const B = 45;
    const pw = W - L - R;
    const ph = H - T - B;

    const dur = Math.max(1, (trajData && trajData.durationDays) || (plan && plan.durationDays) || 30);
    const startBank = (trajData && trajData.startBank) || (plan && plan.startingBank) || 50;
    const targetBank = (trajData && trajData.targetBank) || (plan && plan.targetBank) || 500;

    const getX = (day) => L + (day / dur) * pw;

    let maxVal = targetBank * 1.15;
    if (trajData && trajData.trajectories) {
      for (const k of ['minimum', 'medium', 'high', 'multi', 'cautious', 'balanced', 'aggressive']) {
        const t = trajData.trajectories[k];
        if (t && t.finalP90) maxVal = Math.max(maxVal, t.finalP90);
      }
    }
    if (history && history.length) {
      for (const h of history) {
        if (h.closingBankroll) maxVal = Math.max(maxVal, h.closingBankroll * 1.08);
      }
    }
    const maxY = Math.ceil(Math.min(targetBank * 2.5, Math.max(targetBank * 1.15, maxVal)) / 25) * 25;
    const getY = (val) => T + ph - (Math.max(0, val) / maxY) * ph;

    // Y Ekseni Kılavuz Çizgileri
    let gridLines = '';
    const numYSteps = 5;
    const yStepVal = maxY / numYSteps;
    for (let i = 0; i <= numYSteps; i++) {
      const v = Math.round(i * yStepVal);
      const yPos = getY(v);
      gridLines += `
        <line x1="${L}" y1="${yPos.toFixed(1)}" x2="${W - R}" y2="${yPos.toFixed(1)}" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
        <text x="${(L - 8).toFixed(1)}" y="${(yPos + 3.5).toFixed(1)}" fill="var(--muted)" font-size="10" text-anchor="end" font-family="inherit">${formatCurrency(v, curr)}</text>
      `;
    }

    // X Ekseni Gün Kılavuzları
    let xGuides = '';
    const xSteps = [0, Math.round(dur * 0.25), Math.round(dur * 0.5), Math.round(dur * 0.75), dur];
    const uniqueXSteps = Array.from(new Set(xSteps)).sort((a, b) => a - b);
    for (const d of uniqueXSteps) {
      const xPos = getX(d);
      xGuides += `
        <line x1="${xPos.toFixed(1)}" y1="${T}" x2="${xPos.toFixed(1)}" y2="${(T + ph).toFixed(1)}" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>
        <text x="${xPos.toFixed(1)}" y="${(T + ph + 16).toFixed(1)}" fill="var(--muted)" font-size="10.5" text-anchor="middle" font-family="inherit">${d === 0 ? '0. Gün' : d === dur ? `${d}. Gün (Hedef)` : `${d}. Gün`}</text>
      `;
    }

    // Hedef Kasa Kılavuzu
    const targetY = getY(targetBank);
    const targetGuide = `
      <line x1="${L}" y1="${targetY.toFixed(1)}" x2="${W - R}" y2="${targetY.toFixed(1)}" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="4,4" opacity="0.45"/>
      <text x="${(W - R).toFixed(1)}" y="${(targetY - 6).toFixed(1)}" fill="#f59e0b" font-size="10.5" font-weight="700" text-anchor="end" font-family="inherit">🎯 HEDEF: ${formatCurrency(targetBank, curr)}</text>
    `;

    // 1. Geometrik Hedef Yolu Çizgisi (Altın sarısı kesikli)
    let targetPathD = '';
    let targetAreaD = `M ${L.toFixed(1)},${(T + ph).toFixed(1)} `;
    (trajData && trajData.targetPoints || []).forEach((pt, idx) => {
      const px = getX(pt.day).toFixed(1);
      const py = getY(pt.targetBank).toFixed(1);
      if (idx === 0) {
        targetPathD += `M ${px},${py}`;
        targetAreaD += `L ${px},${py} `;
      } else {
        targetPathD += ` L ${px},${py}`;
        targetAreaD += `L ${px},${py} `;
      }
    });
    targetAreaD += `L ${(W - R).toFixed(1)},${(T + ph).toFixed(1)} Z`;

    const targetCurveSvg = `
      <path d="${targetAreaD}" fill="url(#targetGrad)" opacity="0.35"/>
      <path d="${targetPathD}" fill="none" stroke="#f59e0b" stroke-width="2.2" stroke-dasharray="6,4"/>
    `;

    // 2. Risk Modelleri Çizgileri (Minimum %15, Orta %20, Yüksek %25)
    let profilesSvg = '';
    const normView = (viewMode === 'cautious' ? 'minimum' : viewMode === 'balanced' ? 'medium' : viewMode === 'aggressive' ? 'high' : (viewMode === 'multi' || viewMode === 'excel') ? 'minimum' : viewMode);
    const profilesToDraw = (normView === 'all' || !normView)
      ? ['minimum', 'medium', 'high']
      : [normView];

    const profColors = {
      minimum: { main: '#10b981', fill: 'rgba(16, 185, 129, 0.12)', name: 'Minimum Risk (%50 Rezerv · %15 Büyüme)' },
      medium: { main: '#3b82f6', fill: 'rgba(59, 130, 246, 0.12)', name: 'Orta Risk (%35 Rezerv · %20 Büyüme)' },
      high: { main: '#ef4444', fill: 'rgba(239, 68, 68, 0.14)', name: 'Yüksek Risk (%25 Rezerv · %25 Büyüme)' }
    };

    if (trajData && trajData.trajectories) {
      for (const pKey of profilesToDraw) {
        const pData = trajData.trajectories[pKey];
        if (!pData || !pData.dayPoints) continue;
        const c = profColors[pKey] || profColors.minimum;

        // Tekil risk modu seçildiyse P10-P90 güven koridorunu çiz
        if (normView !== 'all') {
          let p90Path = '';
          let p10Path = '';
          pData.dayPoints.forEach((pt, idx) => {
            const px = getX(pt.day).toFixed(1);
            const py90 = getY(pt.p90).toFixed(1);
            const py10 = getY(pt.p10).toFixed(1);
            if (idx === 0) {
              p90Path += `M ${px},${py90}`;
              p10Path = `L ${px},${py10}`;
            } else {
              p90Path += ` L ${px},${py90}`;
              p10Path = ` L ${px},${py10}` + p10Path;
            }
          });
          const bandD = p90Path + ' ' + p10Path + ' Z';
          profilesSvg += `<path d="${bandD}" fill="${c.fill}" stroke="none"/>`;
        }

        // Medyan çizgisi
        let medD = '';
        pData.dayPoints.forEach((pt, idx) => {
          const px = getX(pt.day).toFixed(1);
          const py = getY(pt.median).toFixed(1);
          if (idx === 0) medD += `M ${px},${py}`;
          else medD += ` L ${px},${py}`;
        });

        const strokeW = normView === pKey ? 3.2 : (normView === 'all' ? 2.4 : 2.8);
        profilesSvg += `<path d="${medD}" fill="none" stroke="${c.main}" stroke-width="${strokeW}" stroke-linejoin="round"/>`;

        const lastPt = pData.dayPoints[pData.dayPoints.length - 1];
        const endX = getX(lastPt.day).toFixed(1);
        const endY = getY(lastPt.median).toFixed(1);
        profilesSvg += `<circle cx="${endX}" cy="${endY}" r="4" fill="${c.main}" stroke="#0d1219" stroke-width="2"/>`;
      }
    }

    // 3. Gerçekleşen Kasa Çizgisi (varsa)
    let realizedSvg = '';
    if (history && history.length > 0) {
      let rD = '';
      let rCircles = '';
      history.forEach((h, idx) => {
        const d = h.dayIndex != null ? h.dayIndex : idx;
        const val = h.closingBankroll != null ? h.closingBankroll : startBank;
        const rx = getX(d).toFixed(1);
        const ry = getY(val).toFixed(1);
        if (idx === 0) rD += `M ${rx},${ry}`;
        else rD += ` L ${rx},${ry}`;
        rCircles += `<circle cx="${rx}" cy="${ry}" r="4" fill="#fbbf24" stroke="#0d1219" stroke-width="1.8"/>`;
      });
      realizedSvg = `
        <path d="${rD}" fill="none" stroke="#fbbf24" stroke-width="3" stroke-linejoin="round"/>
        ${rCircles}
      `;
    }

    const defs = `
      <defs>
        <linearGradient id="targetGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#f59e0b" stop-opacity="0.25"/>
          <stop offset="100%" stop-color="#f59e0b" stop-opacity="0.0"/>
        </linearGradient>
      </defs>
    `;

    return `
      <svg viewBox="0 0 ${W} ${H}" width="100%" class="trajectory-svg" id="planTrajectorySvg" role="img" aria-label="Hedeflenen Sürede Kasa Ulaşma Grafiği" style="display:block;overflow:visible;">
        ${defs}
        <rect x="${L}" y="${T}" width="${pw}" height="${ph}" fill="#0d1219" rx="6"/>
        ${gridLines}
        ${xGuides}
        ${targetGuide}
        ${targetCurveSvg}
        ${profilesSvg}
        ${realizedSvg}
        <line id="cursorGuide" x1="-10" y1="${T}" x2="-10" y2="${(T + ph).toFixed(1)}" stroke="#ffffff" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.6" style="pointer-events:none;display:none;"/>
        <circle id="cursorPointTarget" cx="-10" cy="-10" r="4.5" fill="#f59e0b" stroke="#fff" stroke-width="1.5" style="pointer-events:none;display:none;"/>
        <circle id="cursorPointCau" cx="-10" cy="-10" r="4.5" fill="#10b981" stroke="#fff" stroke-width="1.5" style="pointer-events:none;display:none;"/>
        <circle id="cursorPointBal" cx="-10" cy="-10" r="4.5" fill="#3b82f6" stroke="#fff" stroke-width="1.5" style="pointer-events:none;display:none;"/>
        <circle id="cursorPointAgg" cx="-10" cy="-10" r="4.5" fill="#ef4444" stroke="#fff" stroke-width="1.5" style="pointer-events:none;display:none;"/>
        <circle id="cursorPointExcel" cx="-10" cy="-10" r="4.5" fill="#38bdf8" stroke="#fff" stroke-width="1.5" style="pointer-events:none;display:none;"/>
        <rect id="chartInteractiveOverlay" x="${L}" y="${T}" width="${pw}" height="${ph}" fill="transparent" style="cursor:crosshair;"/>
      </svg>
    `;
  }

  function renderTrajectoryChartCardHtml(plan, curr, viewMode, trajData) {
    if (!trajData) {
      trajData = PE.calculatePlanTrajectories(plan, null);
    }
    const svgHtml = generateTrajectoryChartSvg(trajData, plan, curr, viewMode, (paperState && paperState.history) || []);
    const minM = trajData.trajectories.minimum || trajData.trajectories.cautious;
    const medM = trajData.trajectories.medium || trajData.trajectories.balanced;
    const highM = trajData.trajectories.high || trajData.trajectories.aggressive;
    const em = trajData.excelModel || (PE.calculateExcelGrowthModel && PE.calculateExcelGrowthModel(plan));

    const rawProf = (paperState && paperState.settings && paperState.settings.riskProfile) || 'minimum';
    const activeProf = (rawProf === 'cautious' ? 'minimum' : rawProf === 'balanced' ? 'medium' : rawProf === 'aggressive' ? 'high' : rawProf);

    return `
      <div class="card plan-chart-card" id="planChartCard">
        <div class="chart-head">
          <div>
            <h3>📈 Hedef Kasa Ulaşma Grafiği · Günlük Büyüme Projeksiyonu</h3>
            <p>Hedeflenen <b>${trajData.durationDays} günde</b> ${formatCurrency(trajData.startBank, curr)} ➔ ${formatCurrency(trajData.targetBank, curr)} geometrik hedef yolu ve risk modellerinin bileşik büyüme patikaları.</p>
          </div>
          <div class="chart-view-chips" id="chartViewChips">
            <button type="button" class="cchip ${viewMode === 'all' || !viewMode ? 'active' : ''}" data-view="all">📊 Tümünü Karşılaştır</button>
            <button type="button" class="cchip ${viewMode === 'minimum' || viewMode === 'cautious' ? 'active' : ''}" data-view="minimum">🟢 Minimum Risk (%50 Rezerv · %15)</button>
            <button type="button" class="cchip ${viewMode === 'medium' || viewMode === 'balanced' ? 'active' : ''}" data-view="medium">🔵 Orta Risk (%35 Rezerv · %20)</button>
            <button type="button" class="cchip ${viewMode === 'high' || viewMode === 'aggressive' ? 'active' : ''}" data-view="high">🔴 Yüksek Risk (%25 Rezerv · %25)</button>
          </div>
        </div>

        <div class="chart-svg-box">
          ${svgHtml}
        </div>

        <div class="chart-tooltip-bar" id="planChartTracker">
          <span class="ct-hint">💡 Grafiğin üzerine gelerek gün bazlı hedef ve 3 risk modelinin büyüme projeksiyonlarını inceleyebilirsiniz.</span>
        </div>

        <div class="chart-legend">
          <span class="cl-item"><span class="cl-dot" style="background:#f59e0b;"></span> 🎯 Hedef Yolu (Geometrik Referans)</span>
          <span class="cl-item"><span class="cl-dot" style="background:#10b981;"></span> 🟢 Minimum Risk (%50 Rezerv · %15 Büyüme · 1.15×)</span>
          <span class="cl-item"><span class="cl-dot" style="background:#3b82f6;"></span> 🔵 Orta Risk (%35 Rezerv · %20 Büyüme · 1.20×)</span>
          <span class="cl-item"><span class="cl-dot" style="background:#ef4444;"></span> 🔴 Yüksek Risk (%25 Rezerv · %25 Büyüme · 1.25×)</span>
          ${paperState && paperState.history && paperState.history.length ? '<span class="cl-item"><span class="cl-dot" style="background:#fbbf24;"></span> 🟡 Gerçekleşen Kasa</span>' : ''}
        </div>

        <div class="chart-models-summary">
          <div class="cms-card minimum ${activeProf === 'minimum' ? 'active-profile' : ''}">
            <div class="cms-head">
              <b>🟢 Minimum Risk</b>
              <span class="cms-badge b-min">%50 Rezerv · %15/gün</span>
            </div>
            <div class="cms-row"><span>Kasa Rezervi:</span><b class="good">%50 (Dokunulmaz)</b></div>
            <div class="cms-row"><span>Aktif Kasa Payı:</span><b>%50</b></div>
            <div class="cms-row"><span>Günlük Büyüme:</span><b class="good">+%15,0 (1.15×)</b></div>
            <div class="cms-row"><span>30. Gün Teorik:</span><b class="good">${formatCurrency(PE.round(trajData.startBank * Math.pow(1.15, trajData.durationDays), 2), curr)}</b></div>
            <div class="cms-row"><span>Medyan Kasa:</span><b>${formatCurrency(minM.finalMedian, curr)}</b></div>
            <div class="cms-row"><span>Hedefe Ulaşma:</span><b class="${minM.targetHitPct >= 50 ? 'good' : 'warn'}">%${minM.targetHitPct}</b></div>
          </div>

          <div class="cms-card medium ${activeProf === 'medium' ? 'active-profile' : ''}">
            <div class="cms-head">
              <b>🔵 Orta Risk</b>
              <span class="cms-badge b-med">%35 Rezerv · %20/gün</span>
            </div>
            <div class="cms-row"><span>Kasa Rezervi:</span><b class="good">%35 (Dokunulmaz)</b></div>
            <div class="cms-row"><span>Aktif Kasa Payı:</span><b>%65</b></div>
            <div class="cms-row"><span>Günlük Büyüme:</span><b class="good">+%20,0 (1.20×)</b></div>
            <div class="cms-row"><span>30. Gün Teorik:</span><b class="good">${formatCurrency(PE.round(trajData.startBank * Math.pow(1.20, trajData.durationDays), 2), curr)}</b></div>
            <div class="cms-row"><span>Medyan Kasa:</span><b>${formatCurrency(medM.finalMedian, curr)}</b></div>
            <div class="cms-row"><span>Hedefe Ulaşma:</span><b class="${medM.targetHitPct >= 50 ? 'good' : 'warn'}">%${medM.targetHitPct}</b></div>
          </div>

          <div class="cms-card high aggressive ${activeProf === 'high' ? 'active-profile' : ''}">
            <div class="cms-head">
              <b style="color:#f87171;">🔴 Yüksek Risk</b>
              <span class="cms-badge b-high">%25 Rezerv · %25/gün</span>
            </div>
            <div class="cms-row"><span>Kasa Rezervi:</span><b class="good">%25 (Dokunulmaz)</b></div>
            <div class="cms-row"><span>Aktif Kasa Payı:</span><b>%75</b></div>
            <div class="cms-row"><span>Günlük Büyüme:</span><b style="color:#f87171;">+%25,0 (1.25×)</b></div>
            <div class="cms-row"><span>30. Gün Teorik:</span><b style="color:#f87171;">${formatCurrency(PE.round(trajData.startBank * Math.pow(1.25, trajData.durationDays), 2), curr)}</b></div>
            <div class="cms-row"><span>Medyan Kasa:</span><b style="color:#f87171;">${formatCurrency(highM.finalMedian, curr)}</b></div>
            <div class="cms-row"><span>Hedefe Ulaşma:</span><b class="${highM.targetHitPct >= 50 ? 'good' : 'warn'}">%${highM.targetHitPct}</b></div>
          </div>
        </div>
      </div>
    `;
  }

  function wireChartInteractiveEvents(container, trajData, curr, plan) {
    if (!container || !trajData) return;
    const overlay = container.querySelector('#chartInteractiveOverlay');
    const svg = container.querySelector('#planTrajectorySvg');
    const guide = container.querySelector('#cursorGuide');
    const ptTarget = container.querySelector('#cursorPointTarget');
    const ptCau = container.querySelector('#cursorPointCau');
    const ptBal = container.querySelector('#cursorPointBal');
    const ptAgg = container.querySelector('#cursorPointAgg');
    const ptExcel = container.querySelector('#cursorPointExcel');
    const tracker = container.querySelector('#planChartTracker');

    if (!overlay || !svg || !tracker) return;

    const W = 820; const H = 360; const L = 70; const R = 35; const T = 35; const B = 45;
    const pw = W - L - R; const ph = H - T - B;
    const dur = Math.max(1, trajData.durationDays);
    const targetBank = trajData.targetBank;

    let maxVal = targetBank * 1.15;
    if (trajData.trajectories) {
      const agg = trajData.trajectories.aggressive;
      const bal = trajData.trajectories.balanced;
      const cau = trajData.trajectories.cautious;
      if (agg && agg.finalP90) maxVal = Math.max(maxVal, agg.finalP90);
      if (bal && bal.finalP90) maxVal = Math.max(maxVal, bal.finalP90);
      if (cau && cau.finalP90) maxVal = Math.max(maxVal, cau.finalP90);
    }
    const maxY = Math.ceil(Math.min(targetBank * 2.5, Math.max(targetBank * 1.15, maxVal)) / 25) * 25;
    const getX = (d) => L + (d / dur) * pw;
    const getY = (val) => T + ph - (Math.max(0, val) / maxY) * ph;

    function handleMove(e) {
      const rect = overlay.getBoundingClientRect();
      const clientX = (e.touches && e.touches[0]) ? e.touches[0].clientX : e.clientX;
      const mouseX = clientX - rect.left;
      const pct = Math.max(0, Math.min(1, mouseX / rect.width));
      const day = Math.round(pct * dur);

      const xPos = getX(day);
      if (guide) {
        guide.setAttribute('x1', xPos);
        guide.setAttribute('x2', xPos);
        guide.style.display = 'block';
      }

      const tPt = trajData.targetPoints && trajData.targetPoints[day];
      const minPt = trajData.trajectories && (trajData.trajectories.minimum || trajData.trajectories.cautious) && (trajData.trajectories.minimum || trajData.trajectories.cautious).dayPoints[day];
      const medPt = trajData.trajectories && (trajData.trajectories.medium || trajData.trajectories.balanced) && (trajData.trajectories.medium || trajData.trajectories.balanced).dayPoints[day];
      const highPt = trajData.trajectories && (trajData.trajectories.high || trajData.trajectories.aggressive) && (trajData.trajectories.high || trajData.trajectories.aggressive).dayPoints[day];
      const emData = trajData.excelModel || (PE.calculateExcelGrowthModel && PE.calculateExcelGrowthModel(plan));
      const emPt = emData && emData.dayPoints && emData.dayPoints[day];

      if (ptTarget && tPt) {
        ptTarget.setAttribute('cx', xPos);
        ptTarget.setAttribute('cy', getY(tPt.targetBank));
        ptTarget.style.display = 'block';
      }
      if (ptCau && minPt) {
        ptCau.setAttribute('cx', xPos);
        ptCau.setAttribute('cy', getY(minPt.median));
        ptCau.style.display = 'block';
      }
      if (ptBal && medPt) {
        ptBal.setAttribute('cx', xPos);
        ptBal.setAttribute('cy', getY(medPt.median));
        ptBal.style.display = 'block';
      }
      if (ptAgg && highPt) {
        ptAgg.setAttribute('cx', xPos);
        ptAgg.setAttribute('cy', getY(highPt.median));
        ptAgg.style.display = 'block';
      }
      if (ptExcel && emPt) {
        ptExcel.setAttribute('cx', xPos);
        ptExcel.setAttribute('cy', getY(emPt.theoreticalBank));
        ptExcel.style.display = 'block';
      }

      const tVal = tPt ? formatCurrency(tPt.targetBank, curr) : '-';
      const minVal = minPt ? formatCurrency(minPt.median, curr) : '-';
      const medVal = medPt ? formatCurrency(medPt.median, curr) : '-';
      const highVal = highPt ? formatCurrency(highPt.median, curr) : '-';

      tracker.innerHTML = `
        <span style="font-weight:900;color:var(--text);">📅 Gün ${day}</span>
        <span>🎯 Hedef: <b>${tVal}</b></span>
        <span>🟢 Minimum Risk (%50 Rezerv · %15): <b>${minVal}</b></span>
        <span>🔵 Orta Risk (%35 Rezerv · %20): <b>${medVal}</b></span>
        <span>🔴 Yüksek Risk (%25 Rezerv · %25): <b style="color:#ef4444;">${highVal}</b></span>
      `;
    }

    function handleLeave() {
      if (guide) guide.style.display = 'none';
      if (ptTarget) ptTarget.style.display = 'none';
      if (ptCau) ptCau.style.display = 'none';
      if (ptBal) ptBal.style.display = 'none';
      if (ptAgg) ptAgg.style.display = 'none';
      if (ptExcel) ptExcel.style.display = 'none';
      tracker.innerHTML = '<span class="ct-hint">💡 Grafiğin üzerine gelerek gün bazlı hedef ve 3 risk modelinin büyüme projeksiyonlarını inceleyebilirsiniz.</span>';
    }

    overlay.onmousemove = handleMove;
    overlay.onmouseleave = handleLeave;
    overlay.ontouchmove = (e) => { e.preventDefault(); handleMove(e); };
    overlay.ontouchend = handleLeave;

    const chips = container.querySelectorAll('.cchip');
    chips.forEach(chip => {
      chip.onclick = () => {
        chips.forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        currentChartMode = chip.dataset.view || 'all';
        const newCardHtml = renderTrajectoryChartCardHtml(plan, curr, currentChartMode, trajData);
        const temp = document.createElement('div');
        temp.innerHTML = newCardHtml;
        const newCard = temp.firstElementChild;
        container.replaceWith(newCard);
        wireChartInteractiveEvents(newCard, trajData, curr, plan);
      };
    });
  }

  // ---------------------------------------------------------------------------
  // Excel Günlük Kasa Modeli Çizelgesi (Betavus Kasa Modeli 1)
  // ---------------------------------------------------------------------------

  function renderExcelDailyTableCardHtml(plan, state, curr) {
    if (!PE.generateExcelDailyTable) return '';
    const activeProf = (state && state.settings && state.settings.riskProfile) || 'balanced';
    const data = PE.generateExcelDailyTable(plan, state, activeProf);
    const m = data.model;
    const rows = data.rows;

    return `
      <div class="card excel-model-card" id="excelModelCard">
        <div class="excel-model-head">
          <div class="emh-title">
            <h3>📑 Günlük Kasa Büyüme Modeli (${esc(m.profileName)})</h3>
            <p>Seçilen ${esc(m.profileName)} doğrultusunda kasa rezervi (%${Math.round(m.reservePct * 100)}), günlük büyüme katsayısı (${m.dailyGrowthFactor}×) ve 30 günlük bileşik takip çizelgesi.</p>
          </div>
          <span class="badge b-excel" style="background:rgba(56,189,248,0.15);color:#38bdf8;font-size:11px;font-weight:800;padding:4px 10px;border-radius:8px;border:1px solid rgba(56,189,248,0.3);">
            Günlük Büyüme: ${m.dailyGrowthFactor}× (+%${m.dailyGrowthRatePct}/gün)
          </span>
        </div>

        <div class="excel-params-strip">
          <div class="ep-col">
            <span class="ep-lbl">Kasa Rezervi</span>
            <span class="ep-val">%${Math.round(m.reservePct * 100)} (${formatCurrency(m.startingBank * m.reservePct, curr)})</span>
            <span class="ep-note">Dokunulmaz rezerv</span>
          </div>
          <div class="ep-col">
            <span class="ep-lbl">Aktif Kasa Payı</span>
            <span class="ep-val">%${Math.round((1.0 - m.reservePct) * 100)} (${formatCurrency(m.startingBank * (1.0 - m.reservePct), curr)})</span>
            <span class="ep-note">Bahislerde kullanılan pay</span>
          </div>
          <div class="ep-col">
            <span class="ep-lbl">Günlük Büyüme Oranı</span>
            <span class="ep-val good">+%${m.dailyGrowthRatePct} / gün</span>
            <span class="ep-note">Bileşik katsayı: ${m.dailyGrowthFactor}×</span>
          </div>
          <div class="ep-col">
            <span class="ep-lbl">30. Gün Teorik Kasa</span>
            <span class="ep-val good">${formatCurrency(m.finalTheoreticalBank, curr)}</span>
            <span class="ep-note">Toplam çarpan: ${m.totalGrowthMultiplier}×</span>
          </div>
          <div class="ep-col">
            <span class="ep-lbl">Model Durumu</span>
            <span class="ep-val good">%100 · ${m.controlStatus}</span>
            <span class="ep-note">Rezerv + Aktif = %100</span>
          </div>
        </div>

        <div class="tbl-scroll" style="max-height:360px;overflow-y:auto;border:1px solid var(--line);border-radius:10px;margin-top:12px;">
          <table class="excel-table">
            <thead>
              <tr>
                <th>Gün</th>
                <th>Tarih</th>
                <th>Teorik Kasa (${curr})</th>
                <th>Gerçek Kasa (${curr})</th>
                <th>Gün Sonu Kasa</th>
                <th>Günlük Büyüme</th>
                <th>Total Büyüme</th>
              </tr>
            </thead>
            <tbody>
              ${rows.map(r => `
                <tr class="${r.isSettled ? 'row-settled' : ''}">
                  <td><b>${r.day}</b></td>
                  <td>${r.dateStr}</td>
                  <td><b style="color:#38bdf8;">${formatCurrency(r.theoreticalBank, curr)}</b></td>
                  <td>${r.actualStartBank != null ? formatCurrency(r.actualStartBank, curr) : '—'}</td>
                  <td>${r.actualEndBank != null ? `<b>${formatCurrency(r.actualEndBank, curr)}</b>` : '—'}</td>
                  <td>${r.dailyGrowthPct != null ? `<span class="${r.dailyGrowthPct >= 0 ? 'good' : 'bad'}">${r.dailyGrowthPct >= 0 ? '+' : ''}%${r.dailyGrowthPct}</span>` : '—'}</td>
                  <td>${r.totalGrowthPct != null ? `<span class="${r.totalGrowthPct >= 0 ? 'good' : 'bad'}">${r.totalGrowthPct >= 0 ? '+' : ''}%${r.totalGrowthPct}</span>` : '—'}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>

        <div class="excel-warn-note" style="margin-top:10px;padding:8px 12px;background:rgba(255,255,255,0.03);border:1px solid var(--line);border-radius:8px;font-size:11px;color:var(--muted);">
          ⚠️ <b>Excel Kasa Modeli Uyarısı:</b> ${esc(m.assumptionNote)}
        </div>
      </div>
    `;
  }

  // ---------------------------------------------------------------------------
  // 30 Günlük Geçmiş Kasa & Kuponlarım Simülasyonu (50 € Örnek Model)
  // ---------------------------------------------------------------------------

  function renderPlanMainNavHtml() {
    return `
      <div class="plan-main-nav">
        <button type="button" class="plan-main-nav-btn ${planSubView === 'active' ? 'active' : ''}" id="btnPmnActive">
          🎯 Aktif Kasa Planım &amp; Yönetim
        </button>
        <button type="button" class="plan-main-nav-btn ${planSubView === 'sim30' ? 'active' : ''}" id="btnPmnSim30">
          ⚡ 30 Günlük Geçmiş Kasa &amp; Kuponlarım Simülasyonu (50 € Örnek Model)
        </button>
      </div>
    `;
  }

  function wirePlanNavEvents() {
    const btnActive = document.getElementById('btnPmnActive');
    const btnSim30 = document.getElementById('btnPmnSim30');
    if (btnActive) {
      btnActive.onclick = () => {
        if (planSubView !== 'active') {
          planSubView = 'active';
          renderPlanPane();
        }
      };
    }
    if (btnSim30) {
      btnSim30.onclick = () => {
        if (planSubView !== 'sim30') {
          planSubView = 'sim30';
          renderPlanPane();
        }
      };
    }
  }

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

  function render30DaySimulationViewHtml(simData) {
    if (!simData || !simData.profiles) {
      return `
        <div class="card" style="padding:24px;text-align:center;color:var(--muted);">
          Simülasyon verisi yüklenemedi. Lütfen internet bağlantınızı kontrol edip tekrar deneyin.
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
        <span class="p-badge">30 GÜNLÜK GEÇMİŞ SİMÜLASYON MODELİ</span>
        <p><b>BETAVUS</b> gerçek bahis sitesi değildir; para kabul etmez veya kupon oynatmaz. Aşağıdaki model, <b>${dmy(win.startDate)} – ${dmy(win.endDate)}</b> tarihleri arasında oynanmış <b>${win.matchesInWindow} adet gerçek maç</b> ve gerçek skorlar ile Avrupa bahis piyasası oranları üzerinden çalıştırılmış 30 günlük disiplinli kasa &amp; kupon simülasyonudur.</p>
        <div class="p-quote">« 50 € Başlangıç Kasası · 3 Risk Modeli · Gerçek Maçlar · Disiplinli Kasa Rezervi »</div>
      </div>

      <div class="sim30-hero-card">
        <div class="sim30-hero-title">
          <div>
            <h3>📈 30 Günlük Geçmiş Kasa Büyümesi &amp; Kuponlarım Simülasyonu</h3>
            <div class="sim30-desc">
              Başlangıç Kasası: <b>50,00 €</b> · Dönem: <b>${dmy(win.startDate)} – ${dmy(win.endDate)} (${win.totalDays} Gün)</b> · Maç Havuzu: <b>${win.matchesInWindow} Resmi Maç</b>
            </div>
          </div>
          <div class="sim30-badges">
            <span class="sim30-badge">🇪🇺 Avrupa Gerçek Piyasa Oranları</span>
            <span class="sim30-badge">🛡️ Korumalı Kasa Rezervi</span>
            <span class="sim30-badge">📊 Excel Modeli Uyumlu</span>
          </div>
        </div>
      </div>

      <!-- Risk Profili Seçim & KPI Grid Kartları -->
      <div class="sim30-kpi-grid">
        <!-- Minimum Risk Kartı -->
        <div class="sim30-kpi-card minimum ${sim30ActiveProfile === 'minimum' ? 'active-card' : ''}" data-prof="minimum">
          <div class="sim30-kpi-head">
            <b style="color:#10b981;font-size:13px;">🟢 Minimum Risk (%50 Rezerv)</b>
            <span class="r-badge b-min">5x 0.5 Üst · ~1.25x</span>
          </div>
          <div class="sim30-kpi-rows">
            <div class="sim30-kpi-row"><span>Başlangıç Kasası:</span><b>50,00 €</b></div>
            <div class="sim30-kpi-row"><span>30. Gün Kasa:</span><b style="color:#10b981;font-size:13px;">${formatCurrency(minProf.stats.finalBank, 'EUR')}</b></div>
            <div class="sim30-kpi-row"><span>Toplam Kâr / ROI:</span><b class="${minProf.stats.totalNetProfit >= 0 ? 'good' : 'bad'}">${minProf.stats.totalNetProfit >= 0 ? '+' : ''}${formatCurrency(minProf.stats.totalNetProfit, 'EUR')} (${minProf.stats.totalRoiPct >= 0 ? '+' : ''}%${minProf.stats.totalRoiPct})</b></div>
            <div class="sim30-kpi-row"><span>Kupon Başarısı:</span><b>${minProf.stats.wonCoupons} / ${minProf.stats.totalCoupons} (%${minProf.stats.winRatePct})</b></div>
            <div class="sim30-kpi-row"><span>Rezerv / Aktif:</span><span>${formatCurrency(minProf.stats.reserveBank, 'EUR')} / ${formatCurrency(minProf.stats.activeBank, 'EUR')}</span></div>
            <div class="sim30-kpi-row"><span>Maks. Düşüş (DD):</span><b class="warn">%${minProf.stats.maxDrawdownPct}</b></div>
          </div>
        </div>

        <!-- Orta Risk Kartı -->
        <div class="sim30-kpi-card medium ${sim30ActiveProfile === 'medium' ? 'active-card' : ''}" data-prof="medium">
          <div class="sim30-kpi-head">
            <b style="color:#38bdf8;font-size:13px;">🔵 Orta Risk (%35 Rezerv)</b>
            <span class="r-badge b-med">3x 1.5 Üst · ~1.70x</span>
          </div>
          <div class="sim30-kpi-rows">
            <div class="sim30-kpi-row"><span>Başlangıç Kasası:</span><b>50,00 €</b></div>
            <div class="sim30-kpi-row"><span>30. Gün Kasa:</span><b style="color:#38bdf8;font-size:13px;">${formatCurrency(medProf.stats.finalBank, 'EUR')}</b></div>
            <div class="sim30-kpi-row"><span>Toplam Kâr / ROI:</span><b class="${medProf.stats.totalNetProfit >= 0 ? 'good' : 'bad'}">${medProf.stats.totalNetProfit >= 0 ? '+' : ''}${formatCurrency(medProf.stats.totalNetProfit, 'EUR')} (${medProf.stats.totalRoiPct >= 0 ? '+' : ''}%${medProf.stats.totalRoiPct})</b></div>
            <div class="sim30-kpi-row"><span>Kupon Başarısı:</span><b>${medProf.stats.wonCoupons} / ${medProf.stats.totalCoupons} (%${medProf.stats.winRatePct})</b></div>
            <div class="sim30-kpi-row"><span>Rezerv / Aktif:</span><span>${formatCurrency(medProf.stats.reserveBank, 'EUR')} / ${formatCurrency(medProf.stats.activeBank, 'EUR')}</span></div>
            <div class="sim30-kpi-row"><span>Maks. Düşüş (DD):</span><b class="warn">%${medProf.stats.maxDrawdownPct}</b></div>
          </div>
        </div>

        <!-- Yüksek Risk Kartı -->
        <div class="sim30-kpi-card high ${sim30ActiveProfile === 'high' ? 'active-card' : ''}" data-prof="high">
          <div class="sim30-kpi-head">
            <b style="color:#ef4444;font-size:13px;">🔴 Yüksek Risk (%25 Rezerv)</b>
            <span class="r-badge b-high">3x 2.5 Üst · ~3.25x</span>
          </div>
          <div class="sim30-kpi-rows">
            <div class="sim30-kpi-row"><span>Başlangıç Kasası:</span><b>50,00 €</b></div>
            <div class="sim30-kpi-row"><span>30. Gün Kasa:</span><b style="color:#ef4444;font-size:13px;">${formatCurrency(highProf.stats.finalBank, 'EUR')}</b></div>
            <div class="sim30-kpi-row"><span>Toplam Kâr / ROI:</span><b class="${highProf.stats.totalNetProfit >= 0 ? 'good' : 'bad'}">${highProf.stats.totalNetProfit >= 0 ? '+' : ''}${formatCurrency(highProf.stats.totalNetProfit, 'EUR')} (${highProf.stats.totalRoiPct >= 0 ? '+' : ''}%${highProf.stats.totalRoiPct})</b></div>
            <div class="sim30-kpi-row"><span>Kupon Başarısı:</span><b>${highProf.stats.wonCoupons} / ${highProf.stats.totalCoupons} (%${highProf.stats.winRatePct})</b></div>
            <div class="sim30-kpi-row"><span>Rezerv / Aktif:</span><span>${formatCurrency(highProf.stats.reserveBank, 'EUR')} / ${formatCurrency(highProf.stats.activeBank, 'EUR')}</span></div>
            <div class="sim30-kpi-row"><span>Maks. Düşüş (DD):</span><b class="warn">%${highProf.stats.maxDrawdownPct}</b></div>
          </div>
        </div>
      </div>

      <!-- 30 Günlük Kasa Gelişim Grafiği (SVG) -->
      <div class="card" style="margin-bottom:16px;">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:12px;">
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

      <!-- Alt Sekmeler: Kuponlarım Simülasyonu & Kasa Muhasebe Çizelgesi -->
      <div class="sim30-subtabs-bar">
        <button type="button" class="sim30-subtab ${sim30ActiveSubtab === 'coupons' ? 'active' : ''}" id="btnSim30TabCoupons">
          🎫 Kuponlarım Simülasyonu (${activeProfData.coupons.length} Günlük Kupon)
        </button>
        <button type="button" class="sim30-subtab ${sim30ActiveSubtab === 'ledger' ? 'active' : ''}" id="btnSim30TabLedger">
          📊 Kasa Muhasebe Çizelgesi (31 Gün Finansal Tablo)
        </button>
      </div>

      <!-- Alt Sekme İçeriği -->
      <div id="sim30SubtabContent">
        ${sim30ActiveSubtab === 'coupons'
          ? render30DayCouponsListHtml(simData, sim30ActiveProfile)
          : render30DayLedgerTableHtml(simData, sim30ActiveProfile)}
      </div>
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
              <div>Kupon Stake (%${prof.stakePct}): <b>${formatCurrency(cpn.stake, 'EUR')}</b></div>
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
                    <th>Tahmin / Pazar</th>
                    <th>Piyasa Oranı</th>
                    <th>Biten Skor</th>
                    <th>Toplam Gol</th>
                    <th>Sonuç</th>
                  </tr>
                </thead>
                <tbody>
                  ${cpn.legs.map((leg, lIdx) => `
                    <tr>
                      <td>${lIdx + 1}</td>
                      <td>${flag(leg.league)} ${esc(leg.league)}</td>
                      <td><b>${esc(leg.home)}</b> vs <b>${esc(leg.away)}</b></td>
                      <td style="color:var(--muted);">${timeStr(leg.kickoff_utc)}</td>
                      <td><span class="line-badge">${esc(leg.market)}</span></td>
                      <td><b style="color:#38bdf8;">${leg.odds.toFixed(2)}</b></td>
                      <td><b>${leg.realScore || (leg.homeGoals != null ? `${leg.homeGoals} - ${leg.awayGoals}` : '—')}</b></td>
                      <td>${leg.totalGoals != null ? `${leg.totalGoals} Gol` : '—'}</td>
                      <td>
                        <span class="${leg.isWon ? 'good' : 'bad'}" style="font-weight:700;">
                          ${leg.isWon ? '✅ Geldi' : '❌ Yatış'}
                        </span>
                      </td>
                    </tr>
                  `).join('')}
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
          🛡️ <b>Kasa Yönetim Notu:</b> ${esc(prof.name)} modelinde her gün kasanın %${prof.reservePct}'si (dokunulmaz sermaye tabanı) korunmuştur. Kupon yatması halinde bile kasa sıfırlanmaz, sonraki günün stake tutarı geriye kalan toplam kasanın %${prof.stakePct}'si olarak otomatik küçülür.
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
        if (pt.theoreticalTarget && activeProfile !== 'all') {
          maxVal = Math.max(maxVal, Math.min(pt.theoreticalTarget, 400));
        }
      });
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

    profilesToCheck.forEach(profKey => {
      const prof = simData.profiles[profKey];
      if (!prof || !prof.ledger) return;
      const st = profStyles[profKey] || profStyles.minimum;

      // Realized curve
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
        <path d="${areaD}" fill="url(#${st.gradId})" opacity="${activeProfile === 'all' ? 0.35 : 0.5}"/>
        <path d="${pathD}" fill="none" stroke="${st.stroke}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
      `;

      // Draw theoretical curve if single profile selected
      if (activeProfile !== 'all') {
        let theoPathD = `M ${getX(0).toFixed(1)},${getY(50.0).toFixed(1)}`;
        prof.ledger.forEach(pt => {
          const px = getX(pt.day).toFixed(1);
          const py = getY(Math.min(pt.theoreticalTarget, maxY)).toFixed(1);
          theoPathD += ` L ${px},${py}`;
        });
        curvesSvg += `
          <path d="${theoPathD}" fill="none" stroke="#f59e0b" stroke-width="1.6" stroke-dasharray="4,4" opacity="0.7"/>
        `;
      }
    });

    return `
      <svg viewBox="0 0 ${W} ${H}" class="trajectory-svg" id="sim30Svg" style="width:100%;height:auto;display:block;user-select:none;">
        <defs>
          <linearGradient id="gradSimMin" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#10b981" stop-opacity="0.3"/>
            <stop offset="100%" stop-color="#10b981" stop-opacity="0.0"/>
          </linearGradient>
          <linearGradient id="gradSimMed" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.3"/>
            <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.0"/>
          </linearGradient>
          <linearGradient id="gradSimHigh" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#ef4444" stop-opacity="0.3"/>
            <stop offset="100%" stop-color="#ef4444" stop-opacity="0.0"/>
          </linearGradient>
        </defs>

        <!-- Arka Plan Kılavuz Çizgileri -->
        ${gridLines}
        ${xGuides}

        <!-- 50 Euro Başlangıç Kılavuzu -->
        <line x1="${L}" y1="${getY(50.0).toFixed(1)}" x2="${W - R}" y2="${getY(50.0).toFixed(1)}" stroke="rgba(255,255,255,0.2)" stroke-dasharray="2,2"/>
        <text x="${(W - R).toFixed(1)}" y="${(getY(50.0) - 5).toFixed(1)}" fill="var(--muted)" font-size="9.5" text-anchor="end" font-family="inherit">Başlangıç: 50,00 €</text>

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

  function wire30DaySimulationEvents(simData) {
    // KPI kart tıklamaları -> aktif profili değiştir
    document.querySelectorAll('.sim30-kpi-card').forEach(card => {
      card.onclick = () => {
        const prof = card.dataset.prof;
        if (prof) {
          sim30ActiveProfile = prof;
          const pane = document.getElementById('pane-plan');
          if (pane) {
            pane.innerHTML = `
              ${renderPlanMainNavHtml()}
              ${render30DaySimulationViewHtml(simData)}
            `;
            wirePlanNavEvents();
            wire30DaySimulationEvents(simData);
          }
        }
      };
    });

    // Grafik üstü profil filtreleme butonları
    document.querySelectorAll('#sim30ChartPills .cmp-btn').forEach(btn => {
      btn.onclick = () => {
        const prof = btn.dataset.prof;
        if (prof) {
          sim30ActiveProfile = prof;
          const pane = document.getElementById('pane-plan');
          if (pane) {
            pane.innerHTML = `
              ${renderPlanMainNavHtml()}
              ${render30DaySimulationViewHtml(simData)}
            `;
            wirePlanNavEvents();
            wire30DaySimulationEvents(simData);
          }
        }
      };
    });

    // Subtab butonları: Kuponlarım vs Kasa Muhasebe Çizelgesi
    const btnCpn = document.getElementById('btnSim30TabCoupons');
    const btnLedger = document.getElementById('btnSim30TabLedger');
    const contentEl = document.getElementById('sim30SubtabContent');

    if (btnCpn && contentEl) {
      btnCpn.onclick = () => {
        sim30ActiveSubtab = 'coupons';
        btnCpn.classList.add('active');
        if (btnLedger) btnLedger.classList.remove('active');
        contentEl.innerHTML = render30DayCouponsListHtml(simData, sim30ActiveProfile);
      };
    }
    if (btnLedger && contentEl) {
      btnLedger.onclick = () => {
        sim30ActiveSubtab = 'ledger';
        btnLedger.classList.add('active');
        if (btnCpn) btnCpn.classList.remove('active');
        contentEl.innerHTML = render30DayLedgerTableHtml(simData, sim30ActiveProfile);
      };
    }

    // Grafik tooltip olayları
    const svgEl = document.getElementById('sim30Svg');
    if (svgEl) {
      wire30DaySimulationChartEvents(svgEl);
    }
  }

  function wire30DaySimulationChartEvents(svgEl) {
    const tooltip = svgEl.querySelector('#sim30ChartTooltip');
    const ttBg = svgEl.querySelector('#sim30TooltipBg');
    const ttTitle = svgEl.querySelector('#sim30TtTitle');
    const ttVal = svgEl.querySelector('#sim30TtVal');
    const ttSub = svgEl.querySelector('#sim30TtSub');
    if (!tooltip || !ttBg || !ttTitle || !ttVal || !ttSub) return;

    const dots = svgEl.querySelectorAll('.sim30-dot');
    dots.forEach(dot => {
      dot.onmouseenter = () => {
        const cx = parseFloat(dot.getAttribute('cx'));
        const cy = parseFloat(dot.getAttribute('cy'));
        const day = dot.dataset.day;
        const date = dmy(dot.dataset.date);
        const prof = dot.dataset.prof;
        const bank = formatCurrency(dot.dataset.bank, 'EUR');
        const change = parseFloat(dot.dataset.change);
        const status = dot.dataset.status === 'won' ? '✅ Kupon Kazandı' : '❌ Kupon Kaybetti';

        dot.setAttribute('r', '6');

        ttTitle.textContent = `Gün ${day} · ${date} (${prof})`;
        ttVal.textContent = `Kasa: ${bank}`;
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
        dot.setAttribute('r', '3.5');
        tooltip.setAttribute('opacity', '0');
      };
    });
  }

  // ---------------------------------------------------------------------------
  // Kasa Planım Ekranı (#pane-plan)
  // ---------------------------------------------------------------------------

  function renderPlanPane() {
    const pane = document.getElementById('pane-plan');
    if (!pane) return;

    if (planSubView === 'sim30') {
      pane.innerHTML = `
        ${renderPlanMainNavHtml()}
        <div class="sim30-loading" style="padding:48px 24px;text-align:center;color:var(--muted);">
          <div style="font-size:24px;margin-bottom:8px;">⏳</div>
          <div>30 Günlük Geçmiş Kasa ve Kupon Simülasyonu Hazırlanıyor...</div>
        </div>
      `;
      wirePlanNavEvents();
      getOrGenerateSim30Data((data) => {
        if (planSubView !== 'sim30') return;
        pane.innerHTML = `
          ${renderPlanMainNavHtml()}
          ${render30DaySimulationViewHtml(data)}
        `;
        wirePlanNavEvents();
        wire30DaySimulationEvents(data);
      });
      return;
    }

    if (!paperState || !paperState.plan) {
      pane.innerHTML = `
        ${renderPlanMainNavHtml()}
        ${renderPlanSetupHtml()}
      `;
      wirePlanNavEvents();
      wirePlanSetupEvents();
      return;
    }

    const metrics = PE.getPlanMetrics(paperState);
    const curr = paperState.settings.currency || 'EUR';
    const rawProfKey = paperState.settings.riskProfile || 'minimum';
    const profKey = (rawProfKey === 'cautious' ? 'minimum' : rawProfKey === 'balanced' ? 'medium' : rawProfKey === 'aggressive' ? 'high' : rawProfKey);
    const prof = PE.RISK_PROFILES[profKey] || PE.RISK_PROFILES.minimum;

    // Simülasyonu hesapla (eğer çalıştırılmamışsa veya eski ise)
    if (!paperState.simulation || !paperState.simulation.result) {
      const sim = PE.runPlanSimulation(paperState.plan, prof.id, null, {
        remainingDays: metrics.remainingDays,
        currentBank: metrics.totalBank
      });
      paperState.simulation = {
        lastRunAt: new Date().toISOString(),
        seed: 42,
        result: sim
      };
      saveState();
    }
    const simRes = paperState.simulation.result;

    // Adaptif seçenekler (eğer geride ise)
    const adaptive = PE.buildAdaptiveOptions(paperState.plan, paperState, null);

    // Hedef Kasa Ulaşma Trajektorisi ve Risk Modelleri Projeksiyonları
    const trajData = PE.calculatePlanTrajectories(paperState.plan, null);
    cachedTrajData = trajData;

    pane.innerHTML = `
      ${renderPlanMainNavHtml()}

      <div class="paper-disclaimer">
        <span class="p-badge">SANAL KASA SİMÜLASYONU</span>
        <p><b>BETAVUS</b> bahis kabul etmez, ödeme almaz ve kupon oynatmaz. Gösterilen kasa, stake ve getiriler sanaldır. Tahminler olasılıksaldır ve sonuç garantisi vermez.</p>
        <div class="p-quote">« Önce simüle et. Riskini gör. Stratejini ölç. Sonra karar ver. »</div>
      </div>

      <div class="plan-header-card">
        <div class="plan-title-row">
          <div>
            <h2>Sanal Kasa Planım · ${esc(prof.name)}</h2>
            <div class="plan-sub">Başlangıç: ${dmy(paperState.plan.startDate)} · ${metrics.durationDays} Günlük Plan (${metrics.elapsedDays}. Gün / ${metrics.remainingDays} Gün Kaldı)</div>
          </div>
          <div class="plan-status-badge ${metrics.status.code}">
            <span class="dot">●</span> ${metrics.status.label} (${metrics.status.diffPct > 0 ? '+' : ''}${metrics.status.diffPct}%)
          </div>
        </div>

        <div class="plan-tiles">
          <div class="tile">
            <div class="v">${formatCurrency(metrics.availableBalance, curr)}</div>
            <div class="k">Kullanılabilir Bakiye</div>
            <div class="n">Kuponlara bağlanmamış sanal bakiye</div>
          </div>
          <div class="tile">
            <div class="v">${formatCurrency(metrics.pendingStake, curr)}</div>
            <div class="k">Bekleyen Stake</div>
            <div class="n">Oynanmamış maçlardaki sanal tutar</div>
          </div>
          <div class="tile">
            <div class="v">${formatCurrency(metrics.totalBank, curr)}</div>
            <div class="k">Toplam Sanal Kasa</div>
            <div class="n">Kullanılabilir + Bekleyen Stake (${metrics.totalGrowthPct > 0 ? '+' : ''}${metrics.totalGrowthPct}%)</div>
          </div>
          <div class="tile">
            <div class="v">${formatCurrency(metrics.targetBank, curr)}</div>
            <div class="k">Hedef Kasa</div>
            <div class="n">İlerleme: ${metrics.progressPct}% (Başlangıç: ${formatCurrency(metrics.startingBank, curr)})</div>
          </div>
        </div>

        <!-- İlerleme Çubuğu -->
        <div class="plan-progress-box">
          <div class="pbar-labels">
            <span>Başlangıç: ${formatCurrency(metrics.startingBank, curr)}</span>
            <span>Bugünkü Hedef Yolu: <b>${formatCurrency(metrics.targetToday, curr)}</b></span>
            <span>Hedef: ${formatCurrency(metrics.targetBank, curr)}</span>
          </div>
          <div class="pbar-track">
            <div class="pbar-fill" style="width: ${Math.min(100, metrics.progressPct)}%"></div>
            <div class="pbar-target-marker" style="left: ${Math.min(100, Math.max(0, ((metrics.targetToday - metrics.startingBank) / (metrics.targetBank - metrics.startingBank)) * 100))}%" title="Bugünkü Geometrik Hedef Yolu"></div>
          </div>
          <div class="pbar-hint">
            <span>Hedefe ulaşmak için gereken bileşik günlük oran: <b>%${metrics.dailyReqRate} / gün</b></span>
            <span>Kasa Durumu: <b class="${metrics.status.color}">${metrics.status.label}</b></span>
          </div>
        </div>
      </div>

      <!-- Hedeflenen Sürede Kasa Ulaşma Grafiği -->
      ${renderTrajectoryChartCardHtml(paperState.plan, curr, currentChartMode, trajData)}

      <!-- Monte Carlo Simülasyon Kartı -->
      <div class="card sim-card">
        <div class="sim-head">
          <div>
            <h3>🎲 5.000 İterasyonlu Monte Carlo Kasa Projeksiyonu</h3>
            <p>Seçili risk profili (${esc(prof.name)}) ve kupon olasılıklarına dayalı deterministik simülasyon sonuçları (Garanti içermez).</p>
          </div>
          <button class="btn-subtle" id="btnRerunSim" type="button">↻ Yeniden Hesapla</button>
        </div>
        <div class="sim-tiles">
          <div class="stile">
            <div class="sv">${formatCurrency(simRes.medianBank, curr)}</div>
            <div class="sk">Plan Sonu Medyan Kasa</div>
            <div class="sn">Olası orta senaryo değeri</div>
          </div>
          <div class="stile">
            <div class="sv">${formatCurrency(simRes.p10, curr)} – ${formatCurrency(simRes.p90, curr)}</div>
            <div class="sk">P10 / P90 Senaryo Aralığı</div>
            <div class="sn">%80 olasılıkla bu aralıkta kalır</div>
          </div>
          <div class="stile">
            <div class="sv good">%${simRes.targetHitPct}</div>
            <div class="sk">Hedefe Ulaşma İhtimali</div>
            <div class="sn">Süre içinde hedefe varış</div>
          </div>
          <div class="stile">
            <div class="sv ${simRes.halfBankLossPct > 20 ? 'bad' : 'warn'}">%${simRes.halfBankLossPct}</div>
            <div class="sk">Yarı Kasa Kaybı Riski</div>
            <div class="sn">Kasayı %50 kaybetme riski</div>
          </div>
          <div class="stile">
            <div class="sv warn">%${simRes.maxDrawdownPct}</div>
            <div class="sk">Tahmini Maks. Düşüş</div>
            <div class="sn">Tepe noktadan beklenen çekilme</div>
          </div>
        </div>
      </div>

      <!-- Betavus Çok Kollu Kasa Büyüme Modeli (Excel Çizelgesi) -->
      ${renderExcelDailyTableCardHtml(paperState.plan, paperState, curr)}

      <!-- Adaptif Öneriler (Gerekirse) -->
      ${adaptive && adaptive.showAdaptive ? renderAdaptiveCardHtml(adaptive, curr) : ''}

      <!-- Plan Eylemleri ve Dışa/İçe Aktar -->
      <div class="plan-actions-card">
        <div class="p-act-left">
          <button class="btn-sec" id="btnExportJSON" type="button">📥 Geçmişi Dışa Aktar (JSON)</button>
          <button class="btn-sec" id="btnImportJSON" type="button">📤 JSON İçe Aktar</button>
          <input type="file" id="jsonFileInput" accept=".json" style="display:none">
        </div>
        <div class="p-act-right">
          <button class="btn-danger-subtle" id="btnResetPlan" type="button">⚠️ Planı Sıfırla / Yeni Plan Kur</button>
        </div>
      </div>
    `;

    wirePlanDashboardEvents();
  }

  function renderPlanSetupHtml() {
    return `
      <div class="paper-disclaimer">
        <span class="p-badge">SANAL KASA SİMÜLASYONU</span>
        <p><b>BETAVUS</b> bir bahis platformu değildir; bahis oynatmaz ve gerçek para kabul etmez. Futbol gol tahminleri için yapay zekâ destekli bir <b>paper-betting ve kasa yönetim simülatörüdür</b>.</p>
        <div class="p-quote">« Önce simüle et. Riskini gör. Stratejini ölç. Sonra karar ver. »</div>
      </div>

      <div class="card plan-setup-card">
        <h2>🎯 Sanal Kasa Planı Oluştur</h2>
        <p>Disiplinli kasa yönetimi için sanal başlangıç bütçenizi, hedefinizi ve risk toleransınızı tanımlayın.</p>

        <form id="planSetupForm" onsubmit="return false;">
          <div class="form-row">
            <div class="form-group">
              <label for="setupStartBank">Sanal Başlangıç Kasası *</label>
              <input type="text" id="setupStartBank" class="form-input" placeholder="Örn: 50" value="50" required autocomplete="off">
              <span class="form-hint">Simülasyona başlayacağınız sanal miktar.</span>
            </div>
            <div class="form-group">
              <label for="setupTargetBank">Sanal Hedef Kasa *</label>
              <input type="text" id="setupTargetBank" class="form-input" placeholder="Örn: 500" value="500" required autocomplete="off">
              <span class="form-hint">Ulaşmayı hedeflediğiniz sanal kasa tutarı.</span>
            </div>
            <div class="form-group">
              <label for="setupCurrency">Para Birimi</label>
              <select id="setupCurrency" class="form-select">
                <option value="EUR" selected>EUR (€)</option>
                <option value="TRY">TRY (₺)</option>
                <option value="USD">USD ($)</option>
                <option value="GBP">GBP (£)</option>
              </select>
            </div>
          </div>

          <div class="form-group" style="margin-top:14px;">
            <label>Plan Süresi (Gün) *</label>
            <div class="pill-group" id="durationPills">
              <button type="button" class="pill" data-days="7">7 Gün</button>
              <button type="button" class="pill" data-days="14">14 Gün</button>
              <button type="button" class="pill active" data-days="30">30 Gün</button>
              <button type="button" class="pill" data-days="60">60 Gün</button>
              <button type="button" class="pill" data-days="90">90 Gün</button>
            </div>
            <input type="number" id="setupDuration" class="form-input" value="30" min="1" max="365" style="margin-top:8px;max-width:160px;" placeholder="Özel gün sayısı">
          </div>

          <div class="form-group" style="margin-top:16px;">
            <label>Kasa Modeli &amp; Risk Toleransı *</label>
            <div class="risk-cards">
              <label class="risk-card active">
                <input type="radio" name="setupRisk" value="minimum" checked>
                <div class="r-head">
                  <b>🟢 Minimum Risk</b>
                  <span class="r-badge b-min">Rezerv: %50 · Günlük %15</span>
                </div>
                <p>Kasanın %50'si dokunulmaz rezerv olarak tutulur. %50 aktif payla günlük %15 büyüme hedeflenir (1.15x/gün).</p>
              </label>
              <label class="risk-card">
                <input type="radio" name="setupRisk" value="medium">
                <div class="r-head">
                  <b>🔵 Orta Risk</b>
                  <span class="r-badge b-med">Rezerv: %35 · Günlük %20</span>
                </div>
                <p>Kasanın %35'i dokunulmaz rezerv olarak tutulur. %65 aktif payla günlük %20 büyüme hedeflenir (1.20x/gün).</p>
              </label>
              <label class="risk-card">
                <input type="radio" name="setupRisk" value="high">
                <div class="r-head">
                  <b style="color:#f87171;">🔴 Yüksek Risk</b>
                  <span class="r-badge b-high">Rezerv: %25 · Günlük %25</span>
                </div>
                <p>Kasanın %25'i dokunulmaz rezerv olarak tutulur. %75 aktif payla günlük %25 büyüme hedeflenir (1.25x/gün).</p>
              </label>
            </div>
          </div>

          <div class="setup-chart-preview" id="setupChartPreviewBox">
            <h4>📈 Hedeflenen Sürede Kasa Ulaşma Grafiği Canlı Önizlemesi</h4>
            <p>Seçilen başlangıç kasası, hedef kasa ve süreye göre geometrik hedef yolu ve risk modellerinin büyüme patikaları.</p>
            <div id="setupChartSvgContainer" class="chart-svg-box"></div>
          </div>

          <div id="setupFormError" class="form-error" hidden></div>

          <div class="form-footer">
            <button type="button" id="btnCreatePlan" class="btn-primary" style="padding:14px 28px;font-size:14px;">
              🚀 Sanal Kasa Planı Oluştur
            </button>
            <span class="form-guarantee-note">⚠️ Bu bir simülasyondur; hedef garantisi veya kesin kâr vaadi verilmez.</span>
          </div>
        </form>
      </div>
    `;
  }

  function renderAdaptiveCardHtml(adaptive, curr) {
    return `
      <div class="card adaptive-card">
        <div class="adapt-head">
          <div>
            <h3>⚖️ Hedef Yolu Karşılaştırması &amp; Adaptif Alternatifler</h3>
            <p>Sanal kasanız güncel hedef yolunun gerisinde seyrediyor. Sistem riski otomatik artırmaz; stratejinizi korumak için 3 alternatifi karşılaştırın:</p>
          </div>
        </div>
        <div class="adapt-grid">
          ${adaptive.options.map(opt => `
            <div class="adapt-col ${opt.id}">
              <div class="col-tag">${esc(opt.tag)}</div>
              <h4>${esc(opt.title)}</h4>
              <p class="desc">${esc(opt.desc)}</p>
              <div class="adapt-metrics">
                <div class="am-row"><span>Başarı İhtimali:</span><b>%${opt.metrics.newTargetProbPct}</b></div>
                <div class="am-row"><span>Medyan Kasa:</span><b>${formatCurrency(opt.metrics.medianBank, curr)}</b></div>
                <div class="am-row"><span>P10 / P90:</span><b>${formatCurrency(opt.metrics.p10, curr)} – ${formatCurrency(opt.metrics.p90, curr)}</b></div>
                <div class="am-row"><span>Yarı Kasa Kaybı:</span><b class="${opt.metrics.halfBankLossPct > 25 ? 'bad' : 'warn'}">%${opt.metrics.halfBankLossPct}</b></div>
                <div class="am-row"><span>Maks. Düşüş:</span><b>%${opt.metrics.maxDrawdownPct}</b></div>
              </div>
              ${opt.warning ? `<div class="adapt-warn">⚠️ ${esc(opt.warning)}</div>` : ''}
              <button class="btn-apply-adapt" data-opt="${opt.id}" type="button">Bu Alternatifi Uygula</button>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  function wirePlanSetupEvents() {
    function updateSetupChartPreview() {
      const container = document.getElementById('setupChartSvgContainer');
      if (!container) return;
      const start = parseNumber(document.getElementById('setupStartBank')?.value) || 50;
      const target = parseNumber(document.getElementById('setupTargetBank')?.value) || 500;
      const duration = parseInt(document.getElementById('setupDuration')?.value, 10) || 30;
      const curr = document.getElementById('setupCurrency')?.value || 'EUR';
      if (start <= 0 || target <= start || duration <= 0) {
        container.innerHTML = '<div style="padding:24px;text-align:center;color:var(--muted);font-size:11.5px;">Geçerli başlangıç kasası, hedef kasa ve süre girildiğinde grafik görüntülenecektir.</div>';
        return;
      }
      const dummyPlan = { startingBank: start, targetBank: target, durationDays: duration };
      const previewTraj = PE.calculatePlanTrajectories(dummyPlan, null);
      container.innerHTML = generateTrajectoryChartSvg(previewTraj, dummyPlan, curr, 'all', []);
    }

    const pills = document.querySelectorAll('#durationPills .pill');
    const durInput = document.getElementById('setupDuration');
    pills.forEach(p => {
      p.onclick = () => {
        pills.forEach(x => x.classList.remove('active'));
        p.classList.add('active');
        if (durInput) durInput.value = p.dataset.days;
        updateSetupChartPreview();
      };
    });
    if (durInput) {
      durInput.oninput = () => {
        pills.forEach(x => x.classList.toggle('active', x.dataset.days === durInput.value));
        updateSetupChartPreview();
      };
    }

    const rCards = document.querySelectorAll('.risk-card');
    rCards.forEach(rc => {
      rc.onclick = () => {
        rCards.forEach(x => x.classList.remove('active'));
        rc.classList.add('active');
        const radio = rc.querySelector('input[type="radio"]');
        if (radio) radio.checked = true;
        updateSetupChartPreview();
      };
    });

    const startInp = document.getElementById('setupStartBank');
    const targetInp = document.getElementById('setupTargetBank');
    const currInp = document.getElementById('setupCurrency');
    if (startInp) startInp.oninput = updateSetupChartPreview;
    if (targetInp) targetInp.oninput = updateSetupChartPreview;
    if (currInp) currInp.onchange = updateSetupChartPreview;

    updateSetupChartPreview();

    const btn = document.getElementById('btnCreatePlan');
    if (btn) {
      btn.onclick = () => {
        const start = parseNumber(document.getElementById('setupStartBank')?.value);
        const target = parseNumber(document.getElementById('setupTargetBank')?.value);
        const duration = parseInt(document.getElementById('setupDuration')?.value, 10);
        const curr = document.getElementById('setupCurrency')?.value || 'EUR';
        const riskRadio = document.querySelector('input[name="setupRisk"]:checked');
        const risk = riskRadio ? riskRadio.value : 'minimum';
        const errEl = document.getElementById('setupFormError');

        const errors = [];
        if (start == null || start <= 0) errors.push('Sanal başlangıç kasası 0\'dan büyük olmalıdır.');
        if (target == null || target <= 0) errors.push('Hedef kasa 0\'dan büyük olmalıdır.');
        if (start != null && target != null && target <= start) errors.push('Hedef kasa, başlangıç kasasından büyük olmalıdır.');
        if (isNaN(duration) || duration <= 0) errors.push('Plan süresi geçerli bir pozitif gün sayısı olmalıdır.');

        if (errors.length) {
          if (errEl) {
            errEl.innerHTML = errors.join('<br>');
            errEl.hidden = false;
          }
          return;
        }

        paperState = PE.createInitialState(
          { currency: curr, riskProfile: risk },
          { startingBank: start, targetBank: target, durationDays: duration }
        );
        saveState();
        renderPlanPane();
        renderRecPane();
        renderCouponsPane();
      };
    }
  }

  function wirePlanDashboardEvents() {
    wirePlanNavEvents();

    const chartCard = document.getElementById('planChartCard');
    if (chartCard && cachedTrajData && paperState && paperState.plan) {
      const curr = (paperState.settings && paperState.settings.currency) || 'EUR';
      wireChartInteractiveEvents(chartCard, cachedTrajData, curr, paperState.plan);
    }

    const btnSim = document.getElementById('btnRerunSim');
    if (btnSim) {
      btnSim.onclick = () => {
        if (!paperState || !paperState.plan) return;
        const metrics = PE.getPlanMetrics(paperState);
        const sim = PE.runPlanSimulation(paperState.plan, paperState.settings.riskProfile, null, {
          remainingDays: metrics.remainingDays,
          currentBank: metrics.totalBank,
          seed: Math.floor(Math.random() * 10000)
        });
        paperState.simulation = {
          lastRunAt: new Date().toISOString(),
          seed: sim.seed,
          result: sim
        };
        saveState();
        renderPlanPane();
      };
    }

    const btnReset = document.getElementById('btnResetPlan');
    if (btnReset) {
      btnReset.onclick = () => {
        if (confirm('Mevcut sanal kasa planını ve kupon geçmişini sıfırlamak istediğinize emin misiniz? Bu işlem geri alınamaz.')) {
          paperState = null;
          try { localStorage.removeItem(PE.STORAGE_KEY); } catch (e) {}
          renderPlanPane();
          renderRecPane();
          renderCouponsPane();
        }
      };
    }

    const btnExport = document.getElementById('btnExportJSON');
    if (btnExport) {
      btnExport.onclick = () => {
        if (!paperState) return;
        const jsonStr = PE.exportPaperState(paperState);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `betavus-paper-export-${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        URL.revokeObjectURL(url);
      };
    }

    const btnImport = document.getElementById('btnImportJSON');
    const fileInput = document.getElementById('jsonFileInput');
    if (btnImport && fileInput) {
      btnImport.onclick = () => fileInput.click();
      fileInput.onchange = (e) => {
        const file = e.target.files && e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => {
          const val = PE.validateImportedJSON(evt.target.result);
          if (!val.valid) {
            alert('İçe aktarma hatası: ' + val.error);
            return;
          }
          if (confirm('İçe aktarılan veriler mevcut planınızın ve kuponlarınızın üzerine yazılacaktır. Onaylıyor musunuz?')) {
            paperState = val.data;
            saveState();
            renderPlanPane();
            renderRecPane();
            renderCouponsPane();
            alert('Veriler başarıyla içe aktarıldı.');
          }
        };
        reader.readAsText(file);
      };
    }

    // Adaptif butonlar
    document.querySelectorAll('.btn-apply-adapt').forEach(b => {
      b.onclick = () => {
        const optId = b.dataset.opt;
        const adaptive = PE.buildAdaptiveOptions(paperState.plan, paperState, null);
        const chosen = adaptive && adaptive.options.find(o => o.id === optId);
        if (!chosen) return;

        let confirmMsg = `"${chosen.title}" alternatifini uygulamak istediğinize emin misiniz?`;
        if (optId === 'change_risk') {
          confirmMsg += `\n\nUYARI: Risk profiliniz "${PE.RISK_PROFILES[chosen.changes.riskProfile].name}" olarak değiştirilecektir. Düşüş ve sermaye kaybı riski artar.`;
        }

        if (confirm(confirmMsg)) {
          if (chosen.changes.durationDays) {
            paperState.plan.durationDays = chosen.changes.durationDays;
          }
          if (chosen.changes.targetBank) {
            paperState.plan.targetBank = chosen.changes.targetBank;
          }
          if (chosen.changes.riskProfile) {
            paperState.settings.riskProfile = chosen.changes.riskProfile;
          }
          saveState();
          renderPlanPane();
          renderRecPane();
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
        </div>
        <div class="rec-actions">
          <button class="btn-sec" id="btnRecChangeRisk" type="button">⚙️ Kasa Modelini Değiştir</button>
        </div>
      </div>

      <div class="rec-cards-grid">
        ${Object.keys(PE.COUPON_CLASSES).map(k => renderRecCardHtml(recs[k], k, curr, prof)).join('')}
      </div>
    `;

    wireRecEvents(recs);
  }

  function renderRecCardHtml(rec, classKey, curr, prof) {
    const cls = PE.COUPON_CLASSES[classKey];
    const armPct = Math.round((prof[cls.armKey] || 0) * 100);

    if (!rec || !rec.available) {
      return `
        <div class="card rec-card unavailable ${cls.id}">
          <div class="rc-head">
            <span class="rc-badge ${cls.badgeClass}">${esc(cls.name)}</span>
            <span class="rc-arm-note">Kasadan ayrılan: %${armPct}</span>
          </div>
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
      <div class="card rec-card ${cls.id}">
        <div class="rc-head">
          <div class="rc-title-box">
            <span class="rc-badge ${cls.badgeClass}">${esc(cls.name)}</span>
            <span class="rc-legs-count">${rec.selections.length} Maç</span>
          </div>
          <span class="rc-arm-note">Risk Kolu: %${armPct} (${formatCurrency(rec.recommendedStake, curr)})</span>
        </div>
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

    if (!paperState) {
      paperState = PE.createInitialState();
    }

    const slips = paperState.slips || [];
    const drafts = slips.filter(s => s.status === 'draft');
    const pending = slips.filter(s => s.status === 'pending');
    const settled = slips.filter(s => s.status === 'won' || s.status === 'lost' || s.status === 'void');

    const curr = paperState.settings.currency || 'EUR';
    const metrics = PE.getPlanMetrics(paperState);

    pane.innerHTML = `
      <div class="paper-disclaimer">
        <span class="p-badge">SANAL KASA SİMÜLASYONU</span>
        <p><b>BETAVUS</b> kupon oynatmaz veya ödeme almaz. Bu ekranda planınıza eklediğiniz sanal kuponların (paper slips) durumunu, oranlarını ve kasa hareketlerini takip edersiniz.</p>
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

  function wireCouponsEvents() {
    // Alt sekmeler
    ['pending', 'settled', 'drafts', 'model12'].forEach(key => {
      const btn = document.getElementById('csub-' + key);
      if (btn) {
        btn.onclick = () => {
          activeCpnSubtab = key;
          renderCouponsPane();
          if (key === 'model12' && typeof root.renderCoupons === 'function') {
            root.renderCoupons();
          }
        };
      }
    });

    if (activeCpnSubtab === 'model12' && typeof root.renderCoupons === 'function') {
      root.renderCoupons();
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
