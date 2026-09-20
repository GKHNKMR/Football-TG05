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
  // Kasa Planım Ekranı (#pane-plan)
  // ---------------------------------------------------------------------------

  function renderPlanPane() {
    const pane = document.getElementById('pane-plan');
    if (!pane) return;

    if (!paperState || !paperState.plan) {
      pane.innerHTML = renderPlanSetupHtml();
      wirePlanSetupEvents();
      return;
    }

    const metrics = PE.getPlanMetrics(paperState);
    const curr = paperState.settings.currency || 'EUR';
    const prof = PE.RISK_PROFILES[paperState.settings.riskProfile] || PE.RISK_PROFILES.cautious;

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

    pane.innerHTML = `
      <div class="paper-disclaimer">
        <span class="p-badge">SANAL KASA SİMÜLASYONU</span>
        <p><b>BETAVUS</b> bahis kabul etmez, ödeme almaz ve kupon oynatmaz. Gösterilen kasa, stake ve getiriler sanaldır. Tahminler olasılıksaldır ve sonuç garantisi vermez.</p>
        <div class="p-quote">« Önce simüle et. Riskini gör. Stratejini ölç. Sonra karar ver. »</div>
      </div>

      <div class="plan-header-card">
        <div class="plan-title-row">
          <div>
            <h2>Sanal Kasa Planım · ${esc(prof.name)} Profil</h2>
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
            <label>Genel Risk Toleransı *</label>
            <div class="risk-cards">
              <label class="risk-card active">
                <input type="radio" name="setupRisk" value="cautious" checked>
                <div class="r-head">
                  <b>Temkinli / Minimum Risk (Varsayılan)</b>
                  <span class="r-badge b-cautious">Düşük Risk</span>
                </div>
                <p>Sermaye koruma odaklı. %75 Kasa Rezervinde kalır, %20 Minimum Risk koluna (0.5 Üst), %5 Orta Risk koluna ayrılır.</p>
              </label>
              <label class="risk-card">
                <input type="radio" name="setupRisk" value="balanced">
                <div class="r-head">
                  <b>Dengeli / Medium</b>
                  <span class="r-badge b-balanced">Dengeli</span>
                </div>
                <p>Büyüme ve koruma dengesi. %50 Kasa Rezervi, %30 Minimum Risk, %16 Orta Risk, %4 Yüksek Risk.</p>
              </label>
              <label class="risk-card">
                <input type="radio" name="setupRisk" value="aggressive">
                <div class="r-head">
                  <b>Agresif</b>
                  <span class="r-badge b-aggressive">Yüksek Varyans</span>
                </div>
                <p>Yüksek getiri & yüksek çekilme riski. %35 Kasa Rezervi, %40 Minimum Risk, %18 Orta Risk, %7 Yüksek Risk.</p>
              </label>
            </div>
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
    const pills = document.querySelectorAll('#durationPills .pill');
    const durInput = document.getElementById('setupDuration');
    pills.forEach(p => {
      p.onclick = () => {
        pills.forEach(x => x.classList.remove('active'));
        p.classList.add('active');
        if (durInput) durInput.value = p.dataset.days;
      };
    });
    if (durInput) {
      durInput.oninput = () => {
        pills.forEach(x => x.classList.toggle('active', x.dataset.days === durInput.value));
      };
    }

    const rCards = document.querySelectorAll('.risk-card');
    rCards.forEach(rc => {
      rc.onclick = () => {
        rCards.forEach(x => x.classList.remove('active'));
        rc.classList.add('active');
        const radio = rc.querySelector('input[type="radio"]');
        if (radio) radio.checked = true;
      };
    });

    const btn = document.getElementById('btnCreatePlan');
    if (btn) {
      btn.onclick = () => {
        const start = parseNumber(document.getElementById('setupStartBank')?.value);
        const target = parseNumber(document.getElementById('setupTargetBank')?.value);
        const duration = parseInt(document.getElementById('setupDuration')?.value, 10);
        const curr = document.getElementById('setupCurrency')?.value || 'EUR';
        const riskRadio = document.querySelector('input[name="setupRisk"]:checked');
        const risk = riskRadio ? riskRadio.value : 'cautious';
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
    const profKey = (paperState && paperState.settings && paperState.settings.riskProfile) || 'cautious';
    const prof = PE.RISK_PROFILES[profKey] || PE.RISK_PROFILES.cautious;
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
          <p>Seçili Risk Profiliniz: <b>${esc(prof.name)}</b> (%${Math.round(prof.reservePct * 100)} Rezervde · Kullanılabilir: ${formatCurrency(available, curr)})</p>
        </div>
        <div class="rec-actions">
          <button class="btn-sec" id="btnRecChangeRisk" type="button">⚙️ Risk Profilini Değiştir</button>
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
        const pKeys = Object.keys(PE.RISK_PROFILES);
        const cur = (paperState && paperState.settings.riskProfile) || 'cautious';
        const next = cur === 'cautious' ? 'balanced' : cur === 'balanced' ? 'aggressive' : 'cautious';
        if (confirm(`Risk profilinizi "${PE.RISK_PROFILES[next].name}" olarak değiştirmek istiyor musunuz?`)) {
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
      riskProfile: (paperState && paperState.settings && paperState.settings.riskProfile) || slipOrRec.riskProfile || 'cautious',
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
    openCouponEditor
  };

  // Sayfa yüklendiğinde otomatik başlat
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})(typeof window !== 'undefined' ? window : this);
