/**
 * BETAVUS — Çifte Şans & Skor Tahminleri Kullanıcı Arayüzü (cifte_ui.js)
 * 
 * Hem güncel maç bülteni tahminlerini hem de 5 sezonluk (17.003 maç)
 * Çifte Şans ve Skor Model Doğruluğu (Backtest) analizini sunar.
 */

(function (global) {
  'use strict';

  let cifteSubView = 'pred'; // 'pred' (Güncel Tahminler) | 'bt' (Model Doğruluğu)
  let cifteFilter = 'all';   // 'all', 'high_conf', 'dc_1x', 'dc_12', 'dc_x2'
  let cifteSearch = '';
  let btLeagueFilter = 'Tümü';

  function esc(s) {
    if (!s) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function fmtPct(val) {
    return (Number(val) || 0).toLocaleString('tr-TR', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + '%';
  }

  function num(val) {
    return (Number(val) || 0).toLocaleString('tr-TR');
  }

  function getFlag(league) {
    if (typeof global.flag === 'function') return global.flag(league, 12);
    return '⚽';
  }

  function getLocalDate(iso) {
    if (typeof global.localDate === 'function') return global.localDate(iso);
    if (!iso) return '';
    try {
      const d = new Date(iso);
      return d.toLocaleDateString('tr-TR', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      return iso;
    }
  }

  function dmy(iso) {
    if (!iso) return '—';
    const parts = iso.split('-');
    if (parts.length === 3) return `${parts[2]}.${parts[1]}.${parts[0].slice(2)}`;
    return iso;
  }

  function renderCiftePane() {
    const pane = document.getElementById('pane-cifte');
    if (!pane) return;

    if (cifteSubView === 'bt') {
      renderCifteBacktestView(pane);
    } else {
      renderCiftePredictionsView(pane);
    }
  }

  // ===========================================================================
  // 1. GÜNCEL MAÇ TAHMİNLERİ GÖRÜNÜMÜ
  // ===========================================================================

  function renderCiftePredictionsView(pane) {
    const rawMatches = window.__data || [];
    const CE = global.BETAVUS_CIFTE;
    if (!CE) {
      pane.innerHTML = '<div class="empty">Analiz motoru yükleniyor…</div>';
      return;
    }

    const currentLeague = (window.selectedByTab && window.selectedByTab['cifte']) || window.selected || 'Tümü';
    let filtered = rawMatches;
    if (currentLeague && currentLeague !== 'Tümü') {
      filtered = filtered.filter(m => m.league === currentLeague);
    }

    const analyzed = filtered.map(m => CE.analyzeMatch(m)).filter(Boolean);

    let displayList = analyzed;
    if (cifteFilter === 'high_conf') {
      displayList = displayList.filter(a => a.bestDc.pct >= 75.0);
    } else if (cifteFilter === 'dc_1x') {
      displayList = displayList.filter(a => a.bestDc.pick === '1X');
    } else if (cifteFilter === 'dc_12') {
      displayList = displayList.filter(a => a.bestDc.pick === '12');
    } else if (cifteFilter === 'dc_x2') {
      displayList = displayList.filter(a => a.bestDc.pick === 'X2');
    }

    if (cifteSearch) {
      const q = cifteSearch.toLowerCase();
      displayList = displayList.filter(a =>
        a.home.toLowerCase().includes(q) ||
        a.away.toLowerCase().includes(q) ||
        a.league.toLowerCase().includes(q)
      );
    }

    const totalCount = analyzed.length;
    const highConfCount = analyzed.filter(a => a.bestDc.pct >= 75.0).length;
    const avgBestDc = totalCount > 0 ? (analyzed.reduce((s, a) => s + a.bestDc.pct, 0) / totalCount).toFixed(1) : '0.0';

    let html = `
      <div class="cifte-container" style="max-width:1100px;margin:0 auto;padding:12px 14px;">
        
        <!-- Alt Sekme Geçiş Butonları (Tahminler vs Model Doğruluğu) -->
        <div class="subtabs-bar" style="display:flex;gap:10px;margin-bottom:16px;">
          <button class="chip active subtab-toggle" data-view="pred" type="button" style="padding:7px 16px;font-size:12.5px;font-weight:800;border-radius:8px;cursor:pointer;">
            ⚽ Güncel Fikstür &amp; Tahminler (${totalCount})
          </button>
          <button class="chip subtab-toggle" data-view="bt" type="button" style="padding:7px 16px;font-size:12.5px;font-weight:800;border-radius:8px;cursor:pointer;background:rgba(255,255,255,0.05);color:var(--text);border:1px solid rgba(255,255,255,0.15);">
            📊 Çifte Şans &amp; Skor Model Doğruluğu (17.003 Maç)
          </button>
        </div>

        <!-- Tanıtım ve Bilgi Başlığı -->
        <div class="cifte-intro-card" style="background:linear-gradient(135deg,rgba(15,23,42,0.85),rgba(30,41,59,0.7));border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:16px 20px;margin-bottom:18px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
          <div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
              <span style="font-size:20px;">🎲</span>
              <h2 style="margin:0;font-size:18px;font-weight:900;color:var(--text);">Çifte Şans &amp; Skor Tahminleri</h2>
              <span class="cms-badge" style="background:rgba(16,185,129,0.15);color:#10b981;border:1px solid rgba(16,185,129,0.3);font-size:11px;padding:2px 8px;border-radius:99px;font-weight:700;">Dixon-Coles Matrisi</span>
            </div>
            <p style="margin:0;font-size:12.5px;color:var(--muted);max-width:650px;line-height:1.45;">
              Geçmiş takım formları, hücum/savunma parametreleri ve beklenen gollerden (λ) hesaplanan <b>1X (1-0), 12 (1-2), X2 (0-2)</b> çifte şans olasılıkları ve en muhtemel kesin skor projeksiyonları.
            </p>
          </div>

          <!-- Mini İstatistik Kutuları -->
          <div style="display:flex;gap:12px;flex-wrap:wrap;">
            <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:8px 14px;text-align:center;min-width:85px;">
              <div style="font-size:18px;font-weight:900;color:var(--accent);">${totalCount}</div>
              <div style="font-size:10.5px;color:var(--muted);margin-top:2px;">Analiz Maçı</div>
            </div>
            <div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.25);border-radius:10px;padding:8px 14px;text-align:center;min-width:95px;">
              <div style="font-size:18px;font-weight:900;color:#10b981;">${highConfCount}</div>
              <div style="font-size:10.5px;color:var(--muted);margin-top:2px;">Yüksek Güven (≥%75)</div>
            </div>
            <div style="background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.25);border-radius:10px;padding:8px 14px;text-align:center;min-width:90px;">
              <div style="font-size:18px;font-weight:900;color:#fbbf24;">%${avgBestDc}</div>
              <div style="font-size:10.5px;color:var(--muted);margin-top:2px;">Ort. Güçlü ÇŞ</div>
            </div>
          </div>
        </div>

        <!-- Filtre ve Arama Çubuğu -->
        <div class="cifte-filter-bar" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:16px;">
          <div class="cifte-chips" style="display:flex;gap:8px;flex-wrap:wrap;">
            <button class="chip ${cifteFilter === 'all' ? 'active' : ''}" data-cf="all" type="button" style="cursor:pointer;">🌐 Tümü (${totalCount})</button>
            <button class="chip ${cifteFilter === 'high_conf' ? 'active' : ''}" data-cf="high_conf" type="button" style="cursor:pointer;">⭐ Yüksek Güven (≥%75)</button>
            <button class="chip ${cifteFilter === 'dc_1x' ? 'active' : ''}" data-cf="dc_1x" type="button" style="cursor:pointer;">🛡️ 1X (Ev/Beraberlik)</button>
            <button class="chip ${cifteFilter === 'dc_12' ? 'active' : ''}" data-cf="dc_12" type="button" style="cursor:pointer;">⚡ 12 (Kazanır / Berabere Bitmez)</button>
            <button class="chip ${cifteFilter === 'dc_x2' ? 'active' : ''}" data-cf="dc_x2" type="button" style="cursor:pointer;">🚀 X2 (Beraberlik/Deplasman)</button>
          </div>

          <div style="position:relative;width:240px;">
            <input type="text" id="qCifte" placeholder="Takım veya lig ara…" value="${esc(cifteSearch)}" style="width:100%;padding:7px 12px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:8px;color:var(--text);font-size:12px;outline:none;" />
            ${cifteSearch ? '<span id="qCifteClear" style="position:absolute;right:8px;top:7px;cursor:pointer;color:var(--muted);font-size:13px;">✕</span>' : ''}
          </div>
        </div>

        <!-- Maç Kartları Listesi -->
        <div class="cifte-cards-list" style="display:flex;flex-direction:column;gap:14px;">
    `;

    if (!displayList.length) {
      html += `
        <div class="empty" style="text-align:center;padding:40px;background:rgba(255,255,255,0.02);border-radius:12px;border:1px dashed rgba(255,255,255,0.1);">
          <strong style="font-size:15px;display:block;margin-bottom:6px;">Seçilen kriterlere uygun maç bulunamadı</strong>
          <span style="color:var(--muted);font-size:12px;">Arama teriminizi temizlemeyi veya filtreyi değiştirmeyi deneyebilirsiniz.</span>
        </div>
      `;
    } else {
      html += displayList.map(item => renderMatchCardHtml(item)).join('');
    }

    html += `
        </div>
      </div>
    `;

    pane.innerHTML = html;
    wireCifteEvents(pane);
  }

  function renderMatchCardHtml(a) {
    const isHighConf = a.bestDc.pct >= 75.0;
    const bestPick = a.bestDc.pick;
    const top4Scores = a.topScores.slice(0, 4);

    return `
      <div class="card cifte-match-card" style="background:rgba(18,24,38,0.75);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:16px 18px;position:relative;transition:border-color .2s;">
        
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:12px;">
          <div style="display:flex;align-items:center;gap:7px;font-size:12px;color:var(--muted);font-weight:600;">
            <span>${getFlag(a.league)}</span>
            <span style="color:var(--text);font-weight:700;">${esc(a.league)}</span>
            <span>·</span>
            <span>📅 ${getLocalDate(a.kickoff)}</span>
          </div>

          <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:11px;color:var(--muted);">Beklenen Gol (λ): <b style="color:var(--text);">${a.expGoals}</b> (${a.lamHome} vs ${a.lamAway})</span>
            ${isHighConf ? `<span class="cms-badge" style="background:rgba(16,185,129,0.15);color:#10b981;border:1px solid rgba(16,185,129,0.3);font-size:10.5px;padding:2px 8px;border-radius:6px;font-weight:800;">⭐ Yüksek Güven</span>` : ''}
          </div>
        </div>

        <div style="display:grid;grid-template-columns:1.2fr 1fr;gap:16px;align-items:center;margin-bottom:16px;">
          <div>
            <div style="font-size:16px;font-weight:900;color:var(--text);letter-spacing:0.2px;">
              ${esc(a.home)} <span style="color:var(--muted);font-weight:400;margin:0 4px;">—</span> ${esc(a.away)}
            </div>
            <div style="display:flex;gap:12px;margin-top:6px;font-size:11px;color:var(--muted);">
              <span>1: <b style="color:var(--text);">%${a.prob1}</b></span>
              <span>X: <b style="color:var(--text);">%${a.probX}</b></span>
              <span>2: <b style="color:var(--text);">%${a.prob2}</b></span>
            </div>
          </div>

          <div>
            <div style="height:8px;border-radius:4px;overflow:hidden;display:flex;background:rgba(255,255,255,0.06);">
              <div style="width:${a.prob1}%;background:#3b82f6;" title="Ev Sahibi: %${a.prob1}"></div>
              <div style="width:${a.probX}%;background:#94a3b8;" title="Beraberlik: %${a.probX}"></div>
              <div style="width:${a.prob2}%;background:#ef4444;" title="Deplasman: %${a.prob2}"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--muted);margin-top:4px;">
              <span style="color:#3b82f6;">Ev (%${a.prob1})</span>
              <span style="color:#94a3b8;">Beraberlik (%${a.probX})</span>
              <span style="color:#ef4444;">Deplasman (%${a.prob2})</span>
            </div>
          </div>
        </div>

        <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:10px;margin-bottom:14px;">
          
          <div class="cifte-box ${bestPick === '1X' ? 'is-best' : ''}" style="background:${bestPick === '1X' ? 'rgba(16,185,129,0.08)' : 'rgba(255,255,255,0.025)'};border:1px solid ${bestPick === '1X' ? 'rgba(16,185,129,0.45)' : 'rgba(255,255,255,0.08)'};border-radius:10px;padding:10px 12px;position:relative;">
            ${bestPick === '1X' ? '<span style="position:absolute;top:-8px;right:10px;background:#10b981;color:#0d1219;font-size:9.5px;font-weight:900;padding:1px 6px;border-radius:4px;">MODEL TERCİHİ</span>' : ''}
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <b style="font-size:14px;color:${bestPick === '1X' ? '#10b981' : 'var(--text)'};">1X</b>
              <span style="font-size:16px;font-weight:900;color:${bestPick === '1X' ? '#10b981' : 'var(--text)'};">${fmtPct(a.dc['1X'].pct)}</span>
            </div>
            <div style="font-size:11px;color:var(--muted);margin-top:2px;">Ev Sahibi veya Beraberlik (1-0)</div>
            ${a.dc['1X'].odds ? `<div style="font-size:10.5px;color:var(--accent);margin-top:4px;">Piyasa Oranı: <b>${a.dc['1X'].odds}</b></div>` : ''}
          </div>

          <div class="cifte-box ${bestPick === '12' ? 'is-best' : ''}" style="background:${bestPick === '12' ? 'rgba(16,185,129,0.08)' : 'rgba(255,255,255,0.025)'};border:1px solid ${bestPick === '12' ? 'rgba(16,185,129,0.45)' : 'rgba(255,255,255,0.08)'};border-radius:10px;padding:10px 12px;position:relative;">
            ${bestPick === '12' ? '<span style="position:absolute;top:-8px;right:10px;background:#10b981;color:#0d1219;font-size:9.5px;font-weight:900;padding:1px 6px;border-radius:4px;">MODEL TERCİHİ</span>' : ''}
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <b style="font-size:14px;color:${bestPick === '12' ? '#10b981' : 'var(--text)'};">12</b>
              <span style="font-size:16px;font-weight:900;color:${bestPick === '12' ? '#10b981' : 'var(--text)'};">${fmtPct(a.dc['12'].pct)}</span>
            </div>
            <div style="font-size:11px;color:var(--muted);margin-top:2px;">Ev veya Deplasman (1-2)</div>
            ${a.dc['12'].odds ? `<div style="font-size:10.5px;color:var(--accent);margin-top:4px;">Piyasa Oranı: <b>${a.dc['12'].odds}</b></div>` : ''}
          </div>

          <div class="cifte-box ${bestPick === 'X2' ? 'is-best' : ''}" style="background:${bestPick === 'X2' ? 'rgba(16,185,129,0.08)' : 'rgba(255,255,255,0.025)'};border:1px solid ${bestPick === 'X2' ? 'rgba(16,185,129,0.45)' : 'rgba(255,255,255,0.08)'};border-radius:10px;padding:10px 12px;position:relative;">
            ${bestPick === 'X2' ? '<span style="position:absolute;top:-8px;right:10px;background:#10b981;color:#0d1219;font-size:9.5px;font-weight:900;padding:1px 6px;border-radius:4px;">MODEL TERCİHİ</span>' : ''}
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <b style="font-size:14px;color:${bestPick === 'X2' ? '#10b981' : 'var(--text)'};">X2</b>
              <span style="font-size:16px;font-weight:900;color:${bestPick === 'X2' ? '#10b981' : 'var(--text)'};">${fmtPct(a.dc['X2'].pct)}</span>
            </div>
            <div style="font-size:11px;color:var(--muted);margin-top:2px;">Beraberlik veya Deplasman (0-2)</div>
            ${a.dc['X2'].odds ? `<div style="font-size:10.5px;color:var(--accent);margin-top:4px;">Piyasa Oranı: <b>${a.dc['X2'].odds}</b></div>` : ''}
          </div>

        </div>

        <div style="background:rgba(0,0,0,0.22);border-radius:8px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <span style="font-size:11px;font-weight:700;color:var(--muted);">🎯 En Olası Skorlar:</span>
            ${top4Scores.map(sc => `
              <span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);padding:2px 8px;border-radius:6px;font-size:11px;">
                <b style="color:#fbbf24;">${sc.score}</b> <span style="color:var(--muted);font-size:10px;">(%${sc.pct})</span>
              </span>
            `).join('')}
          </div>

          <div style="display:flex;align-items:center;gap:12px;font-size:11px;color:var(--muted);">
            <span>0.5Ü: <b style="color:${a.goals.o05 >= 90 ? '#10b981' : 'var(--text)'};">%${a.goals.o05}</b></span>
            <span>1.5Ü: <b style="color:${a.goals.o15 >= 80 ? '#3b82f6' : 'var(--text)'};">%${a.goals.o15}</b></span>
            <span>2.5Ü: <b style="color:${a.goals.o25 >= 60 ? '#f59e0b' : 'var(--text)'};">%${a.goals.o25}</b></span>
            <span>KG Var: <b style="color:${a.goals.bttsYes >= 55 ? '#a855f7' : 'var(--text)'};">%${a.goals.bttsYes}</b></span>
          </div>
        </div>

      </div>
    `;
  }

  // ===========================================================================
  // 2. ÇİFTE ŞANS & SKOR MODEL DOĞRULUĞU (BACKTEST) GÖRÜNÜMÜ
  // ===========================================================================

  function renderCifteBacktestView(pane) {
    const data = window.CIFTE_BACKTEST_DATA;
    if (!data) {
      pane.innerHTML = '<div class="empty">Model doğrulama verisi yükleniyor…</div>';
      return;
    }

    // Aktif lig filtresine göre istatistik seçimi
    const isFiltered = (btLeagueFilter && btLeagueFilter !== 'Tümü');
    const st = isFiltered ? (data.by_league[btLeagueFilter] || data.overall) : data.overall;

    const leaguesList = [
      'Premier League', 'Championship', 'LaLiga', 'Bundesliga',
      'Serie A', 'Ligue 1', 'Eredivisie', 'Turkish Süper Lig', 'Primeira Liga'
    ];

    const seasonList = ['2021/22', '2022/23', '2023/24', '2024/25', '2025/26', '2026/27'];

    let html = `
      <div class="cifte-container" style="max-width:1100px;margin:0 auto;padding:12px 14px;">
        
        <!-- Alt Sekme Geçiş Butonları -->
        <div class="subtabs-bar" style="display:flex;gap:10px;margin-bottom:16px;">
          <button class="chip subtab-toggle" data-view="pred" type="button" style="padding:7px 16px;font-size:12.5px;font-weight:800;border-radius:8px;cursor:pointer;background:rgba(255,255,255,0.05);color:var(--text);border:1px solid rgba(255,255,255,0.15);">
            ⚽ Güncel Fikstür &amp; Tahminler
          </button>
          <button class="chip active subtab-toggle" data-view="bt" type="button" style="padding:7px 16px;font-size:12.5px;font-weight:800;border-radius:8px;cursor:pointer;">
            📊 Çifte Şans &amp; Skor Model Doğruluğu (17.003 Maç)
          </button>
        </div>

        <!-- Üst Başlık ve Açıklama -->
        <div class="card" style="margin-bottom:16px;background:var(--panel2);padding:16px 20px;border-radius:12px;">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
            <div>
              <h2 style="margin:0 0 4px;font-size:17px;font-weight:900;">📊 Çifte Şans &amp; Skor Model Doğruluğu (5 Sezonluk Backtest)</h2>
              <p style="margin:0;font-size:12px;color:var(--muted);line-height:1.5;">
                5 sezonluk (<b>17.003 maç</b>) sızıntısız walk-forward doğrulama. Her maçın tahmini yalnızca o maçtan önceki verilerle ve Dixon-Coles skor matrisiyle üretilmiştir.
              </p>
            </div>
            ${isFiltered ? `
              <button class="chip active" id="btnResetCifteLeague" type="button" style="cursor:pointer;background:rgba(239,68,68,0.15);color:#ef4444;border:1px solid rgba(239,68,68,0.3);">
                Filtreyi Sıfırla (Tümü) ✕
              </button>
            ` : ''}
          </div>
        </div>

        <!-- 4 Büyük KPI Kartı (0.5, 1.5, 2.5 Gibi!) -->
        <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:12px;margin-bottom:16px;">
          
          <!-- KPI 1: 1X Çifte Şans -->
          <div class="card" style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.25);border-radius:12px;padding:14px;text-align:center;">
            <div style="font-size:12px;font-weight:800;color:#10b981;margin-bottom:4px;">🛡️ 1X Çifte Şans (≥%75)</div>
            <div style="font-size:28px;font-weight:900;color:#10b981;margin:4px 0;">%${st.dc_1x_pct}</div>
            <div style="font-size:11px;color:var(--text);font-weight:600;">${num(st.dc_1x_h)} / ${num(st.dc_1x_n)} Maç Tuttu</div>
            <div style="font-size:10px;color:var(--muted);margin-top:3px;">${num(st.dc_1x_n - st.dc_1x_h)} Iska · Ev Sahibi / Beraberlik</div>
          </div>

          <!-- KPI 2: 12 Çifte Şans -->
          <div class="card" style="background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.25);border-radius:12px;padding:14px;text-align:center;">
            <div style="font-size:12px;font-weight:800;color:#3b82f6;margin-bottom:4px;">⚡ 12 Çifte Şans (≥%75)</div>
            <div style="font-size:28px;font-weight:900;color:#3b82f6;margin:4px 0;">%${st.dc_12_pct}</div>
            <div style="font-size:11px;color:var(--text);font-weight:600;">${num(st.dc_12_h)} / ${num(st.dc_12_n)} Maç Tuttu</div>
            <div style="font-size:10px;color:var(--muted);margin-top:3px;">${num(st.dc_12_n - st.dc_12_h)} Iska · Berabere Bitmez</div>
          </div>

          <!-- KPI 3: X2 Çifte Şans -->
          <div class="card" style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.25);border-radius:12px;padding:14px;text-align:center;">
            <div style="font-size:12px;font-weight:800;color:#ef4444;margin-bottom:4px;">🚀 X2 Çifte Şans (≥%75)</div>
            <div style="font-size:28px;font-weight:900;color:#ef4444;margin:4px 0;">%${st.dc_x2_pct}</div>
            <div style="font-size:11px;color:var(--text);font-weight:600;">${num(st.dc_x2_h)} / ${num(st.dc_x2_n)} Maç Tuttu</div>
            <div style="font-size:10px;color:var(--muted);margin-top:3px;">${num(st.dc_x2_n - st.dc_x2_h)} Iska · Beraberlik / Deplasman</div>
          </div>

          <!-- KPI 4: Skor Tahmini -->
          <div class="card" style="background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.25);border-radius:12px;padding:14px;text-align:center;">
            <div style="font-size:12px;font-weight:800;color:#fbbf24;margin-bottom:4px;">🎯 Skor Tahmini (Top-3)</div>
            <div style="font-size:28px;font-weight:900;color:#fbbf24;margin:4px 0;">%${st.sc_top3_pct}</div>
            <div style="font-size:11px;color:var(--text);font-weight:600;">${num(st.sc_top3_h)} / ${num(st.total)} Skor Havuzu</div>
            <div style="font-size:10px;color:var(--muted);margin-top:3px;">Top-1 Birebir Skor: <b>%${st.sc_top1_pct}</b> (${num(st.sc_top1_h)} Maç)</div>
          </div>

        </div>

        <!-- 1. Lig Bazında Başarı Tablosu -->
        <div class="card" style="margin-bottom:16px;">
          <h2 style="font-size:15px;margin:0 0 6px;">Lig Bazında Çifte Şans &amp; Skor Başarısı ${isFiltered ? `· ${esc(btLeagueFilter)}` : '· Tüm Ligler'}</h2>
          <p style="font-size:11.5px;color:var(--muted);margin:0 0 10px;">
            Lig satırlarına tıklayarak yukarıdaki KPI kartlarını doğrudan seçtiğiniz lige süzebilirsiniz.
          </p>
          <div class="tbl-scroll">
            <table class="bt-table" id="tblCifteLeague">
              <thead>
                <tr style="font-size:11px;">
                  <th rowspan="2" style="text-align:left;vertical-align:bottom;padding:6px 8px;">Lig</th>
                  <th rowspan="2" style="vertical-align:bottom;padding:6px 4px;">Toplam Maç</th>
                  <th colspan="3" style="border-bottom:2px solid #10b981;color:#10b981;padding:4px;">1X (≥%75)</th>
                  <th colspan="3" style="border-bottom:2px solid #3b82f6;color:#3b82f6;padding:4px;">12 (≥%75)</th>
                  <th colspan="3" style="border-bottom:2px solid #ef4444;color:#ef4444;padding:4px;">X2 (≥%75)</th>
                  <th colspan="2" style="border-bottom:2px solid #fbbf24;color:#fbbf24;padding:4px;">Skor İsabeti</th>
                </tr>
                <tr style="font-size:10px;color:var(--muted);">
                  <th>Vurgu</th><th>Tuttu</th><th>%</th>
                  <th>Vurgu</th><th>Tuttu</th><th>%</th>
                  <th>Vurgu</th><th>Tuttu</th><th>%</th>
                  <th>Top-1 %</th><th>Top-3 %</th>
                </tr>
              </thead>
              <tbody>
                <!-- Genel Satır -->
                <tr class="clickable bt-cifte-league-row ${btLeagueFilter === 'Tümü' ? 'active-row' : ''}" data-league="Tümü" style="font-weight:800;background:rgba(255,255,255,0.03);">
                  <td style="text-align:left;padding:6px 8px;">${btLeagueFilter === 'Tümü' ? '👉 ' : ''}⭐ Tüm Ligler (Genel Toplam)</td>
                  <td>${num(data.overall.total)}</td>
                  <td>${num(data.overall.dc_1x_n)}</td><td class="good">${num(data.overall.dc_1x_h)}</td><td class="good"><b>%${data.overall.dc_1x_pct}</b></td>
                  <td>${num(data.overall.dc_12_n)}</td><td class="good">${num(data.overall.dc_12_h)}</td><td class="good"><b>%${data.overall.dc_12_pct}</b></td>
                  <td>${num(data.overall.dc_x2_n)}</td><td class="good">${num(data.overall.dc_x2_h)}</td><td class="good"><b>%${data.overall.dc_x2_pct}</b></td>
                  <td style="color:#fbbf24;">%${data.overall.sc_top1_pct}</td><td style="color:#fbbf24;"><b>%${data.overall.sc_top3_pct}</b></td>
                </tr>
                ${leaguesList.map(lg => {
                  const lSt = data.by_league[lg] || emptyStat();
                  const isAct = (btLeagueFilter === lg);
                  return `
                    <tr class="clickable bt-cifte-league-row ${isAct ? 'active-row' : ''}" data-league="${esc(lg)}" style="cursor:pointer;">
                      <td style="text-align:left;padding:6px 8px;font-weight:${isAct ? '900' : '600'};">${isAct ? '👉 ' : ''}${getFlag(lg)} ${esc(lg)}</td>
                      <td>${num(lSt.total)}</td>
                      <td>${num(lSt.dc_1x_n)}</td><td>${num(lSt.dc_1x_h)}</td><td class="good"><b>%${lSt.dc_1x_pct}</b></td>
                      <td>${num(lSt.dc_12_n)}</td><td>${num(lSt.dc_12_h)}</td><td class="good"><b>%${lSt.dc_12_pct}</b></td>
                      <td>${num(lSt.dc_x2_n)}</td><td>${num(lSt.dc_x2_h)}</td><td class="good"><b>%${lSt.dc_x2_pct}</b></td>
                      <td style="color:#fbbf24;">%${lSt.sc_top1_pct}</td><td style="color:#fbbf24;"><b>%${lSt.sc_top3_pct}</b></td>
                    </tr>
                  `;
                }).join('')}
              </tbody>
            </table>
          </div>
        </div>

        <!-- 2. Sezon Bazında Tablo -->
        <div class="card" style="margin-bottom:16px;">
          <h2 style="font-size:15px;margin:0 0 6px;">Sezon Bazında Çifte Şans &amp; Skor Başarısı ${isFiltered ? `· ${esc(btLeagueFilter)}` : '· Tüm Ligler'}</h2>
          <div class="tbl-scroll">
            <table class="bt-table">
              <thead>
                <tr style="font-size:11px;">
                  <th rowspan="2" style="text-align:left;vertical-align:bottom;padding:6px 8px;">Sezon</th>
                  <th rowspan="2" style="vertical-align:bottom;padding:6px 4px;">Maç</th>
                  <th colspan="3" style="border-bottom:2px solid #10b981;color:#10b981;padding:4px;">1X (≥%75)</th>
                  <th colspan="3" style="border-bottom:2px solid #3b82f6;color:#3b82f6;padding:4px;">12 (≥%75)</th>
                  <th colspan="3" style="border-bottom:2px solid #ef4444;color:#ef4444;padding:4px;">X2 (≥%75)</th>
                  <th colspan="2" style="border-bottom:2px solid #fbbf24;color:#fbbf24;padding:4px;">Skor İsabeti</th>
                </tr>
                <tr style="font-size:10px;color:var(--muted);">
                  <th>Vurgu</th><th>Tuttu</th><th>%</th>
                  <th>Vurgu</th><th>Tuttu</th><th>%</th>
                  <th>Vurgu</th><th>Tuttu</th><th>%</th>
                  <th>Top-1 %</th><th>Top-3 %</th>
                </tr>
              </thead>
              <tbody>
                ${seasonList.map(sz => {
                  let sSt = isFiltered
                    ? ((data.by_season_league[sz] && data.by_season_league[sz][btLeagueFilter]) || emptyStat())
                    : (data.by_season[sz] || emptyStat());
                  const isCurrent = (sz === '2026/27');
                  return `
                    <tr style="${isCurrent ? 'background:rgba(255,255,255,0.03);font-weight:800;' : ''}">
                      <td style="text-align:left;padding:5px 8px;${isCurrent ? 'color:var(--accent);' : ''}">${sz} ${isCurrent ? '(Güncel)' : ''}</td>
                      <td>${num(sSt.total)}</td>
                      <td>${num(sSt.dc_1x_n)}</td><td>${num(sSt.dc_1x_h)}</td><td class="good"><b>%${sSt.dc_1x_pct}</b></td>
                      <td>${num(sSt.dc_12_n)}</td><td>${num(sSt.dc_12_h)}</td><td class="good"><b>%${sSt.dc_12_pct}</b></td>
                      <td>${num(sSt.dc_x2_n)}</td><td>${num(sSt.dc_x2_h)}</td><td class="good"><b>%${sSt.dc_x2_pct}</b></td>
                      <td style="color:#fbbf24;">%${sSt.sc_top1_pct}</td><td style="color:#fbbf24;"><b>%${sSt.sc_top3_pct}</b></td>
                    </tr>
                  `;
                }).join('')}
              </tbody>
            </table>
          </div>
        </div>

        <!-- 3. Örnek Maçlar — Tahmin vs Gerçek Skor -->
        <div class="card" style="margin-bottom:16px;">
          <h2 style="font-size:15px;margin:0 0 6px;">Örnek Maçlar — Çifte Şans &amp; Skor Doğrulama Havuzu</h2>
          <p style="font-size:11.5px;color:var(--muted);margin:0 0 10px;">
            Geçmiş maç havuzundan düzenli aralıklarla çekilen maçlarda modelin o maç için öngördüğü en güçlü ÇŞ seçimi ve en olası ilk 3 skorunun gerçek sonuçla karşılaştırılması.
          </p>
          <div class="tbl-scroll">
            <table class="bt-table">
              <thead>
                <tr>
                  <th>Tarih</th><th>Lig</th><th>Maç</th><th>Gerçek Skor</th>
                  <th>Model ÇŞ Tercihi</th><th>ÇŞ İsabeti</th><th>En Olası İlk 3 Skor</th><th>Skor İsabeti</th>
                </tr>
              </thead>
              <tbody>
                ${(data.samples || []).filter(x => !isFiltered || x.league === btLeagueFilter).slice(0, 50).map(s => `
                  <tr>
                    <td>${dmy(s.date)}</td>
                    <td>${esc(s.league)}</td>
                    <td style="text-align:left;">${esc(s.home)} — ${esc(s.away)}</td>
                    <td><b style="font-size:13px;">${s.actual_score}</b></td>
                    <td><span class="badge ${s.best_dc === '1X' ? 'b-min' : s.best_dc === '12' ? 'b-med' : 'b-high'}">${s.best_dc} (%${s.best_dc_pct})</span></td>
                    <td><b class="${s.dc_hit ? 'good' : 'bad'}">${s.dc_hit ? '✓ TUTTU' : '✗ ISKA'}</b></td>
                    <td style="font-size:11px;color:var(--muted);">${s.top_scores.join(' · ')}</td>
                    <td><b class="${s.top1_hit ? 'good' : s.top3_hit ? 'warn' : 'bad'}">${s.top1_hit ? '🎯 Tam İsabet' : s.top3_hit ? '⚡ İlk 3 İçinde' : '✗ Iska'}</b></td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>

        <!-- Matematiksel Yöntem Kartı -->
        <div class="card" style="background:var(--panel2);padding:16px 20px;border-radius:12px;">
          <h3 style="margin:0 0 8px;font-size:15px;color:var(--accent);">📐 Matematiksel Yöntem &amp; Çifte Şans Türetimi</h3>
          <p style="font-size:12px;color:var(--muted);line-height:1.6;margin:0 0 8px;">
            Her maç için takımların ev/deplasman hücum-savunma parametreleri ve beklenen golleri (λ) üzerinden <b>Dixon-Coles düşük skor düzeltmeli Bivariate Poisson matrisi</b> oluşturulur. $P(x, y)$ matrisinde $x$ ev sahibi, $y$ deplasman gol sayısı olmak üzere:
          </p>
          <div class="formula" style="font-size:12px;padding:8px 12px;margin-bottom:8px;">
            P(1X) = P(Ev Galibiyeti) + P(Beraberlik) = &sum;<sub>x &gt; y</sub> P(x, y) + &sum;<sub>x = y</sub> P(x, y)
          </div>
          <p style="font-size:12px;color:var(--muted);line-height:1.6;margin:0;">
            Modelin <b>≥%75 güven eşiği</b> üstündeki maçları, 5 sezon boyunca sürdürülebilir biçimde %76'nın üzerinde bir isabet yakalamıştır.
          </p>
        </div>

      </div>
    `;

    pane.innerHTML = html;
    wireCifteBacktestEvents(pane);
  }

  function emptyStat() {
    return {
      total: 0,
      dc_best_n: 0, dc_best_h: 0, dc_best_pct: 0,
      dc_1x_n: 0, dc_1x_h: 0, dc_1x_pct: 0,
      dc_12_n: 0, dc_12_h: 0, dc_12_pct: 0,
      dc_x2_n: 0, dc_x2_h: 0, dc_x2_pct: 0,
      sc_top1_h: 0, sc_top1_pct: 0,
      sc_top3_h: 0, sc_top3_pct: 0
    };
  }

  function wireCifteEvents(container) {
    if (!container) return;

    // Alt sekme geçişi (Tahminler vs Model Doğruluğu)
    container.querySelectorAll('.subtab-toggle').forEach(btn => {
      btn.onclick = () => {
        cifteSubView = btn.dataset.view;
        renderCiftePane();
      };
    });

    // Çip filtreleri dinle
    container.querySelectorAll('.cifte-chips .chip').forEach(btn => {
      btn.onclick = () => {
        cifteFilter = btn.dataset.cf;
        renderCiftePane();
      };
    });

    // Arama kutusu dinle
    const qInp = container.querySelector('#qCifte');
    if (qInp) {
      qInp.oninput = (e) => {
        cifteSearch = e.target.value.trim();
        renderCiftePane();
      };
      const clearBtn = container.querySelector('#qCifteClear');
      if (clearBtn) {
        clearBtn.onclick = () => {
          cifteSearch = '';
          renderCiftePane();
        };
      }
    }
  }

  function wireCifteBacktestEvents(container) {
    if (!container) return;

    // Alt sekme geçişi
    container.querySelectorAll('.subtab-toggle').forEach(btn => {
      btn.onclick = () => {
        cifteSubView = btn.dataset.view;
        renderCiftePane();
      };
    });

    // Filtreyi sıfırla butonu
    const resetBtn = container.querySelector('#btnResetCifteLeague');
    if (resetBtn) {
      resetBtn.onclick = () => {
        btLeagueFilter = 'Tümü';
        renderCiftePane();
      };
    }

    // Lig tablosundaki tıklanabilir satırlar
    container.querySelectorAll('.bt-cifte-league-row').forEach(tr => {
      tr.onclick = () => {
        const lg = tr.dataset.league;
        btLeagueFilter = (btLeagueFilter === lg) ? 'Tümü' : lg;
        renderCiftePane();
      };
    });
  }

  global.BETAVUS_CIFTE_UI = {
    renderCiftePane,
    setSubView: (v) => { cifteSubView = v; renderCiftePane(); },
    setFilter: (f) => { cifteFilter = f; renderCiftePane(); },
    setSearch: (s) => { cifteSearch = s; renderCiftePane(); }
  };

})(typeof window !== 'undefined' ? window : this);
