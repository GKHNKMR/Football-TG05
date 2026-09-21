/**
 * BETAVUS — Çifte Şans & Toplam Gol Aralığı Kullanıcı Arayüzü (cifte_ui.js)
 * 
 * Hem güncel maç bülteni tahminlerini hem de 5 sezonluk (16.478 maç)
 * Çifte Şans ve Toplam Gol Aralığı Model Doğruluğu (Backtest) analizini sunar.
 * Kısıtlı veriler (partial-form / league-avg) analize ve vurgulara dahil edilmez.
 */

(function (global) {
  'use strict';

  let cifteSubView = 'bt'; // Bu sekme yalnızca Model Doğruluğu görünümünü sunar.
  let cifteFilter = 'all';   // 'all', 'high_conf', 'dc_1x', 'dc_12', 'dc_x2'
  let cifteSearch = '';
  let btLeagueFilter = 'Tümü';
  let btSampleExpanded = false;

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

  function isLimitedData(x) {
    if (typeof window !== 'undefined' && typeof window.isLimitedData === 'function') {
      return window.isLimitedData(x);
    }
    if (!x) return false;
    if (x.h2h_tier || (x.h2h_matches_used && x.h2h_matches_used >= 2)) return false;
    const b = typeof x === 'string' ? x : x.basis;
    return b === 'partial-form' || b === 'league-avg' || (typeof b === 'string' && (b.startsWith('partial-form') || b.startsWith('league-avg')));
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

    renderCifteBacktestView(pane);
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
      displayList = displayList.filter(a => !isLimitedData(a.rawMatch) && a.bestDc.pct >= 75.0);
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
    const highConfCount = analyzed.filter(a => !isLimitedData(a.rawMatch) && a.bestDc.pct >= 75.0).length;
    const avgBestDc = totalCount > 0 ? (analyzed.reduce((s, a) => s + a.bestDc.pct, 0) / totalCount).toFixed(1) : '0.0';

    let html = `
      <div class="cifte-container" style="max-width:1100px;margin:0 auto;padding:12px 14px;">
        
        <!-- Alt Sekme Geçiş Butonları (Tahminler vs Model Doğruluğu) -->
        <div class="subtabs-bar" style="display:flex;gap:10px;margin-bottom:16px;">
          <button class="chip active subtab-toggle" data-view="pred" type="button" style="padding:7px 16px;font-size:12.5px;font-weight:800;border-radius:8px;cursor:pointer;">
            ⚽ Güncel Fikstür &amp; Tahminler (${totalCount})
          </button>
          <button class="chip subtab-toggle" data-view="bt" type="button" style="padding:7px 16px;font-size:12.5px;font-weight:800;border-radius:8px;cursor:pointer;background:rgba(255,255,255,0.05);color:var(--text);border:1px solid rgba(255,255,255,0.15);">
            📊 Çifte Şans &amp; Gol Aralığı Model Doğruluğu (16.478 Maç)
          </button>
        </div>

        <!-- Tanıtım ve Bilgi Başlığı -->
        <div class="cifte-intro-card" style="background:linear-gradient(135deg,rgba(15,23,42,0.85),rgba(30,41,59,0.7));border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:16px 20px;margin-bottom:18px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
          <div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
              <span style="font-size:20px;">🎲</span>
              <h2 style="margin:0;font-size:18px;font-weight:900;color:var(--text);">Çifte Şans &amp; Toplam Gol Aralığı</h2>
              <span class="cms-badge" style="background:rgba(16,185,129,0.15);color:#10b981;border:1px solid rgba(16,185,129,0.3);font-size:11px;padding:2px 8px;border-radius:99px;font-weight:700;">Dixon-Coles Matrisi</span>
            </div>
            <p style="margin:0;font-size:12.5px;color:var(--muted);max-width:650px;line-height:1.45;">
              Geçmiş takım formları, hücum/savunma parametreleri ve beklenen gollerden (λ) hesaplanan <b>1X, 12, X2</b> çifte şans olasılıkları ile <b>2–3, 3–4 ve 5+ gol</b> toplam gol aralıkları.
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
    const lim = isLimitedData(a.rawMatch);
    const isHighConf = !lim && a.bestDc.pct >= 75.0;
    const bestPick = a.bestDc.pick;
    const goalRanges = Object.values(a.goalRanges || {});
    const bestGoalRange = a.bestGoalRange || goalRanges[0];

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
            ${lim ? `<span class="cms-badge" style="background:rgba(242,153,74,0.18);color:#f2994a;border:1px solid rgba(242,153,74,0.3);font-size:10px;padding:2px 8px;border-radius:6px;font-weight:800;">⚠️ Kısıtlı Veri (Vurgusuz)</span>` : isHighConf ? `<span class="cms-badge" style="background:rgba(16,185,129,0.15);color:#10b981;border:1px solid rgba(16,185,129,0.3);font-size:10.5px;padding:2px 8px;border-radius:6px;font-weight:800;">⭐ Yüksek Güven</span>` : ''}
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
          
          <div class="cifte-box ${bestPick === '1X' && !lim ? 'is-best' : ''}" style="background:${bestPick === '1X' && !lim ? 'rgba(16,185,129,0.08)' : 'rgba(255,255,255,0.025)'};border:1px solid ${bestPick === '1X' && !lim ? 'rgba(16,185,129,0.45)' : 'rgba(255,255,255,0.08)'};border-radius:10px;padding:10px 12px;position:relative;">
            ${bestPick === '1X' && !lim ? '<span style="position:absolute;top:-8px;right:10px;background:#10b981;color:#0d1219;font-size:9.5px;font-weight:900;padding:1px 6px;border-radius:4px;">MODEL TERCİHİ</span>' : ''}
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <b style="font-size:14px;color:${bestPick === '1X' && !lim ? '#10b981' : 'var(--text)'};">1X</b>
              <span style="font-size:16px;font-weight:900;color:${bestPick === '1X' && !lim ? '#10b981' : 'var(--text)'};">${fmtPct(a.dc['1X'].pct)}</span>
            </div>
            <div style="font-size:11px;color:var(--muted);margin-top:2px;">Ev Sahibi veya Beraberlik (1-0)</div>
            ${a.dc['1X'].odds ? `<div style="font-size:10.5px;color:var(--accent);margin-top:4px;">Piyasa Oranı: <b>${a.dc['1X'].odds}</b></div>` : ''}
          </div>

          <div class="cifte-box ${bestPick === '12' && !lim ? 'is-best' : ''}" style="background:${bestPick === '12' && !lim ? 'rgba(16,185,129,0.08)' : 'rgba(255,255,255,0.025)'};border:1px solid ${bestPick === '12' && !lim ? 'rgba(16,185,129,0.45)' : 'rgba(255,255,255,0.08)'};border-radius:10px;padding:10px 12px;position:relative;">
            ${bestPick === '12' && !lim ? '<span style="position:absolute;top:-8px;right:10px;background:#10b981;color:#0d1219;font-size:9.5px;font-weight:900;padding:1px 6px;border-radius:4px;">MODEL TERCİHİ</span>' : ''}
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <b style="font-size:14px;color:${bestPick === '12' && !lim ? '#10b981' : 'var(--text)'};">12</b>
              <span style="font-size:16px;font-weight:900;color:${bestPick === '12' && !lim ? '#10b981' : 'var(--text)'};">${fmtPct(a.dc['12'].pct)}</span>
            </div>
            <div style="font-size:11px;color:var(--muted);margin-top:2px;">Ev veya Deplasman (1-2)</div>
            ${a.dc['12'].odds ? `<div style="font-size:10.5px;color:var(--accent);margin-top:4px;">Piyasa Oranı: <b>${a.dc['12'].odds}</b></div>` : ''}
          </div>

          <div class="cifte-box ${bestPick === 'X2' && !lim ? 'is-best' : ''}" style="background:${bestPick === 'X2' && !lim ? 'rgba(16,185,129,0.08)' : 'rgba(255,255,255,0.025)'};border:1px solid ${bestPick === 'X2' && !lim ? 'rgba(16,185,129,0.45)' : 'rgba(255,255,255,0.08)'};border-radius:10px;padding:10px 12px;position:relative;">
            ${bestPick === 'X2' && !lim ? '<span style="position:absolute;top:-8px;right:10px;background:#10b981;color:#0d1219;font-size:9.5px;font-weight:900;padding:1px 6px;border-radius:4px;">MODEL TERCİHİ</span>' : ''}
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <b style="font-size:14px;color:${bestPick === 'X2' && !lim ? '#10b981' : 'var(--text)'};">X2</b>
              <span style="font-size:16px;font-weight:900;color:${bestPick === 'X2' && !lim ? '#10b981' : 'var(--text)'};">${fmtPct(a.dc['X2'].pct)}</span>
            </div>
            <div style="font-size:11px;color:var(--muted);margin-top:2px;">Beraberlik veya Deplasman (0-2)</div>
            ${a.dc['X2'].odds ? `<div style="font-size:10.5px;color:var(--accent);margin-top:4px;">Piyasa Oranı: <b>${a.dc['X2'].odds}</b></div>` : ''}
          </div>

        </div>

        <div style="background:rgba(0,0,0,0.22);border-radius:8px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
          <div class="goal-range-list" style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <span style="font-size:11px;font-weight:700;color:var(--muted);">⚽ Toplam Gol Aralığı:</span>
            ${goalRanges.map(range => {
              const selected = !lim && bestGoalRange && range.pick === bestGoalRange.pick;
              return `
              <span class="goal-range-chip${selected ? ' is-best' : ''}" data-goal-range="${range.pick}" style="background:${selected ? 'rgba(251,191,36,0.12)' : 'rgba(255,255,255,0.06)'};border:1px solid ${selected ? 'rgba(251,191,36,0.55)' : 'rgba(255,255,255,0.1)'};padding:4px 9px;border-radius:7px;font-size:11px;">
                <b style="color:${selected ? '#fbbf24' : 'var(--text)'};">${range.label}</b>
                <span style="color:var(--muted);font-size:10px;">(%${range.pct})</span>
                ${selected ? '<span style="display:block;color:#fbbf24;font-size:8.5px;font-weight:900;margin-top:2px;">MODEL ARALIĞI</span>' : ''}
              </span>`;
            }).join('')}
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
  // 2. ÇİFTE ŞANS & GOL ARALIĞI MODEL DOĞRULUĞU (BACKTEST) GÖRÜNÜMÜ
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

    const seasonList = ['2025/26', '2024/25', '2023/24', '2022/23', '2021/22'];

    // 4 KPI Kart Tanımı (0.5, 1.5, 2.5 Model Doğruluğu ile Birebir Uyumlu!)
    const kpiCards = [
      {
        key: '1X',
        label: '1X Çifte Şans',
        desc: 'Ev Sahibi veya Beraberlik (1-0)',
        emoji: '🛡️',
        color: '#10b981',
        threshold: '≥ %75',
        n: st.dc_1x_n,
        h: st.dc_1x_h,
        pct: st.dc_1x_pct
      },
      {
        key: '12',
        label: '12 Çifte Şans',
        desc: 'Ev veya Deplasman (1-2)',
        emoji: '⚡',
        color: '#3b82f6',
        threshold: '≥ %75',
        n: st.dc_12_n,
        h: st.dc_12_h,
        pct: st.dc_12_pct
      },
      {
        key: 'X2',
        label: 'X2 Çifte Şans',
        desc: 'Beraberlik veya Deplasman (0-2)',
        emoji: '🚀',
        color: '#ef4444',
        threshold: '≥ %75',
        n: st.dc_x2_n,
        h: st.dc_x2_h,
        pct: st.dc_x2_pct
      },
      {
        key: 'range',
        label: 'Toplam Gol Aralığı',
        desc: '2–3 · 3–4 · 5+ Gol',
        emoji: '🎯',
        color: '#fbbf24',
        threshold: 'En Olası Aralık',
        n: st.gr_n,
        h: st.gr_h,
        pct: st.gr_pct
      }
    ];

    const renderMetricGroup = (pool, hits, pct, color) => `
      <td class="cifte-bt-stat-pool">${num(pool)}</td>
      <td class="cifte-bt-stat-hit">${num(hits)}</td>
      <td class="cifte-bt-stat-rate" style="--stat-color:${color};">%${pct}</td>
    `;

    const filteredSamples = (data.samples || []).filter(x => !isFiltered || x.league === btLeagueFilter);
    const visibleSamples = filteredSamples.slice(0, btSampleExpanded ? 50 : 12);

    let html = `
      <div class="cifte-container cifte-backtest">
        
        <!-- Üst Başlık ve Bilgilendirme Kartı -->
        <div class="card cifte-bt-hero">
          <div class="cifte-bt-hero-row">
            <div>
              <h2>📊 Çifte Şans &amp; Gol Aralığı Model Doğruluğu</h2>
              <p>
                Beş tamamlanmış sezondaki <b>16.478 maç</b> ile sızıntısız walk-forward doğrulama. Her tahmin yalnızca maçtan önce bilinen verilerle üretildi; kısıtlı veriler güvenli vurgu hesaplarının dışında tutuldu.
              </p>
            </div>
            ${isFiltered ? `
              <button class="chip active" id="btnResetCifteLeague" type="button">
                ${esc(btLeagueFilter)} filtresini kaldır ✕
              </button>
            ` : ''}
          </div>
        </div>

        <!-- Kısa okuma rehberi -->
        <div class="cifte-bt-guide" aria-label="Doğrulama ekranı okuma rehberi">
          <div class="cifte-bt-guide-item">
            <b>Vurgu ne demek?</b>
            Model olasılığı en az %75 olan ve kısıtlı veri taşımayan maçlar.
          </div>
          <div class="cifte-bt-guide-item">
            <b>Başarı oranı nasıl okunur?</b>
            Tutan tahmin sayısının vurgulanan maç sayısına oranı.
          </div>
          <div class="cifte-bt-guide-item">
            <b>Gol aralığı nasıl doğrulanır?</b>
            Modelin seçtiği 2–3, 3–4 veya 5+ gol bandı gerçekleşen toplam golle karşılaştırılır.
          </div>
        </div>

        <!-- 4 KPI Kartı (Model Doğruluğu Tasarımı ile Birebir — Iska Kutusu Yok!) -->
        <div class="cifte-bt-kpi-grid">
          ${kpiCards.map(c => {
            const metrics = c.key === 'range'
              ? [
                  { value: st.total, label: '5 Sezonluk Maç', color: 'var(--text)' },
                  { value: st.gr_n, label: 'Aralık Tahmini', color: '#fbbf24' },
                  { value: st.gr_h, label: 'Aralık Tuttu ✓', color: 'var(--good)' }
                ]
              : [
                  { value: st.total, label: '5 Sezonluk Maç', color: 'var(--text)' },
                  { value: c.n, label: 'Sistem Vurguladı', color: c.color },
                  { value: c.h, label: 'Tahmin Tuttu ✓', color: 'var(--good)' }
                ];
            return `
              <article class="card cifte-bt-kpi" style="--market:${c.color};">
                <div class="cifte-bt-kpi-head">
                  <span class="cifte-bt-kpi-icon">${c.emoji}</span>
                  <span class="cifte-bt-kpi-title">${c.label}</span>
                  ${isFiltered ? `<span class="cifte-bt-kpi-scope">· ${esc(btLeagueFilter)}</span>` : ''}
                  <span class="cifte-bt-kpi-threshold">${c.threshold}</span>
                </div>
                <div class="cifte-bt-metrics">
                  ${metrics.map(metric => `
                    <div class="cifte-bt-metric" style="--metric-color:${metric.color};">
                      <div class="cifte-bt-metric-value">${num(metric.value)}</div>
                      <div class="cifte-bt-metric-label">${metric.label}</div>
                    </div>
                  `).join('')}
                </div>
                <div class="cifte-bt-success">
                  <span class="cifte-bt-success-value">%${c.pct}</span>
                  <span class="cifte-bt-success-label">Başarı Oranı</span>
                  <div class="cifte-bt-success-note">
                    ${c.key === 'range'
                      ? `${num(c.n)} tahminin <b>${num(c.h)}</b> tanesinde toplam gol seçilen aralıkta · 2–3: <b>%${st.gr_23_pct}</b> · 3–4: <b>%${st.gr_34_pct}</b> · 5+: <b>%${st.gr_5p_pct}</b>`
                      : `${num(c.n)} vurgulanan maçın <b>${num(c.h)}</b> tanesinde tahmin tuttu`}
                  </div>
                </div>
              </article>
            `;
          }).join('')}
        </div>

        <!-- 1. Lig bazında başarı tablosu -->
        <section class="card cifte-bt-section">
          <h2>Lig Bazında Çifte Şans &amp; Gol Aralığı Başarısı ${isFiltered ? `· ${esc(btLeagueFilter)}` : '· Tüm Ligler'}</h2>
          <p>Bir lige tıklayarak üstteki dört performans kartını ve sezon tablosunu aynı lig için süzebilirsiniz.</p>
          <div class="cifte-bt-table-wrap">
            <table class="cifte-bt-table" id="tblCifteLeague">
              <thead>
                <tr class="cifte-bt-group-row">
                  <th scope="col" rowspan="2">Lig / Sezon</th>
                  <th scope="col" rowspan="2">Maç</th>
                  <th scope="colgroup" colspan="3" class="cifte-bt-group-head" style="--group-color:#10b981;">1X Çifte Şans (≥%75)</th>
                  <th scope="colgroup" colspan="3" class="cifte-bt-group-head" style="--group-color:#3b82f6;">12 Çifte Şans (≥%75)</th>
                  <th scope="colgroup" colspan="3" class="cifte-bt-group-head" style="--group-color:#ef4444;">X2 Çifte Şans (≥%75)</th>
                  <th scope="colgroup" colspan="3" class="cifte-bt-group-head" style="--group-color:#fbbf24;">Toplam Gol Aralığı</th>
                </tr>
                <tr class="cifte-bt-subhead-row">
                  <th scope="col">Vurgu</th><th scope="col">Tuttu</th><th scope="col">%</th>
                  <th scope="col">Vurgu</th><th scope="col">Tuttu</th><th scope="col">%</th>
                  <th scope="col">Vurgu</th><th scope="col">Tuttu</th><th scope="col">%</th>
                  <th scope="col">Tahmin</th><th scope="col">Tuttu</th><th scope="col">%</th>
                </tr>
              </thead>
              <tbody>
                <tr class="clickable bt-cifte-league-row ${btLeagueFilter === 'Tümü' ? 'active-row' : ''}" data-league="Tümü">
                  <td style="font-weight:900;">${btLeagueFilter === 'Tümü' ? '👉 ' : ''}🌐 Tüm Ligler</td>
                  <td style="font-weight:900;">${num(data.overall.total)}</td>
                  ${renderMetricGroup(data.overall.dc_1x_n, data.overall.dc_1x_h, data.overall.dc_1x_pct, '#10b981')}
                  ${renderMetricGroup(data.overall.dc_12_n, data.overall.dc_12_h, data.overall.dc_12_pct, '#3b82f6')}
                  ${renderMetricGroup(data.overall.dc_x2_n, data.overall.dc_x2_h, data.overall.dc_x2_pct, '#ef4444')}
                  ${renderMetricGroup(data.overall.gr_n, data.overall.gr_h, data.overall.gr_pct, '#fbbf24')}
                </tr>
                ${leaguesList.map(lg => {
                  const lSt = data.by_league[lg] || emptyStat();
                  const isAct = (btLeagueFilter === lg);
                  return `
                    <tr class="clickable bt-cifte-league-row ${isAct ? 'active-row' : ''}" data-league="${esc(lg)}">
                      <td style="font-weight:${isAct ? '900' : '700'};">${isAct ? '👉 ' : ''}${getFlag(lg)} ${esc(lg)}</td>
                      <td>${num(lSt.total)}</td>
                      ${renderMetricGroup(lSt.dc_1x_n, lSt.dc_1x_h, lSt.dc_1x_pct, '#10b981')}
                      ${renderMetricGroup(lSt.dc_12_n, lSt.dc_12_h, lSt.dc_12_pct, '#3b82f6')}
                      ${renderMetricGroup(lSt.dc_x2_n, lSt.dc_x2_h, lSt.dc_x2_pct, '#ef4444')}
                      ${renderMetricGroup(lSt.gr_n, lSt.gr_h, lSt.gr_pct, '#fbbf24')}
                    </tr>
                  `;
                }).join('')}
              </tbody>
            </table>
          </div>
        </section>

        <!-- 2. Sezon bazında tablo (5 tamamlanmış sezon) -->
        <section class="card cifte-bt-section">
          <h2>Sezon Bazında Çifte Şans &amp; Gol Aralığı Başarısı ${isFiltered ? `· ${esc(btLeagueFilter)}` : '· Tüm Ligler'}</h2>
          <p>Sezonlar yalnızca tamamlanmış 2021/22–2025/26 dönemini kapsar; devam eden sezon bu doğrulamaya dahil değildir.</p>
          <div class="cifte-bt-table-wrap">
            <table class="cifte-bt-table" id="tblCifteSeason">
              <thead>
                <tr class="cifte-bt-group-row">
                  <th scope="col" rowspan="2">Lig / Sezon</th>
                  <th scope="col" rowspan="2">Maç</th>
                  <th scope="colgroup" colspan="3" class="cifte-bt-group-head" style="--group-color:#10b981;">1X Çifte Şans (≥%75)</th>
                  <th scope="colgroup" colspan="3" class="cifte-bt-group-head" style="--group-color:#3b82f6;">12 Çifte Şans (≥%75)</th>
                  <th scope="colgroup" colspan="3" class="cifte-bt-group-head" style="--group-color:#ef4444;">X2 Çifte Şans (≥%75)</th>
                  <th scope="colgroup" colspan="3" class="cifte-bt-group-head" style="--group-color:#fbbf24;">Toplam Gol Aralığı</th>
                </tr>
                <tr class="cifte-bt-subhead-row">
                  <th scope="col">Vurgu</th><th scope="col">Tuttu</th><th scope="col">%</th>
                  <th scope="col">Vurgu</th><th scope="col">Tuttu</th><th scope="col">%</th>
                  <th scope="col">Vurgu</th><th scope="col">Tuttu</th><th scope="col">%</th>
                  <th scope="col">Tahmin</th><th scope="col">Tuttu</th><th scope="col">%</th>
                </tr>
              </thead>
              <tbody>
                ${seasonList.map(sz => {
                  const sSt = isFiltered
                    ? ((data.by_season_league[sz] && data.by_season_league[sz][btLeagueFilter]) || emptyStat())
                    : (data.by_season[sz] || emptyStat());
                  return `
                    <tr>
                      <td style="font-weight:800;">${sz}</td>
                      <td>${num(sSt.total)}</td>
                      ${renderMetricGroup(sSt.dc_1x_n, sSt.dc_1x_h, sSt.dc_1x_pct, '#10b981')}
                      ${renderMetricGroup(sSt.dc_12_n, sSt.dc_12_h, sSt.dc_12_pct, '#3b82f6')}
                      ${renderMetricGroup(sSt.dc_x2_n, sSt.dc_x2_h, sSt.dc_x2_pct, '#ef4444')}
                      ${renderMetricGroup(sSt.gr_n, sSt.gr_h, sSt.gr_pct, '#fbbf24')}
                    </tr>
                  `;
                }).join('')}
              </tbody>
            </table>
          </div>
        </section>

        <!-- 3. Örnek Maçlar — Tahmin edilen gol aralığı vs gerçek toplam gol -->
        <section class="card cifte-bt-section">
          <h2>Örnek Maçlar — Gol Aralığı ile Gerçek Sonuç Karşılaştırması</h2>
          <p>Her satırda modelin maçtan önce seçtiği toplam gol aralığı, maçın gerçekleşen toplam golüyle karşılaştırılır.</p>
          <div class="cifte-bt-table-wrap">
            <table class="cifte-bt-table cifte-bt-samples" id="tblCifteSamples">
              <thead>
                <tr>
                  <th scope="col">Tarih · Lig</th>
                  <th scope="col">Maç</th>
                  <th scope="col">Gerçek Skor · Gol</th>
                  <th scope="col">Çifte Şans Tahmini</th>
                  <th scope="col">Tahmin Edilen Gol Aralığı</th>
                  <th scope="col">Aralık Sonucu</th>
                </tr>
              </thead>
              <tbody>
                ${visibleSamples.map(s => `
                  <tr>
                    <td>
                      <div style="font-weight:800;">${dmy(s.date)}</div>
                      <div class="sample-meta">${getFlag(s.league)} ${esc(s.league)}</div>
                    </td>
                    <td class="sample-match">${esc(s.home)} — ${esc(s.away)}</td>
                    <td><b style="font-size:16px;">${s.actual_score}</b><div class="sample-meta">${s.actual_total} gol</div></td>
                    <td>
                      <span class="cifte-bt-pick">${s.best_dc} · %${s.best_dc_pct}</span>
                      ${s.dc_hit
                        ? '<span class="cifte-bt-result">✓ Tahmin tuttu</span>'
                        : '<span class="cifte-bt-result cifte-bt-muted-result">—</span>'}
                    </td>
                    <td>
                      <span class="cifte-bt-pick" style="border-color:rgba(251,191,36,.45);color:#fbbf24;">${s.goal_range_label} · %${s.goal_range_pct}</span>
                    </td>
                    <td>${s.range_hit
                      ? '<span class="cifte-bt-result" style="color:#fbbf24;">✓ Aralık tuttu</span>'
                      : '<span class="cifte-bt-result cifte-bt-muted-result">—</span>'}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
          ${filteredSamples.length > 12 ? `
            <div class="cifte-bt-sample-actions">
              <span>${btSampleExpanded
                ? `${num(visibleSamples.length)} örnek maç gösteriliyor.`
                : `${num(filteredSamples.length)} maçlık örnek havuzundan ilk 12 karşılaşma gösteriliyor.`}</span>
              <button class="chip" id="btnToggleCifteSamples" type="button">
                ${btSampleExpanded ? 'İlk 12 maça dön ↑' : `Daha fazla örnek göster (${num(Math.min(filteredSamples.length, 50))}) ↓`}
              </button>
            </div>
          ` : ''}
        </section>

        <!-- Matematiksel Yöntem Kartı -->
        <div class="card cifte-bt-method">
          <h3>📐 Hesaplama Yöntemi</h3>
          <p>
            Takımların ev/deplasman hücum ve savunma gücü ile beklenen golleri (λ) kullanılarak <b>Dixon-Coles düzeltmeli skor olasılık matrisi</b> oluşturulur. Bu matristen ev galibiyeti (1), beraberlik (X) ve deplasman galibiyeti (2) olasılıkları türetilir.
          </p>
          <div class="formula">
            P(1X) = P(1) + P(X) &nbsp;·&nbsp; P(12) = P(1) + P(2) &nbsp;·&nbsp; P(X2) = P(X) + P(2)
          </div>
          <p>
            Çifte Şans kartları yalnızca <b>≥%75 güven eşiğini</b> geçen maçları değerlendirir. Gol aralığı için P(2–3), P(3–4) ve P(5+) değerleri skor matrisindeki ilgili toplamların olasılıkları toplanarak hesaplanır; en yüksek olasılıklı aralık model tahmini olur.
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
      gr_n: 0, gr_h: 0, gr_pct: 0,
      gr_23_n: 0, gr_23_h: 0, gr_23_pct: 0,
      gr_34_n: 0, gr_34_h: 0, gr_34_pct: 0,
      gr_5p_n: 0, gr_5p_h: 0, gr_5p_pct: 0
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

    // Filtreyi sıfırla butonu
    const resetBtn = container.querySelector('#btnResetCifteLeague');
    if (resetBtn) {
      resetBtn.onclick = () => {
        btLeagueFilter = 'Tümü';
        btSampleExpanded = false;
        renderCiftePane();
      };
    }

    const sampleToggle = container.querySelector('#btnToggleCifteSamples');
    if (sampleToggle) {
      sampleToggle.onclick = () => {
        btSampleExpanded = !btSampleExpanded;
        renderCiftePane();
      };
    }

    // Lig tablosundaki tıklanabilir satırlar
    container.querySelectorAll('.bt-cifte-league-row').forEach(tr => {
      tr.onclick = () => {
        const lg = tr.dataset.league;
        btLeagueFilter = (btLeagueFilter === lg) ? 'Tümü' : lg;
        btSampleExpanded = false;
        renderCiftePane();
      };
    });
  }

  global.BETAVUS_CIFTE_UI = {
    renderCiftePane,
    setSubView: () => { cifteSubView = 'bt'; renderCiftePane(); },
    setFilter: (f) => { cifteFilter = f; renderCiftePane(); },
    setSearch: (s) => { cifteSearch = s; renderCiftePane(); }
  };

})(typeof window !== 'undefined' ? window : this);
