/**
 * BETAVUS — Çifte Şans & Skor Tahminleri Kullanıcı Arayüzü (cifte_ui.js)
 */

(function (global) {
  'use strict';

  let cifteFilter = 'all'; // 'all', 'high_conf', 'dc_1x', 'dc_12', 'dc_x2'
  let cifteSearch = '';

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

  function renderCiftePane() {
    const pane = document.getElementById('pane-cifte');
    if (!pane) return;

    // Fikstür verilerini al
    const rawMatches = window.__data || [];
    const CE = global.BETAVUS_CIFTE;
    if (!CE) {
      pane.innerHTML = '<div class="empty">Analiz motoru yükleniyor…</div>';
      return;
    }

    // Aktif lige göre filtrele
    const currentLeague = (window.selectedByTab && window.selectedByTab['cifte']) || window.selected || 'Tümü';
    let filtered = rawMatches;
    if (currentLeague && currentLeague !== 'Tümü') {
      filtered = filtered.filter(m => m.league === currentLeague);
    }

    // Her maç için analiz yap
    const analyzed = filtered.map(m => CE.analyzeMatch(m)).filter(Boolean);

    // Çifte şans filtresini uygula
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

    // Arama filtresi uygula
    if (cifteSearch) {
      const q = cifteSearch.toLowerCase();
      displayList = displayList.filter(a =>
        a.home.toLowerCase().includes(q) ||
        a.away.toLowerCase().includes(q) ||
        a.league.toLowerCase().includes(q)
      );
    }

    // İstatistik özetleri
    const totalCount = analyzed.length;
    const highConfCount = analyzed.filter(a => a.bestDc.pct >= 75.0).length;
    const avgBestDc = totalCount > 0 ? (analyzed.reduce((s, a) => s + a.bestDc.pct, 0) / totalCount).toFixed(1) : '0.0';

    // HTML Oluştur
    let html = `
      <div class="cifte-container" style="max-width:1100px;margin:0 auto;padding:12px 14px;">
        
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
          
          <!-- Hızlı Filtre Çipleri -->
          <div class="cifte-chips" style="display:flex;gap:8px;flex-wrap:wrap;">
            <button class="chip ${cifteFilter === 'all' ? 'active' : ''}" data-cf="all" type="button" style="cursor:pointer;">🌐 Tümü (${totalCount})</button>
            <button class="chip ${cifteFilter === 'high_conf' ? 'active' : ''}" data-cf="high_conf" type="button" style="cursor:pointer;">⭐ Yüksek Güven (≥%75)</button>
            <button class="chip ${cifteFilter === 'dc_1x' ? 'active' : ''}" data-cf="dc_1x" type="button" style="cursor:pointer;">🛡️ 1X (Ev/Beraberlik)</button>
            <button class="chip ${cifteFilter === 'dc_12' ? 'active' : ''}" data-cf="dc_12" type="button" style="cursor:pointer;">⚡ 12 (Kazanır / Berabere Bitmez)</button>
            <button class="chip ${cifteFilter === 'dc_x2' ? 'active' : ''}" data-cf="dc_x2" type="button" style="cursor:pointer;">🚀 X2 (Beraberlik/Deplasman)</button>
          </div>

          <!-- Arama Kutusu -->
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

    // En olası 4 skor
    const top4Scores = a.topScores.slice(0, 4);

    return `
      <div class="card cifte-match-card" style="background:rgba(18,24,38,0.75);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:16px 18px;position:relative;transition:border-color .2s;">
        
        <!-- Üst Başlık: Lig, Tarih ve Beklenen Gol -->
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

        <!-- Takımlar ve 1-X-2 Olasılık Barı -->
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

          <!-- 1-X-2 Görsel Dağılım Barı -->
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

        <!-- Çifte Şans Kutuları (1X, 12, X2) -->
        <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:10px;margin-bottom:14px;">
          
          <!-- 1X Kutusu -->
          <div class="cifte-box ${bestPick === '1X' ? 'is-best' : ''}" style="background:${bestPick === '1X' ? 'rgba(16,185,129,0.08)' : 'rgba(255,255,255,0.025)'};border:1px solid ${bestPick === '1X' ? 'rgba(16,185,129,0.45)' : 'rgba(255,255,255,0.08)'};border-radius:10px;padding:10px 12px;position:relative;">
            ${bestPick === '1X' ? '<span style="position:absolute;top:-8px;right:10px;background:#10b981;color:#0d1219;font-size:9.5px;font-weight:900;padding:1px 6px;border-radius:4px;">MODEL TERCİHİ</span>' : ''}
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <b style="font-size:14px;color:${bestPick === '1X' ? '#10b981' : 'var(--text)'};">1X</b>
              <span style="font-size:16px;font-weight:900;color:${bestPick === '1X' ? '#10b981' : 'var(--text)'};">${fmtPct(a.dc['1X'].pct)}</span>
            </div>
            <div style="font-size:11px;color:var(--muted);margin-top:2px;">Ev Sahibi veya Beraberlik (1-0)</div>
            ${a.dc['1X'].odds ? `<div style="font-size:10.5px;color:var(--accent);margin-top:4px;">Piyasa Oranı: <b>${a.dc['1X'].odds}</b></div>` : ''}
          </div>

          <!-- 12 Kutusu -->
          <div class="cifte-box ${bestPick === '12' ? 'is-best' : ''}" style="background:${bestPick === '12' ? 'rgba(16,185,129,0.08)' : 'rgba(255,255,255,0.025)'};border:1px solid ${bestPick === '12' ? 'rgba(16,185,129,0.45)' : 'rgba(255,255,255,0.08)'};border-radius:10px;padding:10px 12px;position:relative;">
            ${bestPick === '12' ? '<span style="position:absolute;top:-8px;right:10px;background:#10b981;color:#0d1219;font-size:9.5px;font-weight:900;padding:1px 6px;border-radius:4px;">MODEL TERCİHİ</span>' : ''}
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <b style="font-size:14px;color:${bestPick === '12' ? '#10b981' : 'var(--text)'};">12</b>
              <span style="font-size:16px;font-weight:900;color:${bestPick === '12' ? '#10b981' : 'var(--text)'};">${fmtPct(a.dc['12'].pct)}</span>
            </div>
            <div style="font-size:11px;color:var(--muted);margin-top:2px;">Ev veya Deplasman (1-2)</div>
            ${a.dc['12'].odds ? `<div style="font-size:10.5px;color:var(--accent);margin-top:4px;">Piyasa Oranı: <b>${a.dc['12'].odds}</b></div>` : ''}
          </div>

          <!-- X2 Kutusu -->
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

        <!-- Alt Panel: En Olası Skorlar ve Gol Çizgileri -->
        <div style="background:rgba(0,0,0,0.22);border-radius:8px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
          
          <!-- En Olası Skorlar -->
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <span style="font-size:11px;font-weight:700;color:var(--muted);">🎯 En Olası Skorlar:</span>
            ${top4Scores.map(sc => `
              <span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);padding:2px 8px;border-radius:6px;font-size:11px;">
                <b style="color:#fbbf24;">${sc.score}</b> <span style="color:var(--muted);font-size:10px;">(%${sc.pct})</span>
              </span>
            `).join('')}
          </div>

          <!-- Gol & KG Özeti -->
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

  function wireCifteEvents(container) {
    if (!container) return;

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

  global.BETAVUS_CIFTE_UI = {
    renderCiftePane,
    setFilter: (f) => { cifteFilter = f; renderCiftePane(); },
    setSearch: (s) => { cifteSearch = s; renderCiftePane(); }
  };

})(typeof window !== 'undefined' ? window : this);
