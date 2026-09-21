/**
 * BETAVUS — Çifte Şans & Skor Tahminleri Analiz Motoru (cifte_engine.js)
 * 
 * Bivariate Poisson ve Dixon-Coles düzeltmeli skor olasılıkları matrisi
 * üzerinden 1-X-2 maç sonu, Çifte Şans (1X, 12, X2), kesin skorlar (1-0, 1-1, 2-1 vb.)
 * ve Karşılıklı Gol (KG Var / Yok) tahminlerini üretir.
 */

(function (global) {
  'use strict';

  // Faktöriyel tablosu (hız için önbelleklenmiş)
  const FACT_CACHE = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600];
  function factorial(n) {
    if (n < 0) return 1;
    if (n < FACT_CACHE.length) return FACT_CACHE[n];
    let r = FACT_CACHE[FACT_CACHE.length - 1];
    for (let i = FACT_CACHE.length; i <= n; i++) r *= i;
    return r;
  }

  // Poisson olasılık kütle fonksiyonu: P(X = k) = (lambda^k * e^-lambda) / k!
  function poissonPmf(k, lambda) {
    if (lambda <= 0) return k === 0 ? 1.0 : 0.0;
    if (k < 0) return 0.0;
    return (Math.pow(lambda, k) * Math.exp(-lambda)) / factorial(k);
  }

  // Dixon-Coles tau düzeltme faktörü (düşük skorlu 0-0, 1-0, 0-1, 1-1 hücreleri için)
  function tau(x, y, lamH, lamA, rho) {
    if (x === 0 && y === 0) return 1.0 - lamH * lamA * rho;
    if (x === 0 && y === 1) return 1.0 + lamH * rho;
    if (x === 1 && y === 0) return 1.0 + lamA * rho;
    if (x === 1 && y === 1) return 1.0 - rho;
    return 1.0;
  }

  function safeRho(lamH, lamA, rho) {
    lamH = Math.max(lamH, 1e-6);
    lamA = Math.max(lamA, 1e-6);
    const hi = Math.min(1.0 / (lamH * lamA), 1.0) - 1e-6;
    const lo = Math.max(-1.0 / lamH, -1.0 / lamA) + 1e-6;
    return Math.min(Math.max(rho, lo), hi);
  }

  // 10x10 Skor Olasılık Matrisi
  const MAX_G = 9;
  function buildScoreGrid(lamH, lamA, rho) {
    rho = safeRho(lamH, lamA, rho || 0.02);
    const px = [];
    const py = [];
    for (let i = 0; i <= MAX_G; i++) {
      px.push(poissonPmf(i, lamH));
      py.push(poissonPmf(i, lamA));
    }

    const grid = {};
    let total = 0.0;
    for (let x = 0; x <= MAX_G; x++) {
      for (let y = 0; y <= MAX_G; y++) {
        const t = Math.max(0, tau(x, y, lamH, lamA, rho));
        const p = px[x] * py[y] * t;
        grid[`${x}-${y}`] = p;
        total += p;
      }
    }

    // Normalizasyon (toplam = 1.0)
    if (total > 0) {
      for (const k in grid) {
        grid[k] /= total;
      }
    }
    return grid;
  }

  // Maç nesnesinden ev ve deplasman beklenen gollerini (lambda) çözümler
  function resolveLambdas(match) {
    let lh = match.lam_home || match.base_lam_home;
    let la = match.lam_away || match.base_lam_away;
    let rho = match.rho || match.base_rho || 0.02;

    if (lh != null && la != null) {
      return { lamH: Number(lh), lamA: Number(la), rho: Number(rho) };
    }

    // Eğer doğrudan lam_home/lam_away yoksa fakat pred_lambda ve market oranları varsa:
    const totLam = Number(match.pred_lambda) || Number(match.exp_goals) || 2.65;
    const mkt = match.market || {};
    const hOdds = Number(mkt.h);
    const aOdds = Number(mkt.a);

    if (hOdds > 1.0 && aOdds > 1.0) {
      // Piyasa olasılıklarının oranına göre beklenen golü paylaştır
      const invH = 1.0 / hOdds;
      const invA = 1.0 / aOdds;
      const homeShare = invH / (invH + invA);
      // Ev sahibi avantajı ile yumuşatılmış lambda dağılımı
      lh = totLam * Math.max(0.25, Math.min(0.75, homeShare));
      la = totLam - lh;
    } else {
      // Standart ev sahibi hafif avantajı (%55 / %45)
      lh = totLam * 0.55;
      la = totLam * 0.45;
    }

    return { lamH: Math.max(0.3, lh), lamA: Math.max(0.3, la), rho: 0.02 };
  }

  /**
   * Tek bir maç için tüm gelişmiş olasılıkları ve çifte şans analizini hesaplar
   */
  function analyzeMatch(match) {
    if (!match) return null;
    const { lamH, lamA, rho } = resolveLambdas(match);
    const grid = buildScoreGrid(lamH, lamA, rho);

    let pHome = 0.0;
    let pDraw = 0.0;
    let pAway = 0.0;
    let pBtts = 0.0;
    let pO05 = 0.0;
    let pO15 = 0.0;
    let pO25 = 0.0;
    let pO35 = 0.0;

    const allScores = [];

    for (let x = 0; x <= MAX_G; x++) {
      for (let y = 0; y <= MAX_G; y++) {
        const p = grid[`${x}-${y}`] || 0;
        if (x > y) pHome += p;
        else if (x === y) pDraw += p;
        else pAway += p;

        if (x >= 1 && y >= 1) pBtts += p;

        const tot = x + y;
        if (tot > 0) pO05 += p;
        if (tot > 1) pO15 += p;
        if (tot > 2) pO25 += p;
        if (tot > 3) pO35 += p;

        allScores.push({ score: `${x}-${y}`, home: x, away: y, prob: p, pct: (p * 100).toFixed(1) });
      }
    }

    // Skorları olasılığa göre çoktan aza sırala
    allScores.sort((a, b) => b.prob - a.prob);
    const topScores = allScores.slice(0, 6);

    // Çifte Şans Olasılıkları
    const p1X = pHome + pDraw;
    const p12 = pHome + pAway;
    const pX2 = pDraw + pAway;

    // Piyasa Çifte Şans İma Edilen Oranları (Marketten türetilen)
    const mkt = match.market || {};
    let mktOdds1X = null, mktOdds12 = null, mktOddsX2 = null;
    if (mkt.h && mkt.d && mkt.a) {
      const invH = 1.0 / Number(mkt.h);
      const invD = 1.0 / Number(mkt.d);
      const invA = 1.0 / Number(mkt.a);
      mktOdds1X = (1.0 / (invH + invD)).toFixed(2);
      mktOdds12 = (1.0 / (invH + invA)).toFixed(2);
      mktOddsX2 = (1.0 / (invD + invA)).toFixed(2);
    }

    // En güçlü ve güvenli Çifte Şans seçimini belirle
    let bestDc = '1X';
    let bestDcPct = p1X;
    if (p12 > bestDcPct && p12 >= 0.70) {
      bestDc = '12';
      bestDcPct = p12;
    }
    if (pX2 > bestDcPct) {
      bestDc = 'X2';
      bestDcPct = pX2;
    }

    // Güven Eşiği (Confidence level): %75 üstü çok yüksek, %65-75 orta, altı normal
    const confidenceLevel = bestDcPct >= 0.80 ? 'very_high' : (bestDcPct >= 0.72 ? 'high' : 'normal');

    return {
      matchId: match.match_id || `${match.league}-${match.home}-${match.away}`,
      home: match.home,
      away: match.away,
      league: match.league,
      kickoff: match.kickoff_utc,
      lamHome: Number(lamH.toFixed(2)),
      lamAway: Number(lamA.toFixed(2)),
      expGoals: Number((lamH + lamA).toFixed(2)),

      // 1-X-2 Olasılıkları
      prob1: Number((pHome * 100).toFixed(1)),
      probX: Number((pDraw * 100).toFixed(1)),
      prob2: Number((pAway * 100).toFixed(1)),

      // Çifte Şans Olasılıkları
      dc: {
        '1X': { pct: Number((p1X * 100).toFixed(1)), prob: p1X, odds: mktOdds1X, label: '1X (1 veya 0)', desc: 'Ev Sahibi veya Beraberlik' },
        '12': { pct: Number((p12 * 100).toFixed(1)), prob: p12, odds: mktOdds12, label: '12 (1 veya 2)', desc: 'Ev Sahibi veya Deplasman' },
        'X2': { pct: Number((pX2 * 100).toFixed(1)), prob: pX2, odds: mktOddsX2, label: 'X2 (0 veya 2)', desc: 'Beraberlik veya Deplasman' },
      },
      bestDc: {
        pick: bestDc,
        pct: Number((bestDcPct * 100).toFixed(1)),
        confidence: confidenceLevel,
        odds: (bestDc === '1X' ? mktOdds1X : (bestDc === '12' ? mktOdds12 : mktOddsX2))
      },

      // Skor Tahminleri (Top 6)
      topScores: topScores,
      // Kullanıcının özellikle belirttiği yaygın skorların anlık olasılıkları
      scoreProbs: {
        '1-0': Number(((grid['1-0'] || 0) * 100).toFixed(1)),
        '0-0': Number(((grid['0-0'] || 0) * 100).toFixed(1)),
        '1-1': Number(((grid['1-1'] || 0) * 100).toFixed(1)),
        '0-1': Number(((grid['0-1'] || 0) * 100).toFixed(1)),
        '2-1': Number(((grid['2-1'] || 0) * 100).toFixed(1)),
        '1-2': Number(((grid['1-2'] || 0) * 100).toFixed(1)),
        '2-0': Number(((grid['2-0'] || 0) * 100).toFixed(1)),
        '0-2': Number(((grid['0-2'] || 0) * 100).toFixed(1)),
        '2-2': Number(((grid['2-2'] || 0) * 100).toFixed(1))
      },

      // Gol Çizgileri
      goals: {
        o05: Number((pO05 * 100).toFixed(1)),
        o15: Number((pO15 * 100).toFixed(1)),
        o25: Number((pO25 * 100).toFixed(1)),
        o35: Number((pO35 * 100).toFixed(1)),
        bttsYes: Number((pBtts * 100).toFixed(1)),
        bttsNo: Number(((1.0 - pBtts) * 100).toFixed(1))
      },

      rawMatch: match
    };
  }

  // Dışa aktarım
  global.BETAVUS_CIFTE = {
    poissonPmf,
    buildScoreGrid,
    resolveLambdas,
    analyzeMatch
  };

})(typeof window !== 'undefined' ? window : this);
