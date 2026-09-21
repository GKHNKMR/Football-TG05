/**
 * BETAVUS — Paper-Betting, Kupon Planlama ve Sanal Kasa Yönetim Motoru
 * (Pure Calculation Engine — DOM'dan tamamen bağımsız, test edilebilir saf fonksiyonlar)
 */

(function (root, factory) {
  if (typeof define === 'function' && define.amd) {
    define([], factory);
  } else if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.BETAVUS_PAPER = factory();
  }
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---------------------------------------------------------------------------
  // 1. Merkezi Konfigürasyonlar (Şartname Bölüm 6, 12, 13)
  // ---------------------------------------------------------------------------

  const SCHEMA_VERSION = 1;
  const STORAGE_KEY = 'betavus.paper_v1';

  const RISK_PROFILES = {
    minimum: {
      id: 'minimum',
      name: 'Minimum Risk',
      kol: 'Minimum Risk',
      badgeClass: 'b-min',
      color: '#10b981',
      reservePct: 0.50,          // Kasa Rezerv: 50%
      dailyGrowthRate: 0.15,     // Günlük Büyüme Oranı: 15%
      dailyFactor: 1.15,         // 1.15x
      stakePct: 0.50,            // Aktif oynanabilir kasa payı: %50
      targetOdds: 1.30,
      legsCount: 5,
      marketTarget: 'over_0_5',
      note: '5 adet 0,5 ustu mac',
      marketLabel: '0.5 Üst',
      minRiskArmPct: 0.50,
      midRiskArmPct: 0.00,
      highRiskArmPct: 0.00,
      desc: 'Kasa Rezervi: %50 · Günlük Büyüme Oranı: %15 (1.15x/gün).'
    },
    medium: {
      id: 'medium',
      name: 'Orta Risk',
      kol: 'Orta Risk',
      badgeClass: 'b-med',
      color: '#3b82f6',
      reservePct: 0.35,          // Kasa Rezerv: 35%
      dailyGrowthRate: 0.20,     // Günlük Büyüme Oranı: 20%
      dailyFactor: 1.20,         // 1.20x
      stakePct: 0.65,            // Aktif oynanabilir kasa payı: %65
      targetOdds: 1.50,
      legsCount: 3,
      marketTarget: 'over_1_5',
      note: '3 adet 1,5 ustu mac',
      marketLabel: '1.5 Üst',
      minRiskArmPct: 0.00,
      midRiskArmPct: 0.65,
      highRiskArmPct: 0.00,
      desc: 'Kasa Rezervi: %35 · Günlük Büyüme Oranı: %20 (1.20x/gün).'
    },
    high: {
      id: 'high',
      name: 'Yüksek Risk',
      kol: 'Yuksek Risk',
      badgeClass: 'b-high',
      color: '#ef4444',
      reservePct: 0.25,          // Kasa Rezerv: 25%
      dailyGrowthRate: 0.25,     // Günlük Büyüme Oranı: 25%
      dailyFactor: 1.25,         // 1.25x
      stakePct: 0.75,            // Aktif oynanabilir kasa payı: %75
      targetOdds: 2.00,
      legsCount: 3,
      marketTarget: 'over_2_5',
      note: '3 adet 2,5 ustu mac',
      marketLabel: '2.5 Üst',
      minRiskArmPct: 0.00,
      midRiskArmPct: 0.00,
      highRiskArmPct: 0.75,
      desc: 'Kasa Rezervi: %25 · Günlük Büyüme Oranı: %25 (1.25x/gün).'
    }
  };

  // Geriye dönük uyumluluk takma adları (Aliases)
  RISK_PROFILES.cautious = RISK_PROFILES.minimum;
  RISK_PROFILES.balanced = RISK_PROFILES.medium;
  RISK_PROFILES.aggressive = RISK_PROFILES.high;
  RISK_PROFILES.multi = RISK_PROFILES.minimum;

  const COUPON_CLASSES = {
    minimum: {
      id: 'minimum',
      name: 'Minimum Risk',
      badgeClass: 'b-min',
      color: '#10b981',
      armKey: 'minRiskArmPct',
      stakePct: 0.50,
      targetOdds: 1.25,
      note: '5 adet 0,5 ustu mac',
      market: 'over_0_5',
      line: '0.5',
      marketLabel: '0.5 Üst',
      propKey: 'p_over_0_5',
      minModelProb: 0.94,
      maxLegs: 5,
      fallbackOdds: 1.25,
      desc: '5 adet 0,5 üstü maç (≥ %94 model güveni, %50 kasa rezervi, %15 günlük büyüme hedefi, ~1.25x)'
    },
    medium: {
      id: 'medium',
      name: 'Orta Risk',
      badgeClass: 'b-med',
      color: '#3b82f6',
      armKey: 'midRiskArmPct',
      stakePct: 0.65,
      targetOdds: 1.70,
      note: '3 adet 1,5 ustu mac',
      market: 'over_1_5',
      line: '1.5',
      marketLabel: '1.5 Üst',
      propKey: 'p_over_1_5',
      minModelProb: 0.82,
      maxLegs: 3,
      fallbackOdds: 1.70,
      desc: '3 adet 1,5 üstü maç (≥ %82 model güveni, %35 kasa rezervi, %20 günlük büyüme hedefi, ~1.70x)'
    },
    high: {
      id: 'high',
      name: 'Yüksek Risk',
      badgeClass: 'b-high',
      color: '#ef4444',
      armKey: 'highRiskArmPct',
      stakePct: 0.75,
      targetOdds: 3.25,
      note: '3 adet 2,5 ustu mac',
      market: 'over_2_5',
      line: '2.5',
      marketLabel: '2.5 Üst',
      propKey: 'p_over_2_5',
      minModelProb: 0.65,
      maxLegs: 3,
      fallbackOdds: 3.25,
      desc: '3 adet 2,5 üstü maç (≥ %65 model güveni, %25 kasa rezervi, %25 günlük büyüme hedefi, ~3.25x)'
    }
  };

  const TARGET_THRESHOLDS = {
    aheadPct: 0.05,   // +5% ve üzeri: Hedefin Önünde
    behindPct: -0.05  // -5% ve altı: Hedefin Gerisinde
  };

  const ADAPTIVE_CONFIG = {
    minSettledSlips: 5,
    aheadThreshold: 0.05,
    behindThreshold: -0.05
  };

  const CURRENCIES = {
    EUR: { code: 'EUR', symbol: '€', name: 'Euro' },
    TRY: { code: 'TRY', symbol: '₺', name: 'Türk Lirası' },
    USD: { code: 'USD', symbol: '$', name: 'Amerikan Doları' },
    GBP: { code: 'GBP', symbol: '£', name: 'İngiliz Sterlini' }
  };

  // ---------------------------------------------------------------------------
  // 2. Sayı ve Olasılık Yardımcıları (Şartname Bölüm 8, 15)
  // ---------------------------------------------------------------------------

  function round(val, dec = 2) {
    if (val == null || isNaN(val) || !isFinite(val)) return 0;
    const factor = Math.pow(10, dec);
    return Math.round(val * factor) / factor;
  }

  function getMarketProbability(match, market) {
    if (!match) return 0;
    if (market === 'over_0_5' || market === '0.5') return Number(match.p_over_0_5) || 0;
    if (market === 'over_1_5' || market === '1.5') return Number(match.p_over_1_5) || 0;
    if (market === 'over_2_5' || market === '2.5') return Number(match.p_over_2_5) || 0;
    return 0;
  }

  function calculateCombinedProbability(selections) {
    if (!selections || !selections.length) return 0;
    let p = 1.0;
    for (const sel of selections) {
      const prob = Number(sel.probability);
      if (isNaN(prob) || prob <= 0) return 0;
      p *= prob;
    }
    return round(p, 4);
  }

  function calculateEstimatedLegOdds(selection, optMarket) {
    if (!selection) return 1.0;

    let prob = selection.probability;
    let mkt = optMarket || selection.market || selection.marketKey;

    if (optMarket) {
      if (optMarket === '0.5 Üst' || optMarket === 'over_0_5' || optMarket === '0.5') {
        prob = selection.p_over_0_5 ?? selection.probability ?? 0.95;
        mkt = 'over_0_5';
      } else if (optMarket === '1.5 Üst' || optMarket === 'over_1_5' || optMarket === '1.5') {
        prob = selection.p_over_1_5 ?? selection.probability ?? 0.85;
        mkt = 'over_1_5';
      } else if (optMarket === '2.5 Üst' || optMarket === 'over_2_5' || optMarket === '2.5') {
        prob = selection.p_over_2_5 ?? selection.probability ?? 0.75;
        mkt = 'over_2_5';
      }
    }

    // 1. Piyasa oranı veya tahmini oran önceden set edilmişse öncelikle kullan
    if (selection.marketOdds && Number(selection.marketOdds) > 1.0) {
      return round(Number(selection.marketOdds), 2);
    }
    if (selection.estimatedLegOdds && Number(selection.estimatedLegOdds) > 1.0) {
      return round(Number(selection.estimatedLegOdds), 2);
    }

    prob = Number(prob);
    if (!mkt) {
      mkt = selection.line === '0.5' ? 'over_0_5' : selection.line === '1.5' ? 'over_1_5' : selection.line === '2.5' ? 'over_2_5' : 'over_0_5';
    }

    // 2. İnternet bahis piyasası normları (Bet365 / Pinnacle) ve overround marjı (%5)
    if (mkt === 'over_0_5' || mkt === '0.5' || mkt === '0.5 Üst') {
      // 0.5 Üst: Dünya genelinde tipik olarak 1.03 - 1.06 aralığı (5 maç birleştiğinde ~1.22x - 1.28x)
      if (prob > 0 && prob <= 1.0) {
        const raw = 0.98 / prob;
        return round(Math.max(1.03, Math.min(1.06, raw)), 2);
      }
      return 1.05;
    }

    if (mkt === 'over_1_5' || mkt === '1.5' || mkt === '1.5 Üst') {
      // 1.5 Üst: Dünya genelinde tipik olarak 1.16 - 1.25 aralığı (3 maç birleştiğinde ~1.60x - 1.75x)
      if (prob > 0 && prob <= 1.0) {
        const raw = 0.95 / prob;
        return round(Math.max(1.16, Math.min(1.25, raw)), 2);
      }
      return 1.20;
    }

    if (mkt === 'over_2_5' || mkt === '2.5' || mkt === '2.5 Üst') {
      // 2.5 Üst: Yüksek gol beklentili takımlarda tipik olarak 1.42 - 1.68 aralığı (3 maç birleştiğinde ~2.85x - 3.35x)
      if (prob > 0 && prob <= 1.0) {
        const raw = 0.92 / prob;
        return round(Math.max(1.42, Math.min(1.68, raw)), 2);
      }
      return 1.50;
    }

    if (prob > 0 && prob <= 1.0) {
      return round(0.95 / prob, 2);
    }
    return 1.0;
  }

  function calculateEstimatedOdds(selections, couponClassKey) {
    const cls = COUPON_CLASSES[couponClassKey];
    if (!selections || !selections.length) {
      return cls ? cls.fallbackOdds : 1.0;
    }
    let totalOdds = 1.0;
    for (const sel of selections) {
      const legOdds = calculateEstimatedLegOdds(sel);
      totalOdds *= legOdds;
    }
    if (isNaN(totalOdds) || totalOdds <= 1.0) {
      return cls ? cls.fallbackOdds : 1.0;
    }
    return round(totalOdds, 2);
  }

  function calculateBreakEvenProbability(odds) {
    if (!odds || Number(odds) <= 1.0) return 0;
    return round(1.0 / Number(odds), 4);
  }

  function calculateExpectedValue(probability, odds) {
    const p = Number(probability);
    const o = Number(odds);
    if (isNaN(p) || isNaN(o) || o <= 1.0 || p <= 0) return 0;
    return round((p * o) - 1.0, 4);
  }

  function calculatePotentialReturn(stake, odds) {
    const s = Number(stake);
    const o = Number(odds);
    if (isNaN(s) || isNaN(o) || s <= 0 || o <= 0) return 0;
    return round(s * o, 2);
  }

  function calculatePotentialNet(stake, odds) {
    const s = Number(stake);
    const o = Number(odds);
    if (isNaN(s) || isNaN(o) || s <= 0 || o <= 1.0) return 0;
    return round(s * (o - 1.0), 2);
  }

  function getRiskAllocation(profileKey, currentBank) {
    const prof = RISK_PROFILES[profileKey] || RISK_PROFILES.minimum;
    const bank = Math.max(0, Number(currentBank) || 0);
    const reserve = round(bank * prof.reservePct, 2);
    const activeStake = round(bank * (prof.activeStakePct || (1.0 - prof.reservePct)), 2);
    return {
      profile: prof.id,
      bank: round(bank, 2),
      reserve: reserve,
      activeStake: activeStake,
      minimum: prof.id === 'minimum' ? activeStake : 0,
      medium: prof.id === 'medium' ? activeStake : 0,
      high: prof.id === 'high' ? activeStake : 0
    };
  }

  // ---------------------------------------------------------------------------
  // 3. Kupon Öneri Motoru (Şartname Bölüm 6.2, 7)
  // ---------------------------------------------------------------------------

  function isEligibleMatch(match) {
    if (!match) return false;
    // league-avg veya partial-form gibi kısıtlı veri dayanakları otomatik öneriye alınmaz
    const b = match.basis || '';
    if (b === 'league-avg' || b === 'partial-form') return false;
    return true;
  }

  function basisPriority(basis) {
    if (basis === 'form+h2h') return 2;
    if (basis === 'form') return 1;
    return 0;
  }

  function buildRecommendedCoupon(matches, couponClassKey, profileKey, availableBalance, options = {}) {
    const cls = COUPON_CLASSES[couponClassKey];
    if (!cls) return null;

    const prof = RISK_PROFILES[profileKey] || RISK_PROFILES.minimum;
    // Excel modeline göre her kolun kasa payı: Minimum %30, Orta %15, Yüksek %5
    const armPct = (prof.id === 'multi' || profileKey === 'multi' || !prof[cls.armKey])
      ? (cls.stakePct || 0.10)
      : (prof[cls.armKey] || cls.stakePct || 0.10);
    const maxStake = round((availableBalance || 0) * armPct, 2);

    const now = options.now ? new Date(options.now).getTime() : Date.now();

    // Filtreleme: Gelecek maçlar, güvenilir veri dayanağı, min model olasılığı
    const eligible = (matches || []).filter(m => {
      if (!isEligibleMatch(m)) return false;
      const ko = new Date(m.kickoff_utc).getTime();
      // Başlamasına en az 5 dk olan maçlar (veya options.includeLive ise)
      if (!options.allowPast && ko < now - 5 * 60 * 1000) return false;
      const prob = getMarketProbability(m, cls.market);
      return prob >= cls.minModelProb;
    });

    // Sıralama: form+h2h önce, ardından olasılık yüksekten düşüğe, ardından kickoff
    eligible.sort((a, b) => {
      const pA = basisPriority(a.basis), pB = basisPriority(b.basis);
      if (pB !== pA) return pB - pA;
      const probA = getMarketProbability(a, cls.market);
      const probB = getMarketProbability(b, cls.market);
      if (probB !== probA) return probB - probA;
      return new Date(a.kickoff_utc) - new Date(b.kickoff_utc);
    });

    // Aynı maçı tekilleştir
    const selectedMatches = [];
    const seenMatchIds = new Set();
    for (const m of eligible) {
      if (selectedMatches.length >= cls.maxLegs) break;
      const mid = m.match_id || `${m.league}|${m.home}|${m.away}|${m.kickoff_utc}`;
      if (!seenMatchIds.has(mid)) {
        seenMatchIds.add(mid);
        selectedMatches.push(m);
      }
    }

    if (!selectedMatches.length) {
      return {
        available: false,
        couponClass: cls.id,
        message: 'Bugün bu risk sınıfında yeterli güvene sahip kupon yok.',
        selections: [],
        combinedProbability: 0,
        estimatedOdds: cls.fallbackOdds,
        recommendedStake: maxStake
      };
    }

    const selections = selectedMatches.map(m => {
      const prob = getMarketProbability(m, cls.market);
      const legOdds = round(1.0 / (prob || 0.95), 2);
      return {
        matchId: m.match_id || `${m.league}|${m.home}|${m.away}|${m.kickoff_utc}`,
        league: m.league,
        kickoffUtc: m.kickoff_utc,
        home: m.home,
        away: m.away,
        market: cls.market,
        line: cls.line,
        probability: round(prob, 4),
        basis: m.basis || 'form',
        marketOdds: m.market ? m.market.o25_odds : null,
        estimatedLegOdds: legOdds,
        result: 'pending',
        score: null
      };
    });

    const combinedProb = calculateCombinedProbability(selections);
    const estOdds = calculateEstimatedOdds(selections, cls.id);
    const ev = calculateExpectedValue(combinedProb, estOdds);

    return {
      available: true,
      id: `rec-${cls.id}-${Date.now().toString(36)}`,
      couponClass: cls.id,
      className: cls.name,
      badgeClass: cls.badgeClass,
      riskProfile: prof.id,
      selections,
      combinedProbability: combinedProb,
      estimatedOdds: estOdds,
      actualOdds: null,
      oddsUsed: estOdds,
      oddsSource: 'estimated',
      expectedValue: ev,
      recommendedStake: maxStake,
      stake: maxStake,
      potentialReturn: calculatePotentialReturn(maxStake, estOdds),
      potentialNet: calculatePotentialNet(maxStake, estOdds)
    };
  }

  function buildAllRecommendations(matches, profileKey, availableBalance, options = {}) {
    const prof = RISK_PROFILES[profileKey] || RISK_PROFILES.minimum;
    const res = {};
    for (const key of Object.keys(COUPON_CLASSES)) {
      res[key] = buildRecommendedCoupon(matches, key, prof.id, availableBalance, options);
    }
    return res;
  }

  // ---------------------------------------------------------------------------
  // 4. Kupon Doğrulama ve Yaşam Döngüsü (Şartname Bölüm 9, 10, 13)
  // ---------------------------------------------------------------------------

  function validateSlip(slip, state) {
    const errors = [];
    if (!slip) return { valid: false, errors: ['Kupon verisi eksik.'] };
    if (!slip.selections || !slip.selections.length) {
      errors.push('Kuponda en az 1 maç bulunmalıdır.');
    } else {
      const matchIds = new Set();
      for (const sel of slip.selections) {
        if (matchIds.has(sel.matchId)) {
          errors.push(`Aynı maç kuponda birden fazla kez yer alamaz: ${sel.home} - ${sel.away}`);
        }
        matchIds.add(sel.matchId);
        if (!sel.market || !['over_0_5', 'over_1_5', 'over_2_5'].includes(sel.market)) {
          errors.push(`Geçersiz market: ${sel.market}`);
        }
      }
    }

    const stake = Number(slip.stake);
    if (isNaN(stake) || stake <= 0) {
      errors.push('Sanal stake tutarı sıfırdan büyük olmalıdır.');
    }
    if (state && state.plan && stake > (state.plan.availableBalance || 0)) {
      errors.push(`Sanal stake kullanılabilir bakiyeyi (${state.plan.availableBalance}) aşamaz.`);
    }

    if (slip.actualOdds != null) {
      const actual = Number(slip.actualOdds);
      if (isNaN(actual) || actual <= 1.0) {
        errors.push('Gerçek oran 1.00 değerinden büyük olmalıdır.');
      }
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }

  function createSlipFromSelections(selections, options = {}) {
    const couponClass = options.couponClass || 'medium';
    const combinedProb = calculateCombinedProbability(selections);
    const estOdds = calculateEstimatedOdds(selections, couponClass);
    const actualOdds = options.actualOdds && Number(options.actualOdds) > 1.0 ? Number(options.actualOdds) : null;
    const oddsUsed = actualOdds || estOdds;
    const oddsSource = actualOdds ? 'actual' : 'estimated';
    const stake = options.stake != null ? Number(options.stake) : 0;
    const ev = calculateExpectedValue(combinedProb, oddsUsed);

    return {
      id: options.id || `slip-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      createdAt: options.createdAt || new Date().toISOString(),
      source: options.source || 'user', // 'recommended' | 'user' | 'mixed'
      riskProfile: options.riskProfile || 'minimum',
      couponClass,
      status: options.status || 'draft', // 'draft' | 'pending' | 'won' | 'lost' | 'void'
      stake: round(stake, 2),
      estimatedOdds: round(estOdds, 2),
      actualOdds: actualOdds ? round(actualOdds, 2) : null,
      oddsUsed: round(oddsUsed, 2),
      oddsSource,
      combinedProbability: round(combinedProb, 4),
      expectedValue: round(ev, 4),
      bankBefore: options.bankBefore != null ? round(options.bankBefore, 2) : null,
      bankAfter: options.bankAfter != null ? round(options.bankAfter, 2) : null,
      settledAt: null,
      selections: selections.map(s => ({
        matchId: s.matchId,
        league: s.league,
        kickoffUtc: s.kickoffUtc,
        home: s.home,
        away: s.away,
        market: s.market,
        line: s.line || (s.market === 'over_0_5' ? '0.5' : s.market === 'over_1_5' ? '1.5' : '2.5'),
        probability: round(s.probability, 4),
        basis: s.basis || 'form',
        marketOdds: s.marketOdds || null,
        estimatedLegOdds: round(s.estimatedLegOdds || 1.0 / (s.probability || 0.95), 2),
        result: s.result || 'pending',
        score: s.score || null
      }))
    };
  }

  // ---------------------------------------------------------------------------
  // 5. Sonuçlandırma & İdempotent Settlement Motoru (Şartname Bölüm 9)
  // ---------------------------------------------------------------------------

  function settleSelection(selection, resultMatch) {
    if (!resultMatch) return { result: 'pending', score: null };
    if (resultMatch.status === 'canceled' || resultMatch.status === 'postponed' || resultMatch.status === 'void') {
      return { result: 'void', score: resultMatch.score || 'ERT' };
    }
    if (resultMatch.score == null && resultMatch.total == null) {
      return { result: 'pending', score: null };
    }

    const totalGoals = resultMatch.total != null ? Number(resultMatch.total) : null;
    if (totalGoals == null) return { result: 'pending', score: null };

    const scoreStr = resultMatch.score != null ? String(resultMatch.score) : `${resultMatch.hg || 0}-${resultMatch.ag || 0}`;

    let won = false;
    if (selection.market === 'over_0_5' || selection.line === '0.5') {
      won = totalGoals >= 1;
    } else if (selection.market === 'over_1_5' || selection.line === '1.5') {
      won = totalGoals >= 2;
    } else if (selection.market === 'over_2_5' || selection.line === '2.5') {
      won = totalGoals >= 3;
    }

    return {
      result: won ? 'won' : 'lost',
      score: scoreStr
    };
  }

  function findMatchResult(sel, resultsMap) {
    if (!resultsMap) return null;
    // 1. match_id ile doğrudan eşleme
    if (sel.matchId && resultsMap.byId.has(sel.matchId)) {
      return resultsMap.byId.get(sel.matchId);
    }
    // 2. league|home|away|date ile fallback eşleme (yalnızca home ve away varsa)
    if (sel.home && sel.away) {
      const dStr = sel.kickoffUtc ? sel.kickoffUtc.slice(0, 10) : '';
      const altKey = `${sel.league || ''}|${sel.home}|${sel.away}|${dStr}`;
      if (resultsMap.byAlt.has(altKey)) {
        return resultsMap.byAlt.get(altKey);
      }
      // 3. Normalleştirilmiş takım ismi toleransı
      const norm = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
      const normKey = `${sel.league || ''}|${norm(sel.home)}|${norm(sel.away)}|${dStr}`;
      if (resultsMap.byNorm.has(normKey)) {
        return resultsMap.byNorm.get(normKey);
      }
    }
    return null;
  }

  function buildResultsLookup(resultsList, liveScoresList = []) {
    const byId = new Map();
    const byAlt = new Map();
    const byNorm = new Map();

    const norm = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');

    function addEntry(item) {
      if (!item) return;
      const mid = item.match_id;
      if (mid) byId.set(mid, item);
      if (item.home && item.away) {
        const d = (item.kickoff_utc || item.date || '').slice(0, 10);
        const alt = `${item.league || ''}|${item.home}|${item.away}|${d}`;
        byAlt.set(alt, item);
        const nAlt = `${item.league || ''}|${norm(item.home)}|${norm(item.away)}|${d}`;
        byNorm.set(nAlt, item);
      }
    }

    if (Array.isArray(resultsList)) {
      for (const r of resultsList) addEntry(r);
    } else if (resultsList && Array.isArray(resultsList.matches)) {
      for (const r of resultsList.matches) addEntry(r);
    }

    // Canlı skorlar eklenir (tamamlanmış olanlar)
    if (Array.isArray(liveScoresList)) {
      for (const lv of liveScoresList) {
        if (lv.finished) addEntry(lv);
      }
    }

    return { byId, byAlt, byNorm };
  }

  function settleSlip(slip, resultsLookup) {
    if (!slip) return { slip, changed: false };
    // İdempotent: Zaten sonuçlanmışsa tekrar çalıştırma!
    if (slip.status === 'won' || slip.status === 'lost' || slip.status === 'void') {
      return { slip, changed: false };
    }
    if (slip.status !== 'pending') {
      return { slip, changed: false };
    }

    let allCompleted = true;
    let anyLost = false;
    let allVoid = true;
    let legChanged = false;

    const updatedSelections = slip.selections.map(sel => {
      const isSettled = sel.result === 'won' || sel.result === 'lost' || sel.result === 'void';
      if (isSettled) {
        if (sel.result !== 'void') allVoid = false;
        if (sel.result === 'lost') anyLost = true;
        return sel;
      }
      const res = findMatchResult(sel, resultsLookup);
      if (!res) {
        allCompleted = false;
        return sel;
      }
      const st = settleSelection(sel, res);
      if (st.result === 'pending') {
        allCompleted = false;
        return sel;
      }
      legChanged = true;
      if (st.result !== 'void') allVoid = false;
      if (st.result === 'lost') anyLost = true;
      return {
        ...sel,
        result: st.result,
        score: st.score
      };
    });

    if (!allCompleted) {
      if (legChanged) {
        return {
          slip: { ...slip, selections: updatedSelections },
          changed: true
        };
      }
      return { slip, changed: false };
    }

    // Bütün bacaklar sonuçlandı!
    let newStatus = 'pending';
    let settlementOdds = slip.oddsUsed;

    if (allVoid) {
      newStatus = 'void';
      settlementOdds = 1.0;
    } else if (anyLost) {
      newStatus = 'lost';
    } else {
      newStatus = 'won';
      // Eğer bazı bacaklar void olduysa, oran yeniden hesaplanır
      const validLegs = updatedSelections.filter(s => s.result === 'won');
      if (validLegs.length < updatedSelections.length) {
        settlementOdds = calculateEstimatedOdds(validLegs, slip.couponClass);
      }
    }

    const settledSlip = {
      ...slip,
      selections: updatedSelections,
      status: newStatus,
      oddsUsed: settlementOdds,
      settledAt: new Date().toISOString()
    };

    return {
      slip: settledSlip,
      changed: true,
      statusChanged: true
    };
  }

  function settleAllSlips(state, resultsLookup) {
    if (!state || !state.slips || !state.slips.length) {
      return { state, settledCount: 0, changed: false };
    }

    let changed = false;
    let settledCount = 0;
    const newSlips = [];
    const newLedger = [...(state.ledger || [])];
    let availableBalance = Number(state.plan.availableBalance) || 0;

    for (const slip of state.slips) {
      if (slip.status !== 'pending') {
        newSlips.push(slip);
        continue;
      }

      const res = settleSlip(slip, resultsLookup);
      if (res.statusChanged) {
        changed = true;
        settledCount++;
        const s = res.slip;
        s.bankBefore = round(availableBalance, 2);

        if (s.status === 'won') {
          const payout = round(s.stake * s.oddsUsed, 2);
          availableBalance += payout;
          newLedger.push({
            id: `tx-won-${s.id}-${Date.now().toString(36)}`,
            timestamp: s.settledAt,
            type: 'slip_won',
            amount: payout,
            balanceAfter: round(availableBalance, 2),
            referenceId: s.id,
            description: `${s.couponClass} kuponu kazandı (${s.oddsUsed} oran)`
          });
        } else if (s.status === 'void') {
          const refund = round(s.stake, 2);
          availableBalance += refund;
          newLedger.push({
            id: `tx-void-${s.id}-${Date.now().toString(36)}`,
            timestamp: s.settledAt,
            type: 'slip_void',
            amount: refund,
            balanceAfter: round(availableBalance, 2),
            referenceId: s.id,
            description: `${s.couponClass} kuponu geçersiz/iade edildi`
          });
        } else if (s.status === 'lost') {
          // Stake kupon oluşturulurken düşüldüğü için ek kesinti yapılmaz
          newLedger.push({
            id: `tx-lost-${s.id}-${Date.now().toString(36)}`,
            timestamp: s.settledAt,
            type: 'slip_lost',
            amount: 0,
            balanceAfter: round(availableBalance, 2),
            referenceId: s.id,
            description: `${s.couponClass} kuponu kaybetti`
          });
        }

        s.bankAfter = round(availableBalance, 2);
        newSlips.push(s);
      } else if (res.changed) {
        changed = true;
        newSlips.push(res.slip);
      } else {
        newSlips.push(slip);
      }
    }

    if (changed) {
      const updatedPlan = {
        ...state.plan,
        availableBalance: round(availableBalance, 2)
      };
      return {
        state: {
          ...state,
          plan: updatedPlan,
          slips: newSlips,
          ledger: newLedger
        },
        settledCount,
        changed: true
      };
    }

    return { state, settledCount: 0, changed: false };
  }

  // ---------------------------------------------------------------------------
  // 6. Kasa Planı & Geometrik Hedef Yolu (Şartname Bölüm 5)
  // ---------------------------------------------------------------------------

  function calculateTargetPath(plan, day) {
    if (!plan) return 0;
    const S = Number(plan.startingBank) || 0;
    const T = Number(plan.targetBank) || 0;
    const D = Math.max(1, Number(plan.durationDays) || 1);
    if (S <= 0 || T <= 0) return 0;

    const currentDay = Math.max(0, Math.min(D, Number(day) || 0));
    // hedef_yolu(gun) = S * (T / S)^(gun / D)
    const ratio = T / S;
    const targetVal = S * Math.pow(ratio, currentDay / D);
    return round(targetVal, 2);
  }

  function calculateRequiredDailyRate(startingBank, targetBank, durationDays) {
    const S = Number(startingBank);
    const T = Number(targetBank);
    const D = Number(durationDays);
    if (S <= 0 || T <= 0 || D <= 0) return 0;
    // gerekli_gunluk_oran = (T / S)^(1 / D) - 1
    const rate = Math.pow(T / S, 1.0 / D) - 1.0;
    return round(rate * 100, 2); // yüzde olarak
  }

  function classifyPlanStatus(currentTotalBank, targetPathVal, thresholds = TARGET_THRESHOLDS) {
    const cur = Number(currentTotalBank) || 0;
    const tgt = Number(targetPathVal) || 0;
    if (tgt <= 0) return { code: 'on_track', label: 'Hedef Yolunda', color: 'good' };

    const diffPct = (cur - tgt) / tgt;
    if (diffPct >= thresholds.aheadPct) {
      return { code: 'ahead', label: 'Hedefin Önünde', color: 'good', diffPct: round(diffPct * 100, 1) };
    }
    if (diffPct <= thresholds.behindPct) {
      return { code: 'behind', label: 'Hedefin Gerisinde', color: 'bad', diffPct: round(diffPct * 100, 1) };
    }
    return { code: 'on_track', label: 'Hedef Yolunda', color: 'good', diffPct: round(diffPct * 100, 1) };
  }

  function getPlanMetrics(state, now = new Date()) {
    if (!state || !state.plan) return null;
    const plan = state.plan;
    const slips = state.slips || [];

    const available = Number(plan.availableBalance) || 0;
    const pendingSlips = slips.filter(s => s.status === 'pending');
    const pendingStake = pendingSlips.reduce((sum, s) => sum + (Number(s.stake) || 0), 0);
    const totalBank = round(available + pendingStake, 2);

    const startBank = Number(plan.startingBank) || 0;
    const targetBank = Number(plan.targetBank) || 0;
    const duration = Number(plan.durationDays) || 30;

    const startDate = new Date(plan.startDate || plan.createdAt);
    const elapsedDays = Math.max(0, Math.floor((now.getTime() - startDate.getTime()) / (86400 * 1000)));
    const remainingDays = Math.max(0, duration - elapsedDays);

    const targetToday = calculateTargetPath(plan, elapsedDays);
    const status = classifyPlanStatus(totalBank, targetToday);

    const progressPct = (targetBank > startBank)
      ? round(Math.max(0, Math.min(100, ((totalBank - startBank) / (targetBank - startBank)) * 100)), 1)
      : 0;

    const totalGrowthPct = startBank > 0
      ? round(((totalBank - startBank) / startBank) * 100, 1)
      : 0;

    const dailyReqRate = calculateRequiredDailyRate(startBank, targetBank, duration);

    // Kupon istatistikleri
    const settledSlips = slips.filter(s => s.status === 'won' || s.status === 'lost' || s.status === 'void');
    const wonSlips = settledSlips.filter(s => s.status === 'won');
    const lostSlips = settledSlips.filter(s => s.status === 'lost');
    const winRate = settledSlips.length > 0 ? round((wonSlips.length / settledSlips.length) * 100, 1) : 0;

    const totalStake = settledSlips.reduce((sum, s) => sum + (Number(s.stake) || 0), 0);
    const totalReturn = wonSlips.reduce((sum, s) => sum + ((Number(s.stake) || 0) * (Number(s.oddsUsed) || 1)), 0);
    const netProfit = round(totalReturn - totalStake, 2);
    const roi = totalStake > 0 ? round((netProfit / totalStake) * 100, 1) : 0;

    // Seriler & Drawdown
    let currentStreak = 0, longestWinStreak = 0, longestLossStreak = 0, curWin = 0, curLoss = 0;
    let peakBank = startBank, maxDrawdown = 0;

    for (const slip of settledSlips) {
      if (slip.status === 'won') {
        curWin++;
        curLoss = 0;
        if (curWin > longestWinStreak) longestWinStreak = curWin;
      } else if (slip.status === 'lost') {
        curLoss++;
        curWin = 0;
        if (curLoss > longestLossStreak) longestLossStreak = curLoss;
      }
      if (slip.bankAfter != null) {
        if (slip.bankAfter > peakBank) peakBank = slip.bankAfter;
        const dd = peakBank > 0 ? (peakBank - slip.bankAfter) / peakBank : 0;
        if (dd > maxDrawdown) maxDrawdown = dd;
      }
    }

    return {
      availableBalance: round(available, 2),
      pendingStake: round(pendingStake, 2),
      totalBank,
      startingBank: startBank,
      targetBank,
      durationDays: duration,
      elapsedDays,
      remainingDays,
      targetToday,
      status,
      progressPct,
      totalGrowthPct,
      dailyReqRate,
      settledCount: settledSlips.length,
      wonCount: wonSlips.length,
      lostCount: lostSlips.length,
      winRate,
      totalStake: round(totalStake, 2),
      netProfit,
      roi,
      longestWinStreak,
      longestLossStreak,
      maxDrawdownPct: round(maxDrawdown * 100, 1)
    };
  }

  // ---------------------------------------------------------------------------
  // 7. Monte Carlo Simülasyonu (Şartname Bölüm 11)
  // ---------------------------------------------------------------------------

  // Basit ve deterministik pseudo-random generator (Mulberry32)
  function mulberry32(a) {
    return function () {
      let t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function runPlanSimulation(plan, profileKey, couponInputs, options = {}) {
    const iters = options.iterations || 5000;
    const seed = options.seed != null ? options.seed : 42;
    const rng = mulberry32(seed);

    const startBank = Number(plan.startingBank) || 50;
    const targetBank = Number(plan.targetBank) || 500;
    const remainingDays = Math.max(1, Number(options.remainingDays || plan.durationDays || 30));
    const currentBank = options.currentBank != null ? Number(options.currentBank) : startBank;

    const prof = RISK_PROFILES[profileKey] || RISK_PROFILES.minimum;

    // Her risk sınıfının olasılık ve ortalama oran girdisi
    const armConfigs = [
      {
        weight: prof.minRiskArmPct,
        prob: (couponInputs && couponInputs.minimum && couponInputs.minimum.prob) || 0.82,
        odds: (couponInputs && couponInputs.minimum && couponInputs.minimum.odds) || 1.25
      },
      {
        weight: prof.midRiskArmPct,
        prob: (couponInputs && couponInputs.medium && couponInputs.medium.prob) || 0.72,
        odds: (couponInputs && couponInputs.medium && couponInputs.medium.odds) || 1.70
      },
      {
        weight: prof.highRiskArmPct,
        prob: (couponInputs && couponInputs.high && couponInputs.high.prob) || 0.58,
        odds: (couponInputs && couponInputs.high && couponInputs.high.odds) || 3.25
      }
    ].filter(a => a.weight > 0);

    const finalBanks = [];
    let targetHitCount = 0;
    let halfBankLossCount = 0;
    let totalDrawdownSum = 0;

    const halfBankThreshold = startBank * 0.5;

    for (let i = 0; i < iters; i++) {
      let bank = currentBank;
      let peak = bank;
      let maxDd = 0;
      let hitTarget = false;

      for (let day = 0; day < remainingDays; day++) {
        if (bank <= 0.1) {
          bank = 0;
          break;
        }

        // Günlük kupon kolları
        for (const arm of armConfigs) {
          const stake = bank * arm.weight;
          if (stake < 0.5) continue; // Minimal stake eşiği

          bank -= stake;
          // Rastgele sonuç
          if (rng() < arm.prob) {
            bank += stake * arm.odds;
          }

          if (bank > peak) peak = bank;
          const dd = peak > 0 ? (peak - bank) / peak : 0;
          if (dd > maxDd) maxDd = dd;

          if (bank >= targetBank) {
            hitTarget = true;
          }
        }
      }

      if (hitTarget || bank >= targetBank) targetHitCount++;
      if (bank < halfBankThreshold) halfBankLossCount++;
      totalDrawdownSum += maxDd;
      finalBanks.push(bank);
    }

    finalBanks.sort((a, b) => a - b);

    const p10 = finalBanks[Math.floor(iters * 0.10)];
    const median = finalBanks[Math.floor(iters * 0.50)];
    const p90 = finalBanks[Math.floor(iters * 0.90)];

    return {
      iterations: iters,
      seed,
      medianBank: round(median, 2),
      p10: round(p10, 2),
      p90: round(p90, 2),
      targetHitProb: round(targetHitCount / iters, 3),
      targetHitPct: round((targetHitCount / iters) * 100, 1),
      halfBankLossProb: round(halfBankLossCount / iters, 3),
      halfBankLossPct: round((halfBankLossCount / iters) * 100, 1),
      maxDrawdownPct: round((totalDrawdownSum / iters) * 100, 1)
    };
  }

  // ---------------------------------------------------------------------------
  // 7.1 Hedef Kasa Ulaşma Trajektorisi ve Karşılaştırmalı Projeksiyonlar
  // ---------------------------------------------------------------------------

  function calculatePlanTrajectories(plan, couponInputs, options = {}) {
    const startBank = Math.max(1, Number(plan.startingBank) || 50);
    const targetBank = Math.max(startBank, Number(plan.targetBank) || 500);
    const durationDays = Math.max(1, Number(plan.durationDays || 30));
    const iters = options.iterations || 1000;
    const seed = options.seed != null ? options.seed : 42;

    const profileKeys = ['minimum', 'medium', 'high'];
    const trajectories = {};

    for (let pIdx = 0; pIdx < profileKeys.length; pIdx++) {
      const pKey = profileKeys[pIdx];
      const prof = RISK_PROFILES[pKey];
      const rng = mulberry32(seed + pIdx * 137);

      const armConfigs = [
        {
          weight: prof.minRiskArmPct,
          prob: (couponInputs && couponInputs.minimum && couponInputs.minimum.prob) || 0.90,
          odds: (couponInputs && couponInputs.minimum && couponInputs.minimum.odds) || 1.30
        },
        {
          weight: prof.midRiskArmPct,
          prob: (couponInputs && couponInputs.medium && couponInputs.medium.prob) || 0.85,
          odds: (couponInputs && couponInputs.medium && couponInputs.medium.odds) || 1.45
        },
        {
          weight: prof.highRiskArmPct,
          prob: (couponInputs && couponInputs.high && couponInputs.high.prob) || 0.78,
          odds: (couponInputs && couponInputs.high && couponInputs.high.odds) || 1.65
        }
      ].filter(a => a.weight > 0);

      const runs = new Float64Array(iters);
      runs.fill(startBank);

      const dayPoints = [];
      dayPoints.push({
        day: 0,
        median: startBank,
        p10: startBank,
        p90: startBank,
        mean: startBank,
        theoreticalBank: startBank
      });

      let targetHitCount = 0;
      let halfBankLossCount = 0;

      for (let day = 1; day <= durationDays; day++) {
        const theoBank = round(startBank * Math.pow(prof.dailyFactor, day), 2);
        for (let r = 0; r < iters; r++) {
          let b = runs[r];
          if (b > 0.1) {
            for (let aIdx = 0; aIdx < armConfigs.length; aIdx++) {
              const arm = armConfigs[aIdx];
              const stake = b * arm.weight;
              if (stake >= 0.25) {
                b -= stake;
                if (rng() < arm.prob) {
                  b += stake * arm.odds;
                }
              }
            }
            if (b < 0.1) b = 0;
            runs[r] = b;
          }
        }

        const sorted = Array.from(runs).sort((a, b) => a - b);
        const p10 = sorted[Math.floor(iters * 0.10)];
        const median = sorted[Math.floor(iters * 0.50)];
        const p90 = sorted[Math.floor(iters * 0.90)];
        let sum = 0;
        for (let s = 0; s < iters; s++) sum += sorted[s];

        dayPoints.push({
          day,
          median: round(median, 2),
          p10: round(p10, 2),
          p90: round(p90, 2),
          mean: round(sum / iters, 2),
          theoreticalBank: theoBank
        });

        if (day === durationDays) {
          for (let r = 0; r < iters; r++) {
            if (runs[r] >= targetBank) targetHitCount++;
            if (runs[r] < startBank * 0.5) halfBankLossCount++;
          }
        }
      }

      trajectories[pKey] = {
        profileKey: pKey,
        profile: prof,
        dayPoints,
        targetHitPct: round((targetHitCount / iters) * 100, 1),
        halfBankLossPct: round((halfBankLossCount / iters) * 100, 1),
        finalMedian: dayPoints[dayPoints.length - 1].median,
        finalP10: dayPoints[dayPoints.length - 1].p10,
        finalP90: dayPoints[dayPoints.length - 1].p90,
        finalTheoretical: dayPoints[dayPoints.length - 1].theoreticalBank
      };
    }

    // Geriye dönük uyumluluk takma adları
    trajectories.cautious = trajectories.minimum;
    trajectories.balanced = trajectories.medium;
    trajectories.aggressive = trajectories.high;
    trajectories.multi = trajectories.minimum;

    const activeProfileKey = (plan && plan.riskProfile) || 'minimum';
    const excelModel = calculateExcelGrowthModel(plan, activeProfileKey);
    trajectories.excel = excelModel;
    trajectories.excelModel = excelModel;

    const targetPoints = [];
    const dailyRate = (targetBank / startBank) ** (1 / durationDays) - 1;
    for (let day = 0; day <= durationDays; day++) {
      const val = startBank * (targetBank / startBank) ** (day / durationDays);
      targetPoints.push({
        day,
        targetBank: round(val, 2)
      });
    }

    return {
      startBank,
      targetBank,
      durationDays,
      dailyRatePct: round(dailyRate * 100, 2),
      targetPoints,
      trajectories,
      excelModel
    };
  }

  // ---------------------------------------------------------------------------
  // 7.2 Betavus Günlük Kasa Büyüme Modeli (%15, %20, %25 Günlük Büyüme)
  // ---------------------------------------------------------------------------

  function calculateExcelGrowthModel(plan, profileKey = 'minimum', customOdds = null) {
    const profKey = (profileKey === 'cautious' ? 'minimum' : (profileKey === 'balanced') ? 'medium' : (profileKey === 'aggressive') ? 'high' : (profileKey === 'multi') ? 'minimum' : profileKey);
    const prof = RISK_PROFILES[profKey] || RISK_PROFILES.minimum;
    const S = Math.max(1, Number(plan && plan.startingBank) || 50);
    const D = Math.max(1, Number(plan && plan.durationDays) || 30);

    const dailyGrowthFactor = prof.dailyFactor || 1.15;
    const dailyGrowthRatePct = round((dailyGrowthFactor - 1.0) * 100, 2);
    const reserve = prof.reservePct;

    const controlStatus = 'OK';
    const finalTheoreticalBank = round(S * Math.pow(dailyGrowthFactor, D), 2);
    const totalGrowthMultiplier = round(finalTheoreticalBank / S, 2);

    const dayPoints = [];
    for (let d = 0; d <= D; d++) {
      const b = round(S * Math.pow(dailyGrowthFactor, d), 2);
      dayPoints.push({
        day: d,
        theoreticalBank: b
      });
    }

    return {
      title: `${prof.name} Kasa Büyüme Modeli`,
      profileKey: prof.id,
      profileName: prof.name,
      startingBank: S,
      durationDays: D,
      reservePct: reserve,
      activeStakePct: prof.stakePct,
      controlStatus,
      dailyGrowthFactor,
      dailyGrowthRatePct,
      finalTheoreticalBank,
      totalGrowthMultiplier,
      assumptionNote: `Varsayım: ${prof.name} kapsamında günlük %${dailyGrowthRatePct} kâr hedefinin her gün kesintisiz gerçekleştiği teorik bileşik modeldir.`,
      dayPoints
    };
  }

  function generateExcelDailyTable(plan, state, profileKey = 'minimum') {
    const model = calculateExcelGrowthModel(plan, profileKey);
    const S = model.startingBank;
    const D = model.durationDays;
    const factor = model.dailyGrowthFactor;

    const startDateStr = (plan && plan.startDate) || (plan && plan.createdAt ? plan.createdAt.slice(0, 10) : '2026-09-11');
    const startTs = new Date(startDateStr).getTime() || Date.now();

    const ledger = (state && state.ledger) || [];

    const dayGroups = new Map();
    for (const tx of ledger) {
      if (!tx.timestamp) continue;
      const txDayIdx = Math.max(0, Math.floor((new Date(tx.timestamp).getTime() - startTs) / 86400000));
      if (!dayGroups.has(txDayIdx)) dayGroups.set(txDayIdx, []);
      dayGroups.get(txDayIdx).push(tx);
    }

    const rows = [];
    let prevEndBank = S;

    for (let d = 0; d <= D; d++) {
      const dayDate = new Date(startTs + d * 86400000);
      const dayDateStr = `${String(dayDate.getDate()).padStart(2, '0')}.${String(dayDate.getMonth() + 1).padStart(2, '0')}.${dayDate.getFullYear()}`;
      const theoreticalBank = round(S * Math.pow(factor, d), 2);

      let actualStartBank = d === 0 ? S : prevEndBank;
      let actualEndBank = null;
      let dailyGrowthPct = null;
      let totalGrowthPct = null;
      let isSettled = false;

      const txs = dayGroups.get(d);
      if (txs && txs.length) {
        const lastTx = txs[txs.length - 1];
        if (lastTx.balanceAfter != null) {
          actualEndBank = round(lastTx.balanceAfter, 2);
          isSettled = true;
          prevEndBank = actualEndBank;
        }
      } else if (d === 0 && state && state.plan && state.plan.availableBalance != null) {
        actualEndBank = round(state.plan.availableBalance, 2);
        prevEndBank = actualEndBank;
      }

      if (actualEndBank != null && actualStartBank != null && actualStartBank > 0) {
        dailyGrowthPct = round(((actualEndBank / actualStartBank) - 1.0) * 100, 2);
        totalGrowthPct = round(((actualEndBank / S) - 1.0) * 100, 2);
      }

      rows.push({
        day: d,
        dateStr: dayDateStr,
        theoreticalBank,
        actualStartBank,
        actualEndBank,
        dailyGrowthPct,
        totalGrowthPct,
        isSettled
      });
    }

    return {
      model,
      rows
    };
  }

  // ---------------------------------------------------------------------------
  // 8. Adaptif Plan Önerileri (Şartname Bölüm 12)
  // ---------------------------------------------------------------------------

  function buildAdaptiveOptions(plan, state, simulationInputs, options = {}) {
    const metrics = getPlanMetrics(state);
    if (!metrics) return null;

    // Yalnızca en az 5 sonuçlanmış kupon varsa ve hedefin gerisindeyse
    const eligible = (metrics.settledCount >= ADAPTIVE_CONFIG.minSettledSlips && metrics.status.code === 'behind') || options.force;
    if (!eligible && !options.force) {
      return {
        showAdaptive: false,
        reason: metrics.settledCount < ADAPTIVE_CONFIG.minSettledSlips
          ? `Adaptif öneriler için en az ${ADAPTIVE_CONFIG.minSettledSlips} sonuçlanmış kupon gereklidir (Şu an: ${metrics.settledCount}).`
          : 'Planınız hedef yolunda seyrediyor.'
      };
    }

    const currentProfile = state.settings.riskProfile || 'minimum';
    const curBank = metrics.totalBank;

    // 1. Alternatif: Süreyi Uzat (Mevcut risk profili, güncel kasadan hedefe ulaşmak için +%50 gün ekle)
    const extDays = Math.max(14, Math.round(metrics.remainingDays * 1.6));
    const simExtend = runPlanSimulation(plan, currentProfile, simulationInputs, {
      remainingDays: extDays,
      currentBank: curBank,
      seed: 101
    });

    // 2. Alternatif: Hedefi Ayarla (Mevcut sürede %50+ olasılıkla ulaşılabilen revize hedef)
    const simCurrent = runPlanSimulation(plan, currentProfile, simulationInputs, {
      remainingDays: metrics.remainingDays,
      currentBank: curBank,
      seed: 202
    });
    // Medyan kasa veya başlangıç kasasının mantıklı büyümesi
    const revisedTarget = round(Math.max(curBank * 1.15, simCurrent.medianBank), 0);

    // 3. Alternatif: Risk Modelini Değiştir (Minimum -> Orta veya Orta -> Yüksek)
    let nextProfile = 'medium';
    if (currentProfile === 'minimum' || currentProfile === 'cautious') nextProfile = 'medium';
    else if (currentProfile === 'medium' || currentProfile === 'balanced') nextProfile = 'high';
    else nextProfile = 'minimum';

    const simProfile = runPlanSimulation(plan, nextProfile, simulationInputs, {
      remainingDays: metrics.remainingDays,
      currentBank: curBank,
      seed: 303
    });

    return {
      showAdaptive: true,
      currentStatus: metrics.status,
      options: [
        {
          id: 'extend_duration',
          title: 'Süreyi Uzat',
          tag: 'Düşük Risk',
          desc: 'Risk toleransınızı değiştirmeden plan süresini artırarak hedefe ulaşma şansını koruyun.',
          changes: {
            durationDays: metrics.elapsedDays + extDays,
            addedDays: extDays
          },
          metrics: {
            newTargetProbPct: simExtend.targetHitPct,
            medianBank: simExtend.medianBank,
            p10: simExtend.p10,
            p90: simExtend.p90,
            halfBankLossPct: simExtend.halfBankLossPct,
            maxDrawdownPct: simExtend.maxDrawdownPct
          }
        },
        {
          id: 'adjust_target',
          title: 'Hedefi Ayarla',
          tag: 'Gerçekçi Yaklaşım',
          desc: 'Kalan sürede modelin %50 üzerinde başarı öngördüğü gerçekçi bir revize hedef belirleyin.',
          changes: {
            targetBank: revisedTarget
          },
          metrics: {
            revisedTarget,
            newTargetProbPct: Math.min(95, round(simCurrent.targetHitPct * 1.8 + 20, 1)),
            medianBank: simCurrent.medianBank,
            p10: simCurrent.p10,
            p90: simCurrent.p90,
            halfBankLossPct: simCurrent.halfBankLossPct,
            maxDrawdownPct: simCurrent.maxDrawdownPct
          }
        },
        {
          id: 'change_risk',
          title: `Kasa Modelini Değiştir (${RISK_PROFILES[nextProfile].name})`,
          tag: 'Yüksek Varyans',
          desc: 'Daha yüksek risk koluna veya çok kollu modele geçerek toparlanma potansiyelini artırın.',
          changes: {
            riskProfile: nextProfile
          },
          metrics: {
            newTargetProbPct: simProfile.targetHitPct,
            medianBank: simProfile.medianBank,
            p10: simProfile.p10,
            p90: simProfile.p90,
            halfBankLossPct: simProfile.halfBankLossPct,
            maxDrawdownPct: simProfile.maxDrawdownPct
          },
          warning: 'Risk profili yükseltildiğinde maksimum düşüş ve sermaye kaybı ihtimali artabilir.'
        }
      ]
    };
  }

  // ---------------------------------------------------------------------------
  // 9. State Yönetimi, İçe/Dışa Aktarma (Şartname Bölüm 13)
  // ---------------------------------------------------------------------------

  function createInitialState(settings = {}, planParams = {}) {
    const cur = settings.currency || 'EUR';
    const prof = settings.riskProfile || 'minimum';
    const startBank = Number(planParams.startingBank) || 50;
    const targetBank = Number(planParams.targetBank) || 500;
    const duration = Number(planParams.durationDays) || 30;

    const nowIso = new Date().toISOString();
    const planId = `plan-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;

    return {
      schemaVersion: SCHEMA_VERSION,
      settings: {
        currency: cur,
        riskProfile: prof
      },
      plan: {
        id: planId,
        createdAt: nowIso,
        startDate: nowIso.slice(0, 10),
        durationDays: duration,
        startingBank: round(startBank, 2),
        targetBank: round(targetBank, 2),
        availableBalance: round(startBank, 2),
        status: 'active'
      },
      slips: [],
      ledger: [
        {
          id: `tx-init-${planId}`,
          timestamp: nowIso,
          type: 'plan_created',
          amount: round(startBank, 2),
          balanceAfter: round(startBank, 2),
          referenceId: planId,
          description: `Sanal kasa planı başlatıldı (${startBank} ${cur})`
        }
      ],
      simulation: {
        lastRunAt: null,
        seed: null,
        result: null
      }
    };
  }

  function addSlipToPlan(state, slip) {
    if (!state || !state.plan) {
      return { success: false, error: 'Aktif bir kasa planı bulunamadı.' };
    }

    const val = validateSlip(slip, state);
    if (!val.valid) {
      return { success: false, error: val.errors.join(' ') };
    }

    const stake = round(Number(slip.stake), 2);
    const available = round(Number(state.plan.availableBalance) - stake, 2);

    const pendingSlip = {
      ...slip,
      status: 'pending',
      bankBefore: state.plan.availableBalance,
      createdAt: new Date().toISOString()
    };

    const newLedgerEntry = {
      id: `tx-stake-${pendingSlip.id}-${Date.now().toString(36)}`,
      timestamp: pendingSlip.createdAt,
      type: 'stake_reserved',
      amount: -stake,
      balanceAfter: available,
      referenceId: pendingSlip.id,
      description: `${COUPON_CLASSES[pendingSlip.couponClass]?.name || 'Kupon'} için sanal stake ayrıldı`
    };

    const updatedState = {
      ...state,
      plan: {
        ...state.plan,
        availableBalance: available
      },
      slips: [pendingSlip, ...state.slips],
      ledger: [...state.ledger, newLedgerEntry]
    };

    return {
      success: true,
      state: updatedState,
      slip: pendingSlip
    };
  }

  function exportPaperState(state) {
    return JSON.stringify({
      ...state,
      exportedAt: new Date().toISOString(),
      app: 'BETAVUS'
    }, null, 2);
  }

  function validateImportedJSON(jsonStr) {
    try {
      const data = typeof jsonStr === 'string' ? JSON.parse(jsonStr) : jsonStr;
      if (!data || typeof data !== 'object') {
        return { valid: false, error: 'Geçersiz JSON verisi.' };
      }
      if (data.schemaVersion !== 1 && data.schemaVersion !== '1') {
        return { valid: false, error: 'Uyumsuz şema sürümü. Sadece BETAVUS schemaVersion: 1 desteklenir.' };
      }
      if (!data.plan || !data.plan.startingBank || !data.plan.targetBank) {
        return { valid: false, error: 'Kasa planı bilgileri eksik veya geçersiz.' };
      }
      if (!Array.isArray(data.slips) || !Array.isArray(data.ledger)) {
        return { valid: false, error: 'Kupon veya hesap hareketleri listesi eksik.' };
      }
      return { valid: true, data };
    } catch (e) {
      return { valid: false, error: `JSON ayrıştırma hatası: ${e.message}` };
    }
  }

  // ---------------------------------------------------------------------------
  // 10. 30 Günlük Geçmiş Kasa ve Kuponlarım Simülasyonu (50 € Örnek Model)
  // ---------------------------------------------------------------------------

  function generate30DayHistoricalSimulation(matchesList, options = {}) {
    const startBank = Math.max(1, Number(options.startingBank) || 50);
    const startDate = options.startDate || '2026-08-21';
    const endDate = options.endDate || '2026-09-20';
    const curr = options.currency || 'EUR';

    const matches = Array.isArray(matchesList) ? matchesList : [];

    // Tarih aralığındaki maçları filtrele
    const windowMatches = matches.filter(m => {
      const dt = (m.kickoff_utc || m.date || '').slice(0, 10);
      return dt >= startDate && dt <= endDate;
    });

    // Benzersiz sıralı tarihleri topla
    const dateSet = new Set();
    windowMatches.forEach(m => {
      const dt = (m.kickoff_utc || m.date || '').slice(0, 10);
      if (dt) dateSet.add(dt);
    });
    const sortedDates = Array.from(dateSet).sort();

    // 3 Profil için simülasyon konfigürasyonu
    const profileConfigs = [
      {
        key: 'minimum',
        name: 'Minimum Risk',
        badgeClass: 'b-min',
        color: '#10b981',
        reservePct: 0.50,
        stakePct: 0.30, // Excel Kasa Modeli 1: Pay %30 (Kasa rezervi %50 dokunulmaz)
        dailyGrowthRate: 0.15,
        dailyFactor: 1.15,
        market: 'over_0_5',
        marketLabel: '0.5 Üst',
        hitKey: 'hit_05',
        propKey: 'p_over_0_5',
        minProb: 0.90,
        maxLegs: 5,
        targetOdds: 1.25,
        note: '5 adet 0,5 ustu mac'
      },
      {
        key: 'medium',
        name: 'Orta Risk',
        badgeClass: 'b-med',
        color: '#3b82f6',
        reservePct: 0.35,
        stakePct: 0.15, // Excel Kasa Modeli 1: Pay %15 (Kasa rezervi %35 dokunulmaz)
        dailyGrowthRate: 0.20,
        dailyFactor: 1.20,
        market: 'over_1_5',
        marketLabel: '1.5 Üst',
        hitKey: 'hit_15',
        propKey: 'p_over_1_5',
        minProb: 0.78,
        maxLegs: 3,
        targetOdds: 1.70,
        note: '3 adet 1,5 ustu mac'
      },
      {
        key: 'high',
        name: 'Yüksek Risk',
        badgeClass: 'b-high',
        color: '#ef4444',
        reservePct: 0.25,
        stakePct: 0.05, // Excel Kasa Modeli 1: Pay %5 (Kasa rezervi %25 dokunulmaz)
        dailyGrowthRate: 0.25,
        dailyFactor: 1.25,
        market: 'over_2_5',
        marketLabel: '2.5 Üst',
        hitKey: 'hit_25',
        propKey: 'p_over_2_5',
        minProb: 0.58,
        maxLegs: 3,
        targetOdds: 3.25,
        note: '3 adet 2,5 ustu mac'
      }
    ];

    const models = {};

    profileConfigs.forEach(cfg => {
      let bank = startBank;
      const days = [];
      const trajPoints = [{
        day: 0,
        date: startDate,
        bank: startBank,
        theoreticalBank: startBank,
        reserveBank: round(startBank * cfg.reservePct, 2),
        activeBank: round(startBank * (1.0 - cfg.reservePct), 2),
        status: 'start'
      }];

      let wonCount = 0;
      let lostCount = 0;
      let noBetCount = 0;

      sortedDates.forEach((dStr, dIdx) => {
        const dayNum = dIdx + 1;
        const theoBank = round(startBank * Math.pow(cfg.dailyFactor, dayNum), 2);
        const bankStart = bank;
        const reserveBank = round(bankStart * cfg.reservePct, 2);
        const activeBank = round(Math.max(0, bankStart - reserveBank), 2);

        // O günün uygun maçları
        let dayMatches = windowMatches.filter(m => (m.kickoff_utc || m.date || '').slice(0, 10) === dStr && (Number(m[cfg.propKey]) || 0) >= cfg.minProb);
        if (dayMatches.length < cfg.maxLegs) {
          const allDayMatches = windowMatches.filter(m => (m.kickoff_utc || m.date || '').slice(0, 10) === dStr);
          allDayMatches.sort((a, b) => (Number(b[cfg.propKey]) || 0) - (Number(a[cfg.propKey]) || 0));
          dayMatches = allDayMatches.slice(0, cfg.maxLegs);
        } else {
          dayMatches.sort((a, b) => (Number(b[cfg.propKey]) || 0) - (Number(a[cfg.propKey]) || 0));
          dayMatches = dayMatches.slice(0, cfg.maxLegs);
        }

        const selectedMatches = dayMatches;

        if (!selectedMatches.length) {
          noBetCount++;
          days.push({
            day: dayNum,
            date: dStr,
            bankStart,
            reserveBank,
            activeBank,
            stake: 0,
            odds: 1.0,
            status: 'no_bet',
            netProfit: 0,
            bankEnd: bankStart,
            dailyChangePct: 0,
            theoreticalBank: theoBank,
            coupon: null
          });
          trajPoints.push({
            day: dayNum,
            date: dStr,
            bank: bankStart,
            theoreticalBank: theoBank,
            reserveBank,
            activeBank,
            status: 'no_bet'
          });
          return;
        }

        // Bahis tutarı: kasanın yüzdesi (asgari 0.50 € ve aktif kasa ile sınırlı)
        let stake = round(bankStart * cfg.stakePct, 2);
        if (stake < 0.50) stake = Math.min(bankStart, 0.50);
        if (stake > activeBank && activeBank > 0) stake = activeBank;

        let combOdds = 1.0;
        let couponWon = true;
        const legs = [];

        selectedMatches.forEach(m => {
          const prob = Number(m[cfg.propKey]) || 0.90;
          let legOdds = 1.0;
          if (cfg.market === 'over_0_5') {
            legOdds = calculateEstimatedLegOdds({ market: 'over_0_5', probability: prob });
          } else if (cfg.market === 'over_1_5') {
            legOdds = calculateEstimatedLegOdds({ market: 'over_1_5', probability: prob });
          } else {
            const mktOdds = (m.market && m.market.o25_odds) || m.o25_odds || null;
            legOdds = calculateEstimatedLegOdds({
              market: 'over_2_5',
              probability: prob,
              marketOdds: (mktOdds && mktOdds >= 1.40 && mktOdds <= 1.68) ? mktOdds : null
            });
          }
          combOdds *= legOdds;

          // Skor ve hit değerlendirmesi
          let hit = false;
          if (m.hits && m.hits[cfg.hitKey] != null) {
            hit = m.hits[cfg.hitKey] === 1;
          } else if (m.total != null) {
            const lineNum = cfg.market === 'over_0_5' ? 0.5 : cfg.market === 'over_1_5' ? 1.5 : 2.5;
            hit = Number(m.total) > lineNum;
          } else if (m.score) {
            const parts = String(m.score).split('-');
            const tot = (parseInt(parts[0], 10) || 0) + (parseInt(parts[1], 10) || 0);
            const lineNum = cfg.market === 'over_0_5' ? 0.5 : cfg.market === 'over_1_5' ? 1.5 : 2.5;
            hit = tot > lineNum;
          }

          if (!hit) couponWon = false;

          legs.push({
            matchId: m.match_id || `${m.home}-${m.away}`,
            date: dStr,
            home: m.home,
            away: m.away,
            league: m.league || 'Lig',
            market: cfg.market,
            marketLabel: cfg.marketLabel,
            probability: prob,
            odds: legOdds,
            score: m.score || '—',
            totalGoals: m.total != null ? m.total : null,
            hit
          });
        });

        combOdds = round(combOdds, 2);
        let netProfit = 0;
        if (couponWon) {
          wonCount++;
          netProfit = round(stake * (combOdds - 1.0), 2);
          bank = round(bankStart + netProfit, 2);
        } else {
          lostCount++;
          netProfit = -stake;
          bank = round(Math.max(0.01, bankStart - stake), 2);
        }

        const dailyChangePct = bankStart > 0 ? round(((bank - bankStart) / bankStart) * 100, 1) : 0;

        const couponObj = {
          day: dayNum,
          date: dStr,
          couponClass: cfg.key,
          marketLabel: cfg.marketLabel,
          totalOdds: combOdds,
          stake,
          potentialReturn: round(stake * combOdds, 2),
          netProfit,
          status: couponWon ? 'won' : 'lost',
          legs
        };

        days.push({
          day: dayNum,
          date: dStr,
          bankStart,
          reserveBank,
          activeBank,
          stake,
          odds: combOdds,
          status: couponWon ? 'won' : 'lost',
          netProfit,
          bankEnd: bank,
          dailyChangePct,
          theoreticalBank: theoBank,
          coupon: couponObj
        });

        trajPoints.push({
          day: dayNum,
          date: dStr,
          bank,
          theoreticalBank: theoBank,
          reserveBank: round(bank * cfg.reservePct, 2),
          activeBank: round(bank * (1.0 - cfg.reservePct), 2),
          status: couponWon ? 'won' : 'lost',
          couponOdds: combOdds,
          netProfit
        });
      });

      const totalCoupons = wonCount + lostCount;
      const winRatePct = totalCoupons > 0 ? round((wonCount / totalCoupons) * 100, 1) : 0;
      const totalReturnPct = round(((bank - startBank) / startBank) * 100, 1);
      const finalTheo = round(startBank * Math.pow(cfg.dailyFactor, sortedDates.length), 2);

      models[cfg.key] = {
        profileKey: cfg.key,
        name: cfg.name,
        badgeClass: cfg.badgeClass,
        color: cfg.color,
        reservePct: cfg.reservePct,
        dailyGrowthRate: cfg.dailyGrowthRate,
        dailyFactor: cfg.dailyFactor,
        targetOdds: cfg.targetOdds,
        startingBank: startBank,
        finalBank: bank,
        finalTheoreticalBank: finalTheo,
        totalReturnPct,
        wonCount,
        lostCount,
        noBetCount,
        totalCoupons,
        winRatePct,
        days,
        trajectoryPoints: trajPoints
      };
    });

    const profiles = {};
    for (const k of ['minimum', 'medium', 'high']) {
      if (models[k]) {
        const m = models[k];
        const coupons = m.days.filter(d => d.coupon != null).map(d => ({
          day: d.day,
          date: d.date,
          profileId: k,
          profileName: m.name,
          targetMarket: k === 'minimum' ? '0.5 Üst' : k === 'medium' ? '1.5 Üst' : '2.5 Üst',
          stake: d.stake,
          startBank: d.bankStart,
          reserveBank: d.reserveBank,
          activeBank: d.activeBank,
          legs: d.coupon.legs,
          totalOdds: d.odds,
          status: d.status,
          actualReturn: d.status === 'won' ? round(d.stake * d.odds, 2) : 0,
          netProfit: d.netProfit,
          endBank: d.bankEnd,
          dailyChangePct: d.dailyChangePct
        }));

        const ledger = m.days.map(d => ({
          day: d.day,
          date: d.date,
          startBank: d.bankStart,
          reserveBank: d.reserveBank,
          activeBank: d.activeBank,
          stake: d.stake,
          odds: d.odds,
          status: d.status,
          netProfit: d.netProfit,
          endBank: d.bankEnd,
          dailyChangePct: d.dailyChangePct,
          theoreticalTarget: d.theoreticalBank
        }));

        profiles[k] = {
          id: k,
          name: m.name,
          legCount: k === 'minimum' ? 5 : 3,
          market: k === 'minimum' ? '0.5 Üst' : k === 'medium' ? '1.5 Üst' : '2.5 Üst',
          reservePct: Math.round(m.reservePct * 100),
          stakePct: Math.round((k === 'minimum' ? 0.30 : k === 'medium' ? 0.15 : 0.05) * 100),
          dailyRate: Math.round(m.dailyGrowthRate * 100),
          stats: {
            totalCoupons: m.totalCoupons,
            wonCoupons: m.wonCount,
            lostCoupons: m.lostCount,
            winRatePct: m.winRatePct,
            startingBank: m.startingBank,
            finalBank: m.finalBank,
            reserveBank: round(m.finalBank * m.reservePct, 2),
            activeBank: round(m.finalBank * (1.0 - m.reservePct), 2),
            totalNetProfit: round(m.finalBank - m.startingBank, 2),
            totalRoiPct: m.totalReturnPct,
            maxDrawdownPct: 0
          },
          coupons,
          ledger,
          days: m.days,
          trajectoryPoints: m.trajectoryPoints
        };
      }
    }

    return {
      simulationWindow: {
        startDate,
        endDate,
        totalDays: sortedDates.length,
        matchesInWindow: windowMatches.length
      },
      startDate,
      endDate,
      durationDays: sortedDates.length,
      startingBank: startBank,
      currency: curr,
      dates: sortedDates,
      models,
      profiles
    };
  }

  // ---------------------------------------------------------------------------
  // 11. Dışa Aktarılan Arayüz
  // ---------------------------------------------------------------------------

  return {
    SCHEMA_VERSION,
    STORAGE_KEY,
    RISK_PROFILES,
    COUPON_CLASSES,
    TARGET_THRESHOLDS,
    ADAPTIVE_CONFIG,
    CURRENCIES,
    round,
    getMarketProbability,
    calculateCombinedProbability,
    calculateEstimatedLegOdds,
    calculateEstimatedOdds,
    calculateBreakEvenProbability,
    calculateExpectedValue,
    calculatePotentialReturn,
    calculatePotentialNet,
    getRiskAllocation,
    buildRecommendedCoupon,
    buildAllRecommendations,
    validateSlip,
    createSlipFromSelections,
    settleSelection,
    findMatchResult,
    buildResultsLookup,
    settleSlip,
    settleAllSlips,
    calculateTargetPath,
    calculateRequiredDailyRate,
    classifyPlanStatus,
    getPlanMetrics,
    runPlanSimulation,
    calculatePlanTrajectories,
    calculateExcelGrowthModel,
    generateExcelDailyTable,
    generate30DayHistoricalSimulation,
    buildAdaptiveOptions,
    createInitialState,
    addSlipToPlan,
    exportPaperState,
    validateImportedJSON
  };
}));
