"""Shared BETAVUS goal model: weighted team scoring rates with match-recency
decay, a per-league Dixon-Coles low-score correction, and head-to-head
blending.

update_predictions.py (live predictions), backtest.py (walk-forward
backtest) and build_results.py (leak-free reconstruction) all import
LeagueModel from here instead of keeping their own copies, so there is
exactly one implementation of the scoring math.

Model, in order:
  1. Each team's home/away scoring and conceding rates are weighted by
     season recency (caller-supplied season weight) AND by how many
     matches ago that specific appearance was (exponential decay, half-life
     RECENCY_HALF_LIFE_MATCHES) - a team's last 5-10 matches dominate its
     rate estimate instead of being diluted evenly across a whole season.
  2. lam_home/lam_away come from averaging a team's own scoring rate with
     its opponent's conceding rate, same as before.
  3. If >=2 head-to-head meetings exist, the combined total is blended
     towards the H2H average (0.72/0.28), then rescaled back onto
     lam_home/lam_away keeping their ratio, and clamped to [0.30, 6.0]
     combined - same bounds the live model always used.
  4. Dixon-Coles: independent Poisson(lam_home) x Poisson(lam_away)
     under-counts the low scores (0-0, 1-0, 0-1, 1-1) relative to what
     leagues actually produce. A tau(x,y,rho) correction is applied to
     those four cells and the joint grid is renormalized. rho is fit per
     league by a 1D grid-search MLE against that league's own historical
     low-score frequencies (Dixon & Coles 1997) - since tau=1 everywhere
     else, only matches that actually finished 0-0/1-0/0-1/1-1 affect the
     fit.
"""

import math

H2H_MAX = 8
RECENCY_HALF_LIFE_MATCHES = 6  # a team's Nth-most-recent match counts for 2**-(N/6)
MAX_GOALS = 15                 # scoreline grid bound for the Dixon-Coles sum
RHO_GRID = [round(-0.35 + 0.01 * i, 2) for i in range(41)]  # -0.35 .. 0.05
DEFAULT_RHO = -0.10            # literature default when a league has no/too little data
# Piyasa harmanı: maçın 2.5 üst/alt oranı varsa toplam lambda'nın bu kadarı piyasanın
# ima ettiği toplamdan gelir. Model tek başına yüksek lambda'larda 0.3-0.5 gol iyimser;
# piyasa kadro/sakatlık/motivasyon bilgisini taşıyor. scripts/tune_market_blend.py
# walk-forward ölçümü (2021/22-2026/27, 17.002 maç): Brier 0.1624 -> 0.1584,
# 1.5+ vurgu isabeti %84.6 -> %88.4, vurgu sayısı yalnızca -%7.
MARKET_WEIGHT = 0.9


def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def poisson_over(lam, n):
    """P(X > n) for X ~ Poisson(lam) - plain total-goals tail, no Dixon-Coles.
    Kept for anything that only wants a quick single-lambda estimate."""
    term = math.exp(-lam)
    cdf = term
    for k in range(1, n + 1):
        term *= lam / k
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


def _tau(x, y, lam_h, lam_a, rho):
    if x == 0 and y == 0:
        return 1 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1 + lam_h * rho
    if x == 1 and y == 0:
        return 1 + lam_a * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _safe_rho(lam_h, lam_a, rho):
    """Clamp rho so all four tau cells stay non-negative for this (lam_h, lam_a)."""
    lam_h = max(lam_h, 1e-6)
    lam_a = max(lam_a, 1e-6)
    hi = min(1.0 / (lam_h * lam_a), 1.0) - 1e-6
    lo = max(-1.0 / lam_h, -1.0 / lam_a) + 1e-6
    return min(max(rho, lo), hi)


def dc_score_grid(lam_h, lam_a, rho, max_goals=MAX_GOALS):
    """Dixon-Coles-adjusted, renormalized P(home=x, away=y) for x,y in [0,max_goals]."""
    rho = _safe_rho(lam_h, lam_a, rho)
    px = [poisson_pmf(x, lam_h) for x in range(max_goals + 1)]
    py = [poisson_pmf(y, lam_a) for y in range(max_goals + 1)]
    grid, total = {}, 0.0
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            p = px[x] * py[y] * _tau(x, y, lam_h, lam_a, rho)
            grid[(x, y)] = p
            total += p
    if total > 0:
        for k in grid:
            grid[k] /= total
    return grid


def low_total_cdf(lam_h, lam_a, rho):
    """P(toplam<=0), P(toplam<=1), P(toplam<=2) - dc_score_grid ile aynı sonuç, ama
    yalnızca x+y<=2 hücreleri + kapalı form normalizasyonla (ızgaranın ~1/40'ı)."""
    rho = _safe_rho(lam_h, lam_a, rho)
    px = [poisson_pmf(x, lam_h) for x in range(3)]
    py = [poisson_pmf(y, lam_a) for y in range(3)]
    total = (sum(poisson_pmf(x, lam_h) for x in range(MAX_GOALS + 1))
             * sum(poisson_pmf(y, lam_a) for y in range(MAX_GOALS + 1)))
    cell = {}
    for x in range(3):
        for y in range(3 - x):
            t = _tau(x, y, lam_h, lam_a, rho)
            cell[(x, y)] = px[x] * py[y] * t
            total += px[x] * py[y] * (t - 1)
    c0 = cell[(0, 0)] / total
    c1 = c0 + (cell[(1, 0)] + cell[(0, 1)]) / total
    c2 = c1 + (cell[(2, 0)] + cell[(1, 1)] + cell[(0, 2)]) / total
    return c0, c1, c2


def market_total(lam_h, lam_a, rho, p_over25):
    """Modelin ev/deplasman oranı ve rho'su korunarak, Dixon-Coles P(toplam>2.5)
    değerini piyasanınkine eşitleyen toplam lambda (ikiye bölme)."""
    share = lam_h / (lam_h + lam_a)
    lo, hi = 0.3, 8.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if 1 - low_total_cdf(mid * share, mid * (1 - share), rho)[2] < p_over25:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def market_p_over25(o25_odds, u25_odds):
    """2.5 üst/alt oranlarından marjı ayıklanmış P(2.5 üst); oran yoksa None."""
    try:
        io, iu = 1 / float(o25_odds), 1 / float(u25_odds)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return io / (io + iu)


def fit_rho(low_score_matches):
    """low_score_matches: iterable of (hg, ag, lam_h, lam_a) for historical
    matches whose actual score has hg<=1 and ag<=1 (the only cells tau
    touches). Returns the RHO_GRID value maximizing the log-likelihood of
    those observed low scores - every other historical scoreline has
    tau==1 (log-contribution 0) for every candidate rho, so it can't
    affect the argmax and is skipped for speed."""
    rows = list(low_score_matches)
    if not rows:
        return DEFAULT_RHO
    best_rho, best_ll = DEFAULT_RHO, None
    for rho in RHO_GRID:
        ll = 0.0
        for hg, ag, lh, la in rows:
            t = _tau(hg, ag, lh, la, _safe_rho(lh, la, rho))
            ll += math.log(max(t, 1e-9))
        if best_ll is None or ll > best_ll:
            best_ll, best_rho = ll, rho
    return best_rho


KEY_PLAYER_DAMPING_ALPHA = 0.5  # a missing player's output share only partly
                                 # maps onto team lambda loss - teammates absorb some
KEY_PLAYER_MAX_DAMPING = 0.6    # cap combined damping so lambda never collapses to ~0


def key_player_damping(lam_home, lam_away, rho, home_missing_share, away_missing_share,
                        alpha=KEY_PLAYER_DAMPING_ALPHA, max_damp=KEY_PLAYER_MAX_DAMPING):
    """Damp lam_home/lam_away for a missing key attacking player and
    recompute the Over probabilities from the same rho - shared by
    scripts/fetch_injuries.py (a Transfermarkt-listed absence, known days
    ahead) and scripts/fetch_lineups.py (the confirmed real starting XI,
    known ~60-75 min ahead) so both apply identical math to whichever
    "missing share" they've each independently worked out. Returns None if
    neither side has any damping to apply.
    """
    damp_home = min(max_damp, alpha * home_missing_share)
    damp_away = min(max_damp, alpha * away_missing_share)
    if not damp_home and not damp_away:
        return None
    lam_home = lam_home * (1 - damp_home)
    lam_away = lam_away * (1 - damp_away)
    grid = dc_score_grid(lam_home, lam_away, rho)

    def over(n):
        return max(0.0, min(1.0, sum(p for (x, y), p in grid.items() if x + y > n)))

    return {
        "lam_home": round(lam_home, 3), "lam_away": round(lam_away, 3),
        "exp_goals": round(lam_home + lam_away, 3),
        "p_over_0_5": round(over(0), 4), "p_over_1_5": round(over(1), 4),
        "p_over_2_5": round(over(2), 4),
    }


class LeagueModel:
    """Weighted home/away scoring rates (with match-recency decay), a
    Dixon-Coles rho fit to the league's own low-score frequencies, and a
    head-to-head record.

    seasons: list of (matches, season_weight); each match is a dict with
    "home", "away", "hg", "ag", "date" (ISO string, used only for sorting).

    xg_seasons (optional): same shape, but each match has "home_xg"/"away_xg"
    (see scripts/opta_xg.py) instead of "hg"/"ag" - a team's own recency-
    weighted xG average is blended into its scoring-rate estimate with
    weight xg_weight (0 = pure goals, same as omitting xg_seasons entirely;
    1 = pure xG). Blended independently per team/side, so a team with no xG
    history (not yet covered by the upstream source) just falls back to its
    goals-based rate for that component instead of the whole match. See
    scripts/tune_xg_weight.py for how a league's weight is actually chosen.

    sos_strength (optional, 0..1): Strength-of-Schedule correction - a
    team's own scoring/conceding rate is itself a plain average over
    whichever opponents it happened to face, so it's biased by how tough
    that particular schedule was (see Strength of Schedule (SoS).txt for
    the write-up this implements). Each rate is rescaled by
    (league_average / average_opponent_rate_faced) ** sos_strength, using
    that specific opponent's own raw (pre-SoS) rate as the reference -
    single-pass, not an iterative joint solve.

    Measured with scripts/tune_sos_strength.py's walk-forward backtest
    across all 9 leagues: essentially no effect (0.0000-0.0002 Brier
    points, weaker than even the xG blend's already-marginal gain) - the
    model's existing per-match home-attack/away-defense pairing already
    captures most of what schedule strength would otherwise correct for.
    NOT wired into the live pipeline; default 0.0 (off) everywhere.
    """

    def __init__(self, seasons, half_life_matches=RECENCY_HALF_LIFE_MATCHES,
                 fit_rho_=True, default_rho=DEFAULT_RHO,
                 xg_seasons=None, xg_weight=0.0, sos_strength=0.0):
        home_apps, away_apps = {}, {}  # team -> [(date, hg, ag, season_weight, opponent), ...]
        self.h2h = {}
        all_matches = []
        hs, as_ = [0.0, 0.0], [0.0, 0.0]
        for matches, w in seasons:
            for m in matches:
                home, away, hg, ag = m["home"], m["away"], m["hg"], m["ag"]
                home_apps.setdefault(home, []).append((m.get("date", ""), hg, ag, w, away))
                away_apps.setdefault(away, []).append((m.get("date", ""), hg, ag, w, home))
                self.h2h.setdefault(frozenset((home, away)), []).append((m.get("date", ""), hg + ag))
                hs[0] += hg * w
                hs[1] += w
                as_[0] += ag * w
                as_[1] += w
                all_matches.append(m)
        self.base_home = hs[0] / hs[1] if hs[1] else 1.5
        self.base_away = as_[0] / as_[1] if as_[1] else 1.1

        decay = math.log(2) / half_life_matches
        self.home_gf, self.home_ga = {}, {}
        self.away_gf, self.away_ga = {}, {}
        for team, apps in home_apps.items():
            apps.sort(key=lambda r: r[0], reverse=True)
            for rank, (_, hg, ag, w, _opp) in enumerate(apps):
                rw = w * math.exp(-decay * rank)
                self._add(self.home_gf, team, hg, rw)
                self._add(self.home_ga, team, ag, rw)
        for team, apps in away_apps.items():
            apps.sort(key=lambda r: r[0], reverse=True)
            for rank, (_, hg, ag, w, _opp) in enumerate(apps):
                rw = w * math.exp(-decay * rank)
                self._add(self.away_gf, team, ag, rw)
                self._add(self.away_ga, team, hg, rw)

        self.sos_strength = sos_strength
        self.home_gf_sos, self.home_ga_sos = {}, {}
        self.away_gf_sos, self.away_ga_sos = {}, {}
        if sos_strength:
            def sos_mult(avg_opp, league_avg):
                if not avg_opp or not league_avg:
                    return 1.0
                return max(0.5, min(2.0, league_avg / avg_opp)) ** sos_strength

            for team, apps in home_apps.items():
                def_sum, def_w, att_sum, att_w = 0.0, 0.0, 0.0, 0.0
                for rank, (_, _hg, _ag, w, opp) in enumerate(apps):
                    rw = w * math.exp(-decay * rank)
                    def_sum += self._avg(self.away_ga, opp, self.base_home) * rw
                    att_sum += self._avg(self.away_gf, opp, self.base_away) * rw
                    def_w += rw
                    att_w += rw
                raw_gf = self._avg(self.home_gf, team, self.base_home)
                raw_ga = self._avg(self.home_ga, team, self.base_away)
                self.home_gf_sos[team] = raw_gf * sos_mult(def_w and def_sum / def_w, self.base_home)
                self.home_ga_sos[team] = raw_ga * sos_mult(self.base_away, att_w and att_sum / att_w)
            for team, apps in away_apps.items():
                def_sum, def_w, att_sum, att_w = 0.0, 0.0, 0.0, 0.0
                for rank, (_, _hg, _ag, w, opp) in enumerate(apps):
                    rw = w * math.exp(-decay * rank)
                    def_sum += self._avg(self.home_ga, opp, self.base_away) * rw
                    att_sum += self._avg(self.home_gf, opp, self.base_home) * rw
                    def_w += rw
                    att_w += rw
                raw_gf = self._avg(self.away_gf, team, self.base_away)
                raw_ga = self._avg(self.away_ga, team, self.base_home)
                self.away_gf_sos[team] = raw_gf * sos_mult(def_w and def_sum / def_w, self.base_away)
                self.away_ga_sos[team] = raw_ga * sos_mult(self.base_home, att_w and att_sum / att_w)

        self.xg_weight = xg_weight
        self.home_xgf, self.home_xga = {}, {}
        self.away_xgf, self.away_xga = {}, {}
        if xg_seasons and xg_weight > 0:
            xg_home_apps, xg_away_apps = {}, {}
            for matches, w in xg_seasons:
                for m in matches:
                    home, away = m["home"], m["away"]
                    hxg, axg = m["home_xg"], m["away_xg"]
                    xg_home_apps.setdefault(home, []).append((m.get("date", ""), hxg, axg, w))
                    xg_away_apps.setdefault(away, []).append((m.get("date", ""), hxg, axg, w))
            for team, apps in xg_home_apps.items():
                apps.sort(key=lambda r: r[0], reverse=True)
                for rank, (_, hxg, axg, w) in enumerate(apps):
                    rw = w * math.exp(-decay * rank)
                    self._add(self.home_xgf, team, hxg, rw)
                    self._add(self.home_xga, team, axg, rw)
            for team, apps in xg_away_apps.items():
                apps.sort(key=lambda r: r[0], reverse=True)
                for rank, (_, hxg, axg, w) in enumerate(apps):
                    rw = w * math.exp(-decay * rank)
                    self._add(self.away_xgf, team, axg, rw)
                    self._add(self.away_xga, team, hxg, rw)

        self.rho = self._fit_rho(all_matches) if fit_rho_ else default_rho

    @staticmethod
    def _add(store, key, value, weight):
        e = store.setdefault(key, [0.0, 0.0])
        e[0] += value * weight
        e[1] += weight

    @staticmethod
    def _avg(store, key, fallback):
        e = store.get(key)
        return e[0] / e[1] if e and e[1] else fallback

    def _base_lambdas(self, home, away):
        if self.sos_strength:
            hgf = self.home_gf_sos.get(home, self.base_home)
            hga = self.home_ga_sos.get(home, self.base_away)
            agf = self.away_gf_sos.get(away, self.base_away)
            aga = self.away_ga_sos.get(away, self.base_home)
        else:
            hgf = self._avg(self.home_gf, home, self.base_home)
            hga = self._avg(self.home_ga, home, self.base_away)
            agf = self._avg(self.away_gf, away, self.base_away)
            aga = self._avg(self.away_ga, away, self.base_home)
        if self.xg_weight > 0:
            w = self.xg_weight
            hxgf = self._avg(self.home_xgf, home, None)
            hxga = self._avg(self.home_xga, home, None)
            axgf = self._avg(self.away_xgf, away, None)
            axga = self._avg(self.away_xga, away, None)
            if hxgf is not None:
                hgf = (1 - w) * hgf + w * hxgf
            if hxga is not None:
                hga = (1 - w) * hga + w * hxga
            if axgf is not None:
                agf = (1 - w) * agf + w * axgf
            if axga is not None:
                aga = (1 - w) * aga + w * axga
        return (hgf + aga) / 2, (agf + hga) / 2

    def _fit_rho(self, all_matches):
        rows = []
        for m in all_matches:
            hg, ag = m["hg"], m["ag"]
            if hg > 1 or ag > 1:
                continue
            lh, la = self._base_lambdas(m["home"], m["away"])
            rows.append((hg, ag, lh, la))
        return fit_rho(rows)

    def predict_from_lambdas(self, lam_home, lam_away, basis, h2h_used=0):
        """Build the same output shape as predict(), for a caller that has
        already computed (or adjusted) lam_home/lam_away itself - e.g.
        scripts/fetch_lineups.py damping a team's lambda for a missing key
        player, after predict() has already applied the H2H blend."""
        grid = dc_score_grid(lam_home, lam_away, self.rho)

        def over(n):
            return max(0.0, min(1.0, sum(p for (x, y), p in grid.items() if x + y > n)))

        return {
            "basis": basis,
            "h2h_matches_used": h2h_used,
            "lam_home": round(lam_home, 3),
            "lam_away": round(lam_away, 3),
            "exp_goals": round(lam_home + lam_away, 3),
            "rho": round(self.rho, 3),
            "p_over_0_5": round(over(0), 4),
            "p_over_1_5": round(over(1), 4),
            "p_over_2_5": round(over(2), 4),
        }

    def predict(self, home, away, market_p25=None):
        """market_p25: bahis piyasasının P(2.5 üst) tahmini (marj ayıklanmış), varsa.
        Toplam gol beklentisi MARKET_WEIGHT oranında piyasanın ima ettiği toplama
        çekilir; ev/deplasman oranı ve rho modelden kalır."""
        lam_home, lam_away = self._base_lambdas(home, away)
        known = (home in self.home_gf) + (away in self.away_gf)
        basis = "form" if known == 2 else "partial-form" if known == 1 else "league-avg"

        pair = sorted(self.h2h.get(frozenset((home, away)), []), reverse=True)[:H2H_MAX]
        h2h_used = len(pair)
        base_total = lam_home + lam_away
        if h2h_used >= 2:
            h2h_avg = sum(tg for _, tg in pair) / h2h_used
            blended_total = 0.72 * base_total + 0.28 * h2h_avg
            basis += "+h2h"
        else:
            blended_total = base_total
        blended_total = max(0.30, min(6.0, blended_total))
        if base_total > 1e-6:
            scale = blended_total / base_total
            lam_home *= scale
            lam_away *= scale
        else:
            lam_home = lam_away = blended_total / 2

        # basis'e eklenmez: arayüz basis'i birebir karşılaştırıyor ('form+h2h')
        market_used = market_p25 is not None and 0.02 < market_p25 < 0.98
        if market_used:
            total = lam_home + lam_away
            target = (1 - MARKET_WEIGHT) * total + MARKET_WEIGHT * market_total(lam_home, lam_away, self.rho, market_p25)
            lam_home *= target / total
            lam_away *= target / total

        pred = self.predict_from_lambdas(lam_home, lam_away, basis, h2h_used)
        pred["market_used"] = market_used
        return pred
