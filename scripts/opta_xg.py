"""Shared helper: pull historical + current-season team xG for BETAVUS's 9
leagues from peteowen1/pannadata's public Opta mirror on GitHub (no key,
updated by that project most days), for blending into the goal model.

The upstream release ships a giant player-match file
(opta_xmetrics_bymatch.parquet, ~300MB: one row per player per match) plus a
small fixtures file (opta_fixtures.parquet: match_id -> date/competition/
home+away team+id). We never download either in full - DuckDB's httpfs
extension queries them straight off GitHub's release CDN with column/filter
pushdown, so pulling just our 9 leagues takes a few seconds and a few MB.

Team names: Opta uses long official names ("Manchester City FC",
"Reial Club Deportiu Espanyol de Barcelona") that don't match football-data
.co.uk's short ones ("Man City", "Espanol") - same kind of gap
scripts/live_scores.py already bridges for ESPN, solved the same way here
(strip common club-type words, then a small hand-built table for the
stragglers a plain word-strip can't reach). Verified empirically: 0
unmatched, 0 name collisions across all 9 leagues for every season our own
football-data.co.uk cache covers (see git history for the audit script).
"""

import re
import unicodedata

import duckdb

XMETRICS_URL = "https://github.com/peteowen1/pannadata/releases/download/opta-latest/opta_xmetrics_bymatch.parquet"
FIXTURES_URL = "https://github.com/peteowen1/pannadata/releases/download/opta-latest/opta_fixtures.parquet"

# BETAVUS division code -> Opta's own competition code
DIV_TO_OPTA = {
    "E0": "EPL", "E1": "Championship", "SP1": "La_Liga", "D1": "Bundesliga",
    "I1": "Serie_A", "F1": "Ligue_1", "N1": "Eredivisie", "T1": "Super_Lig",
    "P1": "Primeira_Liga",
}
OPTA_TO_DIV = {v: k for k, v in DIV_TO_OPTA.items()}

STOP = {
    "FC", "AFC", "CF", "SK", "BK", "AS", "SS", "SC", "US", "CD", "RC", "CA", "SD", "UD",
    "AC", "CS", "AA", "EC", "CP", "SL", "GD", "SV", "TSV", "VFL", "VFB", "RB", "FSV",
    "FK", "AND", "THE", "CLUB", "CALCIO", "CLUBE", "FUTEBOL", "FUTBOL", "FOOTBALL",
    "SPORTING", "SPOR", "ESPORTE", "ESPORTIVO", "UNIAO", "KULUBU", "KULUB",
    "DEPORTIVO", "ASSOCIATION",
}

# same idiom as scripts/live_scores.py's ESPN matching - short/colloquial
# words that don't line up with football-data.co.uk's own short names.
WORD_ALIASES = {
    "WOLVES": "WOLVERHAMPTON", "SPURS": "TOTTENHAM", "BORO": "MIDDLESBROUGH",
    "NIJMEGEN": "NEC", "COLOGNE": "KOLN", "HAMBURGER": "HAMBURG", "MUNICH": "MUNCHEN",
    "UTD": "UNITED",
}

# hand-verified stragglers a word-strip can't bridge (wildly different naming
# style, not just abbreviation/suffix) - keyed by norm(opta_team_name) ->
# the exact football-data.co.uk short name.
ALIASES = {
    "MANCHESTER CITY": "Man City", "MANCHESTER UNITED": "Man United",
    "NOTTINGHAM FOREST": "Nott'm Forest", "QUEENS PARK RANGERS": "QPR",
    "SHEFFIELD WEDNESDAY": "Sheffield Weds",
    "ATHLETIC": "Ath Bilbao", "ATLETICO DE MADRID": "Ath Madrid",
    "DE A CORUNA": "La Coruna", "REAL RACING": "Santander",
    "BORUSSIA MONCHENGLADBACH": "M'gladbach",
    "DUSSELDORFER TUS FORTUNA": "Fortuna Dusseldorf",
    "EINTRACHT FRANKFURT": "Ein Frankfurt", "KIELER HOLSTEIN": "Holstein Kiel",
    "ARS ET LABOR FERRARA": "Spal", "INTERNAZIONALE MILANO": "Inter",
    "SAINT ETIENNE": "St Etienne", "JEUNESSE AUXERROISE": "Auxerre",
    "ALKMAAR ZAANSTREEK": "AZ Alkmaar", "FORTUNA SITTARD": "For Sittard",
    "ADANA DEMIR": "Ad. Demirspor", "CAYKUR RIZE": "Rizespor",
    "ISTANBUL BASAKSEHIR": "Buyuksehyr",
    "ACADEMICO DE VISEU": "Academico Viseu", "PACOS DE FERREIRA": "Pacos Ferreira",
    "DE PORTUGAL": "Sp Lisbon",
    "STADE RENNAIS": "Rennes", "AMED SFK": "Amedspor", "ERZURUM BB": "Erzurumspor",
    # a longer official name that happens to *contain* a different, shorter
    # club's fd short name - the substring fallback below would misfire.
    "REIAL DEPORTIU ESPANYOL DE BARCELONA": "Espanol",   # not "Barcelona"
    "PARIS SAINT GERMAIN": "Paris SG",                    # not "Paris FC"
}


def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z0-9]+", " ", s.upper())
    toks = [t for t in s.split() if t and not t.isdigit() and t not in STOP]
    # applied to BOTH sides (fd names and Opta names alike, via build_matcher
    # below) so e.g. fd's short "Wolves" folds up to "WOLVERHAMPTON", which
    # then lines up as a prefix of Opta's "WOLVERHAMPTON WANDERERS" through
    # the substring fallback in build_matcher, instead of needing an exact
    # match neither side alone would produce.
    toks = [WORD_ALIASES.get(t, t) for t in toks]
    return " ".join(toks) or s.strip()


def build_matcher(fd_team_names):
    """fd_team_names: iterable of a division's football-data.co.uk names.
    Returns opta_team_name -> fd_name (or None if it can't be resolved)."""
    fd_norm = {}
    for n in fd_team_names:
        fd_norm.setdefault(norm(n), n)

    def match(opta_name):
        n = norm(opta_name)
        hit = fd_norm.get(n) or ALIASES.get(n)
        if hit:
            return hit
        cands = [fdn for nn, fdn in fd_norm.items() if nn and (nn in n or n in nn)]
        return cands[0] if len(cands) == 1 else None

    return match


def fetch_raw_matches(divisions=None):
    """Query the remote Opta parquet files for our leagues.

    divisions: iterable of BETAVUS division codes (default: all 9). Returns a
    list of dicts: div, opta_competition, season (e.g. "2025-2026"), date
    (ISO), home_opta, away_opta (Opta's own team names - NOT yet resolved to
    football-data.co.uk names), home_xg, away_xg.
    """
    comps = [DIV_TO_OPTA[d] for d in (divisions or DIV_TO_OPTA)]
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    placeholders = ", ".join(f"'{c}'" for c in comps)
    q = f"""
      WITH team_match_xg AS (
        SELECT match_id, team_id, any_value(team_name) AS team_name,
               any_value(competition) AS competition, any_value(season) AS season,
               sum(xg) AS team_xg
        FROM read_parquet('{XMETRICS_URL}')
        WHERE competition IN ({placeholders})
        GROUP BY match_id, team_id
      )
      SELECT f.match_id, f.match_date, tmx.competition, tmx.season,
             f.home_team, f.away_team, f.home_team_id, f.away_team_id,
             h.team_xg AS home_xg, a.team_xg AS away_xg
      FROM team_match_xg tmx
      JOIN read_parquet('{FIXTURES_URL}') f ON f.match_id = tmx.match_id
      JOIN team_match_xg h ON h.match_id = f.match_id AND h.team_id = f.home_team_id
      JOIN team_match_xg a ON a.match_id = f.match_id AND a.team_id = f.away_team_id
      WHERE tmx.team_id = f.home_team_id
    """
    rows = con.execute(q).fetchall()
    cols = [d[0] for d in con.description]
    out = []
    for row in rows:
        r = dict(zip(cols, row))
        div = OPTA_TO_DIV.get(r["competition"])
        if not div:
            continue
        out.append({
            "div": div,
            "season": r["season"],
            "date": r["match_date"].rstrip("Z"),
            "home_opta": r["home_team"], "away_opta": r["away_team"],
            "home_xg": r["home_xg"], "away_xg": r["away_xg"],
        })
    return out


def resolve_team_names(raw_matches, fd_team_names_by_div):
    """raw_matches: fetch_raw_matches() output. fd_team_names_by_div: dict
    div -> iterable of that division's football-data.co.uk team names.

    Returns (resolved, unmatched): resolved matches get "home"/"away" (the
    football-data.co.uk name) added; unmatched rows (a name the aliasing
    couldn't bridge - expected for seasons our own football-data.co.uk cache
    doesn't cover) are dropped from resolved and returned separately so a
    caller can log them without failing the run.
    """
    matchers = {div: build_matcher(names) for div, names in fd_team_names_by_div.items()}
    resolved, unmatched = [], []
    for m in raw_matches:
        matcher = matchers.get(m["div"])
        home = matcher(m["home_opta"]) if matcher else None
        away = matcher(m["away_opta"]) if matcher else None
        if home and away:
            resolved.append({**m, "home": home, "away": away})
        else:
            unmatched.append(m)
    return resolved, unmatched
