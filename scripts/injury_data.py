"""Current injuries/suspensions per club, from Transfermarkt's "Sperren und
Verletzungen" (bans and injuries) page - no official API, but the page is
server-rendered (no JS needed) and, unlike its slug, routes purely off the
numeric club id: /any-text-here/sperrenundverletzungen/verein/{id} works
regardless of what "any-text-here" is (verified empirically), so the only
per-club fact needed is that numeric id.

That id comes from Wikidata's P7223 ("Transfermarkt team ID") on the same
club entity scripts/stadium_geo.py already resolves by name for stadium
coordinates - reuses its club_qid() cache instead of re-solving the
name-matching problem a third time.

Unlike everything else cached under data/cache/ in this project, an
injury list is NOT a fixed historical fact - it changes daily, and this
runs as part of the live hourly pipeline (not a one-off backtest), so only
the club identity (Wikidata qid -> Transfermarkt id) is cached permanently;
the actual injuries/suspensions page is always fetched fresh.

Standard library only.
"""

import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stadium_geo import club_qid, _get_json, _safe_filename  # noqa: E402

CACHE_DIR = Path("data/cache/wikidata_stadiums")  # shares stadium_geo's qid cache dir
TM_URL = "https://www.transfermarkt.com/club/sperrenundverletzungen/verein/{tm_id}"
# Transfermarkt blocks a bare urllib default UA on some paths - a plain
# browser UA gets a normal 200 (verified).
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

#  each player's row nests a whole sub-table (photo + name + position) inside
#  its first <td>, so the name link is followed by several nested </td>
#  closes before the real "age/reason/since/return" cells - .*? (DOTALL)
#  just skips past all of that to the specific 4-cell sequence that follows.
_ROW_RE = re.compile(
    r'<a title="([^"]+)" href="/[^"]+/profil/spieler/\d+">.*?'
    r'<td class="zentriert">\d*</td>'
    r'<td class="[^"]*">([^<]+)</td>'
    r'<td class="zentriert">([\d/]*)</td>'
    r'<td class="zentriert">([\d/]*|[^<]*)</td>',
    re.S,
)
_SECTION_RE = re.compile(r'<td class="extrarow[^"]*"[^>]*>([^<]+)</td>')


def transfermarkt_id(league, fd_name):
    """This club's Wikidata-resolved Transfermarkt team id, or None. Cached
    permanently (like club_qid()) - a club's Transfermarkt id is a fixed
    identity fact, not something that needs re-querying every hourly run."""
    cache_path = CACHE_DIR / f"tmid_{_safe_filename(league, fd_name)}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8")) or None
        except json.JSONDecodeError:
            pass

    qid = club_qid(league, fd_name)
    if not qid:
        return None  # club_qid() already distinguishes "no club" from "lookup failed"

    query = f'SELECT ?id WHERE {{ wd:{qid} wdt:P7223 ?id. }} LIMIT 1'
    url = f"https://query.wikidata.org/sparql?query={query.replace(' ', '%20').replace('{', '%7B').replace('}', '%7D')}&format=json"
    try:
        data = _get_json(url)
    except Exception as exc:
        # a transient failure - do NOT cache, so the next hourly run retries
        # instead of a Wikidata 502 permanently reading as "no Transfermarkt id"
        print(f"  Transfermarkt id lookup failed [{qid}]: {exc}")
        return None
    rows = data["results"]["bindings"]
    tm_id = rows[0]["id"]["value"] if rows else None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(tm_id), encoding="utf-8")
    return tm_id


def _fetch_page(tm_id):
    req = Request(TM_URL.format(tm_id=tm_id), headers={"User-Agent": _UA})
    with urlopen(req, timeout=25) as resp:
        return resp.read().decode("utf-8", errors="replace")


_TABLE_TAG_RE = re.compile(r"<table\b|</table>")


def _table_extent(html, idx):
    """Length of the balanced <table>...</table> starting at idx - the
    player rows nest their own <table class="inline-table">, so a plain
    'first </table>' search closes on that inner table instead."""
    depth = 0
    for m in _TABLE_TAG_RE.finditer(html, idx):
        depth += 1 if m.group().startswith("<table") else -1
        if depth == 0:
            return m.end() - idx
    return 20000  # unbalanced/truncated page - fall back to a generous slice


def fetch_absences(tm_id):
    """[{"player", "category" ("Injuries"/"Suspensions"/...), "reason",
    "since", "expected_return"}, ...] currently listed for this club - []
    if the page has nothing (the common case) or can't be fetched."""
    try:
        html = _fetch_page(tm_id)
    except (HTTPError, URLError) as exc:
        print(f"  Transfermarkt page fetch failed [{tm_id}]: {exc}")
        return []

    idx = html.find('<table class="items"')
    if idx < 0:
        return []
    table_html = html[idx:idx + _table_extent(html, idx)]

    # player rows nest a sub-table inside their own <tr>, so a naive
    # "split into <tr>...</tr> chunks" walk grabs the wrong (inner) tag
    # pair - instead, find section headers and data rows by position in
    # the flat HTML and assign each row to whichever section precedes it.
    sections = [(m.start(), m.group(1).strip()) for m in _SECTION_RE.finditer(table_html)]

    def category_at(pos):
        cat = "Injuries"
        for start, label in sections:
            if start > pos:
                break
            cat = label
        return cat

    out = []
    for m in _ROW_RE.finditer(table_html):
        out.append({
            "player": m.group(1).strip(), "category": category_at(m.start()),
            "reason": m.group(2).strip(), "since": m.group(3).strip(),
            "expected_return": m.group(4).strip(),
        })
    return out


def absences_for_league(league, fd_names):
    """{fd_name: [absence dicts]} for every resolvable club with at least
    one current injury/suspension listed."""
    out = {}
    for fd_name in fd_names:
        tm_id = transfermarkt_id(league, fd_name)
        if not tm_id:
            continue
        absences = fetch_absences(tm_id)
        if absences:
            out[fd_name] = absences
    return out
