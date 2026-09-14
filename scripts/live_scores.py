"""Shared helpers for the ESPN-scoreboard-based live-score layer.

data/live-scores.json is produced once per run by fetch_live_scores.py (a
flat list of matches in our leagues from the last LOOKBACK_DAYS, whichever
have kicked off) and consumed by:

  - update_predictions.py: drop a fixture from Tahminler once it's actually
    finished (openfootball/football-data can take days to record a score,
    so without this a played match keeps showing as "upcoming"), and attach
    a live in-progress score onto ones still being played.
  - build_results.py: grade an archived (real pre-kickoff) prediction before
    football-data.co.uk's CSV has caught up with that result.

Both only need read access to the file - fetching happens once, up front,
in its own workflow step.
"""

import json
import re
import unicodedata
from datetime import timedelta
from pathlib import Path

LIVE_FILE = Path("data/live-scores.json")


def load_live_scores():
    if not LIVE_FILE.exists():
        return []
    try:
        return json.loads(LIVE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


# ESPN uses its own short/nickname/official forms that don't line up with
# either openfootball's long names or football-data's short names (e.g. it
# calls Sheffield United "Sheffield Utd", Wolverhampton Wanderers "Wolves",
# 1. FC Köln "FC Cologne", Hamburger SV "Hamburg SV") - fold the common ones
# to a shared word so substring matching below actually lines them up.
_WORD_ALIASES = {
    "UTD": "UNITED", "WOLVES": "WOLVERHAMPTON", "SPURS": "TOTTENHAM",
    "BORO": "MIDDLESBROUGH", "NIJMEGEN": "NEC", "COLOGNE": "KOLN",
    "HAMBURGER": "HAMBURG", "MUNICH": "MUNCHEN",
}
# A few club names ESPN spells so differently from ours that no amount of
# per-word aliasing lines them up (a rebrand, a different short form, or a
# translation) - map the whole cleaned name instead.
_NAME_ALIASES = {
    "STADE RENNAIS": "RENNES", "ATHLETIC CLUB": "ATHLETIC BILBAO",
    "ERZURUM BB": "ERZURUMSPOR", "AMED SFK": "AMEDSPOR",
}


def norm(s):
    """Accent/case/punctuation-insensitive form for fuzzy team-name matching."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z0-9]+", " ", s.upper())
    s = re.sub(r"\s+", " ", s).strip()
    if s in _NAME_ALIASES:
        s = _NAME_ALIASES[s]
    words = [_WORD_ALIASES.get(w, w) for w in s.split(" ")]
    return " ".join(words)


def find_live_match(live, league, home, away, d, slack=2):
    """d: a date. Returns the live-scores.json entry for this fixture, or None."""
    h_n, a_n = norm(home), norm(away)
    if not h_n or not a_n:
        return None
    for delta in range(-slack, slack + 1):
        target = (d + timedelta(days=delta)).isoformat()
        for item in live:
            if item["league"] != league or item["date"] != target:
                continue
            lh, la = norm(item["home"]), norm(item["away"])
            if (lh in h_n or h_n in lh) and (la in a_n or a_n in la):
                return item
    return None
