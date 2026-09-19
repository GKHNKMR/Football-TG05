"""Shared name-matching helpers for cross-referencing data/key-players.json
(Opta's own player names) against a name list from a different provider -
ESPN's starting-XI roster (scripts/fetch_lineups.py) or Transfermarkt's
injury/suspension list (scripts/fetch_injuries.py). Nothing standardizes
player-name spelling between these three sources, so this compares
normalized SURNAMES only, scoped to one team's own key-player list - not a
global player database. Treat a match as a best-effort signal, not a
certainty.
"""

import re
import unicodedata

_JR_ALIASES = {"JR", "JUNIOR", "JNR"}


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z]+", " ", s.upper()).split()


def surname_key(name):
    toks = fold(name)
    if not toks:
        return ""
    tok = toks[-1]
    return "JUNIOR" if tok in _JR_ALIASES else tok


def name_tokens(full_name):
    """All normalized tokens in a name (JR-aliased), so a key player's
    surname just needs to appear somewhere in it."""
    return {("JUNIOR" if t in _JR_ALIASES else t) for t in fold(full_name)}


def key_players_missing_from(key_players, present_tokens):
    """key_players: data/key-players.json[fd_team] list. present_tokens: a
    POSITIVE set of name-tokens (e.g. the actual starting XI) - a key
    player counts as missing if their surname isn't in it. Returns
    (missing_names, missing_share)."""
    missing = [kp for kp in key_players if surname_key(kp["player"]) not in present_tokens]
    share = sum(kp["share"] for kp in missing)
    return [kp["player"] for kp in missing], share


def key_players_listed_in(key_players, absent_names):
    """key_players: data/key-players.json[fd_team] list. absent_names: a
    NEGATIVE list of full names (e.g. an injury/suspension report) - a key
    player counts as absent if their surname appears in it. Returns
    (absent_names_matched, absent_share)."""
    absent_tokens = set()
    for n in absent_names:
        absent_tokens |= name_tokens(n)
    absent = [kp for kp in key_players if surname_key(kp["player"]) in absent_tokens]
    return [kp["player"] for kp in absent], sum(kp["share"] for kp in absent)
