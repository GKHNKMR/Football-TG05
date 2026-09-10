"""Shared team-name crosswalk: BETAVUS display names <-> football-data.co.uk.

BETAVUS fixtures come from openfootball (long canonical names, e.g. "Manchester
United"); football-data.co.uk CSVs use short names ("Man United"). This maps
between them for every league BETAVUS covers.
"""

DIVISIONS = {
    "E0": ("Premier League", 39),
    "E1": ("Championship", 40),
    "SP1": ("LaLiga", 140),
    "D1": ("Bundesliga", 78),
    "I1": ("Serie A", 135),
    "F1": ("Ligue 1", 61),
    "N1": ("Eredivisie", 88),
    "T1": ("Turkish Süper Lig", 203),
}

LEAGUE_BY_DIV = {d: name for d, (name, _lid) in DIVISIONS.items()}
DIV_BY_LEAGUE = {name: d for d, name in LEAGUE_BY_DIV.items()}

# BETAVUS display name -> football-data.co.uk name. Only the ones that differ;
# identical names fall through untouched.
CROSSWALK = {
    "Premier League": {
        "Brighton & Hove Albion": "Brighton", "Coventry City": "Coventry",
        "Hull City": "Hull", "Ipswich Town": "Ipswich", "Leeds United": "Leeds",
        "Manchester City": "Man City", "Manchester United": "Man United",
        "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
        "Tottenham Hotspur": "Tottenham",
    },
    "LaLiga": {
        "Athletic Bilbao": "Ath Bilbao", "Atlético Madrid": "Ath Madrid",
        "Alavés": "Alaves", "Celta Vigo": "Celta", "Deportivo La Coruña": "La Coruna",
        "Espanyol": "Espanol", "Málaga": "Malaga", "Racing Santander": "Santander",
        "Rayo Vallecano": "Vallecano", "Real Betis": "Betis", "Real Sociedad": "Sociedad",
        "Osasuna": "Osasuna",
    },
    "Bundesliga": {
        "Bayern München": "Bayern Munich", "Mönchengladbach": "M'gladbach",
        "Borussia Dortmund": "Dortmund", "Eintracht Frankfurt": "Ein Frankfurt",
        "Köln": "FC Koln", "Hamburger SV": "Hamburg", "Mainz 05": "Mainz",
    },
    "Championship": {
        "Birmingham City": "Birmingham", "Blackburn Rovers": "Blackburn",
        "Bolton Wanderers": "Bolton", "Cardiff City": "Cardiff",
        "Charlton Athletic": "Charlton", "Coventry City": "Coventry",
        "Derby County": "Derby", "Huddersfield Town": "Huddersfield",
        "Hull City": "Hull", "Ipswich Town": "Ipswich", "Leeds United": "Leeds",
        "Leicester City": "Leicester", "Lincoln City": "Lincoln", "Luton Town": "Luton",
        "Norwich City": "Norwich", "Nottingham Forest": "Nott'm Forest",
        "Oxford United": "Oxford", "Peterborough United": "Peterboro",
        "Plymouth Argyle": "Plymouth", "Preston North End": "Preston",
        "Queens Park Rangers": "QPR", "Rotherham United": "Rotherham",
        "Sheffield Wednesday": "Sheffield Weds", "Stoke City": "Stoke",
        "Swansea City": "Swansea", "West Bromwich Albion": "West Brom",
        "West Ham United": "West Ham", "Wigan Athletic": "Wigan",
        "Wolverhampton Wanderers": "Wolves",
    },
    "Serie A": {},
    "Ligue 1": {"Paris Saint-Germain": "Paris SG"},
    "Eredivisie": {
        "ADO Den Haag": "Den Haag", "AZ": "AZ Alkmaar", "Fortuna Sittard": "For Sittard",
        "NEC": "Nijmegen", "PEC Zwolle": "Zwolle", "PSV": "PSV Eindhoven",
    },
    # football-data.co.uk uses ASCII short forms for Turkish clubs; restore the
    # names Turkish readers expect.
    "Turkish Süper Lig": {
        "Adana Demirspor": "Ad. Demirspor", "Başakşehir": "Buyuksehyr",
        "Beşiktaş": "Besiktas", "Fenerbahçe": "Fenerbahce", "Gaziantep FK": "Gaziantep",
        "Gençlerbirliği": "Genclerbirligi", "Göztepe": "Goztep",
        "Fatih Karagümrük": "Karagumruk", "Kasımpaşa": "Kasimpasa",
        "Ümraniyespor": "Umraniyespor", "İstanbulspor": "Istanbulspor",
    },
}

# reverse: football-data name -> BETAVUS display name, per league
PRETTY = {
    league: {fd: bv for bv, fd in cw.items()} for league, cw in CROSSWALK.items()
}


def to_fd(league, name):
    """BETAVUS display name -> football-data.co.uk name for that league."""
    return CROSSWALK.get(league, {}).get(name, name)


def to_pretty(league, fd_name):
    """football-data.co.uk name -> BETAVUS display name for that league."""
    return PRETTY.get(league, {}).get(fd_name, fd_name)
