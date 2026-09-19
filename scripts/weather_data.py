"""Historical daily weather (precipitation, wind) near a stadium, for
scripts/tune_weather.py - the "Hava Sartlari" third of improvement item D.

Open-Meteo's historical archive (archive-api.open-meteo.com) is the usual
free/keyless choice here, but is unreachable from this network (times out
on every attempt - looks like a corporate firewall block on that specific
subdomain, confirmed by api.open-meteo.com's OTHER endpoints working fine).
Meteostat's bulk data mirror (bulk.meteostat.net, no key, CC BY-NC 4.0) is
reachable and used instead: a global station list plus one gzipped hourly
CSV per station, both plain static-file downloads (not the flaky/blocked
"archive-api" pattern).

Method: nearest weather station to each stadium (great-circle distance
over Meteostat's ~16k-station list, no external geo library needed), then
that station's full hourly history collapsed to one row per day
(precipitation total, max wind speed) - a day-level signal side-steps
having to line up the CSV's local station time against football-data.co.uk's
own local kickoff time, which the data doesn't cleanly support anyway.

Cached to disk (data/cache/meteostat/) - a fixed historical record, a
second run costs zero new requests except for stations not seen before.

Standard library only.
"""

import csv
import gzip
import io
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen

CACHE_DIR = Path("data/cache/meteostat")
STATIONS_URL = "https://bulk.meteostat.net/v2/stations/lite.json.gz"
HOURLY_URL = "https://bulk.meteostat.net/v2/hourly/{station_id}.csv.gz"

# hourly CSV column order per Meteostat's documented bulk format
_COLS = ["date", "hour", "temp", "dwpt", "rhum", "prcp", "snow",
         "wdir", "wspd", "wpgt", "pres", "tsun", "coco"]


def _fetch_gz_bytes(url):
    req = Request(url, headers={"User-Agent": "betavus-research/1.0 (one-off analysis script)"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


_stations_cache = None


def _load_stations(min_hourly_start="2020-01-01"):
    """Stations with usable hourly coverage for our backtest window -
    [{"id", "lat", "lon"}] - cached in-process and on disk."""
    global _stations_cache
    if _stations_cache is not None:
        return _stations_cache
    cache_path = CACHE_DIR / "stations.json"
    if cache_path.exists():
        _stations_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        return _stations_cache

    raw = _fetch_gz_bytes(STATIONS_URL)
    stations = json.loads(gzip.decompress(raw).decode("utf-8"))
    out = []
    for s in stations:
        hourly = (s.get("inventory") or {}).get("hourly") or {}
        start = hourly.get("start")
        loc = s.get("location") or {}
        if not start or start > min_hourly_start or "latitude" not in loc:
            continue
        out.append({"id": s["id"], "lat": loc["latitude"], "lon": loc["longitude"]})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out), encoding="utf-8")
    _stations_cache = out
    return out


def nearest_station_id(lat, lon):
    stations = _load_stations()
    if not stations:
        return None
    best, best_d = None, None
    for s in stations:
        d = _haversine_km(lat, lon, s["lat"], s["lon"])
        if best_d is None or d < best_d:
            best, best_d = s["id"], d
    return best


def station_daily_weather(station_id):
    """{iso_date: (precip_mm_total, max_wind_kmh)} for one station's whole
    history - the hourly CSV collapsed to daily aggregates once, cached, so
    a match lookup is then just a dict get."""
    cache_path = CACHE_DIR / f"daily_{station_id}.json"
    if cache_path.exists():
        return {k: tuple(v) for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()}

    try:
        raw = _fetch_gz_bytes(HOURLY_URL.format(station_id=station_id))
    except Exception as exc:
        print(f"  hourly fetch failed [{station_id}]: {exc}")
        return {}
    text = gzip.decompress(raw).decode("utf-8")

    by_date = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 9:
            continue
        date = row[0]
        prcp = row[5].strip()
        wspd = row[8].strip()
        p = float(prcp) if prcp else 0.0
        w = float(wspd) if wspd else 0.0
        entry = by_date.setdefault(date, [0.0, 0.0])
        entry[0] += p
        entry[1] = max(entry[1], w)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(by_date), encoding="utf-8")
    return {k: tuple(v) for k, v in by_date.items()}


def daily_weather_for_coord(lat, lon, date_iso):
    """(precip_mm, max_wind_kmh) for the nearest station to (lat, lon) on
    that date, or None if unavailable."""
    station_id = nearest_station_id(lat, lon)
    if not station_id:
        return None
    return station_daily_weather(station_id).get(date_iso)
