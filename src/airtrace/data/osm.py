"""OpenStreetMap source-type geography via the Overpass API, cached for offline use.

Overpass rejects clients without an identifying User-Agent (HTTP 406), so one
is always sent. Responses are cached gzipped under ``artifacts/cache/osm``.
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from airtrace.data.geo import LocalProjection

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
USER_AGENT = "AirTrace/0.1 (academic air-quality prototype)"

# Road classes and their relative traffic-emission weight.
ROAD_WEIGHTS = {"motorway": 1.0, "trunk": 1.0, "primary": 0.7, "secondary": 0.4, "tertiary": 0.2}
LANDUSE_TAGS = ("industrial", "construction", "landfill", "farmland", "brownfield")


def _query(body: str, bbox: tuple[float, float, float, float]) -> str:
    s, w, n, e = bbox
    return f"[out:json][timeout:180][bbox:{s:.5f},{w:.5f},{n:.5f},{e:.5f}];({body});out geom;"


def fetch_overpass(name: str, body: str, bbox: tuple[float, float, float, float], cache_dir: Path) -> dict:
    """Run an Overpass query, using ``cache_dir/<name>.json.gz`` if present."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.json.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    data = urllib.parse.urlencode({"data": _query(body, bbox)}).encode()
    last_err: Exception | None = None
    for attempt, url in enumerate(OVERPASS_URLS * 2):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=240) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump(result, fh)
            return result
        except Exception as err:  # network errors, 429/504 from busy mirrors
            last_err = err
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Overpass query {name!r} failed on all mirrors") from last_err


def fetch_delhi_osm(proj: LocalProjection, cache_dir: Path) -> tuple[dict, dict]:
    """Major roads and source-relevant land use for the domain (plus a 2 km margin)."""
    bbox = proj.bbox(margin_m=2000.0)
    roads = fetch_overpass(
        "roads", f'way["highway"~"^({"|".join(ROAD_WEIGHTS)})$"];', bbox, cache_dir
    )
    landuse = fetch_overpass(
        "landuse",
        "".join(f'way["landuse"="{t}"];relation["landuse"="{t}"];' for t in LANDUSE_TAGS),
        bbox,
        cache_dir,
    )
    return roads, landuse


def _xy(proj: LocalProjection, geometry: list[dict]) -> np.ndarray:
    lon = np.array([p["lon"] for p in geometry])
    lat = np.array([p["lat"] for p in geometry])
    return np.column_stack(proj.to_xy(lon, lat))


def road_lines(roads: dict, proj: LocalProjection) -> tuple[list[np.ndarray], list[float]]:
    """Road polylines in metres and their emission weights."""
    lines, weights = [], []
    for el in roads["elements"]:
        if el.get("type") == "way" and "geometry" in el:
            lines.append(_xy(proj, el["geometry"]))
            weights.append(ROAD_WEIGHTS.get(el["tags"].get("highway", ""), 0.0))
    return lines, weights


def landuse_polygons(landuse: dict, proj: LocalProjection) -> dict[str, list[np.ndarray]]:
    """Polygons (outer rings, metres) for each land-use tag."""
    out: dict[str, list[np.ndarray]] = {t: [] for t in LANDUSE_TAGS}
    for el in landuse["elements"]:
        tag = el.get("tags", {}).get("landuse")
        if tag not in out:
            continue
        if el["type"] == "way" and "geometry" in el:
            out[tag].append(_xy(proj, el["geometry"]))
        elif el["type"] == "relation":
            for m in el.get("members", []):
                if m.get("role") == "outer" and "geometry" in m:
                    out[tag].append(_xy(proj, m["geometry"]))
    return out
