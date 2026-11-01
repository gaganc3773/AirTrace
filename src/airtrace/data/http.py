"""Small JSON-over-HTTP helper with a gzip file cache, so every fetched response is replayable offline."""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "AirTrace/0.1 (academic air-quality prototype)"


def read_cache(path: Path) -> dict | list | None:
    """Cached JSON at ``path`` (gzip), or ``None``."""
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def write_cache(path: Path, obj: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh)


def get_json(url: str, params: dict, headers: dict | None = None, retries: int = 4, timeout: float = 60.0) -> dict | list:
    """GET ``url?params`` and decode JSON, backing off on HTTP 429 / 5xx and network errors."""
    query = urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(f"{url}?{query}" if query else url,
                                 headers={"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code not in (429, 500, 502, 503, 504):
                body = err.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(f"HTTP {err.code} from {url}: {body}") from err
            reset = err.headers.get("x-ratelimit-reset") if err.headers else None
            wait = float(reset) if reset and reset.replace(".", "", 1).isdigit() and float(reset) < 120 else 10 * (attempt + 1)
            last = err
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as err:
            last = err
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts") from last
