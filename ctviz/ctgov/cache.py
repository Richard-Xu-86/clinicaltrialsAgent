"""A deliberately simple on-disk cache for API responses, keyed by full URL.

One JSON file per URL. That is plenty for a single-process service and makes the
cache inspectable by hand; the recorded responses shipped with the example runs
are just a copy of this directory.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


class ResponseCache:
    def __init__(self, directory: Path, ttl_s: int | None):
        self.directory = Path(directory)
        self.ttl_s = ttl_s  # None = entries never expire (used for replaying recordings)

    def _path(self, url: str) -> Path:
        return self.directory / f"{cache_key(url)}.json"

    def get(self, url: str) -> Any | None:
        path = self._path(url)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None  # a corrupt entry is just a miss
        if self.ttl_s is not None and time.time() - entry.get("stored_at", 0) > self.ttl_s:
            return None
        if entry.get("url") != url:  # hash collision guard; practically never happens
            return None
        return entry["body"]

    def put(self, url: str, body: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = {"url": url, "stored_at": time.time(), "body": body}
        tmp = self._path(url).with_suffix(".tmp")
        tmp.write_text(json.dumps(entry), encoding="utf-8")
        tmp.replace(self._path(url))  # atomic on POSIX, so readers never see half a file
