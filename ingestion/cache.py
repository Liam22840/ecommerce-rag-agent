"""On-disk JSONL-backed cache for embedding vectors.

Append-only writes mean partial failures never corrupt previously cached entries.
The full cache is loaded into memory at construction time for O(1) lookups.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Optional


def text_key(text: str) -> str:
    return "text:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def image_key(image_bytes: bytes) -> str:
    return "image:" + hashlib.sha256(image_bytes).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path):
        self._path = Path(path)
        self._index: dict[str, list[float]] = {}
        # Guards _index and the append. The server embeds concurrently — the request thread,
        # speculative pre-warm threads, and FastAPI's threadpool all share one cache — so reads
        # and writes must be serialised to avoid a corrupt dict or an interleaved JSONL line.
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._index[entry["key"]] = entry["vector"]

    def get(self, key: str) -> Optional[list[float]]:
        with self._lock:
            return self._index.get(key)

    def put(self, key: str, vector: list[float]) -> None:
        with self._lock:
            self._index[key] = vector
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "vector": vector}) + "\n")
