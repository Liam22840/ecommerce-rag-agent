"""Exact-match answer cache for high-frequency repeated chat questions.

Keyed on the normalised message + top_k, so an identical context-free product question
returns the stored answer without re-running retrieval or the LLM. Persisted as append-only
JSONL and loaded into memory at startup, mirroring ingestion.cache.EmbeddingCache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from server.textutil import normalize


class QueryCache:
    def __init__(self, path: Path, max_entries: int = 500, enabled: bool = True):
        self._path = Path(path)
        self._max_entries = max_entries
        self._enabled = enabled
        # Insertion order is recency order: oldest first, newest last.
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def key(message: str, top_k: int) -> str:
        return hashlib.sha256(f"{normalize(message)}|k={top_k}".encode("utf-8")).hexdigest()

    @staticmethod
    def eligible(compare_ids: list[str], recent_ids: list[str]) -> bool:
        """Only context-free turns are cacheable: identical words can mean different things
        once a conversation carries comparison or recently-shown-product context."""
        return not compare_ids and not recent_ids

    @staticmethod
    def storeable(intent_type: str) -> bool:
        # Only product-search answers are cached (chit-chat and comparison turns are
        # conversational / per-session). The cache holds whatever the current config
        # produces and is disposable, so we don't gate on the degraded flag.
        return intent_type == "product_search"

    def get(self, key: str) -> dict | None:
        if not self._enabled:
            return None
        return self._entries.get(key)

    def put(self, key: str, response: dict) -> None:
        if not self._enabled:
            return
        self._mark_newest(key, response)
        self._evict()
        self._append(key, response)

    def _mark_newest(self, key: str, response: dict) -> None:
        # Insertion order is recency order, so pop-then-reinsert moves this key to newest
        # (and applies last-write-wins).
        self._entries.pop(key, None)
        self._entries[key] = response

    def _evict(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.pop(next(iter(self._entries)), None)

    def _load(self) -> None:
        if not self._enabled or not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = entry.get("key")
                response = entry.get("response")
                if key and isinstance(response, dict):
                    self._mark_newest(key, response)
        self._evict()

    def _append(self, key: str, response: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
