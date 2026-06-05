"""Doubao multimodal embedding client with on-disk cache and simple retry."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Optional

import requests

from ingestion.cache import EmbeddingCache, image_key, text_key
from ingestion.chunk import Chunk

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "doubao-embedding-vision-251215"
MAX_ATTEMPTS = 3


class DoubaoEmbedder:
    def __init__(
        self,
        api_key: str,
        cache: EmbeddingCache,
        dataset_root: Optional[Path] = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        retry_sleep: float = 2.0,
        timeout: float = 60.0,
        max_attempts: int = MAX_ATTEMPTS,
    ):
        self._api_key = api_key
        self._cache = cache
        self._dataset_root = Path(dataset_root) if dataset_root else Path(".")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._retry_sleep = retry_sleep
        self._timeout = timeout
        self._max_attempts = max_attempts

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Return one embedding vector per chunk, in the same order as input."""
        # The Doubao multimodal endpoint fuses all inputs in a single call into
        # one joint embedding, so we cannot batch independent chunks. One API
        # call per chunk is the only correct shape.
        results: list[list[float]] = []
        for chunk in chunks:
            key, item = self._build_input(chunk)
            results.append(self._embed_item(key, item))
        return results

    def embed_text(self, text: str) -> list[float]:
        """Embed a standalone text query with the same model/cache as chunks."""
        key = text_key(text)
        return self._embed_item(key, {"type": "text", "text": text})

    def _build_input(self, chunk: Chunk) -> tuple[str, dict]:
        """Return (cache_key, api_input_item) for a single chunk."""
        if chunk.chunk_type == "image":
            assert chunk.image_path, f"image chunk {chunk.chunk_id} missing image_path"
            img_path = self._dataset_root / chunk.image_path
            img_bytes = img_path.read_bytes()
            key = image_key(img_bytes)
            b64 = base64.b64encode(img_bytes).decode("ascii")
            return key, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        else:
            key = text_key(chunk.text)
            return key, {"type": "text", "text": chunk.text}

    def _embed_item(self, key: str, item: dict) -> list[float]:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        vec = self._call_api(item)
        self._cache.put(key, vec)
        return vec

    def _call_api(self, item: dict) -> list[float]:
        payload = {"model": self._model, "input": [item]}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self._base_url}/embeddings/multimodal"

        last_err: Optional[Exception] = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=self._timeout)
            except requests.RequestException as e:
                last_err = e
                if attempt < self._max_attempts:
                    time.sleep(self._retry_sleep)
                continue

            if resp.status_code == 200:
                return resp.json()["data"]["embedding"]
            if 400 <= resp.status_code < 500:
                raise RuntimeError(
                    f"Doubao API hard failure {resp.status_code}: {resp.text[:300]}"
                )
            last_err = RuntimeError(f"{resp.status_code} {resp.text[:200]}")
            if attempt < self._max_attempts:
                time.sleep(self._retry_sleep)

        raise RuntimeError(f"Doubao API failed after {self._max_attempts} attempts: {last_err}")
