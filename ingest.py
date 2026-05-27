"""Run the full ingestion pipeline: dataset JSONs -> chunks -> embeddings -> Milvus."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from ingestion.cache import EmbeddingCache
from ingestion.chunk import extract_chunks
from ingestion.embed import DEFAULT_BASE_URL, DEFAULT_MODEL, DoubaoEmbedder
from ingestion.milvus_store import MilvusStore

DATA_DIR = Path("data")
DATASET_ROOT = Path("ecommerce_agent_dataset")
MILVUS_PATH = DATA_DIR / "milvus.db"
CACHE_PATH = DATA_DIR / "embedding_cache.jsonl"

# Documented dim for doubao-embedding-vision-251215. If Doubao ever changes
# it, Milvus rejects the insert with a dim mismatch. Update this constant
# and delete data/milvus.db to rebuild the collection.
EMBEDDING_DIM = 2048


def load_products(limit: int | None = None) -> list[dict]:
    files = sorted(DATASET_ROOT.glob("*/data/*.json"))
    if limit is not None:
        files = files[:limit]
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                       help="only ingest the first N products (for smoke testing)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("ARK_EMBEDDING_API_KEY")
    if not api_key:
        print("ERROR: ARK_EMBEDDING_API_KEY not set in environment or .env", file=sys.stderr)
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(CACHE_PATH)
    embedder = DoubaoEmbedder(
        api_key=api_key,
        cache=cache,
        dataset_root=DATASET_ROOT,
        base_url=os.environ.get("ARK_EMBEDDING_BASE_URL", DEFAULT_BASE_URL),
        model=os.environ.get("ARK_EMBEDDING_MODEL", DEFAULT_MODEL),
    )
    store = MilvusStore(uri=str(MILVUS_PATH), dim=EMBEDDING_DIM)
    store.ensure_collection()

    products = load_products(limit=args.limit)
    for product in products:
        chunks = extract_chunks(product)
        vectors = embedder.embed_chunks(chunks)
        store.upsert(chunks, vectors)

    print(f"Milvus collection row count: {store.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
