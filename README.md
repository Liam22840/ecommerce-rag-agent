# ecommerce-rag-agent

A multi-modal e-commerce intelligent shopping agent based on RAG.
基于 RAG 的多模态电商智能导购 Agent.

## What this repo contains (so far)

- `ecommerce_agent_dataset/`: 100 products across 4 categories, with text fields and one image per product.
- `ingestion/`: Python package with chunk extraction, the Doubao multimodal embedding client, and the Milvus Lite store.
- `ingest.py`: CLI to run the full pipeline.

## Setup

```bash
uv venv
uv pip install -r requirements.txt
cp .env.example .env
# Then edit .env and set ARK_API_KEY to your real Doubao key.
```

## Running the ingestion

Smoke run on 2 products (good for verifying the API + Milvus end-to-end before committing API credit to all 100):

```bash
.venv/bin/python ingest.py --limit 2
```

Full run on all 100 products:

```bash
.venv/bin/python ingest.py
```

Output lands at:
- `data/milvus.db`: populated Milvus Lite database
- `data/embedding_cache.jsonl`: embedding cache (re-runs are fast)

Both commands print `Milvus collection row count: N` when finished. That
line is the verification the DB was populated correctly.

## Querying from the backend

The backend reads `data/milvus.db` directly via `pymilvus`:

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="data/milvus.db")
results = client.search(
    collection_name="products",
    data=[query_vector],  # use the same Doubao multimodal endpoint to embed the query
    limit=5,
    output_fields=["chunk_id", "product_id", "chunk_type", "text",
                   "category", "sub_category", "brand", "base_price"],
)
```

For product card rendering, the backend reads the raw dataset JSONs:

```python
import json
from glob import glob

PRODUCTS = {}
for path in glob("ecommerce_agent_dataset/*/data/*.json"):
    p = json.load(open(path, encoding="utf-8"))
    PRODUCTS[p["product_id"]] = p
```

## Running the tests

```bash
.venv/bin/python -m pytest -v
```
