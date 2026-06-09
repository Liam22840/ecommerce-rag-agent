# ecommerce-rag-agent

A multi-modal e-commerce intelligent shopping agent based on RAG.
基于 RAG 的多模态电商智能导购 Agent.

## What this repo contains (so far)

- `ecommerce_agent_dataset/`: 100+ products across 4 categories, with text fields and one image per product.
- `ingestion/`: Python package with chunk extraction, the Doubao multimodal embedding client, and the Milvus Lite store.
- `ingest.py`: CLI to run the full pipeline.

## Setup

```bash
uv venv
uv pip install -r requirements.txt
cp .env.example .env
# Then edit .env and set CHAT_API_KEY / ARK_EMBEDDING_API_KEY.
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

## Running the backend API

Create `.env` from the example and fill in the real API keys locally:

```bash
cp .env.example .env
# edit .env and set CHAT_API_KEY / ARK_EMBEDDING_API_KEY
```

The backend uses separate settings for chat and embeddings. Keep real keys in `.env` only:

- `CHAT_API_KEY`: key for the chat model
- `CHAT_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai`
- `CHAT_MODEL=gemini-3.1-flash-lite` (Gemini Flash-Lite via its OpenAI-compatible endpoint)
- `ARK_EMBEDDING_API_KEY`: key for query/product embeddings
- `ARK_EMBEDDING_BASE_URL=https://ark.cn-beijing.volces.com/api/v3`
- `ARK_EMBEDDING_MODEL=doubao-embedding-vision-251215`

Start the FastAPI service:

```bash
.venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Core endpoints:

- `GET /health`
- `POST /api/chat` for a full JSON response with answer + product cards
- `POST /api/chat/stream` for SSE streaming replies
- `GET /api/products/{product_id}` for product-card detail data

Example:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"推荐一款适合油皮的洗面奶"}'
```

If the embedding key or the vector DB is unavailable, the service degrades to
local lexical retrieval plus a deterministic grounded answer instead of
hallucinating product facts.

## Running the tests

```bash
.venv/bin/python -m pytest -v
```
