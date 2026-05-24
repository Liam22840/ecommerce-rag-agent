from pathlib import Path

from fastapi.testclient import TestClient

from server.app import create_app
from server.config import Settings


DATASET_ROOT = Path(__file__).parent.parent / "ecommerce_agent_dataset"


def _client() -> TestClient:
    settings = Settings(
        dataset_root=DATASET_ROOT,
        chat_api_key=None,
        embedding_api_key=None,
        enable_vector_search=False,
        enable_llm=False,
    )
    return TestClient(create_app(settings=settings))


def test_chat_endpoint_returns_grounded_product_cards():
    client = _client()

    resp = client.post("/api/chat", json={"message": "推荐一款适合油皮的洗面奶"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["degraded"] is True
    assert body["retrieval_source"] == "lexical"
    assert body["products"]
    assert body["products"][0]["product_id"] == "p_beauty_011"
    assert "商品库" in body["answer"]


def test_chat_endpoint_handles_no_exact_match_without_hallucinating():
    client = _client()

    resp = client.post("/api/chat", json={"message": "200 元以下的蓝牙耳机有哪些？"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["products"] == []
    assert "没有在商品库中找到完全匹配" in body["answer"]


def test_product_detail_endpoint():
    client = _client()

    resp = client.get("/api/products/p_beauty_011")

    assert resp.status_code == 200
    assert resp.json()["product_id"] == "p_beauty_011"


def test_stream_endpoint_uses_sse_events():
    client = _client()

    with client.stream("POST", "/api/chat/stream", json={"message": "推荐一款适合油皮的洗面奶"}) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert "event: meta" in body
    assert "event: delta" in body
    assert "event: products" in body
    assert "event: done" in body
