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
    assert "1." not in body["answer"]


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


def test_product_image_asset_endpoint():
    client = _client()

    resp = client.get("/assets/products/1_美妆护肤/images/p_beauty_011_live.jpg")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content


def test_stream_endpoint_uses_sse_events():
    client = _client()

    with client.stream("POST", "/api/chat/stream", json={"message": "推荐一款适合油皮的洗面奶"}) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert "event: token" in body
    assert "event: products" in body
    assert "event: done" in body
    assert '"base_price"' in body
    assert '"reason"' in body
    assert '"items"' in body


def test_stream_endpoint_accepts_ios_payload_and_legacy_path():
    client = _client()

    payload = {
        "conversation_id": "ios-session-1",
        "message": "推荐一款适合油皮的洗面奶",
        "attachments": [],
        "client_context": {"cart_items": []},
    }
    with client.stream("POST", "/api/v1/chat/stream", json=payload) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert '"session_id": "ios-session-1"' in body
    assert "event: token" in body
    assert "event: products" in body


def test_chat_endpoint_defaults_to_three_product_cards():
    client = _client()

    resp = client.post("/api/chat", json={"message": "推荐一款适合油皮的洗面奶"})

    assert resp.status_code == 200
    assert len(resp.json()["products"]) <= 3


def test_chat_endpoint_orders_selected_cards_by_price_when_requested():
    client = _client()

    resp = client.post(
        "/api/chat",
        json={"message": "推荐一个适合敏感肌的保湿护肤品，cheaper is better"},
    )

    assert resp.status_code == 200
    body = resp.json()
    prices = [product["price"] for product in body["products"]]
    assert prices == sorted(prices)
    assert [product["product_id"] for product in body["products"]] == [
        "p_beauty_007",
        "p_beauty_022",
        "p_beauty_012",
    ]
    assert "15g 体验装 89元；50g 标准装 268元" in body["products"][0]["price_summary"]
    assert "卡片" in body["answer"]
    assert "p_beauty_002" not in [product["product_id"] for product in body["products"]]


def test_chat_endpoint_uses_requested_sku_price_for_specs():
    client = _client()

    resp = client.post(
        "/api/chat",
        json={"message": "推荐50g适合敏感肌的保湿霜，cheaper is better"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert [product["product_id"] for product in body["products"]] == ["p_beauty_007"]
    assert body["products"][0]["price"] == 268.0
    assert body["products"][0]["price_label"] == "268元（50g 标准装）"
    assert body["products"][0]["price_summary"] == "15g 体验装 89元；50g 标准装 268元"
    assert "SKU价格" in body["answer"]
