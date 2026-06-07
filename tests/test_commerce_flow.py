from pathlib import Path

from fastapi.testclient import TestClient

from server.app import create_app
from server.config import Settings


DATASET_ROOT = Path(__file__).parent.parent / "ecommerce_agent_dataset"


def _client(tmp_path: Path | None = None, enable_query_cache: bool = False) -> TestClient:
    settings = Settings(
        dataset_root=DATASET_ROOT,
        chat_api_key=None,
        embedding_api_key=None,
        enable_vector_search=False,
        enable_llm=False,
        enable_llm_intent=False,
        enable_query_cache=enable_query_cache,
        query_cache_path=(tmp_path or Path("/tmp")) / "query-cache.jsonl",
    )
    return TestClient(create_app(settings=settings))


def _search(client: TestClient, session_id: str) -> list[dict]:
    resp = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "推荐三款保湿面霜，价格从低到高"},
    )
    assert resp.status_code == 200
    products = resp.json()["products"]
    assert len(products) >= 2
    return products


def _cart_payload(*items: tuple[dict, int]) -> list[dict]:
    return [{"product": product, "product_id": product["product_id"], "quantity": qty} for product, qty in items]


def test_adds_first_shown_product_to_cart_from_conversation_context():
    client = _client()
    session_id = "cart-add-first"
    products = _search(client, session_id)

    resp = client.post("/api/chat", json={"session_id": session_id, "message": "把第一个加到购物车"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["cart"]["items"][0]["product_id"] == products[0]["product_id"]
    assert body["cart"]["items"][0]["quantity"] == 1
    assert body["cart"]["items"][0]["product"]["price_label"] == products[0]["price_label"]
    assert "加入购物车" in body["answer"]
    assert body["intent"]["intent_type"] == "cart_action"


def test_add_uses_client_recent_product_ids_when_server_session_memory_is_empty():
    client = _client()
    session_id = "cart-client-recent"
    products = _search(client, session_id)

    # Simulate stream-cache replay, backend restart, or another path where the app still has
    # visible products but the server has no shown_products for this conversation id.
    resp = client.post(
        "/api/chat",
        json={
            "session_id": "new-server-session",
            "message": "把第一个加到购物车",
            "client_context": {"recent_product_ids": [product["product_id"] for product in products]},
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["cart"]["items"][0]["product_id"] == products[0]["product_id"]
    assert "加入购物车" in body["answer"]


def test_adds_second_shown_product_with_quantity_from_language():
    client = _client()
    session_id = "cart-add-second"
    products = _search(client, session_id)

    resp = client.post("/api/chat", json={"session_id": session_id, "message": "第二个来两件"})

    assert resp.status_code == 200
    item = resp.json()["cart"]["items"][0]
    assert item["product_id"] == products[1]["product_id"]
    assert item["quantity"] == 2


def test_increment_single_cart_item_and_bypasses_query_cache(tmp_path: Path):
    client = _client(tmp_path, enable_query_cache=True)
    session_id = "cart-cache-bypass"
    products = _search(client, session_id)
    current_cart = _cart_payload((products[0], 1))

    first = client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "再加一件",
            "client_context": {"cart_items": current_cart},
        },
    )
    assert first.status_code == 200
    assert first.json()["cart"]["items"][0]["quantity"] == 2

    second = client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "再加一件",
            "client_context": {"cart_items": _cart_payload((products[0], 2))},
        },
    )
    assert second.status_code == 200
    assert second.json()["cart"]["items"][0]["quantity"] == 3


def test_ambiguous_increment_multiple_cart_items_asks_clarification_without_mutation():
    client = _client()
    session_id = "cart-ambiguous-increment"
    products = _search(client, session_id)
    cart = _cart_payload((products[0], 1), (products[1], 1))

    resp = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "再加一件", "client_context": {"cart_items": cart}},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert [item["quantity"] for item in body["cart"]["items"]] == [1, 1]
    assert "哪一件" in body["answer"] or "哪款" in body["answer"]


def test_pending_cart_clarification_accepts_bare_ordinal_reply():
    client = _client()
    session_id = "cart-pending-number"
    products = _search(client, session_id)
    cart = _cart_payload((products[0], 1), (products[1], 1))

    first = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "再加一件", "client_context": {"cart_items": cart}},
    )
    assert first.status_code == 200
    assert first.json()["cart"]["needs_clarification"] is True

    resolved = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "1", "client_context": {"cart_items": cart}},
    )

    assert resolved.status_code == 200
    items = resolved.json()["cart"]["items"]
    assert [item["quantity"] for item in items] == [2, 1]
    assert resolved.json()["intent"]["commerce_action"] == "increment"


def test_pending_cart_clarification_accepts_add_first_wording_as_cart_scope_reply():
    client = _client()
    session_id = "cart-pending-add-first"
    products = _search(client, session_id)
    cart = _cart_payload((products[0], 1), (products[1], 1))

    client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "再加一件", "client_context": {"cart_items": cart}},
    )
    resolved = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "加第一个", "client_context": {"cart_items": cart}},
    )

    assert resolved.status_code == 200
    body = resolved.json()
    assert body["comparison"] is None
    assert [item["quantity"] for item in body["cart"]["items"]] == [2, 1]


def test_add_first_wording_routes_to_cart_not_comparison_without_pending_clarification():
    client = _client()
    session_id = "cart-add-first-short"
    products = _search(client, session_id)

    resp = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "加第一个"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["comparison"] is None
    assert body["intent"]["intent_type"] == "cart_action"
    assert body["cart"]["items"][0]["product_id"] == products[0]["product_id"]


def test_removes_second_cart_item_not_second_shown_product():
    client = _client()
    session_id = "cart-remove-second"
    products = _search(client, session_id)
    cart = _cart_payload((products[0], 1), (products[1], 1))

    resp = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "删除第二个商品", "client_context": {"cart_items": cart}},
    )

    assert resp.status_code == 200
    items = resp.json()["cart"]["items"]
    assert len(items) == 1
    assert items[0]["product_id"] == products[0]["product_id"]


def test_sets_quantity_to_zero_removes_item_and_clear_cart_empties_cart():
    client = _client()
    session_id = "cart-quantity-clear"
    products = _search(client, session_id)
    cart = _cart_payload((products[0], 2))

    removed = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "数量改成0", "client_context": {"cart_items": cart}},
    )
    assert removed.status_code == 200
    assert removed.json()["cart"]["items"] == []

    cleared = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "清空购物车", "client_context": {"cart_items": cart}},
    )
    assert cleared.status_code == 200
    assert cleared.json()["cart"]["items"] == []


def test_checkout_empty_cart_and_confirm_order_clears_cart():
    client = _client()
    empty = client.post("/api/chat", json={"session_id": "order-empty", "message": "下单吧"})
    assert empty.status_code == 200
    assert empty.json()["order"] is None
    assert "购物车为空" in empty.json()["answer"]

    session_id = "order-confirm"
    products = _search(client, session_id)
    cart = _cart_payload((products[0], 2))
    draft = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "下单吧", "client_context": {"cart_items": cart}},
    )
    assert draft.status_code == 200
    assert draft.json()["order"]["status"] == "awaiting_confirmation"
    assert draft.json()["order"]["items"][0]["quantity"] == 2

    confirmed = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "确认", "client_context": {"cart_items": cart}},
    )
    assert confirmed.status_code == 200
    body = confirmed.json()
    assert body["order"]["status"] == "submitted"
    assert body["order"]["order_id"].startswith("EG")
    assert body["cart"]["items"] == []


def test_stream_emits_cart_event_before_done():
    client = _client()
    session_id = "stream-cart"
    _search(client, session_id)

    with client.stream("POST", "/api/chat/stream", json={"session_id": session_id, "message": "把第一个加购物车"}) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert "event: cart" in body
    assert '"type": "cart_updated"' in body
    assert body.index("event: cart") < body.index("event: done")
