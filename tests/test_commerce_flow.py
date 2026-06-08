import json
from pathlib import Path

from fastapi.testclient import TestClient

from server.app import create_app
from server.assistant import ShoppingAssistant
from server.catalog import ProductCatalog
from server.commerce import CommerceActionCandidate, CommerceService, OrderState
from server.config import Settings
from server.retrieval import ProductRetriever


DATASET_ROOT = Path(__file__).parent.parent / "ecommerce_agent_dataset"


class CommerceLLM:
    """Returns a fixed commerce action whenever the commerce intent parser is consulted."""

    available = True

    def __init__(self, action: str):
        self._action = action

    def _reply(self, messages: list[dict[str, str]]) -> str:
        if "购物车/下单" in messages[0]["content"]:
            return (
                '{"action":"%s","refs":[],"product_ids":[],"quantity":null,'
                '"target_scope":"cart_items","confidence":"high"}' % self._action
            )
        return ""

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self._reply(messages)

    def stream(self, messages: list[dict[str, str]]):
        yield self._reply(messages)


def _client_with_commerce_llm(action: str) -> TestClient:
    settings = Settings(
        dataset_root=DATASET_ROOT,
        chat_api_key=None,
        embedding_api_key=None,
        enable_vector_search=False,
        enable_llm=False,
        enable_llm_intent=False,
        enable_query_cache=False,
    )
    catalog = ProductCatalog.load(DATASET_ROOT)
    retriever = ProductRetriever(catalog, settings)
    assistant = ShoppingAssistant(
        catalog=catalog, retriever=retriever, llm=CommerceLLM(action), intent_llm=None, settings=settings
    )
    return TestClient(create_app(settings=settings, assistant=assistant))


def test_checkout_proceeds_when_the_llm_corroborates_the_keyword():
    client = _client_with_commerce_llm("checkout")
    session_id = "checkout-agree"
    products = _search(client, session_id)
    cart = _cart_payload((products[0], 2))

    resp = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "下单吧", "client_context": {"cart_items": cart}},
    )

    assert resp.status_code == 200
    assert resp.json()["order"]["status"] == "awaiting_confirmation"


def test_incidental_checkout_keyword_is_vetoed_when_the_llm_disagrees():
    # "下单" appears but the LLM reads no order intent, so the incidental keyword must not check out.
    client = _client_with_commerce_llm("none")
    session_id = "checkout-veto"
    products = _search(client, session_id)
    cart = _cart_payload((products[0], 2))

    resp = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "这个下单后多久能到货", "client_context": {"cart_items": cart}},
    )

    assert resp.status_code == 200
    assert resp.json()["order"] is None


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


# --- Single-router behaviours: fast-path, corroboration, measure-word quantity, pending replies ----


class _StubLLM:
    """Returns one fixed JSON payload for any prompt. For unit-testing CommerceService directly."""

    available = True

    def __init__(self, payload: dict):
        self._payload = payload

    def complete(self, _messages) -> str:
        return json.dumps(self._payload)


def _a_product_id() -> str:
    return ProductCatalog.load(DATASET_ROOT).products[0]["product_id"]


def test_confirm_order_needs_llm_agreement_before_submitting():
    catalog = ProductCatalog.load(DATASET_ROOT)
    pid = _a_product_id()
    cart = [{"product_id": pid, "quantity": 1}]

    state = OrderState()
    draft = CommerceService(catalog, llm=_StubLLM({"action": "checkout"})).maybe_handle(
        "下单", cart_items=cart, session_products=None, order_state=state
    )
    assert draft.order.status == "awaiting_confirmation"

    # "确认一下评价" contains 确认 but the LLM reads no confirm intent -> must NOT submit.
    vetoed = CommerceService(catalog, llm=_StubLLM({"action": "none"})).maybe_handle(
        "确认一下这个评价", cart_items=cart, session_products=None, order_state=state
    )
    assert vetoed is None
    assert state.draft is not None  # still awaiting confirmation

    submitted = CommerceService(catalog, llm=_StubLLM({"action": "confirm_order"})).maybe_handle(
        "确认", cart_items=cart, session_products=None, order_state=state
    )
    assert submitted.order.status == "submitted"


def test_measure_word_quantity_is_understood_via_the_llm():
    catalog = ProductCatalog.load(DATASET_ROOT)
    pid = _a_product_id()
    product = catalog.get(pid)
    session = [{"id": pid, "title": product["title"], "brand": product["brand"],
                "price": 1.0, "sub_category": product["sub_category"]}]
    # "要五个" is a measure word the 件-only parser can't read; the LLM supplies quantity 5.
    svc = CommerceService(catalog, llm=_StubLLM({"action": "add", "product_ids": [pid], "quantity": 5}))

    res = svc.maybe_handle("加入购物车，要五个", cart_items=[], session_products=session, order_state=OrderState())

    assert res is not None
    assert res.cart.items[0].quantity == 5


def test_multi_ordinal_add_adds_all_referenced_products_deterministically():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0, p1 = catalog.products[0]["product_id"], catalog.products[1]["product_id"]
    session = [{"id": p0, "title": catalog.get(p0)["title"]}, {"id": p1, "title": catalog.get(p1)["title"]}]

    res = CommerceService(catalog, llm=None).maybe_handle(
        "第一个和第二个加入购物车", cart_items=[], session_products=session, order_state=OrderState()
    )

    assert {i.product_id for i in res.cart.items} == {p0, p1}


def test_per_item_quantities_on_a_multi_add():
    # "第一个买两瓶，第二个买三瓶" — different counts per product via the LLM's items list.
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0, p1 = catalog.products[0]["product_id"], catalog.products[1]["product_id"]
    session = [{"id": p0, "title": catalog.get(p0)["title"]}, {"id": p1, "title": catalog.get(p1)["title"]}]
    svc = CommerceService(catalog, llm=_StubLLM(
        {"action": "add", "items": [{"product_id": p0, "quantity": 2}, {"product_id": p1, "quantity": 3}]}
    ))

    res = svc.maybe_handle("第一个买两瓶，第二个买三瓶", cart_items=[], session_products=session, order_state=OrderState())

    assert {i.product_id: i.quantity for i in res.cart.items} == {p0: 2, p1: 3}


def test_add_rejects_an_llm_id_that_was_not_shown():
    # Regression: "最便宜的那个" must resolve among shown items, not let the LLM cart an off-screen
    # product (it once added the globally cheapest catalogue item the user never saw).
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0, p1 = catalog.products[0]["product_id"], catalog.products[1]["product_id"]
    offscreen = catalog.products[5]["product_id"]
    session = [{"id": p0, "title": catalog.get(p0)["title"]}, {"id": p1, "title": catalog.get(p1)["title"]}]
    svc = CommerceService(catalog, llm=_StubLLM({"action": "add", "product_ids": [offscreen]}))

    res = svc.maybe_handle("把最便宜的那个加入购物车", cart_items=[], session_products=session, order_state=OrderState())

    carted = {i.product_id for i in res.cart.items} if res.cart else set()
    assert offscreen not in carted


def test_fuzzy_multi_add_uses_the_llm_resolved_ids():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0, p1 = catalog.products[0]["product_id"], catalog.products[1]["product_id"]
    session = [{"id": p0, "title": catalog.get(p0)["title"]}, {"id": p1, "title": catalog.get(p1)["title"]}]
    # "便宜的两个" has no ordinals; only the LLM can turn it into two ids.
    svc = CommerceService(catalog, llm=_StubLLM({"action": "add", "product_ids": [p0, p1]}))

    res = svc.maybe_handle("把便宜的两个加入购物车", cart_items=[], session_products=session, order_state=OrderState())

    assert {i.product_id for i in res.cart.items} == {p0, p1}


def test_measure_word_quantity_is_filled_by_the_llm():
    # The LLM fills which item and how many; a measure word it understands ("要五个") yields quantity 5.
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0, p1 = catalog.products[0]["product_id"], catalog.products[1]["product_id"]
    session = [
        {"id": p0, "title": catalog.get(p0)["title"], "price": 1.0},
        {"id": p1, "title": catalog.get(p1)["title"], "price": 2.0},
    ]
    svc = CommerceService(catalog, llm=_StubLLM({"action": "add", "product_ids": [p0], "quantity": 5}))

    res = svc.maybe_handle("把第一个加入购物车，要五个", cart_items=[], session_products=session, order_state=OrderState())

    assert res.cart.items[0].product_id == p0
    assert res.cart.items[0].quantity == 5


def test_pending_reply_ordinal_uses_no_llm():
    catalog = ProductCatalog.load(DATASET_ROOT)
    pid = _a_product_id()
    product = catalog.get(pid)
    session = [{"id": pid, "title": product["title"], "price": 1.0}]

    class _CountLLM:
        available = True
        calls = 0

        def complete(self, _messages):
            type(self).calls += 1
            return "{}"

    state = OrderState(pending_action=CommerceActionCandidate(action="add", target_scope="shown_products"))
    res = CommerceService(catalog, llm=_CountLLM()).handle_pending_reply(
        "第一个", cart_items=[], session_products=session, order_state=state
    )

    assert res.cart.items[0].product_id == pid
    assert _CountLLM.calls == 0


def test_pending_reply_natural_language_resolves_via_llm():
    catalog = ProductCatalog.load(DATASET_ROOT)
    pid = _a_product_id()
    product = catalog.get(pid)
    session = [{"id": pid, "title": product["title"], "price": 1.0}]
    state = OrderState(pending_action=CommerceActionCandidate(action="add", target_scope="shown_products"))

    res = CommerceService(catalog, llm=_StubLLM({"outcome": "resolve", "product_id": pid})).handle_pending_reply(
        "那个便宜的", cart_items=[], session_products=session, order_state=state
    )

    assert res.cart.items[0].product_id == pid
    assert state.pending_action is None


def test_pending_reply_abandonment_clears_and_does_not_act():
    catalog = ProductCatalog.load(DATASET_ROOT)
    state = OrderState(pending_action=CommerceActionCandidate(action="add", target_scope="shown_products"))

    res = CommerceService(catalog, llm=_StubLLM({"outcome": "abandon", "product_id": None})).handle_pending_reply(
        "算了，看看别的", cart_items=[], session_products=[{"id": "x"}], order_state=state
    )

    assert res is None
    assert state.pending_action is None
