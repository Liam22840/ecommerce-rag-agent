import json
from pathlib import Path

from fastapi.testclient import TestClient

from server.app import create_app
from server.assistant import ShoppingAssistant
from server.catalog import ProductCatalog
from server.commerce import (
    CommerceActionCandidate,
    CommerceService,
    OrderState,
    _coerce_int,
    _looks_like_add_ref,
    _pool_product_id,
    looks_like_commerce,
)
from server.config import Settings
from server.pricing import MAX_CART_QUANTITY, build_cart_item
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


def test_stream_emits_order_event_on_checkout():
    client = _client()
    session_id = "stream-order"
    products = _search(client, session_id)
    cart = _cart_payload((products[0], 1))

    with client.stream(
        "POST", "/api/chat/stream",
        json={"session_id": session_id, "message": "下单吧", "client_context": {"cart_items": cart}},
    ) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert "event: order" in body
    assert '"type": "order_draft"' in body


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


def test_cart_line_quantity_is_capped_to_a_sane_maximum():
    # A pathological "要1000000件" clamps to MAX_CART_QUANTITY instead of pricing an absurd subtotal.
    catalog = ProductCatalog.load(DATASET_ROOT)
    product = catalog.products[0]
    item = build_cart_item(catalog, product, 1_000_000)
    assert item.quantity == MAX_CART_QUANTITY
    assert item.line_total == round(item.unit_price * MAX_CART_QUANTITY, 2)


def test_add_honours_a_named_sku_price():
    # The user names a 规格; the cart line is priced for that SKU, not the lowest one.
    catalog = ProductCatalog.load(DATASET_ROOT)
    pid = next(p["product_id"] for p in catalog.products
               if len({s["price"] for s in catalog.sku_prices(p)}) >= 2)
    skus = sorted(catalog.sku_prices(catalog.get(pid)), key=lambda s: s["price"])
    dearer = skus[-1]
    session = [{"id": pid, "title": catalog.get(pid)["title"]}]

    svc = CommerceService(catalog, llm=_StubLLM({"action": "add", "product_ids": [pid], "sku": dearer["label"]}))
    res = svc.maybe_handle(f"要{dearer['label']}的加入购物车", cart_items=[], session_products=session, order_state=OrderState())
    assert res.cart.items[0].unit_price == dearer["price"]

    # Without a 规格 phrase the lowest SKU is used, as before.
    svc2 = CommerceService(catalog, llm=_StubLLM({"action": "add", "product_ids": [pid]}))
    res2 = svc2.maybe_handle("加入购物车", cart_items=[], session_products=session, order_state=OrderState())
    assert res2.cart.items[0].unit_price == skus[0]["price"]


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


# --- degraded / edge branches (unit-level) -----------------------------------

class _RaisingLLM:
    """An available LLM whose complete() always raises, for the fall-back-to-deterministic branches."""

    available = True

    def complete(self, _messages):
        raise RuntimeError("llm down")


def _ids(catalog, n: int = 2):
    return [catalog.products[i]["product_id"] for i in range(n)]


def _session(catalog, *pids):
    return [{"id": pid, "title": catalog.get(pid)["title"], "price": 1.0} for pid in pids]


# pending-reply resolver

def test_handle_pending_reply_returns_none_without_a_pending_action():
    catalog = ProductCatalog.load(DATASET_ROOT)
    res = CommerceService(catalog).handle_pending_reply(
        "第一个", cart_items=[], session_products=_session(catalog, _a_product_id()), order_state=OrderState()
    )
    assert res is None


def test_pending_reply_without_ordinal_and_no_llm_leaves_pending_set():
    catalog = ProductCatalog.load(DATASET_ROOT)
    state = OrderState(pending_action=CommerceActionCandidate(action="add", target_scope="shown_products"))
    res = CommerceService(catalog, llm=None).handle_pending_reply(
        "那个便宜的", cart_items=[], session_products=_session(catalog, _a_product_id()), order_state=state
    )
    assert res is None
    assert state.pending_action is not None


def test_pending_reply_llm_failure_leaves_pending_set():
    catalog = ProductCatalog.load(DATASET_ROOT)
    state = OrderState(pending_action=CommerceActionCandidate(action="add", target_scope="shown_products"))
    res = CommerceService(catalog, llm=_RaisingLLM()).handle_pending_reply(
        "那个便宜的", cart_items=[], session_products=_session(catalog, _a_product_id()), order_state=state
    )
    assert res is None
    assert state.pending_action is not None


def test_pending_reply_resolve_to_an_offpool_id_is_rejected():
    catalog = ProductCatalog.load(DATASET_ROOT)
    pid, offscreen = catalog.products[0]["product_id"], catalog.products[5]["product_id"]
    state = OrderState(pending_action=CommerceActionCandidate(action="add", target_scope="shown_products"))
    res = CommerceService(catalog, llm=_StubLLM({"outcome": "resolve", "product_id": offscreen})).handle_pending_reply(
        "那个便宜的", cart_items=[], session_products=_session(catalog, pid), order_state=state
    )
    assert res is None
    assert state.pending_action is not None  # not cleared, the model named an off-pool id


def test_pending_reply_unknown_outcome_keeps_pending():
    catalog = ProductCatalog.load(DATASET_ROOT)
    state = OrderState(pending_action=CommerceActionCandidate(action="add", target_scope="shown_products"))
    res = CommerceService(catalog, llm=_StubLLM({"outcome": "???"})).handle_pending_reply(
        "那个便宜的", cart_items=[], session_products=_session(catalog, _a_product_id()), order_state=state
    )
    assert res is None
    assert state.pending_action is not None  # only abandon/not_a_reply clear it


def test_pending_reply_cart_scope_resolves_against_the_cart_pool():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0, p1 = _ids(catalog)
    cart = [{"product_id": p0, "quantity": 1}, {"product_id": p1, "quantity": 1}]
    state = OrderState(pending_action=CommerceActionCandidate(action="remove", target_scope="cart_items"))
    res = CommerceService(catalog, llm=_StubLLM({"outcome": "resolve", "product_id": p0})).handle_pending_reply(
        "那个便宜的", cart_items=cart, session_products=None, order_state=state
    )
    assert {i.product_id for i in res.cart.items} == {p1}  # p0 removed via the cart pool


# deterministic parser fall-backs

def test_deterministic_cancel_order_keeps_cart_and_clears_draft():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0 = catalog.products[0]["product_id"]
    cart = [{"product_id": p0, "quantity": 1}]
    state = OrderState()
    CommerceService(catalog, llm=None).maybe_handle("下单", cart_items=cart, session_products=None, order_state=state)
    assert state.draft is not None

    res = CommerceService(catalog, llm=None).maybe_handle("取消", cart_items=cart, session_products=None, order_state=state)
    assert res.order.status == "cancelled"
    assert {i.product_id for i in res.cart.items} == {p0}  # cart retained
    assert state.draft is None


def test_deterministic_decrement_reduces_quantity():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0 = catalog.products[0]["product_id"]
    res = CommerceService(catalog, llm=None).maybe_handle(
        "减一件", cart_items=[{"product_id": p0, "quantity": 2}], session_products=None, order_state=OrderState()
    )
    assert res.cart.items[0].quantity == 1
    assert res.intent["commerce_action"] == "decrement"


def test_deterministic_show_cart_summarises_the_cart():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0, p1 = _ids(catalog)
    res = CommerceService(catalog, llm=None).maybe_handle(
        "查看购物车",
        cart_items=[{"product_id": p0, "quantity": 1}, {"product_id": p1, "quantity": 1}],
        session_products=None, order_state=OrderState(),
    )
    assert res.cart.action == "show_cart"
    assert "购物车共" in res.answer


def test_non_commerce_message_returns_none():
    catalog = ProductCatalog.load(DATASET_ROOT)
    res = CommerceService(catalog, llm=None).maybe_handle(
        "今天天气不错", cart_items=[], session_products=None, order_state=OrderState()
    )
    assert res is None


# LLM payload parsing edges

def test_llm_failure_falls_back_to_deterministic_add():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0 = catalog.products[0]["product_id"]
    res = CommerceService(catalog, llm=_RaisingLLM()).maybe_handle(
        "加入购物车", cart_items=[], session_products=_session(catalog, p0), order_state=OrderState()
    )
    assert {i.product_id for i in res.cart.items} == {p0}


def test_llm_invalid_action_yields_no_commerce():
    catalog = ProductCatalog.load(DATASET_ROOT)
    res = CommerceService(catalog, llm=_StubLLM({"action": "banana"})).maybe_handle(
        "帮我看看", cart_items=[], session_products=None, order_state=OrderState()
    )
    assert res is None


def test_llm_items_skips_non_dict_and_empty_pid_entries():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0 = catalog.products[0]["product_id"]
    svc = CommerceService(catalog, llm=_StubLLM(
        {"action": "add", "product_ids": [p0],
         "items": ["junk", {"product_id": "", "quantity": 2}, {"product_id": p0, "quantity": 4}]}
    ))
    res = svc.maybe_handle("加入购物车", cart_items=[], session_products=_session(catalog, p0), order_state=OrderState())
    assert {i.product_id: i.quantity for i in res.cart.items} == {p0: 4}  # per-item qty applied, junk ignored


def test_llm_item_with_null_quantity_does_not_override_base_quantity():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0 = catalog.products[0]["product_id"]
    svc = CommerceService(catalog, llm=_StubLLM(
        {"action": "add", "product_ids": [p0], "quantity": 2, "items": [{"product_id": p0, "quantity": None}]}
    ))
    res = svc.maybe_handle("加入购物车", cart_items=[], session_products=_session(catalog, p0), order_state=OrderState())
    assert res.cart.items[0].quantity == 2


# checkout / confirm state machine

def test_confirm_order_with_empty_cart_reports_and_clears_draft():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0 = catalog.products[0]["product_id"]
    state = OrderState()
    CommerceService(catalog, llm=_StubLLM({"action": "checkout"})).maybe_handle(
        "下单", cart_items=[{"product_id": p0, "quantity": 1}], session_products=None, order_state=state
    )
    assert state.draft is not None

    res = CommerceService(catalog, llm=_StubLLM({"action": "confirm_order"})).maybe_handle(
        "确认", cart_items=[], session_products=None, order_state=state
    )
    assert res.order is None
    assert "购物车为空，无法提交订单" in res.answer
    assert state.draft is None


def test_confirm_order_requotes_when_the_cart_changed_since_the_draft():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0 = catalog.products[0]["product_id"]
    state = OrderState()
    CommerceService(catalog, llm=_StubLLM({"action": "checkout"})).maybe_handle(
        "下单", cart_items=[{"product_id": p0, "quantity": 2}], session_products=None, order_state=state
    )
    res = CommerceService(catalog, llm=_StubLLM({"action": "confirm_order"})).maybe_handle(
        "确认", cart_items=[{"product_id": p0, "quantity": 3}], session_products=None, order_state=state
    )
    assert res.order.status == "awaiting_confirmation"  # re-quoted, not submitted


# cart normalisation filtering

def test_normalize_cart_drops_rows_without_id_unknown_product_or_nonpositive_quantity():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0, p1 = _ids(catalog)
    cart = [
        {"quantity": 1},                                  # no product id
        {"product_id": "p_nope_999", "quantity": 1},      # not in catalog
        {"product_id": p1, "quantity": -1},               # non-positive quantity
        {"product_id": p0, "quantity": 1},                # the only valid row
    ]
    res = CommerceService(catalog, llm=None).maybe_handle(
        "查看购物车", cart_items=cart, session_products=None, order_state=OrderState()
    )
    assert {i.product_id for i in res.cart.items} == {p0}


def test_normalize_cart_sums_duplicate_rows_of_the_same_product():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0 = catalog.products[0]["product_id"]
    res = CommerceService(catalog, llm=None).maybe_handle(
        "查看购物车",
        cart_items=[{"product_id": p0, "quantity": 1}, {"product_id": p0, "quantity": 1}],
        session_products=None, order_state=OrderState(),
    )
    assert res.cart.items[0].quantity == 2


# single-product / cart-item resolver fall-backs

def test_add_deictic_with_a_single_shown_product_resolves_it():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0 = catalog.products[0]["product_id"]
    res = CommerceService(catalog, llm=None).maybe_handle(
        "这个加入购物车", cart_items=[], session_products=_session(catalog, p0), order_state=OrderState()
    )
    assert {i.product_id for i in res.cart.items} == {p0}
    assert _pool_product_id(res.cart.items[0]) == p0  # CartItem branch of _pool_product_id


def test_add_deictic_with_multiple_shown_products_clarifies():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0, p1 = _ids(catalog)
    res = CommerceService(catalog, llm=None).maybe_handle(
        "这个加入购物车", cart_items=[], session_products=_session(catalog, p0, p1), order_state=OrderState()
    )
    assert res.cart.needs_clarification is True
    assert "不够明确" in res.answer


def test_add_without_a_ref_and_a_single_shown_product_auto_resolves():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0 = catalog.products[0]["product_id"]
    res = CommerceService(catalog, llm=None).maybe_handle(
        "加入购物车", cart_items=[], session_products=_session(catalog, p0), order_state=OrderState()
    )
    assert {i.product_id for i in res.cart.items} == {p0}


def test_cart_action_on_an_empty_cart_reports():
    catalog = ProductCatalog.load(DATASET_ROOT)
    res = CommerceService(catalog, llm=None).maybe_handle(
        "删除第一个商品", cart_items=[], session_products=None, order_state=OrderState()
    )
    assert res.cart.needs_clarification is True
    assert "购物车为空" in res.answer


def test_remove_resolves_a_cart_item_by_llm_product_id():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0, p1 = _ids(catalog)
    cart = [{"product_id": p0, "quantity": 1}, {"product_id": p1, "quantity": 1}]
    res = CommerceService(catalog, llm=_StubLLM({"action": "remove", "product_ids": [p0]})).maybe_handle(
        "删掉那个", cart_items=cart, session_products=None, order_state=OrderState()
    )
    assert {i.product_id for i in res.cart.items} == {p1}


def test_cart_ordinal_out_of_range_clarifies():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0, p1 = _ids(catalog)
    cart = [{"product_id": p0, "quantity": 1}, {"product_id": p1, "quantity": 1}]
    res = CommerceService(catalog, llm=None).maybe_handle(
        "删除第五个商品", cart_items=cart, session_products=None, order_state=OrderState()
    )
    assert res.cart.needs_clarification is True
    assert "购物车里没有你说的第几个商品" in res.answer


def test_maybe_handle_resolves_a_pending_action_via_an_ordinal_reply():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0 = catalog.products[0]["product_id"]
    state = OrderState(pending_action=CommerceActionCandidate(action="add", target_scope="shown_products"))
    res = CommerceService(catalog, llm=None).maybe_handle(
        "第一个", cart_items=[], session_products=_session(catalog, p0), order_state=state
    )
    assert {i.product_id for i in res.cart.items} == {p0}


# pure helpers

def test_cart_summary_reports_an_empty_cart():
    catalog = ProductCatalog.load(DATASET_ROOT)
    assert CommerceService(catalog)._cart_summary([]) == "购物车为空。"


def test_coerce_int_handles_int_float_and_string_forms():
    assert _coerce_int(3) == 3
    assert _coerce_int(2.0) == 2
    assert _coerce_int("3") == 3
    assert _coerce_int("三") == 3
    assert _coerce_int(None) is None
    assert _coerce_int(True) is None
    assert _coerce_int(["3"]) is None


def test_looks_like_add_ref_for_bare_ordinal_when_no_cart_exists():
    assert _looks_like_add_ref("第一件", has_cart=False) is True
    assert _looks_like_add_ref("第一件", has_cart=True) is False
    assert _looks_like_add_ref("加这个", has_cart=False) is True       # verb-then-ref
    assert _looks_like_add_ref("第一个加入购物车", has_cart=True) is True  # ref-then-verb


def test_looks_like_commerce_via_regex_fallbacks_without_a_keyword():
    assert looks_like_commerce("第一个数量") is True   # ref + cart noun, no literal hint
    assert looks_like_commerce("加第二个") is True      # verb + ordinal
    assert looks_like_commerce("讲个笑话") is False


def test_add_ordinal_out_of_range_asks_to_clarify():
    catalog = ProductCatalog.load(DATASET_ROOT)
    p0, p1 = _ids(catalog)
    res = CommerceService(catalog, llm=None).maybe_handle(
        "第五个加入购物车", cart_items=[], session_products=_session(catalog, p0, p1), order_state=OrderState()
    )
    assert res.cart.needs_clarification is True
    assert "序号" in res.answer  # asks for a valid position rather than grabbing the first item
