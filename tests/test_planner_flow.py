from pathlib import Path

from fastapi.testclient import TestClient

from server.app import create_app
from server.assistant import ShoppingAssistant
from server.catalog import ProductCatalog
from server.config import Settings
from server.retrieval import ProductRetriever


DATASET_ROOT = Path(__file__).parent.parent / "ecommerce_agent_dataset"


def _client() -> TestClient:
    settings = Settings(
        dataset_root=DATASET_ROOT,
        chat_api_key=None,
        embedding_api_key=None,
        enable_vector_search=False,
        enable_llm=False,
        enable_llm_intent=False,
        enable_query_cache=False,
    )
    return TestClient(create_app(settings=settings))


class PlannerLLM:
    available = True

    def __init__(self, response: str):
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.response

    def stream(self, _messages: list[dict[str, str]]):
        yield from []


def _client_with_planner_llm(llm: PlannerLLM) -> TestClient:
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
    assistant = ShoppingAssistant(catalog=catalog, retriever=retriever, llm=llm, intent_llm=None, settings=settings)
    return TestClient(create_app(settings=settings, assistant=assistant))


def test_planner_searches_selects_cheapest_product_and_adds_it_to_cart():
    client = _client()

    resp = client.post(
        "/api/chat",
        json={
            "session_id": "planner-cheapest-add",
            "message": "帮我推荐跑鞋，并把最便宜的一双加入购物车",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["plan"] is not None
    assert [step["action"] for step in body["plan"]["steps"]] == [
        "product_search",
        "select_products",
        "cart_action",
    ]
    assert all(step["status"] == "done" for step in body["plan"]["steps"])
    assert body["products"]
    assert body["cart"]["items"]

    cheapest = min(body["products"], key=lambda product: product["price"])
    added = body["cart"]["items"][0]
    assert added["product_id"] == cheapest["product_id"]
    assert added["quantity"] == 1
    assert "加入购物车" in body["answer"]


def test_planner_compares_two_cheapest_products_and_adds_comparison_winner():
    client = _client()

    resp = client.post(
        "/api/chat",
        json={
            "session_id": "planner-compare-add",
            "message": "帮我推荐跑鞋，对比最便宜的两双哪个更便宜，把更便宜的加入购物车",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["plan"] is not None
    assert [step["action"] for step in body["plan"]["steps"]] == [
        "product_search",
        "select_products",
        "comparison",
        "cart_action",
    ]
    assert body["comparison"] is not None
    assert body["comparison"]["winner_product_id"] is not None
    assert body["cart"]["items"][0]["product_id"] == body["comparison"]["winner_product_id"]
    assert all(product["sub_category"] == "跑步鞋" for product in body["products"])


def test_planner_does_not_hijack_single_step_cart_followup():
    client = _client()
    session_id = "planner-single-cart"
    search = client.post("/api/chat", json={"session_id": session_id, "message": "推荐三款保湿面霜"})
    assert search.status_code == 200
    first_product = search.json()["products"][0]

    resp = client.post("/api/chat", json={"session_id": session_id, "message": "把第一个加到购物车"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["plan"] is None
    assert body["cart"]["items"][0]["product_id"] == first_product["product_id"]
    assert body["comparison"] is None


def test_stream_emits_plan_event_before_cart_update():
    client = _client()

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "session_id": "planner-stream",
            "message": "帮我推荐跑鞋，并把最便宜的一双加入购物车",
        },
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert "event: plan" in body
    assert "event: cart" in body
    assert body.index("event: plan") < body.index("event: cart")
    assert '"action": "product_search"' in body
    assert '"action": "cart_action"' in body


def test_llm_planner_schema_executes_against_real_catalog_and_cart():
    llm = PlannerLLM(
        '{"steps":['
        '{"action":"product_search","title":"推荐耳机","query":"推荐蓝牙耳机"},'
        '{"action":"select_products","title":"筛选最低价","criteria":"price_asc","count":1},'
        '{"action":"cart_action","title":"加入购物车","target":"selected_products","quantity":1}'
        ']}'
    )
    client = _client_with_planner_llm(llm)

    resp = client.post(
        "/api/chat",
        json={
            "session_id": "planner-llm-schema",
            "message": "帮我找一款蓝牙耳机，然后放进购物车",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert llm.calls
    assert body["plan"]["steps"][0]["title"] == "推荐耳机"
    assert body["products"][0]["sub_category"] == "真无线耳机"
    assert body["cart"]["items"][0]["product_id"] == body["products"][0]["product_id"]
