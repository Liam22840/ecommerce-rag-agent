from pathlib import Path

from fastapi.testclient import TestClient

from server.app import create_app
from server.assistant import ShoppingAssistant
from server.catalog import ProductCatalog
from server.config import Settings
from server.planner import (
    PlannedStep,
    PlannerService,
    _coerce_step,
    _positive_int,
    _search_query,
    _selection_count,
    _selection_criteria,
    _valid_plan,
    looks_like_planned_task,
)
from server.retrieval import ProductRetriever


def test_planner_gate_recognises_a_multistep_request_for_an_unlisted_product():
    # Regression: the search detector hardcoded a handful of product-type characters, so a compound
    # request for anything outside that list (裤子) was silently never treated as a multi-step task.
    assert looks_like_planned_task("买条裤子然后加入购物车") is True


def test_planner_gate_does_not_count_a_deictic_add_as_a_search():
    # A plain cart-add ("买这个") next to a connector must not be miscounted as search + commerce
    # and routed through the planner.
    assert looks_like_planned_task("买这个吧，谢谢") is False


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
    assert body.count("event: plan") >= 4
    assert '"status": "pending"' in body
    assert '"status": "running"' in body
    assert '"status": "done"' in body
    assert '"action": "product_search"' in body
    assert '"action": "cart_action"' in body


class ScriptedLLM:
    """Routes by prompt: a plan for the planner, an intent label for the intent parser, a decline for
    the chitchat answer. The compound message routes to planned_task; the bare watch search step
    parses to chitchat, so the out-of-catalogue decline surfaces inside the plan."""

    available = True

    def __init__(self, plan: str):
        self._plan = plan

    def _reply(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        if "planner" in system:
            return self._plan
        if "意图解析器" in system:
            user = messages[1]["content"]
            route = "planned_task" if "对比" in user else "chitchat"
            return '{"intent_type":"%s"}' % route
        return "抱歉，本店暂不提供手表，您可以看看我们在售的其他品类。"

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self._reply(messages)

    def stream(self, messages: list[dict[str, str]]):
        yield self._reply(messages)


def test_planner_declines_out_of_catalogue_item_instead_of_carting_a_substitute():
    # Regression: a watch isn't in the catalogue. The intent parser classifies it as chitchat, but
    # the planner used to force product_search and cart whatever retrieval returned (AirPods).
    llm = ScriptedLLM(
        '{"steps":['
        '{"action":"product_search","title":"推荐手表","query":"推荐一个手表"},'
        '{"action":"comparison","title":"对比手表","query":"对比手表"},'
        '{"action":"cart_action","title":"加入购物车","target":"comparison_winner","quantity":1}'
        ']}'
    )
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
    assistant = ShoppingAssistant(catalog=catalog, retriever=retriever, llm=llm, intent_llm=llm, settings=settings)
    client = TestClient(create_app(settings=settings, assistant=assistant))

    resp = client.post(
        "/api/chat",
        json={
            "session_id": "planner-out-of-catalogue",
            "message": "推荐一个手表，对比一下哪个好，再加入购物车",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["products"] == []
    assert body["cart"] is None
    assert body["comparison"] is None
    assert body["plan"]["steps"][0]["status"] == "failed"
    assert "手表" in body["answer"]


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


# --- planner helpers and degraded paths (unit-level) --------------------------

class _PlanLLM:
    """A planner LLM whose reply (or raised error) is fixed, for driving the degraded branches."""

    def __init__(self, result: str = "", error: Exception | None = None, available: bool = True):
        self.available = available
        self._result = result
        self._error = error

    def complete(self, _messages):
        if self._error is not None:
            raise self._error
        return self._result

    def stream(self, _messages):
        yield from []


def _planner(llm=None) -> PlannerService:
    return PlannerService(categories=set(), sub_categories=set(), brands=set(), llm=llm)


def test_coerce_step_drops_unknown_action():
    assert _coerce_step({"action": "frobnicate"}) is None
    assert _coerce_step({"title": "无动作"}) is None  # missing action


def test_coerce_step_normalises_invalid_select_criteria_to_relevance():
    step = _coerce_step({"action": "select_products", "criteria": "cheapest"})
    assert step is not None and step.criteria == "relevance"


def test_coerce_step_normalises_invalid_cart_target_to_previous_step():
    step = _coerce_step({"action": "cart_action", "target": "the moon"})
    assert step is not None and step.target == "previous_step"


def test_coerce_step_uses_default_title_when_missing():
    step = _coerce_step({"action": "checkout"})
    assert step is not None and step.title == "创建订单"


def test_positive_int_handles_float_digit_string_and_chinese():
    assert _positive_int(2.0) == 2
    assert _positive_int(0.0) is None
    assert _positive_int("2") == 2
    assert _positive_int("两") == 2
    assert _positive_int("零") is None       # chinese_to_int -> 0, not positive
    assert _positive_int(True) is None       # bool is not a count
    assert _positive_int(["3"]) is None       # unsupported type


def test_valid_plan_rejects_single_step():
    assert _valid_plan([PlannedStep(action="product_search", title="找")]) is False


def test_valid_plan_rejects_duplicate_product_search():
    steps = [
        PlannedStep(action="product_search", title="找一"),
        PlannedStep(action="product_search", title="找二"),
    ]
    assert _valid_plan(steps) is False


def test_valid_plan_rejects_cart_action_without_a_source_step():
    orphan = [PlannedStep(action="cart_action", title="加购"), PlannedStep(action="checkout", title="下单")]
    assert _valid_plan(orphan) is False
    sourced = [PlannedStep(action="product_search", title="找"), PlannedStep(action="cart_action", title="加购")]
    assert _valid_plan(sourced) is True


def test_selection_criteria_maps_expensive_rated_and_default():
    assert _selection_criteria("买最贵的") == "price_desc"
    assert _selection_criteria("挑好评最多的") == "rating_desc"
    assert _selection_criteria("随便看看") == "relevance"


def test_selection_count_recognises_pair_word_and_falls_back_to_default():
    assert _selection_count("买两款", default=1) == 2     # 两 -> at least a pair
    assert _selection_count("买个东西", default=1) == 1    # nothing numeric -> default


def test_search_query_strips_cart_verbs_when_lead_segment_is_empty():
    # A message that opens with a connector leaves an empty first segment, so the query falls back
    # to the whole text with the cart/checkout verbs stripped out.
    assert "加入购物车" not in _search_query("，加入购物车")


def test_plan_returns_none_for_a_non_task_without_force():
    assert _planner().plan("你好呀", force=False) is None


def test_fallback_returns_none_when_forced_but_only_one_action():
    assert _planner().plan("推荐跑鞋", force=True) is None


def test_llm_plan_degrades_when_complete_raises():
    assert _planner(_PlanLLM(error=RuntimeError("boom")))._llm_plan("x", None, []) is None


def test_llm_plan_treats_null_response_as_no_plan():
    assert _planner(_PlanLLM(result="null"))._llm_plan("x", None, []) is None


def test_llm_plan_ignores_payload_without_a_steps_list():
    assert _planner(_PlanLLM(result='{"foo": 1}'))._llm_plan("x", None, []) is None


def test_llm_plan_rejects_a_single_step_plan():
    llm = _PlanLLM(result='{"steps":[{"action":"product_search","title":"找"}]}')
    assert _planner(llm)._llm_plan("x", None, []) is None


def test_fallback_appends_a_checkout_step():
    steps = _planner()._fallback_plan("推荐跑鞋然后加入购物车并下单").steps
    actions = [step.action for step in steps]
    assert "checkout" in actions and "cart_action" in actions


def test_fallback_skips_the_search_step_without_a_search_trigger():
    steps = _planner()._fallback_plan("对比这两款，把便宜的加入购物车").steps
    actions = [step.action for step in steps]
    assert "product_search" not in actions
    assert actions[0] == "select_products" and "comparison" in actions


def test_fallback_search_and_comparison_without_a_cart_step():
    steps = _planner()._fallback_plan("推荐两款跑鞋然后对比一下").steps
    actions = [step.action for step in steps]
    assert actions == ["product_search", "select_products", "comparison"]
    assert "cart_action" not in actions and "checkout" not in actions


# --- planner execution: checkout step, sort criteria, fail-closed -------------

def test_planner_executes_a_checkout_step_and_creates_an_order():
    llm = PlannerLLM(
        '{"steps":['
        '{"action":"product_search","title":"找耳机","query":"推荐蓝牙耳机"},'
        '{"action":"select_products","title":"选最便宜","criteria":"price_asc","count":1},'
        '{"action":"cart_action","title":"加入购物车","target":"selected_products","quantity":1},'
        '{"action":"checkout","title":"下单"}'
        ']}'
    )
    client = _client_with_planner_llm(llm)

    resp = client.post(
        "/api/chat",
        json={"session_id": "planner-checkout", "message": "帮我找蓝牙耳机，选最便宜的加入购物车并下单"},
    )

    assert resp.status_code == 200
    body = resp.json()
    steps = body["plan"]["steps"]
    assert steps[-1]["action"] == "checkout" and steps[-1]["status"] == "done"
    assert body["order"]["status"] == "awaiting_confirmation"


def test_planner_select_price_desc_adds_the_most_expensive_product():
    llm = PlannerLLM(
        '{"steps":['
        '{"action":"product_search","title":"找电脑","query":"推荐笔记本电脑"},'
        '{"action":"select_products","title":"选最贵","criteria":"price_desc","count":1},'
        '{"action":"cart_action","title":"加入购物车","target":"selected_products","quantity":1}'
        ']}'
    )
    client = _client_with_planner_llm(llm)

    resp = client.post(
        "/api/chat",
        json={"session_id": "planner-price-desc", "message": "帮我找笔记本电脑，把最贵的加入购物车"},
    )

    assert resp.status_code == 200
    body = resp.json()
    dearest = max(body["products"], key=lambda product: product["price"])
    assert body["cart"]["items"][0]["product_id"] == dearest["product_id"]


def test_planner_select_rating_desc_adds_the_highest_rated_product():
    llm = PlannerLLM(
        '{"steps":['
        '{"action":"product_search","title":"找跑鞋","query":"推荐跑步鞋"},'
        '{"action":"select_products","title":"选评分最高","criteria":"rating_desc","count":1},'
        '{"action":"cart_action","title":"加入购物车","target":"selected_products","quantity":1}'
        ']}'
    )
    client = _client_with_planner_llm(llm)

    resp = client.post(
        "/api/chat",
        json={"session_id": "planner-rating-desc", "message": "帮我找跑鞋，把评分最高的加入购物车"},
    )

    assert resp.status_code == 200
    body = resp.json()
    # the carted product must genuinely be the highest-rated one, i.e. the rating sort actually ran
    catalog = ProductCatalog.load(DATASET_ROOT)
    rated = {p["product_id"]: catalog.avg_rating(catalog.require(p["product_id"])) for p in body["products"]}
    carted = body["cart"]["items"][0]["product_id"]
    assert rated[carted] == max(rated.values())


def test_planner_cart_step_without_a_resolvable_target_fails_closed():
    # No search runs, so the cart step has no products to act on. It must fail the step rather than
    # crash or cart something arbitrary.
    llm = PlannerLLM(
        '{"steps":['
        '{"action":"select_products","title":"筛选","criteria":"price_asc","count":1},'
        '{"action":"cart_action","title":"加入购物车","target":"comparison_winner","quantity":1}'
        ']}'
    )
    client = _client_with_planner_llm(llm)

    resp = client.post(
        "/api/chat",
        json={"session_id": "planner-fail-closed", "message": "对比这两款，然后把更好的加入购物车"},
    )

    assert resp.status_code == 200
    steps = resp.json()["plan"]["steps"]
    assert steps[-1]["action"] == "cart_action" and steps[-1]["status"] == "failed"
    assert "缺少可执行的商品信息" in steps[-1]["summary"]
