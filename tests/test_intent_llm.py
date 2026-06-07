"""Tests for the LLM-backed intent parser, its validation, and routing."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from server.app import create_app
from server.assistant import CHITCHAT_REPLY, ShoppingAssistant
from server.catalog import CatalogHit, ProductCatalog
from server.config import Settings
from server.intent import IntentParser, SearchFilters
from server.retrieval import ProductRetriever

DATASET_ROOT = Settings().dataset_root

CATEGORIES = {"美妆护肤", "数码电子", "服饰运动", "食品饮料"}
SUB_CATEGORIES = {"唇釉", "化妆水", "智能手机", "面霜", "真无线耳机", "洁面"}
BRANDS = {"华为", "雅诗兰黛", "兰蔻"}


class FakeLLM:
    """Stand-in for ArkChatClient: returns canned JSON (or raises)."""

    def __init__(self, response, available: bool = True):
        self._response = response
        self.available = available
        self.calls: list = []

    def complete(self, messages):
        self.calls.append(messages)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _parser(response, available: bool = True) -> IntentParser:
    return IntentParser(CATEGORIES, SUB_CATEGORIES, BRANDS, llm=FakeLLM(response, available))


# --- LLM mapping: the verified failing queries ---------------------------------

def test_llm_parses_price_range_and_maps_lipstick_synonym():
    resp = json.dumps({
        "intent_type": "product_search", "category": "美妆护肤", "sub_category": "唇釉",
        "min_price": 200, "max_price": 500,
    })
    f = _parser(resp).parse("给我找一个 200 到 500 元价格区间的口红")
    assert f.min_price == 200.0
    assert f.max_price == 500.0
    assert f.sub_category == "唇釉"
    assert f.raw_query == "给我找一个 200 到 500 元价格区间的口红"


def test_llm_maps_toner_synonym_and_negation():
    resp = json.dumps({"sub_category": "化妆水", "excluded_terms": ["酒精"]})
    f = _parser(resp).parse("不含酒精的爽肤水")
    assert f.sub_category == "化妆水"
    assert "酒精" in f.excluded_terms


def test_llm_parses_chinese_number_budget():
    resp = json.dumps({"sub_category": "智能手机", "max_price": 10000})
    f = _parser(resp).parse("推荐个手机，预算不超过1万")
    assert f.max_price == 10000.0
    assert f.sub_category == "智能手机"


def test_llm_classifies_first_purchase_as_product_search_not_comparison():
    resp = json.dumps({"intent_type": "product_search", "sub_category": "真无线耳机"})
    f = _parser(resp).parse("第一次买降噪耳机推荐哪个")
    assert f.intent_type == "product_search"  # despite the "第一" keyword


# --- Merge correctness ---------------------------------------------------------

def test_merge_keeps_rule_specs_when_llm_drops_them():
    resp = json.dumps({"sub_category": "面霜", "requested_specs": []})
    f = _parser(resp).parse("推荐50g适合敏感肌的保湿霜，cheaper is better")
    assert "50g" in f.requested_specs        # union with rule
    assert "敏感肌" in f.required_terms        # union with rule
    assert f.prefer_low_price is True          # OR with rule


def test_merge_validated_llm_category_survives_unmapped_subcategory():
    resp = json.dumps({"category": "美妆护肤", "sub_category": "唇釉"})
    f = _parser(resp).parse("推荐口红")
    assert f.sub_category == "唇釉"
    assert f.category == "美妆护肤"  # 唇釉 is not in SUB_CATEGORY_TO_CATEGORY; came from LLM


# --- Session context carry-over + rewrite --------------------------------------

def test_refinement_carries_previous_topic_with_llm():
    previous = SearchFilters(category="美妆护肤", sub_category="面霜", required_terms=["保湿"])
    resp = json.dumps({"intent_type": "product_search", "sort_by": "price_asc"})
    f = _parser(resp).parse("便宜点的", previous_filters=previous)
    assert f.sub_category == "面霜"        # inherited from previous turn
    assert f.category == "美妆护肤"
    assert "保湿" in f.required_terms       # sellpoints carried (unioned)
    assert f.prefer_low_price is True       # this turn's own constraint applied


def test_new_topic_drops_previous_context():
    previous = SearchFilters(category="美妆护肤", sub_category="面霜")
    resp = json.dumps({"intent_type": "product_search", "sub_category": "智能手机"})
    f = _parser(resp).parse("推荐个手机", previous_filters=previous)
    assert f.sub_category == "智能手机"     # current turn names its own topic
    assert f.category == "数码电子"          # previous 面霜 dropped


def test_deterministic_carry_over_without_llm():
    previous = SearchFilters(category="美妆护肤", sub_category="面霜")
    f = IntentParser(CATEGORIES, SUB_CATEGORIES, BRANDS).parse("便宜点的", previous_filters=previous)
    assert f.sub_category == "面霜"          # degraded-mode backstop still carries
    assert f.category == "美妆护肤"
    assert f.prefer_low_price is True


def test_rewritten_query_extracted():
    resp = json.dumps({"sub_category": "面霜", "rewritten_query": "更便宜的面霜"})
    f = _parser(resp).parse("便宜点的")
    assert f.rewritten_query == "更便宜的面霜"


def test_rewritten_query_empty_when_absent_or_nonstring():
    assert _parser(json.dumps({"sub_category": "面霜"})).parse("推荐面霜").rewritten_query == ""
    assert _parser(json.dumps({"rewritten_query": 123})).parse("推荐面霜").rewritten_query == ""


def test_exclude_seen_extracted_and_defaults_false():
    assert _parser(json.dumps({"sub_category": "面霜", "exclude_seen": True})).parse("换一批").exclude_seen is True
    assert _parser(json.dumps({"sub_category": "面霜"})).parse("推荐面霜").exclude_seen is False


def test_recall_product_ids_extracted_and_default_empty():
    resp = json.dumps({"intent_type": "product_search", "recall_product_ids": ["p_beauty_007"]})
    assert _parser(resp).parse("回到最开始那个").recall_product_ids == ["p_beauty_007"]
    assert _parser(json.dumps({"sub_category": "面霜"})).parse("推荐面霜").recall_product_ids == []


def test_compare_product_ids_extracted_and_default_empty():
    resp = json.dumps({"intent_type": "comparison", "compare_product_ids": ["p_beauty_007", "p_beauty_008"]})
    assert _parser(resp).parse("第一个和第二个哪个好").compare_product_ids == ["p_beauty_007", "p_beauty_008"]
    assert _parser(json.dumps({"sub_category": "面霜"})).parse("推荐面霜").compare_product_ids == []


def test_session_products_passed_to_llm_when_present():
    fake = FakeLLM(json.dumps({}))
    parser = IntentParser(CATEGORIES, SUB_CATEGORIES, BRANDS, llm=fake)
    parser.parse("回到最开始那个", session_products=[{"id": "p_beauty_007", "title": "薇诺娜", "price": 89}])
    assert "session_products" in fake.calls[0][1]["content"]


def test_intent_messages_include_recent_turns_only_when_provided():
    fake = FakeLLM(json.dumps({"sub_category": "面霜"}))
    parser = IntentParser(CATEGORIES, SUB_CATEGORIES, BRANDS, llm=fake)

    parser.parse("便宜点的", history=[{"query": "推荐面霜", "sub_category": "面霜"}])
    parser.parse("推荐面霜")

    with_history = fake.calls[0][1]["content"]
    without_history = fake.calls[1][1]["content"]
    assert "recent_turns" in with_history
    assert "recent_turns" not in without_history


# --- Validation / coercion edge cases ------------------------------------------

def test_invalid_json_falls_back_to_rules():
    f = _parser("this is not json").parse("推荐一款适合油皮的洗面奶")
    assert f.sub_category == "洁面"  # rule path still works
    assert f.intent_type == "product_search"


def test_llm_exception_falls_back_to_rules():
    f = _parser(RuntimeError("boom")).parse("200 元以下的蓝牙耳机")
    assert f.max_price == 200.0
    assert f.sub_category == "真无线耳机"


def test_empty_payload_falls_back_to_rule_values():
    f = _parser("{}").parse("推荐一款适合油皮的洗面奶")
    assert f.sub_category == "洁面"


def test_hallucinated_category_is_dropped():
    resp = json.dumps({"category": "奢侈品", "sub_category": "唇釉"})
    f = _parser(resp).parse("推荐口红")
    # 奢侈品 isn't a real category -> dropped; 唇釉 isn't in the backfill dict -> category stays None.
    assert f.category is None
    assert f.sub_category == "唇釉"


def test_excluded_brands_string_is_not_char_split():
    resp = json.dumps({"excluded_brands": "华为"})
    f = _parser(resp).parse("推荐手机")
    assert f.excluded_brands == ["华为"]
    assert "华" not in f.excluded_brands


def test_min_greater_than_max_is_swapped():
    resp = json.dumps({"min_price": 500, "max_price": 200})
    f = _parser(resp).parse("找个东西")
    assert f.min_price == 200.0
    assert f.max_price == 500.0


def test_negative_and_bool_prices_rejected():
    resp = json.dumps({"max_price": -100, "min_price": True})
    f = _parser(resp).parse("找个东西")
    assert f.max_price is None
    assert f.min_price is None


def test_bad_enum_falls_back_to_default():
    resp = json.dumps({"sort_by": "cheapest", "intent_type": "banter"})
    f = _parser(resp).parse("找个东西")
    assert f.sort_by == "relevance"
    assert f.intent_type == "product_search"


def test_comparison_shaped_payload_ignored():
    resp = json.dumps({"dimensions": [{"label": "降噪"}]})
    f = _parser(resp).parse("推荐一款适合油皮的洗面奶")
    assert f.sub_category == "洁面"  # known keys absent -> rule fallback values


def test_llm_extracts_compare_refs():
    resp = json.dumps({"intent_type": "comparison", "compare_refs": ["理肤泉", "薇诺娜"]})
    f = _parser(resp).parse("理肤泉和薇诺娜哪个更适合敏感肌")
    assert f.intent_type == "comparison"
    assert f.compare_refs == ["理肤泉", "薇诺娜"]


def test_compare_refs_default_empty_without_llm():
    f = IntentParser(CATEGORIES, SUB_CATEGORIES, BRANDS).parse("理肤泉和薇诺娜哪个好")
    assert f.compare_refs == []


def test_no_llm_uses_rule_path():
    f = IntentParser(CATEGORIES, SUB_CATEGORIES, BRANDS).parse("推荐一款适合油皮的洗面奶")
    assert f.sub_category == "洁面"


def test_unavailable_llm_is_not_called():
    parser = _parser(json.dumps({"sub_category": "唇釉"}), available=False)
    f = parser.parse("推荐一款适合油皮的洗面奶")
    assert f.sub_category == "洁面"  # rule path; LLM skipped despite a canned response


def test_json_embedded_in_prose_is_extracted():
    resp = '好的，解析结果如下：{"sub_category": "化妆水"} 以上。'
    f = _parser(resp).parse("爽肤水")
    assert f.sub_category == "化妆水"


def test_coerce_bool_string_true_sets_prefer_low_price():
    resp = json.dumps({"sub_category": "面霜", "prefer_low_price": "true"})
    f = _parser(resp).parse("推荐面霜")
    assert f.prefer_low_price is True


def test_sort_by_price_asc_forces_prefer_low_price():
    resp = json.dumps({"sub_category": "面霜", "sort_by": "price_asc", "prefer_low_price": False})
    f = _parser(resp).parse("推荐面霜")
    assert f.sort_by == "price_asc"
    assert f.prefer_low_price is True


# --- avg_rating + sort_by ------------------------------------------------------

def test_avg_rating_helper():
    product = {"rag_knowledge": {"user_reviews": [{"rating": 5}, {"rating": 3}, {"rating": 4}]}}
    assert ProductCatalog.avg_rating(product) == 4.0
    assert ProductCatalog.avg_rating({"rag_knowledge": {"user_reviews": []}}) == 0.0


def test_order_hits_by_rating_desc():
    settings = Settings(dataset_root=DATASET_ROOT, embedding_api_key=None, enable_vector_search=False)
    catalog = ProductCatalog.load(DATASET_ROOT)
    assistant = ShoppingAssistant(catalog=catalog, retriever=ProductRetriever(catalog, settings))
    low = CatalogHit(product={"rag_knowledge": {"user_reviews": [{"rating": 2}]}}, score=1.0)
    high = CatalogHit(product={"rag_knowledge": {"user_reviews": [{"rating": 5}]}}, score=1.0)
    ordered = assistant._order_hits([low, high], SearchFilters(sort_by="rating_desc"))
    assert ordered[0] is high


# --- App routing (intent_type via the parser) ----------------------------------

def _app(intent_response: str) -> TestClient:
    settings = Settings(dataset_root=DATASET_ROOT, chat_api_key=None, embedding_api_key=None,
                        enable_vector_search=False, enable_llm=False, enable_query_cache=False)
    catalog = ProductCatalog.load(DATASET_ROOT)
    retriever = ProductRetriever(catalog, settings)
    assistant = ShoppingAssistant(
        catalog=catalog, retriever=retriever, llm=None,
        intent_llm=FakeLLM(intent_response),  # type: ignore[arg-type]
    )
    return TestClient(create_app(settings=settings, assistant=assistant))


def test_chitchat_returns_friendly_reply_without_products():
    client = _app(json.dumps({"intent_type": "chitchat"}))
    body = client.post("/api/chat", json={"message": "你好"}).json()
    assert body["answer"] == CHITCHAT_REPLY
    assert body["products"] == []
    assert body["comparison"] is None


def test_first_purchase_routes_to_product_search_not_comparison():
    client = _app(json.dumps({"intent_type": "product_search", "sub_category": "真无线耳机"}))
    body = client.post("/api/chat", json={"message": "第一次买降噪耳机推荐哪个"}).json()
    assert body["comparison"] is None  # not hijacked into comparison


def test_intent_type_comparison_routes_to_comparison():
    client = _app(json.dumps({"intent_type": "comparison"}))
    body = client.post(
        "/api/chat",
        json={"message": "对比一下", "compare_product_ids": ["p_beauty_007", "p_beauty_012"]},
    ).json()
    assert body["comparison"] is not None


def test_llm_relative_cheaper_emits_tighter_max_price_with_carry():
    # The LLM, given the prior turn, resolves "便宜一点的" into a concrete tighter price.
    resp = json.dumps({
        "intent_type": "product_search", "category": "美妆护肤", "sub_category": "面霜",
        "max_price": 80, "prefer_low_price": True,
    })
    f = _parser(resp).parse(
        "便宜一点的",
        previous_filters=SearchFilters(category="美妆护肤", sub_category="面霜"),
    )
    assert f.sub_category == "面霜"
    assert f.max_price == 80.0
    assert f.prefer_low_price is True
