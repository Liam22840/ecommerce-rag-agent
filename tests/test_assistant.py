"""Tests for ShoppingAssistant orchestration: LLM answer/stream paths and memory."""

from __future__ import annotations

import json
from pathlib import Path

from server.assistant import ShoppingAssistant, _chunk_text, _looks_like_greeting, _reason
from server.catalog import CatalogHit, ProductCatalog
from server.config import Settings
from server.filter_cache import FilterCache
from server.intent import SearchFilters
from server.llm import ModelUnavailable
from server.retrieval import ProductRetriever, RetrievalResult
from server.textutil import dedupe_ids


DATASET_ROOT = Path(__file__).parent.parent / "ecommerce_agent_dataset"
PRODUCT_QUERY = "推荐一款适合油皮的洗面奶"


class FakeLLM:
    def __init__(
        self,
        *,
        available: bool = True,
        complete_result: str = "模型生成的回答",
        complete_error: Exception | None = None,
        stream_tokens: list[str] | None = None,
        stream_error: Exception | None = None,
        stream_error_after: int = 0,
    ):
        self.available = available
        self._complete_result = complete_result
        self._complete_error = complete_error
        self._stream_tokens = stream_tokens or []
        self._stream_error = stream_error
        self._stream_error_after = stream_error_after
        self.complete_calls = 0
        self.stream_calls = 0

    def complete(self, messages):
        self.complete_calls += 1
        if self._complete_error is not None:
            raise self._complete_error
        return self._complete_result

    def stream(self, messages):
        self.stream_calls += 1
        for idx, token in enumerate(self._stream_tokens):
            if self._stream_error is not None and idx == self._stream_error_after:
                raise self._stream_error
            yield token
        if self._stream_error is not None and self._stream_error_after >= len(self._stream_tokens):
            raise self._stream_error


def _settings(**overrides) -> Settings:
    base = dict(dataset_root=DATASET_ROOT, embedding_api_key=None, enable_vector_search=False)
    base.update(overrides)
    return Settings(**base)


def _assistant(llm=None, intent_llm=None, settings=None) -> ShoppingAssistant:
    settings = settings or _settings()
    catalog = ProductCatalog.load(DATASET_ROOT)
    retriever = ProductRetriever(catalog, settings)
    retriever._startup_warning = None  # isolate degraded flag from the lexical-mode warning
    return ShoppingAssistant(
        catalog=catalog, retriever=retriever, llm=llm, intent_llm=intent_llm, settings=settings
    )


def _hits(assistant, *product_ids) -> list[CatalogHit]:
    return [
        CatalogHit(product=assistant.catalog.require(pid), score=1.0, snippets=[], source="lexical")
        for pid in product_ids
    ]


def test_excluded_ids_uses_llm_judgment_as_primary():
    llm = FakeLLM(complete_result='{"exclude": ["p_beauty_007"]}')
    assistant = _assistant(llm=llm)
    hits = _hits(assistant, "p_beauty_007", "p_beauty_008")
    assert assistant._excluded_ids(hits, ["油腻"]) == {"p_beauty_007"}


def test_excluded_ids_ignores_ids_not_in_shortlist():
    llm = FakeLLM(complete_result='{"exclude": ["p_not_real", "p_beauty_008"]}')
    assistant = _assistant(llm=llm)
    hits = _hits(assistant, "p_beauty_007", "p_beauty_008")
    assert assistant._excluded_ids(hits, ["油腻"]) == {"p_beauty_008"}


def test_excluded_ids_falls_back_to_deterministic_when_llm_unavailable():
    assistant = _assistant(llm=FakeLLM(available=False))
    hits = _hits(assistant, "p_beauty_007", "p_beauty_008")
    # p_beauty_008's own copy positively claims 烟酰胺, 007 does not.
    assert assistant._excluded_ids(hits, ["烟酰胺"]) == {"p_beauty_008"}


def test_excluded_ids_falls_back_when_llm_returns_garbage():
    assistant = _assistant(llm=FakeLLM(complete_result="not json at all"))
    hits = _hits(assistant, "p_beauty_007", "p_beauty_008")
    assert assistant._excluded_ids(hits, ["烟酰胺"]) == {"p_beauty_008"}


# --- answer(): LLM availability matrix -----------------------------------------

def test_answer_uses_llm_text_when_available():
    llm = FakeLLM(complete_result="这是模型回答")
    assistant = _assistant(llm)

    response = assistant.answer(PRODUCT_QUERY, session_id=None, top_k=3)

    assert response.answer == "这是模型回答"
    assert llm.complete_calls == 1
    assert response.products  # grounded products still attached
    assert response.degraded is False
    assert response.warnings == []
    assert response.retrieval_source == "lexical"


def test_answer_falls_back_to_grounded_when_llm_unavailable_mid_call():
    llm = FakeLLM(complete_error=ModelUnavailable("quota exceeded"))
    assistant = _assistant(llm)

    response = assistant.answer(PRODUCT_QUERY, session_id=None, top_k=3)

    assert "商品库" in response.answer  # deterministic grounded narration
    assert response.degraded is True
    assert any("LLM unavailable" in w for w in response.warnings)
    assert "quota exceeded" in " ".join(response.warnings)


def test_answer_skips_llm_when_not_available():
    llm = FakeLLM(available=False, complete_result="should-not-be-used")
    assistant = _assistant(llm)

    response = assistant.answer(PRODUCT_QUERY, session_id=None, top_k=3)

    assert response.answer != "should-not-be-used"
    assert llm.complete_calls == 0
    assert "商品库" in response.answer


def test_answer_no_match_returns_grounded_no_hit_message():
    assistant = _assistant(llm=None)

    response = assistant.answer("200 元以下的蓝牙耳机有哪些？", session_id=None, top_k=3)

    assert response.products == []
    assert "没有在商品库中找到完全匹配" in response.answer


# --- stream_answer() -----------------------------------------------------------

def test_stream_answer_streams_llm_tokens_when_available():
    llm = FakeLLM(stream_tokens=["你", "好", "呀"])
    assistant = _assistant(llm)
    prepared = assistant.prepare(PRODUCT_QUERY, session_id=None, top_k=3)

    tokens = list(assistant.stream_answer(prepared))

    assert tokens == ["你", "好", "呀"]
    assert llm.stream_calls == 1


def test_stream_answer_falls_back_when_stream_fails_before_first_token():
    llm = FakeLLM(stream_tokens=[], stream_error=RuntimeError("stream down"), stream_error_after=0)
    assistant = _assistant(llm)
    prepared = assistant.prepare(PRODUCT_QUERY, session_id=None, top_k=3)

    tokens = list(assistant.stream_answer(prepared))

    # No tokens streamed -> deterministic grounded answer is chunked out instead.
    assert "".join(tokens) == prepared.grounded_answer
    assert len(tokens) > 1  # chunked


def test_stream_answer_stops_without_duplicate_when_stream_fails_midway():
    llm = FakeLLM(stream_tokens=["部分", "答案"], stream_error=RuntimeError("boom"), stream_error_after=1)
    assistant = _assistant(llm)
    prepared = assistant.prepare(PRODUCT_QUERY, session_id=None, top_k=3)

    tokens = list(assistant.stream_answer(prepared))

    # First token already sent, we stop rather than re-emitting the fallback.
    assert tokens == ["部分"]
    assert prepared.grounded_answer not in "".join(tokens)


def test_stream_answer_uses_grounded_when_no_llm():
    assistant = _assistant(llm=None)
    prepared = assistant.prepare(PRODUCT_QUERY, session_id=None, top_k=3)

    tokens = list(assistant.stream_answer(prepared))

    assert "".join(tokens) == prepared.grounded_answer


# --- session memory ------------------------------------------------------------

def test_recent_product_ids_orders_most_recent_first_preserving_within_turn_order():
    assistant = _assistant(llm=None)
    creams = [p.product_id for p in assistant.prepare("推荐面霜", session_id="s", top_k=3).products]
    phones = [p.product_id for p in assistant.prepare("推荐手机", session_id="s", top_k=3).products]

    recent = assistant._recent_product_ids("s", [])

    # Last turn's products come first, in display order (so comparison's 第一个/第二个 resolve),
    # then the earlier turn's products.
    assert recent[: len(phones)] == phones
    assert recent.index(phones[0]) < recent.index(creams[0])


def test_recent_product_ids_dedupes_when_product_reshown():
    assistant = _assistant(llm=None)
    assistant.prepare("推荐面霜", session_id="s", top_k=3)
    assistant.prepare("推荐面霜", session_id="s", top_k=3)  # same products shown again

    recent = assistant._recent_product_ids("s", [])
    assert len(recent) == len(set(recent))


def test_recent_product_ids_puts_client_ids_first():
    assistant = _assistant(llm=None)
    assistant.prepare("推荐面霜", session_id="s", top_k=3)

    merged = assistant._recent_product_ids("s", ["X"])
    assert merged[0] == "X"


def test_remember_shown_products_ignores_empty_session_or_products():
    assistant = _assistant(llm=None)

    assistant._remember_shown_products("", [])
    assistant._remember_shown_products(None, [])

    assert assistant._sessions == {}


def test_remember_turn_round_trips_and_skips_empty_session():
    assistant = _assistant(llm=None)
    filters = SearchFilters(sub_category="面霜", category="美妆护肤")

    assistant._remember_turn("s", "推荐面霜", filters, [])
    assert assistant._previous_filters("s") is filters

    assistant._remember_turn("", "推荐面霜", filters, [])
    assert assistant._previous_filters("") is None


def test_comparison_turn_does_not_overwrite_carried_filters():
    assistant = _assistant(llm=None)

    assistant.prepare("三百以内的面霜", session_id="s", top_k=3)
    before = assistant._previous_filters("s")
    assert before is not None and before.sub_category == "面霜"

    # A comparison turn reads recent products but must not overwrite the search context.
    assistant.prepare("第一个和第二个对比一下", session_id="s", top_k=3)
    assert assistant._previous_filters("s") is before


class _SpyRetriever:
    """Records the query string handed to retrieve() and the pre-warmed query."""

    def __init__(self):
        self.queries: list[str] = []
        self.prewarmed: list[str] = []
        self.limits: list[int] = []

    def prewarm_query(self, text):
        self.prewarmed.append(text)

    def retrieve(self, query, filters, limit, image_vector=None):
        self.queries.append(query)
        self.limits.append(limit)
        return RetrievalResult(hits=[], source="none")


def test_prepare_uses_rewritten_query_for_retrieval():
    intent_llm = FakeLLM(
        complete_result=json.dumps({"intent_type": "product_search", "rewritten_query": "更便宜的面霜"})
    )
    spy = _SpyRetriever()
    catalog = ProductCatalog.load(DATASET_ROOT)
    assistant = ShoppingAssistant(catalog=catalog, retriever=spy, llm=None, intent_llm=intent_llm)

    assistant.prepare("便宜点的", session_id="s", top_k=3)

    assert spy.queries == ["更便宜的面霜"]


def test_prepare_falls_back_to_raw_query_when_no_rewrite():
    spy = _SpyRetriever()
    catalog = ProductCatalog.load(DATASET_ROOT)
    assistant = ShoppingAssistant(catalog=catalog, retriever=spy, llm=None, intent_llm=None)

    assistant.prepare("三百以内的面霜", session_id="s", top_k=3)

    assert spy.queries == ["三百以内的面霜"]


def test_prepare_prewarms_the_query_embedding():
    spy = _SpyRetriever()
    catalog = ProductCatalog.load(DATASET_ROOT)
    assistant = ShoppingAssistant(catalog=catalog, retriever=spy, llm=None, intent_llm=None)

    assistant.prepare("三百以内的面霜", session_id="s", top_k=3)

    # The raw query is pre-warmed so the embed overlaps the intent call.
    assert spy.prewarmed == ["三百以内的面霜"]


def test_prepare_skips_prewarm_for_explicit_comparison():
    spy = _SpyRetriever()
    catalog = ProductCatalog.load(DATASET_ROOT)
    assistant = ShoppingAssistant(catalog=catalog, retriever=spy, llm=None, intent_llm=None)

    assistant.prepare("对比一下", session_id="s", top_k=3, compare_product_ids=["a", "b"])

    assert spy.prewarmed == []  # comparison never retrieves, so nothing to pre-warm


def test_opener_matches_the_route():
    a = _assistant()
    # A search opener names the product type — the label comes from the instant rule parse.
    assert "面霜" in a.opener("product_search", "三百以内的面霜")
    assert "面霜" in a.opener("product_search", "不要油腻的面霜")  # modifier negation still tailors
    # Negating the type itself yields no label, so a generic search opener (never offers 面霜).
    assert "面霜" not in a.opener("product_search", "不要面霜")
    # Other routes get a route-appropriate opener, not a search line.
    assert "对比" in a.opener("comparison", "随便")
    assert "购物车" in a.opener("cart_action", "把最贵的删了")
    # Chitchat gets no opener — its reply greets for itself.
    assert a.opener("chitchat", "你好") == ""
    # The opener leads with one of the varied acknowledgements (one of which itself contains a 顿号,
    # so match the prefix rather than splitting on the separator).
    leads = ("好的", "好嘞", "没问题", "收到", "好的呀", "嗯，好的")
    opener = a.opener("product_search", "面霜")
    assert any(opener.startswith(lead) for lead in leads)


_OPENER_LEADS = ("好的", "好嘞", "没问题", "收到", "好的呀", "嗯，好的")


def test_streaming_opener_flushes_instant_lead_then_route_tail():
    # 首Token: the instant lead is flushed before the router, then the route-specific tail completes
    # the line once the route is known (intent_llm off -> fallback route -> product_search).
    a = _assistant()
    tokens = [u for u in a.prepare_stream("三百以内的面霜", session_id="s", top_k=3) if isinstance(u, str)]
    assert len(tokens) == 2
    assert tokens[0].startswith(_OPENER_LEADS) and tokens[0].endswith("，")  # instant, route-neutral
    assert "面霜" in tokens[1]  # the tail names the product type once the route is known
    assert "".join(tokens).endswith("～\n")


def test_streaming_greeting_suppresses_instant_lead_and_opener():
    # A greeting routes to chitchat (the stub router answers it) and the greeting gate suppresses the
    # instant lead, so the turn streams no opener at all — the chitchat reply greets for itself.
    a = _assistant_with_intent(_ScriptedLLM())
    tokens = [u for u in a.prepare_stream("你好", session_id="s", top_k=3) if isinstance(u, str)]
    assert tokens == []


def test_greeting_gate_only_catches_short_greetings():
    assert _looks_like_greeting("你好") is True
    assert _looks_like_greeting("谢谢") is True
    assert _looks_like_greeting("hi") is True
    # A real shopping turn that merely opens with a greeting is too long to be gated, so it keeps its
    # instant lead; a plain shopping query has no greeting token at all.
    assert _looks_like_greeting("你好，推荐个面霜") is False
    assert _looks_like_greeting("推荐面霜") is False


def test_comparison_winner_cleared_on_new_search_but_survives_other_turns():
    a = _assistant()
    sid = "win"
    a._session(sid).last_winner_id = "p_beauty_007"
    # A turn that shows no new product list (e.g. a cart turn) must not drop the winner.
    a._remember_turn(sid, "q", SearchFilters(), [])
    assert a._session(sid).last_winner_id == "p_beauty_007"
    # A fresh product list invalidates it, so a later "更好的那个" can't resolve to a stale winner.
    a.prepare("推荐三款面霜", session_id=sid, top_k=3)
    assert a._session(sid).last_winner_id is None


# --- filter-keyed safe cache ---------------------------------------------------

def _cached_assistant(cache: FilterCache, llm) -> ShoppingAssistant:
    settings = _settings()
    catalog = ProductCatalog.load(DATASET_ROOT)
    retriever = ProductRetriever(catalog, settings)
    retriever._startup_warning = None
    return ShoppingAssistant(
        catalog=catalog, retriever=retriever, llm=llm, intent_llm=None,
        settings=settings, filter_cache=cache,
    )


def test_filter_cache_serves_paraphrases_from_one_entry(tmp_path):
    cache = FilterCache(tmp_path / "fc.jsonl", enabled=True)
    llm = FakeLLM(complete_result="缓存答案")
    assistant = _cached_assistant(cache, llm)

    # Session-free turns with the same parsed filters but different wording.
    first = assistant.answer("推荐洗面奶", session_id=None, top_k=3)
    second = assistant.answer("洗面奶推荐", session_id=None, top_k=3)

    assert first.answer == "缓存答案"
    assert second.answer == "缓存答案"
    assert llm.complete_calls == 1  # the paraphrase was served from the filter cache
    assert [p.product_id for p in second.products] == [p.product_id for p in first.products]


def test_filter_cache_key_separates_negation_and_ignores_phrasing_and_order():
    # Opposite price intent -> different keys (the LLM resolved the negation into structure).
    cheap = SearchFilters(sub_category="洁面", prefer_low_price=True)
    pricey = SearchFilters(sub_category="洁面", prefer_low_price=False)
    assert FilterCache.key(cheap, 3) != FilterCache.key(pricey, 3)

    # Same meaning -> same key regardless of raw text or list order.
    a = SearchFilters(sub_category="洁面", required_terms=["保湿", "控油"], raw_query="x")
    b = SearchFilters(sub_category="洁面", required_terms=["控油", "保湿"], raw_query="y")
    assert FilterCache.key(a, 3) == FilterCache.key(b, 3)

    # top_k is part of the key (different result-set size).
    assert FilterCache.key(a, 3) != FilterCache.key(a, 5)


def test_filter_cache_skips_session_context_turns():
    # A turn that carries seen-product context is never cacheable: same words can mean different
    # things once the conversation has state.
    base = SearchFilters(sub_category="洁面")
    assert FilterCache.eligible(base, recent_product_ids=[]) is True
    assert FilterCache.eligible(base, recent_product_ids=["p1"]) is False
    assert FilterCache.eligible(SearchFilters(sub_category="洁面", exclude_seen=True), []) is False
    assert FilterCache.eligible(SearchFilters(intent_type="chitchat"), []) is False


# --- multi-round history + relative refinements --------------------------------

_ROUTE_FROM_INTENT = {
    "product_search": "search", "comparison": "comparison", "chitchat": "chitchat",
    "cart_action": "cart", "checkout": "checkout", "planned_task": "plan",
}


def _route_reply_for(user_message: str, next_intent_json: str) -> str:
    """Derive the focused-router answer. A greeting routes to chitchat (which no longer runs the intent
    parse, so it consumes nothing); otherwise derive from the next queued intent so the router and the
    intent parse agree (the router peeks, the intent parse pops)."""
    if any(g in user_message for g in ("你好", "你是谁", "谢谢")):
        return json.dumps({"route": "chitchat"})
    try:
        intent = json.loads(next_intent_json).get("intent_type", "product_search")
    except (ValueError, AttributeError):
        intent = "product_search"
    return json.dumps({"route": _ROUTE_FROM_INTENT.get(intent, "search")})


class _SeqLLM:
    """Intent LLM that returns a different canned JSON per call (turn-by-turn)."""

    available = True

    def __init__(self, responses):
        self._responses = list(responses)

    def complete(self, messages):
        if "意图路由器" in messages[0]["content"]:
            return _route_reply_for(messages[1]["content"], self._responses[0] if self._responses else "{}")
        return self._responses.pop(0) if self._responses else "{}"


class _ScriptedLLM:
    """Intent LLM whose next response is settable. Defaults to '{}' so untouched turns fall
    back to the rule parser. Set `next_response` right before a turn that needs a canned JSON
    (useful when the value depends on ids produced by earlier turns, e.g. recall)."""

    available = True

    def __init__(self):
        self.next_response = "{}"
        self.calls: list = []

    def complete(self, messages):
        if "意图路由器" in messages[0]["content"]:
            return _route_reply_for(messages[1]["content"], self.next_response)
        self.calls.append(messages)
        return self.next_response


def _assistant_with_intent(intent_llm) -> ShoppingAssistant:
    return _assistant(intent_llm=intent_llm)


def test_remember_turn_records_and_caps_to_history_turns():
    assistant = _assistant(llm=None)  # default history_turns = 3
    for n in range(7):
        assistant._remember_turn("s", f"q{n}", SearchFilters(sub_category="面霜"), [])
    turns = assistant._sessions["s"].turns
    assert [turn.query for turn in turns] == [f"q{n}" for n in range(4, 7)]


def test_history_summaries_include_shown_prices():
    assistant = _assistant(llm=None)
    assistant.prepare("三百以内的面霜", session_id="s", top_k=3)
    history = assistant._history_summaries("s")
    assert history and history[-1]["sub_category"] == "面霜"
    assert history[-1]["shown"] and all("price" in item for item in history[-1]["shown"])


def test_previous_filters_reads_the_last_turn():
    assistant = _assistant(llm=None)
    assistant.prepare("推荐面霜", session_id="s", top_k=3)
    assistant.prepare("推荐手机", session_id="s", top_k=3)
    assert assistant._previous_filters("s").sub_category == "智能手机"


def test_exclude_seen_drops_already_shown_products():
    intent_llm = _SeqLLM([
        json.dumps({"intent_type": "product_search", "category": "美妆护肤", "sub_category": "面霜", "max_price": 100}),
        json.dumps({"intent_type": "product_search", "category": "美妆护肤", "sub_category": "面霜", "exclude_seen": True}),
    ])
    assistant = _assistant_with_intent(intent_llm)
    assistant.prepare("100以内的面霜", session_id="s", top_k=5)  # shows the two ≤100 creams (007, 008)

    prepared = assistant.prepare("换一批", session_id="s", top_k=5)
    ids = [product.product_id for product in prepared.products]
    assert "p_beauty_007" not in ids and "p_beauty_008" not in ids
    assert ids == ["p_beauty_012"]  # the only unseen 面霜


def test_result_status_no_improvement_when_results_already_seen():
    assistant = _assistant(llm=None)
    assistant.prepare("三百以内的面霜", session_id="s", top_k=3)
    repeat = assistant.prepare("三百以内的面霜", session_id="s", top_k=3)
    assert repeat.result_status == "no_improvement"
    assert "最匹配" in repeat.grounded_answer


def test_result_status_no_results_when_empty():
    assistant = _assistant(llm=None)
    prepared = assistant.prepare("便宜点的", session_id="fresh", top_k=3)
    assert prepared.products == []
    assert prepared.result_status == "no_results"


def test_result_status_no_cheaper_when_tightened_below_floor():
    # "便宜一点的" pushed the price ceiling under the cheapest item shown last turn and found
    # nothing -> a distinct status from a plain empty result, so the answer can say "已是最便宜".
    assistant = _assistant(llm=None)
    status = assistant._result_status(
        SearchFilters(prefer_low_price=True), products=[], recent_product_ids=[], prev_floor=50.0,
    )
    assert status == "no_cheaper"


def test_order_hits_sorts_by_price_descending():
    assistant = _assistant(llm=None)
    hits = _hits(assistant, "p_beauty_007", "p_beauty_008", "p_beauty_011")
    ordered = assistant._order_hits(hits, SearchFilters(sort_by="price_desc"))
    prices = [assistant._display_price(hit.product, SearchFilters(sort_by="price_desc")) for hit in ordered]
    assert prices == sorted(prices, reverse=True)


def test_excluded_ids_with_no_hits_returns_empty_set():
    assistant = _assistant(llm=FakeLLM())
    assert assistant._excluded_ids([], ["油腻"]) == set()


def test_excluded_ids_falls_back_when_llm_raises():
    # An exception from the judge (not just unparseable output) must degrade to the deterministic check.
    assistant = _assistant(llm=FakeLLM(complete_error=RuntimeError("judge down")))
    hits = _hits(assistant, "p_beauty_007", "p_beauty_008")
    assert assistant._excluded_ids(hits, ["烟酰胺"]) == {"p_beauty_008"}


def test_exclusion_terms_drop_violating_products_in_prepare():
    # End-to-end: an exclusion turn ("不要油腻") must remove products whose own copy claims the
    # excluded term, via the deterministic fallback when no answer LLM is configured.
    intent = _SeqLLM([json.dumps(
        {"intent_type": "product_search", "category": "美妆护肤", "sub_category": "面霜", "excluded_terms": ["油腻"]}
    )])
    assistant = _assistant_with_intent(intent)
    prepared = assistant.prepare("不要油腻的面霜", session_id="s", top_k=5)
    for product in prepared.products:
        full = assistant.catalog.require(product.product_id)
        assert assistant.catalog.violates_excluded(full, ["油腻"]) is False


def test_reason_credits_required_term_only_when_evidenced():
    assistant = _assistant(llm=None)
    evidenced = {
        "product_id": "x", "title": "补水面霜", "brand": "甲牌",
        "category": "美妆护肤", "sub_category": "面霜", "base_price": 50.0, "skus": [],
        "rag_knowledge": {"marketing_description": "深层补水锁水", "official_faq": [], "user_reviews": []},
    }
    hit = CatalogHit(product=evidenced, score=1.0, snippets=[], source="lexical")
    reason = _reason(hit, SearchFilters(required_terms=["保湿"]), assistant.catalog)
    assert "匹配保湿需求" in reason


def test_result_status_exhausted_when_all_seen_excluded():
    intent_llm = _SeqLLM([
        json.dumps({"intent_type": "product_search", "category": "美妆护肤", "sub_category": "面霜"}),
        json.dumps({"intent_type": "product_search", "category": "美妆护肤", "sub_category": "面霜", "exclude_seen": True}),
    ])
    assistant = _assistant_with_intent(intent_llm)
    assistant.prepare("推荐面霜", session_id="s", top_k=5)  # shows all 3 面霜

    prepared = assistant.prepare("换一批", session_id="s", top_k=5)
    assert prepared.products == []
    assert prepared.result_status == "exhausted"
    assert "没有更多" in prepared.grounded_answer


# --- session-wide product recall (backtracking) --------------------------------

def test_shown_products_accumulate_deduped_and_session_wide():
    assistant = _assistant(llm=None)
    assistant.prepare("推荐面霜", session_id="s", top_k=3)
    assistant.prepare("推荐手机", session_id="s", top_k=3)
    assistant.prepare("推荐面霜", session_id="s", top_k=3)  # repeat: no duplicates

    shown = assistant._session_products("s")
    ids = [item["id"] for item in shown]
    assert len(ids) == len(set(ids))  # deduped
    cats = {item["sub_category"] for item in shown}
    assert "面霜" in cats and "智能手机" in cats  # whole session, not just last turn
    # first-shown order: face creams (turn 1) precede phones (turn 2)
    first_phone = next(i for i, item in enumerate(shown) if item["sub_category"] == "智能手机")
    assert shown[0]["sub_category"] == "面霜" and first_phone > 0


def test_recall_returns_exact_products_by_id():
    intent_llm = FakeLLM(complete_result=json.dumps(
        {"intent_type": "product_search", "recall_product_ids": ["p_beauty_012"]}
    ))
    assistant = _assistant_with_intent(intent_llm)
    prepared = assistant.prepare("回到最开始那个", session_id="s", top_k=5)
    assert [product.product_id for product in prepared.products] == ["p_beauty_012"]


def test_recall_ignores_ids_not_in_catalog():
    intent_llm = FakeLLM(complete_result=json.dumps(
        {"intent_type": "product_search", "recall_product_ids": ["p_nope_999"]}
    ))
    assistant = _assistant_with_intent(intent_llm)
    prepared = assistant.prepare("回到那个", session_id="s", top_k=5)
    # Invalid id is dropped, so it falls through to normal retrieval (never returns the bad id).
    assert all(product.product_id != "p_nope_999" for product in prepared.products)


# --- photo-find (拍照找货) -------------------------------------------------------

class _PhotoRetriever:
    """Fake retriever for photo turns: returns a fixed product and records calls."""

    def __init__(self, catalog, product_id, vector=None):
        self._catalog = catalog
        self._product_id = product_id
        self._vector = vector if vector is not None else [0.5, 0.5]
        self.retrieve_calls = []
        self.embed_calls = []

    def prewarm_query(self, text):
        pass

    def embed_image(self, image_bytes):
        self.embed_calls.append(image_bytes)
        return self._vector

    def retrieve(self, query, filters, limit, image_vector=None):
        self.retrieve_calls.append((query, filters, image_vector))
        hit = CatalogHit(product=self._catalog.require(self._product_id), score=1.0, source="vector")
        return RetrievalResult(hits=[hit], source="vector")


def _photo_assistant(intent_llm, product_id="p_clothes_007"):
    catalog = ProductCatalog.load(DATASET_ROOT)
    retriever = _PhotoRetriever(catalog, product_id)
    assistant = ShoppingAssistant(
        catalog=catalog, retriever=retriever, llm=None, intent_llm=intent_llm, settings=_settings()
    )
    return assistant, retriever


def test_photo_turn_embeds_image_and_retrieves_with_vector():
    intent_llm = FakeLLM(complete_result=json.dumps({
        "intent_type": "product_search", "category": "服饰运动", "sub_category": "跑步鞋",
        "vision_description": "白色跑鞋", "vision_confidence": "high",
    }))
    assistant, retriever = _photo_assistant(intent_llm)

    prepared = assistant.prepare("找同款", session_id="s", top_k=3, image_bytes=b"\xff\xd8\xff\xd9")

    assert retriever.embed_calls == [b"\xff\xd8\xff\xd9"]
    query, filters, image_vector = retriever.retrieve_calls[0]
    assert image_vector == [0.5, 0.5]
    assert query == "白色跑鞋"
    assert [p.product_id for p in prepared.products] == ["p_clothes_007"]
    assert prepared.filters.intent_type == "product_search"


def test_photo_turn_streams_an_instant_lead_before_the_slow_search():
    # 首Token: the photo path flushes an instant opener before the (slow) embed + VLM, then the result.
    intent_llm = FakeLLM(complete_result=json.dumps({"intent_type": "product_search", "vision_confidence": "high"}))
    assistant, _ = _photo_assistant(intent_llm)
    items = list(assistant.prepare_stream("找同款", session_id="s", top_k=3, image_bytes=b"\xff\xd8"))
    assert isinstance(items[0], str) and "识别图片" in items[0]  # instant lead first
    assert any(not isinstance(it, str) for it in items)  # the prepared result follows


def test_photo_low_confidence_drops_category_gate():
    intent_llm = FakeLLM(complete_result=json.dumps({
        "intent_type": "product_search", "category": "服饰运动", "sub_category": "跑步鞋",
        "vision_description": "某种外套", "vision_confidence": "low",
    }))
    assistant, retriever = _photo_assistant(intent_llm)

    assistant.prepare("找同款", session_id="s", top_k=3, image_bytes=b"\xff\xd8\xff\xd9")

    _, filters, _ = retriever.retrieve_calls[0]
    assert filters.category is None and filters.sub_category is None


def test_photo_turn_degrades_to_lexical_when_embed_fails():
    intent_llm = FakeLLM(complete_result=json.dumps({"intent_type": "product_search", "vision_confidence": "high"}))
    assistant, retriever = _photo_assistant(intent_llm)
    retriever.embed_image = lambda b: (_ for _ in ()).throw(RuntimeError("embed down"))

    prepared = assistant.prepare("找同款", session_id="s", top_k=3, image_bytes=b"\xff\xd8")

    _, _, image_vector = retriever.retrieve_calls[0]
    assert image_vector is None
    assert prepared.products


def test_photo_turn_records_session_memory_for_followup():
    intent_llm = FakeLLM(complete_result=json.dumps({"intent_type": "product_search", "vision_confidence": "high"}))
    assistant, _ = _photo_assistant(intent_llm)

    assistant.prepare("找同款", session_id="s", top_k=3, image_bytes=b"\xff\xd8")
    shown = assistant._session_products("s")
    assert shown and shown[0]["id"] == "p_clothes_007"


# --- verification fixes: sort over-fetch, plan-cart pool, session history ------

def test_sort_query_overfetches_full_category_before_truncating():
    # A price/rating sort must retrieve the whole filtered category, not just the relevance-top-k,
    # or the true cheapest/highest-rated gets truncated away before _order_hits sorts.
    from server.assistant import _SORT_CANDIDATE_POOL
    intent_llm = _SeqLLM([json.dumps(
        {"intent_type": "product_search", "category": "数码电子", "sub_category": "笔记本电脑", "sort_by": "price_desc"}
    )])
    spy = _SpyRetriever()
    catalog = ProductCatalog.load(DATASET_ROOT)
    assistant = ShoppingAssistant(catalog=catalog, retriever=spy, llm=None, intent_llm=intent_llm)
    assistant.prepare("高端的笔记本电脑", session_id="s", top_k=3)
    assert spy.limits[0] >= _SORT_CANDIDATE_POOL  # over-fetched, not just top_k=3


def test_relevance_query_does_not_overfetch():
    spy = _SpyRetriever()
    catalog = ProductCatalog.load(DATASET_ROOT)
    assistant = ShoppingAssistant(catalog=catalog, retriever=spy, llm=None, intent_llm=None)
    assistant.prepare("推荐笔记本电脑", session_id="s", top_k=3)
    assert spy.limits[0] == 3  # no sort -> normal top_k, no over-fetch


def test_pool_with_ids_guarantees_the_targets_resolve():
    assistant = _assistant(llm=None)
    pool = assistant._pool_with_ids(["p_clothes_007"], existing=None)
    assert any(entry["id"] == "p_clothes_007" for entry in pool)
    # existing entries are preserved and not duplicated
    pool2 = assistant._pool_with_ids(["p_clothes_007"], existing=[{"id": "p_clothes_007"}])
    assert len(pool2) == 1


def test_has_session_history_only_true_after_a_recorded_turn():
    assistant = _assistant(llm=None)
    assert assistant.has_session_history("s") is False  # no entry, doesn't create one
    assistant.prepare("推荐面霜", session_id="s", top_k=3)
    assert assistant.has_session_history("s") is True
    assert assistant.has_session_history(None) is False


# --- small helpers -------------------------------------------------------------

def test_chunk_text_splits_into_fixed_windows():
    chunks = list(_chunk_text("abcdefgh", chunk_size=3))
    assert chunks == ["abc", "def", "gh"]


def test_chunk_text_empty_string_yields_nothing():
    assert list(_chunk_text("", chunk_size=18)) == []


def test_dedupe_ids_drops_empties_and_duplicates_preserving_order():
    assert dedupe_ids(["a", "", "b", "a", "c", "b"]) == ["a", "b", "c"]


# --- config knobs actually drive memory behaviour ------------------------------

def test_history_turns_setting_controls_the_window():
    assistant = _assistant(settings=_settings(history_turns=1))
    for n in range(4):
        assistant._remember_turn("s", f"q{n}", SearchFilters(sub_category="面霜"), [])
    assert [turn.query for turn in assistant._sessions["s"].turns] == ["q3"]


def test_shown_summary_cap_limits_items_per_turn():
    assistant = _assistant(settings=_settings(shown_summary_cap=1))
    assistant.prepare("推荐面霜", session_id="s", top_k=3)  # 3 creams shown
    history = assistant._history_summaries("s")
    assert len(history[-1]["shown"]) == 1


def test_recent_products_cap_limits_derived_recency():
    assistant = _assistant(settings=_settings(recent_products_cap=2))
    assistant.prepare("推荐手机", session_id="s", top_k=5)  # several phones shown
    assert len(assistant._recent_product_ids("s", [])) == 2


def test_session_products_cap_evicts_least_recently_shown():
    assistant = _assistant(settings=_settings(session_products_cap=2))
    cream = assistant.prepare("推荐面霜", session_id="s", top_k=1).products[0].product_id
    phone = assistant.prepare("推荐手机", session_id="s", top_k=1).products[0].product_id
    shoe = assistant.prepare("推荐跑鞋", session_id="s", top_k=1).products[0].product_id

    ids = [item["id"] for item in assistant._session_products("s")]
    assert cream not in ids          # oldest, least-recently-shown, evicted
    assert phone in ids and shoe in ids


def test_reshowing_a_product_saves_it_from_eviction():
    assistant = _assistant(settings=_settings(session_products_cap=2))
    cream = assistant.prepare("推荐面霜", session_id="s", top_k=1).products[0].product_id
    phone = assistant.prepare("推荐手机", session_id="s", top_k=1).products[0].product_id
    assistant.prepare("推荐面霜", session_id="s", top_k=1)  # re-show cream -> now most recent
    shoe = assistant.prepare("推荐跑鞋", session_id="s", top_k=1).products[0].product_id

    ids = [item["id"] for item in assistant._session_products("s")]
    assert cream in ids and shoe in ids
    assert phone not in ids          # phone is now the least-recently-shown


# --- backtracking depth + recall interactions ----------------------------------

def test_backtracking_recalls_a_product_beyond_the_turn_window():
    llm = _ScriptedLLM()  # "{}" -> rule parses the searches
    assistant = _assistant(intent_llm=llm)
    for query in ["推荐面霜", "推荐手机", "推荐耳机", "推荐跑鞋"]:
        assistant.prepare(query, session_id="s", top_k=2)

    # session_products is newest-first, so the earliest-shown 面霜 is at the tail.
    first = assistant._session_products("s")[-1]
    assert first["sub_category"] == "面霜"
    assert len(assistant._sessions["s"].turns) == 3  # its turn fell out of the window

    llm.next_response = json.dumps(
        {"intent_type": "product_search", "recall_product_ids": [first["id"]]}
    )
    prepared = assistant.prepare("回到我最开始看的那个面霜", session_id="s", top_k=5)
    assert [product.product_id for product in prepared.products] == [first["id"]]


def test_refining_after_a_recall_carries_from_the_recall_turn():
    llm = _ScriptedLLM()
    assistant = _assistant(intent_llm=llm)
    assistant.prepare("推荐面霜", session_id="s", top_k=3)
    assistant.prepare("推荐手机", session_id="s", top_k=3)
    cream = assistant._session_products("s")[-1]["id"]  # newest-first: earliest 面霜 is last

    llm.next_response = json.dumps(
        {"intent_type": "product_search", "category": "美妆护肤", "sub_category": "面霜",
         "recall_product_ids": [cream]}
    )
    assistant.prepare("回到那个面霜", session_id="s", top_k=5)  # records a 面霜 turn

    llm.next_response = json.dumps({"intent_type": "product_search", "sort_by": "price_asc", "prefer_low_price": True})
    refined = assistant.prepare("便宜点的", session_id="s", top_k=5)
    assert refined.filters.sub_category == "面霜"  # carried off the recall turn, not 手机


# --- multi-feature journeys + cross-branch memory ------------------------------

def test_chitchat_turn_preserves_all_memory():
    # The chitchat turn no longer runs the intent parse, so it consumes no queued response (the router
    # recognises the greeting by query); only the two searches pop.
    llm = _SeqLLM([
        json.dumps({"intent_type": "product_search", "category": "美妆护肤", "sub_category": "面霜"}),
        json.dumps({"intent_type": "product_search", "sort_by": "price_asc"}),
    ])
    assistant = _assistant(intent_llm=llm)
    assistant.prepare("推荐面霜", session_id="s", top_k=3)
    shown_before = list(assistant._session_products("s"))

    assistant.prepare("你好你是谁", session_id="s", top_k=3)  # chitchat
    assert assistant._session_products("s") == shown_before        # untouched
    assert assistant._previous_filters("s").sub_category == "面霜"  # window intact

    after = assistant.prepare("便宜点的", session_id="s", top_k=3)
    assert after.filters.sub_category == "面霜"


def test_comparison_updates_product_memory_but_not_the_turn_window():
    assistant = _assistant(llm=None)  # rule routes "第一个和第二个" to comparison
    assistant.prepare("推荐面霜", session_id="s", top_k=3)
    turns_before = len(assistant._sessions["s"].turns)

    assistant.prepare("第一个和第二个哪个更保湿", session_id="s", top_k=3)
    assert len(assistant._sessions["s"].turns) == turns_before          # no search turn added
    assert assistant._previous_filters("s").sub_category == "面霜"        # refinement context unchanged


def test_exclude_seen_with_backstop_carry_drops_seen_within_carried_category():
    llm = _SeqLLM([
        json.dumps({"intent_type": "product_search", "category": "美妆护肤", "sub_category": "面霜", "max_price": 100}),
        json.dumps({"intent_type": "product_search", "exclude_seen": True}),  # no category -> backstop carries
    ])
    assistant = _assistant(intent_llm=llm)
    assistant.prepare("100以内的面霜", session_id="s", top_k=5)  # shows 007, 008

    prepared = assistant.prepare("换一批", session_id="s", top_k=5)
    assert prepared.filters.sub_category == "面霜"  # carried by the deterministic backstop
    ids = [product.product_id for product in prepared.products]
    assert "p_beauty_007" not in ids and "p_beauty_008" not in ids
    assert ids == ["p_beauty_012"]


def test_two_sessions_do_not_share_any_memory():
    assistant = _assistant(intent_llm=None)
    assistant.prepare("推荐面霜", session_id="A", top_k=3)
    assistant.prepare("推荐手机", session_id="B", top_k=3)

    a_cats = {item["sub_category"] for item in assistant._session_products("A")}
    b_cats = {item["sub_category"] for item in assistant._session_products("B")}
    assert a_cats == {"面霜"} and b_cats == {"智能手机"}
    assert assistant._previous_filters("A").sub_category == "面霜"
    assert assistant._previous_filters("B").sub_category == "智能手机"


def test_degraded_mode_carries_category_across_a_refinement_chain():
    assistant = _assistant(llm=None)  # no intent LLM -> deterministic backstop only
    assistant.prepare("推荐面霜", session_id="s", top_k=3)
    first = assistant.prepare("便宜点的", session_id="s", top_k=3)
    second = assistant.prepare("再给我看看别的价位", session_id="s", top_k=3)
    assert first.filters.sub_category == "面霜"
    assert second.filters.sub_category == "面霜"


# --- required attributes: rank, don't gate, flag honestly when unmet -----------

def test_unmet_required_term_surfaces_closest_and_flags_honestly():
    # The LLM asks for 防水 on a 面霜 search, no face cream evidences it.
    intent_llm = _SeqLLM([
        json.dumps({"intent_type": "product_search", "category": "美妆护肤",
                    "sub_category": "面霜", "required_terms": ["防水"]}),
    ])
    assistant = _assistant(intent_llm=intent_llm)

    prepared = assistant.prepare("防水的面霜", session_id="s", top_k=3)

    assert prepared.products  # creams still surface, not silently dropped
    assert "没有" in prepared.grounded_answer and "防水" in prepared.grounded_answer  # honest line
    # no card claims the unmet attribute
    assert all("匹配防水需求" not in (product.matched_reason or "") for product in prepared.products)
