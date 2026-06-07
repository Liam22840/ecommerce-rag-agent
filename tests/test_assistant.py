"""Tests for ShoppingAssistant orchestration: LLM answer/stream paths and memory."""

from __future__ import annotations

import json
from pathlib import Path

from server.assistant import ShoppingAssistant, _chunk_text
from server.catalog import CatalogHit, ProductCatalog
from server.config import Settings
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
    # p_beauty_008's own copy positively claims 烟酰胺; 007 does not.
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

    # First token already sent; we stop rather than re-emitting the fallback.
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
    """Records the query string handed to retrieve()."""

    def __init__(self):
        self.queries: list[str] = []

    def retrieve(self, query, filters, limit):
        self.queries.append(query)
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


# --- multi-round history + relative refinements --------------------------------

class _SeqLLM:
    """Intent LLM that returns a different canned JSON per call (turn-by-turn)."""

    available = True

    def __init__(self, responses):
        self._responses = list(responses)

    def complete(self, messages):
        return self._responses.pop(0) if self._responses else "{}"


class _ScriptedLLM:
    """Intent LLM whose next response is settable. Defaults to '{}' so untouched turns fall
    back to the rule parser; set `next_response` right before a turn that needs a canned JSON
    (useful when the value depends on ids produced by earlier turns, e.g. recall)."""

    available = True

    def __init__(self):
        self.next_response = "{}"
        self.calls: list = []

    def complete(self, messages):
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
    assert cream not in ids          # oldest, least-recently-shown — evicted
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
    llm = _SeqLLM([
        json.dumps({"intent_type": "product_search", "category": "美妆护肤", "sub_category": "面霜"}),
        json.dumps({"intent_type": "chitchat"}),
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


# --- required attributes: rank, don't gate; flag honestly when unmet -----------

def test_unmet_required_term_surfaces_closest_and_flags_honestly():
    # The LLM asks for 防水 on a 面霜 search; no face cream evidences it.
    intent_llm = _SeqLLM([
        json.dumps({"intent_type": "product_search", "category": "美妆护肤",
                    "sub_category": "面霜", "required_terms": ["防水"]}),
    ])
    assistant = _assistant(intent_llm=intent_llm)

    prepared = assistant.prepare("防水的面霜", session_id="s", top_k=3)

    assert prepared.products  # creams still surface — not silently dropped
    assert "没有" in prepared.grounded_answer and "防水" in prepared.grounded_answer  # honest line
    # no card claims the unmet attribute
    assert all("匹配防水需求" not in (product.matched_reason or "") for product in prepared.products)
