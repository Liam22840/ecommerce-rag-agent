"""Tests for ShoppingAssistant orchestration: LLM answer/stream paths and memory."""

from __future__ import annotations

from pathlib import Path

from server.assistant import (
    ShoppingAssistant,
    _chunk_text,
    _dedupe_ids,
)
from server.catalog import ProductCatalog
from server.config import Settings
from server.llm import ModelUnavailable
from server.retrieval import ProductRetriever


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


def _assistant(llm=None) -> ShoppingAssistant:
    settings = Settings(dataset_root=DATASET_ROOT, embedding_api_key=None, enable_vector_search=False)
    catalog = ProductCatalog.load(DATASET_ROOT)
    retriever = ProductRetriever(catalog, settings)
    retriever._startup_warning = None  # isolate degraded flag from the lexical-mode warning
    return ShoppingAssistant(catalog=catalog, retriever=retriever, llm=llm)


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

def test_remember_recent_products_dedupes_and_prepends():
    assistant = _assistant(llm=None)

    assistant._remember_recent_products("s", ["a", "b"])
    assistant._remember_recent_products("s", ["c", "b"])

    # Newest first, deduped across calls.
    assert assistant._recent_product_ids_by_session["s"] == ["c", "b", "a"]


def test_remember_recent_products_caps_at_ten():
    assistant = _assistant(llm=None)

    assistant._remember_recent_products("s", [f"id{n}" for n in range(15)])

    stored = assistant._recent_product_ids_by_session["s"]
    assert len(stored) == 10
    assert stored == [f"id{n}" for n in range(10)]


def test_remember_recent_products_ignores_empty_session_or_ids():
    assistant = _assistant(llm=None)

    assistant._remember_recent_products("", ["a"])
    assistant._remember_recent_products(None, ["a"])
    assistant._remember_recent_products("s", [])

    assert assistant._recent_product_ids_by_session == {}


def test_recent_product_ids_merges_client_and_stored():
    assistant = _assistant(llm=None)
    assistant._remember_recent_products("s", ["b", "a"])

    merged = assistant._recent_product_ids("s", ["x", "b"])

    assert merged == ["x", "b", "a"]


# --- small helpers -------------------------------------------------------------

def test_chunk_text_splits_into_fixed_windows():
    chunks = list(_chunk_text("abcdefgh", chunk_size=3))
    assert chunks == ["abc", "def", "gh"]


def test_chunk_text_empty_string_yields_nothing():
    assert list(_chunk_text("")) == []


def test_dedupe_ids_drops_empties_and_duplicates_preserving_order():
    assert _dedupe_ids(["a", "", "b", "a", "c", "b"]) == ["a", "b", "c"]
