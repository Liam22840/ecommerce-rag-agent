"""Unit tests for the exact-match hot-query cache."""

from __future__ import annotations

from server.query_cache import QueryCache


def _response(answer: str = "hi", intent_type: str = "product_search") -> dict:
    return {
        "answer": answer,
        "products": [],
        "comparison": None,
        "session_id": None,
        "intent": {"intent_type": intent_type},
        "retrieval_source": "lexical",
        "degraded": False,
        "warnings": [],
    }


def test_key_collapses_spacing_and_punctuation(tmp_path):
    cache = QueryCache(tmp_path / "q.jsonl")
    assert cache.key("推荐敏感肌面霜", 3) == cache.key("推荐 敏感肌面霜。", 3)
    assert QueryCache.key("abc", 3) != QueryCache.key("abc", 5)  # top_k is part of the key


def test_put_then_get_round_trip(tmp_path):
    cache = QueryCache(tmp_path / "q.jsonl")
    key = cache.key("q", 3)
    assert cache.get(key) is None
    cache.put(key, _response("answer-1"))
    assert cache.get(key)["answer"] == "answer-1"


def test_persists_and_reloads_from_jsonl(tmp_path):
    path = tmp_path / "q.jsonl"
    first = QueryCache(path)
    key = first.key("q", 3)
    first.put(key, _response("persisted"))
    # a fresh instance reads the same file
    second = QueryCache(path)
    assert second.get(key)["answer"] == "persisted"


def test_load_skips_blank_and_malformed_lines(tmp_path):
    # A torn write (process killed mid-append) or a stray blank line must not break startup:
    # the loader skips them and keeps every well-formed entry.
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"key": "a", "response": {"answer": "A"}}\n'
        "\n"
        "{not valid json\n"
        '{"key": "b", "response": "not-a-dict"}\n'  # response must be a dict to be kept
        '{"key": "c", "response": {"answer": "C"}}\n',
        encoding="utf-8",
    )
    cache = QueryCache(path)
    assert cache.get("a")["answer"] == "A"
    assert cache.get("b") is None
    assert cache.get("c")["answer"] == "C"


def test_lru_cap_drops_oldest(tmp_path):
    cache = QueryCache(tmp_path / "q.jsonl", max_entries=2)
    cache.put("a", _response("A"))
    cache.put("b", _response("B"))
    cache.put("c", _response("C"))  # over cap -> evicts the oldest ("a")
    assert cache.get("a") is None
    assert cache.get("b")["answer"] == "B"
    assert cache.get("c")["answer"] == "C"


def test_eligible_only_without_context():
    assert QueryCache.eligible([], []) is True
    assert QueryCache.eligible(["p1"], []) is False
    assert QueryCache.eligible([], ["p2"]) is False


def test_storeable_only_product_search():
    assert QueryCache.storeable("product_search") is True
    assert QueryCache.storeable("chitchat") is False
    assert QueryCache.storeable("comparison") is False


def test_disabled_cache_never_stores_or_serves(tmp_path):
    cache = QueryCache(tmp_path / "q.jsonl", enabled=False)
    key = cache.key("q", 3)
    cache.put(key, _response())
    assert cache.get(key) is None
    assert not (tmp_path / "q.jsonl").exists()
