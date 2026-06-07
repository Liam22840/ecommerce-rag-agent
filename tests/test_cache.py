import threading
from pathlib import Path

from ingestion.cache import EmbeddingCache, text_key, image_key


def test_text_key_is_stable_and_distinct():
    a = text_key("hello")
    b = text_key("hello")
    c = text_key("hello!")
    assert a == b
    assert a != c
    assert a.startswith("text:")


def test_image_key_uses_byte_hash():
    a = image_key(b"image-bytes-1")
    b = image_key(b"image-bytes-1")
    c = image_key(b"image-bytes-2")
    assert a == b
    assert a != c
    assert a.startswith("image:")


def test_cache_miss_then_hit_then_persist(tmp_path: Path):
    cache_file = tmp_path / "cache.jsonl"
    c1 = EmbeddingCache(cache_file)
    assert c1.get("k1") is None

    c1.put("k1", [0.1, 0.2, 0.3])
    assert c1.get("k1") == [0.1, 0.2, 0.3]

    # Reload from disk
    c2 = EmbeddingCache(cache_file)
    assert c2.get("k1") == [0.1, 0.2, 0.3]


def test_cache_handles_multiple_entries(tmp_path: Path):
    cache_file = tmp_path / "cache.jsonl"
    c = EmbeddingCache(cache_file)
    c.put("a", [1.0])
    c.put("b", [2.0])
    c.put("c", [3.0])
    reloaded = EmbeddingCache(cache_file)
    assert reloaded.get("a") == [1.0]
    assert reloaded.get("b") == [2.0]
    assert reloaded.get("c") == [3.0]


def test_load_skips_blank_lines(tmp_path: Path):
    # Append-only writes plus an editor or a crash can leave blank lines; loading must skip them.
    cache_file = tmp_path / "cache.jsonl"
    cache_file.write_text(
        '{"key": "a", "vector": [1.0]}\n'
        "\n"
        "   \n"
        '{"key": "b", "vector": [2.0]}\n',
        encoding="utf-8",
    )
    cache = EmbeddingCache(cache_file)
    assert cache.get("a") == [1.0]
    assert cache.get("b") == [2.0]


def test_concurrent_put_and_get_is_thread_safe(tmp_path: Path):
    # Speculative pre-warm + the request thread + FastAPI's threadpool all share one cache, so
    # concurrent put/get must not corrupt the dict or interleave a JSONL line.
    cache = EmbeddingCache(tmp_path / "cache.jsonl")
    n = 200

    def writer() -> None:
        for i in range(n):
            cache.put(f"k{i}", [float(i)])

    def reader() -> None:
        for i in range(n):
            cache.get(f"k{i}")  # may be None or the value; just must never raise

    threads = [threading.Thread(target=writer) for _ in range(4)] + [
        threading.Thread(target=reader) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every key is present with the right value, and the file reloads cleanly (no torn lines).
    for i in range(n):
        assert cache.get(f"k{i}") == [float(i)]
    reloaded = EmbeddingCache(tmp_path / "cache.jsonl")
    for i in range(n):
        assert reloaded.get(f"k{i}") == [float(i)]
