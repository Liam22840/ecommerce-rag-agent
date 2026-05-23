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
