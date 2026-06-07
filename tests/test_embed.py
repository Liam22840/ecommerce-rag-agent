import base64
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from ingestion.cache import EmbeddingCache
from ingestion.chunk import Chunk
from ingestion.embed import DoubaoEmbedder


def _make_chunk(text: str = "hello", chunk_id: str = "p1::summary",
                chunk_type: str = "summary", image_path=None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        product_id="p1",
        chunk_type=chunk_type,
        text=text,
        category="美妆护肤",
        sub_category="精华",
        brand="测试",
        base_price=100.0,
        image_path=image_path,
    )


def _ok_response(vector: list[float]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "data": {"embedding": vector, "object": "embedding"},
        "model": "doubao-embedding-vision-251215",
    }
    return resp


def test_embeds_text_chunk_and_caches(tmp_path: Path, mocker):
    mock_post = mocker.patch("ingestion.embed.requests.post",
                             return_value=_ok_response([0.1, 0.2, 0.3]))
    cache = EmbeddingCache(tmp_path / "c.jsonl")
    embedder = DoubaoEmbedder(api_key="fake", cache=cache)

    vecs = embedder.embed_chunks([_make_chunk(text="hello")])

    assert vecs == [[0.1, 0.2, 0.3]]
    assert mock_post.call_count == 1
    # Second call should hit cache, not the API
    vecs2 = embedder.embed_chunks([_make_chunk(text="hello")])
    assert vecs2 == [[0.1, 0.2, 0.3]]
    assert mock_post.call_count == 1  # unchanged


def test_embeds_image_chunk_as_base64(tmp_path: Path, mocker):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG-ish bytes
    mock_post = mocker.patch("ingestion.embed.requests.post",
                             return_value=_ok_response([0.9]))
    cache = EmbeddingCache(tmp_path / "c.jsonl")
    embedder = DoubaoEmbedder(api_key="fake", cache=cache, dataset_root=tmp_path)

    chunk = _make_chunk(chunk_id="p1::image", chunk_type="image",
                       text="", image_path="x.jpg")
    vecs = embedder.embed_chunks([chunk])

    assert vecs == [[0.9]]
    sent_payload = mock_post.call_args.kwargs["json"]
    item = sent_payload["input"][0]
    assert item["type"] == "image_url"
    assert item["image_url"]["url"].startswith("data:image/jpeg;base64,")
    expected_b64 = base64.b64encode(b"\xff\xd8\xff\xd9").decode("ascii")
    assert expected_b64 in item["image_url"]["url"]


def test_retries_on_5xx_then_succeeds(tmp_path: Path, mocker):
    bad = MagicMock()
    bad.status_code = 503
    ok = _ok_response([0.5])
    mocker.patch("ingestion.embed.requests.post", side_effect=[bad, bad, ok])
    cache = EmbeddingCache(tmp_path / "c.jsonl")
    embedder = DoubaoEmbedder(api_key="fake", cache=cache, retry_sleep=0.0)

    vecs = embedder.embed_chunks([_make_chunk(text="t")])
    assert vecs == [[0.5]]


def test_hard_fails_on_4xx(tmp_path: Path, mocker):
    bad = MagicMock()
    bad.status_code = 401
    bad.text = '{"error":"unauthorized"}'
    mocker.patch("ingestion.embed.requests.post", return_value=bad)
    cache = EmbeddingCache(tmp_path / "c.jsonl")
    embedder = DoubaoEmbedder(api_key="fake", cache=cache, retry_sleep=0.0)

    with pytest.raises(RuntimeError, match="401"):
        embedder.embed_chunks([_make_chunk(text="t")])


def test_embed_text_calls_api_then_serves_from_cache(tmp_path: Path, mocker):
    mock_post = mocker.patch("ingestion.embed.requests.post",
                             return_value=_ok_response([0.4, 0.5]))
    cache = EmbeddingCache(tmp_path / "c.jsonl")
    embedder = DoubaoEmbedder(api_key="fake", cache=cache)

    assert embedder.embed_text("query") == [0.4, 0.5]
    assert mock_post.call_count == 1
    sent = mock_post.call_args.kwargs["json"]["input"][0]
    assert sent == {"type": "text", "text": "query"}

    # Repeat query is served from cache, no extra API call.
    assert embedder.embed_text("query") == [0.4, 0.5]
    assert mock_post.call_count == 1


def test_retries_on_network_exception_then_succeeds(tmp_path: Path, mocker):
    ok = _ok_response([0.7])
    mocker.patch("ingestion.embed.requests.post",
                 side_effect=[requests.ConnectionError("boom"), ok])
    cache = EmbeddingCache(tmp_path / "c.jsonl")
    embedder = DoubaoEmbedder(api_key="fake", cache=cache, retry_sleep=0.0)

    assert embedder.embed_chunks([_make_chunk(text="t")]) == [[0.7]]


def test_raises_after_exhausting_attempts_on_network_errors(tmp_path: Path, mocker):
    mocker.patch("ingestion.embed.requests.post",
                 side_effect=requests.ConnectionError("down"))
    cache = EmbeddingCache(tmp_path / "c.jsonl")
    embedder = DoubaoEmbedder(api_key="fake", cache=cache, retry_sleep=0.0, max_attempts=3)

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        embedder.embed_chunks([_make_chunk(text="t")])
