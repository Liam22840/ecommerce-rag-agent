import json
from pathlib import Path

import pytest

from ingestion.chunk import extract_chunks

SAMPLE_PRODUCT = (
    Path(__file__).parent.parent
    / "ecommerce_agent_dataset"
    / "1_美妆护肤"
    / "data"
    / "p_beauty_001.json"
)


@pytest.fixture
def product():
    with SAMPLE_PRODUCT.open(encoding="utf-8") as f:
        return json.load(f)


def test_returns_one_summary_chunk(product):
    chunks = extract_chunks(product)
    summaries = [c for c in chunks if c.chunk_type == "summary"]
    assert len(summaries) == 1
    s = summaries[0]
    assert s.chunk_id == "p_beauty_001::summary"
    assert s.product_id == "p_beauty_001"
    assert "雅诗兰黛" in s.text
    assert s.sub_category in s.text


def test_returns_one_chunk_per_faq(product):
    chunks = extract_chunks(product)
    faqs = [c for c in chunks if c.chunk_type == "faq"]
    assert len(faqs) == len(product["rag_knowledge"]["official_faq"])
    for i, c in enumerate(faqs):
        assert c.chunk_id == f"p_beauty_001::faq::{i}"
        assert c.product_id == "p_beauty_001"
        assert c.text.startswith("问:")
        assert "答:" in c.text


def test_returns_one_chunk_per_review(product):
    chunks = extract_chunks(product)
    reviews = [c for c in chunks if c.chunk_type == "review"]
    assert len(reviews) == len(product["rag_knowledge"]["user_reviews"])
    for i, c in enumerate(reviews):
        assert c.chunk_id == f"p_beauty_001::review::{i}"
        assert c.text.startswith("评分:")


def test_returns_one_image_chunk_with_path(product):
    chunks = extract_chunks(product)
    images = [c for c in chunks if c.chunk_type == "image"]
    assert len(images) == 1
    img = images[0]
    assert img.chunk_id == "p_beauty_001::image"
    assert img.text == ""
    assert img.image_path is not None
    assert img.image_path.endswith("p_beauty_001_live.jpg")


def test_metadata_is_attached_to_each_chunk(product):
    chunks = extract_chunks(product)
    for c in chunks:
        assert c.category == "美妆护肤"
        assert c.sub_category == "精华"
        assert c.brand == "雅诗兰黛"
        assert c.base_price == 720.0
