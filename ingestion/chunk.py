"""Turn a product JSON into a list of typed chunks ready for embedding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    chunk_id: str
    product_id: str
    chunk_type: str  # "summary" | "faq" | "review" | "image"
    text: str
    category: str
    sub_category: str
    brand: str
    base_price: float
    image_path: Optional[str] = None


def _make_chunk(
    product: dict,
    suffix: str,
    chunk_type: str,
    text: str,
    image_path: Optional[str] = None,
) -> Chunk:
    pid = product["product_id"]
    return Chunk(
        chunk_id=f"{pid}::{suffix}",
        product_id=pid,
        chunk_type=chunk_type,
        text=text,
        category=product["category"],
        sub_category=product["sub_category"],
        brand=product["brand"],
        base_price=float(product["base_price"]),
        image_path=image_path,
    )


def _summary_chunk(product: dict) -> Chunk:
    text = "\n".join([
        f"标题: {product['title']}",
        f"品牌: {product['brand']}",
        f"分类: {product['category']} > {product['sub_category']}",
        f"描述: {product['rag_knowledge']['marketing_description']}",
    ])
    return _make_chunk(product, suffix="summary", chunk_type="summary", text=text)


def _faq_chunks(product: dict) -> list[Chunk]:
    return [
        _make_chunk(
            product,
            suffix=f"faq::{i}",
            chunk_type="faq",
            text=f"问: {qa['question']}\n答: {qa['answer']}",
        )
        for i, qa in enumerate(product["rag_knowledge"]["official_faq"])
    ]


def _review_chunks(product: dict) -> list[Chunk]:
    return [
        _make_chunk(
            product,
            suffix=f"review::{i}",
            chunk_type="review",
            text=f"评分: {review['rating']}/5\n{review['content']}",
        )
        for i, review in enumerate(product["rag_knowledge"]["user_reviews"])
    ]


def _image_chunk(product: dict) -> Chunk:
    return _make_chunk(
        product,
        suffix="image",
        chunk_type="image",
        text="",
        image_path=product["image_path"],
    )


def extract_chunks(product: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunks.append(_summary_chunk(product))
    chunks.extend(_faq_chunks(product))
    chunks.extend(_review_chunks(product))
    chunks.append(_image_chunk(product))
    return chunks
