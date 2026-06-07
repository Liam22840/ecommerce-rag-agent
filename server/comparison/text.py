"""Product -> searchable text projection shared by dimension extraction and evidence.

Leaf module: depends only on server.textutil and the standard library, so both
dimensions.py and evidence.py can build on it without an import cycle.
"""

from __future__ import annotations

import re
from typing import Any

from server.textutil import normalize, trim


def _sku_text(product: dict[str, Any]) -> str:
    parts = []
    for sku in product.get("skus", []):
        properties = sku.get("properties", {})
        if isinstance(properties, dict):
            parts.extend(f"{key}{value}" for key, value in properties.items())
        if sku.get("price") is not None:
            parts.append(f"{sku['price']}元")
    return " ".join(str(part) for part in parts if part)


def _source_texts(product: dict[str, Any]) -> list[tuple[str, str, float]]:
    knowledge = product.get("rag_knowledge", {})
    sources = [
        ("商品标题", product.get("title", ""), 0.8),
        ("SKU", _sku_text(product), 1.0),
        ("商品描述", knowledge.get("marketing_description", ""), 1.4),
    ]
    for item in knowledge.get("official_faq", []):
        sources.append(("官方问答", f"{item.get('question', '')} {item.get('answer', '')}", 1.2))
    for item in knowledge.get("user_reviews", []):
        rating = item.get("rating")
        weight = 1.0 if not isinstance(rating, int | float) else max(0.6, min(1.3, rating / 4))
        sources.append(("用户评价", item.get("content", ""), weight))
    return sources


def _chunks(text: str) -> list[str]:
    return [chunk.strip() for chunk in re.split(r"[。！？!?；;\n]", text) if chunk.strip()]


def _strip_source(snippet: str) -> str:
    return snippet.split(": ", 1)[-1]


def _product_corpus(products: list[dict[str, Any]]) -> str:
    parts = []
    for product in products:
        parts.extend([product.get("title", ""), product.get("brand", ""), product.get("category", ""), product.get("sub_category", "")])
        parts.extend(text for _, text, _ in _source_texts(product))
    return normalize(" ".join(str(part) for part in parts if part))


def _product_evidence_for_llm(product: dict[str, Any]) -> dict[str, Any]:
    knowledge = product.get("rag_knowledge", {})
    faq = knowledge.get("official_faq", [])[:3]
    reviews = knowledge.get("user_reviews", [])[:4]
    return {
        "product_id": product.get("product_id"),
        "title": product.get("title"),
        "brand": product.get("brand"),
        "category": product.get("category"),
        "sub_category": product.get("sub_category"),
        "sku_summary": _sku_text(product),
        "marketing_description": trim(str(knowledge.get("marketing_description", "")), 700),
        "official_faq": [
            {
                "question": trim(str(item.get("question", "")), 140),
                "answer": trim(str(item.get("answer", "")), 260),
            }
            for item in faq
        ],
        "user_reviews": [
            {
                "rating": item.get("rating"),
                "content": trim(str(item.get("content", "")), 260),
            }
            for item in reviews
        ],
    }
