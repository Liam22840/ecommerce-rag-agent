"""Deterministic matching of free-text references to catalog products.

Pure helpers used by ComparisonService to map ordinal/name/brand references onto
concrete products. No catalog or LLM dependency, operates on product dicts.
"""

from __future__ import annotations

from typing import Any

from server.textutil import normalize


def _name_score(normalized_query: str, product: dict[str, Any]) -> float:
    title = normalize(product.get("title", ""))
    brand = normalize(product.get("brand", ""))
    sub_category = normalize(product.get("sub_category", ""))
    score = 0.0
    if title and title in normalized_query:
        score += 20
    if brand and brand in normalized_query:
        score += 8
    if sub_category and sub_category in normalized_query:
        score += 4
    for token in _title_tokens(title):
        if token in normalized_query:
            score += 2
    return score


def _best_ref_match(ref: str, pool: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best product in pool for a reference word (brand or product name). Returns None when
    several products tie for the top score (ambiguous) so the caller can fall back to a
    clarification instead of guessing."""
    normalized_ref = normalize(ref)
    if not normalized_ref:
        return None
    scored: list[tuple[float, dict[str, Any]]] = []
    for product in pool:
        # _name_score already rewards the brand/title appearing in the ref. Here we add the
        # reverse direction: a short ref being a fragment of the brand/title.
        score = _name_score(normalized_ref, product)
        brand = normalize(product.get("brand", ""))
        title = normalize(product.get("title", ""))
        if brand and normalized_ref in brand:
            score += 8
        if title and normalized_ref in title:
            score += 6
        if score > 0:
            scored.append((score, product))
    if not scored:
        return None
    top = max(score for score, _ in scored)
    winners = [product for score, product in scored if score == top]
    return winners[0] if len(winners) == 1 else None


def _title_tokens(normalized_title: str) -> list[str]:
    tokens = []
    for size in (6, 4, 3):
        for idx in range(0, max(0, len(normalized_title) - size + 1), size):
            token = normalized_title[idx: idx + size]
            if len(token) == size:
                tokens.append(token)
    return tokens[:8]


def _asks_for_current_two(query: str) -> bool:
    normalized = normalize(query)
    return any(term in normalized for term in ["这两款", "这两个", "两款哪个", "两个哪个", "前两个"])
