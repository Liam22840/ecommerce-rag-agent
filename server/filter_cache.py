"""Negation-safe answer cache keyed on the parsed intent rather than the raw text.

The intent LLM already turns a messy query into a clean SearchFilters object, which *is* the
meaning of the query. Keying the cache on those filters collapses every phrasing of the same
intent ("便宜的洗面奶" / "平价一点的洁面" / "实惠的洗面奶") onto one entry, so the hit rate is far
higher than the exact-text QueryCache. It is safe where an embedding-similarity cache is not:
the LLM resolves negation into structure before the key exists, so "便宜"→prefer_low_price:true
and "不便宜"→prefer_low_price:false get different keys and cannot collide. The match itself is
exact on every field — price and numeric bounds only hit if identical, never "close enough".

Storage (append-only JSONL, in-memory LRU) is inherited from QueryCache; only the key and
eligibility differ.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from server.query_cache import QueryCache
from server.textutil import normalize

if TYPE_CHECKING:
    from server.intent import SearchFilters


# The fields that define WHAT to retrieve and HOW to rank — the meaning of the query. Session/raw
# fields (raw_query, rewritten_query, exclude_seen, recall_product_ids, compare_product_ids) are
# deliberately excluded: they are about phrasing or conversation state, not the intent itself.
_SCALAR_FIELDS = ("max_price", "min_price", "category", "sub_category", "brand", "prefer_low_price", "sort_by")
_LIST_FIELDS = ("required_terms", "requested_specs", "excluded_brands", "excluded_terms")


class FilterCache(QueryCache):
    @staticmethod
    def key(filters: "SearchFilters", top_k: int) -> str:
        # Canonicalise so equivalent intents hash identically: scalars verbatim, list fields
        # normalised + deduped + sorted (order-independent), top_k folded in (it changes the
        # result set). Serialised with sorted keys for a stable hash.
        payload = {field: getattr(filters, field) for field in _SCALAR_FIELDS}
        for field in _LIST_FIELDS:
            payload[field] = sorted({normalize(term) for term in getattr(filters, field) if term})
        payload["top_k"] = top_k
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def eligible(filters: "SearchFilters", recent_product_ids: list[str]) -> bool:
        """Only cache context-free product searches: with no session/seen-product context the
        same filters always mean the same thing and produce the same grounded answer. Novelty
        ("换一批") and backtracking ("回到最开始") are intentionally never cached."""
        return (
            filters.intent_type == "product_search"
            and not filters.exclude_seen
            and not filters.recall_product_ids
            and not recent_product_ids
        )
