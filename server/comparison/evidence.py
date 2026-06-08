"""Judging products on a dimension: deterministic scoring and the LLM judge payload.

_evidence scores a product's catalog text against a dimension's terms. The LLM judge
(messages + confidence validation) lets a model adjudicate, with the deterministic
scorer as the fallback.
"""

from __future__ import annotations

import json
from typing import Any

from server.prompts import EVIDENCE_JUDGE_SYSTEM
from server.textutil import dedupe, normalize, trim
from server.comparison.dimensions import DimensionSpec
from server.comparison.text import _chunks, _product_evidence_for_llm, _source_texts, _strip_source


GENERIC_NEGATIVE_CUES = (
    "不",
    "没",
    "无",
    "差",
    "弱",
    "低",
    "少",
    "慢",
    "短",
    "贵",
    "一般",
    "不足",
    "不够",
    "不太",
    "不是",
    "没有",
    "失望",
    "问题",
    "缺点",
)
GENERIC_POSITIVE_CUES = (
    "好",
    "强",
    "高",
    "足",
    "快",
    "长",
    "低",
    "轻",
    "稳",
    "值",
    "适合",
    "不错",
    "明显",
    "满意",
    "推荐",
    "优秀",
    "友好",
)
LOWER_IS_BETTER_CUES = ("低", "少", "短", "轻", "便宜", "省", "0", "零", "无", "不含", "没有")
HIGHER_IS_BETTER_CUES = ("高", "多", "长", "强", "足", "厚", "贵")


def _evidence(
    product: dict[str, Any],
    terms: tuple[str, ...],
    preference: str,
) -> tuple[float, list[str]]:
    score = 0.0
    snippets = []
    for source, text, weight in _source_texts(product):
        if not text:
            continue
        source_score = 0.0
        for chunk in _chunks(text):
            matched_terms = [term for term in terms if term and term.lower() in chunk.lower()]
            if not matched_terms:
                continue
            polarity = _generic_polarity(chunk, preference)
            source_score += weight * polarity * min(2.0, 0.75 + 0.25 * len(matched_terms))
        if source_score != 0:
            score += source_score
            snippet = _best_snippet(text, terms)
            if snippet:
                snippets.append(f"{source}: {snippet}")
    return score, dedupe(snippets)


def _best_snippet(text: str, terms: tuple[str, ...], max_len: int = 86) -> str:
    chunks = _chunks(text)
    for term in terms:
        for chunk in chunks:
            if term and term.lower() in chunk.lower():
                return trim(chunk, max_len)
    return trim(chunks[0], max_len) if chunks else ""


def _generic_polarity(text: str, preference: str = "higher_is_better") -> float:
    normalized = normalize(text)
    if preference == "lower_is_better":
        lower_hits = sum(1 for cue in LOWER_IS_BETTER_CUES if cue and cue in normalized)
        higher_hits = sum(1 for cue in HIGHER_IS_BETTER_CUES if cue and cue in normalized)
        if lower_hits > higher_hits:
            return 1.25
        if higher_hits > lower_hits:
            return -1.0
    negative_hits = sum(1 for cue in GENERIC_NEGATIVE_CUES if cue and cue in normalized)
    positive_hits = sum(1 for cue in GENERIC_POSITIVE_CUES if cue and cue in normalized)
    if negative_hits > positive_hits:
        return -1.0
    if positive_hits > 0:
        return 1.25
    return 0.8


def _evidence_value(label: str, snippets: list[str]) -> str:
    if snippets:
        return "；".join(_strip_source(snippet) for snippet in snippets[:2])
    return f"商品库未提供足够“{label}”证据"


def _confidence(score: float, snippets: list[str]) -> str:
    if not snippets:
        return "none"
    if abs(score) >= 5:
        return "high"
    if abs(score) >= 2:
        return "medium"
    return "low"


def _winner_from_scores(scores: dict[str, float]) -> str | None:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] <= 0:
        return None
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 2:
        return None
    return ranked[0][0]


def _evidence_judge_messages(
    query: str,
    products: list[dict[str, Any]],
    specs: list[DimensionSpec],
) -> list[dict[str, str]]:
    user_payload = {
        "query": query,
        "dimensions": [{"label": spec.label, "preference": spec.preference} for spec in specs],
        "products": [_product_evidence_for_llm(product) for product in products],
    }
    return [
        {"role": "system", "content": EVIDENCE_JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def _judge_confidence(raw: Any, reason: str, grounded: bool) -> str:
    if not reason and not grounded:
        return "none"
    conf = raw.strip() if isinstance(raw, str) else ""
    if conf not in {"high", "medium", "low", "none"}:
        conf = "medium"
    # No verifiable quote -> don't over-claim.
    if conf in {"high", "medium"} and not grounded:
        conf = "low"
    return conf
