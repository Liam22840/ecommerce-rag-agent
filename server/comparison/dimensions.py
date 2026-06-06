"""Deciding what to compare: extract comparison dimensions from the query.

Two paths produce DimensionSpec lists — an LLM payload (_specs_from_llm_payload) and a
deterministic fallback (_dynamic_specs) — backed by the product corpus so every term is
grounded in real catalog text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from server.intent import SearchFilters
from server.prompts import DIMENSION_EXTRACTION_SYSTEM
from server.textutil import dedupe, normalize
from server.comparison.text import _product_corpus, _product_evidence_for_llm


PRICE_FOCUS_TERMS = ("价格", "预算", "便宜", "划算", "性价比", "省钱", "更值")


@dataclass(frozen=True)
class DimensionSpec:
    label: str
    terms: tuple[str, ...]
    preference: str = "higher_is_better"
    evidence: bool = True


def _dimension_extraction_messages(query: str, products: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DIMENSION_EXTRACTION_SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query": query,
                    "products": [_product_evidence_for_llm(product) for product in products],
                    "rules": [
                        "label 用用户能理解的短词，例如 舒适度、降噪安静、佩戴舒适、水润不拔干。",
                        "aliases 必须来自或贴近商品标题、描述、官方问答、用户评价中的词，便于后端检索证据。",
                        "如果用户关注更低、更少、更轻、更便宜，preference 设为 lower_is_better；其他通常为 higher_is_better。",
                        "价格、容量、库存、SKU 不需要作为 evidence 维度，后端会用结构化字段处理。",
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]


def _specs_from_llm_payload(
    payload: dict[str, Any],
    query: str,
    products: list[dict[str, Any]],
) -> list[DimensionSpec]:
    raw_dimensions = payload.get("dimensions", [])
    if not isinstance(raw_dimensions, list):
        return []
    corpus = _product_corpus(products)
    specs = []
    seen = set()
    for raw in raw_dimensions:
        if not isinstance(raw, dict):
            continue
        label = _clean_attribute_label(str(raw.get("label", "")))
        label = _strip_product_context_words(label, products)
        normalized_label = normalize(label)
        if not normalized_label or normalized_label in seen:
            continue
        if _is_noise_attribute(normalized_label, products):
            continue
        aliases = raw.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        terms = _llm_terms(label, aliases)
        if not terms:
            continue
        if _is_price_dimension(label, terms):
            continue
        matched_terms = tuple(term for term in terms if normalize(term) in corpus)
        if not matched_terms:
            continue
        preference = str(raw.get("preference", "")).strip()
        if preference not in {"higher_is_better", "lower_is_better"}:
            preference = _infer_preference(query, label)
        seen.add(normalized_label)
        specs.append(DimensionSpec(label=label, terms=matched_terms, preference=preference))
    return _dedupe_specs(specs)[:4]


def _llm_terms(label: str, aliases: list[Any]) -> tuple[str, ...]:
    terms = [label]
    terms.extend(str(alias).strip() for alias in aliases if str(alias).strip())
    for term in list(terms):
        terms.extend(_attribute_terms(term))
    return tuple(dedupe([term for term in terms if len(normalize(term)) >= 2])[:16])


def _is_price_dimension(label: str, terms: tuple[str, ...]) -> bool:
    normalized = {normalize(label), *(normalize(term) for term in terms)}
    return any(normalize(term) in normalized for term in PRICE_FOCUS_TERMS)


def _infer_preference(query: str, label: str) -> str:
    normalized = normalize(query)
    normalized_label = normalize(label)
    if any(term in normalized for term in ["更低", "更少", "更短", "更轻", "更便宜", "省钱", "预算"]):
        if not normalized_label or normalized_label in normalized:
            return "lower_is_better"
    return "higher_is_better"


def _dynamic_specs(
    query: str,
    products: list[dict[str, Any]],
) -> list[DimensionSpec]:
    if not products:
        return []
    normalized = normalize(query)
    corpus = _product_corpus(products)
    candidates = _explicit_attribute_candidates(query)
    if not candidates:
        candidates.extend(_corpus_backed_query_ngrams(normalized, corpus, products))
    specs = []
    seen = set()
    for candidate in candidates:
        label = _clean_attribute_label(candidate)
        label = _strip_product_context_words(label, products)
        normalized_label = normalize(label)
        if not normalized_label or normalized_label in seen:
            continue
        if _is_noise_attribute(normalized_label, products):
            continue
        terms = _attribute_terms(label)
        if not any(normalize(term) in corpus for term in terms):
            continue
        seen.add(normalized_label)
        specs.append(DimensionSpec(
            label=label,
            terms=terms,
            preference=_infer_preference(query, label),
        ))
    return _dedupe_specs(specs)[:3]


def _explicit_attribute_candidates(query: str) -> list[str]:
    candidates = []
    clauses = [clause for clause in re.split(r"[。！？!?；;，,]", query) if clause.strip()]
    before_suffix = r"(?:更好|更强|更高|更低|更长|更短|更久|更轻|更重|更快|更慢|更足|更稳|更靠谱|更值|更划算|更便宜|更贵|好|强|高|低|长|短|久|轻|重|快|慢|足|稳)"
    after_prefix = r"(?:更适合|更推荐|更偏向|更满足|更符合|更有利于|更)"
    for clause in clauses:
        for match in re.finditer(rf"(?:哪个|哪款|哪一个|哪种|哪双|哪件|哪瓶|哪台|哪部)?([^，。,.；;？?]{{1,20}}?){before_suffix}", clause, flags=re.IGNORECASE):
            candidates.extend(_split_attribute_candidate(match.group(1)))
        for match in re.finditer(rf"(?:哪个|哪款|哪一个|哪种|哪双|哪件|哪瓶|哪台|哪部)[^，。,.；;？?]{{0,8}}?{after_prefix}([^，。,.；;？?]{{1,18}})", clause, flags=re.IGNORECASE):
            candidates.extend(_split_attribute_candidate(match.group(1)))
        for match in re.finditer(r"(?:看重|关注|侧重|优先|比较|对比|按|从)([^，。,.；;？?]{1,20})", clause, flags=re.IGNORECASE):
            candidates.extend(_split_attribute_candidate(match.group(1)))
    return candidates


def _split_attribute_candidate(value: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?:和|跟|与|、|，|,|以及|还有|并且|或者|或|/)", value)
        if part.strip()
    ]


def _corpus_backed_query_ngrams(normalized_query: str, corpus: str, products: list[dict[str, Any]]) -> list[str]:
    candidates = []
    for size in range(8, 1, -1):
        for idx in range(0, max(0, len(normalized_query) - size + 1)):
            token = normalized_query[idx: idx + size]
            if token in corpus and not _is_noise_attribute(token, products):
                candidates.append(token)
    return candidates


def _clean_attribute_label(value: str) -> str:
    label = value.strip()
    label = re.sub(r"^.*(?:哪个|哪款|哪一个|哪种|哪双|哪件|哪瓶|哪台|哪部)", "", label)
    label = re.sub(r"(?:这两款|这两个|这两种|这两双|这两件|这两瓶|这两台|这两部|两个|两款|两种|两双|两件|两瓶|两台|两部|哪个|哪款|哪一个|哪种|哪双|哪件|哪瓶|哪台|哪部|第[一二三123]个|第[一二三123]款|商品|产品)", "", label)
    label = re.sub(r"(?:更好|更强|更高|更低|更长|更短|更久|更轻|更重|更快|更慢|更足|更稳|更适合|更推荐|更偏向|更靠谱|更值|更划算|更便宜|更贵|好|强|高|低|长|短|久|轻|重|快|慢|足|稳)$", "", label)
    label = re.sub(r"^[的得]+|[的得]+$", "", label)
    label = label.strip("款个只双件瓶盒袋包台部双")
    return label.strip()


def _strip_product_context_words(label: str, products: list[dict[str, Any]]) -> str:
    normalized = normalize(label)
    context_words = set()
    for product in products:
        for key in ("brand", "category", "sub_category"):
            word = normalize(str(product.get(key, "")))
            if len(word) < 2:
                continue
            context_words.add(word)
            for size in range(2, min(4, len(word)) + 1):
                context_words.add(word[:size])
                context_words.add(word[-size:])
    for word in sorted(context_words, key=len, reverse=True):
        if word == normalized:
            return ""
        if normalized.startswith(word):
            normalized = normalized[len(word):]
        if normalized.endswith(word):
            normalized = normalized[: -len(word)]
    return normalized


def _attribute_terms(label: str) -> tuple[str, ...]:
    terms = [label]
    compact = normalize(label)
    for part in re.split(r"(?:和|跟|与|、|，|,|以及|还有|并且|或者|或|/)", label):
        if part.strip():
            terms.append(part.strip())
    if len(compact) > 2:
        for size in range(min(4, len(compact)), 1, -1):
            for idx in range(0, len(compact) - size + 1):
                terms.append(compact[idx: idx + size])
    return tuple(dedupe([term for term in terms if len(normalize(term)) >= 2])[:8])


def _is_noise_attribute(normalized_label: str, products: list[dict[str, Any]]) -> bool:
    if len(normalized_label) < 2:
        return True
    if normalized_label.isdigit():
        return True
    if normalized_label[0] in {"款", "个", "只", "双", "件", "瓶", "盒", "袋", "包", "台", "部"}:
        return True
    noise = {
        "哪个",
        "哪款",
        "哪一个",
        "更好",
        "更强",
        "更高",
        "更低",
        "更足",
        "更适合",
        "更靠谱",
        "这两款",
        "这两个",
        "商品",
        "产品",
        "对比",
        "比较",
        "推荐",
        "适合",
        "更推荐",
        "更偏向",
        "更满足",
        "更符合",
        "买哪个",
        "选哪个",
        "第一个",
        "第二个",
        "第三个",
    }
    if normalized_label in noise:
        return True
    for product in products:
        product_noise = [
            product.get("brand", ""),
            product.get("category", ""),
            product.get("sub_category", ""),
        ]
        if normalized_label in {normalize(value) for value in product_noise if value}:
            return True
    return False


def _dedupe_specs(specs: list[DimensionSpec]) -> list[DimensionSpec]:
    seen = set()
    result = []
    for spec in specs:
        if spec.label not in seen:
            seen.add(spec.label)
            result.append(spec)
    return result


def _price_is_priority(query: str, filters: SearchFilters) -> bool:
    normalized = normalize(query)
    return filters.prefer_low_price or any(
        term in normalized
        for term in ["性价比", "划算", "便宜", "价格", "预算", "省钱", "更值"]
    )
