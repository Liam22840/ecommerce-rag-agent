"""Lightweight intent and constraint parsing.

The parser is deliberately rule-based for the first backend milestone. It gives
the service deterministic filters for price/category constraints, while leaving
room for a later LLM/NLU parser to produce the same SearchFilters structure.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from server.prompts import intent_messages
from server.textutil import dedupe as _dedupe
from server.textutil import json_object as _json_object
from server.textutil import normalize_spec as _normalize_spec


CATEGORY_ALIASES: dict[str, list[str]] = {
    "美妆护肤": ["护肤", "美妆", "化妆品", "彩妆"],
    "数码电子": ["数码", "电子", "手机", "电脑", "耳机", "平板"],
    "服饰运动": ["服饰", "运动", "穿搭", "衣服", "鞋", "跑步"],
    "食品饮料": ["食品", "饮料", "零食", "吃的", "喝的"],
}

SUB_CATEGORY_ALIASES: dict[str, list[str]] = {
    "洁面": ["洗面奶", "洁面", "洗脸", "清洁"],
    "防晒": ["防晒霜", "防晒乳", "防晒"],
    "面霜": ["面霜", "保湿霜"],
    "精华": ["精华", "肌底液"],
    "真无线耳机": ["蓝牙耳机", "无线耳机", "耳机", "降噪耳机"],
    "智能手机": ["手机", "拍照手机", "旗舰手机"],
    "笔记本电脑": ["笔记本", "电脑", "本子"],
    "平板电脑": ["平板", "平板电脑"],
    "跑步鞋": ["跑鞋", "跑步鞋", "慢跑鞋"],
    "篮球鞋": ["篮球鞋"],
    "徒步鞋": ["徒步鞋", "登山鞋"],
    "短袖T恤": ["短袖", "t恤", "T恤"],
    "速干T恤": ["速干", "速干衣"],
    "咖啡": ["咖啡"],
    "茶饮": ["茶", "茶饮"],
    "碳酸饮料": ["汽水", "碳酸饮料"],
    "坚果/零食": ["坚果", "零食"],
}

SUB_CATEGORY_TO_CATEGORY: dict[str, str] = {
    "洁面": "美妆护肤",
    "防晒": "美妆护肤",
    "面霜": "美妆护肤",
    "精华": "美妆护肤",
    "真无线耳机": "数码电子",
    "智能手机": "数码电子",
    "笔记本电脑": "数码电子",
    "平板电脑": "数码电子",
    "跑步鞋": "服饰运动",
    "篮球鞋": "服饰运动",
    "徒步鞋": "服饰运动",
    "短袖T恤": "服饰运动",
    "速干T恤": "服饰运动",
    "咖啡": "食品饮料",
    "茶饮": "食品饮料",
    "碳酸饮料": "食品饮料",
    "坚果/零食": "食品饮料",
}

BUYING_HINTS = ["推荐", "想买", "有哪些", "帮我找", "适合", "预算", "以内", "以下"]

SORT_BY_VALUES = {"relevance", "price_asc", "price_desc", "rating_desc"}
INTENT_TYPE_VALUES = {"product_search", "comparison", "chitchat"}

# Rule-fallback comparison detection. Kept local (not imported from comparison.py,
# which imports SearchFilters from here) so the rule path can set intent_type when the
# LLM is unavailable, preserving comparison routing in degraded mode.
COMPARISON_HINTS = [
    "对比", "比较", "哪个更", "哪款更", "哪一个更", "更适合", "选哪个", "买哪个",
    "二选一", "这两款", "这两个", "第一个", "第二个", "前两个",
]


@dataclass
class SearchFilters:
    max_price: float | None = None
    min_price: float | None = None
    category: str | None = None
    sub_category: str | None = None
    brand: str | None = None
    prefer_low_price: bool = False
    sort_by: str = "relevance"
    intent_type: str = "product_search"
    required_terms: list[str] = field(default_factory=list)
    requested_specs: list[str] = field(default_factory=list)
    excluded_brands: list[str] = field(default_factory=list)
    excluded_terms: list[str] = field(default_factory=list)
    compare_refs: list[str] = field(default_factory=list)
    raw_query: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class IntentParser:
    def __init__(
        self,
        categories: set[str],
        sub_categories: set[str],
        brands: set[str],
        llm: Any | None = None,
    ):
        self._categories = categories
        self._sub_categories = sub_categories
        self._brands = brands
        self._llm = llm

    def parse(self, message: str) -> SearchFilters:
        rule = self._rule_parse(message)
        if not self._should_use_llm():
            return rule
        llm = self._llm_parse(message)
        if llm is None:
            return rule
        return self._merge(rule, llm, message)

    def _rule_parse(self, message: str) -> SearchFilters:
        text = message.strip()
        filters = SearchFilters(raw_query=text)
        filters.max_price = _parse_max_price(text)
        filters.min_price = _parse_min_price(text)
        filters.sub_category = self._match_sub_category(text)
        filters.category = self._match_category(text)
        self._backfill_category(filters)
        filters.brand = self._match_brand(text)
        filters.prefer_low_price = _prefers_low_price(text)
        if filters.prefer_low_price:
            filters.sort_by = "price_asc"
        # Rule-fallback comparison detection keeps comparison routing working when the
        # LLM is unavailable; chitchat is not rule-detectable, so it stays product_search.
        if _looks_like_comparison(text):
            filters.intent_type = "comparison"
        filters.required_terms = _parse_required_terms(text)
        filters.requested_specs = _parse_requested_specs(text)
        filters.excluded_brands = self._match_excluded_brands(text)
        filters.excluded_terms = _parse_excluded_terms(text)
        return filters

    def _should_use_llm(self) -> bool:
        return bool(self._llm) and getattr(self._llm, "available", False)

    def _llm_parse(self, message: str) -> SearchFilters | None:
        try:
            raw = self._llm.complete(
                intent_messages(message, self._categories, self._sub_categories, self._brands)
            )
            payload = _json_object(raw)
        except Exception:  # noqa: BLE001 - LLM parse must degrade to the rule parser.
            return None
        if not payload:
            return None
        return self._filters_from_llm_payload(payload, message)

    def _filters_from_llm_payload(self, payload: dict[str, Any], message: str) -> SearchFilters:
        filters = SearchFilters(raw_query=message.strip())
        filters.category = _coerce_in_set(payload.get("category"), self._categories)
        filters.sub_category = _coerce_in_set(payload.get("sub_category"), self._sub_categories)
        filters.brand = _coerce_in_set(payload.get("brand"), self._brands)
        filters.max_price = _coerce_price(payload.get("max_price"))
        filters.min_price = _coerce_price(payload.get("min_price"))
        if filters.min_price is not None and filters.max_price is not None and filters.min_price > filters.max_price:
            filters.min_price, filters.max_price = filters.max_price, filters.min_price
        filters.sort_by = _coerce_enum(payload.get("sort_by"), SORT_BY_VALUES, "relevance")
        filters.intent_type = _coerce_enum(payload.get("intent_type"), INTENT_TYPE_VALUES, "product_search")
        filters.prefer_low_price = _coerce_bool(payload.get("prefer_low_price")) or filters.sort_by == "price_asc"
        filters.required_terms = _coerce_str_list(payload.get("required_terms"))
        filters.requested_specs = [_normalize_spec(s) for s in _coerce_str_list(payload.get("requested_specs"))]
        filters.excluded_brands = [b for b in _coerce_str_list(payload.get("excluded_brands")) if b in self._brands]
        filters.excluded_terms = _coerce_str_list(payload.get("excluded_terms"))
        filters.compare_refs = _coerce_str_list(payload.get("compare_refs"))
        return filters

    def _merge(self, rule: SearchFilters, llm: SearchFilters, message: str) -> SearchFilters:
        merged = SearchFilters(raw_query=message.strip())
        merged.category = llm.category or rule.category
        merged.sub_category = llm.sub_category or rule.sub_category
        merged.brand = llm.brand or rule.brand
        merged.max_price = llm.max_price if llm.max_price is not None else rule.max_price
        merged.min_price = llm.min_price if llm.min_price is not None else rule.min_price
        merged.sort_by = llm.sort_by if llm.sort_by != "relevance" else rule.sort_by
        merged.intent_type = llm.intent_type if llm.intent_type != "product_search" else rule.intent_type
        merged.required_terms = _dedupe(rule.required_terms + llm.required_terms)
        merged.requested_specs = _dedupe(rule.requested_specs + llm.requested_specs)
        merged.excluded_brands = _dedupe(rule.excluded_brands + llm.excluded_brands)
        merged.excluded_terms = _dedupe(rule.excluded_terms + llm.excluded_terms)
        merged.compare_refs = llm.compare_refs  # references are only extracted by the LLM
        merged.prefer_low_price = rule.prefer_low_price or llm.prefer_low_price or merged.sort_by == "price_asc"
        self._backfill_category(merged)
        return merged

    def _backfill_category(self, filters: SearchFilters) -> None:
        if filters.category is None and filters.sub_category:
            inferred = SUB_CATEGORY_TO_CATEGORY.get(filters.sub_category)
            if inferred in self._categories:
                filters.category = inferred

    def _match_category(self, text: str) -> str | None:
        for category in sorted(self._categories, key=len, reverse=True):
            if category in text:
                return category
        for category, aliases in CATEGORY_ALIASES.items():
            if category in self._categories and any(alias in text for alias in aliases):
                return category
        return None

    def _match_sub_category(self, text: str) -> str | None:
        for sub_category in sorted(self._sub_categories, key=len, reverse=True):
            if sub_category in text:
                return sub_category
        for sub_category, aliases in SUB_CATEGORY_ALIASES.items():
            if sub_category in self._sub_categories and any(alias.lower() in text.lower() for alias in aliases):
                return sub_category
        return None

    def _match_brand(self, text: str) -> str | None:
        for brand in sorted(self._brands, key=len, reverse=True):
            if brand and brand in text:
                return brand
        return None

    def _match_excluded_brands(self, text: str) -> list[str]:
        excluded = []
        for brand in sorted(self._brands, key=len, reverse=True):
            if brand and (f"不要{brand}" in text or f"不想要{brand}" in text or f"除了{brand}" in text):
                excluded.append(brand)
        return excluded


def _parse_max_price(text: str) -> float | None:
    patterns = [
        r"(?:预算|价格|价位)?\s*(\d+(?:\.\d+)?)\s*(?:元|块|rmb|人民币)?\s*(?:以下|以内|内|之内)",
        r"(?:不超过|不超|小于|低于|少于|<=|≤)\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:元|块)?\s*(?:封顶|以下)",
    ]
    return _first_number(patterns, text)


def _parse_min_price(text: str) -> float | None:
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:元|块|rmb|人民币)?\s*(?:以上|起)",
        r"(?:不低于|大于|高于|>=|≥)\s*(\d+(?:\.\d+)?)",
    ]
    return _first_number(patterns, text)


def _first_number(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _parse_excluded_terms(text: str) -> list[str]:
    terms: list[str] = []
    for pattern in [r"不要([^，。,.；;]+)", r"不含([^，。,.；;]+)"]:
        for match in re.finditer(pattern, text):
            term = match.group(1).strip()
            if term and len(term) <= 20:
                terms.append(term)
    return terms


def _prefers_low_price(text: str) -> bool:
    lowered = text.lower()
    hints = [
        "cheaper",
        "cheap",
        "low price",
        "budget",
        "便宜",
        "低价",
        "划算",
        "性价比",
        "不要太贵",
        "价格低",
        "价格从低到高",
    ]
    return any(hint in lowered for hint in hints)


def _parse_required_terms(text: str) -> list[str]:
    required_terms = []
    if any(term in text for term in ["敏感肌", "敏感皮", "易敏", "干敏", "敏皮"]):
        required_terms.append("敏感肌")
    if any(term in text for term in ["保湿", "补水", "锁水", "滋润"]):
        required_terms.append("保湿")
    return required_terms


def _parse_requested_specs(text: str) -> list[str]:
    specs = []
    pattern = r"\d+(?:\.\d+)?\s*(?:g|kg|克|千克|ml|mL|ML|l|L|升|毫升|片|枚|粒|支|瓶|包|盒|寸|英寸|gb|GB|tb|TB)"
    for match in re.finditer(pattern, text):
        specs.append(_normalize_spec(match.group(0)))
    return list(dict.fromkeys(specs))


def _looks_like_comparison(text: str) -> bool:
    return any(hint in text for hint in COMPARISON_HINTS)


def _coerce_in_set(value: Any, allowed: set[str]) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value in allowed else None


def _coerce_enum(value: Any, allowed: set[str], default: str) -> str:
    if isinstance(value, str) and value.strip() in allowed:
        return value.strip()
    return default


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _coerce_price(value: Any) -> float | None:
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price < 0:
        return None
    return price


def _coerce_str_list(value: Any, max_items: int = 16, max_len: int = 20) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if item and len(item) <= max_len:
            items.append(item)
    return _dedupe(items)[:max_items]
