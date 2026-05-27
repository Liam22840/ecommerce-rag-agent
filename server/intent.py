"""Lightweight intent and constraint parsing.

The parser is deliberately rule-based for the first backend milestone. It gives
the service deterministic filters for price/category constraints, while leaving
room for a later LLM/NLU parser to produce the same SearchFilters structure.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


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


@dataclass
class SearchFilters:
    max_price: float | None = None
    min_price: float | None = None
    category: str | None = None
    sub_category: str | None = None
    brand: str | None = None
    prefer_low_price: bool = False
    required_terms: list[str] = field(default_factory=list)
    requested_specs: list[str] = field(default_factory=list)
    excluded_brands: list[str] = field(default_factory=list)
    excluded_terms: list[str] = field(default_factory=list)
    raw_query: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class IntentParser:
    def __init__(self, categories: set[str], sub_categories: set[str], brands: set[str]):
        self._categories = categories
        self._sub_categories = sub_categories
        self._brands = brands

    def parse(self, message: str) -> SearchFilters:
        text = message.strip()
        filters = SearchFilters(raw_query=text)
        filters.max_price = _parse_max_price(text)
        filters.min_price = _parse_min_price(text)
        filters.sub_category = self._match_sub_category(text)
        filters.category = self._match_category(text)
        if filters.category is None and filters.sub_category:
            inferred = SUB_CATEGORY_TO_CATEGORY.get(filters.sub_category)
            if inferred in self._categories:
                filters.category = inferred
        filters.brand = self._match_brand(text)
        filters.prefer_low_price = _prefers_low_price(text)
        filters.required_terms = _parse_required_terms(text)
        filters.requested_specs = _parse_requested_specs(text)
        filters.excluded_brands = self._match_excluded_brands(text)
        filters.excluded_terms = _parse_excluded_terms(text)
        return filters

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


def _normalize_spec(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()
