"""Intent and constraint parsing.

A deterministic rule pass extracts price/category constraints, and when an LLM is
available it refines the result. Both produce the same SearchFilters structure and
the rule pass is the fallback when the model is unavailable.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from server.prompts import intent_messages
from server.textutil import dedupe, json_object, normalize, normalize_spec, trim


# Single source of truth for sellpoint-attribute synonyms: the rule-parser extracts these from a
# query (LLM-off fallback) and the catalog uses them to evidence/rank the same attribute.
REQUIRED_TERM_ALIASES: dict[str, list[str]] = {
    "敏感肌": ["敏感肌", "敏感性", "敏感皮", "干敏", "易敏", "敏皮"],
    "保湿": ["保湿", "补水", "锁水", "滋润"],
}

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

SORT_BY_VALUES = {"relevance", "price_asc", "price_desc", "rating_desc"}
INTENT_TYPE_VALUES = {"product_search", "comparison", "chitchat", "cart_action", "checkout"}

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
    # LLM-rewritten standalone retrieval query for context-dependent follow-ups.
    # Empty means "use raw_query".
    rewritten_query: str = ""
    # LLM-set flag for novelty refinements ("还有别的/换一批"): drop already-shown products.
    exclude_seen: bool = False
    # LLM-picked product ids (copied from session_products) for backtracking recall
    # ("回到最开始那个"), validated against the catalog before use.
    recall_product_ids: list[str] = field(default_factory=list)
    # LLM-resolved product ids (copied from session_products) for a comparison turn
    # ("第一个和第二个"), validated against the catalog, with the ordinal/name waterfall as fallback.
    compare_product_ids: list[str] = field(default_factory=list)
    commerce_action: str | None = None
    commerce_refs: list[str] = field(default_factory=list)
    commerce_quantity: int | None = None
    commerce_target_scope: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class IntentParser:
    def __init__(
        self,
        categories: set[str],
        sub_categories: set[str],
        brands: set[str],
        llm: Any | None = None,
        approx_price_tolerance: float = 0.15,
    ):
        self._categories = categories
        self._sub_categories = sub_categories
        self._brands = brands
        self._llm = llm
        self._approx_price_tolerance = approx_price_tolerance

    def parse(
        self,
        message: str,
        previous_filters: SearchFilters | None = None,
        history: list[dict[str, Any]] | None = None,
        session_products: list[dict[str, Any]] | None = None,
    ) -> SearchFilters:
        # `previous_filters` drives the deterministic backstop (last turn only). `history`
        # is the few-round refinement context. `session_products` is the whole-session list
        # of shown products the LLM uses to resolve backtracking ("回到最开始那个").
        rule = self._rule_parse(message)
        if self._should_use_llm():
            llm = self._llm_parse(message, history, session_products)
            base = rule if llm is None else self._merge(rule, llm, message)
        else:
            base = rule
        # Deterministic carry-over backstop: rescues the common refinement case even
        # when the LLM is unavailable or omits the carry. LLM proposes, this disposes.
        base = self._apply_session_context(base, previous_filters)
        # Safety net for either source contradicting itself: a brand can't be both wanted and
        # excluded. The rule parser already avoids this. This also catches the LLM if it ever
        # emits a brand it simultaneously excluded, which would otherwise match nothing.
        if base.brand and base.brand in base.excluded_brands:
            base.brand = None
        self._widen_approximate_price(base, message)
        return base

    def lead_in_hint(self, message: str) -> tuple[str, str | None]:
        """Instant, deterministic guess for the streaming opener, no LLM call. Rule-parse the
        raw text and report what to acknowledge: ("search", <type>) when a catalog category is
        named, ("compare", None) for a comparison phrasing, else ("neutral", None). Type wins
        over the fuzzy comparison regex. Anything unrecognised falls back to neutral, so chit-chat
        is never mis-opened. The real understanding is still the LLM's job, behind the opener."""
        rule = self._rule_parse(message)
        # Only tailor toward a type the user positively wants. If the detected type is itself
        # excluded ("不要面霜" -> 面霜 lands in both sub_category and excluded_terms), there's no
        # clear positive intent, so stay neutral rather than offering the negated type (or a
        # category backfilled from it). A modifier negation ("不要油腻的面霜") still tailors,
        # because the excluded term is the phrase, not the type itself.
        excluded = {normalize(term) for term in rule.excluded_terms}
        if rule.sub_category:
            return ("neutral", None) if normalize(rule.sub_category) in excluded else ("search", rule.sub_category)
        if rule.category:
            return ("neutral", None) if normalize(rule.category) in excluded else ("search", rule.category)
        if rule.intent_type == "comparison":
            return ("compare", None)
        return ("neutral", None)

    def _widen_approximate_price(self, filters: SearchFilters, message: str) -> None:
        # Fallback for when the LLM (or the rule parser, which never expands) collapsed an
        # approximate price ("三百左右") to a zero-width band (min==max) that matches nothing:
        # when the user signalled approximation, widen it to a tolerance band around the price.
        if (
            filters.min_price is not None
            and filters.min_price == filters.max_price
            and _is_approximate_price(message)
        ):
            centre = filters.min_price
            delta = centre * self._approx_price_tolerance
            filters.min_price = max(0.0, centre - delta)
            filters.max_price = centre + delta

    def _rule_parse(self, message: str) -> SearchFilters:
        text = message.strip()
        filters = SearchFilters(raw_query=text)
        filters.max_price = _parse_max_price(text)
        filters.min_price = _parse_min_price(text)
        filters.sub_category = self._match_sub_category(text)
        filters.category = self._match_category(text)
        self._backfill_category(filters)
        # Exclusions first, so a negated brand ("不要华为") is never matched as a wanted brand.
        filters.excluded_brands = self._match_excluded_brands(text)
        filters.brand = self._match_brand(text, filters.excluded_brands)
        filters.prefer_low_price = _prefers_low_price(text)
        if filters.prefer_low_price:
            filters.sort_by = "price_asc"
        # Rule-fallback comparison detection keeps comparison routing working when the
        # LLM is unavailable. Chitchat is not rule-detectable, so it stays product_search.
        if _looks_like_comparison(text):
            filters.intent_type = "comparison"
        filters.required_terms = _parse_required_terms(text)
        filters.requested_specs = _parse_requested_specs(text)
        filters.excluded_terms = _parse_excluded_terms(text)
        return filters

    def _should_use_llm(self) -> bool:
        return bool(self._llm) and getattr(self._llm, "available", False)

    def _llm_parse(
        self,
        message: str,
        history: list[dict[str, Any]] | None = None,
        session_products: list[dict[str, Any]] | None = None,
    ) -> SearchFilters | None:
        try:
            raw = self._llm.complete(
                intent_messages(
                    message,
                    self._categories,
                    self._sub_categories,
                    self._brands,
                    history=history,
                    session_products=session_products,
                )
            )
            payload = json_object(raw)
        except Exception:  # noqa: BLE001 (LLM parse must degrade to the rule parser)
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
        filters.requested_specs = [normalize_spec(s) for s in _coerce_str_list(payload.get("requested_specs"))]
        filters.excluded_brands = [b for b in _coerce_str_list(payload.get("excluded_brands")) if b in self._brands]
        filters.excluded_terms = _coerce_str_list(payload.get("excluded_terms"))
        filters.compare_refs = _coerce_str_list(payload.get("compare_refs"))
        filters.rewritten_query = _coerce_query(payload.get("rewritten_query"))
        filters.exclude_seen = _coerce_bool(payload.get("exclude_seen"))
        filters.recall_product_ids = _coerce_str_list(payload.get("recall_product_ids"))
        filters.compare_product_ids = _coerce_str_list(payload.get("compare_product_ids"))
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
        merged.required_terms = dedupe(rule.required_terms + llm.required_terms)
        merged.requested_specs = dedupe(rule.requested_specs + llm.requested_specs)
        merged.excluded_brands = dedupe(rule.excluded_brands + llm.excluded_brands)
        merged.excluded_terms = dedupe(rule.excluded_terms + llm.excluded_terms)
        merged.compare_refs = llm.compare_refs  # references are only extracted by the LLM
        merged.prefer_low_price = rule.prefer_low_price or llm.prefer_low_price or merged.sort_by == "price_asc"
        merged.rewritten_query = llm.rewritten_query  # rewrite is only produced by the LLM
        merged.exclude_seen = llm.exclude_seen  # novelty flag is only produced by the LLM
        merged.recall_product_ids = llm.recall_product_ids  # recall ids only come from the LLM
        merged.compare_product_ids = llm.compare_product_ids  # comparison ids only come from the LLM
        self._backfill_category(merged)
        return merged

    def _apply_session_context(
        self, filters: SearchFilters, previous_filters: SearchFilters | None
    ) -> SearchFilters:
        """Deterministic carry-over backstop. When the current turn is a product search that
        names no category or sub_category, treat it as a refinement of the previous turn and
        inherit the prior topic anchor (category / sub_category / sellpoints). Never overrides
        a value the current turn set, and deliberately does not carry brand / price / sort
        (those are commonly changed in a refining turn and are the LLM's job to carry). This is
        the entire carry-over behaviour in degraded mode and a safety net when the LLM omits it."""
        if previous_filters is None:
            return filters
        if filters.intent_type != "product_search":
            return filters
        if filters.category or filters.sub_category:
            return filters
        if not (previous_filters.category or previous_filters.sub_category):
            return filters
        filters.category = previous_filters.category
        filters.sub_category = previous_filters.sub_category
        filters.required_terms = dedupe(previous_filters.required_terms + filters.required_terms)
        self._backfill_category(filters)
        return filters

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

    def _match_brand(self, text: str, excluded: list[str] | None = None) -> str | None:
        excluded = excluded or []
        for brand in sorted(self._brands, key=len, reverse=True):
            if brand and brand in text and brand not in excluded:
                return brand
        return None

    def _match_excluded_brands(self, text: str) -> list[str]:
        excluded = []
        for brand in sorted(self._brands, key=len, reverse=True):
            if brand and (f"不要{brand}" in text or f"不想要{brand}" in text or f"除了{brand}" in text):
                excluded.append(brand)
        return excluded


# A price number, in Arabic digits and/or Chinese numerals (三百, 一万, 1万, 三百五). Scoped to the
# price patterns below so it never touches numerals inside names (e.g. 三只松鼠).
_PRICE_NUMBER = r"([\d零〇一二两三四五六七八九十百千万]+(?:\.\d+)?)"
_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _cn_to_int(token: str) -> int | None:
    """Convert a Chinese or mixed-digit integer (三百 / 一万 / 1万 / 三百五十 / 三百五) to an int,
    or None if it isn't a parseable number. Used only for the deterministic price fallback. The
    chat model handles Chinese numbers itself."""
    total = section = number = last_unit = 0
    for ch in token:
        if ch.isdigit():
            number = number * 10 + int(ch)
        elif ch in _CN_DIGIT:
            number = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                section += (number or 1) * unit
            last_unit = unit
            number = 0
        else:
            return None
    if number and last_unit >= 10:  # trailing bare digit, e.g. 三百五 -> 350
        section += number * (last_unit // 10)
        number = 0
    return (total + section + number) or None


def _to_number(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        value = _cn_to_int(token)
        return float(value) if value is not None else None


def _parse_max_price(text: str) -> float | None:
    patterns = [
        rf"(?:预算|价格|价位)?\s*{_PRICE_NUMBER}\s*(?:元|块|rmb|人民币)?\s*(?:以下|以内|内|之内)",
        rf"(?:不超过|不超|小于|低于|少于|<=|≤)\s*{_PRICE_NUMBER}",
        rf"{_PRICE_NUMBER}\s*(?:元|块)?\s*(?:封顶|以下)",
    ]
    return _first_number(patterns, text)


def _parse_min_price(text: str) -> float | None:
    patterns = [
        rf"{_PRICE_NUMBER}\s*(?:元|块|rmb|人民币)?\s*(?:以上|起)",
        rf"(?:不低于|大于|高于|>=|≥)\s*{_PRICE_NUMBER}",
    ]
    return _first_number(patterns, text)


_APPROX_PRICE_MARKERS = ("左右", "上下", "附近", "大概", "大约", "差不多", "约莫")


def _is_approximate_price(text: str) -> bool:
    return any(marker in text for marker in _APPROX_PRICE_MARKERS)


def _first_number(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = _to_number(match.group(1))
            if value is not None:
                return value
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
    return [
        term
        for term, aliases in REQUIRED_TERM_ALIASES.items()
        if any(alias in text for alias in aliases)
    ]


def _parse_requested_specs(text: str) -> list[str]:
    specs = []
    pattern = r"\d+(?:\.\d+)?\s*(?:g|kg|克|千克|ml|mL|ML|l|L|升|毫升|片|枚|粒|支|瓶|包|盒|寸|英寸|gb|GB|tb|TB)"
    for match in re.finditer(pattern, text):
        specs.append(normalize_spec(match.group(0)))
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
    if isinstance(value, bool):  # bool is an int subclass, reject explicitly
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
    return dedupe(items)[:max_items]


def _coerce_query(value: Any, max_len: int = 80) -> str:
    if not isinstance(value, str):
        return ""
    return trim(value, max_len)
