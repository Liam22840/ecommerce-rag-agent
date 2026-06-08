import json

from server.intent import IntentParser


def _parser() -> IntentParser:
    return IntentParser(
        categories={"美妆护肤", "数码电子", "服饰运动", "食品饮料"},
        sub_categories={"洁面", "真无线耳机", "跑步鞋", "智能手机"},
        brands={"珊珂", "华为", "Apple 苹果"},
    )


class _FixedLLM:
    """Stub intent LLM that always returns the same parsed payload."""

    available = True

    def __init__(self, payload: dict):
        self._payload = payload

    def complete(self, _messages):
        return json.dumps(self._payload)


def _llm_parser(payload: dict) -> IntentParser:
    return IntentParser(
        categories={"美妆护肤"},
        sub_categories={"面霜"},
        brands=set(),
        llm=_FixedLLM(payload),
        approx_price_tolerance=0.15,
    )


def test_parses_basic_recommendation_aliases():
    filters = _parser().parse("推荐一款适合油皮的洗面奶")

    assert filters.category == "美妆护肤"
    assert filters.sub_category == "洁面"
    assert filters.max_price is None


def test_matches_category_named_directly_in_query():
    # A query that names a catalog category outright is matched directly (not via an alias).
    filters = _parser().parse("推荐美妆护肤的产品")
    assert filters.category == "美妆护肤"


def test_lead_in_hint_search_for_subcategory():
    assert _parser().lead_in_hint("推荐一款洗面奶") == ("search", "洁面")


def test_lead_in_hint_search_for_category_only():
    # Category named without a sub-category -> the opener acknowledges the category.
    assert _parser().lead_in_hint("有什么美妆护肤推荐") == ("search", "美妆护肤")


def test_lead_in_hint_neutral_for_chitchat():
    assert _parser().lead_in_hint("你好呀") == ("neutral", None)


def test_parses_budget_and_earphone_alias():
    filters = _parser().parse("200 元以下的蓝牙耳机有哪些？")

    assert filters.category == "数码电子"
    assert filters.sub_category == "真无线耳机"
    assert filters.max_price == 200.0


def test_parses_brand_exclusion_for_future_advanced_cases():
    filters = _parser().parse("除了华为，推荐一个手机，不超过8000")

    assert filters.sub_category == "智能手机"
    assert filters.max_price == 8000.0
    assert "华为" in filters.excluded_brands


def test_excluded_brand_clears_contradictory_positive_brand():
    # Regression: "不要华为的" matches 华为 as both a bare substring (positive brand) and an
    # exclusion. The contradictory positive brand must be dropped so the filter isn't impossible.
    filters = _parser().parse("推荐手机，不要华为的")

    assert "华为" in filters.excluded_brands
    assert filters.brand is None


def test_approximate_price_widens_zero_width_band():
    # Regression: the LLM collapses "三百左右" to min==max==300, which matches nothing, so an
    # approximate marker must widen it to a tolerance band.
    parser = _llm_parser({"sub_category": "面霜", "min_price": 300, "max_price": 300})

    filters = parser.parse("三百左右的面霜")

    assert filters.min_price == 255.0
    assert filters.max_price == 345.0


def test_exact_price_band_is_not_widened():
    # Without an approximate marker, an explicit min==max stays exact.
    parser = _llm_parser({"sub_category": "面霜", "min_price": 300, "max_price": 300})

    filters = parser.parse("正好三百块的面霜")

    assert filters.min_price == 300.0
    assert filters.max_price == 300.0


def test_parses_low_price_preference():
    filters = _parser().parse("推荐一个适合敏感肌的保湿护肤品，cheaper is better")

    assert filters.prefer_low_price is True
    assert filters.required_terms == ["敏感肌", "保湿"]


def test_parses_requested_specs():
    filters = _parser().parse("推荐50g适合敏感肌的保湿霜，cheaper is better")

    assert filters.requested_specs == ["50g"]


def test_parses_min_price_above_and_not_below():
    assert _parser().parse("1000元以上的手机").min_price == 1000.0
    assert _parser().parse("不低于500的耳机").min_price == 500.0


def test_rule_parser_understands_chinese_numeral_prices():
    # The chat model handles Chinese numbers directly. This covers the deterministic fallback for
    # when the model is unavailable, which must also parse Chinese numerals, not only Arabic digits.
    p = _parser()
    assert p.parse("三百以内的洗面奶").max_price == 300.0
    assert p.parse("不超过一万的手机").max_price == 10000.0
    assert p.parse("一千元以上的耳机").min_price == 1000.0
    assert p.parse("三百五十以内的面霜").max_price == 350.0
    # Numerals inside a name must not be read as a price.
    assert p.parse("三只松鼠的零食").max_price is None


def test_llm_planned_task_intent_is_coerced_and_routes():
    # The intent LLM is now the router; planned_task must survive coercion (it's a valid intent value).
    parser = _llm_parser({"intent_type": "planned_task"})
    assert parser.parse("推荐跑鞋并对比最便宜的两双加入购物车").intent_type == "planned_task"


def test_parses_excluded_terms_from_negation():
    filters = _parser().parse("推荐一个面霜，不要香精")
    assert "香精" in filters.excluded_terms


def test_parses_storage_and_volume_specs():
    assert _parser().parse("256GB 的手机").requested_specs == ["256gb"]
    assert _parser().parse("来瓶500ml的汽水").requested_specs == ["500ml"]


def test_rule_detects_comparison_intent():
    filters = _parser().parse("这两款耳机哪个更好")
    assert filters.intent_type == "comparison"


def test_category_matched_via_alias_without_official_word():
    # "数码" is an alias for the 数码电子 category, which isn't spelled out.
    filters = _parser().parse("推荐点数码好物")
    assert filters.category == "数码电子"
