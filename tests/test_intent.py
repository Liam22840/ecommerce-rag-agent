from server.intent import IntentParser


def _parser() -> IntentParser:
    return IntentParser(
        categories={"美妆护肤", "数码电子", "服饰运动", "食品饮料"},
        sub_categories={"洁面", "真无线耳机", "跑步鞋", "智能手机"},
        brands={"珊珂", "华为", "Apple 苹果"},
    )


def test_parses_basic_recommendation_aliases():
    filters = _parser().parse("推荐一款适合油皮的洗面奶")

    assert filters.category == "美妆护肤"
    assert filters.sub_category == "洁面"
    assert filters.max_price is None


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
