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

