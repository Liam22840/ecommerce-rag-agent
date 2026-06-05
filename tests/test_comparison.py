"""Unit tests for the comparison engine: pure helpers and resolution logic."""

from __future__ import annotations

from server.catalog import ProductCatalog
from server.comparison import (
    ComparisonService,
    _asks_for_current_two,
    _attribute_terms,
    _chunks,
    _clean_attribute_label,
    _confidence,
    _dynamic_specs,
    _evidence,
    _generic_polarity,
    _is_noise_attribute,
    _is_price_dimension,
    _json_object,
    _name_score,
    _normalize,
    _price_is_priority,
    _specs_from_llm_payload,
    _strip_product_context_words,
    _trim,
    _winner_from_scores,
)
from server.intent import SearchFilters


def _product(
    pid: str = "p1",
    title: str = "测试耳机",
    brand: str = "某牌",
    category: str = "数码电子",
    sub_category: str = "真无线耳机",
    desc: str = "降噪效果出色，佩戴舒适",
    reviews=None,
    base_price: float = 100.0,
) -> dict:
    return {
        "product_id": pid,
        "title": title,
        "brand": brand,
        "category": category,
        "sub_category": sub_category,
        "base_price": base_price,
        "image_path": f"{pid}.jpg",
        "skus": [],
        "rag_knowledge": {
            "marketing_description": desc,
            "official_faq": [],
            "user_reviews": reviews or [],
        },
    }


def _catalog(*products: dict) -> ProductCatalog:
    return ProductCatalog({p["product_id"]: p for p in products})


# --- _normalize / _trim / _chunks ----------------------------------------------

def test_normalize_strips_punctuation_and_lowercases():
    assert _normalize("A·B, C。D（E）") == "abcde"
    assert _normalize("降 噪_效-果") == "降噪效果"


def test_trim_truncates_with_ellipsis():
    assert _trim("abc", 10) == "abc"
    assert _trim("abcdef", 4) == "abc…"
    assert _trim("  padded  ", 10) == "padded"


def test_chunks_splits_on_terminators_and_drops_blanks():
    assert _chunks("好用。很安静！戴久也舒服\n\n续航强") == ["好用", "很安静", "戴久也舒服", "续航强"]
    assert _chunks("   ") == []


# --- _generic_polarity ----------------------------------------------------------

def test_polarity_higher_is_better_orders_positive_neutral_negative():
    # Assert ordering, not the exact weights, so retuning the heuristic doesn't break this.
    positive = _generic_polarity("降噪效果好")  # positive cue 好
    neutral = _generic_polarity("测试文本")  # no cues
    negative = _generic_polarity("续航差发热")  # negative cue 差
    assert negative < neutral < positive
    assert positive > 0 and negative < 0


def test_polarity_lower_is_better_rewards_lower_terms():
    assert _generic_polarity("糖分很低", preference="lower_is_better") > 0
    assert _generic_polarity("糖分很高", preference="lower_is_better") < 0


# --- _winner_from_scores --------------------------------------------------------

def test_winner_requires_clear_margin():
    assert _winner_from_scores({"a": 5.0, "b": 1.0}) == "a"
    assert _winner_from_scores({"a": 3.0, "b": 2.0}) is None  # margin < 2
    assert _winner_from_scores({"a": 0.0, "b": -1.0}) is None  # no positive score
    assert _winner_from_scores({"a": 3.0}) == "a"  # single candidate
    assert _winner_from_scores({}) is None


# --- _confidence ----------------------------------------------------------------

def test_confidence_levels():
    assert _confidence(10.0, []) == "none"
    assert _confidence(6.0, ["s"]) == "high"
    assert _confidence(3.0, ["s"]) == "medium"
    assert _confidence(0.5, ["s"]) == "low"


# --- _name_score ----------------------------------------------------------------

def test_name_score_rewards_title_brand_subcategory():
    product = {"title": "降噪豆", "brand": "某牌", "sub_category": "真无线耳机"}
    full = _name_score(_normalize("我想要降噪豆这款真无线耳机"), product)
    assert full >= 20  # title match dominates
    none = _name_score(_normalize("完全无关的查询"), product)
    assert none == 0.0


# --- _json_object ---------------------------------------------------------------

def test_json_object_parses_dict_and_rejects_non_dict():
    assert _json_object('{"a": 1}') == {"a": 1}
    assert _json_object("[1, 2, 3]") == {}


def test_json_object_extracts_embedded_object():
    assert _json_object('好的，结果是 {"dimensions": []} 完毕') == {"dimensions": []}


def test_json_object_returns_empty_on_garbage():
    assert _json_object("not json at all") == {}


# --- _is_price_dimension / _price_is_priority -----------------------------------

def test_is_price_dimension_detects_price_terms():
    assert _is_price_dimension("价格", ("价格",)) is True
    assert _is_price_dimension("性价比", ("便宜",)) is True
    assert _is_price_dimension("降噪", ("降噪",)) is False


def test_price_is_priority_from_filters_or_query():
    assert _price_is_priority("随便看看", SearchFilters(prefer_low_price=True)) is True
    assert _price_is_priority("哪个更便宜", SearchFilters()) is True
    assert _price_is_priority("哪个降噪更好", SearchFilters()) is False


def test_asks_for_current_two():
    assert _asks_for_current_two("这两款哪个好") is True
    assert _asks_for_current_two("前两个对比一下") is True
    assert _asks_for_current_two("随便推荐") is False


# --- _attribute_terms / _clean_attribute_label / context stripping --------------

def test_attribute_terms_includes_label_and_ngrams():
    terms = _attribute_terms("降噪")
    assert "降噪" in terms


def test_clean_attribute_label_strips_question_and_comparative_wrappers():
    assert _clean_attribute_label("这两款哪个降噪更好") == "降噪"


def test_strip_product_context_words_removes_brand_prefix():
    product = _product(brand="索尼")
    assert _strip_product_context_words("索尼降噪", [product]) == "降噪"


def test_is_noise_attribute_filters_short_digits_units_and_brands():
    product = _product(brand="某牌")
    assert _is_noise_attribute("x", [product]) is True  # too short
    assert _is_noise_attribute("123", [product]) is True  # numeric
    assert _is_noise_attribute("款式", [product]) is True  # leading unit char
    assert _is_noise_attribute("对比", [product]) is True  # noise word
    assert _is_noise_attribute("某牌", [product]) is True  # equals a brand
    assert _is_noise_attribute("降噪", [product]) is False


# --- _specs_from_llm_payload ----------------------------------------------------

def test_specs_from_llm_payload_keeps_corpus_backed_dimension():
    products = [_product(desc="降噪效果出色，地铁里非常安静")]
    payload = {"dimensions": [{"label": "降噪", "aliases": ["降噪", "噪音"], "preference": "higher_is_better"}]}

    specs = _specs_from_llm_payload(payload, "哪个降噪更好", products)

    assert any(spec.label == "降噪" for spec in specs)
    spec = next(spec for spec in specs if spec.label == "降噪")
    assert spec.preference == "higher_is_better"


def test_specs_from_llm_payload_rejects_non_list():
    assert _specs_from_llm_payload({"dimensions": "nope"}, "q", [_product()]) == []


def test_specs_from_llm_payload_drops_price_dimension():
    products = [_product(desc="价格实惠")]
    payload = {"dimensions": [{"label": "价格", "aliases": ["价格", "便宜"]}]}
    assert _specs_from_llm_payload(payload, "哪个便宜", products) == []


def test_specs_from_llm_payload_drops_dimension_without_corpus_support():
    products = [_product(desc="降噪效果出色")]  # no 防水 anywhere
    payload = {"dimensions": [{"label": "防水", "aliases": ["防水", "防泼溅"]}]}
    assert _specs_from_llm_payload(payload, "哪个防水", products) == []


def test_specs_from_llm_payload_infers_preference_when_invalid():
    products = [_product(desc="糖分很低，热量也低")]
    payload = {"dimensions": [{"label": "糖分", "aliases": ["糖分", "热量"], "preference": "bogus"}]}

    specs = _specs_from_llm_payload(payload, "哪个糖分更低", products)

    spec = next(spec for spec in specs if spec.label == "糖分")
    assert spec.preference == "lower_is_better"


# --- _dynamic_specs (deterministic fallback) ------------------------------------

def test_dynamic_specs_extracts_explicit_attribute_from_query():
    products = [_product(desc="降噪效果出色，地铁里非常安静")]

    specs = _dynamic_specs("哪个降噪更好", products)

    assert any(spec.label == "降噪" for spec in specs)


def test_dynamic_specs_without_products_is_empty():
    assert _dynamic_specs("哪个降噪更好", []) == []


# --- _evidence ------------------------------------------------------------------

def test_evidence_scores_positive_mentions_and_collects_snippets():
    product = _product(
        desc="降噪效果非常好",
        reviews=[{"rating": 5, "content": "地铁里降噪很安静，体验很好"}],
    )

    score, snippets = _evidence(product, ("降噪", "安静"), "higher_is_better")

    assert score > 0
    assert snippets  # at least one evidence snippet collected


def test_evidence_returns_zero_without_term_matches():
    product = _product(desc="完全无关的描述内容")
    score, snippets = _evidence(product, ("降噪",), "higher_is_better")
    assert score == 0.0
    assert snippets == []


# --- ComparisonService.is_comparison_query --------------------------------------

def test_is_comparison_query_by_ids_and_hints():
    svc = ComparisonService(_catalog(_product("p1"), _product("p2")))
    assert svc.is_comparison_query("随便", ["p1", "p2"]) is True  # two ids
    assert svc.is_comparison_query("这两款哪个更好", []) is True  # hint
    assert svc.is_comparison_query("推荐一款耳机", []) is False


# --- ComparisonService resolution / build --------------------------------------

def test_resolve_ordinals_maps_phrases_and_digit_pairs():
    svc = ComparisonService(_catalog(_product("p1"), _product("p2"), _product("p3")))
    assert svc._resolve_ordinals("第一个和第二个", ["p1", "p2", "p3"]) == ["p1", "p2"]
    assert svc._resolve_ordinals("1和3", ["p1", "p2", "p3"]) == ["p1", "p3"]
    assert svc._resolve_ordinals("第一个", []) == []


def test_resolve_names_matches_title_in_query():
    a = _product("p1", title="超静降噪豆Pro")
    b = _product("p2", title="轻量跑鞋X")
    svc = ComparisonService(_catalog(a, b))
    assert svc._resolve_names("我要超静降噪豆Pro", ["p1", "p2"]) == ["p1"]


def test_build_clarifies_when_no_context():
    svc = ComparisonService(_catalog(_product("p1"), _product("p2")))
    comparison = svc.build("哪个更好", SearchFilters(), explicit_product_ids=[], recent_product_ids=[])
    assert comparison.clarification is not None
    assert comparison.products == []
    assert "没有可对比" in comparison.clarification


def test_build_clarifies_when_only_one_product_known():
    svc = ComparisonService(_catalog(_product("p1"), _product("p2")))
    comparison = svc.build(
        "和这个比", SearchFilters(), explicit_product_ids=["p1"], recent_product_ids=[]
    )
    assert comparison.clarification is not None
    assert "另一款" in comparison.clarification


def test_build_price_comparison_picks_cheaper_product():
    cheap = _product("p1", title="A面霜", brand="甲", base_price=100.0, category="美妆护肤", sub_category="面霜", desc="保湿不错")
    pricey = _product("p2", title="B面霜", brand="乙", base_price=300.0, category="美妆护肤", sub_category="面霜", desc="保湿不错")
    svc = ComparisonService(_catalog(cheap, pricey))

    comparison = svc.build(
        "这两款哪个更便宜？",
        SearchFilters(prefer_low_price=True),
        explicit_product_ids=["p1", "p2"],
        recent_product_ids=[],
    )

    assert "价格" in comparison.focus
    assert comparison.winner_product_id == "p1"
    assert "元" in comparison.recommendation
    assert len(comparison.products) == 2
    dimensions = [row.dimension for row in comparison.rows]
    assert "基础定位" in dimensions
    assert "价格与SKU" in dimensions
    assert "规格明细" in dimensions


def test_build_evidence_comparison_produces_focus_row():
    strong = _product(
        "p1",
        title="强降噪豆",
        desc="降噪效果非常出色，地铁里很安静",
        reviews=[{"rating": 5, "content": "降噪强，通勤很安静，非常满意"}],
    )
    weak = _product(
        "p2",
        title="普通耳机",
        desc="音质均衡",
        reviews=[{"rating": 3, "content": "降噪一般，噪音still明显"}],
    )
    svc = ComparisonService(_catalog(strong, weak))

    comparison = svc.build(
        "这两款耳机哪个降噪更好？",
        SearchFilters(),
        explicit_product_ids=["p1", "p2"],
        recent_product_ids=[],
    )

    assert "降噪" in comparison.focus
    assert any(row.dimension == "降噪" for row in comparison.rows)
    assert comparison.winner_product_id in {"p1", None}
    assert comparison.recommendation
    assert comparison.summary
