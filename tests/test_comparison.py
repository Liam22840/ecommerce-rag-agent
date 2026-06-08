"""Unit tests for the comparison engine: pure helpers and resolution logic."""

from __future__ import annotations

from server.catalog import ProductCatalog
from server.comparison import ComparisonService
from server.comparison.dimensions import (
    _attribute_terms,
    _clean_attribute_label,
    _corpus_backed_query_ngrams,
    _dynamic_specs,
    _is_noise_attribute,
    _is_price_dimension,
    _price_is_priority,
    _specs_from_llm_payload,
    _strip_product_context_words,
)
from server.comparison.evidence import (
    _best_snippet,
    _confidence,
    _evidence,
    _generic_polarity,
    _judge_confidence,
    _winner_from_scores,
)
from server.comparison.resolver import _asks_for_current_two, _best_ref_match, _name_score
from server.comparison.text import _chunks
from server.intent import SearchFilters
from server.textutil import json_object, normalize, trim


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


# --- normalize / trim / _chunks ----------------------------------------------

def test_normalize_strips_punctuation_and_lowercases():
    assert normalize("A·B, C。D（E）") == "abcde"
    assert normalize("降 噪_效-果") == "降噪效果"


def test_trim_truncates_with_ellipsis():
    assert trim("abc", 10) == "abc"
    assert trim("abcdef", 4) == "abc…"
    assert trim("  padded  ", 10) == "padded"


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
    full = _name_score(normalize("我想要降噪豆这款真无线耳机"), product)
    assert full >= 20  # title match dominates
    none = _name_score(normalize("完全无关的查询"), product)
    assert none == 0.0


# --- json_object ---------------------------------------------------------------

def test_json_object_parses_dict_and_rejects_non_dict():
    assert json_object('{"a": 1}') == {"a": 1}
    assert json_object("[1, 2, 3]") == {}


def test_json_object_extracts_embedded_object():
    assert json_object('好的，结果是 {"dimensions": []} 完毕') == {"dimensions": []}


def test_json_object_returns_empty_on_garbage():
    assert json_object("not json at all") == {}


# --- _is_price_dimension / _price_is_priority -----------------------------------

def test_is_price_dimension_detects_price_terms():
    assert _is_price_dimension("价格", ("价格",)) is True
    assert _is_price_dimension("性价比", ("便宜",)) is True
    assert _is_price_dimension("降噪", ("降噪",)) is False


def test_price_is_priority_follows_prefer_low_price_flag():
    # Price priority comes from the LLM-set preference, not query keyword spotting.
    assert _price_is_priority(SearchFilters(prefer_low_price=True)) is True
    assert _price_is_priority(SearchFilters()) is False


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


def test_specs_from_llm_payload_skips_non_dict_entries():
    products = [_product(desc="降噪效果出色，地铁里非常安静")]
    payload = {"dimensions": ["not-a-dict", {"label": "降噪", "aliases": ["降噪"]}]}
    specs = _specs_from_llm_payload(payload, "哪个降噪更好", products)
    assert [spec.label for spec in specs] == ["降噪"]


def test_specs_from_llm_payload_dedupes_repeated_label():
    products = [_product(desc="降噪效果出色，地铁里非常安静")]
    payload = {"dimensions": [
        {"label": "降噪", "aliases": ["降噪"]},
        {"label": "降噪", "aliases": ["降噪"]},
    ]}
    specs = _specs_from_llm_payload(payload, "哪个降噪更好", products)
    assert [spec.label for spec in specs] == ["降噪"]


def test_specs_from_llm_payload_coerces_non_list_aliases():
    products = [_product(desc="降噪效果出色，地铁里非常安静")]
    payload = {"dimensions": [{"label": "降噪", "aliases": "降噪"}]}  # aliases should be a list
    specs = _specs_from_llm_payload(payload, "哪个降噪更好", products)
    assert any(spec.label == "降噪" for spec in specs)


def test_specs_from_llm_payload_drops_noise_label():
    # The LLM occasionally emits a meta word ("对比") as a dimension, it must be dropped.
    products = [_product(desc="降噪效果出色")]
    payload = {"dimensions": [{"label": "对比", "aliases": ["对比"]}]}
    assert _specs_from_llm_payload(payload, "哪个更好", products) == []


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


def test_dynamic_specs_falls_back_to_corpus_ngrams_without_comparative_pattern():
    # A bare attribute with no "更好/看重" wrapper has no explicit candidate, so the fallback
    # mines corpus-backed n-grams from the query instead.
    products = [_product(desc="降噪效果出色，地铁里非常安静")]
    specs = _dynamic_specs("降噪", products)
    assert any("降噪" in spec.label for spec in specs)


def test_corpus_backed_query_ngrams_keeps_only_corpus_terms():
    products = [_product(desc="降噪效果出色")]
    corpus = normalize("降噪效果出色 测试耳机 某牌")
    ngrams = _corpus_backed_query_ngrams(normalize("降噪不存在词"), corpus, products)
    assert "降噪" in ngrams
    assert "不存在" not in ngrams  # not in the corpus


def test_strip_product_context_words_drops_exact_match_and_suffix():
    product = _product(brand="森海")
    assert _strip_product_context_words("森海", [product]) == ""      # whole label is the brand
    assert _strip_product_context_words("降噪森海", [product]) == "降噪"  # brand as a suffix


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


def test_best_snippet_prefers_matching_chunk_else_first():
    text = "佩戴很舒适。降噪表现优秀。续航也不错。"
    assert "降噪" in _best_snippet(text, ("降噪",))
    # no term matches -> fall back to the first chunk rather than returning nothing
    assert _best_snippet(text, ("不存在的词",)) == "佩戴很舒适"


def test_best_snippet_empty_text_is_empty():
    assert _best_snippet("", ("降噪",)) == ""


def test_judge_confidence_clamps_and_downgrades():
    # An unrecognised confidence string is normalised to "medium"...
    assert _judge_confidence("bogus", reason="有依据", grounded=True) == "medium"
    # ...but an ungrounded high/medium claim is downgraded so we never over-claim.
    assert _judge_confidence("high", reason="有依据", grounded=False) == "low"
    # No reason and no grounding -> no confidence at all.
    assert _judge_confidence("high", reason="", grounded=False) == "none"


# --- resolver: _best_ref_match --------------------------------------------------

def test_best_ref_match_empty_ref_returns_none():
    assert _best_ref_match("", [_product()]) is None
    assert _best_ref_match("   ", [_product()]) is None


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


def test_resolve_ordinals_handles_front_two_phrase():
    svc = ComparisonService(_catalog(_product("p1"), _product("p2"), _product("p3")))
    assert svc._resolve_ordinals("前两个对比", ["p1", "p2", "p3"]) == ["p1", "p2"]


def test_resolve_ordinals_handles_a_and_b_phrase():
    svc = ComparisonService(_catalog(_product("p1"), _product("p2")))
    assert svc._resolve_ordinals("A和B哪个好", ["p1", "p2"]) == ["p1", "p2"]


def test_title_and_price_subject_fall_back_to_id_when_unknown():
    svc = ComparisonService(_catalog(_product("p1")))
    assert svc._title("p_missing", []) == "p_missing"
    assert svc._price_subject("p_missing", [], SearchFilters()) == "p_missing"


def test_resolve_product_ids_picks_up_two_ids_from_text():
    a, b = _product("p_test_001"), _product("p_test_002")
    svc = ComparisonService(_catalog(a, b))
    ids, clarification = svc._resolve_product_ids("对比 p_test_001 和 p_test_002", [], [])
    assert ids == ["p_test_001", "p_test_002"]
    assert clarification is None


def test_resolve_product_ids_falls_to_name_matching():
    a = _product("p1", title="超静降噪豆Pro")
    b = _product("p2", title="轻量跑鞋X")
    svc = ComparisonService(_catalog(a, b))
    ids, clarification = svc._resolve_product_ids("超静降噪豆Pro和轻量跑鞋X谁好", [], ["p1", "p2"])
    assert set(ids) == {"p1", "p2"}
    assert clarification is None


def test_resolve_product_ids_uses_current_two_when_referenced():
    svc = ComparisonService(_catalog(_product("p1"), _product("p2")))
    ids, clarification = svc._resolve_product_ids("对比这两个", [], ["p1", "p2"])
    assert ids == ["p1", "p2"]
    assert clarification is None


def test_build_trusts_validated_llm_compare_ids():
    # LLM is primary: it resolves the referenced products to ids (it sees session_products
    # newest-first), and the service trusts them after validating against the catalog.
    svc = ComparisonService(_catalog(_product("p1"), _product("p2"), _product("p3")))

    comparison = svc.build(
        "对比这两个",
        SearchFilters(intent_type="comparison", compare_product_ids=["p1", "p3"]),
        explicit_product_ids=[],
        recent_product_ids=["p1", "p2", "p3"],
    )

    assert {product.product_id for product in comparison.products} == {"p1", "p3"}


def test_ordinals_resolve_against_recency_as_fallback():
    # Fallback when the LLM gives no compare ids: the deterministic ordinal resolver counts
    # against the recency-ordered list (most recent first), so "第一个和第二个" = the latest two.
    svc = ComparisonService(_catalog(_product("p_new1"), _product("p_new2"), _product("p_old1")))

    comparison = svc.build(
        "第一个和第二个对比下",
        SearchFilters(intent_type="comparison"),
        explicit_product_ids=[],
        recent_product_ids=["p_new1", "p_new2", "p_old1"],
    )

    assert {product.product_id for product in comparison.products} == {"p_new1", "p_new2"}


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


# --- LLM-resolved compare ids (LLM-first, dictionary-as-fallback) ---------------

def test_llm_resolved_compare_ids_win_over_the_dictionary():
    a = _product("p1", title="A面霜", category="美妆护肤", sub_category="面霜")
    b = _product("p2", title="B面霜", category="美妆护肤", sub_category="面霜")
    svc = ComparisonService(_catalog(a, b))

    # Phrasing the ORDINALS dict cannot parse, and no recent context, so only the LLM-resolved
    # ids (validated against the catalog) can pin this comparison.
    comparison = svc.build(
        "排第一的那个和最后那个哪个更保湿",
        SearchFilters(intent_type="comparison", compare_product_ids=["p1", "p2"]),
        explicit_product_ids=[],
        recent_product_ids=[],
    )
    assert [product.product_id for product in comparison.products] == ["p1", "p2"]


def test_invalid_llm_compare_ids_fall_back_to_the_dictionary_waterfall():
    a = _product("p1", title="A面霜", category="美妆护肤", sub_category="面霜")
    b = _product("p2", title="B面霜", category="美妆护肤", sub_category="面霜")
    svc = ComparisonService(_catalog(a, b))

    # A hallucinated id is dropped by the catalog check, so resolution falls through to the
    # deterministic ordinal dictionary against the recent products.
    comparison = svc.build(
        "第一个和第二个哪个更保湿",
        SearchFilters(intent_type="comparison", compare_product_ids=["p_nope"]),
        explicit_product_ids=[],
        recent_product_ids=["p1", "p2"],
    )
    assert [product.product_id for product in comparison.products] == ["p1", "p2"]
