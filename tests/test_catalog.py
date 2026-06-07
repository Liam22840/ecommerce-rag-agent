import json
from pathlib import Path

import pytest

from server.catalog import ProductCatalog
from server.intent import IntentParser, SearchFilters


DATASET_ROOT = Path(__file__).parent.parent / "ecommerce_agent_dataset"


def _product(
    pid: str = "p1",
    title: str = "测试面霜",
    brand: str = "甲牌",
    category: str = "美妆护肤",
    sub_category: str = "面霜",
    desc: str = "日常护理",
    skus=None,
    faq=None,
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
        "skus": skus or [],
        "rag_knowledge": {
            "marketing_description": desc,
            "official_faq": faq or [],
            "user_reviews": reviews or [],
        },
    }


def _catalog(*products: dict) -> ProductCatalog:
    return ProductCatalog({p["product_id"]: p for p in products})


def test_catalog_loads_all_products_and_builds_cards():
    catalog = ProductCatalog.load(DATASET_ROOT)

    product = catalog.require("p_beauty_011")
    card = catalog.product_card(product)

    assert len(catalog.categories) == 4
    assert card.product_id == "p_beauty_011"
    assert card.price == 52.0
    assert card.detail_path == "/api/products/p_beauty_011"


def test_catalog_exposes_sku_safe_price_labels():
    catalog = ProductCatalog.load(DATASET_ROOT)

    product = catalog.require("p_beauty_007")
    card = catalog.product_card(product)

    assert card.price == 89.0
    assert card.price_label == "89元起（15g 体验装）"
    assert card.price_summary == "15g 体验装 89元；50g 标准装 268元"
    assert card.lowest_price_sku is not None
    assert card.lowest_price_sku.label == "15g 体验装"


def test_catalog_uses_requested_sku_price_when_specified():
    catalog = ProductCatalog.load(DATASET_ROOT)
    parser = IntentParser(catalog.categories, catalog.sub_categories, catalog.brands)
    filters = parser.parse("推荐50g适合敏感肌的保湿霜，cheaper is better")

    product = catalog.require("p_beauty_007")
    card = catalog.product_card(product, filters=filters)

    assert card.price == 268.0
    assert card.price_label == "268元（50g 标准装）"
    assert card.selected_price_sku is not None
    assert card.selected_price_sku.label == "50g 标准装"


def test_lexical_search_handles_basic_skin_care_query():
    catalog = ProductCatalog.load(DATASET_ROOT)
    parser = IntentParser(catalog.categories, catalog.sub_categories, catalog.brands)
    filters = parser.parse("推荐一款适合油皮的洗面奶")

    hits = catalog.search_lexical("推荐一款适合油皮的洗面奶", filters, limit=3)

    assert hits
    assert hits[0].product["sub_category"] == "洁面"
    assert hits[0].product["product_id"] == "p_beauty_011"


def test_lexical_search_respects_budget_filter():
    catalog = ProductCatalog.load(DATASET_ROOT)
    parser = IntentParser(catalog.categories, catalog.sub_categories, catalog.brands)
    filters = parser.parse("200 元以下的蓝牙耳机有哪些？")

    hits = catalog.search_lexical("200 元以下的蓝牙耳机有哪些？", filters, limit=5)

    assert hits == []


def test_sensitive_skin_search_ranks_evidenced_first_without_dropping_others():
    catalog = ProductCatalog.load(DATASET_ROOT)
    parser = IntentParser(catalog.categories, catalog.sub_categories, catalog.brands)
    filters = parser.parse("推荐50g适合敏感肌的保湿霜，cheaper is better")

    hits = catalog.search_lexical("推荐50g适合敏感肌的保湿霜，cheaper is better", filters, limit=5)
    ids = [hit.product["product_id"] for hit in hits]

    # required_terms no longer hard-filter: the sensitive-evidenced cream ranks first, but the
    # other 50g cream still surfaces (ranked below) rather than being silently dropped.
    assert ids[0] == "p_beauty_007"
    assert "p_beauty_008" in ids


def test_requested_spec_ranks_matching_first_without_dropping_others():
    # Regression: a requested spec (e.g. 16寸) must rank matching products first but NOT drop the
    # rest, so "能装16寸笔记本的包" returns bags instead of an empty list.
    fits = _product(
        "p1", title="大容量背包", category="服饰运动", sub_category="背包",
        skus=[{"sku_id": "s1", "properties": {"规格": "16寸隔层"}, "price": 699.0}],
    )
    plain = _product("p2", title="普通背包", category="服饰运动", sub_category="背包", base_price=599.0)
    catalog = _catalog(fits, plain)
    filters = SearchFilters(sub_category="背包", requested_specs=["16寸"])

    # not a hard gate: the non-matching bag is still eligible (would have been False before)
    assert catalog.matches_filters(fits, filters) is True
    assert catalog.matches_filters(plain, filters) is True

    hits = catalog.search_lexical("背包", filters, limit=5)
    ids = [hit.product["product_id"] for hit in hits]
    assert "p2" in ids                       # not dropped
    assert ids.index("p1") < ids.index("p2")  # spec-matching ranks above


def test_unmet_requested_specs_flags_specs_no_hit_matches():
    plain = _product("p1", title="普通背包", category="服饰运动", sub_category="背包")
    catalog = _catalog(plain)
    filters = SearchFilters(requested_specs=["16寸"])
    hits = catalog.search_lexical("背包", filters, limit=5)
    assert catalog.unmet_requested_specs(hits, filters) == ["16寸"]


def test_violates_excluded_skips_negated_own_copy():
    # Regression: "不要油腻" must not flag a cream that advertises being "不油腻" (the deterministic
    # fallback for the LLM exclusion judge).
    catalog = _catalog(_product("p1", title="清爽面霜", desc="质地清爽不油腻不闷痘"))
    assert catalog.violates_excluded(catalog.require("p1"), ["油腻"]) is False


def test_violates_excluded_flags_positive_claim():
    catalog = _catalog(_product("p1", title="厚重面霜", desc="质地油腻厚重滋养"))
    assert catalog.violates_excluded(catalog.require("p1"), ["油腻"]) is True


def test_violates_excluded_ignores_third_party_reviews():
    catalog = _catalog(
        _product("p1", title="清爽面霜", desc="清爽水润", reviews=[{"rating": 3, "content": "我觉得有点油腻"}])
    )
    assert catalog.violates_excluded(catalog.require("p1"), ["油腻"]) is False


def test_matches_filters_no_longer_gates_excluded_terms():
    # excluded_terms moved out of the retrieval gate (now an LLM judge + violates_excluded fallback
    # over the shortlist), so matches_filters ignores it; excluded_brands stays a hard gate.
    catalog = _catalog(_product("p1", desc="含有酒精成分"))
    assert catalog.matches_filters(catalog.require("p1"), SearchFilters(excluded_terms=["酒精"])) is True


# --- construction / loading ----------------------------------------------------

def test_empty_catalog_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        ProductCatalog({})


def test_brand_aliases_canonicalised_at_load():
    # Same company entered under two names ("Nike"/"耐克") is merged at load so a brand filter on
    # one name finds the products stored under the other (otherwise "Nike的跑鞋" returns nothing).
    catalog = _catalog(
        _product("p1", brand="Nike", sub_category="跑步鞋"),
        _product("p2", brand="耐克", sub_category="篮球鞋"),
    )
    assert "Nike" not in catalog.brands
    assert catalog.brands == {"耐克"}
    assert catalog.get("p1")["brand"] == "耐克"


def test_get_and_require_behaviour():
    catalog = _catalog(_product("p1"))
    assert catalog.get("p1")["product_id"] == "p1"
    assert catalog.get("missing") is None
    with pytest.raises(KeyError):
        catalog.require("missing")


def test_load_raises_on_invalid_json(tmp_path):
    data_dir = tmp_path / "1_cat" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "bad.json").write_text("{not valid", encoding="utf-8")
    with pytest.raises(ValueError, match="failed to load"):
        ProductCatalog.load(tmp_path)


def test_load_raises_when_product_id_missing(tmp_path):
    data_dir = tmp_path / "1_cat" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "p.json").write_text(json.dumps({"title": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing product_id"):
        ProductCatalog.load(tmp_path)


# --- price / sku helpers --------------------------------------------------------

def test_lowest_price_falls_back_to_base_price_without_skus():
    catalog = _catalog(_product(base_price=42.0))
    product = catalog.require("p1")
    assert catalog.lowest_price(product) == 42.0


def test_lowest_price_uses_min_numeric_sku_price():
    skus = [
        {"sku_id": "s1", "properties": {"规格": "大"}, "price": 200.0},
        {"sku_id": "s2", "properties": {"规格": "小"}, "price": 120.0},
        {"sku_id": "s3", "properties": {"规格": "坏"}, "price": "n/a"},  # non-numeric ignored
    ]
    catalog = _catalog(_product(skus=skus))
    product = catalog.require("p1")
    assert catalog.lowest_price(product) == 120.0


def test_sku_prices_defaults_to_base_price_when_no_skus():
    catalog = _catalog(_product(base_price=80.0))
    prices = catalog.sku_prices(catalog.require("p1"))
    assert prices == [{"sku_id": None, "label": "默认规格", "price": 80.0}]


def test_price_label_single_sku_shows_label():
    skus = [{"sku_id": "s1", "properties": {"规格": "标准"}, "price": 99.0}]
    catalog = _catalog(_product(skus=skus))
    assert catalog.price_label(catalog.require("p1")) == "99元（标准）"


def test_price_label_uniform_multi_sku_lists_labels():
    skus = [
        {"sku_id": "s1", "properties": {"色": "红"}, "price": 50.0},
        {"sku_id": "s2", "properties": {"色": "蓝"}, "price": 50.0},
    ]
    catalog = _catalog(_product(skus=skus))
    label = catalog.price_label(catalog.require("p1"))
    assert label.startswith("50元（")
    assert "红" in label and "蓝" in label


def test_price_label_varied_multi_sku_shows_from_price():
    skus = [
        {"sku_id": "s1", "properties": {"规格": "小"}, "price": 50.0},
        {"sku_id": "s2", "properties": {"规格": "大"}, "price": 90.0},
    ]
    catalog = _catalog(_product(skus=skus))
    assert catalog.price_label(catalog.require("p1")) == "50元起（小）"


def test_price_label_with_requested_spec_shows_selected_sku():
    skus = [
        {"sku_id": "s1", "properties": {"规格": "15g"}, "price": 50.0},
        {"sku_id": "s2", "properties": {"规格": "50g"}, "price": 90.0},
    ]
    catalog = _catalog(_product(skus=skus))
    filters = SearchFilters(requested_specs=["50g"])
    assert catalog.price_label(catalog.require("p1"), filters) == "90元（50g）"


def test_price_summary_joins_all_skus():
    skus = [
        {"sku_id": "s1", "properties": {"规格": "小"}, "price": 50.0},
        {"sku_id": "s2", "properties": {"规格": "大"}, "price": 90.0},
    ]
    catalog = _catalog(_product(skus=skus))
    assert catalog.price_summary(catalog.require("p1")) == "小 50元；大 90元"


def test_selected_price_sku_matches_requested_spec():
    skus = [
        {"sku_id": "s1", "properties": {"规格": "15g"}, "price": 50.0},
        {"sku_id": "s2", "properties": {"规格": "50g"}, "price": 90.0},
    ]
    catalog = _catalog(_product(skus=skus))
    selected = catalog.selected_price_sku(catalog.require("p1"), SearchFilters(requested_specs=["50g"]))
    assert selected["label"] == "50g"
    # No matching spec falls back to the cheapest SKU.
    fallback = catalog.selected_price_sku(catalog.require("p1"), SearchFilters(requested_specs=["999g"]))
    assert fallback["label"] == "15g"


# --- filtering ------------------------------------------------------------------

def test_matches_filters_min_price_excludes_cheaper():
    catalog = _catalog(_product(base_price=100.0))
    product = catalog.require("p1")
    assert catalog.matches_filters(product, SearchFilters(min_price=150.0)) is False
    assert catalog.matches_filters(product, SearchFilters(min_price=50.0)) is True


def test_matches_filters_brand_and_excluded_brand():
    catalog = _catalog(_product(brand="甲牌"))
    product = catalog.require("p1")
    assert catalog.matches_filters(product, SearchFilters(brand="乙牌")) is False
    assert catalog.matches_filters(product, SearchFilters(excluded_brands=["甲牌"])) is False
    assert catalog.matches_filters(product, SearchFilters(brand="甲牌")) is True


def test_violates_excluded_matches_positive_and_skips_absent():
    catalog = _catalog(_product(desc="含有酒精成分"))
    product = catalog.require("p1")
    assert catalog.violates_excluded(product, ["酒精"]) is True
    assert catalog.violates_excluded(product, ["香精"]) is False


def test_matches_filters_category_and_subcategory():
    catalog = _catalog(_product(category="美妆护肤", sub_category="面霜"))
    product = catalog.require("p1")
    assert catalog.matches_filters(product, SearchFilters(category="数码电子")) is False
    assert catalog.matches_filters(product, SearchFilters(sub_category="洁面")) is False
    assert catalog.matches_filters(product, SearchFilters(category="美妆护肤", sub_category="面霜")) is True


# --- ratings & sensitive-skin scoring ------------------------------------------

def test_avg_rating_ignores_non_numeric_ratings():
    product = {"rag_knowledge": {"user_reviews": [{"rating": 5}, {"rating": "bad"}, {"rating": 3}]}}
    assert ProductCatalog.avg_rating(product) == 4.0


def test_required_terms_no_longer_hard_filter():
    # A product that doesn't evidence the attribute is NOT dropped — required_terms rank, not gate.
    catalog = _catalog(_product(title="普通面霜", desc="温和"))
    product = catalog.require("p1")
    assert catalog.matches_filters(product, SearchFilters(required_terms=["敏感肌"])) is True


def test_sensitive_skin_strong_signal_is_evidenced():
    catalog = _catalog(_product(desc="专为敏感肌打造，温和不刺激"))
    product = catalog.require("p1")
    assert catalog.evidences_required_term(product, "敏感肌") is True


def test_sensitive_skin_weak_only_signal_is_not_evidenced():
    # "敏感肌需先" is a weak/negative cue with no strong signal -> not evidenced (ranks low, not dropped).
    catalog = _catalog(_product(title="普通面霜", desc="敏感肌需先做耐受测试"))
    product = catalog.require("p1")
    assert catalog.evidences_required_term(product, "敏感肌") is False


def test_required_term_moisturizing_alias_is_evidenced():
    catalog = _catalog(_product(title="补水面霜", desc="深层补水锁水"))
    product = catalog.require("p1")
    assert catalog.evidences_required_term(product, "保湿") is True


def test_product_facts_truncates_faq_and_reviews_to_three():
    faq = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(5)]
    reviews = [{"rating": 5, "content": f"c{i}"} for i in range(5)]
    catalog = _catalog(_product(faq=faq, reviews=reviews))
    facts = catalog.product_facts(catalog.require("p1"))
    assert len(facts["faq"]) == 3
    assert len(facts["reviews"]) == 3
    assert facts["sku_count"] == 0
