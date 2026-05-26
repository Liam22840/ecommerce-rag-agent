from pathlib import Path

from server.catalog import ProductCatalog
from server.intent import IntentParser


DATASET_ROOT = Path(__file__).parent.parent / "ecommerce_agent_dataset"


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


def test_sensitive_skin_search_requires_positive_sensitive_fit():
    catalog = ProductCatalog.load(DATASET_ROOT)
    parser = IntentParser(catalog.categories, catalog.sub_categories, catalog.brands)
    filters = parser.parse("推荐50g适合敏感肌的保湿霜，cheaper is better")

    hits = catalog.search_lexical("推荐50g适合敏感肌的保湿霜，cheaper is better", filters, limit=5)

    assert [hit.product["product_id"] for hit in hits] == ["p_beauty_007"]
