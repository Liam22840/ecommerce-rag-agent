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

