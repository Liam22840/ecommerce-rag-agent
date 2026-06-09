"""Shared cart and order pricing helpers.

All commerce totals come from catalog ProductCard/SKU data through this module, not from
LLM text or user-provided cart labels.
"""

from __future__ import annotations

import os

from server.catalog import ProductCatalog
from server.intent import SearchFilters
from server.schemas import CartItem, ProductCard

# Upper bound on a single cart line's quantity. A pathological "要1000000件" clamps to this
# (env-overridable) instead of pricing an absurd subtotal; the cart then shows the capped count.
MAX_CART_QUANTITY = int(os.environ.get("RAG_MAX_CART_QUANTITY", "999"))


def cart_card(catalog: ProductCatalog, product: dict, sku_id: str | None = None) -> ProductCard:
    # The current iOS cart has no SKU picker. When sku_id is absent we deliberately use the
    # catalog's selected/lowest SKU and preserve its label in price_label.
    filters = SearchFilters()
    card = catalog.product_card(product, filters=filters)
    if sku_id is None:
        return card

    for sku in catalog.sku_prices(product):
        if sku.get("sku_id") != sku_id:
            continue
        selected = dict(sku)
        card.selected_price_sku = card.selected_price_sku.__class__(**selected)
        card.price = float(selected["price"])
        card.price_label = f"{float(selected['price']):g}元（{selected['label']}）"
        return card
    return card


def build_cart_item(
    catalog: ProductCatalog,
    product: dict,
    quantity: int,
    sku_id: str | None = None,
) -> CartItem:
    card = cart_card(catalog, product, sku_id)
    selected = card.selected_price_sku or card.lowest_price_sku
    unit_price = float(selected.price if selected else card.price)
    qty = max(0, min(int(quantity), MAX_CART_QUANTITY))
    return CartItem(
        product_id=card.product_id,
        sku_id=selected.sku_id if selected else sku_id,
        quantity=qty,
        product=card,
        unit_price=unit_price,
        price_label=card.price_label,
        line_total=round(unit_price * qty, 2),
    )


def cart_subtotal(items: list[CartItem]) -> float:
    return round(sum(item.line_total for item in items), 2)


def money(value: float) -> str:
    return f"{value:g}元"
