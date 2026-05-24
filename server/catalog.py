"""Product catalog loading, card projection, and local fallback retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from server.intent import SUB_CATEGORY_ALIASES, SearchFilters
from server.schemas import ProductCard


@dataclass
class CatalogHit:
    product: dict[str, Any]
    score: float
    snippets: list[str] = field(default_factory=list)
    source: str = "lexical"


class ProductCatalog:
    def __init__(self, products: dict[str, dict[str, Any]]):
        if not products:
            raise ValueError("product catalog is empty")
        self._products = products

    @classmethod
    def load(cls, dataset_root: Path) -> "ProductCatalog":
        files = sorted(Path(dataset_root).glob("*/data/*.json"))
        products: dict[str, dict[str, Any]] = {}
        for path in files:
            try:
                product = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"failed to load product JSON {path}: {exc}") from exc
            product_id = product.get("product_id")
            if not product_id:
                raise ValueError(f"product JSON {path} missing product_id")
            products[product_id] = product
        return cls(products)

    @property
    def categories(self) -> set[str]:
        return {p["category"] for p in self._products.values()}

    @property
    def sub_categories(self) -> set[str]:
        return {p["sub_category"] for p in self._products.values()}

    @property
    def brands(self) -> set[str]:
        return {p["brand"] for p in self._products.values()}

    def get(self, product_id: str) -> dict[str, Any] | None:
        return self._products.get(product_id)

    def require(self, product_id: str) -> dict[str, Any]:
        product = self.get(product_id)
        if product is None:
            raise KeyError(product_id)
        return product

    def product_card(self, product: dict[str, Any], matched_reason: str | None = None) -> ProductCard:
        return ProductCard(
            product_id=product["product_id"],
            title=product["title"],
            brand=product["brand"],
            category=product["category"],
            sub_category=product["sub_category"],
            price=self.lowest_price(product),
            image_path=product["image_path"],
            detail_path=f"/api/products/{product['product_id']}",
            matched_reason=matched_reason,
        )

    def product_facts(self, product: dict[str, Any]) -> dict[str, Any]:
        reviews = product.get("rag_knowledge", {}).get("user_reviews", [])
        faqs = product.get("rag_knowledge", {}).get("official_faq", [])
        return {
            "product_id": product["product_id"],
            "title": product["title"],
            "brand": product["brand"],
            "category": product["category"],
            "sub_category": product["sub_category"],
            "price": self.lowest_price(product),
            "sku_count": len(product.get("skus", [])),
            "description": product.get("rag_knowledge", {}).get("marketing_description", ""),
            "faq": faqs[:3],
            "reviews": reviews[:3],
        }

    def search_lexical(self, query: str, filters: SearchFilters, limit: int) -> list[CatalogHit]:
        hits: list[CatalogHit] = []
        query_terms = _query_terms(query, filters)
        for product in self._products.values():
            if not self.matches_filters(product, filters):
                continue
            score, snippets = self._score_product(product, query_terms, filters)
            if score > 0:
                hits.append(CatalogHit(product=product, score=score, snippets=snippets))

        hits.sort(key=lambda hit: (hit.score, -self.lowest_price(hit.product)), reverse=True)
        return hits[:limit]

    def matches_filters(self, product: dict[str, Any], filters: SearchFilters) -> bool:
        price = self.lowest_price(product)
        if filters.max_price is not None and price > filters.max_price:
            return False
        if filters.min_price is not None and price < filters.min_price:
            return False
        if filters.category and product["category"] != filters.category:
            return False
        if filters.sub_category and product["sub_category"] != filters.sub_category:
            return False
        if filters.brand and product["brand"] != filters.brand:
            return False
        if product["brand"] in filters.excluded_brands:
            return False

        haystack = self._haystack(product)
        for term in filters.excluded_terms:
            if term and term in haystack:
                return False
        return True

    @staticmethod
    def lowest_price(product: dict[str, Any]) -> float:
        sku_prices = [
            float(sku["price"])
            for sku in product.get("skus", [])
            if isinstance(sku.get("price"), int | float)
        ]
        if sku_prices:
            return min(sku_prices)
        return float(product["base_price"])

    def _score_product(
        self,
        product: dict[str, Any],
        query_terms: list[str],
        filters: SearchFilters,
    ) -> tuple[float, list[str]]:
        score = 0.0
        snippets: list[str] = []
        haystack = self._haystack(product)
        title = product["title"].lower()
        description = product.get("rag_knowledge", {}).get("marketing_description", "")

        if filters.category and product["category"] == filters.category:
            score += 6
        if filters.sub_category and product["sub_category"] == filters.sub_category:
            score += 10
        if filters.brand and product["brand"] == filters.brand:
            score += 5

        for term in query_terms:
            term_l = term.lower()
            if not term_l:
                continue
            if term_l in title:
                score += 4
                snippets.append(product["title"])
            elif term in product["brand"]:
                score += 3
            elif term in product["sub_category"] or term in product["category"]:
                score += 3
            elif term in haystack:
                score += 1
                if description:
                    snippets.append(description[:120])

        if not query_terms and (filters.category or filters.sub_category or filters.max_price):
            score += 1
        return score, _dedupe(snippets)[:3]

    def _haystack(self, product: dict[str, Any]) -> str:
        parts = [
            product.get("title", ""),
            product.get("brand", ""),
            product.get("category", ""),
            product.get("sub_category", ""),
            product.get("rag_knowledge", {}).get("marketing_description", ""),
        ]
        for qa in product.get("rag_knowledge", {}).get("official_faq", []):
            parts.extend([qa.get("question", ""), qa.get("answer", "")])
        for review in product.get("rag_knowledge", {}).get("user_reviews", []):
            parts.append(review.get("content", ""))
        return "\n".join(parts)


def _query_terms(query: str, filters: SearchFilters) -> list[str]:
    terms = [query.strip()]
    if filters.category:
        terms.append(filters.category)
    if filters.sub_category:
        terms.append(filters.sub_category)
        terms.extend(SUB_CATEGORY_ALIASES.get(filters.sub_category, []))
    if filters.brand:
        terms.append(filters.brand)

    for token in ["油皮", "干皮", "敏感肌", "保湿", "控油", "轻量", "续航", "拍照", "降噪", "防水", "户外"]:
        if token in query:
            terms.append(token)
    return _dedupe([term for term in terms if term])


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out

