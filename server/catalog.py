"""Product catalog loading, card projection, and local fallback retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from server.intent import SUB_CATEGORY_ALIASES, SearchFilters
from server.schemas import ProductCard, SkuPrice


REQUIRED_TERM_ALIASES: dict[str, list[str]] = {
    "敏感肌": ["敏感肌", "敏感性", "敏感皮", "干敏", "易敏", "敏皮"],
    "保湿": ["保湿", "补水", "锁水", "滋润"],
}

STRONG_SENSITIVE_SIGNALS = [
    "专为敏感肌",
    "专为干性敏感肌",
    "专为干敏肌",
    "敏感肌打造",
    "敏感肌友好",
    "敏感肌可放心",
    "敏感肌放心",
    "敏感肌福音",
    "干敏皮福音",
    "干敏肌救星",
    "易敏肌适用",
    "通过敏感肌测试",
    "大部分敏感肌",
    "敏感肌可使用",
    "敏感肌能用",
    "对敏感肌友好",
]

WEAK_SENSITIVE_ONLY_SIGNALS = [
    "敏感肌需先",
    "敏感肌先",
    "敏感肌建议先",
    "敏感肌需谨慎",
    "敏感肌谨慎",
    "敏感肌慎入",
]


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

    @property
    def products(self) -> list[dict[str, Any]]:
        return list(self._products.values())

    def get(self, product_id: str) -> dict[str, Any] | None:
        return self._products.get(product_id)

    def require(self, product_id: str) -> dict[str, Any]:
        product = self.get(product_id)
        if product is None:
            raise KeyError(product_id)
        return product

    def product_card(
        self,
        product: dict[str, Any],
        matched_reason: str | None = None,
        filters: SearchFilters | None = None,
    ) -> ProductCard:
        selected_sku = self.selected_price_sku(product, filters)
        lowest_sku = self.lowest_price_sku(product)
        return ProductCard(
            product_id=product["product_id"],
            title=product["title"],
            brand=product["brand"],
            category=product["category"],
            sub_category=product["sub_category"],
            price=selected_sku["price"] if selected_sku else self.lowest_price(product),
            price_label=self.price_label(product, filters),
            price_summary=self.price_summary(product),
            lowest_price_sku=SkuPrice(**lowest_sku) if lowest_sku else None,
            selected_price_sku=SkuPrice(**selected_sku) if selected_sku else None,
            image_path=product["image_path"],
            detail_path=f"/api/products/{product['product_id']}",
            matched_reason=matched_reason,
        )

    def product_facts(self, product: dict[str, Any], filters: SearchFilters | None = None) -> dict[str, Any]:
        reviews = product.get("rag_knowledge", {}).get("user_reviews", [])
        faqs = product.get("rag_knowledge", {}).get("official_faq", [])
        selected_sku = self.selected_price_sku(product, filters)
        return {
            "product_id": product["product_id"],
            "title": product["title"],
            "brand": product["brand"],
            "category": product["category"],
            "sub_category": product["sub_category"],
            "lowest_price": self.lowest_price(product),
            "price_label": self.price_label(product, filters),
            "price_summary": self.price_summary(product),
            "lowest_price_sku": self.lowest_price_sku(product),
            "selected_price_sku": selected_sku,
            "sku_prices": self.sku_prices(product),
            "sku_count": len(product.get("skus", [])),
            "price_instruction": (
                "Use price_label or price_summary verbatim. If the title contains a spec that differs "
                "from the lowest_price_sku label, do not attach the lowest price to the title spec."
            ),
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
        if filters.requested_specs and not self.matches_requested_specs(product, filters.requested_specs):
            return False
        for term in filters.required_terms:
            if not self._matches_required_term(product, term, haystack):
                return False
        for term in filters.excluded_terms:
            if term and term in haystack:
                return False
        return True

    @staticmethod
    def avg_rating(product: dict[str, Any]) -> float:
        ratings = [
            float(review["rating"])
            for review in product.get("rag_knowledge", {}).get("user_reviews", [])
            if isinstance(review.get("rating"), int | float)
        ]
        return sum(ratings) / len(ratings) if ratings else 0.0

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

    def sku_prices(self, product: dict[str, Any]) -> list[dict[str, Any]]:
        sku_prices = []
        for sku in product.get("skus", []):
            price = sku.get("price")
            if not isinstance(price, int | float):
                continue
            properties = sku.get("properties") or {}
            label_parts = [str(value).strip() for value in properties.values() if str(value).strip()]
            label = " ".join(label_parts) if label_parts else "默认规格"
            sku_prices.append({
                "sku_id": sku.get("sku_id"),
                "label": label,
                "price": float(price),
            })
        if sku_prices:
            return sorted(sku_prices, key=lambda item: (item["price"], item["label"]))
        return [{
            "sku_id": None,
            "label": "默认规格",
            "price": float(product["base_price"]),
        }]

    def lowest_price_sku(self, product: dict[str, Any]) -> dict[str, Any] | None:
        sku_prices = self.sku_prices(product)
        return sku_prices[0] if sku_prices else None

    def selected_price_sku(
        self,
        product: dict[str, Any],
        filters: SearchFilters | None = None,
    ) -> dict[str, Any] | None:
        if filters and filters.requested_specs:
            for item in self.sku_prices(product):
                normalized_label = _normalize_spec_text(item["label"])
                if all(spec in normalized_label for spec in filters.requested_specs):
                    return item
        return self.lowest_price_sku(product)

    def price_label(self, product: dict[str, Any], filters: SearchFilters | None = None) -> str:
        sku_prices = self.sku_prices(product)
        if not sku_prices:
            return f"{float(product['base_price']):g}元"

        selected = self.selected_price_sku(product, filters) or sku_prices[0]
        if filters and filters.requested_specs:
            return f"{selected['price']:g}元（{selected['label']}）"

        lowest = selected
        prices = {item["price"] for item in sku_prices}
        if len(sku_prices) == 1:
            return f"{lowest['price']:g}元（{lowest['label']}）"
        if len(prices) == 1:
            labels = " / ".join(item["label"] for item in sku_prices[:3])
            return f"{lowest['price']:g}元（{labels}）"
        return f"{lowest['price']:g}元起（{lowest['label']}）"

    def price_summary(self, product: dict[str, Any]) -> str:
        return "；".join(
            f"{item['label']} {item['price']:g}元"
            for item in self.sku_prices(product)
        )

    def matches_requested_specs(self, product: dict[str, Any], requested_specs: list[str]) -> bool:
        if not requested_specs:
            return True
        normalized_title = _normalize_spec_text(product.get("title", ""))
        normalized_skus = [_normalize_spec_text(item["label"]) for item in self.sku_prices(product)]
        return all(
            spec in normalized_title or any(spec in label for label in normalized_skus)
            for spec in requested_specs
        )

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

        score += self._required_term_score(product, filters)

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

    def _required_term_score(self, product: dict[str, Any], filters: SearchFilters) -> float:
        if not filters.required_terms:
            return 0.0

        score = 0.0
        haystack = self._haystack(product)
        title = product["title"]
        description = product.get("rag_knowledge", {}).get("marketing_description", "")

        if "敏感肌" in filters.required_terms:
            score += self._sensitive_relevance_score(product, haystack) * 4

        if "保湿" in filters.required_terms:
            if "保湿" in title or "补水" in title:
                score += 5
            if any(term in description for term in ["保湿", "补水", "锁水", "滋润"]):
                score += 4

        return score

    def _matches_required_term(self, product: dict[str, Any], term: str, haystack: str) -> bool:
        if term == "敏感肌":
            return self._sensitive_relevance_score(product, haystack) > 0
        aliases = REQUIRED_TERM_ALIASES.get(term, [term])
        return any(alias in haystack for alias in aliases)

    def _sensitive_relevance_score(self, product: dict[str, Any], haystack: str | None = None) -> float:
        haystack = haystack if haystack is not None else self._haystack(product)
        title = product["title"]
        description = product.get("rag_knowledge", {}).get("marketing_description", "")
        score = 0.0
        if "敏感肌" in title or "干敏" in title or "易敏" in title:
            score += 2
        if any(signal in description for signal in STRONG_SENSITIVE_SIGNALS):
            score += 2
        elif any(signal in haystack for signal in STRONG_SENSITIVE_SIGNALS):
            score += 1
        if any(signal in description for signal in WEAK_SENSITIVE_ONLY_SIGNALS):
            score -= 2
        return score

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


def _normalize_spec_text(value: str) -> str:
    return "".join(str(value).lower().split())
