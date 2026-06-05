"""Application orchestration for basic shopping recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

from server.comparison import ComparisonService
from server.catalog import CatalogHit, ProductCatalog
from server.intent import IntentParser, SearchFilters
from server.llm import ArkChatClient
from server.prompts import build_messages
from server.retrieval import ProductRetriever, RetrievalResult
from server.schemas import ChatResponse, ProductCard, ProductComparison


@dataclass
class PreparedChat:
    query: str
    session_id: str | None
    filters: SearchFilters
    retrieval: RetrievalResult
    products: list[ProductCard]
    comparison: ProductComparison | None
    grounded_answer: str
    messages: list[dict[str, str]]


class ShoppingAssistant:
    def __init__(
        self,
        catalog: ProductCatalog,
        retriever: ProductRetriever,
        llm: ArkChatClient | None = None,
    ):
        self._catalog = catalog
        self._retriever = retriever
        self._llm = llm
        self._parser = IntentParser(catalog.categories, catalog.sub_categories, catalog.brands)
        self._comparison = ComparisonService(catalog, llm=llm)
        self._recent_product_ids_by_session: dict[str, list[str]] = {}

    @property
    def catalog(self) -> ProductCatalog:
        return self._catalog

    def prepare(
        self,
        query: str,
        session_id: str | None,
        top_k: int,
        compare_product_ids: list[str] | None = None,
        client_recent_product_ids: list[str] | None = None,
    ) -> PreparedChat:
        filters = self._parser.parse(query)
        recent_product_ids = self._recent_product_ids(session_id, client_recent_product_ids or [])
        compare_product_ids = compare_product_ids or []

        if self._comparison.is_comparison_query(query, compare_product_ids):
            comparison = self._comparison.build(
                query=query,
                filters=filters,
                explicit_product_ids=compare_product_ids,
                recent_product_ids=recent_product_ids,
            )
            products = comparison.products
            grounded_answer = self._comparison_answer(comparison)
            retrieval = RetrievalResult(hits=[], source="lexical")
            self._remember_recent_products(session_id, [product.product_id for product in products])
            return PreparedChat(
                query=query,
                session_id=session_id,
                filters=filters,
                retrieval=retrieval,
                products=products,
                comparison=comparison,
                grounded_answer=grounded_answer,
                messages=[],
            )

        retrieval = self._retriever.retrieve(query=query, filters=filters, limit=top_k)
        hits = self._order_hits(retrieval.hits, filters)
        products = [
            self._catalog.product_card(hit.product, matched_reason=_reason(hit, filters), filters=filters)
            for hit in hits
        ]
        self._remember_recent_products(session_id, [product.product_id for product in products])
        grounded_answer = self._grounded_answer(query, filters, hits)
        messages = build_messages(query, filters, hits, self._catalog)
        return PreparedChat(
            query=query,
            session_id=session_id,
            filters=filters,
            retrieval=retrieval,
            products=products,
            comparison=None,
            grounded_answer=grounded_answer,
            messages=messages,
        )

    def _order_hits(self, hits: list[CatalogHit], filters: SearchFilters) -> list[CatalogHit]:
        if not filters.prefer_low_price:
            return hits
        return sorted(hits, key=lambda hit: self._display_price(hit.product, filters))

    def _display_price(self, product: dict, filters: SearchFilters) -> float:
        selected_sku = self._catalog.selected_price_sku(product, filters)
        if selected_sku:
            return float(selected_sku["price"])
        return self._catalog.lowest_price(product)

    def answer(
        self,
        query: str,
        session_id: str | None,
        top_k: int,
        compare_product_ids: list[str] | None = None,
        client_recent_product_ids: list[str] | None = None,
    ) -> ChatResponse:
        prepared = self.prepare(query, session_id, top_k, compare_product_ids, client_recent_product_ids)
        warnings = list(prepared.retrieval.warnings)

        return ChatResponse(
            answer=prepared.grounded_answer,
            products=prepared.products,
            comparison=prepared.comparison,
            session_id=session_id,
            intent=prepared.filters.to_dict(),
            retrieval_source=prepared.retrieval.source,
            degraded=bool(prepared.retrieval.warnings),
            warnings=warnings,
        )

    def stream_answer(self, prepared: PreparedChat) -> Iterator[str]:
        yield from _chunk_text(prepared.grounded_answer)

    def _recent_product_ids(self, session_id: str | None, client_recent_product_ids: list[str]) -> list[str]:
        stored = self._recent_product_ids_by_session.get(session_id or "", [])
        return _dedupe_ids(client_recent_product_ids + stored)

    def _remember_recent_products(self, session_id: str | None, product_ids: list[str]) -> None:
        if not session_id or not product_ids:
            return
        existing = self._recent_product_ids_by_session.get(session_id, [])
        self._recent_product_ids_by_session[session_id] = _dedupe_ids(product_ids + existing)[:10]

    def _comparison_answer(self, comparison: ProductComparison) -> str:
        if comparison.clarification:
            return comparison.clarification
        lines = [comparison.summary]
        for row in comparison.rows:
            values = []
            for value in row.values:
                title = next(
                    (product.title for product in comparison.products if product.product_id == value.product_id),
                    value.product_id,
                )
                values.append(f"{title}：{value.value}")
            lines.append(f"- {row.dimension}：{'；'.join(values)}。{row.verdict}")
        lines.append(comparison.recommendation)
        lines.append("以上对比仅使用当前商品库的结构化字段、描述、问答和评价证据；证据不足处不会做绝对判断。")
        return "\n".join(lines)

    def _grounded_answer(
        self,
        query: str,
        filters: SearchFilters,
        hits: list[CatalogHit],
    ) -> str:
        if not hits:
            constraints = []
            if filters.sub_category:
                constraints.append(filters.sub_category)
            if filters.category:
                constraints.append(filters.category)
            if filters.max_price is not None:
                constraints.append(f"{filters.max_price:g}元以内")
            constraints.extend(filters.required_terms)
            condition = "、".join(constraints) if constraints else query
            return f"没有在商品库中找到完全匹配“{condition}”的商品。可以放宽预算、换一个类目，或补充你更看重的功能。"

        order_note = "，并按价格从低到高排列" if filters.prefer_low_price else ""
        lines = [f"我按你的条件从商品库里筛选出以下{len(hits[:3])}款{order_note}："]
        for idx, hit in enumerate(hits[:3], start=1):
            product = hit.product
            reason = _reason(hit, filters)
            price_label = self._catalog.price_label(product, filters)
            price_summary = self._catalog.price_summary(product)
            line = f"{idx}. {product['title']}，{product['brand']}，价格：{price_label}。"
            if price_summary and price_summary != price_label:
                line += f"SKU价格明细：{price_summary}。"
            line += f"{reason}。"
            lines.append(line)
        lines.append("以上商品名、品牌、类目、SKU和价格均来自当前商品库；多规格商品以SKU价格明细为准。")
        return "\n".join(lines)


def _reason(hit: CatalogHit, filters: SearchFilters) -> str:
    reasons = []
    product = hit.product
    if filters.sub_category and product["sub_category"] == filters.sub_category:
        reasons.append(f"符合{filters.sub_category}需求")
    if filters.category and product["category"] == filters.category:
        reasons.append(f"属于{filters.category}")
    if filters.max_price is not None:
        reasons.append(f"价格在{filters.max_price:g}元以内")
    for term in filters.required_terms:
        reasons.append(f"匹配{term}需求")
    if filters.prefer_low_price:
        reasons.append("优先低价")
    if hit.snippets:
        reasons.append("商品描述或评价中有相关信息")
    return "，".join(reasons) if reasons else "与当前需求语义匹配"


def _chunk_text(text: str, chunk_size: int = 18) -> Iterator[str]:
    for idx in range(0, len(text), chunk_size):
        yield text[idx: idx + chunk_size]


def _dedupe_ids(product_ids: list[str]) -> list[str]:
    seen = set()
    result = []
    for product_id in product_ids:
        if product_id and product_id not in seen:
            seen.add(product_id)
            result.append(product_id)
    return result
