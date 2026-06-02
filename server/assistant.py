"""Application orchestration for basic shopping recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

from server.catalog import CatalogHit, ProductCatalog
from server.intent import IntentParser, SearchFilters
from server.llm import ArkChatClient
from server.prompts import build_messages
from server.retrieval import ProductRetriever, RetrievalResult
from server.schemas import ChatResponse, ProductCard


@dataclass
class PreparedChat:
    query: str
    session_id: str | None
    filters: SearchFilters
    retrieval: RetrievalResult
    products: list[ProductCard]
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

    @property
    def catalog(self) -> ProductCatalog:
        return self._catalog

    def prepare(self, query: str, session_id: str | None, top_k: int) -> PreparedChat:
        filters = self._parser.parse(query)
        retrieval = self._retriever.retrieve(query=query, filters=filters, limit=top_k)
        hits = self._order_hits(retrieval.hits, filters)
        products = [
            self._catalog.product_card(hit.product, matched_reason=_reason(hit, filters), filters=filters)
            for hit in hits
        ]
        grounded_answer = self._grounded_answer(query, filters, hits)
        messages = build_messages(query, filters, hits, self._catalog)
        return PreparedChat(
            query=query,
            session_id=session_id,
            filters=filters,
            retrieval=retrieval,
            products=products,
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

    def answer(self, query: str, session_id: str | None, top_k: int) -> ChatResponse:
        prepared = self.prepare(query, session_id, top_k)
        warnings = list(prepared.retrieval.warnings)

        return ChatResponse(
            answer=prepared.grounded_answer,
            products=prepared.products,
            session_id=session_id,
            intent=prepared.filters.to_dict(),
            retrieval_source=prepared.retrieval.source,
            degraded=bool(prepared.retrieval.warnings),
            warnings=warnings,
        )

    def stream_answer(self, prepared: PreparedChat) -> Iterator[str]:
        yield from _chunk_text(prepared.grounded_answer)

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

        count = len(hits[:3])
        order_note = "，已按价格从低到高排列" if filters.prefer_low_price else ""
        if count == 1:
            return f"我从商品库里找到1款符合条件的商品{order_note}。价格、规格和推荐理由都放在下方商品卡里，多规格商品以卡片/详情里的SKU价格为准。"
        return f"我从商品库里筛出{count}款符合条件的商品{order_note}。你可以直接看下方商品卡的价格、规格和推荐理由，多规格商品以卡片/详情里的SKU价格为准。"


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
