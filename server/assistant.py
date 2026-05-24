"""Application orchestration for basic shopping recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

from server.catalog import CatalogHit, ProductCatalog
from server.intent import IntentParser, SearchFilters
from server.llm import ArkChatClient, ModelUnavailable
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
    fallback_answer: str
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
        products = [
            self._catalog.product_card(hit.product, matched_reason=_reason(hit, filters))
            for hit in retrieval.hits
        ]
        fallback_answer = self._fallback_answer(query, filters, retrieval.hits)
        messages = build_messages(query, filters, retrieval.hits, self._catalog)
        return PreparedChat(
            query=query,
            session_id=session_id,
            filters=filters,
            retrieval=retrieval,
            products=products,
            fallback_answer=fallback_answer,
            messages=messages,
        )

    def answer(self, query: str, session_id: str | None, top_k: int) -> ChatResponse:
        prepared = self.prepare(query, session_id, top_k)
        warnings = list(prepared.retrieval.warnings)
        degraded = False
        try:
            if self._llm is None or not self._llm.available:
                raise ModelUnavailable("chat model unavailable")
            answer = self._llm.complete(prepared.messages)
        except Exception as exc:  # noqa: BLE001
            degraded = True
            warnings.append(f"LLM unavailable; used deterministic answer: {exc}")
            answer = prepared.fallback_answer

        return ChatResponse(
            answer=answer,
            products=prepared.products,
            session_id=session_id,
            intent=prepared.filters.to_dict(),
            retrieval_source=prepared.retrieval.source,
            degraded=degraded or bool(prepared.retrieval.warnings),
            warnings=warnings,
        )

    def stream_answer(self, prepared: PreparedChat) -> Iterator[str]:
        if self._llm is None or not self._llm.available:
            yield from _chunk_text(prepared.fallback_answer)
            return
        try:
            yielded = False
            for token in self._llm.stream(prepared.messages):
                yielded = True
                yield token
            if not yielded:
                yield from _chunk_text(prepared.fallback_answer)
        except Exception:
            yield from _chunk_text(prepared.fallback_answer)

    def _fallback_answer(
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
            condition = "、".join(constraints) if constraints else query
            return f"没有在商品库中找到完全匹配“{condition}”的商品。可以放宽预算、换一个类目，或补充你更看重的功能。"

        lines = ["我在商品库里找到这些更匹配的选择："]
        for idx, hit in enumerate(hits[:3], start=1):
            product = hit.product
            price = self._catalog.lowest_price(product)
            reason = _reason(hit, filters)
            lines.append(
                f"{idx}. {product['title']}，{product['brand']}，{price:g}元起。{reason}"
            )
        lines.append("以上信息仅来自当前商品库；我没有使用未提供的优惠或库存信息。")
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
    if hit.snippets:
        reasons.append("商品描述或评价中有相关信息")
    return "，".join(reasons) if reasons else "与当前需求语义匹配"


def _chunk_text(text: str, chunk_size: int = 18) -> Iterator[str]:
    for idx in range(0, len(text), chunk_size):
        yield text[idx: idx + chunk_size]
