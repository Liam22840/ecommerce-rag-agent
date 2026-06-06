"""Application orchestration for basic shopping recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator

from server.comparison import ComparisonService
from server.catalog import CatalogHit, ProductCatalog
from server.intent import IntentParser, SearchFilters
from server.llm import ArkChatClient, ModelUnavailable
from server.prompts import (
    CHITCHAT_REPLY,
    build_messages,
    chitchat_messages,
    comparison_narration_messages,
)
from server.retrieval import ProductRetriever, RetrievalResult
from server.schemas import ChatResponse, ProductCard, ProductComparison
from server.textutil import dedupe_ids


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


@dataclass
class SessionState:
    """Per-session short-term memory: products shown so far and the last product-search
    filters, so follow-up turns can resolve references and inherit search context."""

    recent_product_ids: list[str] = field(default_factory=list)
    last_filters: SearchFilters | None = None


class ShoppingAssistant:
    def __init__(
        self,
        catalog: ProductCatalog,
        retriever: ProductRetriever,
        llm: ArkChatClient | None = None,
        intent_llm: ArkChatClient | None = None,
    ):
        self._catalog = catalog
        self._retriever = retriever
        self._llm = llm
        self._parser = IntentParser(
            catalog.categories, catalog.sub_categories, catalog.brands, llm=intent_llm
        )
        self._comparison = ComparisonService(catalog, llm=llm)
        self._sessions: dict[str, SessionState] = {}

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
        filters = self._parser.parse(query, previous_filters=self._previous_filters(session_id))
        recent_product_ids = self._recent_product_ids(session_id, client_recent_product_ids or [])
        compare_product_ids = compare_product_ids or []

        if len(compare_product_ids) >= 2 or filters.intent_type == "comparison":
            comparison = self._comparison.build(
                query=query,
                filters=filters,
                explicit_product_ids=compare_product_ids,
                recent_product_ids=recent_product_ids,
            )
            products = comparison.products
            grounded_answer = self._comparison_answer(comparison)
            # Let the LLM narrate the (deterministic) comparison result; the template above is
            # the fallback. No messages for a clarification — there is nothing to narrate.
            messages = [] if comparison.clarification else comparison_narration_messages(comparison)
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
                messages=messages,
            )

        if filters.intent_type == "chitchat":
            # LLM handles the conversation (kept in character by the system prompt); the fixed
            # reply is the fallback when the model is unavailable.
            return PreparedChat(
                query=query,
                session_id=session_id,
                filters=filters,
                retrieval=RetrievalResult(hits=[], source="none"),
                products=[],
                comparison=None,
                grounded_answer=CHITCHAT_REPLY,
                messages=chitchat_messages(query),
            )

        # The rewrite folds carried context into a standalone retrieval query; the answer
        # itself still replies to what the user actually typed (raw query below).
        search_query = filters.rewritten_query or query
        retrieval = self._retriever.retrieve(query=search_query, filters=filters, limit=top_k)
        hits = self._order_hits(retrieval.hits, filters)
        products = [
            self._catalog.product_card(hit.product, matched_reason=_reason(hit, filters), filters=filters)
            for hit in hits
        ]
        self._remember_recent_products(session_id, [product.product_id for product in products])
        self._remember_filters(session_id, filters)
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
        if filters.sort_by == "price_asc":
            return sorted(hits, key=lambda hit: self._display_price(hit.product, filters))
        if filters.sort_by == "price_desc":
            return sorted(hits, key=lambda hit: self._display_price(hit.product, filters), reverse=True)
        if filters.sort_by == "rating_desc":
            return sorted(hits, key=lambda hit: self._catalog.avg_rating(hit.product), reverse=True)
        return hits

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

        answer_text = prepared.grounded_answer
        if prepared.messages and self._llm is not None and self._llm.available:
            try:
                answer_text = self._llm.complete(prepared.messages)
            except ModelUnavailable as exc:
                warnings.append(f"LLM unavailable, using grounded fallback: {exc}")

        return ChatResponse(
            answer=answer_text,
            products=prepared.products,
            comparison=prepared.comparison,
            session_id=session_id,
            intent=prepared.filters.to_dict(),
            retrieval_source=prepared.retrieval.source,
            degraded=bool(warnings),
            warnings=warnings,
        )

    def stream_answer(self, prepared: PreparedChat) -> Iterator[str]:
        if prepared.messages and self._llm is not None and self._llm.available:
            streamed = False
            try:
                for token in self._llm.stream(prepared.messages):
                    streamed = True
                    yield token
                return
            except Exception:  # noqa: BLE001 - stream must degrade, not crash the response
                if streamed:
                    # Partial answer already sent; ending beats duplicating it with the fallback.
                    return
        yield from _chunk_text(prepared.grounded_answer)

    def _session(self, session_id: str | None) -> SessionState:
        return self._sessions.setdefault(session_id or "", SessionState())

    def _recent_product_ids(self, session_id: str | None, client_recent_product_ids: list[str]) -> list[str]:
        stored = self._session(session_id).recent_product_ids
        return dedupe_ids(client_recent_product_ids + stored)

    def _remember_recent_products(self, session_id: str | None, product_ids: list[str]) -> None:
        if not session_id or not product_ids:
            return
        state = self._session(session_id)
        state.recent_product_ids = dedupe_ids(product_ids + state.recent_product_ids)[:10]

    def _previous_filters(self, session_id: str | None) -> SearchFilters | None:
        return self._session(session_id).last_filters

    def _remember_filters(self, session_id: str | None, filters: SearchFilters) -> None:
        if not session_id:
            return
        self._session(session_id).last_filters = filters

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
