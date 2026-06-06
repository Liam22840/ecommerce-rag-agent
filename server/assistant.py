"""Application orchestration for basic shopping recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator
from typing import Literal

from server.comparison import ComparisonService
from server.catalog import CatalogHit, ProductCatalog
from server.config import Settings
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


# Outcome of a product-search turn, used to narrate honestly instead of re-listing.
ResultStatus = Literal["ok", "no_results", "no_cheaper", "no_improvement", "exhausted"]

# Fields of a shown product remembered for the turn history and the recall log.
_SHOWN_FIELDS = ("id", "title", "brand", "price", "sub_category")


def _product_summary(product: ProductCard) -> dict:
    """Compact record of a shown product, used for the turn history and the recall log."""
    return {
        "id": product.product_id,
        "title": product.title,
        "brand": product.brand,
        "price": product.price,
        "sub_category": product.sub_category,
    }


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
    result_status: ResultStatus = "ok"


@dataclass
class TurnRecord:
    """One product-search turn: the resolved filters and a compact record of what was
    shown, so later turns can resolve relative refinements and backtracking."""

    query: str
    filters: SearchFilters
    shown: list[dict]  # [{"id","title","brand","price","sub_category"}], a few items


@dataclass
class SessionState:
    """Per-session short-term memory: products shown so far (for reference resolution)
    and a short history of recent product-search turns (for carry-over, relative
    refinements and backtracking)."""

    turns: list[TurnRecord] = field(default_factory=list)
    # Single source of truth for "what products has the user seen": session-wide, deduped by
    # id, in first-shown order, each tagged with the turn (last_seq) and within-turn position
    # it was most recently shown. Backtracking reads it in first-shown order; the recency view
    # for comparison ("第一个/前两个") is derived from it (last_seq desc, position asc).
    shown_products: list[dict] = field(default_factory=list)
    turn_seq: int = 0


class ShoppingAssistant:
    def __init__(
        self,
        catalog: ProductCatalog,
        retriever: ProductRetriever,
        llm: ArkChatClient | None = None,
        intent_llm: ArkChatClient | None = None,
        settings: Settings | None = None,
    ):
        self._catalog = catalog
        self._retriever = retriever
        self._llm = llm
        self._settings = settings or Settings()
        self._parser = IntentParser(
            catalog.categories,
            catalog.sub_categories,
            catalog.brands,
            llm=intent_llm,
            approx_price_tolerance=self._settings.approx_price_tolerance,
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
        filters = self._parser.parse(
            query,
            previous_filters=self._previous_filters(session_id),
            history=self._history_summaries(session_id),
            session_products=self._session_products(session_id),
        )
        recent_product_ids = self._recent_product_ids(session_id, client_recent_product_ids or [])
        compare_product_ids = compare_product_ids or []

        if len(compare_product_ids) >= 2 or filters.intent_type == "comparison":
            return self._prepare_comparison(
                query, session_id, filters, compare_product_ids, recent_product_ids
            )
        if filters.intent_type == "chitchat":
            return self._prepare_chitchat(query, session_id, filters)
        return self._prepare_search(query, session_id, filters, top_k, recent_product_ids)

    def _prepare_comparison(
        self,
        query: str,
        session_id: str | None,
        filters: SearchFilters,
        compare_product_ids: list[str],
        recent_product_ids: list[str],
    ) -> PreparedChat:
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
        self._remember_shown_products(session_id, products)
        return PreparedChat(
            query=query,
            session_id=session_id,
            filters=filters,
            retrieval=RetrievalResult(hits=[], source="lexical"),
            products=products,
            comparison=comparison,
            grounded_answer=grounded_answer,
            messages=messages,
        )

    def _prepare_chitchat(
        self, query: str, session_id: str | None, filters: SearchFilters
    ) -> PreparedChat:
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
            messages=chitchat_messages(query, self._catalog.scope_summary()),
        )

    def _prepare_search(
        self,
        query: str,
        session_id: str | None,
        filters: SearchFilters,
        top_k: int,
        recent_product_ids: list[str],
    ) -> PreparedChat:
        # Backtracking ("回到最开始那个"): the LLM picked exact product ids from session_products;
        # return those cards directly. Ids are validated against the catalog (the LLM can only
        # copy from the list we gave it, but we never trust an id we can't resolve).
        recalled = [pid for pid in filters.recall_product_ids if self._catalog.get(pid) is not None]
        if recalled:
            return self._prepare_recall(query, session_id, filters, recalled)
        # The rewrite folds carried context into a standalone retrieval query; the answer
        # itself still replies to what the user actually typed (raw query below).
        search_query = filters.rewritten_query or query
        # For a "换一批" turn, over-fetch so dropping already-seen items still fills top_k.
        limit = top_k + len(recent_product_ids) if filters.exclude_seen else top_k
        retrieval = self._retriever.retrieve(query=search_query, filters=filters, limit=limit)
        hits = self._order_hits(retrieval.hits, filters)
        if filters.exclude_seen:
            seen = set(recent_product_ids)
            hits = [hit for hit in hits if hit.product["product_id"] not in seen]
        hits = hits[:top_k]
        products = [
            self._catalog.product_card(hit.product, matched_reason=_reason(hit, filters, self._catalog), filters=filters)
            for hit in hits
        ]
        prev_floor = self._previous_floor(session_id)
        result_status = self._result_status(filters, products, recent_product_ids, prev_floor)
        context = self._status_context(result_status, products, prev_floor)
        # Required attributes no longer hard-filter; flag any that nothing retrieved clearly
        # evidences, so the answer says so honestly instead of implying every card matches.
        unmet = self._catalog.unmet_required_terms(hits, filters)
        if unmet:
            context = {**(context or {}), "unmet_terms": unmet}
        return self._search_prepared(
            query, session_id, filters, hits, products, retrieval, result_status, context
        )

    def _prepare_recall(
        self, query: str, session_id: str | None, filters: SearchFilters, product_ids: list[str]
    ) -> PreparedChat:
        hits = [CatalogHit(product=self._catalog.require(pid), score=0.0) for pid in product_ids]
        products = [
            self._catalog.product_card(hit.product, matched_reason="你之前看过的商品", filters=filters)
            for hit in hits
        ]
        return self._search_prepared(
            query, session_id, filters, hits, products, RetrievalResult(hits=hits, source="lexical")
        )

    def _search_prepared(
        self,
        query: str,
        session_id: str | None,
        filters: SearchFilters,
        hits: list[CatalogHit],
        products: list[ProductCard],
        retrieval: RetrievalResult,
        result_status: ResultStatus = "ok",
        context: dict | None = None,
    ) -> PreparedChat:
        # Shared tail for the search and recall paths: record the turn, narrate, package.
        self._remember_shown_products(session_id, products)
        self._remember_turn(session_id, query, filters, products)
        grounded_answer = self._grounded_answer(query, filters, hits, result_status, context)
        messages = build_messages(query, filters, hits, self._catalog, result_status, context)
        return PreparedChat(
            query=query,
            session_id=session_id,
            filters=filters,
            retrieval=retrieval,
            products=products,
            comparison=None,
            grounded_answer=grounded_answer,
            messages=messages,
            result_status=result_status,
        )

    def _result_status(
        self,
        filters: SearchFilters,
        products: list[ProductCard],
        recent_product_ids: list[str],
        prev_floor: float | None,
    ) -> ResultStatus:
        if not products:
            if filters.exclude_seen:
                return "exhausted"
            # A "便宜一点的" that tightened below the last shown floor and found nothing.
            if filters.prefer_low_price and prev_floor is not None:
                return "no_cheaper"
            return "no_results"
        seen = set(recent_product_ids)
        # A refinement that surfaced only items already shown earlier — nothing new/better.
        if seen and all(product.product_id in seen for product in products):
            return "no_cheaper" if filters.prefer_low_price and prev_floor is not None else "no_improvement"
        return "ok"

    def _status_context(
        self, result_status: str, products: list[ProductCard], prev_floor: float | None
    ) -> dict | None:
        if result_status == "no_cheaper" and prev_floor is not None:
            return {"cheapest_shown": prev_floor}
        if result_status == "no_improvement" and products:
            return {"cheapest_shown": min(product.price for product in products)}
        return None

    def _previous_floor(self, session_id: str | None) -> float | None:
        turns = self._session(session_id).turns
        if not turns or not turns[-1].shown:
            return None
        return min(item["price"] for item in turns[-1].shown)

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
        yield from _chunk_text(prepared.grounded_answer, self._settings.stream_chunk_size)

    def _session(self, session_id: str | None) -> SessionState:
        return self._sessions.setdefault(session_id or "", SessionState())

    def _shown_by_recency(self, session_id: str | None) -> list[dict]:
        # Single source of truth for the "most recent turn first, display order within a turn"
        # ordering. The intent prompt ("最近展示的排在最前") and the deterministic ordinal
        # fallback both rely on this exact order, so it lives in one place to avoid drift.
        shown = self._session(session_id).shown_products
        return sorted(shown, key=lambda item: (-item["last_seq"], item["position"]))

    def _recent_product_ids(self, session_id: str | None, client_recent_product_ids: list[str]) -> list[str]:
        # Recency view derived from the single shown-products log so comparison's "第一个/前两个"
        # resolve correctly. Client-provided ids stay in front (restart-resilience: the app
        # remembers what it showed).
        ordered = self._shown_by_recency(session_id)
        derived = [item["id"] for item in ordered][: self._settings.recent_products_cap]
        return dedupe_ids(client_recent_product_ids + derived)

    def _previous_filters(self, session_id: str | None) -> SearchFilters | None:
        turns = self._session(session_id).turns
        return turns[-1].filters if turns else None

    def _remember_turn(
        self,
        session_id: str | None,
        query: str,
        filters: SearchFilters,
        products: list[ProductCard],
    ) -> None:
        if not session_id:
            return
        shown = [_product_summary(product) for product in products[: self._settings.shown_summary_cap]]
        turns = self._session(session_id).turns
        turns.append(TurnRecord(query=query, filters=filters, shown=shown))
        del turns[: -self._settings.history_turns]  # keep the most recent N turns

    def _remember_shown_products(self, session_id: str | None, products: list[ProductCard]) -> None:
        """Record shown products in the single session-wide log. New products are appended in
        first-shown order (for recall); a re-shown product keeps its place but updates its
        last_seq/position (so the derived recency view stays correct)."""
        if not session_id or not products:
            return
        state = self._session(session_id)
        state.turn_seq += 1
        seq = state.turn_seq
        by_id = {item["id"]: item for item in state.shown_products}
        for position, product in enumerate(products):
            existing = by_id.get(product.product_id)
            if existing is not None:
                existing["last_seq"] = seq
                existing["position"] = position
                continue
            entry = {**_product_summary(product), "last_seq": seq, "position": position}
            state.shown_products.append(entry)
            by_id[product.product_id] = entry
        self._cap_shown_products(state)

    def _cap_shown_products(self, state: SessionState) -> None:
        # Keep the most-recently-shown N distinct products, but preserve first-shown order in
        # the stored list so recall ("最开始那个") still reads oldest-first.
        cap = self._settings.session_products_cap
        if len(state.shown_products) <= cap:
            return
        keep = {
            item["id"]
            for item in sorted(state.shown_products, key=lambda e: (e["last_seq"], e["position"]), reverse=True)[:cap]
        }
        state.shown_products = [item for item in state.shown_products if item["id"] in keep]

    def _session_products(self, session_id: str | None) -> list[dict] | None:
        # Compact view for the intent LLM (drops the internal seq/position bookkeeping), in the
        # shared recency order so the LLM resolves "第一个/第二个" against the latest search.
        ordered = self._shown_by_recency(session_id)
        if not ordered:
            return None
        return [{key: entry[key] for key in _SHOWN_FIELDS} for entry in ordered]

    def _history_summaries(self, session_id: str | None) -> list[dict] | None:
        turns = self._session(session_id).turns[-self._settings.history_turns :]
        if not turns:
            return None
        return [
            {
                "query": turn.query,
                "category": turn.filters.category,
                "sub_category": turn.filters.sub_category,
                "brand": turn.filters.brand,
                "min_price": turn.filters.min_price,
                "max_price": turn.filters.max_price,
                "required_terms": turn.filters.required_terms,
                "shown": turn.shown,
            }
            for turn in turns
        ]

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
        result_status: str = "ok",
        context: dict | None = None,
    ) -> str:
        if result_status == "exhausted":
            return "没有更多没看过的商品了。可以换个类目，或调整一下需求再看看。"
        if result_status == "no_cheaper":
            cheapest = (context or {}).get("cheapest_shown")
            floor = f"，最低约{cheapest:g}元" if cheapest is not None else ""
            return f"已经没有更便宜的了{floor}。可以换个类目，或放宽其它条件再看看。"
        if result_status == "no_improvement":
            return "这些已经是当前最匹配的结果了，没有更合适的了。可以换个类目或调整需求。"
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
        unmet = (context or {}).get("unmet_terms")
        if unmet:
            lines = [f"没有在商品库里找到明确标注“{'、'.join(unmet)}”的商品，以下是最接近的{len(hits[:3])}款{order_note}："]
        else:
            lines = [f"我按你的条件从商品库里筛选出以下{len(hits[:3])}款{order_note}："]
        for idx, hit in enumerate(hits[:3], start=1):
            product = hit.product
            reason = _reason(hit, filters, self._catalog)
            price_label = self._catalog.price_label(product, filters)
            price_summary = self._catalog.price_summary(product)
            line = f"{idx}. {product['title']}，{product['brand']}，价格：{price_label}。"
            if price_summary and price_summary != price_label:
                line += f"SKU价格明细：{price_summary}。"
            line += f"{reason}。"
            lines.append(line)
        lines.append("以上商品名、品牌、类目、SKU和价格均来自当前商品库；多规格商品以SKU价格明细为准。")
        return "\n".join(lines)


def _reason(hit: CatalogHit, filters: SearchFilters, catalog: ProductCatalog) -> str:
    reasons = []
    product = hit.product
    if filters.sub_category and product["sub_category"] == filters.sub_category:
        reasons.append(f"符合{filters.sub_category}需求")
    if filters.category and product["category"] == filters.category:
        reasons.append(f"属于{filters.category}")
    if filters.max_price is not None:
        reasons.append(f"价格在{filters.max_price:g}元以内")
    for term in filters.required_terms:
        # Only credit the attribute when the product actually evidences it (not just because
        # it was requested) — required_terms no longer hard-filter, so a card may not match.
        if catalog.evidences_required_term(product, term):
            reasons.append(f"匹配{term}需求")
    if filters.prefer_low_price:
        reasons.append("优先低价")
    if hit.snippets:
        reasons.append("商品描述或评价中有相关信息")
    return "，".join(reasons) if reasons else "与当前需求语义匹配"


def _chunk_text(text: str, chunk_size: int) -> Iterator[str]:
    for idx in range(0, len(text), chunk_size):
        yield text[idx: idx + chunk_size]
