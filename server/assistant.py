"""Application orchestration for basic shopping recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator
from typing import Literal

from server.comparison import ComparisonService
from server.catalog import CatalogHit, ProductCatalog
from server.commerce import (
    CommerceActionCandidate,
    CommerceService,
    CommerceResult,
    OrderState,
    looks_like_commerce,
)
from server.config import Settings
from server.filter_cache import FilterCache
from server.intent import IntentParser, SearchFilters
from server.llm import ArkChatClient, ModelUnavailable
from server.planner import PlannedTask, PlannerService
from server.prompts import (
    CHITCHAT_REPLY,
    build_messages,
    chitchat_messages,
    comparison_narration_messages,
    exclusion_judge_messages,
    lead_in_text,
)
from server.retrieval import ProductRetriever, RetrievalResult
from server.schemas import CartUpdate, ChatResponse, ExecutionPlan, OrderDraft, PlanStep, ProductCard, ProductComparison
from server.textutil import dedupe_ids, json_object


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
    cart: CartUpdate | None
    order: OrderDraft | None
    grounded_answer: str
    messages: list[dict[str, str]]
    result_status: ResultStatus = "ok"
    plan: ExecutionPlan | None = None
    # Set on a filter-cacheable product search so the answer can be stored under the parsed
    # intent; None when the turn isn't cacheable. from_filter_cache marks a turn served from
    # that cache, so it isn't re-stored and its (cached) answer streams via the fallback path.
    filter_cache_key: str | None = None
    from_filter_cache: bool = False


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
    order: OrderState = field(default_factory=OrderState)


class ShoppingAssistant:
    def __init__(
        self,
        catalog: ProductCatalog,
        retriever: ProductRetriever,
        llm: ArkChatClient | None = None,
        intent_llm: ArkChatClient | None = None,
        settings: Settings | None = None,
        filter_cache: FilterCache | None = None,
    ):
        self._catalog = catalog
        self._retriever = retriever
        self._llm = llm
        self._settings = settings or Settings()
        # Off unless wired with a real cache (create_app does this); keeps direct test
        # construction side-effect-free, mirroring how QueryCache lives only at the API layer.
        self._filter_cache = filter_cache or FilterCache(self._settings.filter_cache_path, enabled=False)
        self._parser = IntentParser(
            catalog.categories,
            catalog.sub_categories,
            catalog.brands,
            llm=intent_llm,
            approx_price_tolerance=self._settings.approx_price_tolerance,
        )
        self._comparison = ComparisonService(catalog, llm=llm)
        self._commerce = CommerceService(catalog, llm=llm)
        self._planner = PlannerService(catalog.categories, catalog.sub_categories, catalog.brands, llm=llm)
        self._sessions: dict[str, SessionState] = {}

    @property
    def catalog(self) -> ProductCatalog:
        return self._catalog

    def lead_in(self, query: str, compare_product_ids: list[str] | None = None) -> str:
        """Instant, deterministic streaming opener (no model call). An explicit comparison is
        known from the request; otherwise the rule parser guesses what to acknowledge. The real
        understanding still happens in the LLM parse that runs behind this opener."""
        if compare_product_ids:
            return lead_in_text("compare")
        if looks_like_commerce(query):
            return lead_in_text("neutral")
        kind, label = self._parser.lead_in_hint(query)
        return lead_in_text(kind, label)

    def prepare(
        self,
        query: str,
        session_id: str | None,
        top_k: int,
        compare_product_ids: list[str] | None = None,
        client_recent_product_ids: list[str] | None = None,
        cart_items: list[dict] | None = None,
    ) -> PreparedChat:
        session_products = self._commerce_products(session_id, client_recent_product_ids or [])
        planned = self._planner.plan(
            query,
            session_products=session_products,
            cart_items=cart_items or [],
        )
        if planned is not None:
            return self._prepare_planned_task(
                query,
                session_id,
                planned,
                top_k,
                client_recent_product_ids or [],
                cart_items or [],
            )
        commerce = self._commerce.maybe_handle(
            query,
            cart_items=cart_items or [],
            session_products=session_products,
            order_state=self._session(session_id).order,
        )
        if commerce is not None:
            return self._prepare_commerce(query, session_id, commerce)
        # Pipeline parallelism: start embedding the query now so it overlaps the intent call
        # below. Retrieval's later embed_text then hits the warm cache instead of paying the
        # cold round-trip in series. Skipped for an explicit comparison (it never retrieves).
        if not compare_product_ids:
            self._retriever.prewarm_query(query)
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

    def _prepare_commerce(
        self,
        query: str,
        session_id: str | None,
        commerce: CommerceResult,
    ) -> PreparedChat:
        filters = SearchFilters(
            intent_type=commerce.intent.get("intent_type", "cart_action"),
            raw_query=query,
            commerce_action=commerce.intent.get("commerce_action"),
            commerce_refs=list(commerce.intent.get("commerce_refs", [])),
            commerce_quantity=commerce.intent.get("quantity"),
            commerce_target_scope=commerce.intent.get("target_scope"),
        )
        return PreparedChat(
            query=query,
            session_id=session_id,
            filters=filters,
            retrieval=RetrievalResult(hits=[], source="none"),
            products=[],
            comparison=None,
            cart=commerce.cart,
            order=commerce.order,
            grounded_answer=commerce.answer,
            messages=[],
        )

    def _prepare_planned_task(
        self,
        query: str,
        session_id: str | None,
        planned: PlannedTask,
        top_k: int,
        client_recent_product_ids: list[str],
        cart_items: list[dict],
    ) -> PreparedChat:
        plan_steps = [
            PlanStep(
                step_id=f"step-{idx}",
                title=step.title,
                action=step.action,
                status="pending",
            )
            for idx, step in enumerate(planned.steps, start=1)
        ]

        retrieval = RetrievalResult(hits=[], source="none")
        products: list[ProductCard] = self._product_cards_from_ids(
            self._recent_product_ids(session_id, client_recent_product_ids)
        )
        selected_ids: list[str] = [product.product_id for product in products]
        comparison: ProductComparison | None = None
        cart: CartUpdate | None = None
        order: OrderDraft | None = None
        summaries: list[str] = []
        active_filters = SearchFilters(intent_type="planned_action", raw_query=query)
        raw_cart = list(cart_items)

        for idx, step in enumerate(planned.steps):
            plan_steps[idx].status = "running"
            try:
                if step.action == "product_search":
                    prepared = self._run_planned_search(
                        step.query or query,
                        session_id,
                        top_k,
                        client_recent_product_ids,
                        planned,
                    )
                    retrieval = prepared.retrieval
                    products = prepared.products
                    selected_ids = [product.product_id for product in products]
                    active_filters = prepared.filters
                    summary = f"找到 {len(products)} 款候选商品。"
                elif step.action == "select_products":
                    products = self._select_products(products, step.criteria, step.count or 1)
                    selected_ids = [product.product_id for product in products]
                    summary = f"已选出 {len(selected_ids)} 款候选商品。"
                elif step.action == "comparison":
                    compare_ids = selected_ids[: max(2, step.count or len(selected_ids))]
                    filters = self._parser.parse(
                        step.query or query,
                        previous_filters=self._previous_filters(session_id),
                        history=self._history_summaries(session_id),
                        session_products=self._session_products(session_id),
                    )
                    filters.intent_type = "comparison"
                    if step.criteria == "price_asc":
                        filters.prefer_low_price = True
                        filters.sort_by = "price_asc"
                    prepared = self._prepare_comparison(
                        step.query or query,
                        session_id,
                        filters,
                        compare_ids,
                        self._recent_product_ids(session_id, client_recent_product_ids),
                    )
                    comparison = prepared.comparison
                    products = prepared.products
                    selected_ids = [product.product_id for product in products]
                    active_filters = filters
                    summary = comparison.summary if comparison else "已完成对比。"
                elif step.action == "cart_action":
                    target_ids = self._cart_target_ids(step.target, selected_ids, comparison, products)
                    if not target_ids:
                        raise ValueError("no product selected for cart action")
                    cart_result = self._apply_cart_targets(target_ids, step.quantity or 1, query, raw_cart, session_id)
                    cart = cart_result.cart
                    order = cart_result.order
                    raw_cart = [item.model_dump() for item in cart.items] if cart is not None else raw_cart
                    summary = cart_result.answer
                elif step.action == "checkout":
                    checkout = CommerceActionCandidate(action="checkout", target_scope="cart_items", confidence="high")
                    result = self._commerce.apply_candidate(
                        checkout,
                        query,
                        cart_items=raw_cart,
                        session_products=self._commerce_products(session_id, client_recent_product_ids),
                        order_state=self._session(session_id).order,
                    )
                    cart = result.cart
                    order = result.order
                    summary = result.answer
                else:
                    summary = "需要补充信息后才能继续。"
                plan_steps[idx].status = "done"
                plan_steps[idx].summary = summary
                summaries.append(summary)
            except Exception as exc:  # noqa: BLE001 - planned execution should fail closed.
                plan_steps[idx].status = "failed"
                plan_steps[idx].summary = "这一步缺少可执行的商品信息，请补充说明。"
                summaries.append(plan_steps[idx].summary or "")
                break

        plan = ExecutionPlan(steps=plan_steps, summary="；".join(summaries) if summaries else None)
        return PreparedChat(
            query=query,
            session_id=session_id,
            filters=SearchFilters(intent_type="planned_action", raw_query=query),
            retrieval=retrieval,
            products=products,
            comparison=comparison,
            cart=cart,
            order=order,
            grounded_answer=self._planned_answer(plan, cart, comparison),
            messages=[],
            plan=plan,
            result_status="ok" if products or cart or comparison else "no_results",
        )

    def _run_planned_search(
        self,
        query: str,
        session_id: str | None,
        top_k: int,
        client_recent_product_ids: list[str],
        planned: PlannedTask,
    ) -> PreparedChat:
        filters = self._parser.parse(
            query,
            previous_filters=self._previous_filters(session_id),
            history=self._history_summaries(session_id),
            session_products=self._session_products(session_id),
        )
        filters.intent_type = "product_search"
        criteria = next((step.criteria for step in planned.steps if step.action == "select_products"), None)
        if criteria == "price_asc":
            filters.prefer_low_price = True
            filters.sort_by = "price_asc"
        elif criteria == "price_desc":
            filters.sort_by = "price_desc"
        elif criteria == "rating_desc":
            filters.sort_by = "rating_desc"
        return self._prepare_search(
            query,
            session_id,
            filters,
            top_k,
            self._recent_product_ids(session_id, client_recent_product_ids),
        )

    def _select_products(
        self,
        products: list[ProductCard],
        criteria: str | None,
        count: int,
    ) -> list[ProductCard]:
        if criteria == "price_asc":
            ordered = sorted(products, key=lambda product: product.price)
        elif criteria == "price_desc":
            ordered = sorted(products, key=lambda product: product.price, reverse=True)
        else:
            ordered = products
        return ordered[: max(1, count)]

    def _cart_target_ids(
        self,
        target: str | None,
        selected_ids: list[str],
        comparison: ProductComparison | None,
        products: list[ProductCard],
    ) -> list[str]:
        if target == "comparison_winner" and comparison is not None and comparison.winner_product_id:
            return [comparison.winner_product_id]
        if selected_ids:
            return selected_ids
        return [product.product_id for product in products[:1]]

    def _apply_cart_targets(
        self,
        product_ids: list[str],
        quantity: int,
        query: str,
        raw_cart: list[dict],
        session_id: str | None,
    ) -> CommerceResult:
        result: CommerceResult | None = None
        for product_id in product_ids:
            candidate = CommerceActionCandidate(
                action="add",
                product_ids=[product_id],
                quantity=quantity,
                target_scope="shown_products",
                confidence="high",
            )
            result = self._commerce.apply_candidate(
                candidate,
                query,
                cart_items=raw_cart,
                session_products=self._session_products(session_id),
                order_state=self._session(session_id).order,
            )
            if result.cart is not None:
                raw_cart = [item.model_dump() for item in result.cart.items]
        if result is None:
            raise ValueError("no cart target")
        return result

    def _product_cards_from_ids(self, product_ids: list[str]) -> list[ProductCard]:
        filters = SearchFilters()
        cards = []
        for product_id in product_ids:
            product = self._catalog.get(product_id)
            if product is not None:
                cards.append(self._catalog.product_card(product, matched_reason="已展示商品", filters=filters))
        return cards

    def _planned_answer(
        self,
        plan: ExecutionPlan,
        cart: CartUpdate | None,
        comparison: ProductComparison | None,
    ) -> str:
        lines = ["我已按计划完成："]
        for step in plan.steps:
            marker = "✓" if step.status == "done" else "!"
            detail = f"：{step.summary}" if step.summary else ""
            lines.append(f"{marker} {step.title}{detail}")
        if comparison is not None and comparison.recommendation:
            lines.append(comparison.recommendation)
        if cart is not None:
            lines.append(cart.summary)
        return "\n".join(lines)

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
            cart=None,
            order=None,
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
            cart=None,
            order=None,
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
        # Filter-keyed cache: a context-free product search keyed on the parsed intent. A hit
        # replays the stored answer + cards, skipping embed, retrieval and the answer LLM. The
        # key is set on a miss too, so the generated answer gets stored once it's produced.
        filter_key: str | None = None
        if self._filter_cache.enabled and FilterCache.eligible(filters, recent_product_ids):
            filter_key = self._filter_cache.key(filters, top_k)
            cached = self._filter_cache.get(filter_key)
            if cached is not None:
                return self._prepared_from_cache(query, session_id, filters, cached, filter_key)
        # The rewrite folds carried context into a standalone retrieval query; the answer
        # itself still replies to what the user actually typed (raw query below).
        search_query = filters.rewritten_query or query
        # Over-fetch so that dropping already-seen ("换一批") or excluded ("不要油腻") items still
        # leaves enough to fill top_k.
        buffer = (len(recent_product_ids) if filters.exclude_seen else 0) + (top_k if filters.excluded_terms else 0)
        retrieval = self._retriever.retrieve(query=search_query, filters=filters, limit=top_k + buffer)
        hits = self._order_hits(retrieval.hits, filters)
        if filters.exclude_seen:
            seen = set(recent_product_ids)
            hits = [hit for hit in hits if hit.product["product_id"] not in seen]
        if filters.excluded_terms:
            excluded = self._excluded_ids(hits, filters.excluded_terms)
            hits = [hit for hit in hits if hit.product["product_id"] not in excluded]
        hits = hits[:top_k]
        products = [
            self._catalog.product_card(hit.product, matched_reason=_reason(hit, filters, self._catalog), filters=filters)
            for hit in hits
        ]
        prev_floor = self._previous_floor(session_id)
        result_status = self._result_status(filters, products, recent_product_ids, prev_floor)
        context = self._status_context(result_status, products, prev_floor)
        # Required attributes and requested specs no longer hard-filter; flag any that nothing
        # retrieved matches, so the answer says so honestly instead of implying every card fits.
        unmet = self._catalog.unmet_required_terms(hits, filters) + self._catalog.unmet_requested_specs(hits, filters)
        if unmet:
            context = {**(context or {}), "unmet_terms": unmet}
        return self._search_prepared(
            query, session_id, filters, hits, products, retrieval, result_status, context, filter_key
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
        filter_cache_key: str | None = None,
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
            cart=None,
            order=None,
            grounded_answer=grounded_answer,
            messages=messages,
            result_status=result_status,
            filter_cache_key=filter_cache_key,
        )

    def _prepared_from_cache(
        self, query: str, session_id: str | None, filters: SearchFilters, cached: dict, key: str
    ) -> PreparedChat:
        # Rebuild a PreparedChat from the cached response so both the streaming and non-streaming
        # paths replay it unchanged: messages=[] routes through the fallback, which emits the
        # cached answer text and cards. Session memory is still updated so follow-ups resolve.
        products = [ProductCard(**product) for product in cached.get("products", [])]
        self._remember_shown_products(session_id, products)
        self._remember_turn(session_id, query, filters, products)
        source = cached.get("retrieval_source") or "none"
        return PreparedChat(
            query=query,
            session_id=session_id,
            filters=filters,
            retrieval=RetrievalResult(hits=[], source=source, warnings=list(cached.get("warnings", []))),
            products=products,
            comparison=None,
            cart=None,
            order=None,
            grounded_answer=cached.get("answer", ""),
            messages=[],
            filter_cache_key=key,
            from_filter_cache=True,
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

    def _excluded_ids(self, hits: list[CatalogHit], excluded_terms: list[str]) -> set[str]:
        """Which shortlisted products to drop for an exclusion ("不要油腻"). The LLM judges meaning
        and negation over the small shortlist (primary); the deterministic negation-aware catalog
        check is the fallback when the LLM is unavailable."""
        if not hits:
            return set()
        if self._llm is not None and self._llm.available:
            products = [
                {
                    "id": hit.product["product_id"],
                    "名称": hit.product.get("title", ""),
                    "描述": hit.product.get("rag_knowledge", {}).get("marketing_description", ""),
                }
                for hit in hits
            ]
            try:
                payload = json_object(self._llm.complete(exclusion_judge_messages(excluded_terms, products)))
                if "exclude" in payload:  # a parseable verdict; garbage -> fall through
                    valid = {hit.product["product_id"] for hit in hits}
                    return {pid for pid in payload["exclude"] if pid in valid}
            except Exception:  # noqa: BLE001 - any judge failure must degrade to the deterministic check
                pass
        return {
            hit.product["product_id"]
            for hit in hits
            if self._catalog.violates_excluded(hit.product, excluded_terms)
        }

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
        cart_items: list[dict] | None = None,
    ) -> ChatResponse:
        prepared = self.prepare(query, session_id, top_k, compare_product_ids, client_recent_product_ids, cart_items)
        warnings = list(prepared.retrieval.warnings)

        answer_text = prepared.grounded_answer
        if prepared.messages and self._llm is not None and self._llm.available:
            try:
                answer_text = self._llm.complete(prepared.messages)
            except ModelUnavailable as exc:
                warnings.append(f"LLM unavailable, using grounded fallback: {exc}")

        response = ChatResponse(
            answer=answer_text,
            products=prepared.products,
            comparison=prepared.comparison,
            cart=prepared.cart,
            order=prepared.order,
            plan=prepared.plan,
            session_id=session_id,
            intent=prepared.filters.to_dict(),
            retrieval_source=prepared.retrieval.source,
            degraded=bool(warnings),
            warnings=warnings,
        )
        self.maybe_store_filter_cache(prepared, response)
        return response

    def maybe_store_filter_cache(self, prepared: PreparedChat, response: ChatResponse) -> None:
        """Store a freshly produced answer under its parsed-intent key, so later paraphrases hit
        it. No-op for non-cacheable turns and for answers already served from this cache."""
        if prepared.filter_cache_key is None or prepared.from_filter_cache:
            return
        self._filter_cache.put(prepared.filter_cache_key, response.model_dump())

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

    def _commerce_products(self, session_id: str | None, client_recent_product_ids: list[str]) -> list[dict] | None:
        """Products available for cart references.

        Prefer the client's visible/recent product ids because cart commands often arrive after a
        cached stream replay or a backend restart where server-side shown_products is empty. Append
        session memory as a fallback while preserving each source's recency order.
        """
        items: list[dict] = []
        seen: set[str] = set()
        for pid in client_recent_product_ids:
            product = self._catalog.get(pid)
            if product is None or pid in seen:
                continue
            items.append({
                "id": pid,
                "title": product["title"],
                "brand": product["brand"],
                "price": self._catalog.lowest_price(product),
                "sub_category": product["sub_category"],
            })
            seen.add(pid)
        for entry in self._session_products(session_id) or []:
            pid = entry["id"]
            if pid not in seen:
                items.append(entry)
                seen.add(pid)
        return items or None

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
