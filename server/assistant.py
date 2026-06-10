"""Application orchestration for basic shopping recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
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
from server.llm import ChatClient, ModelUnavailable
from server.planner import PlannedTask, PlannerService, looks_like_planned_task
from server.prompts import (
    CHITCHAT_REPLY,
    CLARIFY_FALLBACK,
    build_messages,
    chitchat_messages,
    comparison_narration_messages,
    exclusion_judge_messages,
    opener_continuation,
    opener_lead,
    opener_text,
    photo_answer_messages,
    photo_opener,
)
from server.retrieval import ProductRetriever, RetrievalResult
from server.schemas import CartUpdate, ChatResponse, ExecutionPlan, OrderDraft, PlanStep, ProductCard, ProductComparison
from server.textutil import dedupe_ids, json_object


# Outcome of a product-search turn, used to narrate honestly instead of re-listing.
ResultStatus = Literal["ok", "no_results", "no_cheaper", "no_improvement", "exhausted"]

# Fields of a shown product remembered for the turn history and the recall log.
_SHOWN_FIELDS = ("id", "title", "brand", "price", "sub_category")

# When a turn sorts by price/rating, retrieval fetches up to this many candidates so the sort sees
# the whole filtered category (larger than any single sub-category) instead of the relevance-top-k.
_SORT_CANDIDATE_POOL = 50

# Valid SearchFilters keys, computed once. Used to drop unknown keys when rebuilding filters from
# a (possibly older-schema) cached response, so the reconstruction can't choke on a stale field.
_SEARCH_FILTER_FIELDS = frozenset(f.name for f in fields(SearchFilters))

# Obvious greeting / small-talk openers, kept short so a real shopping query that merely starts with
# a greeting ("你好，推荐个面霜") is not caught.
_GREETING_HINTS = (
    "你好", "您好", "你是谁", "您是谁", "在吗", "在不在", "谢谢", "多谢", "感谢",
    "早上好", "中午好", "晚上好", "嗨", "哈喽", "hi", "hello",
)


def _looks_like_greeting(message: str) -> bool:
    """Conservative greeting check, used ONLY to suppress the instant pre-router lead so a "好的，" never
    lands in front of the router's inline greeting reply. Not a router: a wrong answer here only shifts
    or duplicates the opener, it never changes the route the LLM picks."""
    text = message.strip().lower()
    return len(text) <= 6 and any(hint in text for hint in _GREETING_HINTS)


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
    # intent. None when the turn isn't cacheable. from_filter_cache marks a turn served from
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
    # The winner of the most recent comparison, so a follow-up "买更适合的/更好的那个" resolves to it.
    last_winner_id: str | None = None
    # Single source of truth for "what products has the user seen": session-wide, deduped by
    # id, in first-shown order, each tagged with the turn (last_seq) and within-turn position
    # it was most recently shown. Backtracking reads it in first-shown order. The recency view
    # for comparison ("第一个/前两个") is derived from it (last_seq desc, position asc).
    shown_products: list[dict] = field(default_factory=list)
    turn_seq: int = 0
    order: OrderState = field(default_factory=OrderState)
    # The route the previous turn took, given to the router as context so it can tell, e.g., a
    # content-free "确认" after a finished order (-> chitchat) from a real command.
    last_route: str | None = None


class ShoppingAssistant:
    def __init__(
        self,
        catalog: ProductCatalog,
        retriever: ProductRetriever,
        llm: ChatClient | None = None,
        intent_llm: ChatClient | None = None,
        settings: Settings | None = None,
        filter_cache: FilterCache | None = None,
    ):
        self._catalog = catalog
        self._retriever = retriever
        self._llm = llm
        self._settings = settings or Settings()
        # Off unless wired with a real cache (create_app does this). Keeps direct test
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

    def opener(self, route: str, query: str) -> str:
        """The whole route-tailored opener as one string (empty for chitchat). Used by the cached replay
        path, which has no router to wait on. The streaming path instead splits it into an instant lead
        and a route-specific tail (see `_route`)."""
        return opener_text(route, self._opener_label(route, query))

    def _opener_label(self, route: str, query: str) -> str | None:
        # The category label for a search opener comes from the instant rule parse, not the LLM, so it
        # costs nothing.
        return self._parser.lead_in_hint(query)[1] if route == "product_search" else None

    def prepare(
        self,
        query: str,
        session_id: str | None,
        top_k: int,
        compare_product_ids: list[str] | None = None,
        client_recent_product_ids: list[str] | None = None,
        cart_items: list[dict] | None = None,
        image_bytes: bytes | None = None,
        client_address: str | None = None,
    ) -> PreparedChat:
        if image_bytes is not None:
            recent = self._recent_product_ids(session_id, client_recent_product_ids or [])
            return self._prepare_photo_search(query, session_id, top_k, image_bytes, recent)
        result = None
        for item in self._route(
            query, session_id, top_k, compare_product_ids or [], client_recent_product_ids or [], cart_items or [],
            client_address,
        ):
            if isinstance(item, tuple):  # the opener-continuation strings are ignored off-stream
                result = item
        kind, payload = result
        if kind == "planned":
            return self._prepare_planned_task(
                query, session_id, payload, top_k, client_recent_product_ids or [], cart_items or []
            )
        return payload

    def prepare_stream(
        self,
        query: str,
        session_id: str | None,
        top_k: int,
        compare_product_ids: list[str] | None = None,
        client_recent_product_ids: list[str] | None = None,
        cart_items: list[dict] | None = None,
        image_bytes: bytes | None = None,
        client_address: str | None = None,
    ) -> Iterator[str | ExecutionPlan | PreparedChat]:
        if image_bytes is not None:
            # The visual search is slow; flush an instant lead first so 首Token still lands under 1s.
            yield photo_opener()
            yield self.prepare(
                query, session_id, top_k, compare_product_ids,
                client_recent_product_ids, cart_items, image_bytes=image_bytes,
            )
            return
        for item in self._route(
            query, session_id, top_k, compare_product_ids or [], client_recent_product_ids or [], cart_items or [],
            client_address,
        ):
            if isinstance(item, str):
                yield item  # opener continuation — stream it the moment the route is known
                continue
            kind, payload = item
            if kind == "planned":
                yield from self._prepare_planned_task_updates(
                    query, session_id, payload, top_k, client_recent_product_ids or [], cart_items or []
                )
            else:
                yield payload
            return

    def _route(
        self,
        query: str,
        session_id: str | None,
        top_k: int,
        compare_product_ids: list[str],
        client_recent_product_ids: list[str],
        cart_items: list[dict],
        client_address: str | None = None,
    ) -> Iterator[str | tuple[str, PreparedChat | PlannedTask]]:
        """Generator: yields the route-tailored opener continuation as soon as the router decides, then
        a final (kind, payload) tuple — ("prepared", PreparedChat) or ("planned", PlannedTask). A
        focused LLM router classifies the turn (a deterministic clarification-continuation sits in
        front, keyword router as the LLM-off fallback); the heavy intent parse runs only for
        search/comparison; commerce lets the LLM fill the action."""
        state = self._session(session_id)
        order_state = state.order
        if client_address:
            order_state.address = client_address
        session_products = self._commerce_products(session_id, client_recent_product_ids)

        # 1. Continue an open "which item?" clarification before anything routes.
        if order_state.pending_action is not None:
            pending = self._commerce.handle_pending_reply(
                query, cart_items=cart_items, session_products=session_products, order_state=order_state
            )
            if pending is not None:
                yield self.opener("cart_action", query)
                yield ("prepared", self._prepare_commerce(query, session_id, pending))
                return

        # 2. Instant lead, flushed before the router so 首Token lands < 1s on a shopping turn. Suppressed
        # for an obvious greeting so a "好的，" never lands in front of the router's inline chitchat reply.
        # This is a latency gate, not a router: a wrong guess only shifts or duplicates the opener, it
        # never changes the route the LLM picks.
        greeting = _looks_like_greeting(query)
        if not greeting:
            yield opener_lead()

        # 3. Speculative embed overlaps the router/parse calls (only the search path needs it).
        if not compare_product_ids:
            self._retriever.prewarm_query(query)

        recent_product_ids = self._recent_product_ids(session_id, client_recent_product_ids)

        # 4. Focused router (LLM-primary). When the LLM is off/failed, fall back to the keyword router
        # (which needs a rule parse). `filters` is parsed lazily and reused.
        filters: SearchFilters | None = None
        route, chitchat_reply = self._parser.classify_route(
            query,
            has_cart=bool(cart_items),
            has_results=bool(session_products),
            has_draft=order_state.draft is not None,
            just_compared=state.last_winner_id is not None,
            last_route=state.last_route,
        )
        if route is None:
            filters = self._parse_intent(query, session_id, cart_items)
            route = self._fallback_route(query, order_state, filters)
        if len(compare_product_ids) >= 2:
            route = "comparison"
        # clarify backstop: the router proposed a clarifying question, but veto it down to a plain search
        # when asking would be wrong — the turn already carries a distinguishing constraint, we asked
        # last turn (state.last_route is still the previous route here, so no double-ask), or the item is
        # out-of-catalogue. Done before recording last_route and the opener so both reflect the final route.
        if route == "clarify":
            filters = filters or self._parse_intent(query, session_id, cart_items)
            if state.last_route == "clarify" or self._has_constraint(filters) or filters.intent_type == "chitchat":
                route = "product_search"
        # Remember this turn's route as context for the next turn's router (e.g. so a content-free
        # "确认" after a finished order reads as an acknowledgement, not a fresh search).
        state.last_route = route

        # Complete the opener now the route is known. Normally the lead is already out, so only the
        # route-specific tail remains (empty for chitchat, where the reply greets for itself). If the
        # greeting gate suppressed the lead but the turn isn't chitchat after all, emit the whole opener.
        if greeting:
            opener = self.opener(route, query)
            if opener:
                yield opener
        else:
            tail = opener_continuation(route, self._opener_label(route, query))
            if tail:
                yield tail

        # 4. Dispatch.
        if route == "planned_task":
            planned = self._planner.plan(
                query, force=True, session_products=session_products, cart_items=cart_items
            )
            if planned is not None:
                yield ("planned", planned)
                return
            route = "product_search"  # planner declined -> treat as a plain search
        if route in {"cart_action", "checkout"}:
            commerce = self._commerce.maybe_handle(
                query, cart_items=cart_items, session_products=session_products,
                order_state=order_state, comparison_winner_id=state.last_winner_id,
                latest_batch_ids=self._latest_batch_ids(session_id),
            )
            if commerce is not None:
                yield ("prepared", self._prepare_commerce(query, session_id, commerce))
                return
            route = "product_search"  # routed to cart but nothing actionable -> fall through
        if route == "comparison":
            filters = filters or self._parse_intent(query, session_id, cart_items)
            yield ("prepared", self._prepare_comparison(
                query, session_id, filters, compare_product_ids, recent_product_ids
            ))
            return
        if route == "clarify":
            # The query named a product type but gave nothing to narrow on. The router wrote the
            # clarifying question into reply (no extra model call). Record the topic so the next answer
            # inherits the category.
            assert filters is not None  # the clarify backstop above always parses it first
            filters.intent_type = "clarify"
            self._remember_turn(session_id, query, filters, [])
            yield ("prepared", self._prepare_chitchat(
                query, session_id, filters, reply=chitchat_reply or CLARIFY_FALLBACK,
            ))
            return
        if route == "chitchat":
            # The router already wrote the reply inline, so chitchat needs no further model call.
            yield ("prepared", self._prepare_chitchat(
                query, session_id, SearchFilters(intent_type="chitchat", raw_query=query),
                reply=chitchat_reply or CHITCHAT_REPLY,
            ))
            return
        # product_search: the parse still gets to downgrade to chitchat for an out-of-catalogue item.
        filters = filters or self._parse_intent(query, session_id, cart_items)
        if filters.intent_type == "chitchat":
            yield ("prepared", self._prepare_chitchat(query, session_id, filters))
            return
        filters.intent_type = "product_search"
        yield ("prepared", self._prepare_search(query, session_id, filters, top_k, recent_product_ids))

    @staticmethod
    def _has_constraint(filters: SearchFilters) -> bool:
        """True if the parse carries a distinguishing constraint — anything that narrows the search
        beyond the bare product type. category / sub_category are deliberately excluded: a clarify query
        always names a type (that's the topic), so they are not 'enough to search on'."""
        return bool(
            filters.max_price is not None
            or filters.min_price is not None
            or filters.brand
            or filters.prefer_low_price
            or filters.sort_by != "relevance"
            or filters.required_terms
            or filters.requested_specs
            or filters.excluded_brands
            or filters.excluded_terms
        )

    def _parse_intent(self, query: str, session_id: str | None, cart_items: list[dict]) -> SearchFilters:
        return self._parser.parse(
            query,
            previous_filters=self._previous_filters(session_id),
            history=self._history_summaries(session_id),
            session_products=self._session_products(session_id),
            cart=self._cart_for_intent(cart_items),
        )

    def _fallback_route(self, message: str, order_state: OrderState, filters: SearchFilters) -> str:
        """Keyword route resolution for when the intent LLM is unavailable. planned_task is checked
        before commerce because a multi-step request also contains commerce keywords; commerce before
        the rule comparison so an ordinal add ("加第一个") routes to the cart, not a comparison."""
        if looks_like_planned_task(message):
            return "planned_task"
        if looks_like_commerce(message) or order_state.draft is not None:
            return "cart_action"  # maybe_handle's deterministic parse picks add vs checkout vs ...
        return filters.intent_type  # rule parser: comparison or product_search

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
        prepared = None
        for update in self._prepare_planned_task_updates(
            query,
            session_id,
            planned,
            top_k,
            client_recent_product_ids,
            cart_items,
        ):
            if isinstance(update, PreparedChat):
                prepared = update
        if prepared is None:
            raise RuntimeError("planned task did not produce a prepared chat")
        return prepared

    def _prepare_planned_task_updates(
        self,
        query: str,
        session_id: str | None,
        planned: PlannedTask,
        top_k: int,
        client_recent_product_ids: list[str],
        cart_items: list[dict],
    ) -> Iterator[ExecutionPlan | PreparedChat]:
        plan_steps = [
            PlanStep(
                step_id=f"step-{idx}",
                title=step.title,
                action=step.action,
                status="pending",
            )
            for idx, step in enumerate(planned.steps, start=1)
        ]
        yield _copy_plan(ExecutionPlan(steps=plan_steps))

        retrieval = RetrievalResult(hits=[], source="none")
        products: list[ProductCard] = self._product_cards_from_ids(
            self._recent_product_ids(session_id, client_recent_product_ids)
        )
        selected_ids: list[str] = [product.product_id for product in products]
        comparison: ProductComparison | None = None
        cart: CartUpdate | None = None
        order: OrderDraft | None = None
        summaries: list[str] = []
        requested_specs: list[str] = []
        raw_cart = list(cart_items)

        for idx, step in enumerate(planned.steps):
            plan_steps[idx].status = "running"
            yield _copy_plan(ExecutionPlan(steps=plan_steps, summary="；".join(summaries) if summaries else None))
            try:
                if step.action == "product_search":
                    prepared = self._run_planned_search(
                        step.query or query,
                        session_id,
                        top_k,
                        client_recent_product_ids,
                        planned,
                    )
                    if prepared.filters.intent_type == "chitchat":
                        # The thing isn't in our catalogue. Abandon the plan and decline politely
                        # instead of carting an unrelated product.
                        plan_steps[idx].status = "failed"
                        plan_steps[idx].summary = "本店暂不提供这件商品。"
                        summaries.append(plan_steps[idx].summary)
                        prepared.plan = ExecutionPlan(steps=plan_steps, summary="；".join(summaries))
                        yield _copy_plan(prepared.plan)
                        yield prepared
                        return
                    retrieval = prepared.retrieval
                    products = prepared.products
                    selected_ids = [product.product_id for product in products]
                    requested_specs = list(prepared.filters.requested_specs)
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
                    summary = comparison.summary if comparison else "已完成对比。"
                elif step.action == "cart_action":
                    target_ids = self._cart_target_ids(step.target, selected_ids, comparison, products)
                    if not target_ids:
                        raise ValueError("no product selected for cart action")
                    cart_result = self._apply_cart_targets(
                        target_ids, step.quantity or 1, query, raw_cart, session_id, requested_specs
                    )
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
                yield _copy_plan(ExecutionPlan(steps=plan_steps, summary="；".join(summaries)))
            except Exception:  # noqa: BLE001 (planned execution should fail closed)
                plan_steps[idx].status = "failed"
                plan_steps[idx].summary = "这一步缺少可执行的商品信息，请补充说明。"
                summaries.append(plan_steps[idx].summary or "")
                yield _copy_plan(ExecutionPlan(steps=plan_steps, summary="；".join(summaries)))
                break

        plan = ExecutionPlan(steps=plan_steps, summary="；".join(summaries) if summaries else None)
        yield PreparedChat(
            query=query,
            session_id=session_id,
            filters=SearchFilters(intent_type="planned_task", raw_query=query),
            retrieval=retrieval,
            products=products,
            comparison=comparison,
            cart=cart,
            order=order,
            grounded_answer=self._planned_answer(plan),
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
        # An out-of-catalogue request (e.g. 手表, which we don't sell) parses as chitchat. Forcing
        # product_search here would make retrieval return nearest-neighbour junk (AirPods for a
        # watch) and the plan would then cart it. Honour the decline; the caller abandons the plan.
        if filters.intent_type == "chitchat":
            return self._prepare_chitchat(query, session_id, filters)
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
        requested_specs: list[str] | None = None,
    ) -> CommerceResult:
        # The planner already resolved these ids this turn. Commerce only accepts product ids it can
        # see in session_products ∪ cart, so include the targets in the pool directly — otherwise a
        # plan whose search hasn't been written to session memory yet (e.g. a session-less turn)
        # would fail the cart step and ask for clarification instead of adding.
        pool = self._pool_with_ids(product_ids, self._session_products(session_id))
        # The user may have named a 规格 ("512GB高配版"); carry it so each line is priced for that SKU
        # instead of the default cheapest one. The direct cart path does this via candidate.sku.
        sku = " ".join(requested_specs) if requested_specs else None
        result: CommerceResult | None = None
        for product_id in product_ids:
            candidate = CommerceActionCandidate(
                action="add",
                product_ids=[product_id],
                quantity=quantity,
                target_scope="shown_products",
                confidence="high",
                sku=sku,
            )
            result = self._commerce.apply_candidate(
                candidate,
                query,
                cart_items=raw_cart,
                session_products=pool,
                order_state=self._session(session_id).order,
            )
            if result.cart is not None:
                raw_cart = [item.model_dump() for item in result.cart.items]
        if result is None:
            raise ValueError("no cart target")
        return result

    def _pool_with_ids(self, product_ids: list[str], existing: list[dict] | None) -> list[dict]:
        """A commerce reference pool (session_products shape) that's guaranteed to contain the given
        catalog ids, so an explicit, already-resolved target always resolves."""
        pool = list(existing or [])
        have = {entry.get("id") for entry in pool}
        for pid in product_ids:
            if pid in have:
                continue
            product = self._catalog.get(pid)
            if product is None:
                continue
            pool.append(self._commerce_entry(pid, product))
            have.add(pid)
        return pool

    def _commerce_entry(self, pid: str, product: dict) -> dict:
        """A single commerce reference row (the session_products shape) for a catalog product."""
        return {
            "id": pid,
            "title": product["title"],
            "brand": product["brand"],
            "price": self._catalog.lowest_price(product),
            "sub_category": product["sub_category"],
        }

    def _product_cards_from_ids(self, product_ids: list[str]) -> list[ProductCard]:
        filters = SearchFilters()
        cards = []
        for product_id in product_ids:
            product = self._catalog.get(product_id)
            if product is not None:
                cards.append(self._catalog.product_card(product, matched_reason="已展示商品", filters=filters))
        return cards

    def _planned_answer(self, plan: ExecutionPlan) -> str:
        lines = ["我已按计划完成："]
        for step in plan.steps:
            marker = "✓" if step.status == "done" else "!"
            detail = f"：{step.summary}" if step.summary else ""
            lines.append(f"{marker} {step.title}{detail}")
        # Each step's bullet already carries its own summary (the cart line, the comparison verdict),
        # so the outcome isn't repeated again at the tail.
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
        # Let the LLM narrate the (deterministic) comparison result. The template above is
        # the fallback. No messages for a clarification, there is nothing to narrate.
        messages = [] if comparison.clarification else comparison_narration_messages(comparison)
        self._remember_shown_products(session_id, products)
        if session_id and comparison.winner_product_id:
            self._session(session_id).last_winner_id = comparison.winner_product_id
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
        self, query: str, session_id: str | None, filters: SearchFilters, reply: str | None = None
    ) -> PreparedChat:
        # When the router already wrote the reply inline (greeting chitchat), use it and make no
        # further model call (messages=[]). Otherwise (out-of-catalogue downgrade) let the chitchat
        # LLM narrate the polite decline, with the fixed reply as the model-unavailable fallback.
        if reply is None and session_id:
            # An out-of-catalogue product query is a topic change: it shows no products, so the
            # usual clear in _remember_shown_products never runs. Drop any stale comparison winner
            # here too, or a later "买胜出的那款" would resolve to an unrelated earlier list. A bare
            # greeting (reply set) keeps the winner.
            self._session(session_id).last_winner_id = None
        return PreparedChat(
            query=query,
            session_id=session_id,
            filters=filters,
            retrieval=RetrievalResult(hits=[], source="none"),
            products=[],
            comparison=None,
            cart=None,
            order=None,
            grounded_answer=reply if reply is not None else CHITCHAT_REPLY,
            messages=[] if reply is not None else chitchat_messages(query, self._catalog.scope_summary()),
        )

    def _prepare_photo_search(
        self,
        query: str,
        session_id: str | None,
        top_k: int,
        image_bytes: bytes,
        recent_product_ids: list[str],
    ) -> PreparedChat:
        # Understand the photo (VLM proposes the same SearchFilters the text parser does, plus a
        # description and a confidence), embed the photo for visual search, then run the existing
        # hybrid retriever with the image vector as an extra RRF source. A photo turn never declines
        # (unlike out-of-catalog text, which routes to chitchat); honesty comes from the confidence.
        image_vector = self._safe_embed_image(image_bytes)
        filters = self._parser.parse_image(
            image_bytes,
            text=query,
            history=self._history_summaries(session_id),
            session_products=self._session_products(session_id),
        )
        filters.intent_type = "product_search"
        # parse_image already relaxed an uncertain visual category to only what the text named, so
        # low confidence here just drives the honest "approximate" narration, not the gate.
        low_conf = filters.vision_confidence != "high"
        search_query = filters.vision_description or query or "相似商品"
        retrieval = self._retriever.retrieve(
            query=search_query, filters=filters, limit=top_k, image_vector=image_vector
        )
        hits = self._order_hits(retrieval.hits, filters)[:top_k]
        products = [
            self._catalog.product_card(hit.product, matched_reason=_photo_reason(low_conf), filters=filters)
            for hit in hits
        ]
        result_status: ResultStatus = "ok" if products else "no_results"
        self._remember_shown_products(session_id, products)
        self._remember_turn(session_id, query, filters, products)
        grounded = self._photo_grounded_answer(filters, hits, low_conf)
        available_by_id = self._available_by_id(session_id, hits)
        messages = photo_answer_messages(query, filters, hits, self._catalog, low_conf, available_by_id) if hits else []
        return PreparedChat(
            query=query,
            session_id=session_id,
            filters=filters,
            retrieval=retrieval,
            products=products,
            comparison=None,
            cart=None,
            order=None,
            grounded_answer=grounded,
            messages=messages,
            result_status=result_status,
        )

    def _safe_embed_image(self, image_bytes: bytes) -> list[float] | None:
        try:
            return self._retriever.embed_image(image_bytes)
        except Exception:  # noqa: BLE001 (a failed embed degrades to lexical, never crashes)
            return None

    def _photo_grounded_answer(
        self, filters: SearchFilters, hits: list[CatalogHit], low_conf: bool
    ) -> str:
        if not hits:
            return "没能从图片里识别出本店在售的商品。可以换个角度再拍一张，或直接告诉我你想找什么。"
        count = len(hits[:3])
        lead = (
            f"没找到完全同款，这{count}款风格或品类接近你的图片："
            if low_conf
            else f"根据你的图片，这{count}款最接近："
        )
        lines = [lead]
        for idx, hit in enumerate(hits[:3], start=1):
            product = hit.product
            price_label = self._catalog.price_label(product, filters)
            lines.append(f"{idx}. {product['title']}，{product['brand']}，价格：{price_label}。")
        lines.append("以上商品均来自当前商品库；图片仅用于检索，不代表本店有完全相同的商品。")
        return "\n".join(lines)

    def _prepare_search(
        self,
        query: str,
        session_id: str | None,
        filters: SearchFilters,
        top_k: int,
        recent_product_ids: list[str],
    ) -> PreparedChat:
        # Backtracking ("回到最开始那个"): the LLM picked exact product ids from session_products,
        # return those cards directly. Ids are validated against the catalog (the LLM can only
        # copy from the list we gave it, but we never trust an id we can't resolve).
        recalled = [pid for pid in filters.recall_product_ids if self._catalog.get(pid) is not None]
        if recalled:
            return self._prepare_recall(query, session_id, filters, recalled)
        result_count = _effective_result_count(top_k, filters)
        # Filter-keyed cache: a context-free product search keyed on the parsed intent. A hit
        # replays the stored answer + cards, skipping embed, retrieval and the answer LLM. The
        # key is set on a miss too, so the generated answer gets stored once it's produced.
        filter_key: str | None = None
        if self._filter_cache.enabled and FilterCache.eligible(filters, recent_product_ids):
            filter_key = self._filter_cache.key(filters, result_count)
            cached = self._filter_cache.get(filter_key)
            if cached is not None:
                return self._prepared_from_cache(query, session_id, filters, cached, filter_key)
        # The rewrite folds carried context into a standalone retrieval query. The answer
        # itself still replies to what the user actually typed (raw query below).
        search_query = filters.rewritten_query or query
        # Over-fetch so that dropping already-seen ("换一批") or excluded ("不要油腻") items still
        # leaves enough to fill top_k.
        buffer = (len(recent_product_ids) if filters.exclude_seen else 0) + (result_count if filters.excluded_terms else 0)
        limit = result_count + buffer
        # A pure price/rating sort must see the WHOLE filtered category, not just the relevance-top-k,
        # or retrieval truncates the true cheapest/highest-rated away before _order_hits sorts. But
        # only when there are no soft relevance signals: with required_terms/requested_specs ("便宜的
        # 敏感肌面霜") the relevance ranking carries the real intent, so over-fetching then price-sorting
        # would surface the cheapest-overall instead of the cheapest-relevant. The category gate keeps
        # the pool small, so fetching it all is cheap.
        if filters.sort_by != "relevance" and not filters.required_terms and not filters.requested_specs:
            limit = max(limit, _SORT_CANDIDATE_POOL)
        if _asks_for_apple_and_android_phones(query, filters):
            limit = max(limit, _SORT_CANDIDATE_POOL)
        retrieval = self._retriever.retrieve(query=search_query, filters=filters, limit=limit)
        hits = self._order_hits(retrieval.hits, filters)
        if filters.exclude_seen:
            seen = set(recent_product_ids)
            hits = [hit for hit in hits if hit.product["product_id"] not in seen]
        if filters.excluded_terms:
            excluded = self._excluded_ids(hits, filters.excluded_terms)
            hits = [hit for hit in hits if hit.product["product_id"] not in excluded]
        # The user named a product type the catalogue can only gate as a sub-category group (运动鞋 ->
        # 跑步鞋/篮球鞋/徒步鞋). Keep only cards in that group so an out-of-type "closest" match (a hoodie
        # for a shoe query) is dropped and the answer is an honest no-match rather than off-type padding.
        type_subs = self._type_subcategories(filters)
        if type_subs:
            hits = [hit for hit in hits if hit.product["sub_category"] in type_subs]
        hits = _apply_requested_diversity(query, filters, hits)
        hits = hits[:result_count]
        products = [
            self._catalog.product_card(hit.product, matched_reason=_reason(hit, filters, self._catalog), filters=filters)
            for hit in hits
        ]
        prev_floor = self._previous_floor(session_id)
        result_status = self._result_status(filters, products, recent_product_ids, prev_floor)
        context = self._status_context(result_status, products, prev_floor)
        # Required attributes and requested specs rank rather than hard-filter. Flag any that
        # nothing retrieved matches, so the answer says so honestly instead of implying every card fits.
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
        # A single recalled product is a detail/focus view ("第一个怎么样"); it must not reshuffle the
        # ordinal order. A multi-item recall ("回到那批") is a real list and becomes the current batch.
        return self._search_prepared(
            query, session_id, filters, hits, products, RetrievalResult(hits=hits, source="lexical"),
            advances_order=len(product_ids) > 1,
        )

    def _available_by_id(self, session_id: str | None, hits: list[CatalogHit]) -> dict[str, int]:
        """Session-aware availability for narration: each product's seeded base stock (already on the
        product dict) minus what this session has ordered, so a re-narration after a buy-out shows it
        sold out. Shared by the text and photo answer paths so they never disagree."""
        sold = self._session(session_id).order.stock_sold if session_id else {}
        return {
            hit.product["product_id"]: max(0, int(hit.product.get("stock", 0)) - sold.get(hit.product["product_id"], 0))
            for hit in hits
        }

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
        advances_order: bool = True,
    ) -> PreparedChat:
        # Shared tail for the search and recall paths: record the turn, narrate, package.
        self._remember_shown_products(session_id, products, advances_order)
        self._remember_turn(session_id, query, filters, products)
        grounded_answer = self._grounded_answer(query, filters, hits, result_status, context)
        # No retrieved products means no facts to narrate: skip the answer LLM (it would invent
        # plausible-but-fake products) and let the deterministic grounded "no match" answer stand.
        # Mirrors the photo path, which already guards this the same way.
        # Availability the answer states is session-aware (see _available_by_id), so a re-search
        # after buying a product out shows it as sold out.
        available_by_id = self._available_by_id(session_id, hits)
        messages = (
            build_messages(query, filters, hits, self._catalog, result_status, context, available_by_id) if hits else []
        )
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
        products = self._remember_cached_turn(session_id, query, filters, cached)
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

    def _type_subcategories(self, filters: SearchFilters) -> set[str]:
        # Acceptable sub-categories when the user named a product type that didn't resolve to a single
        # sub_category (运动鞋). Empty when a sub_category already gates, or no required term is a type.
        if filters.sub_category:
            return set()
        subs: set[str] = set()
        for term in filters.required_terms:
            subs |= self._catalog.type_group_for_term(term)
        return subs

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
        # A refinement that surfaced only items already shown earlier, nothing new/better.
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
        and negation over the small shortlist (primary). The deterministic negation-aware catalog
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
                if "exclude" in payload:  # a parseable verdict, garbage -> fall through
                    valid = {hit.product["product_id"] for hit in hits}
                    return {pid for pid in payload["exclude"] if pid in valid}
            except Exception:  # noqa: BLE001 (any judge failure must degrade to the deterministic check)
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
        image_bytes: bytes | None = None,
        client_address: str | None = None,
    ) -> ChatResponse:
        prepared = self.prepare(
            query, session_id, top_k, compare_product_ids,
            client_recent_product_ids, cart_items, image_bytes=image_bytes,
            client_address=client_address,
        )
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

    def record_cached_turn(self, session_id: str | None, query: str, cached: dict) -> None:
        """A query-cache hit is served by the API layer without running prepare(), so session
        memory never sees the turn and the next message has nothing to carry over or resolve
        references against. Rebuild the shown products and parsed filters from the cached
        response and record the turn, so a follow-up after a cache hit behaves the same as after
        a freshly computed answer."""
        if not session_id:
            return
        intent = {key: value for key, value in (cached.get("intent") or {}).items() if key in _SEARCH_FILTER_FIELDS}
        filters = SearchFilters(**intent) if intent else SearchFilters(raw_query=query)
        self._remember_cached_turn(session_id, query, filters, cached)

    def _remember_cached_turn(
        self, session_id: str | None, query: str, filters: SearchFilters, cached: dict
    ) -> list[ProductCard]:
        """Rebuild the shown products from a cached response and record the turn in session memory.
        Shared by both cache-hit paths (filter-cache and query-cache). Returns the products so the
        caller can reuse them."""
        products = [ProductCard(**product) for product in cached.get("products", [])]
        self._remember_shown_products(session_id, products)
        self._remember_turn(session_id, query, filters, products)
        return products

    def stream_answer(self, prepared: PreparedChat) -> Iterator[str]:
        if prepared.messages and self._llm is not None and self._llm.available:
            streamed = False
            try:
                for token in self._llm.stream(prepared.messages):
                    streamed = True
                    yield token
                return
            except Exception:  # noqa: BLE001 (stream must degrade, not crash the response)
                if streamed:
                    # Partial answer already sent, ending beats duplicating it with the fallback.
                    return
        yield from _chunk_text(prepared.grounded_answer, self._settings.stream_chunk_size)

    def _session(self, session_id: str | None) -> SessionState:
        return self._sessions.setdefault(session_id or "", SessionState())

    def has_session_history(self, session_id: str | None) -> bool:
        """Whether this session has any prior turns/shown products. The API uses this to bypass the
        text-keyed query cache for follow-ups (which may be context-dependent refinements), without
        peeking only at the client-provided recent ids. Does not create a session entry."""
        state = self._sessions.get(session_id or "")
        return bool(state and (state.turns or state.shown_products))

    def _shown_by_recency(self, session_id: str | None) -> list[dict]:
        # Single source of truth for the "most recent turn first, display order within a turn"
        # ordering. The intent prompt ("最近展示的排在最前") and the deterministic ordinal
        # fallback both rely on this exact order, so it lives in one place to avoid drift.
        shown = self._session(session_id).shown_products
        return sorted(shown, key=lambda item: (-item["last_seq"], item["position"]))

    def _latest_batch_ids(self, session_id: str | None) -> list[str]:
        # Ids of just the most-recently-shown batch (one turn). "都/全部加入" means this batch, not
        # every product seen all session; commerce uses it to deterministically scope an add-all.
        shown = self._session(session_id).shown_products
        if not shown:
            return []
        latest = max(item["last_seq"] for item in shown)
        return [item["id"] for item in shown if item["last_seq"] == latest]

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

    def _remember_shown_products(
        self, session_id: str | None, products: list[ProductCard], advances_order: bool = True
    ) -> None:
        """Record shown products in the single session-wide log. New products are appended in
        first-shown order (for recall). A re-shown product keeps its place but updates its
        last_seq/position (so the derived recency view stays correct)."""
        if not session_id:
            return
        state = self._session(session_id)
        # A fresh product list invalidates the previous comparison's winner, so a later "更好的那个"
        # can't resolve to a stale winner from an unrelated earlier list. A comparison re-sets it right
        # after this call; cart turns never get here, so it survives a compare -> "把更好的加入" follow-up.
        state.last_winner_id = None
        if not products:
            return
        # A single-product detail/focus re-display ("第一个怎么样") must not advance the batch order:
        # the product is already in memory, and bumping its recency floats it to the front of the
        # ordinal view, so a later "第N个" would resolve against a reshuffled pool and target the
        # wrong card. Only a genuine list (search / 换一批 / multi-item recall) advances the order.
        if not advances_order:
            return
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
        # Server batch memory is authoritative for ordinal order ("第N个" = the display order of the
        # latest list). The client's recent buffer is only a fallback for when server memory is empty
        # (a cached stream replay or a restart); used first it would resolve ordinals against the
        # client's recency order, which a single-item detail view silently moves to the front.
        for entry in self._session_products(session_id) or []:
            pid = entry["id"]
            if pid not in seen:
                items.append(entry)
                seen.add(pid)
        for pid in client_recent_product_ids:
            product = self._catalog.get(pid)
            if product is None or pid in seen:
                continue
            items.append(self._commerce_entry(pid, product))
            seen.add(pid)
        return items or None

    def _cart_for_intent(self, cart_items: list[dict] | None) -> list[dict] | None:
        """Compact cart view for the intent router, so it knows a cart exists and can route cart-view
        / remove-by-description turns ("购物车里有什么", "把最贵的删了") to cart_action instead of search."""
        items: list[dict] = []
        for raw in cart_items or []:
            pid = raw.get("product_id") or (raw.get("product") or {}).get("product_id")
            product = self._catalog.get(pid) if pid else None
            if product is None:
                continue
            items.append({
                "title": product["title"],
                "price": self._catalog.lowest_price(product),
                "quantity": raw.get("quantity", 1),
            })
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

        display_hits = hits[: _answer_product_count(filters, len(hits))]
        order_note = "，并按价格从低到高排列" if filters.prefer_low_price else ""
        unmet = (context or {}).get("unmet_terms")
        if unmet:
            lines = [f"没有在商品库里找到明确标注“{'、'.join(unmet)}”的商品，以下是最接近的{len(display_hits)}款{order_note}："]
        else:
            lines = [f"我按你的条件从商品库里筛选出以下{len(display_hits)}款{order_note}："]
        for idx, hit in enumerate(display_hits, start=1):
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


def _effective_result_count(top_k: int, filters: SearchFilters) -> int:
    if filters.requested_count is not None:
        return max(1, min(filters.requested_count, 10))
    return max(1, top_k)


def _answer_product_count(filters: SearchFilters, available_count: int) -> int:
    if filters.requested_count is not None:
        return min(available_count, max(1, min(filters.requested_count, 10)))
    return min(available_count, 3)


def _apply_requested_diversity(query: str, filters: SearchFilters, hits: list[CatalogHit]) -> list[CatalogHit]:
    if not _asks_for_apple_and_android_phones(query, filters):
        return hits
    apple = [hit for hit in hits if hit.product.get("brand") == "Apple 苹果"]
    android = [hit for hit in hits if hit.product.get("brand") != "Apple 苹果"]
    if not apple or not android:
        return hits
    selected_ids = {apple[0].product["product_id"], android[0].product["product_id"]}
    rest = [hit for hit in hits if hit.product["product_id"] not in selected_ids]
    return [apple[0], android[0], *rest]


def _asks_for_apple_and_android_phones(query: str, filters: SearchFilters) -> bool:
    text = query.lower()
    wants_apple = "apple" in text or "iphone" in text or "苹果" in query
    wants_android = "android" in text or "安卓" in query
    wants_phone = filters.sub_category == "智能手机" or "phone" in text or "手机" in query
    return wants_apple and wants_android and wants_phone


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
        # it was requested). required_terms rank rather than hard-filter, so a card may not match.
        if catalog.evidences_required_term(product, term):
            reasons.append(f"匹配{term}需求")
    if filters.prefer_low_price:
        reasons.append("优先低价")
    if hit.snippets:
        reasons.append("商品描述或评价中有相关信息")
    return "，".join(reasons) if reasons else "与当前需求语义匹配"


def _photo_reason(low_conf: bool) -> str:
    return "与你的图片整体风格接近" if low_conf else "与你的图片在品类与外观上接近"


def _copy_plan(plan: ExecutionPlan) -> ExecutionPlan:
    if hasattr(plan, "model_dump"):
        return ExecutionPlan(**plan.model_dump())
    return ExecutionPlan(**plan.dict())


def _chunk_text(text: str, chunk_size: int) -> Iterator[str]:
    for idx in range(0, len(text), chunk_size):
        yield text[idx: idx + chunk_size]
