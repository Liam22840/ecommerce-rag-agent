"""FastAPI entrypoint for the shopping assistant backend."""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from server.assistant import PreparedChat, ShoppingAssistant, _chunk_text
from server.catalog import ProductCatalog
from server.config import Settings
from server.filter_cache import FilterCache
from server.llm import ArkChatClient
from server.query_cache import QueryCache
from server.retrieval import ProductRetriever
from server.schemas import ChatRequest, ChatResponse


def create_app(settings: Settings | None = None, assistant: ShoppingAssistant | None = None) -> FastAPI:
    settings = settings or Settings.load()
    app = FastAPI(title="E-commerce RAG Agent API", version="0.1.1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount(
        "/assets/products",
        StaticFiles(directory=settings.dataset_root),
        name="product-assets",
    )

    if assistant is None:
        catalog = ProductCatalog.load(settings.dataset_root)
        retriever = ProductRetriever(catalog, settings)
        llm = ArkChatClient(
            api_key=settings.chat_api_key if settings.enable_llm else None,
            base_url=settings.chat_base_url,
            model=settings.chat_model,
            timeout_seconds=settings.chat_timeout_seconds,
        )
        intent_llm = llm if settings.enable_llm_intent else None
        filter_cache = FilterCache(
            settings.filter_cache_path,
            max_entries=settings.filter_cache_max_entries,
            enabled=settings.enable_filter_cache,
        )
        assistant = ShoppingAssistant(
            catalog=catalog,
            retriever=retriever,
            llm=llm,
            intent_llm=intent_llm,
            settings=settings,
            filter_cache=filter_cache,
        )

    app.state.settings = settings
    app.state.assistant = assistant
    app.state.catalog = assistant.catalog
    app.state.query_cache = QueryCache(
        settings.query_cache_path,
        max_entries=settings.query_cache_max_entries,
        enabled=settings.enable_query_cache,
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="message cannot be empty")
        cache: QueryCache = app.state.query_cache
        key, cached = _cache_lookup(cache, message, request)
        if cached is not None:
            return ChatResponse(**cached)
        response = app.state.assistant.answer(
            message,
            request.effective_session_id,
            request.top_k,
            request.effective_compare_product_ids,
            request.client_context.recent_product_ids,
        )
        if key is not None:
            _maybe_store(cache, key, response)
        return response

    @app.post("/api/v1/chat/stream", include_in_schema=False)
    @app.post("/api/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="message cannot be empty")
        cache: QueryCache = app.state.query_cache
        key, cached = _cache_lookup(cache, message, request)
        if cached is not None:
            return _sse_response(_sse_replay(cached, settings.stream_chunk_size))
        # prepare() runs inside the generator (after the lead-in is flushed) so the first
        # streamed token lands in <1s instead of after intent + retrieval.
        return _sse_response(_sse_stream(app.state.assistant, message, request, cache, key))

    @app.get("/api/products/{product_id}")
    def product_detail(product_id: str) -> dict:
        product = app.state.catalog.get(product_id)
        if product is None:
            raise HTTPException(status_code=404, detail="product not found")
        return product

    return app


def _cache_lookup(cache: QueryCache, message: str, request: ChatRequest) -> tuple[str | None, dict | None]:
    compare_ids = request.effective_compare_product_ids
    recent_ids = request.client_context.recent_product_ids
    if not cache.eligible(compare_ids, recent_ids):
        return None, None
    key = cache.key(message, request.top_k)
    return key, cache.get(key)


def _maybe_store(cache: QueryCache, key: str, response: ChatResponse) -> None:
    if cache.storeable(response.intent.get("intent_type", "")):
        cache.put(key, _model_dump(response))


def _sse_stream(
    assistant: ShoppingAssistant,
    message: str,
    request: ChatRequest,
    cache: QueryCache | None = None,
    key: str | None = None,
) -> Iterator[str]:
    # Lead-in first: a deterministic, rule-picked opener flushed before intent+retrieval so 首
    # Token lands in <1s. Streaming-only — not collected into the stored answer.
    yield _token_event(assistant.lead_in(message, request.effective_compare_product_ids))
    try:
        prepared = assistant.prepare(
            message,
            request.effective_session_id,
            request.top_k,
            request.effective_compare_product_ids,
            request.client_context.recent_product_ids,
        )
    except Exception:  # noqa: BLE001 - lead-in already sent; close the stream cleanly
        yield _token_event("抱歉，出了一点问题，请再试一次。")
        yield _done_frame(request.effective_session_id, "none", ["prepare failed"])
        return

    # Cards first: emit the products the instant retrieval finishes, then stream the prose.
    products = [_enrich_product_dict(_model_dump(p)) for p in prepared.products]
    comparison = _model_dump(prepared.comparison) if prepared.comparison is not None else None
    yield _cards_frame(products, comparison)

    tokens: list[str] = []
    for token in assistant.stream_answer(prepared):
        tokens.append(token)
        yield _token_event(token)
    yield _done_frame(prepared.session_id, prepared.retrieval.source, list(prepared.retrieval.warnings))

    response = _response_from_prepared(prepared, "".join(tokens))
    if cache is not None and key is not None:
        _maybe_store(cache, key, response)
    assistant.maybe_store_filter_cache(prepared, response)


def _sse_replay(cached: dict, chunk_size: int) -> Iterator[str]:
    products = [_enrich_product_dict(dict(p)) for p in cached.get("products", [])]
    yield _cards_frame(products, cached.get("comparison"))
    for token in _chunk_text(cached.get("answer", ""), chunk_size):
        yield _token_event(token)
    yield _done_frame(cached.get("session_id"), cached.get("retrieval_source"), cached.get("warnings", []))


def _cards_frame(products: list[dict], comparison: dict | None) -> str:
    if comparison is not None:
        comparison = dict(comparison)
        comparison["products"] = products
        comparison["items"] = products
        comparison["type"] = "comparison"
        return _sse("comparison", comparison)
    return _sse("products", {"type": "products", "products": products, "items": products})


def _done_frame(session_id: str | None, retrieval_source: str | None, warnings: list[str]) -> str:
    return _sse("done", {
        "type": "done",
        "ok": True,
        "session_id": session_id,
        "retrieval_source": retrieval_source,
        "warnings": warnings,
    })


def _response_from_prepared(prepared: PreparedChat, answer: str) -> ChatResponse:
    return ChatResponse(
        answer=answer,
        products=prepared.products,
        comparison=prepared.comparison,
        session_id=prepared.session_id,
        intent=prepared.filters.to_dict(),
        retrieval_source=prepared.retrieval.source,
        degraded=bool(prepared.retrieval.warnings),
        warnings=list(prepared.retrieval.warnings),
    )


def _sse_response(stream: Iterator[str]) -> StreamingResponse:
    return StreamingResponse(
        stream,
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _token_event(token: str) -> str:
    return _sse("token", {"type": "token", "token": token, "delta": token, "text": token})


def _enrich_product_dict(product: dict) -> dict:
    product["base_price"] = product.get("price")
    product["reason"] = product.get("matched_reason")
    return product


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _model_dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


app = create_app()
