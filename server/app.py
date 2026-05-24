"""FastAPI entrypoint for the shopping assistant backend."""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from server.assistant import PreparedChat, ShoppingAssistant
from server.catalog import ProductCatalog
from server.config import Settings
from server.llm import ArkChatClient
from server.retrieval import ProductRetriever
from server.schemas import ChatRequest, ChatResponse


def create_app(settings: Settings | None = None, assistant: ShoppingAssistant | None = None) -> FastAPI:
    settings = settings or Settings.load()
    app = FastAPI(title="E-commerce RAG Agent API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
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
        assistant = ShoppingAssistant(catalog=catalog, retriever=retriever, llm=llm)

    app.state.settings = settings
    app.state.assistant = assistant
    app.state.catalog = assistant.catalog

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="message cannot be empty")
        return app.state.assistant.answer(message, request.session_id, request.top_k)

    @app.post("/api/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="message cannot be empty")
        prepared = app.state.assistant.prepare(message, request.session_id, request.top_k)
        return StreamingResponse(
            _sse_stream(app.state.assistant, prepared),
            media_type="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/products/{product_id}")
    def product_detail(product_id: str) -> dict:
        product = app.state.catalog.get(product_id)
        if product is None:
            raise HTTPException(status_code=404, detail="product not found")
        return product

    return app


def _sse_stream(assistant: ShoppingAssistant, prepared: PreparedChat) -> Iterator[str]:
    yield _sse("meta", {
        "session_id": prepared.session_id,
        "intent": prepared.filters.to_dict(),
        "retrieval_source": prepared.retrieval.source,
        "warnings": prepared.retrieval.warnings,
    })
    for token in assistant.stream_answer(prepared):
        yield _sse("delta", {"text": token})
    yield _sse("products", {"products": [_model_dump(p) for p in prepared.products]})
    yield _sse("done", {"ok": True})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _model_dump(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


app = create_app()
