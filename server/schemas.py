"""API and internal data models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=128)
    conversation_id: str | None = Field(default=None, max_length=128)
    top_k: int = Field(default=3, ge=1, le=10)

    @property
    def effective_session_id(self) -> str | None:
        return self.session_id or self.conversation_id


class SkuPrice(BaseModel):
    sku_id: str | None = None
    label: str
    price: float


class ProductCard(BaseModel):
    product_id: str
    title: str
    brand: str
    category: str
    sub_category: str
    price: float
    price_label: str
    price_summary: str
    lowest_price_sku: SkuPrice | None = None
    selected_price_sku: SkuPrice | None = None
    image_path: str
    detail_path: str
    matched_reason: str | None = None


class ChatResponse(BaseModel):
    answer: str
    products: list[ProductCard]
    session_id: str | None = None
    intent: dict[str, Any]
    retrieval_source: Literal["vector", "lexical", "hybrid", "none"]
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
