"""API and internal data models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from server.textutil import dedupe_ids


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=128)
    conversation_id: str | None = Field(default=None, max_length=128)
    top_k: int = Field(default=3, ge=1, le=10)
    compare_product_ids: list[str] = Field(default_factory=list, max_length=3)
    client_context: "ClientContext" = Field(default_factory=lambda: ClientContext())

    @property
    def effective_session_id(self) -> str | None:
        return self.session_id or self.conversation_id

    @property
    def effective_compare_product_ids(self) -> list[str]:
        return dedupe_ids(self.compare_product_ids + self.client_context.compare_product_ids)


class ClientContext(BaseModel):
    cart_items: list[dict[str, Any]] = Field(default_factory=list)
    recent_product_ids: list[str] = Field(default_factory=list, max_length=10)
    compare_product_ids: list[str] = Field(default_factory=list, max_length=3)


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


class ComparisonValue(BaseModel):
    product_id: str
    value: str
    evidence: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low", "none"] = "none"


class ComparisonRow(BaseModel):
    dimension: str
    values: list[ComparisonValue]
    winner_product_id: str | None = None
    verdict: str


class ProductComparison(BaseModel):
    products: list[ProductCard]
    focus: list[str]
    rows: list[ComparisonRow]
    winner_product_id: str | None = None
    recommendation: str
    summary: str
    clarification: str | None = None


class ChatResponse(BaseModel):
    answer: str
    products: list[ProductCard]
    comparison: ProductComparison | None = None
    session_id: str | None = None
    intent: dict[str, Any]
    retrieval_source: Literal["vector", "lexical", "hybrid", "none"]
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
