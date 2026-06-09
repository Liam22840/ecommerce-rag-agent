"""API and internal data models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from server.textutil import dedupe_ids


class Attachment(BaseModel):
    type: Literal["image"] = "image"
    data: str = Field(..., max_length=14_000_000)  # base64 of an image up to ~10MB
    mime: str = "image/jpeg"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=128)
    conversation_id: str | None = Field(default=None, max_length=128)
    top_k: int = Field(default=3, ge=1, le=10)
    compare_product_ids: list[str] = Field(default_factory=list, max_length=3)
    attachments: list[Attachment] = Field(default_factory=list, max_length=1)
    client_context: "ClientContext" = Field(default_factory=lambda: ClientContext())

    @property
    def effective_session_id(self) -> str | None:
        return self.session_id or self.conversation_id

    @property
    def image_attachment(self) -> Attachment | None:
        for attachment in self.attachments:
            if attachment.type == "image":
                return attachment
        return None

    @property
    def effective_compare_product_ids(self) -> list[str]:
        return dedupe_ids(self.compare_product_ids + self.client_context.compare_product_ids)


class ClientContext(BaseModel):
    cart_items: list[dict[str, Any]] = Field(default_factory=list)
    recent_product_ids: list[str] = Field(default_factory=list, max_length=10)
    compare_product_ids: list[str] = Field(default_factory=list, max_length=3)
    # The shipping address the client currently shows. Sent every turn so the order carries the
    # real address; a conversational "把地址改成…" overrides it for that turn (see set_address).
    address: str | None = Field(default=None, max_length=200)


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


class CartItem(BaseModel):
    product_id: str
    quantity: int = Field(default=1, ge=0)
    product: ProductCard
    sku_id: str | None = None
    unit_price: float
    price_label: str
    line_total: float


class CartUpdate(BaseModel):
    items: list[CartItem] = Field(default_factory=list)
    summary: str
    action: str
    subtotal: float = 0.0
    needs_clarification: bool = False


class OrderDraft(BaseModel):
    order_id: str | None = None
    status: Literal["awaiting_confirmation", "submitted", "cancelled"]
    items: list[CartItem] = Field(default_factory=list)
    subtotal: float = 0.0
    address: str = "默认地址"
    summary: str


class PlanStep(BaseModel):
    step_id: str
    title: str
    action: Literal[
        "product_search",
        "select_products",
        "comparison",
        "cart_action",
        "checkout",
        "ask_clarification",
    ]
    status: Literal["pending", "running", "done", "skipped", "failed"] = "pending"
    summary: str | None = None


class ExecutionPlan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    summary: str | None = None


class ChatResponse(BaseModel):
    answer: str
    products: list[ProductCard]
    comparison: ProductComparison | None = None
    cart: CartUpdate | None = None
    order: OrderDraft | None = None
    plan: ExecutionPlan | None = None
    session_id: str | None = None
    intent: dict[str, Any]
    retrieval_source: Literal["vector", "lexical", "hybrid", "none"]
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
