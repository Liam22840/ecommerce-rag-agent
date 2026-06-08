"""Commerce intent parsing and execution for cart/order turns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from server.catalog import ProductCatalog
from server.llm import ChatClient
from server.pricing import build_cart_item, cart_subtotal, money
from server.prompts import commerce_intent_messages
from server.schemas import CartItem, CartUpdate, OrderDraft
from server.textutil import json_object


ActionType = Literal[
    "add",
    "remove",
    "set_quantity",
    "increment",
    "decrement",
    "clear",
    "show_cart",
    "checkout",
    "confirm_order",
    "cancel_order",
    "none",
]

Scope = Literal["shown_products", "cart_items", "unknown"]


@dataclass
class CommerceActionCandidate:
    action: ActionType = "none"
    refs: list[str] = field(default_factory=list)
    product_ids: list[str] = field(default_factory=list)
    quantity: int | None = None
    target_scope: Scope = "unknown"
    confidence: str = "low"

    @property
    def is_commerce(self) -> bool:
        return self.action != "none"


@dataclass
class CommerceResult:
    answer: str
    cart: CartUpdate | None = None
    order: OrderDraft | None = None
    intent: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderState:
    draft: OrderDraft | None = None
    cart_signature: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    pending_action: CommerceActionCandidate | None = None


COMMERCE_ACTIONS = {
    "add",
    "remove",
    "set_quantity",
    "increment",
    "decrement",
    "clear",
    "show_cart",
    "checkout",
    "confirm_order",
    "cancel_order",
    "none",
}

_ADD_HINTS = ("加入购物车", "加到购物车", "加购物车", "加购", "放购物车", "放进购物车", "买这个", "买它", "来一件", "来两件", "来2件")
_REMOVE_HINTS = ("删除", "移除", "去掉", "不要了", "不要这个", "删掉")
_CLEAR_HINTS = ("清空购物车", "购物车清空")
_SHOW_HINTS = ("购物车里有什么", "查看购物车", "看购物车", "购物车")
_CHECKOUT_HINTS = ("下单", "结算", "提交订单", "去支付")
_CONFIRM_HINTS = ("确认", "确定", "可以了", "提交吧", "用默认地址")
_CANCEL_HINTS = ("取消", "先不买", "不下单")
_INCREMENT_HINTS = ("再加一件", "多加一件", "加一件", "再来一件")
_DECREMENT_HINTS = ("减一件", "少一件")
_SET_QTY_HINTS = ("数量改成", "数量设为", "改成")


def looks_like_commerce(message: str) -> bool:
    text = message.strip()
    hints = (
        _ADD_HINTS
        + _REMOVE_HINTS
        + _CLEAR_HINTS
        + _CHECKOUT_HINTS
        + _CONFIRM_HINTS
        + _CANCEL_HINTS
        + _INCREMENT_HINTS
        + _DECREMENT_HINTS
        + _SET_QTY_HINTS
    )
    if any(hint in text for hint in hints):
        return True
    if re.search(r"(第[一二三四五六七八九十\d]+|这个|刚才).*(件|买|加|删|不要|数量|购物车)", text):
        return True
    return bool(re.search(r"(加|买|放|加入).*(第[一二三四五六七八九十\d]+|这个|刚才)", text))


class CommerceService:
    def __init__(self, catalog: ProductCatalog, llm: ChatClient | None = None):
        self._catalog = catalog
        self._llm = llm

    def maybe_handle(
        self,
        message: str,
        *,
        cart_items: list[dict[str, Any]],
        session_products: list[dict[str, Any]] | None,
        order_state: OrderState,
    ) -> CommerceResult | None:
        if order_state.pending_action is not None and _looks_like_pending_reply(message):
            candidate = _candidate_from_pending(order_state.pending_action, message)
            return self._apply(candidate, message, cart_items, session_products, order_state)

        if not looks_like_commerce(message) and order_state.draft is None:
            return None
        deterministic = self._deterministic_parse(message, bool(cart_items), bool(order_state.draft))
        llm_candidate = None
        if not self._candidate_is_complete(deterministic, cart_items, session_products):
            llm_candidate = self._llm_parse(message, cart_items, session_products)
        candidate = self._choose_candidate(deterministic, llm_candidate)
        if not candidate.is_commerce:
            return None
        return self._apply(candidate, message, cart_items, session_products, order_state)

    def apply_candidate(
        self,
        candidate: CommerceActionCandidate,
        message: str,
        *,
        cart_items: list[dict[str, Any]],
        session_products: list[dict[str, Any]] | None,
        order_state: OrderState,
    ) -> CommerceResult:
        return self._apply(candidate, message, cart_items, session_products, order_state)

    def _deterministic_parse(
        self,
        message: str,
        has_cart: bool,
        has_order_draft: bool,
    ) -> CommerceActionCandidate:
        text = message.strip()
        quantity = _parse_quantity(text)
        refs = _parse_refs(text)

        if any(hint in text for hint in _CANCEL_HINTS):
            return CommerceActionCandidate("cancel_order", refs, quantity=quantity, confidence="high")
        if has_order_draft and any(hint in text for hint in _CONFIRM_HINTS):
            return CommerceActionCandidate("confirm_order", refs, quantity=quantity, confidence="high")
        if any(hint in text for hint in _CLEAR_HINTS):
            return CommerceActionCandidate("clear", refs, quantity=quantity, target_scope="cart_items", confidence="high")
        if any(hint in text for hint in _CHECKOUT_HINTS):
            return CommerceActionCandidate("checkout", refs, quantity=quantity, target_scope="cart_items", confidence="high")
        if any(hint in text for hint in _REMOVE_HINTS):
            return CommerceActionCandidate("remove", refs, quantity=quantity, target_scope="cart_items", confidence="high")
        if any(hint in text for hint in _DECREMENT_HINTS):
            return CommerceActionCandidate("decrement", refs, quantity=quantity, target_scope="cart_items", confidence="high")
        if any(hint in text for hint in _INCREMENT_HINTS):
            return CommerceActionCandidate("increment", refs, quantity=quantity or 1, target_scope="cart_items", confidence="high")
        if any(hint in text for hint in _SET_QTY_HINTS) and quantity is not None:
            return CommerceActionCandidate("set_quantity", refs, quantity=quantity, target_scope="cart_items", confidence="high")
        if any(hint in text for hint in _ADD_HINTS) or _looks_like_add_ref(text, has_cart):
            return CommerceActionCandidate("add", refs, quantity=quantity or 1, target_scope="shown_products", confidence="high")
        if has_cart and any(hint in text for hint in _SHOW_HINTS):
            return CommerceActionCandidate("show_cart", refs, target_scope="cart_items", confidence="medium")
        return CommerceActionCandidate()

    def _llm_parse(
        self,
        message: str,
        cart_items: list[dict[str, Any]],
        session_products: list[dict[str, Any]] | None,
    ) -> CommerceActionCandidate | None:
        if self._llm is None or not self._llm.available:
            return None
        try:
            payload = json_object(self._llm.complete(commerce_intent_messages(message, cart_items, session_products)))
        except Exception:  # noqa: BLE001 (commerce must fall back to deterministic handling)
            return None
        action = payload.get("action")
        if action not in COMMERCE_ACTIONS:
            return None
        return CommerceActionCandidate(
            action=action,
            refs=[str(ref).strip() for ref in payload.get("refs", []) if str(ref).strip()],
            product_ids=[str(pid).strip() for pid in payload.get("product_ids", []) if str(pid).strip()],
            quantity=_coerce_int(payload.get("quantity")),
            target_scope=payload.get("target_scope") if payload.get("target_scope") in {"shown_products", "cart_items", "unknown"} else "unknown",
            confidence=payload.get("confidence") if payload.get("confidence") in {"high", "medium", "low"} else "low",
        )

    def _candidate_is_complete(
        self,
        candidate: CommerceActionCandidate,
        cart_items: list[dict[str, Any]],
        session_products: list[dict[str, Any]] | None,
    ) -> bool:
        if candidate.action in {"none"}:
            return False
        if candidate.action in {"clear", "show_cart", "checkout", "confirm_order", "cancel_order"}:
            return candidate.confidence == "high"
        if candidate.action in {"increment", "decrement", "set_quantity"} and len(cart_items) == 1:
            return True
        if candidate.product_ids:
            return True
        if candidate.refs:
            pool = cart_items if candidate.target_scope == "cart_items" else (session_products or [])
            return _ordinal_ref(candidate.refs) is not None and len(pool) >= _ordinal_ref(candidate.refs)
        return False

    def _choose_candidate(
        self,
        deterministic: CommerceActionCandidate,
        llm_candidate: CommerceActionCandidate | None,
    ) -> CommerceActionCandidate:
        if deterministic.confidence == "high" and deterministic.action != "none":
            if llm_candidate and llm_candidate.action != deterministic.action and llm_candidate.confidence == "high":
                return CommerceActionCandidate(action="none")
            if llm_candidate and (not deterministic.refs and llm_candidate.refs):
                deterministic.refs = llm_candidate.refs
            if llm_candidate and deterministic.quantity is None:
                deterministic.quantity = llm_candidate.quantity
            return deterministic
        return llm_candidate or deterministic

    def _apply(
        self,
        candidate: CommerceActionCandidate,
        message: str,
        raw_cart_items: list[dict[str, Any]],
        session_products: list[dict[str, Any]] | None,
        order_state: OrderState,
    ) -> CommerceResult:
        cart = self._normalize_cart(raw_cart_items)
        intent = {
            "intent_type": "checkout" if candidate.action in {"checkout", "confirm_order", "cancel_order"} else "cart_action",
            "commerce_action": candidate.action,
            "commerce_refs": candidate.refs,
            "quantity": candidate.quantity,
            "target_scope": candidate.target_scope,
        }

        if candidate.action == "clear":
            update = self._cart_update([], "clear", "购物车已清空。")
            order_state.pending_action = None
            order_state.draft = None
            return CommerceResult(update.summary, update, None, intent)
        if candidate.action == "show_cart":
            summary = self._cart_summary(cart) if cart else "购物车为空。"
            return CommerceResult(summary, self._cart_update(cart, "show_cart", summary), None, intent)
        if candidate.action == "checkout":
            order_state.pending_action = None
            return self._checkout(cart, order_state, intent)
        if candidate.action == "confirm_order":
            order_state.pending_action = None
            return self._confirm_order(cart, order_state, intent)
        if candidate.action == "cancel_order":
            order_state.pending_action = None
            order_state.draft = None
            order = OrderDraft(status="cancelled", items=cart, subtotal=cart_subtotal(cart), summary="已取消本次下单。")
            return CommerceResult("已取消本次下单，购物车商品仍会保留。", self._cart_update(cart, "cancel_order", self._cart_summary(cart)), order, intent)

        if candidate.action == "add":
            resolved = self._resolve_product(candidate, session_products or [], cart, prefer_cart=False)
            if isinstance(resolved, str):
                order_state.pending_action = candidate
                return self._clarify(cart, resolved, intent)
            qty = candidate.quantity or 1
            cart = self._upsert(cart, resolved.product_id, qty)
            summary = f"已将「{resolved.product.title}」加入购物车，数量 {self._quantity_for(cart, resolved.product_id)}。"
            order_state.pending_action = None
            order_state.draft = None
            return CommerceResult(summary, self._cart_update(cart, "add", summary), None, intent)

        resolved_item = self._resolve_cart_item(candidate, cart)
        if isinstance(resolved_item, str):
            order_state.pending_action = candidate
            return self._clarify(cart, resolved_item, intent)
        if candidate.action == "remove":
            cart = [item for item in cart if item.product_id != resolved_item.product_id]
            summary = f"已从购物车删除「{resolved_item.product.title}」。"
        elif candidate.action == "increment":
            delta = candidate.quantity or 1
            cart = self._upsert(cart, resolved_item.product_id, delta)
            summary = f"已将「{resolved_item.product.title}」数量增加到 {self._quantity_for(cart, resolved_item.product_id)}。"
        elif candidate.action == "decrement":
            cart = self._set_quantity(cart, resolved_item.product_id, resolved_item.quantity - (candidate.quantity or 1))
            summary = f"已更新「{resolved_item.product.title}」数量。"
        elif candidate.action == "set_quantity":
            qty = max(0, candidate.quantity or 0)
            cart = self._set_quantity(cart, resolved_item.product_id, qty)
            summary = f"已将「{resolved_item.product.title}」数量改成 {qty}。" if qty > 0 else f"已从购物车删除「{resolved_item.product.title}」。"
        else:
            order_state.pending_action = candidate
            return self._clarify(cart, "我还不能确定要怎么操作购物车，请换一种说法。", intent)
        order_state.draft = None
        order_state.pending_action = None
        update = self._cart_update(cart, candidate.action, summary)
        return CommerceResult(summary, update, None, intent)

    def _checkout(self, cart: list[CartItem], order_state: OrderState, intent: dict[str, Any]) -> CommerceResult:
        if not cart:
            update = self._cart_update([], "checkout", "购物车为空，先添加商品后再下单。")
            return CommerceResult(update.summary, update, None, intent)
        subtotal = cart_subtotal(cart)
        summary = f"订单待确认：共 {sum(item.quantity for item in cart)} 件商品，合计 {money(subtotal)}，收货地址为默认地址。确认后我会提交模拟订单。"
        order = OrderDraft(status="awaiting_confirmation", items=cart, subtotal=subtotal, summary=summary)
        order_state.draft = order
        order_state.cart_signature = _cart_signature(cart)
        return CommerceResult(summary, self._cart_update(cart, "checkout", self._cart_summary(cart)), order, intent)

    def _confirm_order(self, cart: list[CartItem], order_state: OrderState, intent: dict[str, Any]) -> CommerceResult:
        if not cart:
            update = self._cart_update([], "confirm_order", "购物车为空，无法提交订单。")
            order_state.draft = None
            return CommerceResult(update.summary, update, None, intent)
        if order_state.draft is None or order_state.cart_signature != _cart_signature(cart):
            return self._checkout(cart, order_state, intent)
        order_id = _order_id()
        subtotal = cart_subtotal(cart)
        summary = f"订单已提交，订单号 {order_id}，合计 {money(subtotal)}。"
        order = OrderDraft(order_id=order_id, status="submitted", items=cart, subtotal=subtotal, summary=summary)
        order_state.draft = None
        return CommerceResult(summary, self._cart_update([], "confirm_order", "订单已提交，购物车已清空。"), order, intent)

    def _normalize_cart(self, raw_items: list[dict[str, Any]]) -> list[CartItem]:
        items: list[CartItem] = []
        for raw in raw_items or []:
            pid = _item_product_id(raw)
            if not pid:
                continue
            product = self._catalog.get(pid)
            if product is None:
                continue
            quantity = _coerce_int(raw.get("quantity")) or 1
            if quantity <= 0:
                continue
            items.append(build_cart_item(self._catalog, product, quantity, raw.get("sku_id")))
        return _dedupe_cart(items, self._catalog)

    def _resolve_product(
        self,
        candidate: CommerceActionCandidate,
        session_products: list[dict[str, Any]],
        cart: list[CartItem],
        prefer_cart: bool,
    ) -> CartItem | str:
        for pid in candidate.product_ids:
            product = self._catalog.get(pid)
            if product is not None:
                return build_cart_item(self._catalog, product, 1)
        ordinal = _ordinal_ref(candidate.refs)
        if ordinal is not None:
            pool = cart if prefer_cart else session_products
            if 1 <= ordinal <= len(pool):
                pid = _pool_product_id(pool[ordinal - 1])
                product = self._catalog.get(pid)
                if product is not None:
                    return build_cart_item(self._catalog, product, 1)
            return "我没找到你说的第几个商品，请先让我推荐或明确商品名。"
        if _has_deictic_ref(candidate.refs):
            if len(session_products) == 1:
                product = self._catalog.get(_pool_product_id(session_products[0]))
                if product is not None:
                    return build_cart_item(self._catalog, product, 1)
            return "这里的“这个”不够明确，请告诉我是第几个商品。"
        if len(session_products) == 1:
            product = self._catalog.get(_pool_product_id(session_products[0]))
            if product is not None:
                return build_cart_item(self._catalog, product, 1)
        return "我还不能确定要加入哪款商品，请说“第一个/第二个”或商品名。"

    def _resolve_cart_item(self, candidate: CommerceActionCandidate, cart: list[CartItem]) -> CartItem | str:
        if not cart:
            return "购物车为空，无法执行这个操作。"
        for pid in candidate.product_ids:
            for item in cart:
                if item.product_id == pid:
                    return item
        ordinal = _ordinal_ref(candidate.refs)
        if ordinal is not None:
            if 1 <= ordinal <= len(cart):
                return cart[ordinal - 1]
            return "购物车里没有你说的第几个商品。"
        if len(cart) == 1:
            return cart[0]
        return "购物车里有多件商品，请告诉我想调整哪一件，或说明第几个商品。"

    def _upsert(self, cart: list[CartItem], product_id: str, quantity_delta: int) -> list[CartItem]:
        for index, item in enumerate(cart):
            if item.product_id == product_id:
                return self._set_quantity(cart, product_id, item.quantity + quantity_delta)
        product = self._catalog.require(product_id)
        return cart + [build_cart_item(self._catalog, product, max(1, quantity_delta))]

    def _set_quantity(self, cart: list[CartItem], product_id: str, quantity: int) -> list[CartItem]:
        next_items: list[CartItem] = []
        for item in cart:
            if item.product_id != product_id:
                next_items.append(item)
                continue
            if quantity > 0:
                product = self._catalog.require(product_id)
                next_items.append(build_cart_item(self._catalog, product, quantity, item.sku_id))
        return next_items

    def _quantity_for(self, cart: list[CartItem], product_id: str) -> int:
        return next((item.quantity for item in cart if item.product_id == product_id), 0)

    def _cart_update(
        self,
        items: list[CartItem],
        action: str,
        summary: str,
        needs_clarification: bool = False,
    ) -> CartUpdate:
        return CartUpdate(
            items=items,
            action=action,
            summary=summary,
            subtotal=cart_subtotal(items),
            needs_clarification=needs_clarification,
        )

    def _clarify(self, cart: list[CartItem], answer: str, intent: dict[str, Any]) -> CommerceResult:
        return CommerceResult(answer, self._cart_update(cart, "clarify", answer, True), None, intent)

    def _cart_summary(self, cart: list[CartItem]) -> str:
        if not cart:
            return "购物车为空。"
        count = sum(item.quantity for item in cart)
        return f"购物车共 {count} 件商品，合计 {money(cart_subtotal(cart))}。"


def _item_product_id(raw: dict[str, Any]) -> str | None:
    product = raw.get("product")
    if isinstance(product, dict):
        return product.get("product_id") or product.get("id")
    return raw.get("product_id") or raw.get("productID")


def _pool_product_id(item: dict[str, Any] | CartItem) -> str:
    if isinstance(item, CartItem):
        return item.product_id
    return item.get("id") or item.get("product_id") or item.get("productID")


def _dedupe_cart(items: list[CartItem], catalog: ProductCatalog) -> list[CartItem]:
    by_id: dict[str, int] = {}
    order: list[str] = []
    for item in items:
        if item.product_id not in by_id:
            order.append(item.product_id)
        by_id[item.product_id] = by_id.get(item.product_id, 0) + item.quantity
    return [build_cart_item(catalog, catalog.require(pid), qty) for pid, qty in ((pid, by_id[pid]) for pid in order)]


def _looks_like_add_ref(text: str, has_cart: bool) -> bool:
    if re.search(r"(加|买|放|加入).*(第[一二三四五六七八九十\d]+|这个|刚才)", text):
        return True
    if re.search(r"(第[一二三四五六七八九十\d]+|这个|刚才).*(加购物车|加入购物车|加购|买)", text):
        return True
    return "第" in text and "件" in text and not has_cart


def _looks_like_pending_reply(text: str) -> bool:
    stripped = text.strip()
    if _parse_refs(stripped):
        return True
    if _chinese_number(stripped) is not None:
        return True
    return bool(re.search(r"(加|买|放|加入).*(第[一二三四五六七八九十\d]+|这个|刚才)", stripped))


def _candidate_from_pending(
    pending: CommerceActionCandidate,
    message: str,
) -> CommerceActionCandidate:
    return CommerceActionCandidate(
        action=pending.action,
        refs=_pending_reply_refs(message),
        product_ids=[],
        quantity=pending.quantity,
        target_scope=pending.target_scope,
        confidence="high",
    )


def _pending_reply_refs(text: str) -> list[str]:
    refs = _parse_refs(text)
    if refs:
        return refs
    ordinal = _chinese_number(text.strip())
    if ordinal is not None:
        return [f"第{ordinal}个"]
    return []


def _parse_refs(text: str) -> list[str]:
    refs = []
    for match in re.finditer(r"第[一二三四五六七八九十\d]+个?", text):
        refs.append(match.group(0))
    if any(word in text for word in ("这个", "它", "刚才那个", "刚刚那个")):
        refs.append("这个")
    return refs


def _ordinal_ref(refs: list[str]) -> int | None:
    for ref in refs:
        match = re.search(r"第([一二三四五六七八九十\d]+)", ref)
        if match:
            return _chinese_number(match.group(1))
    return None


def _has_deictic_ref(refs: list[str]) -> bool:
    return any(ref in {"这个", "它", "刚才那个", "刚刚那个"} for ref in refs)


def _parse_quantity(text: str) -> int | None:
    patterns = [
        r"(?:数量|个数)\s*(?:改成|设为|设置为|变成)?\s*([零一二两三四五六七八九十\d]+)",
        r"来\s*([一二两三四五六七八九十\d]+)\s*件",
        r"买\s*([一二两三四五六七八九十\d]+)\s*件",
        r"([一二两三四五六七八九十\d]+)\s*件",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _chinese_number(match.group(1))
    return None


def _chinese_number(value: str) -> int | None:
    value = value.strip()
    if value.isdigit():
        return int(value)
    digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value in digits:
        return digits[value]
    if value == "十":
        return 10
    if "十" in value:
        left, _, right = value.partition("十")
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return _chinese_number(value) if not value.isdigit() else int(value)
    return None


def _cart_signature(cart: list[CartItem]) -> tuple[tuple[str, int], ...]:
    return tuple((item.product_id, item.quantity) for item in cart)


def _order_id() -> str:
    from datetime import datetime
    from uuid import uuid4

    return "EG" + datetime.now().strftime("%Y%m%d") + uuid4().hex[:6].upper()
