"""Commerce intent parsing and execution for cart/order turns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from server.catalog import ProductCatalog
from server.llm import ChatClient
from server.pricing import build_cart_item, cart_subtotal, money
from server.prompts import commerce_intent_messages, pending_reply_messages
from server.schemas import CartItem, CartUpdate, OrderDraft
from server.textutil import chinese_to_int, json_object


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
    # Per-item quantities for a multi-add with different counts ("第一个买两瓶，第二个买三瓶"):
    # product_id -> quantity. Falls back to `quantity` (then 1) for ids not listed here.
    item_quantities: dict[str, int] = field(default_factory=dict)
    # The 规格/SKU phrase the user named ("50g标准装"), resolved to a real sku_id at add time.
    sku: str | None = None

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
    # Per-session inventory ledger: product_id -> quantity already ordered this session. Available
    # stock is the catalogue's seeded base minus this. It decrements only when an order is submitted,
    # and persists for the session (it lives on the session's single OrderState).
    stock_sold: dict[str, int] = field(default_factory=dict)


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
        + _SHOW_HINTS
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

    def _available(self, product_id: str, order_state: OrderState) -> int:
        """Stock a shopper can still buy: the catalogue's seeded base minus what this session has
        already ordered. Pending cart lines don't count — only a submitted order decrements."""
        return max(0, self._catalog.stock(product_id) - order_state.stock_sold.get(product_id, 0))

    def maybe_handle(
        self,
        message: str,
        *,
        cart_items: list[dict[str, Any]],
        session_products: list[dict[str, Any]] | None,
        order_state: OrderState,
        comparison_winner_id: str | None = None,
        latest_batch_ids: list[str] | None = None,
    ) -> CommerceResult | None:
        if order_state.pending_action is not None and _looks_like_pending_reply(message):
            candidate = _candidate_from_pending(order_state.pending_action, message)
            return self._apply(candidate, message, cart_items, session_products, order_state)

        # The LLM fills the action structure (which item, action, quantity). Deterministic parsing is
        # only the fallback when the LLM is unavailable. Verification (the id was actually shown,
        # coercing quantity to an int) and execution (cart maths) stay deterministic in _apply.
        candidate = self._llm_parse(
            message, cart_items, session_products, comparison_winner_id, has_order_draft=bool(order_state.draft)
        )
        if candidate is None:
            candidate = self._deterministic_parse(message, bool(cart_items), bool(order_state.draft))
        if not candidate.is_commerce:
            return None
        return self._apply(
            candidate, message, cart_items, session_products, order_state, latest_batch_ids, comparison_winner_id
        )

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

    def handle_pending_reply(
        self,
        message: str,
        *,
        cart_items: list[dict[str, Any]],
        session_products: list[dict[str, Any]] | None,
        order_state: OrderState,
    ) -> CommerceResult | None:
        """Resolve a reply to an open "which item?" clarification. Returns a result when the pending
        action is continued; returns None when the message isn't a reply (caller routes it fresh).
        An ordinal/deictic reply is resolved deterministically; otherwise the LLM understands a
        natural-language answer ("那个便宜的") or detects abandonment ("算了，看看别的")."""
        pending = order_state.pending_action
        if pending is None:
            return None
        if _looks_like_pending_reply(message):
            candidate = _candidate_from_pending(pending, message)
            return self._apply(candidate, message, cart_items, session_products, order_state)
        if self._llm is None or not self._llm.available:
            return None  # deterministic-only: leave pending set, let the caller route fresh
        pool = self._pending_pool(pending, cart_items, session_products)
        try:
            payload = json_object(self._llm.complete(pending_reply_messages(message, pending.action, pool)))
        except Exception:  # noqa: BLE001 (resolver failure degrades to leaving pending set)
            return None
        outcome = payload.get("outcome")
        if outcome == "resolve":
            pid = str(payload.get("product_id") or "").strip()
            if pid and pid in {str(item.get("id")) for item in pool}:
                candidate = CommerceActionCandidate(
                    action=pending.action,
                    product_ids=[pid],
                    quantity=pending.quantity,
                    target_scope=pending.target_scope,
                    confidence="high",
                )
                return self._apply(candidate, message, cart_items, session_products, order_state)
            return None
        if outcome in {"abandon", "not_a_reply"}:
            order_state.pending_action = None
        return None

    def _pending_pool(
        self,
        pending: CommerceActionCandidate,
        cart_items: list[dict[str, Any]],
        session_products: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if pending.target_scope == "cart_items":
            return [
                {"id": item.product_id, "名称": item.product.title, "价格": item.price_label,
                 "评分": self._catalog.avg_rating(self._catalog.require(item.product_id))}
                for item in self._normalize_cart(cart_items)
            ]
        pool: list[dict[str, Any]] = []
        for p in (session_products or []):
            product = self._catalog.get(p.get("id"))
            pool.append({
                "id": p.get("id"), "名称": p.get("title"), "价格": p.get("price"),
                "评分": self._catalog.avg_rating(product) if product else None,
            })
        return pool

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
            # Leave quantity as None when none was parsed so a deferred LLM can supply it (e.g. a
            # measure word the parser can't read, "要五个"); _apply defaults to 1.
            return CommerceActionCandidate("add", refs, quantity=quantity, target_scope="shown_products", confidence="high")
        if has_cart and any(hint in text for hint in _SHOW_HINTS):
            return CommerceActionCandidate("show_cart", refs, target_scope="cart_items", confidence="medium")
        return CommerceActionCandidate()

    def _llm_parse(
        self,
        message: str,
        cart_items: list[dict[str, Any]],
        session_products: list[dict[str, Any]] | None,
        comparison_winner_id: str | None = None,
        has_order_draft: bool = False,
    ) -> CommerceActionCandidate | None:
        if self._llm is None or not self._llm.available:
            return None
        try:
            payload = json_object(self._llm.complete(
                commerce_intent_messages(message, cart_items, session_products, comparison_winner_id, has_order_draft)
            ))
        except Exception:  # noqa: BLE001 (commerce must fall back to deterministic handling)
            return None
        action = payload.get("action")
        if action not in COMMERCE_ACTIONS:
            return None
        product_ids = [str(pid).strip() for pid in payload.get("product_ids", []) if str(pid).strip()]
        # Optional per-item list ([{product_id, quantity}]) for different counts per product. Its ids
        # extend product_ids so resolution + the shown-item check treat them the same.
        item_quantities: dict[str, int] = {}
        for entry in payload.get("items", []) if isinstance(payload.get("items"), list) else []:
            if not isinstance(entry, dict):
                continue
            pid = str(entry.get("product_id", "")).strip()
            qty = _coerce_int(entry.get("quantity"))
            if pid:
                if pid not in product_ids:
                    product_ids.append(pid)
                if qty is not None:
                    item_quantities[pid] = qty
        sku_raw = payload.get("sku")
        sku = sku_raw.strip() or None if isinstance(sku_raw, str) else None
        return CommerceActionCandidate(
            action=action,
            refs=[str(ref).strip() for ref in payload.get("refs", []) if str(ref).strip()],
            product_ids=product_ids,
            quantity=_coerce_int(payload.get("quantity")),
            target_scope=payload.get("target_scope") if payload.get("target_scope") in {"shown_products", "cart_items", "unknown"} else "unknown",
            confidence=payload.get("confidence") if payload.get("confidence") in {"high", "medium", "low"} else "low",
            item_quantities=item_quantities,
            sku=sku,
        )

    def _apply(
        self,
        candidate: CommerceActionCandidate,
        message: str,
        raw_cart_items: list[dict[str, Any]],
        session_products: list[dict[str, Any]] | None,
        order_state: OrderState,
        latest_batch_ids: list[str] | None = None,
        comparison_winner_id: str | None = None,
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
            # A reference to a comparison result ("买之前对比胜出的那个") when no winner is on record
            # (a newer list cleared it) must clarify, not resolve to a stale or cart-biased product.
            if comparison_winner_id is None and _looks_like_winner_ref(message):
                return self._clarify(cart, "刚才的对比结果已经更新了，请直接说要加入哪一款（可以说序号或商品名）。", intent)
            resolved = self._resolve_products(candidate, session_products or [], cart, prefer_cart=False, message=message)
            if isinstance(resolved, str):
                order_state.pending_action = candidate
                return self._clarify(cart, resolved, intent)
            # "都/全部/所有加入" means the most-recently-shown batch, not every product seen this
            # session. The LLM is told this; scope it deterministically as a backstop when it over-reaches.
            if latest_batch_ids and len(resolved) > 1 and _looks_like_select_all(message):
                scoped = [item for item in resolved if item.product_id in set(latest_batch_ids)]
                if scoped:
                    resolved = scoped
            # quantity is per item; the LLM is told a count of products ("两双") goes in product_ids,
            # not here, so we trust it and default to 1 when unset. item_quantities overrides per id.
            qty = candidate.quantity or 1
            # A named 规格 ("50g标准装") only disambiguates a single-product add; resolve it to a real
            # sku_id so the line is priced for that SKU (None -> the default lowest SKU, as before).
            sku_id = None
            if candidate.sku and len(resolved) == 1:
                sku_id = self._catalog.sku_id_for_phrase(self._catalog.require(resolved[0].product_id), candidate.sku)
            # Each line can't exceed available stock (seeded base minus what this session already
            # ordered). The cap is on the resulting quantity, so an existing line + the new amount
            # is bounded. A sold-out product is skipped; an over-order is added up to the limit.
            added: list[tuple[Any, bool]] = []  # (resolved item, was it capped below the request?)
            sold_out: list[Any] = []
            for item in resolved:
                want = candidate.item_quantities.get(item.product_id, qty)
                room = self._available(item.product_id, order_state) - self._quantity_for(cart, item.product_id)
                take = min(want, room)
                if take <= 0:
                    sold_out.append(item)
                    continue
                cart = self._upsert(cart, item.product_id, take, sku_id)
                added.append((item, take < want))
            order_state.pending_action = None
            order_state.draft = None
            if not added:
                names = "、".join(f"「{item.product.title}」" for item in sold_out)
                summary = f"{names}库存不足，暂时无法加入购物车。"
                return CommerceResult(summary, self._cart_update(cart, "add", summary), None, intent)
            if len(resolved) == 1:
                item, capped = added[0]
                pid = item.product_id
                summary = f"已将「{item.product.title}」加入购物车，数量 {self._quantity_for(cart, pid)}。"
                if capped:
                    summary += f"（该商品库存仅剩 {self._available(pid, order_state)} 件，已按上限加入。）"
            else:
                summary = "已将 " + "、".join(f"「{item.product.title}」" for item, _ in added) + " 加入购物车。"
                if sold_out:
                    summary += "（" + "、".join(f"「{item.product.title}」" for item in sold_out) + " 库存不足，未加入。）"
                elif any(capped for _, capped in added):
                    summary += "（部分商品已按库存上限加入。）"
            return CommerceResult(summary, self._cart_update(cart, "add", summary), None, intent)

        resolved_item = self._resolve_cart_item(candidate, cart)
        if isinstance(resolved_item, str):
            order_state.pending_action = candidate
            return self._clarify(cart, resolved_item, intent)
        if candidate.action == "remove":
            cart = [item for item in cart if item.product_id != resolved_item.product_id]
            summary = f"已从购物车删除「{resolved_item.product.title}」。"
        elif candidate.action == "increment":
            avail = self._available(resolved_item.product_id, order_state)
            take = min(candidate.quantity or 1, avail - resolved_item.quantity)
            if take <= 0:
                summary = f"「{resolved_item.product.title}」库存仅剩 {avail} 件，已无法再增加。"
            else:
                cart = self._upsert(cart, resolved_item.product_id, take)
                summary = f"已将「{resolved_item.product.title}」数量增加到 {self._quantity_for(cart, resolved_item.product_id)}。"
                if take < (candidate.quantity or 1):
                    summary += f"（库存仅剩 {avail} 件，已按上限。）"
        elif candidate.action == "decrement":
            new_qty = resolved_item.quantity - (candidate.quantity or 1)
            cart = self._set_quantity(cart, resolved_item.product_id, new_qty)
            summary = (f"已将「{resolved_item.product.title}」数量减到 {new_qty}。" if new_qty > 0
                       else f"已从购物车删除「{resolved_item.product.title}」。")
        elif candidate.action == "set_quantity":
            sku_id = (self._catalog.sku_id_for_phrase(self._catalog.require(resolved_item.product_id), candidate.sku)
                      if candidate.sku else None)
            if candidate.quantity is None and sku_id is not None:
                # A pure variant switch ("换成家庭装"): keep the quantity, re-price to the named SKU.
                cart = self._set_quantity(cart, resolved_item.product_id, resolved_item.quantity, sku_id)
                summary = f"已将「{resolved_item.product.title}」换成{candidate.sku}。"
            else:
                avail = self._available(resolved_item.product_id, order_state)
                requested = max(0, candidate.quantity or 0)
                qty = min(requested, avail)
                cart = self._set_quantity(cart, resolved_item.product_id, qty, sku_id)
                summary = (f"已将「{resolved_item.product.title}」数量改成 {qty}。" if qty > 0
                           else f"已从购物车删除「{resolved_item.product.title}」。")
                if qty > 0 and requested > avail:
                    summary += f"（库存仅剩 {avail} 件，已按上限。）"
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
        # The order is real now, so its quantities leave inventory for the rest of the session: a
        # later add or search sees the reduced (or zero) availability. This is the "实时生效" part.
        for item in cart:
            order_state.stock_sold[item.product_id] = order_state.stock_sold.get(item.product_id, 0) + item.quantity
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

    def _resolve_products(
        self,
        candidate: CommerceActionCandidate,
        session_products: list[dict[str, Any]],
        cart: list[CartItem],
        prefer_cart: bool,
        message: str = "",
    ) -> list[CartItem] | str:
        """Resolve one or more products for an add. The LLM's ids (which can be several, e.g. a fuzzy
        "便宜的两个") and explicit ordinals ("第一个和第二个") resolve to multiple items; everything
        else falls back to the single-item resolver. LLM ids are verified against what was actually
        shown/in the cart, so the model can't add an off-screen product (e.g. resolving "最便宜的那个"
        to the globally cheapest item the user never saw)."""
        pool = cart if prefer_cart else session_products
        # An out-of-range ordinal ("第一百个" / "第零个" when 3 are shown) must clarify, even if the LLM
        # optimistically resolved it to a product_id — verify the named position against what's shown.
        # Read it from the original message too: the LLM often drops the ref once it has resolved an id.
        ordinals = _ordinal_refs(candidate.refs) or _ordinal_refs([message])
        if ordinals and any(not (1 <= o <= len(pool)) for o in ordinals):
            return f"现在只展示了 {len(pool)} 款商品，没有你说的那一个，请说一个有效的序号或商品名。"
        shown = {_pool_product_id(p) for p in session_products} | {item.product_id for item in cart}
        ids = [pid for pid in candidate.product_ids if pid in shown and self._catalog.get(pid) is not None]
        if ids:
            return [build_cart_item(self._catalog, self._catalog.require(pid), 1) for pid in ids]
        if ordinals:
            items = [
                build_cart_item(self._catalog, self._catalog.require(_pool_product_id(pool[o - 1])), 1)
                for o in ordinals
                if 1 <= o <= len(pool) and self._catalog.get(_pool_product_id(pool[o - 1])) is not None
            ]
            if items:
                return items
            return "我没找到你说的第几个商品，请先让我推荐或明确商品名。"
        single = self._resolve_product(candidate, session_products, cart, prefer_cart)
        return single if isinstance(single, str) else [single]

    def _resolve_product(
        self,
        candidate: CommerceActionCandidate,
        session_products: list[dict[str, Any]],
        cart: list[CartItem],
        prefer_cart: bool,
    ) -> CartItem | str:
        # product_ids and ordinals are resolved (and verified against shown items) in
        # _resolve_products; here we only handle the deictic / single-shown fallbacks.
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

    def _upsert(self, cart: list[CartItem], product_id: str, quantity_delta: int, sku_id: str | None = None) -> list[CartItem]:
        for item in cart:
            if item.product_id == product_id:
                return self._set_quantity(cart, product_id, item.quantity + quantity_delta, sku_id)
        product = self._catalog.require(product_id)
        return cart + [build_cart_item(self._catalog, product, max(1, quantity_delta), sku_id)]

    def _set_quantity(self, cart: list[CartItem], product_id: str, quantity: int, sku_id: str | None = None) -> list[CartItem]:
        next_items: list[CartItem] = []
        for item in cart:
            if item.product_id != product_id:
                next_items.append(item)
                continue
            if quantity > 0:
                product = self._catalog.require(product_id)
                # A new sku_id overrides; otherwise keep the line's existing SKU (decrement/set_quantity).
                next_items.append(build_cart_item(self._catalog, product, quantity, sku_id if sku_id is not None else item.sku_id))
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
    sku_by_id: dict[str, str | None] = {}
    order: list[str] = []
    for item in items:
        if item.product_id not in by_id:
            order.append(item.product_id)
            sku_by_id[item.product_id] = item.sku_id  # keep the chosen SKU; rebuilding without it reverts to the cheapest
        by_id[item.product_id] = by_id.get(item.product_id, 0) + item.quantity
    return [build_cart_item(catalog, catalog.require(pid), by_id[pid], sku_by_id[pid]) for pid in order]


def _looks_like_winner_ref(text: str) -> bool:
    """An explicit reference back to a comparison's result ("之前对比胜出的那款"). Used only to clarify
    when there is no winner on record; a bare "更划算的那个" (a description) is intentionally excluded."""
    return bool(re.search(r"胜出|之前对比|刚才对比|对比[^。]{0,6}(赢|胜|更好|那[个款])", text))


def _looks_like_select_all(text: str) -> bool:
    """A "select everything just shown" add ("都加入"/"全部要了"/"这些都要"). Used only to scope such
    an add to the latest batch; the LLM still decides the action."""
    return bool(re.search(r"都|全部|所有|全都|这些", text))


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
    if chinese_to_int(stripped) is not None:
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
    ordinal = chinese_to_int(text.strip())
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
    return next(iter(_ordinal_refs(refs)), None)


def _ordinal_refs(refs: list[str]) -> list[int]:
    out: list[int] = []
    for ref in refs:
        # Include 零/百/千 so an out-of-range ordinal ("第一百"=100, "第零"=0) parses to its real value
        # and the caller can reject it, instead of "第一百" mis-reading as "第一".
        match = re.search(r"第([零一二三四五六七八九十百千\d]+)", ref)
        if match:
            value = chinese_to_int(match.group(1))
            if value is not None:
                out.append(value)
    return out


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
            return chinese_to_int(match.group(1))
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return chinese_to_int(value) if not value.isdigit() else int(value)
    return None


def _cart_signature(cart: list[CartItem]) -> tuple[tuple[str, int], ...]:
    return tuple((item.product_id, item.quantity) for item in cart)


def _order_id() -> str:
    from datetime import datetime
    from uuid import uuid4

    return "EG" + datetime.now().strftime("%Y%m%d") + uuid4().hex[:6].upper()
