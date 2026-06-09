"""Multi-step task planning for compound shopping requests.

The planner only decomposes a user turn into executable steps. Product facts, prices,
comparison winners and cart mutations stay in the existing catalog/comparison/commerce
modules so the LLM cannot invent state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from server.commerce import looks_like_commerce
from server.llm import ChatClient
from server.prompts import planner_messages
from server.textutil import chinese_to_int, json_object


PlanAction = Literal[
    "product_search",
    "select_products",
    "comparison",
    "cart_action",
    "checkout",
    "ask_clarification",
]

ALLOWED_ACTIONS = {
    "product_search",
    "select_products",
    "comparison",
    "cart_action",
    "checkout",
    "ask_clarification",
}
SELECT_CRITERIA = {"price_asc", "price_desc", "rating_desc", "relevance"}
TARGETS = {"selected_products", "comparison_winner", "previous_step"}

# How many candidates the deterministic fallback shortlists: a comparison needs at least two to
# weigh against each other, anything else defaults to a single pick.
_COMPARISON_SELECTION_COUNT = 2
_DEFAULT_SELECTION_COUNT = 1
# A step title is the raw query when it's short enough to read at a glance, otherwise a generic label.
_MAX_PLAN_TITLE_LEN = 18


@dataclass
class PlannedStep:
    action: PlanAction
    title: str
    query: str = ""
    criteria: str | None = None
    count: int | None = None
    target: str | None = None
    quantity: int | None = None


@dataclass
class PlannedTask:
    steps: list[PlannedStep]


def looks_like_planned_task(message: str) -> bool:
    return _has_composite_connector(message) and _action_count(message) >= 2


class PlannerService:
    def __init__(
        self,
        categories: set[str],
        sub_categories: set[str],
        brands: set[str],
        llm: ChatClient | None = None,
    ):
        self._categories = categories
        self._sub_categories = sub_categories
        self._brands = brands
        self._llm = llm

    def plan(
        self,
        message: str,
        *,
        force: bool = False,
        session_products: list[dict[str, Any]] | None = None,
        cart_items: list[dict[str, Any]] | None = None,
    ) -> PlannedTask | None:
        # The router decides whether this is a multi-step task. `force` is set when the intent LLM
        # routed here; otherwise `looks_like_planned_task` is the fallback router's own pre-check.
        if not force and not looks_like_planned_task(message):
            return None
        llm_plan = self._llm_plan(message, session_products, cart_items or [])
        if llm_plan is not None:
            return llm_plan
        return self._fallback_plan(message)

    def _llm_plan(
        self,
        message: str,
        session_products: list[dict[str, Any]] | None,
        cart_items: list[dict[str, Any]],
    ) -> PlannedTask | None:
        if self._llm is None or not getattr(self._llm, "available", False):
            return None
        try:
            raw = self._llm.complete(
                planner_messages(
                    message,
                    self._categories,
                    self._sub_categories,
                    self._brands,
                    session_products=session_products,
                    cart_items=cart_items,
                )
            )
        except Exception:  # noqa: BLE001 (planner must degrade to deterministic handling)
            return None
        if raw.strip().lower() in {"null", "none", ""}:
            return None
        payload = json_object(raw)
        steps = payload.get("steps") if payload else None
        if not isinstance(steps, list):
            return None
        parsed = [_coerce_step(step) for step in steps if isinstance(step, dict)]
        parsed = [step for step in parsed if step is not None]
        return PlannedTask(parsed) if _valid_plan(parsed) else None

    def _fallback_plan(self, message: str) -> PlannedTask | None:
        actions = _action_count(message)
        if actions < 2:
            return None

        search_query = _search_query(message)
        criteria = _selection_criteria(message)
        has_comparison = _looks_like_comparison(message)
        has_cart = looks_like_commerce(message)
        has_checkout = _looks_like_checkout(message)
        count = _selection_count(message, default=_COMPARISON_SELECTION_COUNT if has_comparison else _DEFAULT_SELECTION_COUNT)

        steps: list[PlannedStep] = []
        if _looks_like_search(message):
            steps.append(PlannedStep(
                action="product_search",
                title=_title("推荐商品", search_query),
                query=search_query,
            ))
        if criteria != "relevance" or has_comparison or has_cart:
            steps.append(PlannedStep(
                action="select_products",
                title=_select_title(criteria, count),
                criteria=criteria,
                count=count,
            ))
        if has_comparison:
            steps.append(PlannedStep(
                action="comparison",
                title="对比候选商品",
                query=message,
                criteria="price_asc" if criteria == "price_asc" and _asks_cheaper(message) else None,
            ))
        if has_cart:
            steps.append(PlannedStep(
                action="cart_action",
                title="加入购物车",
                target="comparison_winner" if has_comparison else "selected_products",
                quantity=_quantity(message) or 1,
            ))
        if has_checkout:
            steps.append(PlannedStep(action="checkout", title="创建订单"))

        return PlannedTask(steps) if _valid_plan(steps) else None


def _coerce_step(raw: dict[str, Any]) -> PlannedStep | None:
    action = raw.get("action")
    if action not in ALLOWED_ACTIONS:
        return None
    criteria = raw.get("criteria")
    if criteria is not None:
        criteria = str(criteria).strip() or None
    if action == "select_products" and criteria not in SELECT_CRITERIA:
        criteria = "relevance"
    target = raw.get("target")
    if target is not None:
        target = str(target).strip() or None
    if action == "cart_action" and target not in TARGETS:
        target = "previous_step"
    return PlannedStep(
        action=action,
        title=str(raw.get("title") or _default_title(action)).strip(),
        query=str(raw.get("query") or "").strip(),
        criteria=criteria,
        count=_positive_int(raw.get("count")),
        target=target,
        quantity=_positive_int(raw.get("quantity")),
    )


def _valid_plan(steps: list[PlannedStep]) -> bool:
    actions = [step.action for step in steps]
    if len(actions) < 2:
        return False
    if actions.count("product_search") > 1:
        return False
    if "cart_action" in actions and not any(action in actions for action in {"product_search", "select_products", "comparison"}):
        return False
    return True


def _has_composite_connector(text: str) -> bool:
    return bool(re.search(r"(并且|并|然后|之后|再|同时|顺便|，|,|。)", text))


def _action_count(text: str) -> int:
    return sum([
        _looks_like_search(text),
        _looks_like_comparison(text),
        looks_like_commerce(text),
        _looks_like_checkout(text),
    ])


def _looks_like_search(text: str) -> bool:
    # Deliberately broad: this only pre-filters which turns are worth handing to the planner LLM,
    # which is the real decider (it returns null for anything that isn't a genuine multi-step task).
    # Enumerating product nouns here would silently drop requests for anything not in the list. The
    # bare 买 excludes a deictic add ("买这个/买它") so a plain cart-add isn't miscounted as a search.
    return bool(re.search(r"(推荐|找|看看|筛选|有没有|有哪些|想买|想要|要买|需要|买(?![这它那]))", text))


def _looks_like_comparison(text: str) -> bool:
    return bool(re.search(r"(对比|比较|哪个更|哪款更|哪一个更|更适合|选哪个|买哪个|二选一|这两款|这两个)", text))


def _looks_like_checkout(text: str) -> bool:
    return bool(re.search(r"(下单|结算|提交订单|去支付)", text))


def _asks_cheaper(text: str) -> bool:
    return bool(re.search(r"(便宜|价格低|低价|更省|更划算|性价比)", text))


def _selection_criteria(text: str) -> str:
    if _asks_cheaper(text) or re.search(r"(价格.*低|低.*价格)", text):
        return "price_asc"
    if re.search(r"(贵|高价|价格.*高)", text):
        return "price_desc"
    if re.search(r"(评分|评价|口碑|好评)", text):
        return "rating_desc"
    return "relevance"


def _selection_count(text: str, default: int) -> int:
    match = re.search(r"最[^，。,.]*?([一二两三四五六七八九十\d]+)\s*(?:个|件|款|双)", text)
    if match:
        return chinese_to_int(match.group(1)) or default
    if "两" in text or "二" in text:
        return max(default, _COMPARISON_SELECTION_COUNT)
    return default


def _quantity(text: str) -> int | None:
    match = re.search(r"([一二两三四五六七八九十\d]+)\s*(?:个|件|款|双)", text)
    return chinese_to_int(match.group(1)) if match else None


def _search_query(text: str) -> str:
    parts = re.split(r"(?:并且|然后|之后|同时|顺便|，|,|。)", text, maxsplit=1)
    query = parts[0].strip() if parts else text.strip()
    if query:
        return query
    return re.sub(r"(加入购物车|加到购物车|加购物车|加购|下单|结算)", "", text).strip()


def _title(default: str, query: str) -> str:
    return query if 0 < len(query) <= _MAX_PLAN_TITLE_LEN else default


def _select_title(criteria: str, count: int) -> str:
    label = {
        "price_asc": "筛选低价商品",
        "price_desc": "筛选高价商品",
        "rating_desc": "筛选高评分商品",
        "relevance": "筛选候选商品",
    }.get(criteria, "筛选候选商品")
    return f"{label} {count} 款"


def _default_title(action: str) -> str:
    return {
        "product_search": "推荐商品",
        "select_products": "筛选候选商品",
        "comparison": "对比候选商品",
        "cart_action": "加入购物车",
        "checkout": "创建订单",
        "ask_clarification": "确认需求",
    }.get(action, "执行任务")


def _positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    if isinstance(value, str):
        parsed = chinese_to_int(value) if not value.isdigit() else int(value)
        return parsed if parsed and parsed > 0 else None
    return None
