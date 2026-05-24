"""Prompt construction for grounded shopping responses."""

from __future__ import annotations

import json

from server.catalog import CatalogHit, ProductCatalog
from server.intent import SearchFilters


SYSTEM_PROMPT = """你是一个电商智能导购助手。
你只能依据提供的商品事实回答，不能编造商品、价格、库存、优惠券、功效或参数。
如果候选商品不足以满足用户条件，要明确说明没有找到完全匹配项，并给出可继续筛选的问题。
回答要简洁、可执行，优先说明为什么推荐，以及和用户条件的对应关系。
价格必须使用商品事实中的 price 字段。
"""


def build_messages(
    query: str,
    filters: SearchFilters,
    hits: list[CatalogHit],
    catalog: ProductCatalog,
) -> list[dict[str, str]]:
    facts = [catalog.product_facts(hit.product) for hit in hits]
    user_payload = {
        "user_query": query,
        "parsed_filters": filters.to_dict(),
        "candidate_products": facts,
        "instruction": "请基于候选商品回答。最多推荐3款。不要提到不存在的优惠、库存或平台活动。",
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

