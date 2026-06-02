"""Prompt construction for grounded shopping responses."""

from __future__ import annotations

import json

from server.catalog import CatalogHit, ProductCatalog
from server.intent import SearchFilters


SYSTEM_PROMPT = """你是一个电商智能导购助手。
你只能依据提供的商品事实回答，不能编造商品、价格、库存、优惠券、功效或参数。
如果候选商品不足以满足用户条件，要明确说明没有找到完全匹配项，并给出可继续筛选的问题。
回答要简洁、可执行。当前端会展示商品卡时，不要在文本中逐条复述商品名、价格、SKU或推荐理由。
有商品卡时，文本最多2句话，只做结果概览、排序说明或下一步筛选建议；商品细节交给商品卡呈现。
商品名、品牌、类目、规格、SKU 和价格都必须使用商品事实中的结构化字段。
价格必须优先照抄 price_label；需要解释多规格时照抄 price_summary。
禁止把 title 里的规格和 lowest_price 混在一起表达；如果 title 中的规格不同于 lowest_price_sku，只能说“xx元起（最低价SKU）”，并列出 SKU 价格明细。
"""


def build_messages(
    query: str,
    filters: SearchFilters,
    hits: list[CatalogHit],
    catalog: ProductCatalog,
) -> list[dict[str, str]]:
    facts = [catalog.product_facts(hit.product, filters) for hit in hits]
    user_payload = {
        "user_query": query,
        "parsed_filters": filters.to_dict(),
        "candidate_products": facts,
        "instruction": (
            "请基于候选商品回答。最多推荐3款。不要提到不存在的优惠、库存或平台活动。"
            "所有商品事实和价格必须来自 candidate_products，不允许自行推断或改写 SKU 价格。"
            "如果会返回商品卡，回答文本不要编号列商品，不要重复商品卡里的名称、价格、SKU明细或推荐理由。"
        ),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
