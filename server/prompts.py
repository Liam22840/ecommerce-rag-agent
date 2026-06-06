"""All LLM prompts and message builders for the assistant.

This is the single home for prompt text. Domain types are imported only under
TYPE_CHECKING so this module pulls in nothing from the business modules at runtime,
which lets intent.py / comparison.py / assistant.py import from here without cycles.
The message builders operate on the objects passed to them (duck-typed at runtime).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.catalog import CatalogHit, ProductCatalog
    from server.intent import SearchFilters
    from server.schemas import ProductComparison


# --- Recommendation answer -----------------------------------------------------

SYSTEM_PROMPT = """你是一个电商智能导购助手。
你只能依据提供的商品事实回答，不能编造商品、价格、库存、优惠券、功效或参数。
如果候选商品不足以满足用户条件，要明确说明没有找到完全匹配项，并给出可继续筛选的问题。
回答要简洁、可执行，优先说明为什么推荐，以及和用户条件的对应关系。
商品名、品牌、类目、规格、SKU 和价格都必须使用商品事实中的结构化字段。
价格必须优先照抄 price_label；需要解释多规格时照抄 price_summary。
禁止把 title 里的规格和 lowest_price 混在一起表达；如果 title 中的规格不同于 lowest_price_sku，只能说“xx元起（最低价SKU）”，并列出 SKU 价格明细。
当 result_status 不是 ok 时，按字段含义如实说明，不要把上一轮已展示的商品当作新推荐再列一遍；no_cheaper 时点出 context.cheapest_shown 作为当前最低价。这些情况都顺势建议换品类或调整条件。
当 context.unmet_terms 非空时，商品库里没有明确标注这些属性的商品：要如实说明（以下只是最接近的几款），并且不能声称某个候选具备它的商品事实里没有体现的属性。
用纯文本回答，不要使用任何 Markdown 标记（不要出现 **、*、#、`、列表符号等）；需要分条时直接用“1. 2. 3.”和换行。
"""


def build_messages(
    query: str,
    filters: SearchFilters,
    hits: list[CatalogHit],
    catalog: ProductCatalog,
    result_status: str = "ok",
    context: dict | None = None,
) -> list[dict[str, str]]:
    facts = [catalog.product_facts(hit.product, filters) for hit in hits]
    user_payload = {
        "user_query": query,
        "parsed_filters": filters.to_dict(),
        "candidate_products": facts,
        "result_status": result_status,
        "instruction": (
            "请基于候选商品回答。最多推荐3款。不要提到不存在的优惠、库存或平台活动。"
            "所有商品事实和价格必须来自 candidate_products，不允许自行推断或改写 SKU 价格。"
        ),
    }
    if context:
        user_payload["context"] = context
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


# --- Intent parsing ------------------------------------------------------------

INTENT_SYSTEM_PROMPT = (
    "你是电商导购的查询意图解析器。只输出 JSON，不写任何解释或多余文字。"
    "任务：把用户的中文购物查询解析成结构化筛选条件。\n"
    "规则：\n"
    "1. category、sub_category、brand 必须从给定的可选值列表里原样选取；找不到对应项时填 null，禁止自造。\n"
    "2. 把口语词按词义映射到 categories/sub_categories/brands 里给出的官方词（如 口红→唇釉）；拿不准就就近匹配，不要新造。\n"
    "3. 价格区间：「200到500」→ min_price=200, max_price=500；「不超过1万/一万」→ max_price=10000；中文数字要换算成阿拉伯数字；没有约束填 null。"
    "约数价格要展开成一个合理区间而不是精确值（如「三百左右」→ min_price=255, max_price=345），只有明确说「正好/刚好」时才用精确值。\n"
    "4. 否定：「不含X」「不要X牌」→ 写进 excluded_terms / excluded_brands（品牌必须是列表里的官方品牌词，否则不写）。\n"
    "5. intent_type：纯打招呼/闲聊/与购物无关→chitchat；在比较/二选一具体商品→comparison；其余购物需求→product_search。"
    "若用户想要的商品明显不属于本目录的任何 category/sub_category（既无法归类，也不是承接上文的追问或“随便看看”这类泛需求），也按 chitchat 处理，礼貌说明暂不提供该品类，不要硬塞不相关的商品。\n"
    "6. sort_by：用户要便宜/低价优先→price_asc；要评分高/口碑好→rating_desc；要贵/高端优先→price_desc；否则→relevance。\n"
    "7. required_terms 放明确卖点词（如 敏感肌、保湿、防水）；requested_specs 放容量规格（如 50g、256GB、500ml）。\n"
    "8. compare_refs：仅当 intent_type=comparison 时，填用户点名要对比的商品，用其原话里最具体的指代（带型号/系列，如「理肤泉特安」而非只写「理肤泉」）；否则填 []。\n"
    "9. 列表字段没内容返回 []，标量没内容返回 null。\n"
    "10. 给了 recent_turns（最近几轮的解析结果，以及当时展示过的商品和价格）时，如果本轮是承接上文的追问"
    "（改写、追加条件、对刚才结果提相对要求、或指回之前提过的商品），就结合 recent_turns 把本轮理解成具体筛选条件："
    "继承仍然适用的 category/sub_category/required_terms，并把相对要求落到已有字段上（例如想更便宜，就把 max_price 设到"
    " recent_turns 里已展示的最低价以下）。只能用 recent_turns 里真实出现过的信息，不要编造；若本轮是新品类或无关话题，就忽略 recent_turns。\n"
    "11. rewritten_query：本轮依赖上文时，结合 recent_turns 改写成一句可独立检索的完整中文查询；本轮本身已完整，或属于 chitchat/comparison 时填 null。\n"
    "12. exclude_seen：用户想看和刚才展示过的不一样的商品（看过的别再给）时设为 true，否则 false。\n"
    "13. session_products 是本轮会话里展示过的商品（含 id，按展示先后排列，最近展示的排在最前）。"
    "用户指回之前看过的某个/某些商品时，定位到对应商品并把其 id 原样填进 recall_product_ids；否则填 []。\n"
    "14. intent_type=comparison 且用户要对比的是 session_products 里展示过的商品时，定位到对应商品，把要对比的 id（通常两个）"
    "原样填进 compare_product_ids。用序号指代（如「第一个」）时按 session_products 顺序数（第一个=列表最前=最近展示的第一款），"
    "定位到对应 id；定位不到具体 id 的点名商品仍走 compare_refs；否则 compare_product_ids 填 []。\n"
    '只输出如下 JSON：{"intent_type":"product_search|comparison|chitchat",'
    '"category":string|null,"sub_category":string|null,"brand":string|null,'
    '"min_price":number|null,"max_price":number|null,'
    '"sort_by":"relevance|price_asc|price_desc|rating_desc","prefer_low_price":boolean,'
    '"required_terms":[string],"requested_specs":[string],'
    '"excluded_brands":[string],"excluded_terms":[string],"compare_refs":[string],'
    '"rewritten_query":string|null,"exclude_seen":boolean,"recall_product_ids":[string],'
    '"compare_product_ids":[string]}'
)


def intent_messages(
    query: str,
    categories: set[str],
    sub_categories: set[str],
    brands: set[str],
    history: list[dict] | None = None,
    session_products: list[dict] | None = None,
) -> list[dict[str, str]]:
    user_payload = {
        "query": query,
        "categories": sorted(categories),
        "sub_categories": sorted(sub_categories),
        "brands": sorted(brands),
    }
    if history:
        user_payload["recent_turns"] = history
    if session_products:
        user_payload["session_products"] = session_products
    return [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


# --- Comparison: dimension extraction + evidence judging -----------------------
# (the message builders live in comparison.py next to their product serialization;
#  the prompt text lives here.)

DIMENSION_EXTRACTION_SYSTEM = (
    "你是电商导购的对比维度抽取器。只输出 JSON，不写解释。"
    "任务：从用户问题中抽取用户真正关心的对比维度，并根据给定商品证据生成可检索同义词。"
    "不要判断赢家，不要编造商品事实，不要输出价格/SKU 事实。"
    'JSON 格式：{"dimensions":[{"label":"维度名","aliases":["检索词"],"preference":"higher_is_better|lower_is_better"}]}'
)

EVIDENCE_JUDGE_SYSTEM = (
    "你是电商导购的对比证据裁判。只输出 JSON，不写任何解释。"
    "任务：对每个给定维度，只依据所给的商品证据（标题、描述、官方问答、用户评价），判断哪款商品在该维度更好。\n"
    "规则：\n"
    "1. 只能用提供的证据，禁止编造。要读懂评价语气：比如“噪音几乎没了/全没了”是好评，“有底噪/不好/一般”是差评。\n"
    "2. winner_product_id 必须是给定的某个 product_id；证据接近或不足就填 null。\n"
    "3. evidence 里每个商品引用一句你判断所依据的原文（尽量逐字照抄），没有合适证据就留空字符串。\n"
    "4. 遵守每个维度的 preference：lower_is_better 表示越低/越少越好。\n"
    "5. 不要判断价格或 SKU，这部分系统会单独处理。\n"
    "6. reasons 每个商品用一句话说明理由；confidence 取 high|medium|low|none。\n"
    '只输出如下 JSON：{"judgments":[{"dimension":"维度名","winner_product_id":"pid 或 null",'
    '"reasons":{"pid":"一句话理由"},"evidence":{"pid":"原文引用"},"confidence":"high|medium|low|none"}]}'
)


# --- Chit-chat -----------------------------------------------------------------

CHITCHAT_REPLY = (
    "你好呀～我是你的购物助手。告诉我你想买什么就行，比如品类、预算或使用场景"
    "（例如“两三百的敏感肌面霜”“适合通勤的降噪耳机”），我来帮你挑选和对比。"
)

CHITCHAT_SYSTEM = (
    "你是电商导购助手。用户这句话要么与具体购物需求无关（打招呼、道谢、闲聊、问你是谁或你能做什么等），"
    "要么想买的是本店并不经营的商品。请用一两句友好、简短的中文回应："
    "属于闲聊就自然把话题引导回购物（可以问他想买什么品类、预算或使用场景）；"
    "如果用户想买的品类不在本店经营范围内，就礼貌说明本店暂不提供该类商品、只售下列品类，"
    "并欢迎他在这些品类里选购，不要假装能卖、也不要追问这件商品的细节（如预算、型号等）。"
    "不要回答与购物无关的专业问题（医疗、法律、金融、时政等），礼貌说明你只负责帮挑选商品。"
    "纯文本中文，不要使用任何 Markdown 标记。"
)


def chitchat_messages(query: str, categories: set[str] | None = None) -> list[dict[str, str]]:
    system = CHITCHAT_SYSTEM
    if categories:
        system += "\n本店经营品类：" + "、".join(sorted(categories)) + "。"
    return [{"role": "system", "content": system}, {"role": "user", "content": query}]


# --- Comparison narration ------------------------------------------------------

COMPARISON_NARRATION_SYSTEM = (
    "你是电商导购助手。下面给你的是系统已经算好的商品对比结果，请用自然、简洁的中文把结论讲给用户，帮他做决定。\n"
    "要求：\n"
    "1. 不要改变“总体更推荐”的结论，也不要推翻任何逐维度结论。\n"
    "2. 只能用给到的信息，不要编造商品库里没有的参数、功效或评价。\n"
    "3. 价格直接照抄给的“价格”字段。\n"
    "4. 先给结论（更推荐哪个、为什么），再点出主要差异；2 到 4 句话即可。\n"
    "5. 纯文本中文，不要使用任何 Markdown 标记。"
)


def comparison_narration_messages(comparison: ProductComparison) -> list[dict[str, str]]:
    id_to_title = {product.product_id: product.title for product in comparison.products}
    rows = [
        {
            "维度": row.dimension,
            "本维度更优": id_to_title.get(row.winner_product_id, "不明显"),
            "各商品": {id_to_title.get(v.product_id, v.product_id): v.value for v in row.values},
        }
        for row in comparison.rows
    ]
    payload = {
        "商品": [{"名称": product.title, "价格": product.price_label} for product in comparison.products],
        "对比维度": comparison.focus,
        "逐维度结论": rows,
        "总体更推荐": id_to_title.get(comparison.winner_product_id, "无明显赢家"),
        "系统结论": comparison.recommendation,
    }
    return [
        {"role": "system", "content": COMPARISON_NARRATION_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
