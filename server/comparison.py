"""Grounded multi-product comparison over catalog facts and evidence snippets."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Any

from server.catalog import ProductCatalog
from server.intent import SearchFilters
from server.schemas import ComparisonRow, ComparisonValue, ProductComparison


COMPARE_HINTS = [
    "对比",
    "比较",
    "哪个更",
    "哪款更",
    "哪一个更",
    "更适合",
    "选哪个",
    "买哪个",
    "二选一",
    "a和b",
    "a 和 b",
    "第一",
    "第二",
    "这两款",
    "这两个",
]

ORDINALS = {
    "第一个": 0,
    "第一款": 0,
    "第1个": 0,
    "第1款": 0,
    "1号": 0,
    "第一个商品": 0,
    "第二个": 1,
    "第二款": 1,
    "第2个": 1,
    "第2款": 1,
    "2号": 1,
    "第二个商品": 1,
    "第三个": 2,
    "第三款": 2,
    "第3个": 2,
    "第3款": 2,
    "3号": 2,
}

GENERIC_NEGATIVE_CUES = (
    "不",
    "没",
    "无",
    "差",
    "弱",
    "低",
    "少",
    "慢",
    "短",
    "贵",
    "一般",
    "不足",
    "不够",
    "不太",
    "不是",
    "没有",
    "失望",
    "问题",
    "缺点",
)
GENERIC_POSITIVE_CUES = (
    "好",
    "强",
    "高",
    "足",
    "快",
    "长",
    "低",
    "轻",
    "稳",
    "值",
    "适合",
    "不错",
    "明显",
    "满意",
    "推荐",
    "优秀",
    "友好",
)
PRICE_FOCUS_TERMS = ("价格", "预算", "便宜", "划算", "性价比", "省钱", "更值")
LOWER_IS_BETTER_CUES = ("低", "少", "短", "轻", "便宜", "省", "0", "零", "无", "不含", "没有")
HIGHER_IS_BETTER_CUES = ("高", "多", "长", "强", "足", "厚", "贵")


@dataclass(frozen=True)
class DimensionSpec:
    label: str
    terms: tuple[str, ...]
    preference: str = "higher_is_better"
    evidence: bool = True


class ComparisonService:
    def __init__(self, catalog: ProductCatalog, llm: Any | None = None):
        self._catalog = catalog
        self._llm = llm

    def is_comparison_query(self, query: str, explicit_product_ids: list[str]) -> bool:
        if len(explicit_product_ids) >= 2:
            return True
        normalized = _normalize(query)
        return any(hint in normalized for hint in COMPARE_HINTS)

    def build(
        self,
        query: str,
        filters: SearchFilters,
        explicit_product_ids: list[str],
        recent_product_ids: list[str],
    ) -> ProductComparison:
        product_ids, clarification = self._resolve_product_ids(
            query=query,
            explicit_product_ids=explicit_product_ids,
            recent_product_ids=recent_product_ids,
        )

        if clarification:
            focus_specs = self._focus_specs(query, filters, products=[])
            return ProductComparison(
                products=[],
                focus=[spec.label for spec in focus_specs],
                rows=[],
                winner_product_id=None,
                recommendation=clarification,
                summary=clarification,
                clarification=clarification,
            )

        products = [self._catalog.require(product_id) for product_id in product_ids]
        focus_specs = self._focus_specs(query, filters, products=products)
        cards = [self._catalog.product_card(product, filters=filters) for product in products]
        rows = [
            self._basic_row(products),
            self._price_row(products, filters),
            self._sku_row(products),
        ]
        for spec in focus_specs:
            if spec.evidence:
                rows.append(self._evidence_row(products, spec))

        winner_product_id, recommendation = self._recommend(products, rows, query, filters)
        summary = self._summary(products, focus_specs, winner_product_id, recommendation, filters)
        return ProductComparison(
            products=cards,
            focus=[spec.label for spec in focus_specs],
            rows=rows,
            winner_product_id=winner_product_id,
            recommendation=recommendation,
            summary=summary,
        )

    def _resolve_product_ids(
        self,
        query: str,
        explicit_product_ids: list[str],
        recent_product_ids: list[str],
    ) -> tuple[list[str], str | None]:
        product_ids = _dedupe([
            product_id
            for product_id in explicit_product_ids
            if self._catalog.get(product_id) is not None
        ])
        if len(product_ids) >= 2:
            return product_ids[:3], None

        text_ids = re.findall(r"p_[a-z]+_\d+", query, flags=re.IGNORECASE)
        product_ids = _dedupe(product_ids + [pid for pid in text_ids if self._catalog.get(pid)])
        if len(product_ids) >= 2:
            return product_ids[:3], None

        recent_ids = [pid for pid in recent_product_ids if self._catalog.get(pid)]
        ordinal_ids = self._resolve_ordinals(query, recent_ids)
        product_ids = _dedupe(product_ids + ordinal_ids)
        if len(product_ids) >= 2:
            return product_ids[:3], None

        name_ids = self._resolve_names(query, recent_ids)
        product_ids = _dedupe(product_ids + name_ids)
        if len(product_ids) >= 2:
            return product_ids[:3], None

        if _asks_for_current_two(query) and len(recent_ids) >= 2:
            return recent_ids[:2], None

        if product_ids:
            return [], "我只确认到一款商品。请再指定另一款商品，或点选两个商品后发起对比。"
        if recent_ids:
            return [], "我还不能确定你要对比哪两款。可以说“第一个和第二个”，或直接点选两个商品。"
        return [], "我还没有可对比的商品上下文。请先让助手推荐商品，或直接输入两款商品名。"

    def _resolve_ordinals(self, query: str, recent_product_ids: list[str]) -> list[str]:
        if not recent_product_ids:
            return []
        normalized = _normalize(query)
        indexes = []
        for phrase, index in ORDINALS.items():
            if phrase in normalized and index < len(recent_product_ids):
                indexes.append(index)
        for first, second in re.findall(r"(?<!\d)([1-3])\s*(?:和|跟|与|、|,|，|vs|VS)\s*([1-3])(?!\d)", query):
            for raw in (first, second):
                index = int(raw) - 1
                if index < len(recent_product_ids):
                    indexes.append(index)
        if "前两个" in normalized and len(recent_product_ids) >= 2:
            indexes.extend([0, 1])
        if ("a和b" in normalized or "a跟b" in normalized or "a与b" in normalized) and len(recent_product_ids) >= 2:
            indexes.extend([0, 1])
        return [recent_product_ids[index] for index in _dedupe_int(indexes)]

    def _resolve_names(self, query: str, recent_product_ids: list[str]) -> list[str]:
        normalized_query = _normalize(query)
        candidates = []
        pool = [self._catalog.require(pid) for pid in recent_product_ids] if recent_product_ids else self._catalog.products
        for product in pool:
            score = _name_score(normalized_query, product)
            if score > 0:
                candidates.append((score, self._catalog.lowest_price(product), product["product_id"]))
        candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        if candidates and candidates[0][0] <= 8:
            # Brand-only matches are ambiguous when the catalog contains multiple products per brand.
            return []
        return _dedupe([item[2] for item in candidates[:3]])

    def _focus_specs(
        self,
        query: str,
        filters: SearchFilters,
        products: list[dict[str, Any]],
    ) -> list[DimensionSpec]:
        llm_specs = self._llm_specs(query, products)
        specs = llm_specs if llm_specs else _dynamic_specs(query, products)
        if _price_is_priority(query, filters):
            specs.append(DimensionSpec(label="价格", terms=PRICE_FOCUS_TERMS, preference="lower_is_better", evidence=False))
        return _dedupe_specs(specs)[:4]

    def _llm_specs(self, query: str, products: list[dict[str, Any]]) -> list[DimensionSpec]:
        if not products or not self._llm or not getattr(self._llm, "available", False):
            return []
        try:
            raw = self._llm.complete(_dimension_extraction_messages(query, products))
            payload = _json_object(raw)
        except Exception:  # noqa: BLE001 - LLM dimension extraction must degrade to deterministic fallback.
            return []
        return _specs_from_llm_payload(payload, query, products)

    def _basic_row(self, products: list[dict[str, Any]]) -> ComparisonRow:
        values = [
            ComparisonValue(
                product_id=product["product_id"],
                value=f"{product['brand']} · {product['sub_category']}",
                evidence=[product["title"]],
                confidence="high",
            )
            for product in products
        ]
        return ComparisonRow(dimension="基础定位", values=values, verdict="先确认两款商品的品牌和类目，避免跨类目误比。")

    def _price_row(self, products: list[dict[str, Any]], filters: SearchFilters) -> ComparisonRow:
        values = [
            ComparisonValue(
                product_id=product["product_id"],
                value=self._catalog.price_label(product, filters),
                evidence=[self._catalog.price_summary(product)],
                confidence="high",
            )
            for product in products
        ]
        prices = {product["product_id"]: self._catalog.lowest_price(product) for product in products}
        winner = min(prices, key=prices.get)
        verdict = f"{self._price_subject(winner, products, filters)}最低价更低；多规格商品请以 SKU 明细为准。"
        return ComparisonRow(dimension="价格与SKU", values=values, winner_product_id=winner, verdict=verdict)

    def _sku_row(self, products: list[dict[str, Any]]) -> ComparisonRow:
        values = [
            ComparisonValue(
                product_id=product["product_id"],
                value=self._catalog.price_summary(product),
                evidence=[],
                confidence="high",
            )
            for product in products
        ]
        return ComparisonRow(dimension="规格明细", values=values, verdict="规格和价格按商品库 SKU 原文展示，不把低价体验装误写成正装价格。")

    def _evidence_row(self, products: list[dict[str, Any]], spec: DimensionSpec) -> ComparisonRow:
        values = []
        scores: dict[str, float] = {}
        for product in products:
            score, snippets = _evidence(product, spec.terms, spec.preference)
            scores[product["product_id"]] = score
            values.append(ComparisonValue(
                product_id=product["product_id"],
                value=_evidence_value(spec.label, snippets),
                evidence=snippets[:3],
                confidence=_confidence(score, snippets),
            ))
        winner = _winner_from_scores(scores)
        if winner:
            verdict = f"按商品库证据，{self._title(winner, products)}在“{spec.label}”维度更有支撑。"
        else:
            verdict = f"两款在“{spec.label}”维度证据接近或不足，不能做绝对判断。"
        return ComparisonRow(dimension=spec.label, values=values, winner_product_id=winner, verdict=verdict)

    def _recommend(
        self,
        products: list[dict[str, Any]],
        rows: list[ComparisonRow],
        query: str,
        filters: SearchFilters,
    ) -> tuple[str | None, str]:
        scores = {product["product_id"]: 0.0 for product in products}
        reason_dimensions: dict[str, list[str]] = {product["product_id"]: [] for product in products}
        for row in rows:
            if not row.winner_product_id:
                continue
            weight = 1.0
            if row.dimension in {"基础定位", "规格明细"}:
                weight = 0.0
            elif row.dimension == "价格与SKU":
                weight = 1.5 if _price_is_priority(query, filters) else 0.0
            else:
                weight = 2.0
            scores[row.winner_product_id] += weight
            if weight > 0:
                reason_dimensions[row.winner_product_id].append(row.dimension)
        winner = max(scores, key=scores.get)
        if scores[winner] <= 0:
            return None, "这几款商品各有侧重，商品库证据不足以给出单一绝对赢家。建议根据你更看重的维度继续筛选。"
        reasons = reason_dimensions[winner]
        reason_text = "、".join(reasons[:3]) if reasons else "综合证据"
        subject = self._price_subject(winner, products, filters) if "价格与SKU" in reasons else self._title(winner, products)
        suffix = "；如果你要对比指定规格，请直接说明容量/规格，系统会按对应 SKU 比价。" if "价格与SKU" in reasons else "；如果你的优先级不同，可以继续指定预算、肤质或使用场景。"
        return winner, f"更推荐{subject}：它在{reason_text}上更符合当前问题{suffix}"

    def _summary(
        self,
        products: list[dict[str, Any]],
        specs: list[DimensionSpec],
        winner_product_id: str | None,
        recommendation: str,
        filters: SearchFilters,
    ) -> str:
        price_priority = any(spec.label == "价格" for spec in specs)
        names = " vs ".join(
            self._price_subject(product["product_id"], products, filters) if price_priority else product["title"]
            for product in products
        )
        focus = "、".join(spec.label for spec in specs) or "商品库证据"
        if winner_product_id:
            return f"已按{focus}对比：{names}。结论：{recommendation}"
        return f"已按{focus}对比：{names}。结论：{recommendation}"

    @staticmethod
    def _title(product_id: str, products: list[dict[str, Any]]) -> str:
        for product in products:
            if product["product_id"] == product_id:
                return f"「{product['title']}」"
        return product_id

    def _price_subject(self, product_id: str, products: list[dict[str, Any]], filters: SearchFilters) -> str:
        for product in products:
            if product["product_id"] != product_id:
                continue
            sku = self._catalog.selected_price_sku(product, filters) or self._catalog.lowest_price_sku(product)
            if sku:
                return f"「{product['brand']} {sku['label']}（{sku['price']:g}元）」"
            return f"「{product['title']}（{self._catalog.lowest_price(product):g}元）」"
        return product_id


def _evidence(
    product: dict[str, Any],
    terms: tuple[str, ...],
    preference: str,
) -> tuple[float, list[str]]:
    score = 0.0
    snippets = []
    for source, text, weight in _source_texts(product):
        if not text:
            continue
        source_score = 0.0
        for chunk in _chunks(text):
            matched_terms = [term for term in terms if term and term.lower() in chunk.lower()]
            if not matched_terms:
                continue
            polarity = _generic_polarity(chunk, preference)
            source_score += weight * polarity * min(2.0, 0.75 + 0.25 * len(matched_terms))
        if source_score != 0:
            score += source_score
            snippet = _best_snippet(text, terms)
            if snippet:
                snippets.append(f"{source}: {snippet}")
    return score, _dedupe(snippets)


def _source_texts(product: dict[str, Any]) -> list[tuple[str, str, float]]:
    knowledge = product.get("rag_knowledge", {})
    sources = [
        ("商品标题", product.get("title", ""), 0.8),
        ("SKU", _sku_text(product), 1.0),
        ("商品描述", knowledge.get("marketing_description", ""), 1.4),
    ]
    for item in knowledge.get("official_faq", []):
        sources.append(("官方问答", f"{item.get('question', '')} {item.get('answer', '')}", 1.2))
    for item in knowledge.get("user_reviews", []):
        rating = item.get("rating")
        weight = 1.0 if not isinstance(rating, int | float) else max(0.6, min(1.3, rating / 4))
        sources.append(("用户评价", item.get("content", ""), weight))
    return sources


def _sku_text(product: dict[str, Any]) -> str:
    parts = []
    for sku in product.get("skus", []):
        properties = sku.get("properties", {})
        if isinstance(properties, dict):
            parts.extend(f"{key}{value}" for key, value in properties.items())
        if sku.get("price") is not None:
            parts.append(f"{sku['price']}元")
    return " ".join(str(part) for part in parts if part)


def _best_snippet(text: str, terms: tuple[str, ...], max_len: int = 86) -> str:
    chunks = _chunks(text)
    for term in terms:
        for chunk in chunks:
            if term and term.lower() in chunk.lower():
                return _trim(chunk, max_len)
    return _trim(chunks[0], max_len) if chunks else ""


def _chunks(text: str) -> list[str]:
    return [chunk.strip() for chunk in re.split(r"[。！？!?；;\n]", text) if chunk.strip()]


def _generic_polarity(text: str, preference: str = "higher_is_better") -> float:
    normalized = _normalize(text)
    if preference == "lower_is_better":
        lower_hits = sum(1 for cue in LOWER_IS_BETTER_CUES if cue and cue in normalized)
        higher_hits = sum(1 for cue in HIGHER_IS_BETTER_CUES if cue and cue in normalized)
        if lower_hits > higher_hits:
            return 1.25
        if higher_hits > lower_hits:
            return -1.0
    negative_hits = sum(1 for cue in GENERIC_NEGATIVE_CUES if cue and cue in normalized)
    positive_hits = sum(1 for cue in GENERIC_POSITIVE_CUES if cue and cue in normalized)
    if negative_hits > positive_hits:
        return -1.0
    if positive_hits > 0:
        return 1.25
    return 0.8


def _evidence_value(label: str, snippets: list[str]) -> str:
    if snippets:
        return "；".join(_strip_source(snippet) for snippet in snippets[:2])
    return f"商品库未提供足够“{label}”证据"


def _strip_source(snippet: str) -> str:
    return snippet.split(": ", 1)[-1]


def _confidence(score: float, snippets: list[str]) -> str:
    if not snippets:
        return "none"
    if abs(score) >= 5:
        return "high"
    if abs(score) >= 2:
        return "medium"
    return "low"


def _winner_from_scores(scores: dict[str, float]) -> str | None:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] <= 0:
        return None
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 2:
        return None
    return ranked[0][0]


def _name_score(normalized_query: str, product: dict[str, Any]) -> float:
    title = _normalize(product.get("title", ""))
    brand = _normalize(product.get("brand", ""))
    sub_category = _normalize(product.get("sub_category", ""))
    score = 0.0
    if title and title in normalized_query:
        score += 20
    if brand and brand in normalized_query:
        score += 8
    if sub_category and sub_category in normalized_query:
        score += 4
    for token in _title_tokens(title):
        if token in normalized_query:
            score += 2
    return score


def _title_tokens(normalized_title: str) -> list[str]:
    tokens = []
    for size in (6, 4, 3):
        for idx in range(0, max(0, len(normalized_title) - size + 1), size):
            token = normalized_title[idx: idx + size]
            if len(token) == size:
                tokens.append(token)
    return tokens[:8]


def _dimension_extraction_messages(query: str, products: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是电商导购的对比维度抽取器。只输出 JSON，不写解释。"
                "任务：从用户问题中抽取用户真正关心的对比维度，并根据给定商品证据生成可检索同义词。"
                "不要判断赢家，不要编造商品事实，不要输出价格/SKU 事实。"
                "JSON 格式：{\"dimensions\":[{\"label\":\"维度名\",\"aliases\":[\"检索词\"],\"preference\":\"higher_is_better|lower_is_better\"}]}"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query": query,
                    "products": [_product_evidence_for_llm(product) for product in products],
                    "rules": [
                        "label 用用户能理解的短词，例如 舒适度、降噪安静、佩戴舒适、水润不拔干。",
                        "aliases 必须来自或贴近商品标题、描述、官方问答、用户评价中的词，便于后端检索证据。",
                        "如果用户关注更低、更少、更轻、更便宜，preference 设为 lower_is_better；其他通常为 higher_is_better。",
                        "价格、容量、库存、SKU 不需要作为 evidence 维度，后端会用结构化字段处理。",
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]


def _product_evidence_for_llm(product: dict[str, Any]) -> dict[str, Any]:
    knowledge = product.get("rag_knowledge", {})
    faq = knowledge.get("official_faq", [])[:3]
    reviews = knowledge.get("user_reviews", [])[:4]
    return {
        "product_id": product.get("product_id"),
        "title": product.get("title"),
        "brand": product.get("brand"),
        "category": product.get("category"),
        "sub_category": product.get("sub_category"),
        "sku_summary": _sku_text(product),
        "marketing_description": _trim(str(knowledge.get("marketing_description", "")), 700),
        "official_faq": [
            {
                "question": _trim(str(item.get("question", "")), 140),
                "answer": _trim(str(item.get("answer", "")), 260),
            }
            for item in faq
        ],
        "user_reviews": [
            {
                "rating": item.get("rating"),
                "content": _trim(str(item.get("content", "")), 260),
            }
            for item in reviews
        ],
    }


def _json_object(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


def _specs_from_llm_payload(
    payload: dict[str, Any],
    query: str,
    products: list[dict[str, Any]],
) -> list[DimensionSpec]:
    raw_dimensions = payload.get("dimensions", [])
    if not isinstance(raw_dimensions, list):
        return []
    corpus = _product_corpus(products)
    specs = []
    seen = set()
    for raw in raw_dimensions:
        if not isinstance(raw, dict):
            continue
        label = _clean_attribute_label(str(raw.get("label", "")))
        label = _strip_product_context_words(label, products)
        normalized_label = _normalize(label)
        if not normalized_label or normalized_label in seen:
            continue
        if _is_noise_attribute(normalized_label, products):
            continue
        aliases = raw.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        terms = _llm_terms(label, aliases)
        if not terms:
            continue
        if _is_price_dimension(label, terms):
            continue
        matched_terms = tuple(term for term in terms if _normalize(term) in corpus)
        if not matched_terms:
            continue
        preference = str(raw.get("preference", "")).strip()
        if preference not in {"higher_is_better", "lower_is_better"}:
            preference = _infer_preference(query, label)
        seen.add(normalized_label)
        specs.append(DimensionSpec(label=label, terms=matched_terms, preference=preference))
    return _dedupe_specs(specs)[:4]


def _llm_terms(label: str, aliases: list[Any]) -> tuple[str, ...]:
    terms = [label]
    terms.extend(str(alias).strip() for alias in aliases if str(alias).strip())
    for term in list(terms):
        terms.extend(_attribute_terms(term))
    return tuple(_dedupe([term for term in terms if len(_normalize(term)) >= 2])[:16])


def _is_price_dimension(label: str, terms: tuple[str, ...]) -> bool:
    normalized = {_normalize(label), *(_normalize(term) for term in terms)}
    return any(_normalize(term) in normalized for term in PRICE_FOCUS_TERMS)


def _infer_preference(query: str, label: str) -> str:
    normalized = _normalize(query)
    normalized_label = _normalize(label)
    if any(term in normalized for term in ["更低", "更少", "更短", "更轻", "更便宜", "省钱", "预算"]):
        if not normalized_label or normalized_label in normalized:
            return "lower_is_better"
    return "higher_is_better"


def _dynamic_specs(
    query: str,
    products: list[dict[str, Any]],
) -> list[DimensionSpec]:
    if not products:
        return []
    normalized = _normalize(query)
    corpus = _product_corpus(products)
    candidates = _explicit_attribute_candidates(query)
    if not candidates:
        candidates.extend(_corpus_backed_query_ngrams(normalized, corpus, products))
    specs = []
    seen = set()
    for candidate in candidates:
        label = _clean_attribute_label(candidate)
        label = _strip_product_context_words(label, products)
        normalized_label = _normalize(label)
        if not normalized_label or normalized_label in seen:
            continue
        if _is_noise_attribute(normalized_label, products):
            continue
        terms = _attribute_terms(label)
        if not any(_normalize(term) in corpus for term in terms):
            continue
        seen.add(normalized_label)
        specs.append(DimensionSpec(
            label=label,
            terms=terms,
            preference=_infer_preference(query, label),
        ))
    return _dedupe_specs(specs)[:3]


def _product_corpus(products: list[dict[str, Any]]) -> str:
    parts = []
    for product in products:
        parts.extend([product.get("title", ""), product.get("brand", ""), product.get("category", ""), product.get("sub_category", "")])
        parts.extend(text for _, text, _ in _source_texts(product))
    return _normalize(" ".join(str(part) for part in parts if part))


def _explicit_attribute_candidates(query: str) -> list[str]:
    candidates = []
    clauses = [clause for clause in re.split(r"[。！？!?；;，,]", query) if clause.strip()]
    before_suffix = r"(?:更好|更强|更高|更低|更长|更短|更久|更轻|更重|更快|更慢|更足|更稳|更靠谱|更值|更划算|更便宜|更贵|好|强|高|低|长|短|久|轻|重|快|慢|足|稳)"
    after_prefix = r"(?:更适合|更推荐|更偏向|更满足|更符合|更有利于|更)"
    for clause in clauses:
        for match in re.finditer(rf"(?:哪个|哪款|哪一个|哪种|哪双|哪件|哪瓶|哪台|哪部)?([^，。,.；;？?]{{1,20}}?){before_suffix}", clause, flags=re.IGNORECASE):
            candidates.extend(_split_attribute_candidate(match.group(1)))
        for match in re.finditer(rf"(?:哪个|哪款|哪一个|哪种|哪双|哪件|哪瓶|哪台|哪部)[^，。,.；;？?]{{0,8}}?{after_prefix}([^，。,.；;？?]{{1,18}})", clause, flags=re.IGNORECASE):
            candidates.extend(_split_attribute_candidate(match.group(1)))
        for match in re.finditer(r"(?:看重|关注|侧重|优先|比较|对比|按|从)([^，。,.；;？?]{1,20})", clause, flags=re.IGNORECASE):
            candidates.extend(_split_attribute_candidate(match.group(1)))
    return candidates


def _split_attribute_candidate(value: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?:和|跟|与|、|，|,|以及|还有|并且|或者|或|/)", value)
        if part.strip()
    ]


def _corpus_backed_query_ngrams(normalized_query: str, corpus: str, products: list[dict[str, Any]]) -> list[str]:
    candidates = []
    for size in range(8, 1, -1):
        for idx in range(0, max(0, len(normalized_query) - size + 1)):
            token = normalized_query[idx: idx + size]
            if token in corpus and not _is_noise_attribute(token, products):
                candidates.append(token)
    return candidates


def _clean_attribute_label(value: str) -> str:
    label = value.strip()
    label = re.sub(r"^.*(?:哪个|哪款|哪一个|哪种|哪双|哪件|哪瓶|哪台|哪部)", "", label)
    label = re.sub(r"(?:这两款|这两个|这两种|这两双|这两件|这两瓶|这两台|这两部|两个|两款|两种|两双|两件|两瓶|两台|两部|哪个|哪款|哪一个|哪种|哪双|哪件|哪瓶|哪台|哪部|第[一二三123]个|第[一二三123]款|商品|产品)", "", label)
    label = re.sub(r"(?:更好|更强|更高|更低|更长|更短|更久|更轻|更重|更快|更慢|更足|更稳|更适合|更推荐|更偏向|更靠谱|更值|更划算|更便宜|更贵|好|强|高|低|长|短|久|轻|重|快|慢|足|稳)$", "", label)
    label = re.sub(r"^[的得]+|[的得]+$", "", label)
    label = label.strip("款个只双件瓶盒袋包台部双")
    return label.strip()


def _strip_product_context_words(label: str, products: list[dict[str, Any]]) -> str:
    normalized = _normalize(label)
    context_words = set()
    for product in products:
        for key in ("brand", "category", "sub_category"):
            word = _normalize(str(product.get(key, "")))
            if len(word) < 2:
                continue
            context_words.add(word)
            for size in range(2, min(4, len(word)) + 1):
                context_words.add(word[:size])
                context_words.add(word[-size:])
    for word in sorted(context_words, key=len, reverse=True):
        if word == normalized:
            return ""
        if normalized.startswith(word):
            normalized = normalized[len(word):]
        if normalized.endswith(word):
            normalized = normalized[: -len(word)]
    return normalized


def _attribute_terms(label: str) -> tuple[str, ...]:
    terms = [label]
    compact = _normalize(label)
    for part in re.split(r"(?:和|跟|与|、|，|,|以及|还有|并且|或者|或|/)", label):
        if part.strip():
            terms.append(part.strip())
    if len(compact) > 2:
        for size in range(min(4, len(compact)), 1, -1):
            for idx in range(0, len(compact) - size + 1):
                terms.append(compact[idx: idx + size])
    return tuple(_dedupe([term for term in terms if len(_normalize(term)) >= 2])[:8])


def _is_noise_attribute(normalized_label: str, products: list[dict[str, Any]]) -> bool:
    if len(normalized_label) < 2:
        return True
    if normalized_label.isdigit():
        return True
    if normalized_label[0] in {"款", "个", "只", "双", "件", "瓶", "盒", "袋", "包", "台", "部"}:
        return True
    noise = {
        "哪个",
        "哪款",
        "哪一个",
        "更好",
        "更强",
        "更高",
        "更低",
        "更足",
        "更适合",
        "更靠谱",
        "这两款",
        "这两个",
        "商品",
        "产品",
        "对比",
        "比较",
        "推荐",
        "适合",
        "更推荐",
        "更偏向",
        "更满足",
        "更符合",
        "买哪个",
        "选哪个",
        "第一个",
        "第二个",
        "第三个",
    }
    if normalized_label in noise:
        return True
    for product in products:
        product_noise = [
            product.get("brand", ""),
            product.get("category", ""),
            product.get("sub_category", ""),
        ]
        if normalized_label in {_normalize(value) for value in product_noise if value}:
            return True
    return False


def _dedupe_specs(specs: list[DimensionSpec]) -> list[DimensionSpec]:
    seen = set()
    result = []
    for spec in specs:
        if spec.label not in seen:
            seen.add(spec.label)
            result.append(spec)
    return result


def _asks_for_current_two(query: str) -> bool:
    normalized = _normalize(query)
    return any(term in normalized for term in ["这两款", "这两个", "两款哪个", "两个哪个", "前两个"])


def _price_is_priority(query: str, filters: SearchFilters) -> bool:
    normalized = _normalize(query)
    return filters.prefer_low_price or any(
        term in normalized
        for term in ["性价比", "划算", "便宜", "价格", "预算", "省钱", "更值"]
    )


def _normalize(value: str) -> str:
    return re.sub(r"[\s·,，。；;:：/\\()（）「」『』【】\[\]_-]+", "", value.lower())


def _trim(value: str, max_len: int) -> str:
    value = value.strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "…"


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _dedupe_int(items: list[int]) -> list[int]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
