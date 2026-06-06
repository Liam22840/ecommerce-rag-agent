"""Grounded multi-product comparison over catalog facts and evidence snippets.

ComparisonService orchestrates the flow: resolve which products to compare (resolver),
decide what to compare (dimensions), judge each dimension (evidence), then narrate the
result. Each step degrades to a deterministic fallback when the LLM is unavailable.
"""

from __future__ import annotations

import re
from typing import Any

from server.catalog import ProductCatalog
from server.intent import SearchFilters
from server.schemas import ComparisonRow, ComparisonValue, ProductComparison
from server.textutil import dedupe, dedupe_int, json_object, normalize
from server.comparison.dimensions import (
    PRICE_FOCUS_TERMS,
    DimensionSpec,
    _dedupe_specs,
    _dimension_extraction_messages,
    _dynamic_specs,
    _price_is_priority,
    _specs_from_llm_payload,
)
from server.comparison.evidence import (
    _confidence,
    _evidence,
    _evidence_judge_messages,
    _evidence_value,
    _judge_confidence,
    _winner_from_scores,
)
from server.comparison.resolver import _asks_for_current_two, _best_ref_match, _name_score
from server.comparison.text import _source_texts


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


class ComparisonService:
    def __init__(self, catalog: ProductCatalog, llm: Any | None = None):
        self._catalog = catalog
        self._llm = llm

    def is_comparison_query(self, query: str, explicit_product_ids: list[str]) -> bool:
        if len(explicit_product_ids) >= 2:
            return True
        normalized = normalize(query)
        return any(hint in normalized for hint in COMPARE_HINTS)

    def build(
        self,
        query: str,
        filters: SearchFilters,
        explicit_product_ids: list[str],
        recent_product_ids: list[str],
    ) -> ProductComparison:
        product_ids, clarification = self._resolve_compared_products(
            query, filters, explicit_product_ids, recent_product_ids
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
        rows.extend(self._evidence_rows(products, focus_specs, query))

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
        product_ids = dedupe([
            product_id
            for product_id in explicit_product_ids
            if self._catalog.get(product_id) is not None
        ])
        if len(product_ids) >= 2:
            return product_ids[:3], None

        text_ids = re.findall(r"p_[a-z]+_\d+", query, flags=re.IGNORECASE)
        product_ids = dedupe(product_ids + [pid for pid in text_ids if self._catalog.get(pid)])
        if len(product_ids) >= 2:
            return product_ids[:3], None

        recent_ids = [pid for pid in recent_product_ids if self._catalog.get(pid)]
        ordinal_ids = self._resolve_ordinals(query, recent_ids)
        product_ids = dedupe(product_ids + ordinal_ids)
        if len(product_ids) >= 2:
            return product_ids[:3], None

        name_ids = self._resolve_names(query, recent_ids)
        product_ids = dedupe(product_ids + name_ids)
        if len(product_ids) >= 2:
            return product_ids[:3], None

        if _asks_for_current_two(query) and len(recent_ids) >= 2:
            return recent_ids[:2], None

        if product_ids:
            return [], "我只确认到一款商品。请再指定另一款商品，或点选两个商品后发起对比。"
        if recent_ids:
            return [], "我还不能确定你要对比哪两款。可以说“第一个和第二个”，或直接点选两个商品。"
        return [], "我还没有可对比的商品上下文。请先让助手推荐商品，或直接输入两款商品名。"

    def _resolve_compared_products(
        self,
        query: str,
        filters: SearchFilters,
        explicit_product_ids: list[str],
        recent_product_ids: list[str],
    ) -> tuple[list[str], str | None]:
        """Confidence-ordered: explicit (structured) ids first, then the LLM-extracted
        references mapped deterministically (precise), then the deterministic waterfall as
        the fallback. No extra LLM call (references come from the parser's existing call)."""
        valid_explicit = dedupe([
            pid for pid in explicit_product_ids if self._catalog.get(pid) is not None
        ])
        if len(valid_explicit) >= 2:
            return valid_explicit[:3], None
        # LLM references are more precise than the deterministic name matcher, so map them
        # before falling to the full deterministic waterfall.
        mapped = self._map_refs_to_products(filters.compare_refs, recent_product_ids)
        if len(mapped) >= 2:
            return mapped[:3], None
        return self._resolve_product_ids(query, explicit_product_ids, recent_product_ids)

    def _map_refs_to_products(self, refs: list[str], recent_product_ids: list[str]) -> list[str]:
        if not refs:
            return []
        recent = [product for product in (self._catalog.get(pid) for pid in recent_product_ids) if product]
        result: list[str] = []
        for ref in refs:
            product = _best_ref_match(ref, recent) or _best_ref_match(ref, self._catalog.products)
            if product and product["product_id"] not in result:
                result.append(product["product_id"])
        return result

    def _resolve_ordinals(self, query: str, recent_product_ids: list[str]) -> list[str]:
        if not recent_product_ids:
            return []
        normalized = normalize(query)
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
        return [recent_product_ids[index] for index in dedupe_int(indexes)]

    def _resolve_names(self, query: str, recent_product_ids: list[str]) -> list[str]:
        normalized_query = normalize(query)
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
        return dedupe([item[2] for item in candidates[:3]])

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
            payload = json_object(raw)
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

    def _evidence_rows(
        self,
        products: list[dict[str, Any]],
        focus_specs: list[DimensionSpec],
        query: str,
    ) -> list[ComparisonRow]:
        """LLM judges each evidence dimension; any dimension it can't judge falls back
        to the deterministic _evidence_row."""
        evidence_specs = [spec for spec in focus_specs if spec.evidence]
        judged = self._llm_judge(products, evidence_specs, query)
        return [judged.get(spec.label) or self._evidence_row(products, spec) for spec in evidence_specs]

    def _llm_judge(
        self,
        products: list[dict[str, Any]],
        evidence_specs: list[DimensionSpec],
        query: str,
    ) -> dict[str, ComparisonRow]:
        if not evidence_specs or not self._llm or not getattr(self._llm, "available", False):
            return {}
        try:
            raw = self._llm.complete(_evidence_judge_messages(query, products, evidence_specs))
            payload = json_object(raw)
        except Exception:  # noqa: BLE001 - evidence judging must degrade to the deterministic scorer.
            return {}
        if not isinstance(payload.get("judgments"), list):
            return {}
        return self._rows_from_judgments(payload, products, evidence_specs)

    def _rows_from_judgments(
        self,
        payload: dict[str, Any],
        products: list[dict[str, Any]],
        evidence_specs: list[DimensionSpec],
    ) -> dict[str, ComparisonRow]:
        product_ids = {product["product_id"] for product in products}
        grounding = {
            product["product_id"]: normalize(" ".join(text for _, text, _ in _source_texts(product)))
            for product in products
        }
        spec_labels = {spec.label for spec in evidence_specs}
        rows: dict[str, ComparisonRow] = {}
        for judgment in payload["judgments"]:
            if not isinstance(judgment, dict):
                continue
            label = str(judgment.get("dimension", "")).strip()
            if label not in spec_labels or label in rows:
                continue
            winner = judgment.get("winner_product_id")
            if winner not in product_ids:
                winner = None
            reasons = judgment.get("reasons")
            reasons = reasons if isinstance(reasons, dict) else {}
            quotes = judgment.get("evidence")
            quotes = quotes if isinstance(quotes, dict) else {}
            raw_conf = judgment.get("confidence")
            values = []
            for product in products:
                pid = product["product_id"]
                reason = str(reasons.get(pid, "")).strip()
                quote = str(quotes.get(pid, "")).strip()
                grounded = bool(quote) and normalize(quote) in grounding[pid]
                values.append(ComparisonValue(
                    product_id=pid,
                    value=reason or (quote if grounded else f"商品库未提供足够“{label}”证据"),
                    evidence=[quote] if grounded else [],
                    confidence=_judge_confidence(raw_conf, reason, grounded),
                ))
            if winner:
                verdict = f"按商品库证据，{self._title(winner, products)}在“{label}”维度更有支撑。"
            else:
                verdict = f"两款在“{label}”维度证据接近或不足，不能做绝对判断。"
            rows[label] = ComparisonRow(dimension=label, values=values, winner_product_id=winner, verdict=verdict)
        return rows

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
