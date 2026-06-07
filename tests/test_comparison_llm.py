"""Tests for the LLM-backed comparison evidence judge, its grounding, and fallback."""

from __future__ import annotations

import json

from server.catalog import ProductCatalog
from server.comparison import ComparisonService
from server.config import Settings
from server.intent import SearchFilters

DATASET_ROOT = Settings().dataset_root
P7, P18 = "p_digital_007", "p_digital_018"  # two real earbuds
REAL_QUOTE = "地铁的轨道轰鸣声直接几乎全没了"  # a real substring of p_digital_007's reviews
FAKE_QUOTE = "这是一段商品库里不存在的编造引用XYZ"


class FakeJudgeLLM:
    """Returns dimension JSON for the extraction prompt and judgment JSON for the judge prompt."""

    def __init__(self, judgment_json: str, dimension_json: str | None = None):
        self.judgment_json = judgment_json
        self.dimension_json = dimension_json or json.dumps(
            {"dimensions": [{"label": "降噪", "aliases": ["降噪"], "preference": "higher_is_better"}]},
            ensure_ascii=False,
        )
        self.calls: list = []

    @property
    def available(self) -> bool:
        return True

    def complete(self, messages):
        self.calls.append(messages)
        if "对比证据裁判" in messages[0]["content"]:
            return self.judgment_json
        return self.dimension_json


class _RaisingLLM:
    """Raises on either the dimension-extraction or the evidence-judge prompt (and returns valid
    JSON for the other), to exercise the degrade-to-deterministic paths on each LLM failure."""

    available = True

    def __init__(self, raise_on: str):
        self._raise_on = raise_on  # "dimension" | "judge"

    def complete(self, messages):
        is_judge = "对比证据裁判" in messages[0]["content"]
        if (self._raise_on == "judge") == is_judge:
            raise RuntimeError("llm down")
        if is_judge:
            return json.dumps({"judgments": []}, ensure_ascii=False)
        return json.dumps({"dimensions": [{"label": "降噪", "aliases": ["降噪"]}]}, ensure_ascii=False)


def _build_with_llm(llm):
    catalog = ProductCatalog.load(DATASET_ROOT)
    svc = ComparisonService(catalog, llm=llm)
    return svc.build(
        query="这两款降噪哪个更好",
        filters=SearchFilters(raw_query="这两款降噪哪个更好"),
        explicit_product_ids=[P7, P18],
        recent_product_ids=[],
    )


def test_dimension_extraction_exception_degrades_to_deterministic():
    comp = _build_with_llm(_RaisingLLM(raise_on="dimension"))
    assert [card.product_id for card in comp.products] == [P7, P18]  # still builds
    assert _row(comp, "价格与SKU") is not None


def test_evidence_judge_exception_degrades_to_deterministic():
    comp = _build_with_llm(_RaisingLLM(raise_on="judge"))
    row = _row(comp, "降噪")
    assert row is not None  # deterministic evidence row still produced


def test_judgments_skip_non_dict_and_unknown_dimension():
    judgment = json.dumps({"judgments": [
        "not-a-dict",                                       # skipped: not a dict
        {"dimension": "未知维度", "winner_product_id": P7},  # skipped: not an extracted dimension
        {"dimension": "降噪", "winner_product_id": P7,
         "reasons": {P7: "降噪强", P18: "一般"}, "evidence": {P7: REAL_QUOTE}, "confidence": "high"},
    ]}, ensure_ascii=False)
    comp = _build(judgment)
    row = _row(comp, "降噪")
    assert row is not None
    assert row.winner_product_id == P7
    assert _row(comp, "未知维度") is None


def _build(judgment_json: str):
    return _build_with_llm(FakeJudgeLLM(judgment_json))


def _row(comp, dimension):
    return next((r for r in comp.rows if r.dimension == dimension), None)


def test_judgment_sets_winner_and_grounds_real_quote():
    judgment = json.dumps({"judgments": [{
        "dimension": "降噪", "winner_product_id": P7,
        "reasons": {P7: "评价说降噪很强", P18: "有底噪"},
        "evidence": {P7: REAL_QUOTE, P18: FAKE_QUOTE},
        "confidence": "high",
    }]}, ensure_ascii=False)
    comp = _build(judgment)
    row = _row(comp, "降噪")
    assert row is not None
    assert row.winner_product_id == P7
    v7 = next(v for v in row.values if v.product_id == P7)
    v18 = next(v for v in row.values if v.product_id == P18)
    assert REAL_QUOTE in v7.evidence          # real quote kept
    assert v18.evidence == []                 # fabricated quote dropped (grounding)


def test_invalid_winner_is_nulled():
    judgment = json.dumps({"judgments": [{
        "dimension": "降噪", "winner_product_id": "p_not_real",
        "reasons": {P7: "x", P18: "y"}, "evidence": {}, "confidence": "medium",
    }]}, ensure_ascii=False)
    row = _row(_build(judgment), "降噪")
    assert row.winner_product_id is None


def test_ungrounded_winner_confidence_downgraded():
    judgment = json.dumps({"judgments": [{
        "dimension": "降噪", "winner_product_id": P7,
        "reasons": {P7: "", P18: ""},
        "evidence": {P7: FAKE_QUOTE},  # not in product text
        "confidence": "high",
    }]}, ensure_ascii=False)
    row = _row(_build(judgment), "降噪")
    v7 = next(v for v in row.values if v.product_id == P7)
    assert v7.evidence == []
    assert v7.confidence in {"low", "none"}   # no verifiable quote -> not high


def test_malformed_judgment_falls_back_to_deterministic():
    # Judge returns non-JSON -> _llm_judge returns {} -> deterministic _evidence_row used.
    comp = _build("not json at all")
    row = _row(comp, "降噪")
    assert row is not None                     # row still produced via fallback
    assert row.winner_product_id in {P7, P18, None}


def test_empty_judgments_falls_back():
    comp = _build(json.dumps({"judgments": []}, ensure_ascii=False))
    assert _row(comp, "降噪") is not None       # fell back, still has the row


def _build_with_refs(refs, recent):
    catalog = ProductCatalog.load(DATASET_ROOT)
    svc = ComparisonService(catalog, llm=None)  # refs come pre-set in filters (parser's job)
    return svc.build(
        query="理肤泉和薇诺娜哪个更适合敏感肌",
        filters=SearchFilters(raw_query="理肤泉和薇诺娜哪个更适合敏感肌",
                              intent_type="comparison", compare_refs=refs),
        explicit_product_ids=[],
        recent_product_ids=recent,
    )


def test_compare_refs_resolve_against_recent_products():
    # One 理肤泉 (p_beauty_012) and one 薇诺娜 (p_beauty_007) on screen -> unambiguous.
    comp = _build_with_refs(["理肤泉", "薇诺娜"], recent=["p_beauty_012", "p_beauty_007"])
    ids = [card.product_id for card in comp.products]
    assert "p_beauty_012" in ids and "p_beauty_007" in ids
    assert comp.clarification is None


def test_compare_refs_ambiguous_cold_falls_back_to_clarification():
    # Cold (nothing shown) and each brand has 2 products -> ambiguous -> clarify, don't guess.
    comp = _build_with_refs(["理肤泉", "薇诺娜"], recent=[])
    assert comp.clarification is not None
    assert comp.products == []


def test_compare_refs_unknown_falls_back_to_clarification():
    comp = _build_with_refs(["不存在的牌子甲", "不存在的牌子乙"], recent=[])
    assert comp.clarification is not None


def test_no_llm_uses_deterministic_engine():
    # With no LLM the comparison still builds via the deterministic engine. (Dimension
    # extraction without the LLM is the teammate's separate logic, so we only assert the
    # structural rows are produced, not a specific evidence dimension.)
    catalog = ProductCatalog.load(DATASET_ROOT)
    svc = ComparisonService(catalog, llm=None)
    comp = svc.build(
        query="这两款降噪哪个更好",
        filters=SearchFilters(raw_query="这两款降噪哪个更好"),
        explicit_product_ids=[P7, P18],
        recent_product_ids=[],
    )
    assert [card.product_id for card in comp.products] == [P7, P18]
    assert _row(comp, "价格与SKU") is not None
