import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from server.app import create_app
from server.assistant import ShoppingAssistant
from server.catalog import ProductCatalog
from server.config import Settings
from server.retrieval import ProductRetriever


DATASET_ROOT = Path(__file__).parent.parent / "ecommerce_agent_dataset"


class FakeDimensionLLM:
    def __init__(self):
        self.calls: list[list[dict[str, str]]] = []

    @property
    def available(self) -> bool:
        return True

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        query = ""
        for message in messages:
            if message["role"] != "user":
                continue
            try:
                query = json.loads(message["content"]).get("query", "")
            except json.JSONDecodeError:
                query = message["content"]
        if "穿一天" in query or "不累" in query:
            return json.dumps({
                "dimensions": [
                    {
                        "label": "舒适度",
                        "aliases": ["缓震", "脚感", "酸疼", "软弹", "通勤", "不累"],
                        "preference": "higher_is_better",
                    }
                ]
            }, ensure_ascii=False)
        if "安静" in query or "戴久" in query:
            return json.dumps({
                "dimensions": [
                    {
                        "label": "降噪安静",
                        "aliases": ["降噪", "噪音", "安静", "地铁", "环境噪音"],
                        "preference": "higher_is_better",
                    },
                    {
                        "label": "佩戴舒适",
                        "aliases": ["佩戴", "贴耳", "胀耳", "耳塞", "小耳", "滑"],
                        "preference": "higher_is_better",
                    },
                ]
            }, ensure_ascii=False)
        if "水润" in query or "拔干" in query:
            return json.dumps({
                "dimensions": [
                    {
                        "label": "水润不拔干",
                        "aliases": ["保湿", "补水", "滋润", "锁水", "干燥", "拔干", "紧绷"],
                        "preference": "higher_is_better",
                    }
                ]
            }, ensure_ascii=False)
        return json.dumps({"dimensions": []}, ensure_ascii=False)


def _client(llm: FakeDimensionLLM) -> TestClient:
    settings = Settings(
        dataset_root=DATASET_ROOT,
        chat_api_key="fake",
        embedding_api_key=None,
        enable_vector_search=False,
        enable_llm=True,
    )
    catalog = ProductCatalog.load(DATASET_ROOT)
    retriever = ProductRetriever(catalog, settings)
    assistant = ShoppingAssistant(catalog=catalog, retriever=retriever, llm=llm)  # type: ignore[arg-type]
    return TestClient(create_app(settings=settings, assistant=assistant))


def test_comparison_dimension_extraction_evaluation_black_box():
    llm = FakeDimensionLLM()
    client = _client(llm)
    cases: list[dict[str, Any]] = [
        {
            "message": "这两双跑鞋穿一天哪个不累？",
            "product_ids": ["p_clothes_007", "p_clothes_009"],
            "expected_focus": ["舒适度"],
            "expected_winner": "p_clothes_009",
        },
        {
            "message": "这两个耳机通勤地铁里哪个更安静，戴久了也舒服？",
            "product_ids": ["p_digital_007", "p_digital_018"],
            "expected_focus": ["降噪安静", "佩戴舒适"],
            "expected_winner": "p_digital_007",
        },
        {
            "message": "这两款面霜哪个上脸更水润不拔干？",
            "product_ids": ["p_beauty_007", "p_beauty_012"],
            "expected_focus": ["水润不拔干"],
            "expected_winner": "p_beauty_012",
        },
        {
            "message": "这两款饮料哪个糖分更低、气泡口感更好？",
            "product_ids": ["p_food_004", "p_food_015"],
            "expected_focus": ["糖分", "气泡口感"],
            "expected_winner": "p_food_004",
        },
    ]

    score = 0
    total = 0
    diagnostics = []
    for case in cases:
        resp = client.post(
            "/api/chat",
            json={
                "message": case["message"],
                "compare_product_ids": case["product_ids"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        comparison = body["comparison"]
        assert comparison is not None
        focus = comparison["focus"]
        rows = [row["dimension"] for row in comparison["rows"]]
        winner = comparison["winner_product_id"]

        focus_ok = all(expected in focus for expected in case["expected_focus"])
        rows_ok = all(expected in rows for expected in case["expected_focus"])
        winner_ok = winner == case["expected_winner"]
        score += int(focus_ok) + int(rows_ok) + int(winner_ok)
        total += 3
        diagnostics.append({
            "message": case["message"],
            "focus": focus,
            "rows": rows,
            "winner": winner,
            "focus_ok": focus_ok,
            "rows_ok": rows_ok,
            "winner_ok": winner_ok,
        })

    assert llm.calls, "comparison flow should call the dimension extraction LLM"
    assert score >= 10, f"comparison evaluation score {score}/{total}: {diagnostics}"


def test_comparison_keeps_price_and_sku_facts_black_box():
    llm = FakeDimensionLLM()
    client = _client(llm)

    resp = client.post(
        "/api/chat",
        json={
            "message": "这两款面霜哪个更便宜？",
            "compare_product_ids": ["p_beauty_007", "p_beauty_012"],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["comparison"]["winner_product_id"] == "p_beauty_007"
    assert "15g 体验装 89元；50g 标准装 268元" in body["answer"]
    assert "40ml 滋润型 260元" in body["answer"]
