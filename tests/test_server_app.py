from pathlib import Path

from fastapi.testclient import TestClient

from server.app import create_app
from server.config import Settings


DATASET_ROOT = Path(__file__).parent.parent / "ecommerce_agent_dataset"


def _client() -> TestClient:
    settings = Settings(
        dataset_root=DATASET_ROOT,
        chat_api_key=None,
        embedding_api_key=None,
        enable_vector_search=False,
        enable_llm=False,
    )
    return TestClient(create_app(settings=settings))


def test_health_endpoint():
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_endpoint_rejects_blank_message():
    resp = _client().post("/api/chat", json={"message": "   "})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "message cannot be empty"


def test_stream_endpoint_rejects_blank_message():
    resp = _client().post("/api/chat/stream", json={"message": "   "})
    assert resp.status_code == 400


def test_product_detail_returns_404_for_unknown_id():
    resp = _client().get("/api/products/does_not_exist")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "product not found"


def test_chat_carries_search_context_across_turns():
    client = _client()
    session_id = "carry-1"

    first = client.post("/api/chat", json={"session_id": session_id, "message": "三百以内的面霜"})
    assert first.status_code == 200
    assert first.json()["intent"]["sub_category"] == "面霜"

    # "便宜点的" names no category; the deterministic carry-over keeps the face-cream context
    # (this runs with enable_llm=False, so it exercises the degraded-mode backstop).
    second = client.post("/api/chat", json={"session_id": session_id, "message": "便宜点的"})
    assert second.status_code == 200
    body = second.json()
    assert body["intent"]["sub_category"] == "面霜"
    assert body["intent"]["prefer_low_price"] is True
    assert body["products"]
    assert all(product["sub_category"] == "面霜" for product in body["products"])


def test_cheaper_refinement_says_nothing_cheaper_instead_of_relisting():
    client = _client()
    session_id = "cheaper-1"

    client.post("/api/chat", json={"session_id": session_id, "message": "三百以内的面霜"})
    # "便宜一点的" carries 面霜 but the shown creams are already the cheapest -> honest answer,
    # not a silent re-list of the same products as if they were new.
    second = client.post("/api/chat", json={"session_id": session_id, "message": "便宜一点的"}).json()
    assert second["intent"]["sub_category"] == "面霜"
    assert "没有更便宜" in second["answer"]
    assert all(product["sub_category"] == "面霜" for product in second["products"])


def test_chat_endpoint_returns_grounded_product_cards():
    client = _client()

    resp = client.post("/api/chat", json={"message": "推荐一款适合油皮的洗面奶"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["degraded"] is True
    assert body["retrieval_source"] == "lexical"
    assert body["products"]
    assert body["products"][0]["product_id"] == "p_beauty_011"
    assert "商品库" in body["answer"]


def test_chat_endpoint_handles_no_exact_match_without_hallucinating():
    client = _client()

    resp = client.post("/api/chat", json={"message": "200 元以下的蓝牙耳机有哪些？"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["products"] == []
    assert "没有在商品库中找到完全匹配" in body["answer"]


def test_product_detail_endpoint():
    client = _client()

    resp = client.get("/api/products/p_beauty_011")

    assert resp.status_code == 200
    assert resp.json()["product_id"] == "p_beauty_011"


def test_product_image_asset_endpoint():
    client = _client()

    resp = client.get("/assets/products/1_美妆护肤/images/p_beauty_011_live.jpg")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content


def test_stream_endpoint_uses_sse_events():
    client = _client()

    with client.stream("POST", "/api/chat/stream", json={"message": "推荐一款适合油皮的洗面奶"}) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert "event: token" in body
    assert "event: products" in body
    assert "event: done" in body
    assert '"base_price"' in body
    assert '"reason"' in body
    assert '"items"' in body


def test_stream_endpoint_accepts_ios_payload_and_legacy_path():
    client = _client()

    payload = {
        "conversation_id": "ios-session-1",
        "message": "推荐一款适合油皮的洗面奶",
        "attachments": [],
        "client_context": {"cart_items": []},
    }
    with client.stream("POST", "/api/v1/chat/stream", json=payload) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert '"session_id": "ios-session-1"' in body
    assert "event: token" in body
    assert "event: products" in body


def test_chat_endpoint_defaults_to_three_product_cards():
    client = _client()

    resp = client.post("/api/chat", json={"message": "推荐一款适合油皮的洗面奶"})

    assert resp.status_code == 200
    assert len(resp.json()["products"]) <= 3


def test_chat_endpoint_orders_selected_cards_by_price_when_requested():
    client = _client()

    resp = client.post(
        "/api/chat",
        json={"message": "推荐一个适合敏感肌的保湿护肤品，cheaper is better"},
    )

    assert resp.status_code == 200
    body = resp.json()
    prices = [product["price"] for product in body["products"]]
    assert prices == sorted(prices)
    assert [product["product_id"] for product in body["products"]] == [
        "p_beauty_007",
        "p_beauty_022",
        "p_beauty_012",
    ]
    assert "15g 体验装 89元；50g 标准装 268元" in body["answer"]
    assert "p_beauty_002" not in [product["product_id"] for product in body["products"]]


def test_chat_endpoint_uses_requested_sku_price_for_specs():
    client = _client()

    resp = client.post(
        "/api/chat",
        json={"message": "推荐50g适合敏感肌的保湿霜，cheaper is better"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert [product["product_id"] for product in body["products"]] == ["p_beauty_007"]
    assert body["products"][0]["price"] == 268.0
    assert body["products"][0]["price_label"] == "268元（50g 标准装）"
    assert "价格：268元（50g 标准装）" in body["answer"]
    assert "15g 体验装 89元；50g 标准装 268元" in body["answer"]


def test_chat_endpoint_compares_explicit_product_ids_with_structured_rows():
    client = _client()

    resp = client.post(
        "/api/chat",
        json={
            "message": "这两款面霜哪个更保湿？",
            "compare_product_ids": ["p_beauty_007", "p_beauty_012"],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert [product["product_id"] for product in body["products"]] == ["p_beauty_007", "p_beauty_012"]
    assert body["comparison"] is not None
    assert body["comparison"]["focus"] == ["保湿"]
    assert body["comparison"]["rows"]
    assert any(row["dimension"] == "保湿" for row in body["comparison"]["rows"])
    assert "15g 体验装 89元；50g 标准装 268元" in body["answer"]
    assert "证据不足处不会做绝对判断" in body["answer"]


def test_chat_endpoint_compares_recent_products_by_ordinal_reference():
    client = _client()
    session_id = "compare-session-1"

    first = client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "推荐一个适合敏感肌的保湿护肤品，cheaper is better",
        },
    )
    assert first.status_code == 200
    assert [product["product_id"] for product in first.json()["products"]][:2] == [
        "p_beauty_007",
        "p_beauty_022",
    ]

    second = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "第一个和第二个哪个更保湿？"},
    )

    assert second.status_code == 200
    body = second.json()
    assert [product["product_id"] for product in body["products"]] == ["p_beauty_007", "p_beauty_022"]
    assert body["comparison"]["winner_product_id"] in {"p_beauty_007", "p_beauty_022", None}
    assert "第一个" not in body["answer"]
    assert "保湿" in body["answer"]


def test_chat_endpoint_price_comparison_recommends_the_compared_sku_not_title_spec():
    client = _client()
    session_id = "compare-session-price-sku"

    first = client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "推荐一个适合敏感肌的保湿护肤品，cheaper is better",
        },
    )
    assert first.status_code == 200
    assert [product["product_id"] for product in first.json()["products"]][:2] == [
        "p_beauty_007",
        "p_beauty_022",
    ]

    second = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "第一个和第二个哪个更便宜？"},
    )

    assert second.status_code == 200
    body = second.json()
    assert body["comparison"]["winner_product_id"] == "p_beauty_007"
    assert "薇诺娜 15g 体验装（89元）" in body["comparison"]["recommendation"]
    assert "薇诺娜 15g 体验装（89元）" in body["comparison"]["summary"]
    assert "15g 体验装 89元；50g 标准装 268元" in body["answer"]


def test_chat_endpoint_preserves_original_recommendation_context_after_comparison():
    client = _client()
    session_id = "compare-session-preserve-context"

    first = client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "推荐一个适合敏感肌的保湿护肤品，cheaper is better",
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "第一个和第二个哪个更保湿？"},
    )
    assert second.status_code == 200

    third = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "第一个和第三个哪个更适合敏感肌？"},
    )

    assert third.status_code == 200
    assert [product["product_id"] for product in third.json()["products"]] == [
        "p_beauty_007",
        "p_beauty_012",
    ]


def test_chat_endpoint_asks_for_products_when_comparison_context_is_missing():
    client = _client()

    resp = client.post("/api/chat", json={"message": "第一个和第二个哪个更保湿？"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["products"] == []
    assert body["comparison"]["clarification"] is not None
    assert "不能确定" in body["answer"] or "还没有可对比" in body["answer"]


def test_chat_endpoint_asks_for_clarification_on_brand_level_comparison():
    client = _client()

    resp = client.post("/api/chat", json={"message": "薇诺娜和理肤泉哪个更保湿？"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["products"] == []
    assert body["comparison"]["clarification"] is not None
    assert "直接输入两款商品名" in body["answer"]


def test_chat_endpoint_asks_for_clarification_on_contextual_brand_level_comparison():
    client = _client()
    session_id = "compare-session-brand-context"
    client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "推荐一个适合敏感肌的保湿护肤品，cheaper is better",
        },
    )

    resp = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "薇诺娜和理肤泉哪个更保湿？"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["products"] == []
    assert body["comparison"]["clarification"] is not None
    assert "不能确定" in body["answer"]


def test_chat_endpoint_compares_digital_products_with_dynamic_attribute():
    client = _client()

    resp = client.post(
        "/api/chat",
        json={
            "message": "这两款耳机哪个音质更好？",
            "compare_product_ids": ["p_digital_007", "p_digital_018"],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert [product["product_id"] for product in body["products"]] == ["p_digital_007", "p_digital_018"]
    assert "音质" in body["comparison"]["focus"]
    assert any(row["dimension"] == "音质" for row in body["comparison"]["rows"])
    assert "商品库" in body["answer"]


def test_chat_endpoint_compares_sports_products_without_beauty_specific_logic():
    client = _client()

    resp = client.post(
        "/api/chat",
        json={
            "message": "这两双跑鞋哪个缓震和抓地更好？",
            "compare_product_ids": ["p_clothes_007", "p_clothes_009"],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert [product["category"] for product in body["products"]] == ["服饰运动", "服饰运动"]
    assert "缓震" in body["comparison"]["focus"]
    assert "抓地" in body["comparison"]["focus"]
    assert any(row["dimension"] == "缓震" for row in body["comparison"]["rows"])
    assert any(row["dimension"] == "抓地" for row in body["comparison"]["rows"])
    assert any(row["dimension"] == "价格与SKU" for row in body["comparison"]["rows"])


def test_chat_endpoint_compares_food_products_with_dynamic_attributes():
    client = _client()

    resp = client.post(
        "/api/chat",
        json={
            "message": "这两款饮料哪个糖分更低、气泡口感更好？",
            "compare_product_ids": ["p_food_004", "p_food_015"],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert [product["category"] for product in body["products"]] == ["食品饮料", "食品饮料"]
    assert "糖分" in body["comparison"]["focus"]
    assert any("气泡" in focus for focus in body["comparison"]["focus"])
    assert any(row["dimension"] == "糖分" for row in body["comparison"]["rows"])
    assert any("气泡" in row["dimension"] for row in body["comparison"]["rows"])


def test_stream_endpoint_emits_structured_comparison_event():
    client = _client()
    session_id = "compare-session-stream"
    client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "推荐一个适合敏感肌的保湿护肤品，cheaper is better",
        },
    )

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"session_id": session_id, "message": "第一个和第二个哪个更保湿？"},
    ) as resp:
        body = "".join(resp.iter_text())

    assert resp.status_code == 200
    assert "event: comparison" in body
    assert '"rows"' in body
    assert '"winner_product_id"' in body
    assert '"price_summary"' in body
    assert "event: done" in body


# --- comparison ordinal resolution against the unified recency memory ----------

def test_comparison_前两个_resolves_to_last_results_first_two():
    client = _client()
    session_id = "cmp-front2"

    first = client.post("/api/chat", json={"session_id": session_id, "message": "推荐几款跑步鞋"}).json()
    shoes = [product["product_id"] for product in first["products"]]
    assert len(shoes) >= 2

    second = client.post("/api/chat", json={"session_id": session_id, "message": "前两个哪个更好"}).json()
    compared = [product["product_id"] for product in second["products"]]
    assert compared == shoes[:2]


def test_comparison_ordinal_refers_to_the_most_recent_search():
    client = _client()
    session_id = "cmp-recency"

    # Two searches in different categories; the ordinal must point at the *latest* result.
    client.post("/api/chat", json={"session_id": session_id, "message": "推荐面霜"})
    shoes = client.post("/api/chat", json={"session_id": session_id, "message": "推荐几款跑步鞋"}).json()
    shoe_ids = [product["product_id"] for product in shoes["products"]]
    assert len(shoe_ids) >= 2

    compared = client.post(
        "/api/chat", json={"session_id": session_id, "message": "第一个和第二个哪个更好"}
    ).json()
    compared_ids = [product["product_id"] for product in compared["products"]]
    assert compared_ids == shoe_ids[:2]  # the shoes (latest), not the earlier creams
    assert all(product["category"] == "服饰运动" for product in compared["products"])


def test_comparison_without_enough_context_asks_for_clarification():
    client = _client()
    # A comparison intent with nothing shown yet -> clarification, no products.
    body = client.post(
        "/api/chat", json={"session_id": "cmp-empty", "message": "第一个和第二个哪个更好"}
    ).json()
    assert body["products"] == []
    assert body["comparison"] is not None
