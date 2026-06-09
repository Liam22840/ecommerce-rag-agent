from server.prompts import chitchat_messages, photo_answer_messages, vision_intent_messages


def test_vision_intent_messages_embed_image_and_vocab():
    messages = vision_intent_messages(
        "便宜点的", {"服饰运动"}, {"短袖T恤"}, {"耐克"},
        "data:image/jpeg;base64,QUJD",
    )
    assert messages[0]["role"] == "system"
    # The user message carries both the text payload and the image part.
    parts = messages[1]["content"]
    kinds = {part["type"] for part in parts}
    assert kinds == {"text", "image_url"}
    image_part = next(p for p in parts if p["type"] == "image_url")
    assert image_part["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
    text_part = next(p for p in parts if p["type"] == "text")
    assert "短袖T恤" in text_part["text"]      # catalog vocab is given to the VLM
    assert "便宜点的" in text_part["text"]      # accompanying text is included


def test_vision_intent_messages_include_session_context_only_when_present():
    with_ctx = vision_intent_messages(
        "", {"服饰运动"}, {"短袖T恤"}, {"耐克"}, "data:image/jpeg;base64,QUJD",
        session_products=[{"id": "p1", "title": "T", "price": 99}],
    )
    text_part = next(p for p in with_ctx[1]["content"] if p["type"] == "text")
    assert "session_products" in text_part["text"]

    without_ctx = vision_intent_messages("", {"服饰运动"}, {"短袖T恤"}, {"耐克"}, "data:image/jpeg;base64,QUJD")
    text_part = next(p for p in without_ctx[1]["content"] if p["type"] == "text")
    assert "session_products" not in text_part["text"]


def test_photo_answer_messages_carry_confidence_and_facts():
    class _Cat:
        def product_facts(self, product, filters=None, available=None):
            return {"product_id": product["product_id"], "price_label": "99元"}

    from server.intent import SearchFilters
    from server.catalog import CatalogHit
    hit = CatalogHit(product={"product_id": "p1"}, score=1.0)
    filters = SearchFilters(vision_description="黑色短袖")
    messages = photo_answer_messages("找同款", filters, [hit], _Cat(), low_confidence=True)
    assert messages[0]["role"] == "system"
    assert '"match_confidence": "low"' in messages[1]["content"]
    assert "黑色短袖" in messages[1]["content"]
    assert "p1" in messages[1]["content"]


def test_chitchat_messages_includes_store_scope():
    # The chitchat responder must know the store's precise scope (sub-categories) so it can
    # decline out-of-catalog requests instead of role-playing that it sells them.
    scope = "数码电子（智能手机、笔记本电脑）；美妆护肤（面霜）"
    messages = chitchat_messages("推荐一辆车", scope)
    assert scope in messages[0]["content"]


def test_chitchat_messages_without_categories_still_valid():
    messages = chitchat_messages("你好")
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "你好"}


from server.prompts import opener_continuation, intent_messages, commerce_intent_messages


def test_opener_continuation_is_empty_for_chitchat_but_set_for_a_search():
    # chitchat's reply greets for itself, so no route tail is prepended; a real search gets one.
    assert opener_continuation("chitchat") == ""
    assert opener_continuation("product_search", "面霜") != ""


def test_intent_messages_include_cart_and_recent_turns_when_present():
    content = intent_messages(
        "再加一件", {"美妆护肤"}, {"面霜"}, {"雅诗兰黛"},
        history=[{"query": "推荐面霜"}], cart=[{"title": "面霜", "price": 10.0, "quantity": 1}],
    )[1]["content"]
    assert "recent_turns" in content and "cart" in content


def test_commerce_intent_messages_include_the_comparison_winner_when_present():
    content = commerce_intent_messages("买胜出的那个", [], [], comparison_winner_id="p_beauty_007")[1]["content"]
    assert "comparison_winner_id" in content and "p_beauty_007" in content
