from server.prompts import chitchat_messages


def test_chitchat_messages_includes_store_categories():
    # The chitchat responder must know the store's scope so it can politely decline
    # out-of-catalog product requests instead of role-playing that it sells them.
    messages = chitchat_messages("推荐一辆车", {"美妆护肤", "数码电子"})
    system = messages[0]["content"]
    assert "美妆护肤" in system and "数码电子" in system


def test_chitchat_messages_without_categories_still_valid():
    messages = chitchat_messages("你好")
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "你好"}
