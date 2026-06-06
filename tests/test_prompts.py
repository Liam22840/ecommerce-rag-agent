from server.prompts import chitchat_messages


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
