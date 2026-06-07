"""Tests for the Ark chat client: request shaping, error handling, SSE parsing."""

from __future__ import annotations

import pytest

from server.llm import ArkChatClient, ModelUnavailable


def _client(api_key: str | None = "secret") -> ArkChatClient:
    # Trailing slash on base_url is intentional: the client must strip it.
    return ArkChatClient(api_key=api_key, base_url="https://ark.example/api/v3/", model="m1")


class _FakeStreamResponse:
    """Minimal stand-in for the streaming requests.Response context manager."""

    def __init__(self, status_code: int, lines: list[bytes], text: str = ""):
        self.status_code = status_code
        self._lines = lines
        self.text = text

    def __enter__(self) -> "_FakeStreamResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def iter_lines(self, decode_unicode: bool = False):
        return iter(self._lines)


def test_available_reflects_presence_of_api_key():
    assert _client("k").available is True
    assert _client(None).available is False


# --- complete() ----------------------------------------------------------------

def test_complete_without_key_raises_model_unavailable():
    with pytest.raises(ModelUnavailable, match="ARK_CHAT_API_KEY"):
        _client(None).complete([{"role": "user", "content": "hi"}])


def test_complete_returns_stripped_content_and_shapes_request(mocker):
    resp = mocker.Mock(status_code=200)
    resp.json.return_value = {"choices": [{"message": {"content": "  你好世界  "}}]}
    post = mocker.patch("server.llm.requests.post", return_value=resp)

    out = _client("secret").complete([{"role": "user", "content": "hi"}])

    assert out == "你好世界"
    _, kwargs = post.call_args
    assert post.call_args[0][0] == "https://ark.example/api/v3/chat/completions"
    assert kwargs["json"]["model"] == "m1"
    assert kwargs["json"]["temperature"] <= 0.5  # low temperature for deterministic answers
    assert kwargs["json"]["messages"] == [{"role": "user", "content": "hi"}]
    assert "stream" not in kwargs["json"]
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    assert kwargs["timeout"] == 60.0
    assert "thinking" not in kwargs["json"]  # not sent unless explicitly disabled


def test_disable_thinking_adds_the_thinking_field(mocker):
    resp = mocker.Mock(status_code=200)
    resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    post = mocker.patch("server.llm.requests.post", return_value=resp)

    client = ArkChatClient(
        api_key="k", base_url="https://ark.example/api/v3", model="m1", disable_thinking=True
    )
    client.complete([{"role": "user", "content": "hi"}])

    _, kwargs = post.call_args
    assert kwargs["json"]["thinking"] == {"type": "disabled"}


def test_complete_raises_on_http_error_with_status_and_body(mocker):
    resp = mocker.Mock(status_code=500, text="upstream exploded" * 50)
    mocker.patch("server.llm.requests.post", return_value=resp)

    with pytest.raises(ModelUnavailable, match="500"):
        _client("k").complete([{"role": "user", "content": "hi"}])


# --- stream() ------------------------------------------------------------------

def test_stream_without_key_raises_model_unavailable():
    with pytest.raises(ModelUnavailable, match="ARK_CHAT_API_KEY"):
        list(_client(None).stream([{"role": "user", "content": "hi"}]))


def test_stream_yields_content_deltas_and_stops_on_done(mocker):
    lines = [
        b"",  # blank line skipped
        b'data: {"choices":[{"delta":{"content":"\xe4\xbd\xa0"}}]}',  # 你
        b'data: {"choices":[{"delta":{"content":"\xe5\xa5\xbd"}}]}',  # 好
        b'data: {"choices":[{"delta":{}}]}',  # no content key, skipped
        b"data: not-json",  # undecodable JSON, skipped
        b"data: [DONE]",  # terminator
        b'data: {"choices":[{"delta":{"content":"unreached"}}]}',
    ]
    mocker.patch(
        "server.llm.requests.post",
        return_value=_FakeStreamResponse(200, lines),
    )

    out = list(_client("k").stream([{"role": "user", "content": "hi"}]))

    assert out == ["你", "好"]


def test_stream_sets_stream_flag_in_payload(mocker):
    post = mocker.patch(
        "server.llm.requests.post",
        return_value=_FakeStreamResponse(200, [b"data: [DONE]"]),
    )

    list(_client("k").stream([{"role": "user", "content": "hi"}]))

    _, kwargs = post.call_args
    assert kwargs["json"]["stream"] is True
    assert kwargs["stream"] is True


def test_stream_raises_on_http_error(mocker):
    mocker.patch(
        "server.llm.requests.post",
        return_value=_FakeStreamResponse(429, [], text="rate limited"),
    )

    with pytest.raises(ModelUnavailable, match="429"):
        list(_client("k").stream([{"role": "user", "content": "hi"}]))
