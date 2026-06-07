"""Volcengine Ark chat client with OpenAI-compatible streaming parsing."""

from __future__ import annotations

import json
from collections.abc import Iterator

import requests


class ModelUnavailable(RuntimeError):
    """Raised when the model cannot be called and the app should degrade."""


class ArkChatClient:
    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout_seconds: float = 60.0,
        disable_thinking: bool = False,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds
        # Doubao "Seed" models are reasoning models with deep thinking on by default, which adds
        # hundreds of hidden reasoning tokens and several seconds per call. For a grounded
        # shopping reply we don't need it, so disable it for those models (their low-latency
        # mode). Off for models that don't accept the field (e.g. the Gemini-compatible endpoint).
        self._disable_thinking = disable_thinking

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def _body(self, messages: list[dict[str, str]], **extra) -> dict:
        body = {"model": self._model, "messages": messages, "temperature": 0.2, **extra}
        if self._disable_thinking:
            body["thinking"] = {"type": "disabled"}
        return body

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self._api_key:
            raise ModelUnavailable("ARK_CHAT_API_KEY not set")
        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=self._body(messages),
            timeout=self._timeout_seconds,
        )
        if resp.status_code >= 400:
            raise ModelUnavailable(f"chat completion failed {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        if not self._api_key:
            raise ModelUnavailable("ARK_CHAT_API_KEY not set")
        with requests.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=self._body(messages, stream=True),
            timeout=self._timeout_seconds,
            stream=True,
        ) as resp:
            if resp.status_code >= 400:
                raise ModelUnavailable(f"chat stream failed {resp.status_code}: {resp.text[:300]}")
            for raw_line in resp.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8").strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = payload.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
