"""OpenAI-compatible chat client with streaming parsing."""

from __future__ import annotations

import json
from collections.abc import Iterator

import requests


class ModelUnavailable(RuntimeError):
    """Raised when the model cannot be called and the app should degrade."""


class ChatClient:
    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout_seconds: float = 60.0,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self._api_key:
            raise ModelUnavailable("CHAT_API_KEY not set")
        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json={"model": self._model, "messages": messages, "temperature": 0.2},
            timeout=self._timeout_seconds,
        )
        if resp.status_code >= 400:
            raise ModelUnavailable(f"chat completion failed {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        if not self._api_key:
            raise ModelUnavailable("CHAT_API_KEY not set")
        with requests.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self._model,
                "messages": messages,
                "temperature": 0.2,
                "stream": True,
            },
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
