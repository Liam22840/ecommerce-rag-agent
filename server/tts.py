"""Gemini text-to-speech integration."""

from __future__ import annotations

import base64
import struct

import requests


class TextToSpeechUnavailable(RuntimeError):
    """Raised when cloud TTS cannot generate audio."""


class GeminiTextToSpeechClient:
    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        voice: str,
        timeout_seconds: float = 30.0,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._voice = voice
        self._timeout_seconds = timeout_seconds

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def synthesize_wav(self, text: str) -> bytes:
        if not self._api_key:
            raise TextToSpeechUnavailable("TTS_API_KEY not set")

        prompt = (
            "Say in a warm, friendly Mandarin shopping-guide voice, with natural pacing "
            "and clear product names. Read exactly this text:\n"
            f"{text.strip()}"
        )
        response = requests.post(
            f"{self._base_url}/models/{self._model}:generateContent",
            headers={
                "x-goog-api-key": self._api_key,
                "Content-Type": "application/json",
            },
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": self._voice,
                            }
                        }
                    },
                },
                "model": self._model,
            },
            timeout=self._timeout_seconds,
        )
        if response.status_code >= 400:
            raise TextToSpeechUnavailable(
                f"TTS generation failed {response.status_code}: {response.text[:300]}"
            )

        try:
            payload = response.json()
            encoded_audio = payload["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
            pcm = base64.b64decode(encoded_audio)
        except (KeyError, IndexError, ValueError) as exc:
            raise TextToSpeechUnavailable("TTS response did not contain audio") from exc

        if not pcm:
            raise TextToSpeechUnavailable("TTS response audio was empty")

        return _wav_bytes(pcm)


def _wav_bytes(
    pcm: bytes,
    channels: int = 1,
    sample_rate: int = 24_000,
    bits_per_sample: int = 16,
) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm)
    riff_size = 36 + data_size
    header = b"".join(
        [
            b"RIFF",
            struct.pack("<I", riff_size),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample),
            b"data",
            struct.pack("<I", data_size),
        ]
    )
    return header + pcm
