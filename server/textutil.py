"""Shared, dependency-free text and JSON helpers.

This module sits at the bottom of the import graph (it imports nothing from server/),
so any module can use it without creating a circular import. It is the single home for
the small helpers shared by intent.py, comparison.py, catalog.py and assistant.py.
"""

from __future__ import annotations

import json
import re
from typing import Any


def json_object(raw: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from an LLM response."""
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


def normalize(value: str) -> str:
    """Lowercase and strip whitespace plus common punctuation/delimiters."""
    return re.sub(r"[\s·,，。；;:：/\\()（）「」『』【】\[\]_-]+", "", value.lower())


def normalize_spec(value: str) -> str:
    """Lowercase and strip whitespace only (e.g. '50 G' -> '50g')."""
    return re.sub(r"\s+", "", str(value)).lower()


def dedupe(items: list[str]) -> list[str]:
    """Order-preserving dedupe of strings, dropping empties."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def dedupe_int(items: list[int]) -> list[int]:
    """Order-preserving dedupe of ints."""
    seen: set[int] = set()
    result: list[int] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def dedupe_ids(ids: list[str]) -> list[str]:
    """Order-preserving dedupe of ids, trimming whitespace and dropping empties."""
    seen: set[str] = set()
    result: list[str] = []
    for value in ids:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def trim(value: str, max_len: int) -> str:
    """Trim a string to max_len characters, adding an ellipsis when truncated."""
    value = value.strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "…"
