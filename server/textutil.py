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


_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def chinese_to_int(token: str) -> int | None:
    """Convert a Chinese or mixed-digit integer (三百 / 一万 / 1万 / 三百五十 / 三百五 / 二十三 / 7)
    to an int, or None if it isn't a parseable number. The single home for this conversion, shared
    by the intent, planner and commerce parsers. The chat model handles Chinese numbers itself;
    this is the deterministic fallback for when it is unavailable."""
    if not token:
        return None
    total = section = number = last_unit = 0
    for ch in token:
        if ch.isdigit():
            number = number * 10 + int(ch)
        elif ch in _CN_DIGIT:
            number = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                section += (number or 1) * unit
            last_unit = unit
            number = 0
        else:
            return None
    if number and last_unit >= 10:  # trailing bare digit, e.g. 三百五 -> 350
        section += number * (last_unit // 10)
        number = 0
    return total + section + number


def trim(value: str, max_len: int) -> str:
    """Trim a string to max_len characters, adding an ellipsis when truncated."""
    value = value.strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "…"
