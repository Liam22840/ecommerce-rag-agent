"""Unit tests for the shared text and JSON helpers in server.textutil."""

from __future__ import annotations

from server.textutil import (
    dedupe,
    dedupe_ids,
    dedupe_int,
    json_object,
    normalize,
    normalize_spec,
    trim,
)


# --- json_object ----------------------------------------------------------------

def test_json_object_parses_a_plain_object():
    assert json_object('{"a": 1, "b": {"c": 2}}') == {"a": 1, "b": {"c": 2}}


def test_json_object_rejects_valid_json_that_is_not_an_object():
    assert json_object("[1, 2, 3]") == {}
    assert json_object('"a string"') == {}


def test_json_object_extracts_an_object_embedded_in_prose():
    assert json_object('好的，结果是 {"dimensions": []} 完毕') == {"dimensions": []}


def test_json_object_returns_empty_on_unparseable_input():
    assert json_object("not json at all") == {}
    assert json_object("text {still not json} more") == {}


# --- normalize ------------------------------------------------------------------

def test_normalize_lowercases_and_strips_punctuation_and_whitespace():
    assert normalize("A·B, C。D（E）") == "abcde"
    assert normalize("降 噪_效-果") == "降噪效果"
    assert normalize("Hello, World!") == "helloworld!"  # comma/space stripped, '!' is not a listed delimiter
    assert normalize("") == ""


# --- normalize_spec -------------------------------------------------------------

def test_normalize_spec_strips_only_whitespace_and_lowercases():
    assert normalize_spec("50 G") == "50g"
    assert normalize_spec("  16 TB  ") == "16tb"
    assert normalize_spec("A-B") == "a-b"  # punctuation is preserved, unlike normalize


def test_normalize_spec_coerces_non_strings():
    assert normalize_spec(100) == "100"


# --- dedupe / dedupe_int / dedupe_ids -------------------------------------------

def test_dedupe_preserves_order_and_drops_empties():
    assert dedupe(["a", "", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert dedupe([]) == []


def test_dedupe_int_preserves_order_and_keeps_zero():
    assert dedupe_int([1, 2, 1, 3, 2]) == [1, 2, 3]
    assert dedupe_int([0, 0]) == [0]


def test_dedupe_ids_trims_drops_empties_and_dedupes():
    assert dedupe_ids(["a", "", "b", "a"]) == ["a", "b"]
    assert dedupe_ids([" x ", "x"]) == ["x"]
    assert dedupe_ids(["  ", ""]) == []


# --- trim -----------------------------------------------------------------------

def test_trim_returns_stripped_value_when_within_limit():
    assert trim("abc", 10) == "abc"
    assert trim("  padded  ", 10) == "padded"
    assert trim("abcdef", 6) == "abcdef"  # exactly at the limit


def test_trim_truncates_with_ellipsis_when_too_long():
    assert trim("abcdef", 4) == "abc…"
