"""Tests for the ingestion entrypoint script (ingest.py)."""

from __future__ import annotations

import ingest


def test_load_products_returns_all_with_product_ids():
    products = ingest.load_products()
    assert len(products) == 100  # full bundled dataset
    assert all("product_id" in p for p in products)


def test_load_products_respects_limit():
    products = ingest.load_products(limit=3)
    assert len(products) == 3


def test_main_returns_error_when_api_key_missing(mocker, monkeypatch, capsys):
    mocker.patch("ingest.load_dotenv")  # don't let .env supply a key
    monkeypatch.delenv("ARK_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["ingest"])

    rc = ingest.main()

    assert rc == 1
    assert "ARK_EMBEDDING_API_KEY not set" in capsys.readouterr().err


def test_main_runs_pipeline_and_reports_count(mocker, monkeypatch):
    mocker.patch("ingest.load_dotenv")
    monkeypatch.setenv("ARK_EMBEDDING_API_KEY", "fake-key")
    monkeypatch.setattr("sys.argv", ["ingest"])

    one_product = ingest.load_products(limit=1)
    mocker.patch("ingest.load_products", return_value=one_product)
    mocker.patch("ingest.EmbeddingCache")

    embedder = mocker.Mock()
    embedder.embed_chunks.return_value = [[0.0]]
    mocker.patch("ingest.DoubaoEmbedder", return_value=embedder)

    store = mocker.Mock()
    store.count.return_value = 7
    mocker.patch("ingest.MilvusStore", return_value=store)

    rc = ingest.main()

    assert rc == 0
    store.ensure_collection.assert_called_once()
    store.upsert.assert_called_once()
    embedder.embed_chunks.assert_called_once()
