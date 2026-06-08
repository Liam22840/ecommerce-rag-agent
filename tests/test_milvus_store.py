from pathlib import Path

from ingestion.chunk import Chunk
from ingestion.milvus_store import MilvusStore


def _chunk(chunk_id: str, **overrides) -> Chunk:
    defaults = dict(
        product_id="p1",
        chunk_type="summary",
        text="some text",
        category="美妆护肤",
        sub_category="精华",
        brand="测试",
        base_price=100.0,
        image_path=None,
    )
    defaults.update(overrides)
    return Chunk(chunk_id=chunk_id, **defaults)


def test_insert_and_search_roundtrip(tmp_path: Path):
    db = tmp_path / "milvus.db"
    store = MilvusStore(uri=str(db), dim=4)
    store.ensure_collection()

    chunks = [
        _chunk("p1::summary", text="A"),
        _chunk("p2::summary", product_id="p2", text="B"),
    ]
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    store.upsert(chunks, vectors)
    # Milvus Lite buffers writes, flush so the rows are searchable deterministically.
    store.client.flush(collection_name=store.collection_name)

    hits = store.search(query_vector=[1.0, 0.0, 0.0, 0.0], k=2)
    assert len(hits) == 2
    top = hits[0]
    assert top["chunk_id"] == "p1::summary"
    assert top["product_id"] == "p1"


def test_count_reflects_stored_rows(tmp_path: Path):
    db = tmp_path / "milvus.db"
    store = MilvusStore(uri=str(db), dim=4)
    store.ensure_collection()
    assert store.count() == 0

    store.upsert(
        [_chunk("p1::summary", text="A"), _chunk("p2::summary", product_id="p2", text="B")],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    store.client.flush(collection_name=store.collection_name)
    assert store.count() == 2


def test_upsert_is_idempotent_on_chunk_id(tmp_path: Path):
    db = tmp_path / "milvus.db"
    store = MilvusStore(uri=str(db), dim=4)
    store.ensure_collection()

    c = _chunk("p1::summary", text="A")
    store.upsert([c], [[1.0, 0.0, 0.0, 0.0]])
    store.upsert([c], [[1.0, 0.0, 0.0, 0.0]])
    # Milvus Lite buffers writes, flush so the rows are searchable deterministically.
    store.client.flush(collection_name=store.collection_name)

    hits = store.search(query_vector=[1.0, 0.0, 0.0, 0.0], k=10)
    chunk_ids = [h["chunk_id"] for h in hits]
    assert chunk_ids.count("p1::summary") == 1
