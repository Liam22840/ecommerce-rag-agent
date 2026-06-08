"""Tests for hybrid retrieval: vector path, degradation, merge, and source labels."""

from __future__ import annotations

import time
from pathlib import Path

from server.catalog import CatalogHit, ProductCatalog
from server.config import Settings
from server.intent import IntentParser, SearchFilters
from server.retrieval import ProductRetriever, RetrievalResult, RRF_K


DATASET_ROOT = Path(__file__).parent.parent / "ecommerce_agent_dataset"


class FakeEmbedder:
    def __init__(self, vector=None, error: Exception | None = None):
        self._vector = vector if vector is not None else [0.1, 0.2]
        self._error = error
        self.calls: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.calls.append(text)
        if self._error is not None:
            raise self._error
        return self._vector


class FakeStore:
    def __init__(self, raw_hits: list[dict]):
        self._raw_hits = raw_hits
        self.calls: list[tuple] = []

    def search(self, vector, k: int) -> list[dict]:
        self.calls.append((vector, k))
        return self._raw_hits


def _lexical_settings() -> Settings:
    return Settings(
        dataset_root=DATASET_ROOT,
        embedding_api_key=None,
        enable_vector_search=False,
    )


def _retriever_with_vector(catalog, raw_hits, embedder=None) -> ProductRetriever:
    """Build a lexical-only retriever then inject fake vector components."""
    retriever = ProductRetriever(catalog, _lexical_settings())
    retriever._vector_ready = True
    retriever._embedder = embedder or FakeEmbedder()
    retriever._store = FakeStore(raw_hits)
    retriever._startup_warning = None
    return retriever


def _filters(catalog, query: str) -> SearchFilters:
    parser = IntentParser(catalog.categories, catalog.sub_categories, catalog.brands)
    return parser.parse(query)


# --- startup warnings -----------------------------------------------------------

def test_disabled_vector_search_records_startup_warning():
    catalog = ProductCatalog.load(DATASET_ROOT)
    retriever = ProductRetriever(catalog, _lexical_settings())

    result = retriever.retrieve("推荐一款适合油皮的洗面奶", _filters(catalog, "推荐一款适合油皮的洗面奶"), limit=3)

    assert any("disabled by configuration" in w for w in result.warnings)
    assert result.source == "lexical"


def test_enabled_vector_search_without_api_key_warns():
    catalog = ProductCatalog.load(DATASET_ROOT)
    settings = Settings(dataset_root=DATASET_ROOT, embedding_api_key=None, enable_vector_search=True)

    retriever = ProductRetriever(catalog, settings)

    assert retriever._vector_ready is False
    assert "ARK_EMBEDDING_API_KEY" in (retriever._startup_warning or "")


def test_vector_init_failure_is_caught_and_warned(mocker):
    catalog = ProductCatalog.load(DATASET_ROOT)
    settings = Settings(dataset_root=DATASET_ROOT, embedding_api_key="x", enable_vector_search=True)
    mocker.patch("server.retrieval.DoubaoEmbedder", side_effect=RuntimeError("boom"))

    retriever = ProductRetriever(catalog, settings)

    assert retriever._vector_ready is False
    assert "initialization failed" in (retriever._startup_warning or "")
    assert "boom" in retriever._startup_warning


# --- lexical-only behaviour -----------------------------------------------------

def test_lexical_only_returns_hits_with_lexical_source():
    catalog = ProductCatalog.load(DATASET_ROOT)
    retriever = ProductRetriever(catalog, _lexical_settings())
    retriever._startup_warning = None  # isolate from the disabled-warning noise

    result = retriever.retrieve("推荐一款适合油皮的洗面奶", _filters(catalog, "推荐一款适合油皮的洗面奶"), limit=3)

    assert result.source == "lexical"
    assert result.warnings == []
    assert result.hits
    assert result.hits[0].product["sub_category"] == "洁面"


def test_no_matches_yields_none_source():
    catalog = ProductCatalog.load(DATASET_ROOT)
    retriever = ProductRetriever(catalog, _lexical_settings())
    retriever._startup_warning = None

    query = "200 元以下的蓝牙耳机有哪些？"
    result = retriever.retrieve(query, _filters(catalog, query), limit=3)

    assert result.hits == []
    assert result.source == "none"


# --- vector path ----------------------------------------------------------------

def test_vector_hit_with_no_lexical_overlap_is_vector_source():
    catalog = ProductCatalog.load(DATASET_ROOT)
    raw = [{"product_id": "p_beauty_011", "score": 0.5, "text": "向量片段"}]
    retriever = _retriever_with_vector(catalog, raw)

    # "zzzzz" has no lexical overlap, so only the vector hit survives.
    result = retriever.retrieve("zzzzz", SearchFilters(), limit=3)

    assert result.source == "vector"
    assert [hit.product["product_id"] for hit in result.hits] == ["p_beauty_011"]
    assert result.hits[0].source == "vector"
    assert "向量片段" in result.hits[0].snippets
    assert retriever._embedder.calls == ["zzzzz"]


def test_vector_and_lexical_overlap_is_hybrid_and_scores_merge():
    catalog = ProductCatalog.load(DATASET_ROOT)
    raw = [{"product_id": "p_beauty_011", "score": 1.0, "text": "向量片段"}]
    retriever = _retriever_with_vector(catalog, raw)

    query = "推荐一款适合油皮的洗面奶"
    result = retriever.retrieve(query, _filters(catalog, query), limit=3)

    assert result.source == "hybrid"
    top = next(hit for hit in result.hits if hit.product["product_id"] == "p_beauty_011")
    # RRF: a product found by BOTH sources sums two reciprocal-rank contributions, so its score
    # exceeds the most any single source can give (1/RRF_K), and merging flips it to hybrid.
    assert top.score > 1.0 / RRF_K
    assert top.source == "hybrid"


def test_vector_hits_failing_filters_are_dropped():
    catalog = ProductCatalog.load(DATASET_ROOT)
    raw = [
        {"product_id": "p_beauty_011", "score": 0.9, "text": "片段"},
        {"product_id": "does_not_exist", "score": 0.9, "text": "片段"},
    ]
    retriever = _retriever_with_vector(catalog, raw)

    # max_price below every product price drops the real vector hit too.
    result = retriever.retrieve("zzzzz", SearchFilters(max_price=1.0), limit=3)

    assert result.hits == []
    assert result.source == "none"


def test_vector_search_exception_degrades_to_lexical():
    catalog = ProductCatalog.load(DATASET_ROOT)
    retriever = _retriever_with_vector(
        catalog, raw_hits=[], embedder=FakeEmbedder(error=RuntimeError("embed down"))
    )

    query = "推荐一款适合油皮的洗面奶"
    result = retriever.retrieve(query, _filters(catalog, query), limit=3)

    assert result.source == "lexical"
    assert any("vector search unavailable" in w for w in result.warnings)
    assert "embed down" in " ".join(result.warnings)
    assert result.hits  # lexical fallback still produced results


# --- _merge_hit -----------------------------------------------------------------

def test_merge_hit_inserts_new_product():
    merged: dict = {}
    hit = CatalogHit(product={"product_id": "p1"}, score=3.0, snippets=["a"], source="lexical")
    ProductRetriever._merge_hit(merged, hit)
    assert merged["p1"] is hit


def test_merge_hit_accumulates_score_marks_hybrid_and_unions_snippets():
    merged: dict = {}
    base = CatalogHit(product={"product_id": "p1"}, score=2.0, snippets=["a"], source="lexical")
    ProductRetriever._merge_hit(merged, base)
    incoming = CatalogHit(product={"product_id": "p1"}, score=5.0, snippets=["a", "b"], source="vector")

    ProductRetriever._merge_hit(merged, incoming)

    assert merged["p1"].score == 7.0
    assert merged["p1"].source == "hybrid"
    assert merged["p1"].snippets == ["a", "b"]


def test_merge_hit_same_source_keeps_source():
    merged: dict = {}
    base = CatalogHit(product={"product_id": "p1"}, score=2.0, snippets=[], source="lexical")
    ProductRetriever._merge_hit(merged, base)
    ProductRetriever._merge_hit(
        merged, CatalogHit(product={"product_id": "p1"}, score=1.0, snippets=[], source="lexical")
    )
    assert merged["p1"].source == "lexical"
    assert merged["p1"].score == 3.0


def test_rrf_fusion_combines_by_rank_not_raw_magnitude():
    # The point of RRF: a product ranked #1 by one source and #2 by the other ties with the
    # mirror-image product, regardless of how different the raw scores are (here vector 999 vs 1).
    catalog = ProductCatalog.load(DATASET_ROOT)
    retriever = ProductRetriever(catalog, _lexical_settings())
    merged: dict = {}
    vector_ranked = [
        CatalogHit(product={"product_id": "A"}, score=999.0, snippets=[], source="vector"),
        CatalogHit(product={"product_id": "B"}, score=1.0, snippets=[], source="vector"),
    ]
    lexical_ranked = [
        CatalogHit(product={"product_id": "B"}, score=5.0, snippets=[], source="lexical"),
        CatalogHit(product={"product_id": "A"}, score=4.0, snippets=[], source="lexical"),
    ]
    retriever._fuse_by_rank(merged, vector_ranked)
    retriever._fuse_by_rank(merged, lexical_ranked)

    # A = 1/RRF_K (vec #1) + 1/(RRF_K+1) (lex #2), B is the mirror image -> equal scores.
    assert abs(merged["A"].score - merged["B"].score) < 1e-12
    assert merged["A"].score == 1.0 / RRF_K + 1.0 / (RRF_K + 1)
    assert merged["A"].source == "hybrid" and merged["B"].source == "hybrid"


def test_retrieval_result_dataclass_defaults_warnings():
    result = RetrievalResult(hits=[], source="none")
    assert result.warnings == []


# --- speculative pre-warm -------------------------------------------------------

def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_prewarm_query_is_noop_without_vector_search():
    catalog = ProductCatalog.load(DATASET_ROOT)
    retriever = ProductRetriever(catalog, _lexical_settings())  # vector disabled, no embedder
    # Must not raise and must not need an embedder.
    retriever.prewarm_query("推荐一款洗面奶")


def test_prewarm_query_embeds_in_background_when_vector_ready():
    catalog = ProductCatalog.load(DATASET_ROOT)
    embedder = FakeEmbedder()
    retriever = _retriever_with_vector(catalog, raw_hits=[], embedder=embedder)

    retriever.prewarm_query("推荐一款洗面奶")

    # The daemon thread embeds the raw query so retrieval's later embed_text is a cache hit.
    assert _wait_until(lambda: embedder.calls == ["推荐一款洗面奶"])


def test_prewarm_query_ignores_embed_failure():
    catalog = ProductCatalog.load(DATASET_ROOT)
    embedder = FakeEmbedder(error=RuntimeError("embed down"))
    retriever = _retriever_with_vector(catalog, raw_hits=[], embedder=embedder)

    retriever.prewarm_query("推荐一款洗面奶")  # the embed runs on a worker, the failure stays in the future

    assert _wait_until(lambda: embedder.calls == ["推荐一款洗面奶"])
    # When retrieval awaits the failed pre-warm it degrades to lexical, never crashes.
    result = retriever.retrieve("推荐一款洗面奶", _filters(catalog, "推荐一款洗面奶"), limit=3)
    assert result.source == "lexical"


def test_prewarm_then_retrieve_embeds_the_query_only_once():
    # The whole point of the fix: retrieval awaits the in-flight pre-warm instead of embedding
    # the query a second time (which the cold-embed-slower-than-intent case would otherwise cause).
    catalog = ProductCatalog.load(DATASET_ROOT)
    raw = [{"product_id": "p_beauty_011", "score": 0.5, "text": "片段"}]
    embedder = FakeEmbedder()
    retriever = _retriever_with_vector(catalog, raw_hits=raw, embedder=embedder)

    query = "推荐一款适合油皮的洗面奶"
    retriever.prewarm_query(query)
    retriever.retrieve(query, _filters(catalog, query), limit=3)

    assert embedder.calls.count(query) == 1
