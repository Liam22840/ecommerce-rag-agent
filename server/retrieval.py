"""Hybrid product retrieval over Milvus and the local product catalog."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from ingestion.cache import EmbeddingCache
from ingestion.embed import DoubaoEmbedder
from server.catalog import CatalogHit, ProductCatalog
from server.config import Settings
from server.intent import SearchFilters


# Reciprocal Rank Fusion constant (standard value). Larger -> rank differences matter less;
# 60 is the widely-used default. Rank-based fusion is scale-invariant, so it's an algorithm
# constant, not an operational tuning knob.
RRF_K = 60


@dataclass
class RetrievalResult:
    hits: list[CatalogHit]
    source: str
    warnings: list[str] = field(default_factory=list)


class ProductRetriever:
    def __init__(self, catalog: ProductCatalog, settings: Settings):
        self._catalog = catalog
        self._settings = settings
        self._vector_ready = False
        self._embedder: DoubaoEmbedder | None = None
        self._store = None
        self._startup_warning: str | None = None
        # Pipeline parallelism for the cold embed: prewarm_query() embeds the query on a worker
        # while the intent LLM runs, and retrieval awaits that same future instead of embedding
        # again. A plain background thread isn't enough — when the cold embed outlasts the intent
        # call, retrieval would miss the not-yet-written cache and embed a second time.
        self._embed_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="embed")
        self._pending: dict[str, Future] = {}
        self._pending_lock = threading.Lock()
        self._init_vector_search()

    def prewarm_query(self, text: str) -> None:
        """Start embedding `text` on a worker so it overlaps the intent LLM call; retrieval later
        awaits the same future (see _embed_query) instead of re-embedding. No-op when vector
        search is off. Best-effort: a failed embed surfaces when its future is awaited and is
        handled there, never here."""
        embedder = self._embedder
        if not (self._vector_ready and embedder is not None) or not text:
            return
        with self._pending_lock:
            # Drop finished futures: their vector is already in the embed cache, so a later
            # lookup hits the cache directly. This bounds the map to in-flight work only.
            self._pending = {k: f for k, f in self._pending.items() if not f.done()}
            if text not in self._pending:
                self._pending[text] = self._embed_pool.submit(embedder.embed_text, text)

    def _embed_query(self, text: str) -> list[float]:
        """Embed the retrieval query, reusing an in-flight pre-warm if one exists so a query is
        never embedded twice. Falls back to a direct embed (which hits the warm cache when a
        completed pre-warm already stored the vector)."""
        with self._pending_lock:
            future = self._pending.pop(text, None)
        if future is not None:
            return future.result()
        assert self._embedder is not None  # guarded by the caller (_vector_candidates)
        return self._embedder.embed_text(text)

    def retrieve(self, query: str, filters: SearchFilters, limit: int) -> RetrievalResult:
        warnings: list[str] = []
        if self._startup_warning:
            warnings.append(self._startup_warning)

        vector_ranked, used_vector = self._vector_candidates(query, filters, warnings)
        lexical_hits = self._catalog.search_lexical(query, filters, limit=max(limit * 3, 12))

        # Reciprocal Rank Fusion: each source contributes by RANK, not raw score, so the two
        # (cosine similarity vs lexical term-overlap, on very different scales) combine fairly
        # instead of lexical magnitude dominating. _merge_hit then sums the contributions,
        # flips overlapping products to "hybrid", and unions snippets.
        merged: dict[str, CatalogHit] = {}
        self._fuse_by_rank(merged, vector_ranked)
        self._fuse_by_rank(merged, lexical_hits)

        hits = sorted(
            merged.values(),
            key=lambda hit: (hit.score, -self._catalog.lowest_price(hit.product)),
            reverse=True,
        )[:limit]
        source = "none"
        if hits:
            source = "hybrid" if used_vector and lexical_hits else "vector" if used_vector else "lexical"
        return RetrievalResult(hits=hits, source=source, warnings=warnings)

    def _vector_candidates(
        self, query: str, filters: SearchFilters, warnings: list[str]
    ) -> tuple[list[CatalogHit], bool]:
        """Vector hits as one CatalogHit per product (best-scoring chunk), filtered and ranked by
        similarity. Returns (ranked_hits, used_vector). Degrades to ([], False) on any failure."""
        if not (self._vector_ready and self._embedder is not None and self._store is not None):
            return [], False
        try:
            vector = self._embed_query(query)
            raw_hits = self._store.search(vector, k=self._settings.vector_search_k)
        except Exception as exc:  # noqa: BLE001 - vector retrieval must degrade, not crash demo
            warnings.append(f"vector search unavailable: {exc}")
            return [], False
        # Keep the best-scoring chunk per product (score only orders the vector list — RRF
        # overwrites it with the rank contribution later, so the raw magnitude doesn't matter).
        best: dict[str, CatalogHit] = {}
        for raw in raw_hits:
            product = self._catalog.get(raw["product_id"])
            if product is None or not self._catalog.matches_filters(product, filters):
                continue
            pid = product["product_id"]
            score = float(raw.get("score") or 0.0)
            existing = best.get(pid)
            if existing is None or score > existing.score:
                snippet = raw.get("text") or product["title"]
                best[pid] = CatalogHit(product=product, score=score, snippets=[snippet], source="vector")
        ranked = sorted(best.values(), key=lambda hit: hit.score, reverse=True)
        return ranked, True

    def _fuse_by_rank(self, merged: dict[str, CatalogHit], ranked_hits: list[CatalogHit]) -> None:
        """Replace each hit's raw score with its reciprocal-rank contribution and merge it in.
        Input lists are already sorted best-first, so list position is the rank."""
        for rank, hit in enumerate(ranked_hits):
            hit.score = 1.0 / (RRF_K + rank)
            self._merge_hit(merged, hit)

    def _init_vector_search(self) -> None:
        if not self._settings.enable_vector_search:
            self._startup_warning = "vector search disabled by configuration"
            return
        if not self._settings.embedding_api_key:
            self._startup_warning = "ARK_EMBEDDING_API_KEY not set; using lexical fallback retrieval"
            return

        try:
            from ingestion.milvus_store import MilvusStore

            self._embedder = DoubaoEmbedder(
                api_key=self._settings.embedding_api_key,
                cache=EmbeddingCache(self._settings.embedding_cache_path),
                dataset_root=self._settings.dataset_root,
                base_url=self._settings.embedding_base_url,
                model=self._settings.embedding_model,
                timeout=self._settings.embedding_timeout_seconds,
                max_attempts=1,
            )
            self._store = MilvusStore(uri=str(self._settings.milvus_path), dim=self._settings.embedding_dim)
            self._store.ensure_collection()
            self._vector_ready = True
        except Exception as exc:  # noqa: BLE001
            self._startup_warning = f"vector search initialization failed: {exc}"

    @staticmethod
    def _merge_hit(merged: dict[str, CatalogHit], hit: CatalogHit) -> None:
        product_id = hit.product["product_id"]
        existing = merged.get(product_id)
        if existing is None:
            merged[product_id] = hit
            return
        existing.score += hit.score
        existing.source = "hybrid" if existing.source != hit.source else existing.source
        for snippet in hit.snippets:
            if snippet and snippet not in existing.snippets:
                existing.snippets.append(snippet)
