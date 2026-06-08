"""Runtime configuration for the backend service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    dataset_root: Path = PROJECT_ROOT / "ecommerce_agent_dataset"
    data_dir: Path = PROJECT_ROOT / "data"
    milvus_path: Path = PROJECT_ROOT / "data" / "milvus.db"
    embedding_cache_path: Path = PROJECT_ROOT / "data" / "embedding_cache.jsonl"
    query_cache_path: Path = PROJECT_ROOT / "data" / "query_cache.jsonl"
    filter_cache_path: Path = PROJECT_ROOT / "data" / "filter_cache.jsonl"
    embedding_dim: int = 2048
    chat_api_key: str | None = None
    embedding_api_key: str | None = None
    chat_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    chat_model: str = "gemini-3.1-flash-lite"
    embedding_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    embedding_model: str = "doubao-embedding-vision-251215"
    chat_timeout_seconds: float = 60.0
    # Query-time embedding must fail fast: if the embedding endpoint is slow,
    # degrade to lexical search (with a warning) rather than hang the request.
    embedding_timeout_seconds: float = 10.0
    retrieval_top_k: int = 5
    vector_search_k: int = 24
    # Short-term session memory:
    #  - history_turns: recent product-search turns kept and fed to the intent LLM (refinement context)
    #  - session_products_cap: distinct products remembered across the whole session (for "go back to" recall)
    #  - recent_products_cap: recent product ids for "第一个/前两个" reference resolution
    #  - shown_summary_cap: how many shown items to summarise per remembered turn
    history_turns: int = 3
    session_products_cap: int = 40
    recent_products_cap: int = 10
    shown_summary_cap: int = 5
    # Character window used when streaming the deterministic fallback answer token-by-token.
    stream_chunk_size: int = 18
    # Tolerance band applied to an approximate price ("三百左右") when it would otherwise
    # collapse to a zero-width min==max band (±fraction of the stated price).
    approx_price_tolerance: float = 0.15
    enable_vector_search: bool = True
    enable_llm: bool = True
    enable_llm_intent: bool = True
    # Hot-query cache: store answers to context-free product questions so an identical
    # repeat returns instantly without re-running retrieval or the LLM.
    enable_query_cache: bool = True
    query_cache_max_entries: int = 500
    # Filter-keyed answer cache: same answers keyed on the parsed intent instead of the raw
    # text, so paraphrases of one intent share an entry. Complements the exact query cache.
    enable_filter_cache: bool = True
    filter_cache_max_entries: int = 500

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")
        return cls(
            chat_api_key=_optional_env("ARK_CHAT_API_KEY"),
            embedding_api_key=_optional_env("ARK_EMBEDDING_API_KEY"),
            chat_base_url=os.environ.get("ARK_CHAT_BASE_URL", cls.chat_base_url),
            chat_model=os.environ.get("ARK_CHAT_MODEL", cls.chat_model),
            embedding_base_url=os.environ.get(
                "ARK_EMBEDDING_BASE_URL", cls.embedding_base_url
            ),
            embedding_model=os.environ.get("ARK_EMBEDDING_MODEL", cls.embedding_model),
            chat_timeout_seconds=float(
                os.environ.get("ARK_CHAT_TIMEOUT_SECONDS", cls.chat_timeout_seconds)
            ),
            embedding_timeout_seconds=float(
                os.environ.get("RAG_EMBED_TIMEOUT_SECONDS", cls.embedding_timeout_seconds)
            ),
            retrieval_top_k=int(os.environ.get("RAG_TOP_K", cls.retrieval_top_k)),
            vector_search_k=int(os.environ.get("RAG_VECTOR_SEARCH_K", cls.vector_search_k)),
            history_turns=int(os.environ.get("RAG_HISTORY_TURNS", cls.history_turns)),
            session_products_cap=int(os.environ.get("RAG_SESSION_PRODUCTS_CAP", cls.session_products_cap)),
            recent_products_cap=int(os.environ.get("RAG_RECENT_PRODUCTS_CAP", cls.recent_products_cap)),
            shown_summary_cap=int(os.environ.get("RAG_SHOWN_SUMMARY_CAP", cls.shown_summary_cap)),
            stream_chunk_size=int(os.environ.get("RAG_STREAM_CHUNK_SIZE", cls.stream_chunk_size)),
            approx_price_tolerance=float(
                os.environ.get("RAG_APPROX_PRICE_TOLERANCE", cls.approx_price_tolerance)
            ),
            enable_vector_search=_bool_env("ENABLE_VECTOR_SEARCH", True),
            enable_llm=_bool_env("ENABLE_LLM", True),
            enable_llm_intent=_bool_env("ENABLE_LLM_INTENT", True),
            enable_query_cache=_bool_env("ENABLE_QUERY_CACHE", True),
            query_cache_max_entries=int(
                os.environ.get("RAG_QUERY_CACHE_MAX", cls.query_cache_max_entries)
            ),
            enable_filter_cache=_bool_env("ENABLE_FILTER_CACHE", True),
            filter_cache_max_entries=int(
                os.environ.get("RAG_FILTER_CACHE_MAX", cls.filter_cache_max_entries)
            ),
        )


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
