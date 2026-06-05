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
    embedding_dim: int = 2048
    chat_api_key: str | None = None
    embedding_api_key: str | None = None
    chat_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    chat_model: str = "ep-20260514111645-lmgt2"
    embedding_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    embedding_model: str = "doubao-embedding-vision-251215"
    chat_timeout_seconds: float = 60.0
    # Query-time embedding must fail fast: if the embedding endpoint is slow,
    # degrade to lexical search (with a warning) rather than hang the request.
    embedding_timeout_seconds: float = 10.0
    retrieval_top_k: int = 5
    vector_search_k: int = 24
    enable_vector_search: bool = True
    enable_llm: bool = True
    enable_llm_intent: bool = True

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
            enable_vector_search=_bool_env("ENABLE_VECTOR_SEARCH", True),
            enable_llm=_bool_env("ENABLE_LLM", True),
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
