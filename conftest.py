# Presence of this file marks the project root for pytest, so the
# `ingestion` package is importable from test files without extra setup.

import pytest

from server.config import Settings


@pytest.fixture(autouse=True)
def _guard_query_cache_leak():
    """Safety net: tests must not persist the real data/query_cache.jsonl. If a test builds
    the app with default settings and writes the cache, remove the file afterwards (but leave
    a pre-existing one, e.g. from a dev's running server, untouched)."""
    path = Settings().query_cache_path
    existed = path.exists()
    yield
    if path.exists() and not existed:
        path.unlink()
