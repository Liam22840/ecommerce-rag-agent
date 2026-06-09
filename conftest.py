# Presence of this file marks the project root for pytest, so the
# `ingestion` package is importable from test files without extra setup.

import pytest

from server.config import Settings


@pytest.fixture(autouse=True)
def _isolate_caches():
    """Tests must run against EMPTY on-disk caches and must not persist into the real ones. Both the
    query cache and the intent-keyed filter cache default to data/*.jsonl, which a dev's running
    server also writes — a stale entry there would otherwise serve a pre-change answer into a test
    (and tests would pollute the dev's cache). Move any existing files aside for the test, then
    restore them; remove anything the test created."""
    settings = Settings()
    paths = [settings.query_cache_path, settings.filter_cache_path]
    backups = {}
    for path in paths:
        if path.exists():
            backup = path.with_suffix(path.suffix + ".pytest-bak")
            path.rename(backup)
            backups[path] = backup
    yield
    for path in paths:
        if path.exists():
            path.unlink()  # whatever the test wrote
        if path in backups:
            backups[path].rename(path)  # restore the dev's cache
