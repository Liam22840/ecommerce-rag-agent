from server.config import Settings, _bool_env, _optional_env


def test_settings_keeps_chat_and_embedding_keys_separate():
    settings = Settings(
        chat_api_key="chat",
        embedding_api_key="embedding",
    )

    assert settings.chat_api_key == "chat"
    assert settings.embedding_api_key == "embedding"
    assert settings.embedding_model == "doubao-embedding-vision-251215"


def test_settings_loads_dedicated_keys_from_env(monkeypatch):
    monkeypatch.setenv("CHAT_API_KEY", "chat")
    monkeypatch.setenv("ARK_EMBEDDING_API_KEY", "embedding")

    settings = Settings.load()

    assert settings.chat_api_key == "chat"
    assert settings.embedding_api_key == "embedding"


def test_bool_env_parses_truthy_and_falsy(monkeypatch):
    for raw in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("RAG_FLAG", raw)
        assert _bool_env("RAG_FLAG", False) is True
    for raw in ("0", "false", "no", "off", "garbage"):
        monkeypatch.setenv("RAG_FLAG", raw)
        assert _bool_env("RAG_FLAG", True) is False


def test_bool_env_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("RAG_FLAG", raising=False)
    assert _bool_env("RAG_FLAG", True) is True
    assert _bool_env("RAG_FLAG", False) is False


def test_optional_env_treats_blank_as_none(monkeypatch):
    monkeypatch.setenv("RAG_OPT", "   ")
    assert _optional_env("RAG_OPT") is None
    monkeypatch.setenv("RAG_OPT", " value ")
    assert _optional_env("RAG_OPT") == " value "
    monkeypatch.delenv("RAG_OPT", raising=False)
    assert _optional_env("RAG_OPT") is None


def test_load_applies_scalar_env_overrides(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "custom-model")
    monkeypatch.setenv("CHAT_BASE_URL", "https://chat.example")
    monkeypatch.setenv("CHAT_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("RAG_TOP_K", "8")
    monkeypatch.setenv("RAG_VECTOR_SEARCH_K", "30")
    monkeypatch.setenv("RAG_EMBED_TIMEOUT_SECONDS", "3.0")

    settings = Settings.load()

    assert settings.chat_model == "custom-model"
    assert settings.chat_base_url == "https://chat.example"
    assert settings.chat_timeout_seconds == 12.5
    assert settings.retrieval_top_k == 8
    assert settings.vector_search_k == 30
    assert settings.embedding_timeout_seconds == 3.0


def test_load_parses_boolean_feature_flags(monkeypatch):
    monkeypatch.setenv("ENABLE_VECTOR_SEARCH", "false")
    monkeypatch.setenv("ENABLE_LLM", "off")

    settings = Settings.load()

    assert settings.enable_vector_search is False
    assert settings.enable_llm is False


def test_enable_llm_intent_defaults_true():
    # enable_llm_intent is not env-driven, it defaults on for the LLM intent parser.
    assert Settings().enable_llm_intent is True
