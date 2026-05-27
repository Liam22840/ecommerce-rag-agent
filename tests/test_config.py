import os

from server.config import Settings


def test_settings_keeps_chat_and_embedding_keys_separate():
    settings = Settings(
        chat_api_key="chat",
        embedding_api_key="embedding",
    )

    assert settings.chat_api_key == "chat"
    assert settings.embedding_api_key == "embedding"
    assert settings.embedding_model == "doubao-embedding-vision-251215"


def test_settings_loads_dedicated_keys_from_env(monkeypatch):
    monkeypatch.setenv("ARK_CHAT_API_KEY", "chat")
    monkeypatch.setenv("ARK_EMBEDDING_API_KEY", "embedding")

    settings = Settings.load()

    assert settings.chat_api_key == "chat"
    assert settings.embedding_api_key == "embedding"
