import sys
import types
import os
import importlib.util
import pytest


class FakeGenerateResponse:
    def __init__(self, text):
        self.response = text


class FakeClient:
    def __init__(self, host=None):
        self.host = host

    def generate(self, **kwargs):
        return FakeGenerateResponse(" text with space \x00")


# Patch the ollama module before importing OllamaClient
fake_ollama = types.SimpleNamespace(Client=FakeClient)
sys.modules.setdefault("ollama", fake_ollama)
sys.modules.setdefault("requests", types.SimpleNamespace(get=lambda *a, **k: types.SimpleNamespace(status_code=200)))
sys.modules.setdefault("loguru", types.SimpleNamespace(logger=types.SimpleNamespace(error=lambda *a, **k: None, warning=lambda *a, **k: None, info=lambda *a, **k: None)))
sys.modules.setdefault("config", types.SimpleNamespace(config=types.SimpleNamespace(get=lambda *a, **k: None)))
package_llm = types.ModuleType("llm")
package_llm.prompts = types.SimpleNamespace(
    FINAL_ANSWER_SYSTEM_PROMPT="",
    FINAL_ANSWER_PROMPT="",
    EVALUATE_ANSWER_SYSTEM_PROMPT="",
    EVALUATE_ANSWER_PROMPT="",
)
sys.modules.setdefault("llm", package_llm)
sys.modules.setdefault("llm.prompts", package_llm.prompts)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OLLAMA_PATH = os.path.join(ROOT_DIR, "llm", "ollama_client.py")
spec = importlib.util.spec_from_file_location("llm.ollama_client", OLLAMA_PATH)
ollama_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ollama_client)
OllamaClient = ollama_client.OllamaClient

def test_generate_returns_cleaned_string(monkeypatch):
    client = OllamaClient(base_url="http://test", model="dummy")
    monkeypatch.setattr(client, "_quick_health_check", lambda: True)
    result = client.generate("prompt")
    assert result == "text with space "
