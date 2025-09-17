import sys
import threading
import time
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from llm.lmstudio_client import LMStudioClient
from llm.ollama_client import OllamaClient


class DummyLMStudioClient(LMStudioClient):
    def __init__(self):
        self.concurrent_enabled = True
        self.max_concurrent_requests = 2
        self.executor = None
        self._executor_lock = threading.Lock()

    def generate(self, prompt: str, **kwargs) -> str:  # type: ignore[override]
        time.sleep(0.01)
        return f"lm:{prompt}"


class DummyOllamaClient(OllamaClient):
    def __init__(self):
        self.concurrent_enabled = True
        self.max_concurrent_requests = 2
        self.executor = None
        self._executor_lock = threading.Lock()

    def generate(self, prompt: str, *args, **kwargs) -> str:  # type: ignore[override]
        time.sleep(0.01)
        return f"ollama:{prompt}"


def _assert_results(results, prefix, prompts):
    expected = Counter(f"{prefix}:{prompt}" for prompt in prompts)
    assert Counter(results) == expected


def test_lmstudio_generate_concurrent_reused_executor():
    client = DummyLMStudioClient()
    prompts = ["a", "b", "c"]

    try:
        first_results = client.generate_concurrent(prompts)
        first_executor = client.executor
        assert first_executor is not None

        second_results = client.generate_concurrent(prompts)
        second_executor = client.executor
        assert second_executor is first_executor
    finally:
        client.close()

    _assert_results(first_results, "lm", prompts)
    _assert_results(second_results, "lm", prompts)
    assert client.executor is None


def test_ollama_generate_concurrent_reused_executor():
    client = DummyOllamaClient()
    prompts = ["1", "2", "3"]

    try:
        first_results = client.generate_concurrent(prompts)
        first_executor = client.executor
        assert first_executor is not None

        second_results = client.generate_concurrent(prompts)
        second_executor = client.executor
        assert second_executor is first_executor
    finally:
        client.close()

    _assert_results(first_results, "ollama", prompts)
    _assert_results(second_results, "ollama", prompts)
    assert client.executor is None
