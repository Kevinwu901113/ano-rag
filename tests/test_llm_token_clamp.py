from relrag.utils import llm_client as llm_client_mod
from relrag.utils import openai_client as openai_client_mod


class _FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 1},
        }


def test_llm_client_clamps_max_tokens(monkeypatch):
    monkeypatch.setattr(llm_client_mod.TokenCounter, "count_messages", lambda _messages: 8100)
    monkeypatch.setattr(
        llm_client_mod.global_config,
        "load_config",
        lambda: {
            "vllm": {"max_model_len": 8192, "context_safety_margin": 256},
            "llm": {"max_context_len": 8192, "safety_margin_tokens": 256},
        },
    )

    client = llm_client_mod.LLMChatClient(endpoint="http://127.0.0.1:8000/v1", model="qwen3-30b-a3b")
    payload = client._build_payload(
        [{"role": "user", "content": "hello"}],
        temperature=0.0,
        max_tokens=256,
        response_format=None,
        extra_body=None,
        stop=None,
        profile="generate",
    )
    client._clamp_payload_max_tokens(payload, [{"role": "user", "content": "hello"}])
    # allowed = 8192 - 256 - 8100 = -164 => clamp to MIN_OUTPUT_TOKENS=16
    assert payload["max_tokens"] == 16


def test_openai_client_clamps_max_tokens(monkeypatch):
    monkeypatch.setattr(openai_client_mod.TokenCounter, "count_messages", lambda _messages: 7900)
    monkeypatch.setattr(
        openai_client_mod.global_config,
        "load_config",
        lambda: {
            "vllm": {"max_model_len": 8192, "context_safety_margin": 256},
            "llm": {"max_context_len": 8192, "safety_margin_tokens": 256},
        },
    )

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None, proxies=None):
        captured["payload"] = dict(json or {})
        return _FakeResponse(status_code=200)

    monkeypatch.setattr(openai_client_mod.requests, "post", _fake_post)

    content = openai_client_mod.chat_completion(
        [{"role": "user", "content": "hello"}],
        model="deepseek-chat",
        api_key="sk-test",
        base_url="http://127.0.0.1:8000/v1",
        max_tokens=256,
        max_retries=0,
    )
    # allowed = 8192 - 256 - 7900 = 36
    assert captured["payload"]["max_tokens"] == 36
    assert content == "ok"
