from types import SimpleNamespace

from relrag.generator import answerer as answerer_mod


def test_call_llm_uses_generate_profile_max_tokens_when_not_provided(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            return None

        def chat(self, messages, *, temperature, max_tokens, stop=None):
            captured["chat_max_tokens"] = max_tokens
            return SimpleNamespace(content="FINAL: test")

    def fake_budget_answer_prompt(*args, **kwargs):
        captured["requested_max_tokens"] = kwargs["requested_max_tokens"]
        report = SimpleNamespace(
            estimated_input_tokens=128,
            requested_max_tokens=kwargs["requested_max_tokens"],
            effective_max_tokens=kwargs["requested_max_tokens"],
            model_ctx_len=8192,
            available_tokens=8000,
            dropped_items_count=0,
            truncated_items_count=0,
            deduped_items_count=0,
            prompt_head="",
            items_count=0,
        )
        return SimpleNamespace(prompt="q", messages=[{"role": "user", "content": "q"}], items=[], report=report)

    monkeypatch.setattr(answerer_mod, "LLMChatClient", FakeClient)
    monkeypatch.setattr(answerer_mod, "budget_answer_prompt", fake_budget_answer_prompt)
    monkeypatch.setattr(answerer_mod, "log_budget_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(answerer_mod, "get_active_llm_stats", lambda: None)

    cfg = {
        "llm_profiles": {"generate": {"max_tokens": 256}},
        "vllm": {"max_tokens": 1024},
        "answer": {"max_evidence_items": 12, "max_evidence_tokens": 512},
        "answerer": {"stop": None},
    }
    out = answerer_mod.call_llm(
        endpoint="http://127.0.0.1:8000/v1",
        model="qwen3-30b-a3b",
        question="q",
        evidences=[],
        cfg=cfg,
    )
    assert out == "FINAL: test"
    assert captured["requested_max_tokens"] == 256
    assert captured["chat_max_tokens"] == 256


def test_call_llm_explicit_max_tokens_still_overrides_config(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            return None

        def chat(self, messages, *, temperature, max_tokens, stop=None):
            captured["chat_max_tokens"] = max_tokens
            return SimpleNamespace(content="FINAL: test")

    def fake_budget_answer_prompt(*args, **kwargs):
        captured["requested_max_tokens"] = kwargs["requested_max_tokens"]
        report = SimpleNamespace(
            estimated_input_tokens=128,
            requested_max_tokens=kwargs["requested_max_tokens"],
            effective_max_tokens=kwargs["requested_max_tokens"],
            model_ctx_len=8192,
            available_tokens=8000,
            dropped_items_count=0,
            truncated_items_count=0,
            deduped_items_count=0,
            prompt_head="",
            items_count=0,
        )
        return SimpleNamespace(prompt="q", messages=[{"role": "user", "content": "q"}], items=[], report=report)

    monkeypatch.setattr(answerer_mod, "LLMChatClient", FakeClient)
    monkeypatch.setattr(answerer_mod, "budget_answer_prompt", fake_budget_answer_prompt)
    monkeypatch.setattr(answerer_mod, "log_budget_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(answerer_mod, "get_active_llm_stats", lambda: None)

    cfg = {
        "llm_profiles": {"generate": {"max_tokens": 1024}},
        "vllm": {"max_tokens": 1024},
        "answer": {"max_evidence_items": 12, "max_evidence_tokens": 512},
        "answerer": {"stop": None},
    }
    out = answerer_mod.call_llm(
        endpoint="http://127.0.0.1:8000/v1",
        model="qwen3-30b-a3b",
        question="q",
        evidences=[],
        max_tokens=128,
        cfg=cfg,
    )
    assert out == "FINAL: test"
    assert captured["requested_max_tokens"] == 128
    assert captured["chat_max_tokens"] == 128
