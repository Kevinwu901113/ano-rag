import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.parallel_task_atomic_note_generator import ParallelTaskAtomicNoteGenerator


class DummyLLM:
    def __init__(self):
        self.is_hybrid_mode = False
        self.client = None
        self.lmstudio_client = None

    def generate(self, prompt, system_prompt=None, **kwargs):
        return "[]"


class RecordingClient:
    def __init__(self, responses, should_fail=False):
        if isinstance(responses, str):
            responses = [responses]
        self._responses = list(responses)
        self.calls = []
        self.should_fail = should_fail

    def generate(self, prompt, system_prompt=None, **kwargs):
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "kwargs": kwargs,
        })
        if self.should_fail:
            raise RuntimeError("simulated failure")

        if not self._responses:
            return "[]"

        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def patch_parallel_helpers(monkeypatch):
    monkeypatch.setattr(
        ParallelTaskAtomicNoteGenerator,
        "_init_parallel_clients",
        lambda self: None,
        raising=False,
    )
    monkeypatch.setattr(
        ParallelTaskAtomicNoteGenerator,
        "_format_atomic_note_prompt",
        lambda self, chunk: chunk.get("text", ""),
        raising=False,
    )
    monkeypatch.setattr(
        ParallelTaskAtomicNoteGenerator,
        "_batch_convert",
        lambda self, parsed_notes, chunk_data: parsed_notes,
        raising=False,
    )
    monkeypatch.setattr(
        ParallelTaskAtomicNoteGenerator,
        "_get_atomic_note_system_prompt",
        lambda self: "SYS",
        raising=False,
    )
    monkeypatch.setattr(
        "llm.parallel_task_atomic_note_generator.extract_json_from_response",
        lambda response: response,
    )
    monkeypatch.setattr(
        "llm.parallel_task_atomic_note_generator.parse_notes_response",
        lambda response, sentinel='~': json.loads(response),
    )


def _create_generator(monkeypatch, llm_params):
    monkeypatch.setattr(
        ParallelTaskAtomicNoteGenerator,
        "_get_optimized_llm_params",
        lambda self: llm_params,
        raising=False,
    )

    generator = ParallelTaskAtomicNoteGenerator(DummyLLM())
    generator.parallel_enabled = True
    generator.parallel_strategy = "task_division"
    generator.allocation_method = "round_robin"
    generator.enable_fallback = True
    generator.fallback_timeout = 5
    return generator


def test_parallel_clients_receive_shared_llm_params(monkeypatch):
    llm_params = {
        "temperature": 0.3,
        "top_p": 0.8,
        "max_tokens": 256,
        "stop": ["END"],
    }
    generator = _create_generator(monkeypatch, llm_params)

    ollama_stub = RecordingClient('[{"content": "ok", "source_sent_ids": [1], "paragraph_idxs": [1]}]')
    lmstudio_stub = RecordingClient('[{"content": "ok", "source_sent_ids": [1], "paragraph_idxs": [1]}]')
    generator.ollama_client = ollama_stub
    generator.lmstudio_client = lmstudio_stub

    chunk = {"text": "chunk text"}
    generator._process_with_ollama(chunk, 0, "SYS")
    generator._process_with_lmstudio(chunk, 1, "SYS")

    assert ollama_stub.calls[0]["kwargs"]["temperature"] == llm_params["temperature"]
    assert ollama_stub.calls[0]["kwargs"]["top_p"] == llm_params["top_p"]
    assert ollama_stub.calls[0]["kwargs"]["max_tokens"] == llm_params["max_tokens"]
    assert ollama_stub.calls[0]["kwargs"]["stop"] == llm_params["stop"]
    assert ollama_stub.calls[0]["kwargs"]["timeout"] == 30

    assert lmstudio_stub.calls[0]["kwargs"]["temperature"] == llm_params["temperature"]
    assert lmstudio_stub.calls[0]["kwargs"]["top_p"] == llm_params["top_p"]
    assert lmstudio_stub.calls[0]["kwargs"]["max_tokens"] == llm_params["max_tokens"]
    assert lmstudio_stub.calls[0]["kwargs"]["stop"] == llm_params["stop"]


def test_parallel_generation_with_fallback(monkeypatch):
    llm_params = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 128,
        "stop": ["END"],
    }
    generator = _create_generator(monkeypatch, llm_params)

    fallback_response = '[{"content": "fallback note", "source_sent_ids": [1], "paragraph_idxs": [1]}]'
    lm_response = '[{"content": "lm note", "source_sent_ids": [1], "paragraph_idxs": [1]}]'

    generator.ollama_client = RecordingClient([], should_fail=True)
    generator.lmstudio_client = RecordingClient([fallback_response, lm_response])

    chunks = [
        {"text": "ollama chunk"},
        {"text": "lm chunk"},
    ]

    start = time.time()
    notes = generator._generate_atomic_notes_parallel_task_division(chunks)
    elapsed = time.time() - start

    assert elapsed < 30
    assert len(notes) == 2
    assert {note["content"] for note in notes} == {"fallback note", "lm note"}
    assert generator.stats["fallback_count"] == 1

    lm_calls = generator.lmstudio_client.calls
    assert any(call["prompt"] == "ollama chunk" for call in lm_calls)
    assert any(call["prompt"] == "lm chunk" for call in lm_calls)
    for call in lm_calls:
        assert call["kwargs"]["max_tokens"] == llm_params["max_tokens"]
        assert call["kwargs"]["stop"] == llm_params["stop"]
