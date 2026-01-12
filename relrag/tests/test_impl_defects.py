import io
import json
import re
import concurrent.futures
from unittest.mock import MagicMock, patch
from pathlib import Path

from relrag.retriever.ir import QueryIR
from relrag.retriever.parser import parse_question
from relrag.schema.note_schema_v1 import ALLOWED_PREDICATES
from relrag.validators.note_validator import validate_and_normalize
from hotpot_entry import _drain_futures, _process_example


def test_max_hops_not_forced_to_single():
    ir = parse_question("Who is the spouse of Ada Lovelace?")
    assert ir is not None
    default_hops = QueryIR.__dataclass_fields__["max_hops"].default
    assert ir.max_hops >= default_hops


def test_born_on_not_normalized_to_born_in():
    raw_notes = [
        {
            "subj": "Ada Lovelace",
            "pred": "born_on",
            "obj": "1815-12-10",
            "subj_type": "PERSON",
            "obj_type": "TIME",
            "evidence": "Ada Lovelace was born on 1815-12-10.",
            "meta": {"source": "doc#c0000", "confidence": 0.9},
        }
    ]
    result = validate_and_normalize(json.dumps(raw_notes), "doc", "c0000")
    assert result["valid_notes"]
    assert result["valid_notes"][0]["pred"] == "born_on"


def test_prompt_predicates_match_schema():
    prompt_path = Path(__file__).resolve().parents[1] / "prompt" / "note_extract.txt"
    text = prompt_path.read_text(encoding="utf-8")
    match = re.search(r"\"pred\"\s+must be one of\s+(\[[^\]]+\])", text)
    assert match, "Prompt predicate whitelist not found"
    prompt_predicates = json.loads(match.group(1))
    assert set(prompt_predicates) == set(ALLOWED_PREDICATES)


def test_drain_futures_aborts_on_stall():
    fut = concurrent.futures.Future()
    future_map = {fut: "qid_1"}
    handle = io.StringIO()
    completed, succeeded = _drain_futures(
        future_map,
        handle,
        progress=None,
        stall_warn_sec=1.0,
        stall_abort_sec=1.0,
    )
    assert completed == 1
    assert succeeded == 0
    assert fut.cancelled()


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = json.dumps(payload)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _dummy_chat_payload(content: str) -> dict:
    return {
        "id": "chatcmpl-dummy",
        "object": "chat.completion",
        "created": 0,
        "model": "dummy-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def test_hotpot_entry_smoke_debug_artifacts(tmp_path):
    example = {
        "_id": "q_smoke",
        "question": "Who wrote Python?",
        "context": [["Python", ["Python was created by Guido van Rossum."]]],
    }
    dummy_notes = [
        {
            "subj": "Python",
            "pred": "authored_by",
            "obj": "Guido van Rossum",
            "subj_type": "WORK",
            "obj_type": "PERSON",
            "evidence": "Python was created by Guido van Rossum.",
            "meta": {"source": "doc#c0000", "confidence": 0.9},
        }
    ]
    payload = _dummy_chat_payload(json.dumps(dummy_notes))

    def _mock_post_chat(*_args, **_kwargs):
        return DummyResponse(payload)

    cache_root = tmp_path / "cache"
    debug_dir = tmp_path / "debug"

    with patch("relrag.utils.llm_client.LLMChatClient.post_chat", side_effect=_mock_post_chat):
        with patch("relrag.utils.llm_client.LLMChatClient.chat") as mock_chat:
            mock_resp = MagicMock()
            mock_resp.content = "FINAL: Guido van Rossum"
            mock_resp.raw = {}
            mock_chat.return_value = mock_resp
            record = _process_example(
                example,
                cache_root=cache_root,
                llm_endpoint="http://mock",
                llm_model="mock-model",
                top_k=5,
                force_build=True,
                debug_dir=debug_dir,
                debug_max_notes=10,
            )

    assert record["answer"] == "Guido van Rossum"
    assert (debug_dir / "q_smoke.json").exists()
    stats_path = cache_root / "q_smoke" / "notes_stats.json"
    assert stats_path.exists()
