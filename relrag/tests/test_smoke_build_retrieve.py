import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from relrag.api import build_index, retrieve, answer
from relrag.config.config_loader import config as global_config


DUMMY_NOTES = [
    {
        "subj": "Python",
        "pred": "authored_by",
        "obj": "Guido van Rossum",
        "subj_type": "WORK",
        "obj_type": "PERSON",
        "evidence": "Guido van Rossum wrote Python.",
        "meta": {
            "source": "doc_1#c0000",
            "confidence": 0.9,
            "subject_profile": {"type": "WORK", "aliases": []},
            "attribute": {
                "name": "authored_by",
                "values": [{"value": "Guido van Rossum", "evidence": "Guido van Rossum wrote Python."}],
            },
        },
    }
]


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


@pytest.fixture
def workspace(tmp_path):
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    docs_path = fixtures_dir / "docs.jsonl"
    if not docs_path.exists():
        docs_path.write_text(
            json.dumps({"id": "doc_1", "text": "Test doc", "title": "Test"}) + "\n",
            encoding="utf-8",
        )
    shutil.copy(docs_path, tmp_path / "docs.jsonl")
    return tmp_path


@pytest.fixture
def disable_hybrid_config():
    cfg = global_config.load_config()
    retriever_cfg = cfg.setdefault("retriever", {})
    structured_cfg = retriever_cfg.setdefault("structured", {})
    embedding_cfg = retriever_cfg.setdefault("embedding", {})
    bm25_cfg = retriever_cfg.setdefault("bm25", {})
    reranker_cfg = cfg.setdefault("reranker", {})

    original = {
        "embedding_enabled": embedding_cfg.get("enabled"),
        "bm25_enabled": bm25_cfg.get("enabled"),
        "reranker_enabled": reranker_cfg.get("enabled"),
        "vector_fallback_enabled": structured_cfg.get("vector_fallback_enabled"),
    }

    embedding_cfg["enabled"] = False
    bm25_cfg["enabled"] = False
    reranker_cfg["enabled"] = False
    structured_cfg["vector_fallback_enabled"] = False
    yield

    embedding_cfg["enabled"] = original["embedding_enabled"]
    bm25_cfg["enabled"] = original["bm25_enabled"]
    reranker_cfg["enabled"] = original["reranker_enabled"]
    structured_cfg["vector_fallback_enabled"] = original["vector_fallback_enabled"]


def test_smoke_build_retrieve(disable_hybrid_config, workspace):
    docs_input = str(workspace / "docs.jsonl")
    output_dir = str(workspace / "output")

    payload = _dummy_chat_payload(json.dumps(DUMMY_NOTES))

    def _mock_post_chat(*_args, **_kwargs):
        return DummyResponse(payload)

    with patch("relrag.utils.llm_client.LLMChatClient.post_chat", side_effect=_mock_post_chat):
        build_index(
            docs_input=docs_input,
            output_dir=output_dir,
            llm_endpoint="http://mock",
            llm_model="mock-model",
        )

    out_path = Path(output_dir)
    assert (out_path / "notes.jsonl").exists()
    assert (out_path / "indexes").exists()
    assert (out_path / "indexes" / "entity_to_notes.json").exists()
    assert (out_path / "indexes" / "predicate_to_notes.json").exists()
    assert (out_path / "indexes" / "graph_edges.jsonl").exists()
    assert (out_path / "indexes" / "inverse_edges.jsonl").exists()
    stats_path = out_path / "notes_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert "raw_generated" in stats
    assert "validated" in stats
    assert "written" in stats

    notes_path = str(out_path / "notes.jsonl")
    index_dir = str(out_path / "indexes")
    result = retrieve(
        question="Who wrote Python?",
        index_dir=index_dir,
        notes_path=notes_path,
        top_k=5,
    )

    assert "paths" in result
    assert "evidence" in result
    assert result["evidence"]
    assert result.get("answer") == "Guido van Rossum"

    with patch("relrag.utils.llm_client.LLMChatClient.chat") as mock_chat_answer:
        mock_resp = MagicMock()
        mock_resp.content = "FINAL: Guido van Rossum"
        mock_resp.raw = {}
        mock_chat_answer.return_value = mock_resp

        ans = answer(
            question="Who wrote Python?",
            evidences=result["evidence"],
            llm_endpoint="http://mock",
            llm_model="mock-model",
        )

    assert "Guido van Rossum" in ans
