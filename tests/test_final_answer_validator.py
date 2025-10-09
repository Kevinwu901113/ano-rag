"""Unit tests for the final answer validator."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validators.final_answer_validator import validate_final_answer


def test_validator_ok():
    ctx = ["Alice married Bob in 1999.", "Steve Hillage is a musician."]
    raw = r'''{
      "disambiguation": "Green performer refers to Steve Hillage [L2]",
      "evidence_spans": ["[L2] Steve Hillage is a musician."],
      "reason": "The context states Steve Hillage is the performer.",
      "answer": "Steve Hillage",
      "used_candidate": true
    }'''
    ok, obj, rep = validate_final_answer(raw, ctx, "Q", "Steve Hillage")
    assert ok
    assert rep["valid_span_count"] == 1
    assert obj["used_candidate"] is True


def test_validator_reject_no_spans():
    ctx = ["Phylicia Rashād is the former spouse of Victor Willis."]
    raw = r'''{"disambiguation":"x","evidence_spans":[],"reason":"x","answer":"Victor Willis","used_candidate":false}'''
    ok, _, rep = validate_final_answer(raw, ctx, "Q", None)
    assert not ok
    assert rep["error"] == "no_valid_evidence_spans"


def test_used_candidate_must_be_supported():
    ctx = ["Grant Green recorded an album in 1971."]
    raw = r'''{
      "disambiguation":"x",
      "evidence_spans":["[L1] Grant Green recorded an album in 1971."],
      "reason":"x",
      "answer":"Grant Green",
      "used_candidate": true
    }'''
    ok, _, rep = validate_final_answer(raw, ctx, "Q", "Steve Hillage")
    assert not ok
    assert rep["error"] == "used_candidate_not_supported_by_spans"
