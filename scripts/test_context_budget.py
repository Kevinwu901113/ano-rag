from copy import deepcopy
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relrag.config.config_loader import config as global_config
from relrag.utils.context_budget import budget_answer_prompt, budget_rerank_prompt


def _long_text(seed: str, repeat: int) -> str:
    return (seed + " ") * repeat


def _assert_budget(report, safety_margin: int) -> None:
    total = report.estimated_input_tokens + report.effective_max_tokens + safety_margin
    assert total <= report.model_ctx_len, f"budget overflow: {total} > {report.model_ctx_len}"


def test_rerank_budget() -> None:
    cfg = deepcopy(global_config.load_config())
    cfg.setdefault("llm", {})["max_context_len"] = 512
    cfg.setdefault("llm", {})["safety_margin_tokens"] = 64
    cfg.setdefault("rerank", {})["max_candidates"] = 12
    cfg.setdefault("rerank", {})["max_candidate_tokens"] = 64
    question = _long_text("rerank question", 50)
    long_evidence = _long_text("evidence", 400)
    candidates = []
    for idx in range(20):
        note_id = f"note_{idx}"
        candidates.append({"note_id": note_id, "note": {"note_id": note_id, "evidence": long_evidence}})
    budgeted = budget_rerank_prompt(
        question,
        candidates,
        prompt_name="rerank.txt",
        cfg=cfg,
        llm_cfg={"max_context_len": 512},
        requested_max_tokens=64,
    )
    _assert_budget(budgeted.report, cfg["llm"]["safety_margin_tokens"])
    assert budgeted.report.items_count <= cfg["rerank"]["max_candidates"]


def test_answer_budget() -> None:
    cfg = deepcopy(global_config.load_config())
    cfg.setdefault("llm", {})["max_context_len"] = 512
    cfg.setdefault("llm", {})["safety_margin_tokens"] = 64
    cfg.setdefault("answer", {})["max_evidence_items"] = 8
    cfg.setdefault("answer", {})["max_evidence_tokens"] = 64
    question = _long_text("answer question", 50)
    long_evidence = _long_text("evidence", 400)
    evidences = []
    for idx in range(12):
        evidences.append(
            {
                "note_id": f"note_{idx}",
                "canonical": long_evidence,
                "evidence": long_evidence,
                "score": 0.5,
            }
        )
    budgeted = budget_answer_prompt(
        question,
        evidences,
        prompt_name="answerer.txt",
        label_instruction="If you can answer, output the canonical label only.",
        system_prompt="",
        cfg=cfg,
        llm_cfg={"max_context_len": 512},
        requested_max_tokens=128,
    )
    _assert_budget(budgeted.report, cfg["llm"]["safety_margin_tokens"])
    assert budgeted.report.items_count <= cfg["answer"]["max_evidence_items"]


if __name__ == "__main__":
    test_rerank_budget()
    test_answer_budget()
    print("context budget tests passed")
