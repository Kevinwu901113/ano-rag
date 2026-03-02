from baseline.runners.common import (
    normalize_answer_for_eval,
    render_aligned_reader_prompt,
    resolve_effective_reader_params,
)


def test_normalize_answer_handles_think_and_final_line() -> None:
    raw = """<think>reasoning</think>\nFINAL: The Beatles"""
    assert normalize_answer_for_eval(raw) == "The Beatles"


def test_normalize_answer_handles_unclosed_think_block() -> None:
    raw = "FINAL: Boston\n<think>I should reason further"
    assert normalize_answer_for_eval(raw) == "Boston"


def test_render_prompt_includes_final_instruction_text() -> None:
    prompt = render_aligned_reader_prompt(
        question="Who wrote Hamlet?",
        evidence_rows=[{"id": "c1", "title": "Hamlet", "text": "William Shakespeare wrote Hamlet.", "rank": 1}],
    )
    assert "FINAL:" in prompt
    assert "Output your step-by-step reasoning first" not in prompt


def test_resolve_effective_reader_params_has_required_keys() -> None:
    params = resolve_effective_reader_params(
        dataset="hotpotqa",
        backend="qwen",
        answer_max_tokens=None,
        temperature=None,
        config_path="relrag/config/config.yaml",
    )
    assert set(params.keys()) == {"answer_max_tokens", "temperature", "source"}
    assert isinstance(params["answer_max_tokens"], int)
    assert isinstance(params["temperature"], float)


def test_resolve_effective_reader_params_enforces_deepseek_token_floor() -> None:
    params = resolve_effective_reader_params(
        dataset="hotpotqa",
        backend="deepseek",
        answer_max_tokens=None,
        temperature=None,
        config_path="this/path/does/not/exist.yaml",
    )
    assert params["answer_max_tokens"] >= 512
