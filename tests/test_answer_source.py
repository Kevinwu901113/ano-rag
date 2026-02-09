from relrag.utils.answer_source import resolve_short_answer


def test_llm_final_preferred_over_structured_answer() -> None:
    short_answer, source, detail = resolve_short_answer(
        structured_answer="Boston",
        llm_raw="FINAL: Cambridge",
        question="Where did he convene?",
    )
    assert short_answer == "Cambridge"
    assert source == "llm_final"
    assert detail.get("source") == "llm_raw:FINAL"
    assert detail.get("llm_has_final") is True


def test_structured_answer_used_when_llm_has_no_final() -> None:
    short_answer, source, detail = resolve_short_answer(
        structured_answer="Boston",
        llm_raw="Answer: Cambridge",
        question="Where did he convene?",
    )
    assert short_answer == "Boston"
    assert source == "structured_answer"
    assert detail.get("source") == "intermediate.structured_answer"
