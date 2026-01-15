from relrag.api import answer


def test_between_compare_solver_uses_counts() -> None:
    evidences = [
        {"subj": "Greyia", "pred": "has_species_count", "obj": "3", "evidence": "Greyia contains three species."},
        {"subj": "Calibanus", "pred": "has_species_count", "obj": "2", "evidence": "Calibanus is a genus of two species."},
    ]
    response = answer(
        question="Between Greyia and Calibanus, which genus contains more species?",
        evidences=evidences,
        llm_endpoint="http://mock",
        llm_model="mock-model",
    )
    assert "Greyia" in response
