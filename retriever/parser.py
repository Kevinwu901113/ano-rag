from .ir import ConstraintIR


def parse_question(question: str) -> ConstraintIR:
    normalized = question.strip().lower()

    if "spouse of the" in normalized and "performer" in normalized:
        alias = (
            normalized.split("spouse of the", 1)[1].split("performer", 1)[0].strip()
        )
        return ConstraintIR(
            variables=["X", "Y", "Z"],
            constraints=[
                {"bind": {"var": "X", "alias": alias, "types": ["WORK", "PERSON"]}},
                {"edge": {"from": "X", "pred": "performed_by", "to": "Y"}},
                {"edge": {"from": "Y", "pred": "spouse", "to": "Z"}},
            ],
            ask={"target": "Z", "kind": "who"},
        )

    return ConstraintIR(variables=[], constraints=[], ask={"target": None})
