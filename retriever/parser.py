import re

from .ir import ConstraintIR


def parse_question(question: str) -> ConstraintIR:
    normalized = question.strip().lower()

    spouse_match = re.search(r"spouse of (?:the )?(?P<entity>.+?) performer", normalized)
    if spouse_match:
        alias = spouse_match.group("entity").strip()
        return ConstraintIR(
            variables=["X", "Y", "Z"],
            constraints=[
                {"bind": {"var": "X", "alias": alias, "types": ["WORK", "PERSON"]}},
                {"edge": {"from": "X", "pred": "performed_by", "to": "Y"}},
                {"edge": {"from": "Y", "pred": "spouse", "to": "Z"}},
            ],
            ask={"target": "Z", "kind": "who"},
        )

    author_match = re.search(r"who wrote (?P<work>.+)", normalized)
    if author_match:
        alias = author_match.group("work")
        return ConstraintIR(
            variables=["X", "Y"],
            constraints=[
                {"bind": {"var": "X", "alias": alias, "types": ["WORK"]}},
                {"edge": {"from": "X", "pred": "authored_by", "to": "Y"}},
            ],
            ask={"target": "Y", "kind": "who"},
        )

    birth_match = re.search(r"where was (?P<person>.+?) born", normalized)
    if birth_match:
        alias = birth_match.group("person")
        return ConstraintIR(
            variables=["X", "Y"],
            constraints=[
                {"bind": {"var": "X", "alias": alias, "types": ["PERSON"]}},
                {"edge": {"from": "X", "pred": "born_in", "to": "Y"}},
            ],
            ask={"target": "Y", "kind": "where"},
        )

    return ConstraintIR(variables=[], constraints=[], ask={"target": None})
