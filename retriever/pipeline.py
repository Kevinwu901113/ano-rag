import json
from typing import Any, Dict

from .operators import BIND, EXPAND_from, Indexes
from .parser import parse_question
from .scorer import score_path


def retrieve_answer(
    question: str, indexes_dir: str, notes_path: str
) -> Dict[str, Any]:
    ir = parse_question(question)
    if not ir.constraints:
        return {"answer": None, "path": [], "reason": "No parser rule matched."}

    indexes = Indexes(indexes_dir)

    bind_op = next(c for c in ir.constraints if "bind" in c)["bind"]
    seeds = BIND(indexes, bind_op["alias"], bind_op.get("types", []))

    perf_edge = next(
        c for c in ir.constraints if "edge" in c and c["edge"]["pred"] == "performed_by"
    )["edge"]
    spouse_edge = next(
        c for c in ir.constraints if "edge" in c and c["edge"]["pred"] == "spouse"
    )["edge"]

    best = {"score": -1.0, "answer": None, "path": []}

    for subj in seeds:
        y_candidates = EXPAND_from(indexes, subj, perf_edge["pred"])
        for actor, note_id_1 in y_candidates:
            for spouse, note_id_2 in EXPAND_from(indexes, actor, spouse_edge["pred"]):
                path = [
                    {
                        "subj": subj,
                        "pred": perf_edge["pred"],
                        "obj": actor,
                        "note_id": note_id_1,
                    },
                    {
                        "subj": actor,
                        "pred": spouse_edge["pred"],
                        "obj": spouse,
                        "note_id": note_id_2,
                    },
                ]
                score = score_path(path)
                if score > best["score"]:
                    best = {"score": score, "answer": spouse, "path": path}

    evidence = []
    if best["path"]:
        wanted = {edge["note_id"] for edge in best["path"]}
        with open(notes_path, "r", encoding="utf-8") as handle:
            for line in handle:
                note = json.loads(line)
                if note["note_id"] in wanted:
                    evidence.append(
                        {"note_id": note["note_id"], "evidence": note["evidence"]}
                    )

    best["evidence"] = evidence
    return best
