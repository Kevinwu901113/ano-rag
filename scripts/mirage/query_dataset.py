#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import List

from loguru import logger
from baselines.direct_llm import DirectLLMRunner


def _list_workspaces(root: Path, dataset: str) -> List[Path]:
    if not root.exists():
        return []
    pattern = re.compile(r"^(?P<idx>\d{3})-(?P<name>.+)$")
    candidates = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if match and match.group("name") == dataset:
            candidates.append((int(match.group("idx")), entry))
    candidates.sort(key=lambda x: x[0])
    return [entry for _, entry in candidates]


def _select_workspace(root: Path, dataset: str, new: bool) -> Path:
    workspaces = _list_workspaces(root, dataset)
    if new or not workspaces:
        next_idx = workspaces[-1].name.split("-")[0] if workspaces else "-1"
        next_val = int(next_idx) + 1
        name = f"{next_val:03d}-{dataset}"
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        return target
    return workspaces[-1]


def _strip_reasoning(answer: str) -> str:
    text = answer
    while True:
        start = text.find("<think>")
        if start == -1:
            break
        end = text.find("</think>", start + 7)
        if end == -1:
            text = text[:start] + text[start + 7 :]
            break
        text = text[:start] + text[end + len("</think>") :]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LM Studio answers for MIRAGE dataset")
    parser.add_argument("--dataset", default="mirage", help="Dataset name (used for workspace naming)")
    parser.add_argument("--dataset-path", default=None, help="Path to dataset.json")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--work-dir", default=None, help="Explicit workspace path")
    parser.add_argument("--indexes-dir", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--lmstudio-endpoint", required=True)
    parser.add_argument("--lmstudio-model", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--qa-log", default=None, help="Optional plain text QA log path (question \t answer)")
    parser.add_argument("--new", action="store_true", help="Force create a new workspace copy")
    parser.add_argument("--direct-llm", action="store_true", help="Skip retrieval and answer directly with LLM")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override LLM max new tokens")
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_path = Path(args.dataset_path) if args.dataset_path else Path(f"data/{dataset_name}_sample/dataset.json")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset JSON not found at {dataset_path}")

    result_root = Path(args.result_root)
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        result_root.mkdir(parents=True, exist_ok=True)
        work_dir = _select_workspace(result_root, dataset_name, args.new)
    logger.info("Using workspace: {}", work_dir)

    override_cfg = work_dir / "config.override.yaml"
    if override_cfg.exists():
        os.environ["ANO_RAG_CONFIG"] = str(override_cfg)

    output_path = Path(args.out) if args.out else work_dir / "answers.json"
    qa_log_path = Path(args.qa_log) if args.qa_log else work_dir / "qa.tsv"

    with open(dataset_path, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    if args.limit > 0:
        dataset = dataset[: args.limit]

    if args.direct_llm:
        runner = DirectLLMRunner(
            lm_endpoint=args.lmstudio_endpoint,
            lm_model=args.lmstudio_model,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        artifacts = runner.run_dataset(dataset, work_dir=str(work_dir))
        logger.info("Direct LLM baseline complete: {}", artifacts.get("qa"))
        if output_path and Path(artifacts["answers_json"]) != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(Path(artifacts["answers_json"]).read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("Copied answers.json to {}", output_path)
        if qa_log_path and Path(artifacts["qa"]) != qa_log_path:
            qa_log_path.parent.mkdir(parents=True, exist_ok=True)
            qa_log_path.write_text(Path(artifacts["qa"]).read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("Copied QA log to {}", qa_log_path)
        return

    indexes_dir = Path(args.indexes_dir) if args.indexes_dir else work_dir / "indexes"
    notes_path = Path(args.notes) if args.notes else work_dir / "notes" / f"notes.{dataset_name}.jsonl"
    if not indexes_dir.exists():
        raise FileNotFoundError(f"Indexes dir not found: {indexes_dir}")
    if not notes_path.exists():
        raise FileNotFoundError(f"Notes file not found: {notes_path}")

    from query.query_processor import QueryProcessor

    qp = QueryProcessor(
        indexes_dir=str(indexes_dir),
        notes_path=str(notes_path),
        lmstudio_endpoint=args.lmstudio_endpoint,
        lmstudio_model=args.lmstudio_model,
    )

    results = []
    qa_lines: List[str] = []
    for item in dataset:
        question = item.get("query") or item.get("question")
        if not question:
            continue
        attr_hint = "occupation" if "occupation" in question.lower() else None
        qid = item.get("query_id")
        doc_hint = None
        if qid:
            doc_hint = f"{dataset_name}/{qid}"
        res = qp.process(question, doc_hint=doc_hint, attribute_hint=attr_hint)
        results.append(
            {
                "query_id": item.get("query_id"),
                "question": question,
                "answer": res.get("answer"),
                "structured": res.get("structured"),
                "decision": res.get("decision"),
            }
        )
        answer_text = res.get("answer")
        if answer_text is None:
            clean_answer = ""
        else:
            stripped = _strip_reasoning(str(answer_text))
            clean_answer = " ".join(stripped.splitlines())
        qa_lines.append(f"{question}\t{clean_answer}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    logger.info("Wrote {} answers to {}", len(results), output_path)

    if qa_log_path:
        qa_log_path.parent.mkdir(parents=True, exist_ok=True)
        qa_log_path.write_text("\n".join(qa_lines), encoding="utf-8")
        logger.info("Wrote QA log to {}", qa_log_path)


if __name__ == "__main__":
    main()
