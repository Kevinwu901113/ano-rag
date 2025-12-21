from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load MuSiQue dataset from .jsonl or .json.

    - .jsonl: one JSON object per line.
    - .json: list, or dict containing list under data/examples/questions.
    """
    dataset_path = Path(path)
    if dataset_path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("data") or data.get("examples") or data.get("questions") or []
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list or a dict containing list under data/examples/questions")
    return data


def save_musique_results_and_qa(
    work_dir: Path,
    results: List[Dict[str, Any]],
    qa_rows: List[Tuple[str, str]],
    *,
    output_path: Optional[Path] = None,
    qa_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Save MuSiQue official results.jsonl and qa.tsv."""
    work_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = work_dir / "preds"
    out_path = Path(output_path) if output_path else preds_dir / "musique_results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    qa_file = Path(qa_path) if qa_path else preds_dir / "qa.tsv"
    qa_file.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for question, answer in qa_rows:
        q = " ".join((question or "").replace("\t", " ").split())
        a = " ".join((answer or "").replace("\t", " ").split())
        lines.append(f"{q}\t{a}")
    qa_file.write_text("\n".join(lines), encoding="utf-8")
    return out_path, qa_file
