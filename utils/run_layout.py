from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Optional


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_workdir(
    workdir: Optional[str],
    *,
    result_root: Optional[str],
    dataset: str,
    run_id: Optional[str] = None,
) -> Path:
    """
    Resolve a dataset workdir. If workdir is provided, use it; otherwise place the
    dataset under result_root/run_<timestamp>/<dataset>.
    """
    if workdir:
        return Path(workdir)
    base = Path(result_root or "result_relrag")
    run_name = f"run_{run_id or _timestamp()}"
    return base / run_name / dataset


def ensure_workdir_layout(work_dir: Path) -> Dict[str, Path]:
    """
    Ensure standard layout under work_dir and return subdir paths.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts = work_dir / "artifacts"
    preds = work_dir / "preds"
    metrics = work_dir / "metrics"
    artifacts.mkdir(parents=True, exist_ok=True)
    preds.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    cache_dir = artifacts / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    return {"workdir": work_dir, "artifacts": artifacts, "preds": preds, "metrics": metrics}
