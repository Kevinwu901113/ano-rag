from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

from scripts.ultradomain.common import (
    DOMAIN_LABELS,
    OUTPUT_ROOT,
    RUN_META_DIR,
    ensure_dirs,
    normalize_domain,
    now_iso,
    ultradomain_get,
    write_json,
    write_jsonl,
)


def _is_column_mismatch_error(exc: Exception) -> bool:
    msg = str(exc)
    cls = exc.__class__.__name__
    keywords = [
        "DatasetGenerationCastError",
        "CastError",
        "column names don't match",
        "new columns",
        "missing columns",
    ]
    return any(token in msg or token in cls for token in keywords)


def _iter_local_json_records(path: Path, *, domain_hint: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    if domain_hint and "__domain_hint" not in row:
                        row["__domain_hint"] = domain_hint
                    yield row
        return
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError:
                return
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    if domain_hint and "__domain_hint" not in row:
                        row["__domain_hint"] = domain_hint
                    yield row
        elif isinstance(payload, dict):
            rows = payload.get("data") or payload.get("rows") or payload.get("items")
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        if domain_hint and "__domain_hint" not in row:
                            row["__domain_hint"] = domain_hint
                        yield row


def _pick_repo_files(
    files: List[str],
    split: str,
    config_name: str | None,
    domain: str,
) -> List[str]:
    json_files = [f for f in files if f.lower().endswith((".jsonl", ".json"))]
    if not json_files:
        return []

    selected = json_files
    if config_name:
        cfg = str(config_name).strip().lower()
        cfg_files = [
            f
            for f in selected
            if cfg in Path(f).stem.lower() or f"/{cfg}/" in f.lower() or f"_{cfg}." in f.lower()
        ]
        if cfg_files:
            selected = cfg_files

    split_key = str(split or "").strip().lower()
    if split_key and split_key not in {"all", "*"}:
        split_files = [
            f
            for f in selected
            if split_key in Path(f).stem.lower() or f"/{split_key}/" in f.lower() or f"_{split_key}." in f.lower()
        ]
        if split_files:
            selected = split_files

    domain_arg = str(domain or "all").strip().lower()
    target_tokens: List[str]
    if domain_arg == "all":
        target_tokens = ["mix", "legal"]
    elif domain_arg in {"mix", "mixed"}:
        target_tokens = ["mix"]
    elif domain_arg == "legal":
        target_tokens = ["legal"]
    else:
        target_tokens = []

    # Bias toward selected target domain files if available.
    domain_files = [f for f in selected if any(token in Path(f).stem.lower() for token in target_tokens)]
    if domain_files:
        selected = domain_files

    return sorted(selected)


def _load_dataset_via_repo_files(
    dataset_id: str,
    split: str,
    cache_dir: str | None,
    config_name: str | None,
    domain: str,
) -> Iterable[Dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for fallback dataset loading.") from exc

    files: List[str] = []
    local_snapshot_root: Optional[Path] = None
    try:
        files = list_repo_files(dataset_id, repo_type="dataset")
    except Exception as exc:
        local_snapshot_root = _latest_cached_snapshot_dir(dataset_id)
        if local_snapshot_root is None:
            raise RuntimeError(
                f"list_repo_files failed for dataset={dataset_id} and no local snapshot is available."
            ) from exc
        print(
            "[prepare_docs] list_repo_files unavailable; "
            f"using local snapshot at {local_snapshot_root}."
        )
        files = [
            str(path.relative_to(local_snapshot_root))
            for path in local_snapshot_root.rglob("*")
            if path.is_file()
        ]

    picked = _pick_repo_files(files, split=split, config_name=config_name, domain=domain)
    if not picked:
        raise RuntimeError(f"No JSON/JSONL files found for dataset={dataset_id} (split={split}, config={config_name}).")

    def _iter_rows() -> Iterator[Dict[str, Any]]:
        for rel_path in picked:
            domain_hint = Path(rel_path).stem
            if local_snapshot_root is not None:
                local_path = local_snapshot_root / rel_path
            else:
                local_path = Path(
                    hf_hub_download(
                        repo_id=dataset_id,
                        filename=rel_path,
                        repo_type="dataset",
                        cache_dir=cache_dir,
                    )
                )
            for row in _iter_local_json_records(local_path, domain_hint=domain_hint):
                yield row

    return _iter_rows()


def _latest_cached_snapshot_dir(dataset_id: str) -> Optional[Path]:
    # Example: dataset_id="TommyChien/UltraDomain" -> datasets--TommyChien--UltraDomain
    repo_key = "datasets--" + str(dataset_id).replace("/", "--")
    candidates: List[Path] = []

    env_hub_cache = os.environ.get("HF_HUB_CACHE")
    if env_hub_cache:
        candidates.append(Path(env_hub_cache))
    env_home = os.environ.get("HF_HOME")
    if env_home:
        candidates.append(Path(env_home) / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    for hub_root in candidates:
        snapshots_dir = hub_root / repo_key / "snapshots"
        if not snapshots_dir.exists():
            continue
        snaps = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        if not snaps:
            continue
        snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return snaps[0]
    return None


def _load_dataset(
    dataset_id: str,
    split: str,
    cache_dir: str | None,
    config_name: str | None,
    domain: str,
):
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("datasets is required. Install with: pip install datasets") from exc
    kwargs: Dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if config_name:
        kwargs["name"] = config_name
    load_exc: Optional[Exception] = None
    try:
        return load_dataset(dataset_id, split=split, **kwargs)
    except Exception as exc:
        load_exc = exc
        reason = "column mismatch" if _is_column_mismatch_error(exc) else f"{exc.__class__.__name__}: {exc}"
        print(
            "[prepare_docs] load_dataset failed "
            f"({reason}); trying direct JSON/JSONL fallback via repo files/local snapshot."
        )

    try:
        return _load_dataset_via_repo_files(
            dataset_id=dataset_id,
            split=split,
            cache_dir=cache_dir,
            config_name=config_name,
            domain=domain,
        )
    except Exception as fallback_exc:
        if load_exc is None:
            raise
        raise RuntimeError(
            "Failed to load dataset with both load_dataset and repo-file fallback. "
            f"load_dataset_error={load_exc.__class__.__name__}: {load_exc}; "
            f"fallback_error={fallback_exc.__class__.__name__}: {fallback_exc}"
        ) from load_exc


def _doc_record(row: Dict[str, Any], domain: str, fallback_id: str) -> Dict[str, Any] | None:
    text = row.get("context") or ""
    if not isinstance(text, str) or not text.strip():
        return None
    doc_id = row.get("context_id") or row.get("_id") or fallback_id
    doc_id = str(doc_id)
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    title = meta.get("title") if isinstance(meta.get("title"), str) else None
    return {
        "doc_id": doc_id,
        "title": title,
        "text": text,
        "dataset": domain,
        "meta": meta or {},
    }


def main() -> None:
    dataset_id_default = ultradomain_get("dataset.dataset_id", "TBD_ULTRADOMAIN_DATASET_ID")
    split_default = ultradomain_get("dataset.split", "train")
    config_name_default = ultradomain_get("dataset.config_name", None)
    cache_dir_default = ultradomain_get("dataset.cache_dir", None)
    limit_default = int(ultradomain_get("dataset.limit", 0) or 0)
    domain_default = ultradomain_get("dataset.domain", "all")

    parser = argparse.ArgumentParser(description="Prepare UltraDomain docs for Mix/Legal.")
    parser.add_argument(
        "--dataset_id",
        default=dataset_id_default,
        help="HuggingFace dataset id for UltraDomain (must be set explicitly).",
    )
    parser.add_argument("--split", default=split_default)
    parser.add_argument("--config_name", default=config_name_default)
    parser.add_argument("--cache_dir", default=cache_dir_default)
    parser.add_argument("--limit", type=int, default=limit_default, help="Optional per-domain limit")
    parser.add_argument("--domain", default=domain_default, help="mix|legal|all")
    args = parser.parse_args()

    if str(args.dataset_id).startswith("TBD_") or "TBD" in str(args.dataset_id):
        raise SystemExit("Please provide a real --dataset_id (placeholder TBD_* is not allowed).")

    ensure_dirs()
    dataset = _load_dataset(args.dataset_id, args.split, args.cache_dir, args.config_name, args.domain)
    domain_arg = str(args.domain or "all").strip().lower()
    selected_domains: Set[str]
    if domain_arg == "all":
        selected_domains = set(DOMAIN_LABELS.keys())
    else:
        normalized = normalize_domain(domain_arg)
        if normalized not in DOMAIN_LABELS:
            raise SystemExit(f"Invalid --domain={args.domain}. Use mix|legal|all.")
        selected_domains = {normalized}

    docs_by_domain: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    dup_counts = defaultdict(int)
    counts = defaultdict(int)

    for idx, row in enumerate(dataset):
        domain = None
        for candidate in (
            row.get("label"),
            row.get("dataset"),
            row.get("domain"),
            row.get("__domain_hint"),
        ):
            normalized = normalize_domain(candidate)
            if normalized:
                domain = normalized
                break
        if domain not in selected_domains:
            continue
        if args.limit and counts[domain] >= args.limit:
            continue
        record = _doc_record(row, domain, fallback_id=f"{domain}_{idx}")
        if record is None:
            continue
        doc_id = record["doc_id"]
        if doc_id in docs_by_domain[domain]:
            dup_counts[domain] += 1
            continue
        docs_by_domain[domain][doc_id] = record
        counts[domain] += 1

    stats: Dict[str, Any] = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "config_name": args.config_name,
        "generated_at": now_iso(),
        "domains": {},
    }

    for domain in sorted(selected_domains):
        docs = list(docs_by_domain[domain].values())
        docs.sort(key=lambda item: item["doc_id"])
        out_path = RUN_META_DIR / f"docs_{domain}.jsonl"
        write_jsonl(out_path, docs)
        lengths = [len(doc.get("text") or "") for doc in docs]
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            min_len = min(lengths)
            max_len = max(lengths)
        else:
            avg_len = min_len = max_len = 0
        stats["domains"][domain] = {
            "doc_count": len(docs),
            "avg_chars": round(avg_len, 2),
            "min_chars": min_len,
            "max_chars": max_len,
            "duplicate_docs": dup_counts.get(domain, 0),
            "output": str(out_path),
        }

    write_json(RUN_META_DIR / "doc_stats.json", stats)

    empty_domains = [domain for domain in selected_domains if int((stats["domains"].get(domain) or {}).get("doc_count", 0)) == 0]
    if empty_domains:
        raise SystemExit(
            "No documents found for selected domain(s): "
            + ", ".join(sorted(empty_domains))
            + ". Check dataset split/config and domain mapping."
        )
    print(f"Wrote docs to {RUN_META_DIR}")


if __name__ == "__main__":
    main()
