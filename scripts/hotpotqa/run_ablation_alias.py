#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_DATA = "data/hotpot_dev_distractor_500_jsonl.jsonl"
DEFAULT_CONFIG = "relrag/config/config.yaml"
DEFAULT_OUTPUT_ROOT = "result/hotpot_ablation_alias"

def _run_cmd(cmd: List[str], cwd: Path) -> None:
    # Set capture_output=False to allow child process stdout/stderr (e.g. tqdm) to be shown
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=False, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")

def _read_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def main() -> None:
    parser = argparse.ArgumentParser(description="Run HotpotQA Alias Binding Ablation Experiment (G0-G3)")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="hotpot_entry config")
    parser.add_argument("--data", default=DEFAULT_DATA, help="HotpotQA jsonl dataset")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT, help="Output root for experiments")
    parser.add_argument("--cache_dir", help="Retriever cache dir (default: <output_root>/cache)")
    parser.add_argument("--reader", default="vllm", help="Reader backend (vllm or openai)")
    parser.add_argument("--retriever", default="structured", help="Retriever mode (must be structured or hybrid)")
    parser.add_argument("--workers", type=int, default=1, help="Workers per run")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit")
    parser.add_argument("--top_k", type=int, default=10, help="Retrieval top_k")
    parser.add_argument("--skip_run", action="store_true", help="Skip runs, only summarize")
    parser.add_argument("--retrieval_only", action="store_true", help="Skip answer generation (faster)")
    parser.add_argument("--force_build", action="store_true", help="Force rebuild indexes")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    data_path = (repo_root / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    output_root = (repo_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    if args.cache_dir:
        cache_dir = (repo_root / args.cache_dir).resolve() if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir).resolve()
    else:
        cache_dir = output_root / "cache"
    
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Define Ablation Groups
    groups = [
        {
            "name": "G0_Full",
            "use_alias_binding": "true",
            "use_alias_lookup": "true",
        },
        {
            "name": "G1_NoAlias",
            "use_alias_binding": "false",
            "use_alias_lookup": "false",
        },
        {
            "name": "G2_SeedOnly",
            "use_alias_binding": "true",
            "use_alias_lookup": "false",
        },
        {
            "name": "G3_ScoreOnly",
            "use_alias_binding": "false",
            "use_alias_lookup": "true",
        },
    ]

    summary_rows = []

    for idx, group in enumerate(groups):
        run_dir = output_root / group["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        
        if not args.skip_run:
            print(f"\n[{time.strftime('%H:%M:%S')}] Step {idx+1}/{len(groups)}: Running {group['name']} ...")
            print(f"  Config: use_alias_binding={group['use_alias_binding']}, use_alias_lookup={group['use_alias_lookup']}")
            
            cmd = [
                args.python,
                str(repo_root / "hotpot_entry.py"),
                "--config", str(config_path),
                "--data", str(data_path),
                "--cache_dir", str(cache_dir),
                "--output_dir", str(run_dir),
                "--reader", args.reader,
                "--retriever", args.retriever,
                "--top_k", str(args.top_k),
                "--use_alias_binding", group["use_alias_binding"],
                "--use_alias_lookup", group["use_alias_lookup"],
                "--workers", str(args.workers),
            ]
            if args.limit > 0:
                cmd.extend(["--limit", str(args.limit)])
            if args.retrieval_only:
                cmd.extend(["--retrieval_only", "true", "--disable_retriever_llm", "true"])
            if args.force_build:
                cmd.append("--force_build")
            
            start_t = time.time()
            try:
                _run_cmd(cmd, cwd=repo_root)
                print(f"  > {group['name']} completed in {time.time() - start_t:.1f}s")
            except Exception as e:
                print(f"[error] {group['name']} failed: {e}")
                continue

        # Collect metrics
        metrics_file = list(run_dir.glob("metrics*.json"))
        if not metrics_file:
            print(f"[warn] No metrics file found for {group['name']}")
            continue
        
        metrics = _read_optional_json(metrics_file[0])
        stats = metrics.get("stats", {})
        retrieval_stats = metrics.get("retrieval_stats", {})

        # Try to find alignment metrics (Official EM/F1)
        align_dirs = list(run_dir.glob("align_*"))
        official_metrics = {}
        if align_dirs:
            manifest_path = align_dirs[0] / "alignment_manifest.json"
            if manifest_path.exists():
                manifest = _read_optional_json(manifest_path)
                official_metrics = manifest.get("metrics", {})

        # Calculate ablation metrics from JSONL
        pred_file = list(run_dir.glob("pred_*.jsonl"))
        ablation_stats = {
            "SeedBindRate": 0.0,
            "AliasHitRate": 0.0,
            "StructuredHitRate": 0.0,
            "FallbackRate": 0.0,
            "count": 0
        }
        if pred_file:
            try:
                with open(pred_file[0], "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        ablation_stats["count"] += 1
                        
                        # Extract meta from intermediate.retrieve_result.meta.ablation
                        inter = record.get("intermediate", {})
                        retr_res = inter.get("retrieve_result", {})
                        meta = retr_res.get("meta", {}).get("ablation", {})
                        
                        if meta.get("seed_entities"): # Non-empty list
                            ablation_stats["SeedBindRate"] += 1
                        bind_alias_hit = bool(meta.get("bind_alias_matched"))
                        score_alias_hit = int(meta.get("alias_lookup_hit_count") or 0) > 0
                        # backward compatibility for old artifacts
                        legacy_alias_hit = bool(meta.get("bind_used_alias_index")) and "bind_alias_matched" not in meta
                        if bind_alias_hit or score_alias_hit or legacy_alias_hit:
                            ablation_stats["AliasHitRate"] += 1
                        if meta.get("structured_hit"):
                            ablation_stats["StructuredHitRate"] += 1
                        
                        # FallbackRate: check fallback status
                        fallback = retr_res.get("fallback", {})
                        if fallback.get("status") != "structured_hit":
                            ablation_stats["FallbackRate"] += 1
            except Exception as e:
                print(f"[warn] Failed to parse {pred_file[0]}: {e}")
            
            if ablation_stats["count"] > 0:
                for k in ["SeedBindRate", "AliasHitRate", "StructuredHitRate", "FallbackRate"]:
                    ablation_stats[k] = round(ablation_stats[k] / ablation_stats["count"], 4)

        row = {
            "Group": group["name"],
            "EM": official_metrics.get("em", stats.get("bleu1", 0.0)),
            "F1": official_metrics.get("f1", stats.get("bleu4", 0.0)),
            "Recall": retrieval_stats.get("top_k_final_mean", 0.0),
        }
        # Add stats
        row.update(stats)
        row.update(retrieval_stats)
        row.update(official_metrics)
        row.update(ablation_stats)
        summary_rows.append(row)

    # Print Summary
    print("\n=== Ablation Summary ===")
    headers = ["Group", "EM", "F1", "SeedBindRate", "AliasHitRate", "StructuredHitRate", "FallbackRate"]
    print("\t".join(headers))
    for row in summary_rows:
        print("\t".join(str(row.get(h, "")) for h in headers))

    summary_path = output_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"\nFull summary written to {summary_path}")

if __name__ == "__main__":
    main()
