import argparse
import json
import random
import subprocess
import sys
import shutil
from pathlib import Path
from typing import List, Dict

def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(data: List[Dict], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def run_cmd(cmd: List[str]):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/hotpot_dev_distractor_500_jsonl.jsonl")
    parser.add_argument("--output_root", default="result/ablation_study")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    
    # 1. Prepare Data Splits
    full_data = load_jsonl(Path(args.data))
    # Ensure consistent order before split
    full_data.sort(key=lambda x: x["_id"])
    
    # Stratified split by "type" (comparison vs bridge) if available
    # HotpotQA has "type" field
    
    comparison = [x for x in full_data if x.get("type") == "comparison"]
    bridge = [x for x in full_data if x.get("type") == "bridge"]
    others = [x for x in full_data if x.get("type") not in ("comparison", "bridge")]
    
    # Shuffle with fixed seed
    random.seed(42)
    random.shuffle(comparison)
    random.shuffle(bridge)
    random.shuffle(others)
    
    splits = [[] for _ in range(args.folds)]
    
    def distribute(items):
        for i, item in enumerate(items):
            splits[i % args.folds].append(item)
            
    distribute(comparison)
    distribute(bridge)
    distribute(others)
    
    split_paths = []
    for i, split_data in enumerate(splits):
        p = root / f"split_{i}.jsonl"
        save_jsonl(split_data, p)
        split_paths.append(p)
        print(f"Fold {i}: {len(split_data)} samples")
        
    # 2. Run Experiments
    # Methods: sentence, fixed
    methods = [
        {"name": "sentence", "args": ["--chunking_method", "sentence"]},
        {"name": "fixed", "args": ["--chunking_method", "fixed", "--chunking_size", "256", "--chunking_overlap", "32"]}
    ]
    
    final_preds = {m["name"]: [] for m in methods}
    
    for fold_idx, data_path in enumerate(split_paths):
        print(f"=== Processing Fold {fold_idx} ===")
        for method in methods:
            m_name = method["name"]
            run_dir = root / f"fold_{fold_idx}" / m_name
            
            # Check if done
            # hotpot_entry.py generates filenames like pred_dev_vllm_hybrid.jsonl
            # We look for any jsonl file that starts with pred_
            found_preds = list(run_dir.glob("pred_*.jsonl"))
            pred_file = found_preds[0] if found_preds else None
            
            if pred_file and pred_file.exists():
                print(f"Found existing results: {pred_file}")
                final_preds[m_name].append(pred_file)
            else:
                cmd = [
                    sys.executable, "hotpot_entry.py",
                    "--config", "relrag/config/exp_ablation_vllm.yaml", # Use vLLM-only config
                    "--data", str(data_path),
                    "--output_dir", str(run_dir),
                    "--workers", str(args.workers),
                    "--force_build", # Important: force rebuild chunks for each method!
                    "--disable_retriever_llm", "true"
                ] + method["args"]
                
                # We need to make sure they don't share cache if chunking is different
                # hotpot_entry.py uses --cache_dir. Default is result/cache.
                # If we share cache, the "docs/chunks.jsonl" might be reused.
                # We MUST use separate cache dirs.
                cmd.extend(["--cache_dir", str(run_dir / "cache")])
                
                try:
                    run_cmd(cmd)
                except subprocess.CalledProcessError as e:
                    print(f"Error running {m_name} fold {fold_idx}: {e}")
                    continue
            
            # Collect results
            found_preds = list(run_dir.glob("pred_*.jsonl"))
            if found_preds:
                final_preds[m_name].append(found_preds[0])
    
    # 3. Aggregate Results
    for m_name, files in final_preds.items():
        combined = []
        for f in files:
            combined.extend(load_jsonl(f))
        
        out_f = root / f"all_preds_{m_name}.jsonl"
        save_jsonl(combined, out_f)
        print(f"Aggregated {len(combined)} predictions for {m_name}")

    # 4. Evaluate
    print("=== Running Evaluation ===")
    eval_cmd = [
        sys.executable, "scripts/hotpotqa/eval_ablation.py",
        "--baseline", str(root / "all_preds_fixed.jsonl"),
        "--experiment", str(root / "all_preds_sentence.jsonl"),
        "--output_dir", str(root / "evaluation_report")
    ]
    run_cmd(eval_cmd)

if __name__ == "__main__":
    main()
