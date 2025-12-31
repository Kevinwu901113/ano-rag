import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# Configuration
DATASET_MAP = {
    "hotpotqa_sample_200": "data/hotpotqa/dataset_distractor_200.json",
    "musique_sample_200": "data/musique_sample/musique.jsonl",
    "mirage_sample_200": "data/mirage_sample_200/dataset.json",
}

# Dataset type for script selection
DATASET_TYPE = {
    "hotpotqa_sample_200": "hotpotqa",
    "musique_sample_200": "musique",
    "mirage_sample_200": "mirage",
}

def get_runner_command(dataset_key, method, dataset_path, budget, workdir):
    dtype = DATASET_TYPE[dataset_key]
    
    # Base command structure
    cmd = [sys.executable]
    
    # Force embed model environment variable or args if possible, but runners load from config.
    # We will update the config loader via CLI override if supported, or ensure config.yaml is correct.
    # Most runners use config/config_loader.py which reads from config/config.yaml.
    # RelRAG accepts --embed-model.
    # VanillaRAG and others read from config.
    # We can pass an env var override if config loader supports it, or assume config.yaml is set.
    # But user asked to ensure it in run_experiment1.py.
    # Let's add an environment variable override for the subprocess.
    env = os.environ.copy()
    env["RAG_EMBED_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Determine script path and extra args
    script_path = ""
    extra_args = []
    
    # Common arguments
    common_args = [
        "--workdir", str(workdir),
        "--context-budget", str(budget),
    ]
    
    if dtype == "hotpotqa":
        common_args.extend(["--dataset", str(dataset_path)])
        common_args.extend(["--embed-device", "auto"])
        base_dir = "scripts/hotpotqa/baselines"
        if method in ["bm25", "dense", "hybrid"]:
            script_path = f"{base_dir}/run_vanilla_rag.py"
            extra_args = ["--retriever", method]
        elif method == "raptor":
            script_path = f"{base_dir}/run_raptor.py"
        elif method == "selfrag":
            script_path = f"{base_dir}/run_selfrag.py"
        elif method == "relrag_full":
            script_path = f"{base_dir}/run_relrag.py"
            extra_args = ["--embed-model", "sentence-transformers/all-MiniLM-L6-v2", "--embed-device", "cpu"]
        elif method == "lightrag":
            script_path = f"{base_dir}/run_graphrag.py"
            
    elif dtype == "musique":
        common_args.extend(["--dataset", str(dataset_path)])
        common_args.extend(["--emb-device", "auto"])
        base_dir = "scripts/musique/baselines"
        if method in ["bm25", "dense", "hybrid"]:
            script_path = f"{base_dir}/run_vanilla_rag.py"
            extra_args = ["--retriever", method]
        elif method == "raptor":
            script_path = f"{base_dir}/run_raptor.py"
        elif method == "selfrag":
            script_path = f"{base_dir}/run_selfrag.py"
        elif method == "relrag_full":
            script_path = f"{base_dir}/run_relrag.py"
            extra_args = ["--embed-model", "sentence-transformers/all-MiniLM-L6-v2"]
        elif method == "lightrag":
            script_path = f"{base_dir}/run_graphrag.py"
            
    elif dtype == "mirage":
        common_args.extend(["--dataset-path", str(dataset_path)])
        # Mirage uses config for embed device, no flag
        base_dir = "scripts/mirage"
        if method in ["bm25", "dense", "hybrid"]:
            script_path = f"{base_dir}/run_vanilla_rag.py"
            extra_args = ["--retriever", method]
        elif method == "raptor":
            script_path = f"{base_dir}/run_simple_raptor.py"
        elif method == "selfrag":
            script_path = f"{base_dir}/run_simple_selfrag.py"
        elif method == "relrag_full":
            script_path = f"{base_dir}/run_relrag.py"
            extra_args = ["--embed-model", "sentence-transformers/all-MiniLM-L6-v2"]
        elif method == "lightrag":
            script_path = f"{base_dir}/run_simple_graphrag.py"

    if not script_path:
        raise ValueError(f"Unknown method {method} for dataset {dtype}")
        
    cmd.append(script_path)
    cmd.extend(common_args)
    cmd.extend(extra_args)
    
    return cmd

def main():
    parser = argparse.ArgumentParser(description="Experiment 1 Runner")
    parser.add_argument("--dataset", nargs="+", default=list(DATASET_MAP.keys()), 
                        choices=list(DATASET_MAP.keys()), help="Datasets to run")
    parser.add_argument("--method", nargs="+", required=True, 
                        choices=["bm25", "dense", "hybrid", "lightrag", "raptor", "selfrag", "relrag_full"],
                        help="Methods to run")
    parser.add_argument("--budget", type=int, default=4096, help="Token budget")
    parser.add_argument("--outdir", default=None, help="Base output directory")
    parser.add_argument("--dry-run", action="store_true", help="Run only 3 samples for testing")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output exists")
    
    args = parser.parse_args()
    
    # Setup output directory
    date_str = datetime.now().strftime("%Y%m%d")
    base_outdir = Path(args.outdir) if args.outdir else Path(f"result_relrag/{date_str}")
    
    for dataset_key in args.dataset:
        for method in args.method:
            print(f"=== Running {dataset_key} with {method} (Budget: {args.budget}) ===")
            
            dataset_path = Path(DATASET_MAP[dataset_key]).resolve()
            if not dataset_path.exists():
                print(f"Error: Dataset not found at {dataset_path}")
                continue
                
            run_dir = base_outdir / dataset_key / method / f"budget_{args.budget}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = get_runner_command(dataset_key, method, dataset_path, args.budget, run_dir)
            
            if args.limit > 0:
                cmd.extend(["--limit", str(args.limit)])
            elif args.dry_run:
                cmd.extend(["--limit", "3"])
                # Also set a smaller retrieval topk for speed in dry run if possible, 
                # but better to test defaults. Limit 3 is key.
            
            print(f"Command: {' '.join(cmd)}")
            
            # Env override for embedding model
            env = os.environ.copy()
            env["RAG_EMBED_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
            
            try:
                subprocess.run(cmd, check=True, env=env)
                print(f"✅ Finished {dataset_key} - {method}")
                
                # Ensure run_meta.json exists (copy config.resolved.json)
                config_resolved = run_dir / "config.resolved.json"
                run_meta = run_dir / "run_meta.json"
                if config_resolved.exists():
                    import shutil
                    shutil.copy(config_resolved, run_meta)
                    print(f"Updated run_meta.json from config.resolved.json")
                
                # Run Evaluation
                print(f"Running evaluation for {run_dir}")
                eval_cmd = [
                    sys.executable, "scripts/evaluate_standard.py",
                    "--run-dir", str(run_dir),
                    "--dataset-path", str(dataset_path),
                    "--dataset-type", DATASET_TYPE[dataset_key]
                ]
                subprocess.run(eval_cmd, check=True)
                print(f"✅ Evaluated {dataset_key} - {method}")
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed {dataset_key} - {method}: {e}")

if __name__ == "__main__":
    main()
