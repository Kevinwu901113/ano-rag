#!/usr/bin/env python3
import os
import glob
import subprocess
import json
import time
from pathlib import Path

# Configs
TARGET_DIRS = [
    "result/experiment_19",
    "result/musique_experiment_1"
]

JUDGES = {
    "vllm": {
        "model": "qwen3-30b-a3b",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_env": "EMPTY", # vLLM often uses EMPTY or ignores
        "concurrency": 16
    },
    "openai": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-c003ed6c90c24b5f8caad6252b4d6a24",
        "concurrency": 8 # Limit concurrency for external API
    }
}

def run_judge(pred_file, judge_name, judge_cfg, out_root):
    # Construct output dir
    # e.g. result/experiment_19/judge/vllm/pred_dev_openai_bm25/
    # Actually, user might prefer result/experiment_19/judge/pred_dev_openai_bm25_vllm_judge/
    # Let's keep it clean: result/experiment_19/judge/{judge_name}/{pred_filename_no_ext}
    
    pred_path = Path(pred_file)
    exp_dir = pred_path.parent
    pred_name = pred_path.stem
    
    out_dir = exp_dir / "judge" / judge_name / pred_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running {judge_name} judge on {pred_file} -> {out_dir}")
    
    cmd = [
        "python3", "relrag/eval/llm_judge_emf1.py",
        "--input", str(pred_path),
        "--out_dir", str(out_dir),
        "--model", judge_cfg["model"],
        "--base_url", judge_cfg["base_url"],
        "--concurrency", str(judge_cfg["concurrency"]),
        "--gold_key", "references",  # Force correct gold key
        "--pred_key", "prediction", # Force correct pred key (optional but safe)
    ]
    
    env = os.environ.copy()
    if "api_key" in judge_cfg:
        env["OPENAI_API_KEY"] = judge_cfg["api_key"]
    else:
        env["OPENAI_API_KEY"] = "EMPTY"

    # Only judge if output doesn't exist or resume?
    # The script supports resume.
    
    try:
        subprocess.run(cmd, env=env, check=True)
        return out_dir / "summary_llm_judge.json"
    except subprocess.CalledProcessError as e:
        print(f"Error running judge for {pred_file}: {e}")
        return None

def main():
    report_lines = ["# Comprehensive LLM Judge Report", "", f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    
    for dir_path in TARGET_DIRS:
        print(f"Processing directory: {dir_path}")
        report_lines.append(f"## Experiment: {dir_path}")
        
        # Find prediction files
        pred_files = sorted(glob.glob(os.path.join(dir_path, "pred_dev_*.jsonl")))
        if not pred_files:
            print(f"No prediction files found in {dir_path}")
            continue
            
        # Table Header
        report_lines.append("| Prediction File | Judge | EM | F1 | Samples | Bad |")
        report_lines.append("|---|---|---|---|---|---|")
        
        for pred_file in pred_files:
            for judge_name, judge_cfg in JUDGES.items():
                summary_path = run_judge(pred_file, judge_name, judge_cfg, dir_path)
                
                if summary_path and summary_path.exists():
                    with open(summary_path, "r") as f:
                        s = json.load(f)
                    
                    pred_name = Path(pred_file).name
                    em = f"{s.get('avg_em', 0):.4f}"
                    f1 = f"{s.get('avg_f1', 0):.4f}"
                    count = s.get("samples_count", 0)
                    bad = s.get("bad_count", 0)
                    
                    report_lines.append(f"| {pred_name} | {judge_name} | {em} | {f1} | {count} | {bad} |")
                else:
                    report_lines.append(f"| {Path(pred_file).name} | {judge_name} | Error | Error | - | - |")
        
        report_lines.append("")
        
    # Write Final Report
    with open("LLM_JUDGE_FULL_REPORT.md", "w") as f:
        f.write("\n".join(report_lines))
        
    print("Done. Report written to LLM_JUDGE_FULL_REPORT.md")

if __name__ == "__main__":
    main()
