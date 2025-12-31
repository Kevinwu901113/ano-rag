import os
import json
import glob
import sys
from pathlib import Path

def find_latest_run_dir(base_path):
    runs = glob.glob(os.path.join(base_path, "run_*"))
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)

def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0

def verify_job(job_dir, run_root):
    required_files = [
        "preds/pred_raw.jsonl",
        "metrics/qa_metrics_final.json",
        "metrics/format_metrics.json",
        "config.resolved.json",
        "summary.json",
        "run.log"
    ]
    
    missing_files = []
    for f in required_files:
        if not (job_dir / f).exists():
            missing_files.append(f)
            
    if missing_files:
        return {"status": "pending", "reason": f"Missing files: {missing_files}", "dir": str(job_dir)}
        
    # Count check
    pred_path = job_dir / "preds/pred_raw.jsonl"
    actual_count = count_lines(pred_path)
    
    # Heuristic for expected count
    expected_count = None
    dataset_name = ""
    try:
        config_data = json.loads((job_dir / "config.resolved.json").read_text())
        # Try to infer dataset name from path or config
        rel_path = job_dir.relative_to(run_root)
        dataset_name = rel_path.parts[0]
    except:
        pass
        
    if "200" in dataset_name:
        expected_count = 200
    elif "musique" in dataset_name and "sample" in dataset_name:
        # Assuming musique_sample is around 200-500, but let's just log it if we can't be sure
        pass
        
    if expected_count is not None and actual_count != expected_count:
        return {
            "status": "failed", 
            "reason": f"Count mismatch: expected {expected_count}, got {actual_count}", 
            "dir": str(job_dir),
            "no_final_tag_rate": 0.0 # Placeholder
        }

    # Quality checks
    try:
        format_metrics = json.loads((job_dir / "metrics/format_metrics.json").read_text())
        qa_metrics = json.loads((job_dir / "metrics/qa_metrics_final.json").read_text())
        config = json.loads((job_dir / "config.resolved.json").read_text())
    except Exception as e:
        return {"status": "failed", "reason": f"JSON parse error: {e}", "dir": str(job_dir)}
        
    failures = []
    if format_metrics.get("invalid_rate", 1.0) != 0.0:
        failures.append(f"invalid_rate {format_metrics.get('invalid_rate')} != 0.0")
    if format_metrics.get("budget_violation_rate", 1.0) != 0.0:
        failures.append(f"budget_violation_rate {format_metrics.get('budget_violation_rate')} != 0.0")
    
    no_final_tag_rate = format_metrics.get("no_final_tag_rate", 1.0)
    if no_final_tag_rate > 0.05:
        failures.append(f"no_final_tag_rate {no_final_tag_rate} > 0.05")
        
    # Config checks
    embedding = config.get("embedding", {})
    if embedding.get("model") != "/home/wjk/models/qwen3-emb":
        failures.append(f"embedding.model mismatch: {embedding.get('model')}")
    if embedding.get("device") != "cpu":
        failures.append(f"embedding.device mismatch: {embedding.get('device')}")
    
    if config.get("llm_model") == "qwen3-30b-a3b":
        pass 
    elif config.get("llm", {}).get("model") == "qwen3-30b-a3b":
        pass
    elif config.get("model") == "qwen3-30b-a3b":
        pass
    else:
        # Check profile
        pass
        
    if "git_commit" not in config:
        failures.append("git_commit missing")
        
    if failures:
        return {
            "status": "failed", 
            "reason": "; ".join(failures), 
            "dir": str(job_dir),
            "no_final_tag_rate": no_final_tag_rate,
            "count": actual_count
        }
    
    # Extract metadata from path
    try:
        rel_path = job_dir.relative_to(run_root)
        parts = rel_path.parts
        # parts: (dataset, method, llm, budget, variant)
        dataset = parts[0]
        method = parts[1]
        budget_str = parts[3]
        budget = budget_str.replace("budget_", "")
    except:
        dataset = config.get("dataset", "unknown")
        method = config.get("method", "unknown")
        budget = config.get("context_budget_tokens", "unknown")
        
    return {
        "status": "passed",
        "dir": str(job_dir),
        "dataset": dataset, 
        "method": method,
        "budget": budget, 
        "em": qa_metrics.get("em", qa_metrics.get("EM", 0.0)),
        "f1": qa_metrics.get("f1", qa_metrics.get("AnswerF1", qa_metrics.get("F1", 0.0))),
        "no_final_tag_rate": no_final_tag_rate,
        "job_id": job_dir.name,
        "count": actual_count
    }

def main():
    base_path = "result_relrag/e1_main_qwen"
    latest_run = find_latest_run_dir(base_path)
    if not latest_run:
        print("No run directory found.")
        sys.exit(1)
        
    print(f"Verifying run: {latest_run}")
    
    run_path = Path(latest_run)
    job_configs = list(run_path.rglob("config.resolved.json"))
    
    results = []
    job_dirs = [p.parent for p in job_configs if p.parent != run_path]
    
    for job_dir in job_dirs:
        res = verify_job(job_dir, run_path)
        results.append(res)
        
    passed = [r for r in results if r["status"] == "passed"]
    failed = [r for r in results if r["status"] == "failed"]
    pending = [r for r in results if r["status"] == "pending"]
    
    print(f"Total found: {len(results)}")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    print(f"Pending: {len(pending)}")
    
    # Generate Report
    lines = []
    lines.append("# E1 Results Summary")
    lines.append("")
    lines.append("## 1. 实验说明")
    lines.append("- **Dataset**: HotpotQA distractor_200, Mirage, Musique")
    lines.append("- **Model**: qwen3-30b-a3b (vLLM)")
    lines.append("- **Embedding**: /home/wjk/models/qwen3-emb (CPU)")
    lines.append("- **Output Protocol**: FINAL-tag enforced")
    lines.append("")
    lines.append("## 2. 结果汇总表")
    lines.append("| Job ID | Dataset | Method | Budget | Count | EM | F1 | no_final_tag_rate |")
    lines.append("|---|---|---|---|---|---|---|---|")
    
    sorted_results = sorted(results, key=lambda x: (x.get("dataset", ""), x.get("method", ""), str(x.get("budget", ""))))
    
    for r in sorted_results:
        if r["status"] == "passed":
            budget = r.get("budget")
            if isinstance(budget, list):
                budget = str(budget)
            rel_path = os.path.relpath(r["dir"], latest_run)
            lines.append(f"| {rel_path} | {r.get('dataset')} | {r.get('method')} | {budget} | {r.get('count')} | {r.get('em'):.4f} | {r.get('f1'):.4f} | {r.get('no_final_tag_rate'):.4f} |")
            
    lines.append("")
    lines.append("## 3. 异常说明")
    if failed:
        lines.append(f"Found {len(failed)} failed jobs:")
        for f in failed:
            lines.append(f"- **Path**: `{f['dir']}`")
            lines.append(f"  - Reason: {f['reason']}")
    else:
        lines.append("No failed jobs.")
        
    high_nft = [r for r in sorted_results if r.get("no_final_tag_rate", 0) > 0.05]
    if high_nft:
        lines.append("\nJobs with high no_final_tag_rate (> 0.05):")
        for r in high_nft:
             lines.append(f"- {r['dir']} ({r.get('no_final_tag_rate')})")
             
    lines.append("")
    lines.append("## 4. 结论")
    all_passed = len(failed) == 0 and len(pending) == 0
    lines.append(f"- E1 是否全部跑通: {'YES' if all_passed else 'NO'}")
    if pending:
        lines.append(f"  - (Note: {len(pending)} jobs are still pending/incomplete)")
    lines.append("- 结果是否稳定: (See rates)")
    lines.append(f"- 是否可以作为论文主结果: {'YES' if all_passed else 'NO'}")
    
    report_content = "\n".join(lines)
    
    os.makedirs("doc", exist_ok=True)
    with open("doc/e1_results_summary.md", "w") as f:
        f.write(report_content)
        
    print("Report generated at doc/e1_results_summary.md")

if __name__ == "__main__":
    main()
