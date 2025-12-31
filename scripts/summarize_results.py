import json
import os
import argparse
from pathlib import Path
import pandas as pd

def load_json(path):
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def main():
    print("Starting summary...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root directory to scan (e.g. result_relrag/20251231)")
    parser.add_argument("--output", default="results_overview", help="Output filename prefix")
    args = parser.parse_args()
    
    root_path = Path(args.root)
    if not root_path.exists():
        print(f"Error: {root_path} does not exist")
        return

    records = []
    
    # Walk through directory
    # Expected structure: root / dataset / method / budget_xxxx / metrics / ...
    
    for dataset_dir in root_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        
        for method_dir in dataset_dir.iterdir():
            if not method_dir.is_dir():
                continue
            method_name = method_dir.name
            
            for run_dir in method_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                # Check for metrics
                metrics_dir = run_dir / "metrics"
                if not metrics_dir.exists():
                    continue
                
                ret_metrics = load_json(metrics_dir / "retrieval_metrics.json")
                gen_metrics = load_json(metrics_dir / "generation_metrics.json")
                fmt_metrics = load_json(metrics_dir / "format_metrics.json")
                
                if not ret_metrics and not gen_metrics:
                    continue
                
                record = {
                    "dataset": dataset_name,
                    "method": method_name,
                    "run_dir": str(run_dir.relative_to(root_path)),
                    "EM": gen_metrics.get("EM", 0.0),
                    "F1": gen_metrics.get("F1", 0.0),
                    "R@1": ret_metrics.get("Recall@1", 0.0),
                    "R@3": ret_metrics.get("Recall@3", 0.0),
                    "R@5": ret_metrics.get("Recall@5", 0.0),
                    "R@10": ret_metrics.get("Recall@10", 0.0),
                    "MRR": ret_metrics.get("MRR", 0.0),
                    "Hit@1": ret_metrics.get("Hit@1", 0.0),
                    "Hit@3": ret_metrics.get("Hit@3", 0.0),
                    "Hit@5": ret_metrics.get("Hit@5", 0.0),
                    "invalid_rate": fmt_metrics.get("invalid_rate", 0.0),
                    "no_final_tag_rate": fmt_metrics.get("no_final_tag_rate", 0.0),
                }
                records.append(record)
    
    if not records:
        print("No results found.")
        return
        
    df = pd.DataFrame(records)
    
    # Sort
    df = df.sort_values(by=["dataset", "method"])
    
    # Save JSON
    df.to_json(f"{args.output}.json", orient="records", indent=2)
    
    # Save Markdown
    # Columns to show
    cols = ["dataset", "method", "EM", "F1", "R@1", "R@3", "R@5", "MRR", "Hit@5", "invalid_rate", "run_dir"]
    # Filter cols that exist
    cols = [c for c in cols if c in df.columns]
    
    # Manual markdown table generation
    def simple_markdown(df, columns):
        # Header
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        lines = [header, sep]
        
        for _, row in df.iterrows():
            vals = []
            for c in columns:
                val = row[c]
                if isinstance(val, float):
                    vals.append(f"{val:.2f}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    md_table = simple_markdown(df, cols)
    
    with open(f"{args.output}.md", "w") as f:
        f.write("# Results Overview\n\n")
        f.write(md_table)
        
    print(f"Saved summary to {args.output}.json and {args.output}.md")
    print(md_table)

if __name__ == "__main__":
    main()
