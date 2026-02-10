import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Find high recall examples in MuSiQue results')
    parser.add_argument('--file', type=str, default='result/musique_experiment_4/pred_dev_dense.jsonl', help='Path to predictions.jsonl')
    parser.add_argument('--threshold', type=float, default=1.0, help='Recall threshold (inclusive)')
    parser.add_argument('--limit', type=int, default=5, help='Number of examples to show per category')
    args = parser.parse_args()

    print(f"Reading from {args.file}...")
    
    high_recall_examples = []
    
    try:
        with open(args.file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    metrics = data.get('metrics', {})
                    # gold_sp_subset is the retrieval recall (fraction of gold supporting paragraphs found)
                    retrieval_recall = metrics.get('gold_sp_subset', 0.0)
                    
                    if retrieval_recall >= args.threshold:
                        high_recall_examples.append(data)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: File {args.file} not found.")
        sys.exit(1)

    print(f"Found {len(high_recall_examples)} examples with retrieval recall >= {args.threshold}")
    
    if not high_recall_examples:
        return

    # Sort by F1 to show both high and low performing QA given high retrieval
    high_recall_examples.sort(key=lambda x: x.get('metrics', {}).get('f1', 0.0), reverse=True)
    
    print(f"\n=== Top {args.limit} High Recall Examples with BEST QA Performance (F1) ===")
    for i, ex in enumerate(high_recall_examples[:args.limit]):
        print_example(ex, i+1)

    print(f"\n=== Top {args.limit} High Recall Examples with WORST QA Performance (F1) ===")
    for i, ex in enumerate(high_recall_examples[-args.limit:]):
        print_example(ex, i+1)

def print_example(ex, idx):
    metrics = ex.get('metrics', {})
    print(f"\n[{idx}] ID: {ex.get('id')} | Recall: {metrics.get('gold_sp_subset'):.2f} | F1: {metrics.get('f1'):.2f} | EM: {metrics.get('em'):.2f}")
    print(f"Question: {ex.get('question')}")
    print(f"Gold Answer: {ex.get('gold_answer')}")
    print(f"Prediction: {ex.get('prediction')}")
    print(f"Num Supporting Facts Found: {len(ex.get('retrieved_context_raw', []))} (Note: raw list size)")
    # Check gold_sp coverage
    gold_sp = ex.get('gold_sp', [])
    print(f"Gold SP count: {len(gold_sp)}")

if __name__ == "__main__":
    main()
