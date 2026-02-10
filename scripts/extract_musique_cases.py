import json
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Extract specific MuSiQue cases by ID')
    parser.add_argument('--file', type=str, default='result/musique_experiment_4/pred_dev_dense.jsonl', help='Path to predictions.jsonl')
    parser.add_argument('--ids', type=str, required=True, help='Comma-separated list of IDs to extract')
    parser.add_argument('--output', type=str, default='debug/musique_failure_cases.jsonl', help='Output file path')
    args = parser.parse_args()

    target_ids = set(args.ids.split(','))
    found_count = 0
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"Extracting {len(target_ids)} cases from {args.file}...")
    
    with open(args.output, 'w') as out_f:
        try:
            with open(args.file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data.get('id') in target_ids:
                            # Dump the full record
                            out_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                            found_count += 1
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"Error: File {args.file} not found.")
            sys.exit(1)

    print(f"Extracted {found_count} cases to {args.output}")

if __name__ == "__main__":
    main()
