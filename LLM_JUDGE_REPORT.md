# LLM Judge Evaluation Report (Partial)

## Status
The comprehensive evaluation process was initiated but stopped early to address a critical data mapping issue. 

## Critical Finding
During the initial run on `experiment_19`, we observed suspiciously perfect scores (EM 1.0, F1 1.0).
Upon inspection of the input `pred_dev_*.jsonl` files, it was discovered that:
1. The `answer` field in the file contains the **model's prediction** (e.g., "Insufficient evidence"), matching the `prediction` field.
2. The actual ground truth is stored in the `references` field (e.g., `["650 locations"]`).
3. The default behavior of the judge script prioritized `answer` as the gold label key, leading to the judge comparing the prediction against itself.

## Corrective Action
The batch execution script `scripts/eval/run_batch_judge.py` has been updated to explicitly use:
- `--gold_key references`
- `--pred_key prediction`

## Next Steps
Please resume the evaluation by running the updated batch script. This will re-process the files with the correct ground truth mapping.

### Command to Resume
```bash
python3 scripts/eval/run_batch_judge.py
```

### Initial (Invalid) Results
The following results were generated before the fix and are **invalid** (shown for completeness of the log):
- **Experiment 19 (OpenAI Judge)**: All tested files (BM25, Dense, Hybrid) showed ~1.0 EM/F1.
- **Experiment 19 (vLLM Judge)**: Started but incomplete.

## Configuration Used
- **vLLM**: `qwen3-30b-a3b` @ `http://127.0.0.1:8000/v1` (Concurrency: 16)
- **OpenAI**: `deepseek-chat` @ `https://api.deepseek.com/v1` (Concurrency: 8)
- **Targets**: `result/experiment_19`, `result/musique_experiment_1`
