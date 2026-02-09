# LLM-as-a-Judge for Exact Match & F1 Evaluation

This tool evaluates QA predictions using an LLM to compute Exact Match (EM) and F1 scores, following strict normalization and scoring rules. It is designed for datasets like HotpotQA, MuSiQue, and 2WikiMultihopQA.

## Features

- **Robust Scoring**: Uses an LLM to enforce normalization (lowercase, punctuation removal, article removal) and multiset token overlap for F1.
- **Cache-enabled**: Avoids re-evaluating identical (model + prompt + input) samples. Supports resume.
- **Concurrency**: Supports parallel LLM calls.
- **Reliable**: Automatic JSON repair and validation.
- **Flexible Input**: Auto-infers column names for ID, Question, Prediction, Gold.

## Usage

### Prerequisites

Ensure you have the `openai` python package installed and access to an OpenAI-compatible API (e.g., vLLM or OpenAI official).

```bash
pip install openai
```

### Basic Command

```bash
python3 relrag/eval/llm_judge_emf1.py \
  --input path/to/predictions.jsonl \
  --out_dir path/to/output_dir \
  --model qwen3-30b-a3b
```

### Arguments

- `--input`: Path to input JSONL file.
- `--out_dir`: Directory to save results.
- `--model`: Model name to use (e.g., `gpt-4o`, `qwen3-30b-a3b`).
- `--base_url`: LLM API endpoint (default: `http://127.0.0.1:8000/v1`).
- `--api_key_env`: Environment variable name for API Key (default: `OPENAI_API_KEY`).
- `--limit N`: Process only first N samples.
- `--concurrency K`: Number of parallel requests (default: 1).
- `--only_failed_official`: If input contains `official_em`/`official_f1`, only evaluate samples where `official_em=0` or `official_f1` is below threshold.
- `--id_key`, `--question_key`, `--pred_key`, `--gold_key`: Manually specify JSON keys if auto-inference fails.

### Input Format

Input file must be a JSONL (one JSON object per line) containing at least:
- ID
- Question
- Prediction
- Gold Answer(s) (string or list of strings)

Example:
```json
{"id": "q1", "question": "...", "pred": "paris", "gold": "Paris"}
```

### Output Files

1. **`judge_emf1.jsonl`**: Detailed per-sample results.
   ```json
   {
     "id": "...",
     "llm_judge": {
       "em": 1,
       "f1": 1.0,
       "debug": { ... }
     }
   }
   ```
2. **`summary_llm_judge.json`**: Aggregated metrics.
3. **`summary_llm_judge.md`**: Human-readable summary.
4. **`cache.jsonl`**: Cache file for resume capability.

## Example: vLLM

```bash
python3 relrag/eval/llm_judge_emf1.py \
  --input results/hotpot_experiment/predictions.jsonl \
  --out_dir results/hotpot_experiment/judge \
  --model qwen3-30b-a3b \
  --base_url http://localhost:8000/v1 \
  --concurrency 4
```

## Example: OpenAI

```bash
export OPENAI_API_KEY=sk-...
python3 relrag/eval/llm_judge_emf1.py \
  --input results/hotpot_experiment/predictions.jsonl \
  --out_dir results/hotpot_experiment/judge \
  --model gpt-4o \
  --base_url https://api.openai.com/v1 \
  --concurrency 10
```
