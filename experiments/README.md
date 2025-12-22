# Experiments (RelRAG)

This folder contains experiment orchestration configs and a thin runner.
All runs should write to `result_relrag/` using the workdir convention.

## Output layout
Each run uses:
`result_relrag/<exp_name>/run_<timestamp>/<dataset>/<method>/<llm>/<budget>/<variant>/`

Each workdir contains:
- run.log
- preds/
- metrics/
- artifacts/
- config.resolved.json (resolved config snapshot at run root)
- summary.json (run root; per-job metadata + metrics)

## How to run
Example:
```bash
PYTHONPATH=. python experiments/run.py \
  --config experiments/configs/e1_main_qwen.yaml \
  |& tee result_relrag/e1_main_qwen/run_$(date +%Y%m%d_%H%M%S).log
```

## Config overview
- `configs/*.yaml`: experiment recipes (datasets, methods, budgets).
- `manifests/datasets.yaml`: dataset name -> path mapping.
- `manifests/methods.yaml`: method name -> script entry mapping.

The runner supports `include:` to merge YAML files and simple env expansion
like `${OPENAI_BASE_URL:-https://api.openai.com/v1}`.

## Notes
- If a method entry is missing for a dataset, the runner skips that combo.
- `ablations[].overrides` are recorded in metadata; if you already have
  CLI flags for an ablation, add `cli_args` in the ablation entry.
- Token budgets are passed to known flags if the script supports them.
  If your script uses a different arg, add it to `cli_args`.
- `runtime.resume` defaults to false for auditability; enable explicitly per experiment.
- `fairness.enforce_budget/topk` will skip jobs when the entry script does not
  expose the required flags (to prevent unfair comparisons).
- `runtime.seed` sets `PYTHONHASHSEED`; if a script lacks `--seed`, the runner
  wraps it with `experiments/seeded_run.py` to seed Python/NumPy/Torch.
- `summary.json` records full command lines, applied flags, and code hashes.
- To hash prompts explicitly, add `prompt_paths` (or `prompt_paths_<dataset>`)
  to the method entry in `experiments/manifests/methods.yaml`.

## Reproducibility checklist
- Record git commit hash in the run root summary.
- Fix token budgets and top-k across methods.
- Use the same dataset split files (no regenerated samples).
- Enforce a strict answer format (avoid empty / think-only outputs).
- Track prompt/code hashes in summary to lock prompt versions.
