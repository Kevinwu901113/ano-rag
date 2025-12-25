
## 2. Fix Phase

### B1. Model Name Mismatch
- **Status**: Checked. `SERVED_MODEL_NAME` in `utils/llm_client.py` is `qwen3-30b-a3b`, which matches the vLLM served model ID. No changes needed for model name.

### B2. Payload Issues
- **Status**: No 400 errors encountered during verification run. The instrumented logging code did not trigger.

### B3. Minimal Acceptance Config
- **File Created**: `experiments/configs/audit_min.yaml`
- **Settings**:
  - `runtime.limit`: 5
  - `meta.result_root`: `result_relrag/miniaudit`
  - `methods`: `[dense_rag]`
  - `datasets`: `[hotpotqa_distractor_200]`
  - `embedding`: `/home/wjk/models/qwen3-emb` (CPU)

## 3. Verification Phase

### Execution
- **Command**: `PYTHONPATH=. python3 experiments/run.py --config experiments/configs/audit_min.yaml`
- **Workdir**: `/home/wjk/workplace/nq/ano-rag/result_relrag/miniaudit/audit_min_check/run_20251224_042803/hotpotqa_distractor_200/dense_rag/llm_default/budget_default/full`

### Results
- **Completion**: Success (Exit code 0)
- **400 Errors**: None observed in `run.log`.
- **Predictions**: `preds/pred_raw.jsonl` contains 5 valid entries (no "error" strings found).
- **Embedding Config**: Verified in `config.resolved.json` and `run.log` (`/home/wjk/models/qwen3-emb` on cpu).

### Key Log Snippets
```
INFO     | __main__:main:371 - Embedding model /home/wjk/models/qwen3-emb (prefer=cpu, batch_size=4, max_length=512, normalize=True, dtype=auto)
...
INFO     | __main__:main:430 - Predictions generated for 5 examples
INFO     | __main__:main:440 - Saved predictions to .../preds/pred.json
```

### Conclusion
Passed minimum acceptance criteria.
