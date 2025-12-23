# Experiment Runbook

## 0. Entry And Prerequisites
- Unified entry: `python experiments/run.py --config <config>`
- Evaluator: `scripts/evaluate_relrag.py` (triggered by `runtime.run_eval=true`)
- Result root: `result_relrag/<exp_name>/run_<timestamp>/...`
- Ensure vLLM and embedding endpoints are available.

## 1. Recommended Order (HotpotQA -> Mirage -> MuSiQue)

| Experiment | Config | Datasets | Methods | Default Budgets | LLM Profiles |
| --- | --- | --- | --- | --- | --- |
| E1 | `experiments/configs/e1_main_qwen.yaml` | hotpotqa_distractor_200 -> mirage_sample_200 -> musique_sample | bm25_rag, dense_rag, hybrid_rag, raptor, relrag_full | 4096/8192/16384 | default |
| E2 | `experiments/configs/e2_retrieval.yaml` | hotpotqa_distractor_200 -> musique_sample | bm25_rag, dense_rag, hybrid_rag, relrag_full | [] | default |
| E3 | `experiments/configs/e3_noise_robustness.yaml` | hotpotqa_distractor_200 -> mirage_sample_200 | relrag_full, hybrid_rag | 2000 | default |
| E5 | `experiments/configs/e5_ablation_structure.yaml` | hotpotqa_distractor_200 -> musique_sample | relrag_full | 4096/8192/16384 | default |
| E7 | `experiments/configs/e7_cross_llm.yaml` | hotpotqa_distractor_200 | hybrid_rag, relrag_full | 4096/8192/16384 | qwen3_30b_a3b |
| E8 | `experiments/configs/e8_budget_scaling.yaml` | hotpotqa_distractor_200 | relrag_full | 500/1000/2000/4000 | default |
| E9 | `experiments/configs/e9_generator_scaling.yaml` | hotpotqa_distractor_200 | relrag_full | 2000 | default/small/large (same model) |
| E10 | `experiments/configs/e10_end_to_end.yaml` | hotpotqa_distractor_200 | relrag_full, hybrid_rag | 2000 | default |

E4 and E6 are optional missing.
E5 ablations: `w_o_graph_expansion`, `w_o_context_scheduler`, `hop_1/2/3/4`.

## 2. Example Commands
```bash
python experiments/run.py --config experiments/configs/e1_main_qwen.yaml
python experiments/run.py --config experiments/configs/e2_retrieval.yaml
python experiments/run.py --config experiments/configs/e3_noise_robustness.yaml
python experiments/run.py --config experiments/configs/e5_ablation_structure.yaml
python experiments/run.py --config experiments/configs/e7_cross_llm.yaml
python experiments/run.py --config experiments/configs/e8_budget_scaling.yaml
python experiments/run.py --config experiments/configs/e9_generator_scaling.yaml
python experiments/run.py --config experiments/configs/e10_end_to_end.yaml
```

To run a subset:
- `--only <substring>` to filter job_id matches
- Or edit the config `datasets` / `methods` / `llm_profiles` lists

## 3. Output Layout
Workdir shape:
`result_relrag/<exp_name>/run_<timestamp>/<dataset>/<method>/llm_<profile>/budget_<budget>/<variant>/`

Each workdir must contain:
- `preds/pred_raw.jsonl`, `preds/pred_norm.jsonl`, `preds/pred_final.jsonl`
- `artifacts/retrieval.jsonl`
- `metrics/qa_metrics_norm.json`, `metrics/qa_metrics_final.json`, `metrics/format_metrics.json`, `metrics/retrieval_metrics.json`
- `config.resolved.json`, `summary.json`, `run.log`

## 4. Failure And Resume Strategy
- Resume: set `runtime.resume: true` and keep `runtime.save_every` (runner passes `--resume` when supported).
- Force re-run: add `--force` to ignore existing outputs.
- Targeted re-run: use `--only` or shrink config datasets/methods.
- If LLM or embedding fails, fix the environment and resume in the same workdir.
