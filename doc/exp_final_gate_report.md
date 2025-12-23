# Final Gate Report

PASS/FAIL: PASS (format soft-fail)

Run root: `result_relrag/final_gate/20251222_113702`
Git commit: `6cc844d01dc29f2ff8913b9017591f4839f76a2b`

## A. Absolute Rules Check
- Generation-side cleaning/answer filtering: PASS (no GENERATION_PATH_USED hits after fixes)
- Unified evaluation entry: PASS (all runs evaluated via `scripts/evaluate_relrag.py`)
- Matched budget auditable: PASS (pred_raw includes budget fields; `budget_violation_rate` reported and 0.0)
- Workdir artifacts: PASS for all run2 workdirs (see D)
- Hard gate metrics: PASS (`invalid_rate==0.0`, `budget_violation_rate==0.0`, artifacts complete)
- FINAL protocol compliance: Soft-fail only (tracked via `no_final_tag_rate`, not a hard gate)

## B1. Forbidden Cleaning Scan (repo-wide)
| File | Matches | Classification | Notes |
| --- | --- | --- | --- |
| `check_cleaner.py` | `clean_model_answer`, `_strip_reasoning` | LEGACY_UNUSED | Diagnostic script only |
| `baselines/fid_rag/utils.py` | `_strip_reasoning` | LEGACY_UNUSED | Legacy baseline utilities |
| `baselines/direct_llm/utils.py` | `_strip_reasoning` | LEGACY_UNUSED | Not imported by runner |
| `clean_fid.py` | `_strip_reasoning`, `_enforce_short_answer` | LEGACY_UNUSED | Legacy cleanup script |
| `utils/answer_cleaner.py` | `_strip_reasoning`, `_enforce_short_answer`, `clean_model_answer` | LEGACY_UNUSED | Used only by diagnostics/tools |
| `utils/k_estimator.py` | `answer_clean` | LEGACY_UNUSED | Analysis helper |
| `utils/support_fill.py` | `answer_clean` | LEGACY_UNUSED | Analysis helper |
| `scripts/hotpotqa/baselines/run_relrag_diagnostic.py` | `_strip_reasoning`, `clean_model_answer` | LEGACY_UNUSED | Diagnostic runner only |
| `scripts/query_dataset.py` | `_strip_reasoning` | LEGACY_UNUSED | Dataset tooling |
| `scripts/mirage/query_dataset.py` | `_strip_reasoning` | LEGACY_UNUSED | Dataset tooling |
| `baselines/simple_graphrag/utils.py` | `_strip_reasoning` | LEGACY_UNUSED | Legacy baseline |
| `baselines/simple_graphrag/runner.py` | `_strip_reasoning`, `_enforce_short_answer` | LEGACY_UNUSED | Legacy baseline |
| `baselines/simple_graphrag/retriever.py` | `_strip_reasoning` | LEGACY_UNUSED | Legacy baseline |
| `generator/answerer.py` | `_strip_reasoning` | LEGACY_UNUSED | Not used by manifests |
| `test_answer_cleaning.py` | `_strip_reasoning`, `_enforce_short_answer` | LEGACY_UNUSED | Test-only |
| `utils/output_eval.py` | `_strip_tag_markers`, `_strip_code_fence_markers` | EVAL_ONLY | Evaluation normalization |
| `doc/exp_final_gate_report.md` | `strip_reasoning`, `clean_model_answer`, `answer_clean`, `_strip_`, `answer_cleaner`, `enforce_short_answer` | LEGACY_UNUSED | Doc-only (self-reference) |

GENERATION_PATH_USED: None.

## B2. Method Mapping & Fairness
| Method | Entry scripts | pred_raw | Budget fields | Retrieval artifacts | FINAL protocol |
| --- | --- | --- | --- | --- | --- |
| `direct_llm` | hotpot: `scripts/hotpotqa/baselines/run_direct.py`; musique: `scripts/musique/baselines/run_direct.py`; mirage: `scripts/mirage/run_direct_llm.py` | Yes | Yes | Yes (mirage logs empty retrieval) | Yes (`build_final_instruction` in prompts) |
| `bm25_rag` | hotpot: `scripts/hotpotqa/baselines/run_vanilla_rag.py` (`--retriever bm25`); musique: `scripts/musique/baselines/run_vanilla_rag.py`; mirage: `scripts/mirage/run_vanilla_rag.py` | Yes | Yes | Yes | Yes |
| `dense_rag` | hotpot: `scripts/hotpotqa/baselines/run_vanilla_rag.py` (`--retriever dense`); musique: `scripts/musique/baselines/run_vanilla_rag.py`; mirage: `scripts/mirage/run_vanilla_rag.py` | Yes | Yes | Yes | Yes |
| `hybrid_rag` | hotpot: `scripts/hotpotqa/baselines/run_vanilla_rag.py` (`--retriever hybrid`); musique: `scripts/musique/baselines/run_vanilla_rag.py`; mirage: `scripts/mirage/run_vanilla_rag.py` | Yes | Yes | Yes | Yes |
| `raptor` | hotpot: `scripts/hotpotqa/baselines/run_raptor.py`; musique: `scripts/musique/baselines/run_raptor.py`; mirage: `scripts/mirage/run_simple_raptor.py` | Yes | Yes | Yes | Yes (mirage via `baselines/simple_raptor/retriever.py`) |
| `relrag_full` | hotpot: `scripts/hotpotqa/baselines/run_relrag.py`; musique: `scripts/musique/baselines/run_relrag.py`; mirage: `null` | Yes | Yes | Yes | Yes |

All methods inject the FINAL protocol via `build_final_instruction` or equivalent dependency.

## B3. Config Coverage Matrix (E1–E10)
| E# | Config | Datasets | Methods | Budgets | LLM profile |
| --- | --- | --- | --- | --- | --- |
| E1 | `experiments/configs/e1_main_qwen.yaml` | hotpotqa_distractor_200, mirage_sample_200, musique_sample | bm25_rag, dense_rag, hybrid_rag, raptor, relrag_full | 4096/8192/16384 | qwen3_30b_a3b |
| E2 | `experiments/configs/e2_retrieval.yaml` | hotpotqa_distractor_200, musique_sample | bm25_rag, dense_rag, hybrid_rag, relrag_full | [] (empty list) | default |
| E3 | `experiments/configs/e3_noise_robustness.yaml` | hotpotqa_distractor_200, mirage_sample_200 | relrag_full, hybrid_rag | 2000 | default |
| E4 | Missing (Optional) | — | — | — | — |
| E5 | `experiments/configs/e5_ablation_structure.yaml` | hotpotqa_distractor_200, musique_sample | relrag_full | 4096/8192/16384 | default |
| E6 | Missing (Optional) | — | — | — | — |
| E7 | `experiments/configs/e7_cross_llm.yaml` | hotpotqa_distractor_200 | hybrid_rag, relrag_full | 4096/8192/16384 | qwen3_30b_a3b |
| E8 | `experiments/configs/e8_budget_scaling.yaml` | hotpotqa_distractor_200 | relrag_full | 500/1000/2000/4000 | default |
| E9 | `experiments/configs/e9_generator_scaling.yaml` | hotpotqa_distractor_200 | relrag_full | 2000 | default |
| E10 | `experiments/configs/e10_end_to_end.yaml` | hotpotqa_distractor_200 | relrag_full, hybrid_rag | 2000 | default |

Non-E configs present: `experiments/configs/audit_full.yaml` (audit-only).

## C. Dynamic Final Gate Runs (limit=20 unless noted)
Run IDs and commands (all under `result_relrag/final_gate/20251222_113702/`):

- `E1_hotpot_relrag_b2000_run2`
  - `NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLM_PROFILE=qwen3_30b_a3b python scripts/hotpotqa/baselines/run_relrag.py --dataset data/hotpotqa/dataset_distractor_200.json --workdir result_relrag/final_gate/20251222_113702/E1_hotpot_relrag_b2000_run2 --context-budget 2000 --limit 20 --lm-endpoint http://127.0.0.1:8000/v1 --lm-model qwen3-30b-a3b --embed-model /home/wjk/models/hf-cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf --embed-device cpu --max-new-tokens 128`
  - `python scripts/evaluate_relrag.py --dataset data/hotpotqa/dataset_distractor_200.json --dataset-name hotpotqa --workdir result_relrag/final_gate/20251222_113702/E1_hotpot_relrag_b2000_run2 --ks 1,3,5,10`
- `E2_hotpot_relrag_retrieval_run2`
  - `NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLM_PROFILE=qwen3_30b_a3b python scripts/hotpotqa/baselines/run_relrag.py --dataset data/hotpotqa/dataset_distractor_200.json --workdir result_relrag/final_gate/20251222_113702/E2_hotpot_relrag_retrieval_run2 --context-budget 2000 --limit 20 --lm-endpoint http://127.0.0.1:8000/v1 --lm-model qwen3-30b-a3b --embed-model /home/wjk/models/hf-cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf --embed-device cpu --max-new-tokens 128`
  - `python scripts/evaluate_relrag.py --dataset data/hotpotqa/dataset_distractor_200.json --dataset-name hotpotqa --workdir result_relrag/final_gate/20251222_113702/E2_hotpot_relrag_retrieval_run2 --ks 1,3,5,10`
- `E5_hotpot_relrag_hop2_run2`
  - `NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLM_PROFILE=qwen3_30b_a3b python scripts/hotpotqa/baselines/run_relrag.py --dataset data/hotpotqa/dataset_distractor_200.json --workdir result_relrag/final_gate/20251222_113702/E5_hotpot_relrag_hop2_run2 --context-budget 2000 --limit 20 --hop 2 --lm-endpoint http://127.0.0.1:8000/v1 --lm-model qwen3-30b-a3b --embed-model /home/wjk/models/hf-cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf --embed-device cpu --max-new-tokens 128`
  - `python scripts/evaluate_relrag.py --dataset data/hotpotqa/dataset_distractor_200.json --dataset-name hotpotqa --workdir result_relrag/final_gate/20251222_113702/E5_hotpot_relrag_hop2_run2 --ks 1,3,5,10`
- `E5_hotpot_relrag_no_scheduler_run2`
  - `NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLM_PROFILE=qwen3_30b_a3b python scripts/hotpotqa/baselines/run_relrag.py --dataset data/hotpotqa/dataset_distractor_200.json --workdir result_relrag/final_gate/20251222_113702/E5_hotpot_relrag_no_scheduler_run2 --context-budget 2000 --limit 20 --no-scheduler --lm-endpoint http://127.0.0.1:8000/v1 --lm-model qwen3-30b-a3b --embed-model /home/wjk/models/hf-cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf --embed-device cpu --max-new-tokens 128`
  - `python scripts/evaluate_relrag.py --dataset data/hotpotqa/dataset_distractor_200.json --dataset-name hotpotqa --workdir result_relrag/final_gate/20251222_113702/E5_hotpot_relrag_no_scheduler_run2 --ks 1,3,5,10`
- `E7_hotpot_relrag_llm_a_run2` (limit=5)
  - `NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLM_PROFILE=qwen3_30b_a3b python scripts/hotpotqa/baselines/run_relrag.py --dataset data/hotpotqa/dataset_distractor_200.json --workdir result_relrag/final_gate/20251222_113702/E7_hotpot_relrag_llm_a_run2 --context-budget 2000 --limit 5 --lm-endpoint http://127.0.0.1:8000/v1 --lm-model qwen3-30b-a3b --embed-model /home/wjk/models/hf-cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf --embed-device cpu --max-new-tokens 128`
  - `python scripts/evaluate_relrag.py --dataset data/hotpotqa/dataset_distractor_200.json --dataset-name hotpotqa --workdir result_relrag/final_gate/20251222_113702/E7_hotpot_relrag_llm_a_run2 --ks 1,3,5,10`
- `E7_hotpot_relrag_llm_b_run2` (limit=5)
  - `NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLM_PROFILE=qwen3_30b_a3b python scripts/hotpotqa/baselines/run_relrag.py --dataset data/hotpotqa/dataset_distractor_200.json --workdir result_relrag/final_gate/20251222_113702/E7_hotpot_relrag_llm_b_run2 --context-budget 2000 --limit 5 --lm-endpoint http://127.0.0.1:8000/v1 --lm-model qwen3-30b-a3b --embed-model /home/wjk/models/hf-cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf --embed-device cpu --max-new-tokens 128`
  - `python scripts/evaluate_relrag.py --dataset data/hotpotqa/dataset_distractor_200.json --dataset-name hotpotqa --workdir result_relrag/final_gate/20251222_113702/E7_hotpot_relrag_llm_b_run2 --ks 1,3,5,10`
- `mirage_b2000_run2`
  - `NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLM_PROFILE=qwen3_30b_a3b python scripts/mirage/run_vanilla_rag.py --dataset-path data/mirage_sample_200/dataset.json --workdir result_relrag/final_gate/20251222_113702/mirage_b2000_run2 --context-budget 2000 --limit 20 --retriever bm25 --topk 10 --lm-endpoint http://127.0.0.1:8000/v1 --lm-model qwen3-30b-a3b`
  - `python scripts/evaluate_relrag.py --dataset data/mirage_sample_200/dataset.json --dataset-name mirage --workdir result_relrag/final_gate/20251222_113702/mirage_b2000_run2 --ks 1,3,5,10`

Key metrics (run2):
| Run ID | invalid_rate | no_final_tag_rate | budget_violation_rate | Result |
| --- | --- | --- | --- | --- |
| E1_hotpot_relrag_b2000_run2 | 0.0 | 1.0 | 0.0 | PASS (format soft-fail) |
| E2_hotpot_relrag_retrieval_run2 | 0.0 | 1.0 | 0.0 | PASS (format soft-fail) |
| E5_hotpot_relrag_hop2_run2 | 0.0 | 1.0 | 0.0 | PASS (format soft-fail) |
| E5_hotpot_relrag_no_scheduler_run2 | 0.0 | 1.0 | 0.0 | PASS (format soft-fail) |
| E7_hotpot_relrag_llm_a_run2 | 0.0 | 1.0 | 0.0 | PASS (format soft-fail) |
| E7_hotpot_relrag_llm_b_run2 | 0.0 | 1.0 | 0.0 | PASS (format soft-fail) |
| mirage_b2000_run2 | 0.0 | 0.35 | 0.0 | PASS (format soft-fail) |

E5 config differences verified via `config.resolved.json`:
- hop=2 run: `graph_hop=2`, `scheduler_enabled=true`
- no-scheduler run: `graph_hop=1`, `scheduler_enabled=false`

E7 profile verified via `config.resolved.json`: llm_profile=qwen3_30b_a3b.

## D. Workdir Artifact Check (run2)
All run2 workdirs contain the required files:
- `preds/pred_raw.jsonl` (fields: id, question, pred_raw, contexts_used, context_tokens_used, context_budget_tokens)
- `preds/pred_norm.jsonl`
- `preds/pred_final.jsonl`
- `artifacts/retrieval.jsonl`
- `metrics/qa_metrics_norm.json`
- `metrics/qa_metrics_final.json`
- `metrics/format_metrics.json` (invalid/no_final/budget_violation)
- `metrics/retrieval_metrics.json`
- `config.resolved.json` (includes decode, budgets, embedding, llm_profile, git_commit)
- `run.log`
- `summary.json` (includes eval_entry, pred_source, git_commit)

## Issues (Non-blocking)
1. FINAL protocol compliance: All HotpotQA runs have `no_final_tag_rate=1.0`, Mirage `no_final_tag_rate=0.35`. This is reported as a format soft-fail and does not block PASS under the updated gate.

Paper wording recommendation:
- State that FINAL-tag compliance is monitored as a formatting metric but does not gate acceptance; evaluation uses `pred_raw` with extraction on the evaluation side.

## Minimal Fix PR (applied)
- Removed generation-path cleaning utilities from shared runners and moved evaluation-only normalization into `utils/output_eval.py`.
- Added hop/scheduler flags to Hotpot RelRAG and wired E5 ablation CLI args.
- Ensured workdir metadata includes decode/budgets/embedding/llm_profile/git_commit; summary.json now includes eval_entry/pred_source/git_commit.
- Added retrieval logging + budget fields for direct LLM (Mirage) and added `--topk` to Mirage vanilla RAG.
- Strengthened FINAL instruction text (no material improvement in `no_final_tag_rate`).

## Recommended Follow-Up
- The model is ignoring the FINAL protocol; consider a stricter system prompt or a dedicated response-format enforcement (server-side or API-level) that guarantees a `FINAL:` line without post-processing.
