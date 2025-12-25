## e1_cache_smoke run summary

- Config: `experiments/configs/e1_cache_smoke.yaml`
- Result root: `result_relrag/e1_cache_smoke/e1_cache_smoke`
- Cache key:
  `hotpotqa_distractor_200__home_wjk_models_qwen3-emb__cpu__len512__norm1__dtypeauto__batch4__maxctx10__indexin_memory_cosine__data8b74681d6856`
- Cache path:
  `result_relrag/e1_cache_smoke/cache/hotpotqa_distractor_200__home_wjk_models_qwen3-emb__cpu__len512__norm1__dtypeauto__batch4__maxctx10__indexin_memory_cosine__data8b74681d6856`

## Cache hit/miss evidence (run.log)

Cache miss (budget 4096, first run):
```
2025-12-25 04:04:54.451 | INFO | __main__:main:530 - Embedding cache miss: hotpotqa_distractor_200__home_wjk_models_qwen3-emb__cpu__len512__norm1__dtypeauto__batch4__maxctx10__indexin_memory_cosine__data8b74681d6856 -> result_relrag/e1_cache_smoke/cache/hotpotqa_distractor_200__home_wjk_models_qwen3-emb__cpu__len512__norm1__dtypeauto__batch4__maxctx10__indexin_memory_cosine__data8b74681d6856
```
Source: `result_relrag/e1_cache_smoke/e1_cache_smoke/run_20251225_040444/hotpotqa_distractor_200/dense_rag/llm_default/budget_4096/full/run.log`

Cache hit (budget 8192, same run):
```
2025-12-25 04:13:50.038 | INFO | __main__:main:520 - Embedding cache hit: hotpotqa_distractor_200__home_wjk_models_qwen3-emb__cpu__len512__norm1__dtypeauto__batch4__maxctx10__indexin_memory_cosine__data8b74681d6856 -> result_relrag/e1_cache_smoke/cache/hotpotqa_distractor_200__home_wjk_models_qwen3-emb__cpu__len512__norm1__dtypeauto__batch4__maxctx10__indexin_memory_cosine__data8b74681d6856
```
Source: `result_relrag/e1_cache_smoke/e1_cache_smoke/run_20251225_040444/hotpotqa_distractor_200/dense_rag/llm_default/budget_8192/full/run.log`

## Timing evidence

- Cache build duration (first build):
  `result_relrag/e1_cache_smoke/cache/hotpotqa_distractor_200__home_wjk_models_qwen3-emb__cpu__len512__norm1__dtypeauto__batch4__maxctx10__indexin_memory_cosine__data8b74681d6856/build.log` reports `duration_s=431.72`
- Cache hit run duration (budget 8192, completed run):
  start `04:13:50`, end `04:15:29` (~99s)
  from `result_relrag/e1_cache_smoke/e1_cache_smoke/run_20251225_040444/hotpotqa_distractor_200/dense_rag/llm_default/budget_8192/full/run.log`

## Format gate check

From summaries in `result_relrag/e1_cache_smoke/e1_cache_smoke/run_20251225_040444/.../summary.json`:
- invalid_rate: 0.0
- budget_violation_rate: 0.0
- no_final_tag_rate: 1.0 (FAIL, should be <= 0.05)
