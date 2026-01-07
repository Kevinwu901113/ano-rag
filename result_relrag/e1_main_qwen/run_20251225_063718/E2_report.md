# E2 Retrieval vs QA Diagnosis

## 数据可用性说明
- 仅在 result_relrag/e1_main_qwen/run_20251225_063718 目录下的 artifacts/preds/metrics 中发现检索结果与预测文本。
- 未发现包含 per-query gold_ids 或 gold answers 的文件（用于 Hit@5 与 EM/F1 判定）。
- 因此四象限统计与样例分类无法在不引入外部数据的前提下完成。

## B1. 四象限统计表
- 计算规则（按需求定义）：
  - Retrieval Hit@5：retrieval.jsonl，k=5，hit=是否存在 canonical_id ∈ gold_ids
  - QA Correctness：correct = (EM == 1) 或 (F1 > 0)

### HotpotQA

| method | Hit & Correct | Hit & Wrong | Miss & Correct | Miss & Wrong |
| --- | --- | --- | --- | --- |
| bm25 | N/A | N/A | N/A | N/A |
| dense | N/A | N/A | N/A | N/A |
| hybrid | N/A | N/A | N/A | N/A |
| raptor | N/A | N/A | N/A | N/A |
| relrag_full | N/A | N/A | N/A | N/A |

### MuSiQue

| method | Hit & Correct | Hit & Wrong | Miss & Correct | Miss & Wrong |
| --- | --- | --- | --- | --- |
| bm25 | N/A | N/A | N/A | N/A |
| dense | N/A | N/A | N/A | N/A |
| hybrid | N/A | N/A | N/A | N/A |
| raptor | N/A | N/A | N/A | N/A |
| relrag_full | N/A | N/A | N/A | N/A |

### MIRAGE

| method | Hit & Correct | Hit & Wrong | Miss & Correct | Miss & Wrong |
| --- | --- | --- | --- | --- |
| bm25 | N/A | N/A | N/A | N/A |
| dense | N/A | N/A | N/A | N/A |
| hybrid | N/A | N/A | N/A | N/A |
| raptor | N/A | N/A | N/A | N/A |
| relrag_full | N/A | N/A | N/A | N/A |


## B2. 方法级诊断总结
- 基于当前可用 artifacts/preds/metrics，无法判定各方法主要失败于检索阶段或生成阶段（缺少 per-query gold_ids 与 EM/F1）。
- 特别对比（RelRAG vs Hybrid、LightRAG vs Dense、自带 Self-RAG 行为）均需四象限统计支持，当前无法给出。
- LightRAG / Self-RAG 结果目录未出现于本次 run_20251225_063718。

## B3. 典型样例分析
- 需求为每个数据集至少 5 条 Hit & Wrong 和 5 条 Miss & Correct，但当前无法判定命中与正确性。
- 若补充 gold_ids 与 gold answers，可基于 retrieval.jsonl + pred_raw.jsonl 直接复现统计与样例抽取。
