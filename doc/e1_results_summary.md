# E1 Results Summary

## 1. 实验说明
- **Dataset**: HotpotQA distractor_200, Mirage, Musique
- **Model**: qwen3-30b-a3b (vLLM)
- **Embedding**: /home/wjk/models/qwen3-emb (CPU)
- **Output Protocol**: FINAL-tag enforced

## 2. 结果汇总表
| Job ID | Dataset | Method | Budget | Count | EM | F1 | no_final_tag_rate |
|---|---|---|---|---|---|---|---|
| hotpotqa_distractor_200/bm25_rag/llm_default/budget_16384/full | hotpotqa_distractor_200 | bm25_rag | 16384 | 200 | 0.4950 | 0.6593 | 0.0200 |
| hotpotqa_distractor_200/bm25_rag/llm_default/budget_4096/full | hotpotqa_distractor_200 | bm25_rag | 4096 | 200 | 0.5000 | 0.6598 | 0.0200 |
| hotpotqa_distractor_200/bm25_rag/llm_default/budget_8192/full | hotpotqa_distractor_200 | bm25_rag | 8192 | 200 | 0.5000 | 0.6695 | 0.0200 |
| hotpotqa_distractor_200/dense_rag/llm_default/budget_16384/full | hotpotqa_distractor_200 | dense_rag | 16384 | 200 | 0.5400 | 0.6973 | 0.0150 |
| hotpotqa_distractor_200/dense_rag/llm_default/budget_4096/full | hotpotqa_distractor_200 | dense_rag | 4096 | 200 | 0.5300 | 0.6932 | 0.0150 |
| hotpotqa_distractor_200/dense_rag/llm_default/budget_8192/full | hotpotqa_distractor_200 | dense_rag | 8192 | 200 | 0.5550 | 0.7093 | 0.0100 |
| hotpotqa_distractor_200/hybrid_rag/llm_default/budget_16384/full | hotpotqa_distractor_200 | hybrid_rag | 16384 | 200 | 0.5150 | 0.6641 | 0.0150 |
| hotpotqa_distractor_200/hybrid_rag/llm_default/budget_4096/full | hotpotqa_distractor_200 | hybrid_rag | 4096 | 200 | 0.5200 | 0.6719 | 0.0100 |
| hotpotqa_distractor_200/hybrid_rag/llm_default/budget_8192/full | hotpotqa_distractor_200 | hybrid_rag | 8192 | 200 | 0.5050 | 0.6554 | 0.0150 |

## 3. 异常说明
No failed jobs.

## 4. 结论
- E1 是否全部跑通: NO
  - (Note: 1 jobs are still pending/incomplete)
- 结果是否稳定: (See rates)
- 是否可以作为论文主结果: NO