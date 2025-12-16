# HotpotQA Distractor200：Embedding-only baselines 的检索召回（CPU）

本页汇总 **只依赖 embedding（不调用 LLM）** 的 HotpotQA Distractor baselines，在 `data/hotpotqa/dataset_distractor_200.json`（200 条）上的检索指标（Title-level）。

## Baselines（embedding-only）

这些 baseline 都只在题内给定的 10 段 paragraph 上做向量检索/重排，不需要 LLM：
- `Vanilla RAG`：题内向量检索 top-k
- `Self-RAG`：`--retrieval-only` 下不做 LLM reflection，退化为题内向量检索 top-k
- `Mini RAPTOR`：`--retrieval-only` 下不生成 summaries，退化为题内向量检索 top-k
- `RelRAG`：相似度图 + 中心性/邻居分数重排（不调用 LLM）

## 运行配置（本次记录）

- 数据集：`data/hotpotqa/dataset_distractor_200.json`（200 samples，each has 10 context paragraphs）
- 设备：CPU（`--embed-device cpu`）
- Embedding 模型：`sentence-transformers/all-MiniLM-L6-v2`（本机 HF cache，离线可加载）
- Embedding 参数：`--embed-batch-size 32 --embed-max-length 256 --embed-normalize`
- 排序 top-k：`--topk 10`（用于计算 `@5/@10`）
- 离线：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`

## 复现命令

使用共享 embedding 的评测脚本（避免重复 encode 10 段 * 200 次）：

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
python scripts/hotpotqa/evaluate_embedonly_recall.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --embed-model /home/wjk/models/hf-cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf \
  --embed-device cpu \
  --embed-batch-size 32 \
  --embed-max-length 256 \
  --embed-normalize \
  --topk 10 \
  --ks 5,10 \
  --output-json result/hotpotqa_distractor200_embedonly/embedonly_recall_allminilm.json \
  --output-md result/hotpotqa_distractor200_embedonly/embedonly_recall_allminilm_table.md
```

## 结果（Title-level）

> 指标解释：`TitleRecall@k` 按 supporting_facts 的 gold titles 计算；`Hit@k` 表示 top-k 至少命中 1 个 gold title。

| run | TitleRecall@5 | TitleRecall@10 | TitlePrec@5 | TitlePrec@10 | Hit@5 | Hit@10 | count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hotpot_raptor_embedonly | 0.785 | 1.000 | 0.317 | 0.204 | 0.975 | 1.000 | 200 |
| hotpot_relrag_embedonly | 0.790 | 1.000 | 0.319 | 0.204 | 0.975 | 1.000 | 200 |
| hotpot_selfrag_embedonly | 0.785 | 1.000 | 0.317 | 0.204 | 0.975 | 1.000 | 200 |
| hotpot_vanilla_rag_embedonly | 0.785 | 1.000 | 0.317 | 0.204 | 0.975 | 1.000 | 200 |

原始输出：
- `result/hotpotqa_distractor200_embedonly/embedonly_recall_allminilm.json`
- `result/hotpotqa_distractor200_embedonly/embedonly_recall_allminilm_table.md`

