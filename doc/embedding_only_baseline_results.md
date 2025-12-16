# Embedding-only baselines：本地可复现运行结果（不依赖 LM Studio / vLLM）

运行环境（本次记录）：
- 日期：2025-12-15
- CUDA：不可用（CPU-only）
- Embedding 模型：`/home/wjk/models/qwen3-emb`
- 统一参数：`--embed-device auto`（实际为 cpu）、`--embed-batch-size 2`、`--embed-max-length 256`、`--embed-normalize`
- 禁止联网下载：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`

> 说明：HotpotQA 的 `--retrieval-only` 会写 `pred.json`，但答案为空字符串，所以 Answer metrics 全 0；本文只关心 Retrieval metrics。

## MIRAGE（FiD index，检索指标）

构建索引（doc embedding + FAISS CPU）：
```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/mirage/build_fid_index.py \
  --doc-pool data/mirage_sample/doc_pool.json \
  --out-dir result/verify_mirage_fid_cpu \
  --embed-model /home/wjk/models/qwen3-emb \
  --embed-device auto \
  --embed-batch-size 2 \
  --embed-max-length 256 \
  --embed-normalize
```

评测 doc-level 检索召回（query embedding + FAISS search）：
```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/evaluate_mirage_retrieval.py naive \
  --dataset data/mirage_sample/dataset.json \
  --index-dir result/verify_mirage_fid_cpu \
  --ks 1,3,5
```

输出（20 条样本，`data/mirage_sample/dataset.json`）：
| run | DocRecall@1 | DocRecall@3 | DocRecall@5 | Hit@1 | Hit@3 | Hit@5 |
| --- | --- | --- | --- | --- | --- | --- |
| verify_mirage_fid_cpu | 0.600 | 0.700 | 0.700 | 0.600 | 0.700 | 0.700 |

## HotpotQA（retrieval-only baselines，检索指标）

本次只跑前 20 条（`--limit 20`），结果目录：`result/hotpotqa_embedonly/`。

Vanilla RAG（retrieval-only）：
```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/hotpotqa/baselines/run_vanilla_rag.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --result-root result/hotpotqa_embedonly \
  --new \
  --retrieval-only \
  --limit 20 \
  --num-workers 1 \
  --topk 10 \
  --embed-model /home/wjk/models/qwen3-emb \
  --embed-device auto \
  --embed-batch-size 2 \
  --embed-max-length 256 \
  --embed-normalize
```

Self-RAG（retrieval-only，不做 LLM reflection）：
```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/hotpotqa/baselines/run_selfrag.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --result-root result/hotpotqa_embedonly \
  --new \
  --retrieval-only \
  --limit 20 \
  --num-workers 1 \
  --topk 10 \
  --embed-model /home/wjk/models/qwen3-emb \
  --embed-device auto \
  --embed-batch-size 2 \
  --embed-max-length 256 \
  --embed-normalize
```

Mini RAPTOR（retrieval-only，不生成 summaries）：
```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/hotpotqa/baselines/run_raptor.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --result-root result/hotpotqa_embedonly \
  --new \
  --retrieval-only \
  --limit 20 \
  --num-workers 1 \
  --topk 10 \
  --embed-model /home/wjk/models/qwen3-emb \
  --embed-device auto \
  --embed-batch-size 2 \
  --embed-max-length 256 \
  --embed-normalize
```

RelRAG（retrieval-only，不调用 LLM）：
```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/hotpotqa/baselines/run_relrag.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --result-root result/hotpotqa_embedonly \
  --new \
  --retrieval-only \
  --limit 20 \
  --num-workers 1 \
  --topk 10 \
  --embed-model /home/wjk/models/qwen3-emb \
  --embed-device auto \
  --embed-batch-size 2 \
  --embed-max-length 256 \
  --embed-normalize
```

聚合评测（读取各 run 的 `retrieval.jsonl`）：
```bash
python scripts/evaluate_hotpotqa_metrics.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --root result/hotpotqa_embedonly \
  --output result/hotpotqa_embedonly/hotpot_metrics.json
```

输出（20 条样本，`data/hotpotqa/dataset_distractor_200.json` 的前 20 条）：
| run | TitleRecall@5 | TitleRecall@10 | TitlePrec@5 | TitlePrec@10 | Hit@5 | Hit@10 |
| --- | --- | --- | --- | --- | --- | --- |
| hotpot_raptor_000 | 0.375 | 1.000 | 0.150 | 0.200 | 0.600 | 1.000 |
| hotpot_relrag_000 | 0.375 | 1.000 | 0.150 | 0.200 | 0.600 | 1.000 |
| hotpot_selfrag_000 | 0.375 | 1.000 | 0.150 | 0.200 | 0.600 | 1.000 |
| hotpot_vanilla_rag_000 | 0.375 | 1.000 | 0.150 | 0.200 | 0.600 | 1.000 |

## 本次未跑的 baseline（原因：依赖 LLM）

- MIRAGE 的 QA baselines（`run_*_rag.py` / Direct / SimpleRAPTOR / SimpleGraphRAG 等）：需要 LLM 作答或生成 summaries/triples。
- HotpotQA 的 GraphRAG：需要 LLM 做三元组抽取/推理。
