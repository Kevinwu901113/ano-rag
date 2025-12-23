## HotpotQA 基线建议命令

说明：各脚本会自动在 `result/hotpotqa` 下创建工作目录（如 `hotpot_direct_000`），默认输出 `pred.json` 和 `qa.tsv`，可用 `--work-dir`/`--result-root`/`--output`/`--qa-path` 覆盖。`--max-context` 默认 10（官方 distractor 正好 10 段）。

统一使用：
- 数据：`data/hotpotqa/dataset_distractor_200.json`
- 大模型：`--lm-endpoint http://127.0.0.1:8000/v1 \`  
          `--lm-model qwen3-30b-a3b`
- 嵌入模型：脚本默认 `Qwen/Qwen3-Embedding-8B`，可不显式指定；如需覆盖可加 `--emb-model ...`

### Direct（无检索）
```bash
python scripts/hotpotqa/baselines/run_direct.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b
```

### Vanilla RAG（10 段内检索 topk）
```bash
python scripts/hotpotqa/baselines/run_vanilla_rag.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b \
  --topk 3
```

### Self-RAG（检索 + LLM 反思挑段落）
```bash
python scripts/hotpotqa/baselines/run_selfrag.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b
```

### Mini RAPTOR（聚类+摘要树检索）
```bash
python scripts/hotpotqa/baselines/run_raptor.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b
```

### Mini GraphRAG（抽取三元组+图推理）
```bash
python scripts/hotpotqa/baselines/run_graphrag.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b
```

### RelRAG（相似度图重排）
```bash
python scripts/hotpotqa/baselines/run_relrag.py \
  --dataset data/hotpotqa/dataset_distractor_200.json \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b
```
