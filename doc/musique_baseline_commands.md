## MuSiQue 主流程与基线命令

说明：MuSiQue 数据为 `.jsonl`（每行一个样本，含 `paragraphs`）。本仓库对 MuSiQue 的评测分两类：

1. **项目主流程（两阶段 StructRAG 管线）**：`scripts/musique/run.py` / `run_musique.sh`。
2. **提供段落设定的基线**：在每个问题给定段落内构建临时检索，不需要离线索引。

统一使用：
- 数据：`data/musique/musique_full_v1.0_dev.jsonl`（或 `data/musique_sample/musique.jsonl` 做快速验证）
- vLLM：`--vllm-endpoint http://127.0.0.1:8000/v1 \`  
        `--vllm-model qwen3-30b-a3b`
- 嵌入模型：基线默认 `Qwen/Qwen3-Embedding-8B`，可用 `--emb-model` 覆盖。

### 结构化 RAG（项目主流程）

推荐使用单卡编排脚本（自动拉起 vLLM → 跑管线 → 收尾）：
```bash
DATASET_PATH=data/musique/musique_full_v1.0_dev.jsonl \
bash scripts/musique/run_musique.sh --new --tag dev-run1
```

或直接跑 Python（需自行保证 vLLM 已启动）：
```bash
python scripts/musique/run.py \
  --dataset-path data/musique/musique_full_v1.0_dev.jsonl \
  --result-root result \
  --new --tag dev-run1 \
  --vllm-endpoint http://127.0.0.1:8000/v1 \
  --vllm-model qwen3-30b-a3b
```

输出位于 `result/musique_*` 工作目录下，包含 `notes/notes.musique.jsonl`、`answers/pred.jsonl` 和 `musique_results.jsonl`。

### Direct（无检索）
```bash
python scripts/musique/baselines/run_direct.py \
  --dataset data/musique/musique_full_v1.0_dev.jsonl \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b
```

### Vanilla RAG（题内段落向量检索）
```bash
python scripts/musique/baselines/run_vanilla_rag.py \
  --dataset data/musique/musique_full_v1.0_dev.jsonl \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b \
  --topk 3
```

### Self-RAG（检索 + LLM 反思重排）
```bash
python scripts/musique/baselines/run_selfrag.py \
  --dataset data/musique/musique_full_v1.0_dev.jsonl \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b \
  --topk 3
```

### Mini RAPTOR（聚类摘要树检索）
```bash
python scripts/musique/baselines/run_raptor.py \
  --dataset data/musique/musique_full_v1.0_dev.jsonl \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b
```

### Mini GraphRAG
```bash
python scripts/musique/baselines/run_graphrag.py \
  --dataset data/musique/musique_full_v1.0_dev.jsonl \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b
```

### RelRAG（相似度图重排）
```bash
python scripts/musique/baselines/run_relrag.py \
  --dataset data/musique/musique_full_v1.0_dev.jsonl \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b
```

### 可选：Full‑Wiki Baselines

如果希望复用 `baselines/` 下的 Full‑Wiki 设定基线，可先将 MuSiQue 段落汇总为全局 doc_pool：
```bash
python scripts/musique/build_doc_pool.py \
  --dataset data/musique/musique_full_v1.0_dev.jsonl \
  --out data/musique/doc_pool.json
```
然后用 Mirage 的索引构建脚本（如 `scripts/mirage/build_naive_index.py`）生成 FAISS/树/图索引，再自行加载 `.jsonl` 数据调用对应 Runner。
