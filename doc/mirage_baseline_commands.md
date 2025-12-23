## MIRAGE 主流程与基线命令

说明：各脚本默认在 `result/` 下创建工作目录（如 `mirage_naive_000`、`000-mirage`），可用 `--work-dir`/`--result-root`/`--new` 覆盖。所有 QA 命令都会写出 `qa.tsv`，便于用 `python evaluate_mirage.py data/mirage_sample/dataset.json <qa.tsv>` 评测。

统一使用：
- QA 数据：`data/mirage_sample/dataset.json`
- 文档池：`data/mirage_sample/doc_pool.json`
- 大模型：`--lm-endpoint http://127.0.0.1:8000/v1 \`  
          `--lm-model qwen3-30b-a3b`
- 嵌入模型：默认 `Qwen/Qwen3-Embedding-8B`，可在 `config.yaml` 或 CLI 覆盖。

### 结构化 RAG（项目主流程）
- 构建笔记 + 结构索引（自动启动单实例 vLLM（TP 多卡），输出 notes/indexes + `config.override.yaml`）：
```bash
DATA_DIR=data/mirage_sample \
bash scripts/mirage/build_notes.sh --new
```
- 用结构化检索批量回答 MIRAGE 数据集（默认读取工作目录里的 notes/indexes 与 override 配置）：
```bash
python scripts/mirage/query_dataset.py \
  --dataset mirage \
  --dataset-path data/mirage_sample/dataset.json \
  --work-dir result/000-mirage \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b \
  --limit 50
```

### Direct（无检索，直接调用 LLM）
```bash
python scripts/mirage/run_direct_llm.py \
  --dataset-path data/mirage_sample/dataset.json \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b \
  --new
```

### Naive RAG（向量检索）
构建索引：
```bash
python scripts/mirage/build_naive_index.py \
  --doc-pool data/mirage_sample/doc_pool.json \
  --out-dir result/mirage_naive
```
跑 QA：
```bash
python scripts/mirage/run_naive_rag.py \
  --dataset-path data/mirage_sample/dataset.json \
  --index-dir result/mirage_naive \
  --topk 5 \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b \
  --new
```

### FiD-style RAG
构建索引：
```bash
python scripts/mirage/build_fid_index.py \
  --doc-pool data/mirage_sample/doc_pool.json \
  --out-dir result/mirage_fid_index
```
跑 QA：
```bash
python scripts/mirage/run_fid_rag.py \
  --dataset-path data/mirage_sample/dataset.json \
  --index-dir result/mirage_fid_index \
  --topk 5 \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b \
  --new
```

### Vanilla RAG（检索 + LLM 作答）
```bash
python scripts/mirage/run_vanilla_rag.py \
  --dataset-path data/mirage_sample/dataset.json \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b \
  --new
```
未提供索引时会在工作目录下从 `doc_pool.json` 自动构建 `vanilla_rag_index.faiss` + `vanilla_rag_chunk_store.pkl`。

### Simple Self-RAG
```bash
python scripts/mirage/run_simple_selfrag.py \
  --dataset-path data/mirage_sample/dataset.json \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b \
  --new
```

### Simple RAPTOR（聚类摘要树）
构建索引：
```bash
python scripts/mirage/build_simple_raptor_index.py \
  --doc-pool data/mirage_sample/doc_pool.json \
  --out-dir result/mirage_raptor \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b
```
跑 QA：
```bash
python scripts/mirage/run_simple_raptor.py \
  --dataset-path data/mirage_sample/dataset.json \
  --index-dir result/mirage_raptor \
  --topk 5 \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b \
  --new
```

### Simple GraphRAG（抽取三元组 + 图推理）
```bash
python scripts/mirage/run_simple_graphrag.py \
  --dataset-path data/mirage_sample/dataset.json \
  --result-root result \
  --index-dir result/mirage_simple_graphrag \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b \
  --new
```
若 `simple_graphrag_graph.pkl` / `simple_graphrag_chunk_store.pkl` 不存在，会先读取 `doc_pool.json` 自动构建。
