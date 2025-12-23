# Baselines Usage Guide

本文档整理了 `baselines` 目录下各个 RAG 基线模型的调用方法，包括索引构建（Index Building）和推理（Inference）两个阶段。

## HotpotQA Distractor Setting

HotpotQA Distractor setting has been unified to strictly use only the provided 10 paragraphs per question. The following scripts implement this restricted setting:

> 说明：所有 HotpotQA 基线会自动在 `result/hotpotqa` 下创建一个工作目录（例如 `hotpot_direct_000`），默认写入 `pred.json`（官方评测格式）和 `qa.tsv`（仅问题和清洗后的答案）。可以通过 `--output` / `--qa-path` 覆盖输出路径，或用 `--work-dir` / `--result-root` 自定义工作目录。

### 1. Direct / Naive
Concatenates 10 paragraphs and prompts LLM directly.
```bash
python scripts/hotpotqa/baselines/run_direct.py \
  --dataset path/to/hotpot_dev_distractor_v1.json \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b
```

### 2. Vanilla RAG (Distractor)
Builds an in-memory index for the 10 paragraphs and retrieves top-k before prompting.
```bash
python scripts/hotpotqa/baselines/run_vanilla_rag.py \
  --dataset path/to/hotpot_dev_distractor_v1.json \
  --topk 3
```

### 3. Self-RAG (Distractor)
Retrieves from 10 paragraphs, then uses LLM reflection to select best paragraphs.
```bash
python scripts/hotpotqa/baselines/run_selfrag.py \
  --dataset path/to/hotpot_dev_distractor_v1.json
```

### 4. Raptor (Distractor)
Clusters the 10 paragraphs into a mini-tree (Leaves -> Summaries) and retrieves.
```bash
python scripts/hotpotqa/baselines/run_raptor.py \
  --dataset path/to/hotpot_dev_distractor_v1.json
```

### 5. GraphRAG (Distractor)
Extracts triples from 10 paragraphs to build a mini-graph, then answers.
```bash
python scripts/hotpotqa/baselines/run_graphrag.py \
  --dataset path/to/hotpot_dev_distractor_v1.json
```

### 6. RelRAG (Distractor)
Builds a relation graph between paragraphs (similarity-based) and uses centrality to re-rank.
```bash
python scripts/hotpotqa/baselines/run_relrag.py \
  --dataset path/to/hotpot_dev_distractor_v1.json
```

---

## MuSiQue (Provided Paragraphs Setting)

MuSiQue 数据集为 `.jsonl`（每行一个样本），每个问题自带若干段落 `paragraphs`。本仓库的 MuSiQue 基线沿用 HotpotQA Distractor 的“提供段落”设定：**不依赖离线索引，按题内段落构建临时检索**。

> 说明：所有 MuSiQue 基线会自动在 `result/musique` 下创建工作目录（例如 `musique_vanilla_rag_000`），默认写入 `musique_results.jsonl`（官方格式，含 `predicted_answer`/`predicted_evidence`）和 `qa.tsv`。可通过 `--output` / `--qa-path` 覆盖输出路径。

### 1. Direct (No Retrieval)
直接拼接所有段落提示 LLM。
```bash
python scripts/musique/baselines/run_direct.py \
  --dataset data/musique/musique_full_v1.0_dev.jsonl \
  --lm-endpoint http://127.0.0.1:8000/v1 \
  --lm-model qwen3-30b-a3b
```

### 2. Vanilla RAG
对题内段落做向量检索后回答。
```bash
python scripts/musique/baselines/run_vanilla_rag.py \
  --dataset data/musique/musique_full_v1.0_dev.jsonl \
  --topk 3 \
  --emb-model Qwen/Qwen3-Embedding-8B
```

### 3. Self-RAG
先检索候选段落，再用 LLM 反思重排。
```bash
python scripts/musique/baselines/run_selfrag.py \
  --dataset data/musique/musique_full_v1.0_dev.jsonl \
  --topk 3
```

### 4. Raptor
对题内段落聚类生成摘要树后检索。
```bash
python scripts/musique/baselines/run_raptor.py \
  --dataset data/musique/musique_full_v1.0_dev.jsonl
```

### 5. GraphRAG
基于题内段落的简单图检索/重排。
```bash
python scripts/musique/baselines/run_graphrag.py \
  --dataset data/musique/musique_full_v1.0_dev.jsonl
```

### 6. RelRAG
段落相似度图 + 中心性重排。
```bash
python scripts/musique/baselines/run_relrag.py \
  --dataset data/musique/musique_full_v1.0_dev.jsonl
```

### 可选：Full‑Wiki Baselines

`baselines/` 下的 `naive_rag` / `fid_rag` / `simple_*` 等 Full‑Wiki 设定基线依赖全局 `doc_pool.json` 与离线索引。MuSiQue 原始数据无全局 doc_pool，可先用：

```bash
python scripts/musique/build_doc_pool.py \
  --dataset data/musique/musique_full_v1.0_dev.jsonl \
  --out data/musique/doc_pool.json
```

再复用 Mirage 的索引构建脚本（如 `python scripts/mirage/build_naive_index.py --doc-pool data/musique/doc_pool.json --out-dir result/musique_naive`）。推理脚本需自行加载 `.jsonl` 数据并调用对应 Runner。

## MIRAGE / Full-Wiki Baselines

(Below are existing instructions for full-retrieval settings)

## 1. Direct LLM (No Retrieval)

直接使用 LLM 回答问题，不进行任何检索。

**Python API:**

```python
from baselines.direct_llm.runner import DirectLLMRunner

runner = DirectLLMRunner(
    lm_endpoint="http://127.0.0.1:8000/v1", # 或从 config 读取
    lm_model="qwen3-30b-a3b"
)

# dataset 需包含 "question" 或 "query" 字段
results = runner.run_dataset(dataset, work_dir="./output/direct_llm")
```

## 2. Naive RAG

朴素 RAG：切分文档 -> 向量化 -> FAISS 检索 -> LLM 回答。

**构建索引 (Indexing):**

```python
from baselines.naive_rag.index import MirageNaiveIndexer

indexer = MirageNaiveIndexer()
# doc_pool_path 为 mirage 格式的文档池 jsonl
indexer.build(doc_pool_path="data/docs.jsonl", out_dir="data/indices/naive")
```

**推理 (Inference):**

```python
from baselines.naive_rag.runner import LLMClient, NaiveIndex

# 1. 初始化检索器
retriever = NaiveIndex(
    index_path="data/indices/naive/index.faiss",
    chunks_path="data/indices/naive/chunks.jsonl"
)

# 2. 检索
hits = retriever.search("What is X?", topk=5)

# 3. 生成回答
llm = LLMClient(endpoint="http://127.0.0.1:8000/v1", model="qwen3-30b-a3b")
context = "\n".join([h["text"] for h in hits])
answer = llm.answer(question="What is X?", context=context)
```

## 3. FiD RAG (Fusion-in-Decoder style)

检索后将每条 Passage 单独与 Question 拼接输入 LLM，要求 LLM 输出 Answer 和 引用的 Passage ID。

**构建索引:**
直接复用 `Naive RAG` 的索引（即 `NaiveIndex` 和对应的 chunks）。

**推理 (Inference):**

```python
from baselines.fid_rag.runner import FiDRAGRunner

runner = FiDRAGRunner(
    index_path="data/indices/naive/index.faiss",
    chunks_path="data/indices/naive/chunks.jsonl",
    topk=5
)

# dataset: list of dicts with "question"
results = runner.run_dataset(dataset, work_dir="./output/fid_rag")
```

## 4. Simple Raptor

递归树状摘要索引 (Recursive Abstractive Processing for Tree-Organized Retrieval)。

**构建索引 (Indexing):**

```python
from baselines.simple_raptor.index import SimpleRaptorIndexer

# 需配置 Embedding (支持 huggingface/vllm) 和 LLM
indexer = SimpleRaptorIndexer(
    embedding_config={
        "provider": "huggingface", 
        "model": "Qwen/Qwen3-Embedding-8B", 
        "device": "cuda:0"
    },
    llm_config={
        "endpoint": "http://127.0.0.1:8000/v1", 
        "model": "qwen3-30b-a3b"
    }
)

# docs: dict {doc_id: text}
# build 方法会执行：切分 -> 聚类 -> 摘要 -> 递归 -> 向量化所有节点
indexer.build(docs, cluster_size=16)

# 保存索引
indexer.save(
    index_path="path/to/index.faiss",
    nodes_path="path/to/nodes.pkl",
    chunk_store_path="path/to/chunk_store.pkl"
)
```

**推理 (Inference):**

```python
from baselines.simple_raptor.retriever import SimpleRaptorRetriever

retriever = SimpleRaptorRetriever(
    index_path="path/to/index.faiss",
    nodes_path="path/to/nodes.pkl",
    chunk_store_path="path/to/chunk_store.pkl",
    embedding_client=encoder, # 可选传入已初始化的 encoder
    llm_client=llm_client     # 可选传入已初始化的 llm
)

# 检索
nodes = retriever.retrieve("Question", top_k=5)

# 生成回答 (需自行调用 LLM)
context = "\n".join([n.text for n in nodes])
```

## 5. Simple Self-RAG

包含检索、生成、反思（Critique）循环的简化版 Self-RAG。

**构建索引 (Indexing):**

```python
from baselines.simple_selfrag.index import SimpleSelfRAGIndexer

indexer = SimpleSelfRAGIndexer(
    embedding_config={"provider": "huggingface", "model": "..."}
)

# docs: dict {doc_id: text}
indexer.build(docs)
indexer.save(index_path="path/to/index.faiss", chunks_path="path/to/chunk_store.pkl")
```

**推理 (Inference):**

```python
from baselines.simple_selfrag.retriever import SimpleSelfRAGRetriever

retriever = SimpleSelfRAGRetriever(
    index_path="path/to/index.faiss",
    chunk_store_path="path/to/chunk_store.pkl"
)

# answer 方法内部实现了 Retrieve -> Generate -> Critique -> (Retry) 流程
final_answer = retriever.answer("Question")
```

## 6. Simple GraphRAG

基于图的检索增强生成。

**构建索引 (Indexing):**

```python
import asyncio
from baselines.simple_graphrag.build_graph import GraphBuilder
from structrag.llm_client import LLMChatClient

llm_client = LLMChatClient(endpoint="...", model="...")
builder = GraphBuilder(llm_client=llm_client)

# docs: dict {doc_id: text}
# 异步构建
await builder.build(docs)

builder.save(
    graph_path="path/to/graph.pkl", 
    chunk_store_path="path/to/chunk_store.pkl"
)
```

**推理 (Inference):**

```python
from baselines.simple_graphrag.retriever import GraphRetriever

retriever = GraphRetriever(
    graph_path="path/to/graph.pkl",
    chunk_store_path="path/to/chunk_store.pkl",
    llm_client=llm_client
)
```
