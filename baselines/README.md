# Baselines / 对比实验

`baselines/` 目录集中存放可复用的对比实验实现（Indexer / Retriever / Runner），用于和 Ano‑RAG 主结构化检索流程做并行评测。  
数据集相关的一键脚本位于 `scripts/*`，这里更偏“算法最小实现 + Python API”。

## 目录说明

- `direct_llm/`：无检索，直接用 LLM 回答问题。
- `vanilla_rag/`：题内/全局向量检索 + LLM 作答的朴素 RAG。
- `naive_rag/`：面向 MIRAGE/Full‑Wiki 设定的向量索引构建与检索基线（FAISS + chunks）。
- `fid_rag/`：FiD‑style RAG（检索后将 top‑k passage 分别拼接输入 LLM）。
- `simple_raptor/`：递归摘要树（RAPTOR）检索基线。
- `simple_selfrag/`：Self‑RAG（Retrieve → Generate → Critique → Retry）简化实现。
- `simple_graphrag/`：抽取三元组并构图后做图推理的 GraphRAG 基线。
- `common/`：各基线共享的分块、Embedding、LLM Client 等轻量工具。

## 如何运行

- 具体命令与参数见 `baselines/USAGE.md`。
- 按数据集整理的推荐命令见：
  - `doc/mirage_baseline_commands.md`
  - `doc/hotpotqa_baseline_commands.md`
  - `doc/musique_baseline_commands.md`

如果你要新增 LightRAG / GraphRAG / 其他方法，请按子目录放置，并保持依赖与脚本相对独立。  
