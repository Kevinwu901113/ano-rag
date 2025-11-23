# Baselines / 对比实验

集中存放简单或外部的对比实验实现，方便与主结构化 RAG 流程并行评测。

- `direct_llm`：不做检索、直接用 LLM 先验回答数据集问题，输出 `answers.json`/`answers_direct_llm.jsonl` 与 `qa.tsv`。可用脚本 `scripts/mirage/run_direct_llm.py`。
- `naive_rag`：朴素向量 RAG 基线，用于 MIRAGE 示例数据。入口脚本见 `scripts/mirage/build_naive_index.py` 与 `scripts/mirage/run_naive_rag.py`。
- 后续新增的 LightRAG、GraphRAG 等对比方法也可以按子目录放在这里，保持独立的依赖与脚本。
