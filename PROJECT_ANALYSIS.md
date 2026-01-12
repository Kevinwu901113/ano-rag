# 项目结构与方法概览

## 项目定位与目标
本仓库实现了一套结构化 RAG（RelRAG）流程，并提供 HotpotQA 数据集的入口脚本。核心思路是：
- 用 LLM 从原始文档中抽取结构化“笔记”（主语/谓语/宾语）。
- 基于这些结构化笔记构建索引（FAISS + BM25 + 结构化索引）。
- 通过结构化 + 混合检索召回证据。
- 用证据约束的提示词让 LLM 生成最终答案。

## 顶层目录结构
- `hotpot_entry.py`：HotpotQA 入口脚本，按问题构建缓存、检索、回答，并输出 JSONL 和官方格式文件。
- `relrag/`：核心库（抽取、索引、检索、生成、配置等）。
- `data/`, `data_full/`：HotpotQA 数据集及 JSONL 版本。
- `result/`, `result_relrag/`：输出结果与缓存目录。
- `eval/`：HotpotQA 评估脚本（`hotpot_evaluate_v1.py`）。
- `run_vllm_dual_gpu.py`, `start_vllm_dual_gpu.sh`：vLLM 启动脚本。
- `logs/`：运行日志与 vLLM 启动记录。

## 核心流程（Build → Retrieve → Answer）

### 1) 文档预处理与切分
- 入口：`hotpot_entry.py` 将每个问题的 context 写入 `result/cache/<qid>/docs/*.txt`。
- 切分：`relrag/doc/chunker.py` 将文档切成 chunk，并附带 `doc_id` / `chunk_id` / 标题等元信息。

### 2) 结构化笔记抽取
- 抽取：`relrag/generator/note_generator.py` 调用 `relrag/utils/llm_client.py`，使用 `relrag/prompt/note_extract.txt` 提示词从文本中抽取结构化笔记。
- 解析与修复：`relrag/generator/note_parsing.py` 处理 LLM 结果，做 JSON 修复、字段补齐等。
- 验证与规范化：`relrag/validators/note_validator.py` 规范谓词、实体类型、证据字段，并生成 `note_id`。
- 代词处理与后处理：在 generator 与 postprocess 模块中进行，保证主语/宾语清晰可解析。

### 3) 索引构建
- 笔记输出：写入 `notes.jsonl`。
- 向量索引：`relrag/indexer/embedding_index.py` 构建 FAISS 索引。
- BM25 索引：`relrag/indexer/bm25_index.py` 构建 BM25 索引。
- 路径与参数来自 `relrag/config.yaml` 与 `relrag/config/config_loader.py`。

### 4) 检索
- 总控：`relrag/retriever/pipeline.py`。
- 查询解析：`relrag/retriever/parser.py` 将问题解析成 `QueryIR`（包含种子实体、谓词链、fanout）。
- 意图识别：`relrag/retriever/intent_detector.py` 推断问题类型与属性提示。
- 结构化检索：基于 BIND/EXPAND 进行路径搜索。
- 混合检索：可融合 embedding 和 BM25（由 `relrag/retriever/hybrid.py` 与配置控制）。
- 证据调度：选择强/弱证据并传给答案生成。

### 5) 回答生成
- `relrag/generator/answerer.py` 组织证据与问题，使用 `relrag/prompt/answerer.txt` 输出答案。
- 可选证据压缩：`relrag/generator/extractor.py`。
- 输出格式处理：`relrag/utils/output_eval.py` 提取 `FINAL:` 行作为最终答案。

## 配置与运行控制
- 默认配置：`relrag/config.yaml`。
- 环境变量覆盖：`ANO_RAG_CONFIG`、`EMB_ENDPOINT` 等在 `relrag/config/config_loader.py` 中生效。
- vLLM 参数：endpoint、model、并发、timeout 等。
- Retriever 参数：structured/hybrid/fusion 权重、fanout、embedding/BM25 开关。

## CLI 与脚本
- Build：`python -m relrag.cli.build`
- Retrieve：`python -m relrag.cli.retrieve`
- Answer：`python -m relrag.cli.answer`
- Hotpot 入口：`python hotpot_entry.py --data ... --cache_dir ... --output_dir ...`

## 数据与评估
- 数据集：`data/`, `data_full/`。
- 评估脚本：`eval/hotpot_evaluate_v1.py`。
- 输出：
  - `result_*.jsonl`：包含中间过程（检索结果、证据等）。
  - `result_*_official.json`：HotpotQA 官方格式（answer/sp）。

## 方法总结
整体采用结构化 RAG：先用 LLM 抽取结构化笔记，再建多种索引进行检索，最后基于证据生成答案。系统效果强依赖“问题解析 + 属性识别 + 结构化路径构造”；若解析失败或路径缺失，通常需要混合检索作为兜底。
