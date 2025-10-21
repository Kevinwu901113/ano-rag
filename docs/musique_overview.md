# ANO-RAG Musique 批处理与系统概览

本文档对 ANO-RAG 系统的架构与 `main_musique.py` 脚本的批处理流程进行概述与说明，帮助你快速理解从文档处理到查询答复的完整路径，以及关键配置与使用方法。

参考资料： [Kevinwu901113/ano-rag | DeepWiki](https://deepwiki.com/Kevinwu901113/ano-rag)

## 项目简介
ANO-RAG 是一个检索增强生成（Retrieval Augmented Generation, RAG）系统，旨在将非结构化文档转换为可检索的知识库，并使用多种检索策略回答复杂多跳问题。系统将文档拆分为“原子笔记”，构建知识图谱，并结合向量检索、BM25 词法匹配与图遍历进行查询处理。

- 阶段划分：
  - 文档处理（构建知识库）：分块 → 原子笔记生成 → 向量嵌入 → 知识图谱 → 向量索引
  - 查询处理（回答问题）：混合检索召回 → 证据融合与排序 → LLM 生成最终答案 → 答案校验与评分
- 配置中心：使用 YAML 配置文件统一管理参数，并在运行时通过增强的配置加载模块进行缓存与同步。
- LLM 选择：支持本地/在线/混合路由，兼容多种 Provider（如 Ollama、LM Studio、OpenAI、Transformers）。

## 系统架构（核心组件）
- 配置模块：`config/config_loader.py`、`config.yaml`
- 文档处理：`doc/document_processor.py`、`llm/atomic_note_generator.py`、`llm/parallel_task_atomic_note_generator.py`
- 向量嵌入与索引：`vector_store/embedding_manager.py`、`vector_store/vector_index.py`
- 图谱构建与检索：`graph/graph_builder.py`、`graph/index.py`、`graph/graph_retriever.py`
- 查询处理：`query/query_processor.py`、`context/structure_pack.py`、`utils/context_dispatcher.py`
- LLM 层：`llm/local_llm.py`、`llm/ollama_client.py`、`llm/lmstudio_client.py`、`llm/openai_client.py`、`llm/multi_model_client.py`

DeepWiki 概览指出系统通过混合 LLM 路由与三重检索（向量、BM25、图）协同工作，支持并行化策略以提升整体吞吐与响应性能（参考：Kevinwu901113/ano-rag | DeepWiki）。

## `main_musique.py` 脚本概览
该脚本用于批量处理 Musique 测试集；对每个样本：
- 将每个段落保存为独立文档（JSON 文件），严格保证 question 不参与文档构建。
- 调用文档处理流程构建原子笔记与知识图谱。
- 使用 question 进行查询并生成最终答案与支持证据索引。
- 输出包含 `predicted_answer`、`predicted_support_idxs` 等结果，另保存详细的召回/选择日志，便于审计与复盘。

关键特性：
- 工作目录管理：自动创建版本化工作目录（`result/N/`），并将所有存储路径重定向到该目录下（向量、图谱、缓存、索引等）。
- 并行与共享：预初始化共享 `EmbeddingManager`，并支持线程池并行处理；可选引擎级并行策略。
- LLM 复用：入口初始化并共享一个 `LocalLLM` 实例给文档处理与查询处理；查询阶段根据配置也可采用混合或专用客户端策略。
- CoR（检索链）可选：启用 `--enable-cor` 时，将在笔记图谱上进行链式检索并融合答案与元数据（置信度、证据实体等）。
- 调试与审计：调试模式下保留中间文件；提示语审计写入 `promptin.log`，便于追踪 LLM 输入与输出。

## 处理流程（逐项）
1. 段落文件生成：每个 `item.paragraphs[i]` 生成一个独立 JSON 文件，结构保持一致（仅段落，不含 question）。
2. 文档处理：
   - `DocumentProcessor.process_documents(...)` 生成原子笔记（必要时并行/分配），并写入向量与图谱相关产物。
   - 若未生成原子笔记，返回占位结果并记录原因。
3. 查询处理：
   - 初始化 `QueryProcessor`（传入原子笔记、可选图文件、嵌入矩阵等）。
   - 执行混合检索与上下文构建，调用 LLM 生成最终答案与评分。
   - 提取答案与支持段落索引，汇总候选与选中笔记的明细。
4. 产物写出与清理：
   - 非调试模式下清理段落临时文件；调试模式保留便于定位问题。
   - 写出 `atomic_notes_recall.jsonl` 与 `selected_atomic_notes.jsonl` 两个细粒度审计文件。

## 存储与输出
- 工作目录：`result/N/`（自动递增），含 `vector_store/`、`graph_store/`、`processed/`、`cache/`、`vector_index/`、`embedding_cache/` 等。
- 召回审计：`atomic_notes_recall.jsonl`（所有候选）与 `selected_atomic_notes.jsonl`（最终选中）。
- 图与嵌入：`graph.json`（节点-边结构），`embeddings.npy`（NumPy 形式）。
- 最终结果：按样本写出包含 `predicted_answer` 与 `predicted_support_idxs` 的条目。

## LLM 策略与配置要点
- 入口共享：脚本入口创建共享 `LocalLLM` 实例并传入处理器。
- Provider 与客户端：支持 `ollama`、`lmstudio`、`openai`、`transformers`，以及混合模式 `hybrid_llm`；并可在原子笔记与最终答案阶段采用不同策略。
- 典型用法：
  - 文档阶段倾向使用本地、并行与流式早停（如 Ollama/LM Studio）。
  - 查询阶段可使用同一个 `LocalLLM` 或根据配置启用混合路由/专用客户端。
- 审计与评分：查询阶段构造上下文提示语并写入 `promptin.log`；支持答案评分与重试解析。

## 并行与性能
- 线程池并行：面向数据项或任务的并发处理。
- 引擎并行（可选）：`ParallelInterface` 与 `ParallelEngine` 提供任务处理器（文档/查询/Musique）与策略控制（如数据拆分、任务分发、混合）。
- 任务分配：原子笔记生成可按轮询或分批策略在多个客户端（如 Ollama 与 LM Studio）间分发，并在失败时回退。

## 命令行与运行示例
- 常用参数（以脚本实际 `--help` 为准）：
  - `--work-dir`：指定工作目录（默认自动创建 `result/N/`）。
  - `--debug`：启用调试模式，保留中间产物与更多日志。
  - `--use-engine-parallel`、`--parallel-workers`、`--parallel-strategy`：控制引擎并行策略与线程数。
  - `--enable-cor`：启用 Chain-of-Retrieval，融合图谱证据与答案。
  - 以及输入/输出文件相关参数（具体名称请查看脚本帮助）。
- 运行示例：
  - `python main_musique.py --work-dir ./work --debug --parallel-workers 4 --parallel-strategy hybrid`
  - 按需新增输入与输出文件参数，例如 `--input-file <path>`、`--output-file <path>`（具体以脚本帮助为准）。

## 适配与排查建议
- 服务连通性：本地服务（Ollama/LM Studio）不可达时检查 `base_url`、端口与模型是否就绪；增大 `timeout` 或调整并发数。
- OpenAI 兼容：确保 `api_key` 有效并满足速率限制；如走代理或私有部署，配置 `base_url`。
- 数据一致性：确保段落文件结构与项目标准一致（仅段落，无 question），避免污染笔记内容。
- 观察日志：查看 `promptin.log`、召回/选择 JSONL 文件以审计提示语与检索路径。

## 参考
- 系统架构与数据流概览参考： [Kevinwu901113/ano-rag | DeepWiki](https://deepwiki.com/Kevinwu901113/ano-rag)
- 核心代码入口与 Musique 批处理：`main_musique.py`
- 文档处理与查询处理的具体实现：`doc/` 与 `query/` 目录下各模块