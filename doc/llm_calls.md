# 项目内大模型（LLM/Embedding）调用总览

本文档系统性整理 Ano-RAG 项目中所有涉及大模型推理的调用点、端点、请求负载、并发与重试策略，以及相关配置入口，便于统一审计与后续优化。

## 概览
- 生成阶段（vLLM）：`generator/note_generator.py::NoteGenerator` 调用 OpenAI 兼容 `chat/completions` 端点抽取严格 JSON 的“原子笔记”。
- 查询阶段（LM Studio）：`generator/answerer.py::call_lmstudio` 调用 `chat/completions` 端点生成最终自然语言回答。
- 证据压缩（LM Studio/兼容端）：`generator/extractor.py::EvidenceExtractor` 对候选笔记进行判定与压缩，返回严格 JSON（keep/summary/labels）。
- 重排（LLM）：`retriever/rerank.py::LLMReranker` 分批调用 `chat/completions` 端点为候选笔记打分与标注。
- 向量嵌入（本地模型）：`utils/embedding_utils.py::EmbeddingEncoder` 使用 `transformers` 或 `sentence_transformers` 在本地编码文本；`retriever/embedding_client.py` 与 `indexer/embedding_index.py` 负责在线检索与离线索引构建。

> 以上所有 HTTP LLM 调用均使用 OpenAI 兼容接口 `POST {endpoint}/chat/completions`。

## 详细调用点

### 1) 生成原子笔记（vLLM）
- 位置：`generator/note_generator.py`
- 类/方法：`NoteGenerator._call(prompt, stop?, max_tokens?)`
- 端点与模型：
  - `endpoint`：来自 CLI 或 `config.vllm.endpoint`，可被 `config.vllm.concurrency.endpoints`（列表）或环境变量 `VLLM_ENDPOINT{N}` 覆盖，实现多端点轮询。
  - `model`：`config.vllm.model`。
- 请求负载（payload）：
  - `model`: `<string>`
  - `temperature`: 来自 `config.vllm.temperature`
  - `max_tokens`: 默认 `min(config.vllm.max_tokens, 2000)`，可传入上限（解析阶段通常受 `config.parsing.max_tokens` 与 stop 序列约束）。
  - `messages`: `[{"role":"user","content": <严格 JSON 提示词>}]`
  - 可选 `stop`: 来自解析配置 `config.parsing.stop`
- 并发与连接：
  - 线程池策略：由上层构建器维持满载并发，线程数由 `config.vllm.concurrency.max_workers` 指定。
  - 会话：每线程一个 `requests.Session`，连接池大小约为 `max(16, max_workers*2)`；`strict_endpoint` 模式下设置 `Connection: close`，否则 `keep-alive`。
  - 端点池：支持轮询（`itertools.cycle`），维护健康标记；故障时熔断黑名单，半开恢复延迟 `config.vllm.concurrency.blacklist_duration_sec`（默认 15s）。
- 超时与重试：
  - 超时：连接超时 `connect_timeout_sec`（默认 3.05s）/ 读取超时 `read_timeout_sec`（默认 20s）。
  - 重试：`retry_max_attempts`（默认 2）；退避参数：`retry_backoff_base`、`retry_backoff_max_sec`、`retry_jitter_frac`；总时长上限 `retry_total_cap_sec`。
  - 失败处理：在 `timeout/http` 失败时重置会话并将端点标记为不健康。
- 适配与观测：
  - 自适应并发：可选 `config.vllm.adaptive.*`（目标延迟 P50/P95、步进、冷却等），通过延迟采样调整工作线程数。
  - 日志：`endpoint_log_every` 控制端点选择日志频率；阶段计时（call/parse/validate）。
- 提示词要点：`build_prompt(doc_text, doc_id)` 要求严格 JSON/JSONL，禁止代词主体，提供 `meta.attribute` 与 `subject_profile/object_profile` 结构，附带中英文规范说明与示例。

### 2) 最终答案生成（LM Studio）
- 位置：`generator/answerer.py`
- 函数：`call_lmstudio(endpoint, model, question, evidences, temperature=0.2, max_tokens=64, retries=2)`
- 端点与模型：
  - `endpoint`/`model`：来自 `QueryProcessor` 初始化参数或 `config.lmstudio.*`。
- 请求负载（payload）：
  - `model`, `temperature`, `max_tokens`
  - `messages`: `[{"role":"user","content": <ANS_PROMPT>}]`
  - `ANS_PROMPT`：严格指示“仅使用证据回答，不足则输出 Insufficient evidence”，证据以 `canonical | original` 对齐展示。
- 超时与重试：
  - HTTP 超时：固定 60 秒。
  - 失败重试：指数退避（`sleep(2**attempt)`），默认重试 2 次。
- 证据预处理：
  - `_compress_evidence(...)` 先调用证据压缩（见下节），返回带 `summary` 的证据以缩短上下文；失败时回退原证据集合。

### 3) 证据压缩与判定（LLM）
- 位置：`generator/extractor.py`
- 类/方法：`EvidenceExtractor.judge_and_compress(question, notes)`
- 端点与模型：
  - 从 `llm_cfg = {endpoint, model, timeout_s}` 传入；默认超时 15 秒。
- 请求负载（payload）：
  - `model`, `messages: [{"role":"user","content": EXTRACT_PROMPT}]`, `temperature: 0.0`, `max_tokens: 128`
  - `EXTRACT_PROMPT` 要求返回严格 JSON：`{"keep": bool, "summary": str, "labels": [str]}`。
- 失败处理：
  - 异常时记录日志并回退：`{"keep": true, "summary": 原 evidence, "labels": ["fallback"]}`。
- 解析：
  - `_parse_response(content)` JSON 解析与容错；字符串标签转换为列表。

### 4) 候选重排（LLM）
- 位置：`retriever/rerank.py`
- 类/方法：`LLMReranker.score(question, candidates)`（按 `batch` 批次循环）
- 端点与模型：
  - 来自 `cfg.reranker.llm` 或传入的 `lm_cfg`，默认指向 `config.lmstudio.*`。
- 请求负载（payload）：
  - `model`, `messages: [{"role":"user","content": PROMPT_TEMPLATE}]`, `temperature: 0.0`, `max_tokens: 64`
  - `PROMPT_TEMPLATE` 要求返回包含 `idx/score/labels` 的严格 JSON 列表。
- 超时与失败回退：
  - `timeout_s`（默认 10 秒）。
  - 异常时回退到词汇重叠度打分（`_fallback_scores`），并打标 `labels=["fallback"]`。
- 解析：
  - `_parse_scores` 支持提取非标准包裹文本中的 JSON 数组；字符串标签归一为列表。

## 向量嵌入调用（本地推理）

### 5) 在线嵌入检索（FAISS + 本地编码）
- 位置：`retriever/embedding_client.py`
- 类/方法：`EmbeddingClient.search(question, topn)`
- 行为：
  - 加载离线 FAISS 索引与元数据（`indexes/faiss/*.faiss`、`*.parquet`）。
  - 使用共享编码器 `EmbeddingEncoder` 在本地将 `question` 编码为向量；可选 L2 归一化。
  - 配置 FAISS 参数（如 `nprobe`、`efSearch`），在索引中检索相似项并返回 note_id/score。
- 配置：`config.retriever.embedding.*`
  - `provider`: `"qwen3"`（transformers）或 `"st"`（sentence_transformers）
  - `model`: 例如 `Qwen/Qwen3-Embedding-8B`
  - `max_len_note`, `topn`, `faiss.kind`, `normalize`

### 6) 离线嵌入索引构建
- 位置：`indexer/embedding_index.py`
- 类/方法：`EmbeddingIndexBuilder.build()`
- 行为：
  - 读取 `notes.jsonl`，用 `build_note_text_for_embed` 生成文本；调用 `EmbeddingEncoder.encode(texts)` 本地编码。
  - 可选 L2 归一化，训练/追加写入 FAISS 索引，更新元数据 parquet。
- 配置：沿用 `config.retriever.embedding.*`。

#### 嵌入配置 / 预下载
- 关键键：`retriever.embedding.cache_dir`（Hugging Face 缓存位置）、`download_dir`（`huggingface-cli --local-dir` 默认目录）、`model_path_override`（本地模型文件夹）、`device`（默认继承 `system.device`）、`dtype`（如 `bfloat16`/`float16` 限制显存）、`auto_build`（`scripts/mirage/build_notes.sh` 结束后自动执行 `scripts/build_indexes.sh`）。
- 自动覆盖：环境变量 `EMB_CACHE_DIR`、`EMB_MODEL_PATH`、`EMB_DOWNLOAD_DIR`、`EMB_DEVICE`、`EMB_DTYPE` 会在 `config_loader` 读取时覆盖对应配置，与 `VLLM_DOWNLOAD_DIR` 行为一致。
- 预下载脚本：`scripts/download_embedding_model.sh` 读取上述配置并封装 `huggingface-cli download`，可通过 `--model` / `--cache-dir` / `--local-dir` 指定 repo 与目标目录，适合在构建索引前预拉 `Qwen/Qwen3-Embedding-8B` 等大模型。
- 多源检索开关：`retriever.structured.enabled` / `retriever.embedding.enabled` / `retriever.bm25.enabled` 控制三路召回；混合检索由 `retriever/pipeline.py::_maybe_run_hybrid` 根据配置自动融合。

### 7) 向量搜索（仅兜底/预筛）
- 位置：`utils/vector_search.py`
- 类/方法：`VectorSearcher.search_in_notes/search_note_ids`
- 行为：
  - 使用本地 `SentenceTransformer`（默认 `all-MiniLM-L6-v2`）编码问题与笔记文本，进行余弦相似度检索，仅用于预筛或兜底，不参与最终融合排序。

## 配置总表与来源
- 文件：`config/config_loader.py`（默认值）与项目根 `config.yaml`（可覆盖）。
- 关键键：
  - `vllm.*`：`endpoint`、`model`、`temperature`、`max_tokens`、`concurrency.*`（`max_workers`、`endpoints`、`connect_timeout_sec`、`read_timeout_sec`、`retry_*`、`blacklist_duration_sec`、`endpoint_log_every`）、`adaptive.*`。
  - `lmstudio.*`：`endpoint`、`model`、`temperature`、`max_tokens`。
  - `reranker.llm.*`：`endpoint`、`model`、`batch`、`timeout_s`。
  - `retriever.embedding.*`：`enabled`、`provider`、`model`、`model_path_override`、`cache_dir`、`download_dir`、`device`、`dtype`、`offline_index_path`、`meta_path`、`max_len_note`、`faiss.*`、`normalize`、`topn`、`auto_build`。
  - `notes.*`、`chunk.*`、`parsing.*`：与生成/解析相关的辅助配置。
  - 环境变量：`VLLM_ENDPOINT{N}`（如 `VLLM_ENDPOINT0`, `VLLM_ENDPOINT1`）优先于配置实现多端点轮询；`EMB_CACHE_DIR` / `EMB_MODEL_PATH` / `EMB_DOWNLOAD_DIR` / `EMB_DEVICE` / `EMB_DTYPE` 覆盖嵌入相关配置；`ANO_RAG_CONFIG` 可指向任意配置文件，方便 per-run 覆盖。

## 端到端调用链
- 构建阶段：`main.py process` → `pipeline/structured_builder.py` → `NoteGenerator`（vLLM → JSON 解析 → 校验）→ 索引构建（倒排/图/嵌入/BM25）。
- 查询阶段：`main.py query` → 结构化检索（`retriever/pipeline.py`）→ 候选重排（`LLMReranker`，可回退）→ 证据压缩（`EvidenceExtractor`）→ 最终回答（`call_lmstudio`）。

## 安全与稳定性注意事项
- 所有对 `chat/completions` 的调用均要求严格 JSON 或受控输出；解析端有健壮容错与回退。
- vLLM 并发与会话复用需结合服务吞吐与 `--max-num-batched-tokens` 等参数调优；多端点下建议开启健康标记与短重试上限。
- LM Studio 与重排/压缩阶段均设有超时与回退，防止链路阻塞。
- 嵌入编码在本地完成，不依赖外部 HTTP；需准备 GPU/CPU 与对应库版本。

## 审计清单（代码引用）
- 生成：`generator/note_generator.py::_call` → `requests.Session.post("{endpoint}/chat/completions", json=payload, timeout=(connect,read))`
- 答案：`generator/answerer.py::call_lmstudio` → `requests.post("{endpoint}/chat/completions", json=payload, timeout=60)`
- 压缩：`generator/extractor.py::EvidenceExtractor.judge_and_compress` → `requests.post("{endpoint}/chat/completions", json=payload, timeout=timeout_s)`
- 重排：`retriever/rerank.py::LLMReranker.score` → `requests.post("{endpoint}/chat/completions", json=payload, timeout=timeout_s)`
- 嵌入：`utils/embedding_utils.py::EmbeddingEncoder.encode`（本地 `transformers/sentence_transformers`），`retriever/embedding_client.py`, `indexer/embedding_index.py`

## 维护建议
- 将关键提示词（生成/压缩/重排）抽出至独立模板文件，便于版本化与审查。
- 在 `config.yaml` 中集中管理端点与模型名，并提供环境变量覆盖说明（现已支持 vLLM 多端点）。
- 为 LLM 调用增加统一的指标采集与失败告警，完善可观测性。
