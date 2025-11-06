# Ano-RAG（结构化版）

以三元组“原子笔记”为核心的最小结构化 RAG。通过 vLLM 提取规范化笔记、构建倒排与图索引；查询阶段以结构化检索获取证据，再交给 LM Studio 生成最终答案。

## 快速开始

- 安装依赖：`pip install -r requirements.txt`
- 配置服务：确保有可用的 vLLM 与 LM Studio HTTP 接口。
- 可选：在 `config.yaml` 中设置 `vllm.*`、`lmstudio.*`、`notes.*`、`chunk.*`。

构建笔记与索引：

```bash
python main.py process \
  --data-dir data/sample \
  --vllm-endpoint http://127.0.0.1:8000/v1 \
  --vllm-model qwen2.5-7b-instruct
```

- 输出文件：`notes/notes.jsonl`、`notes/chunks.jsonl`、`indexes/`（见下文）

查询：

```bash
python main.py query \
  "Who is the spouse of the Green performer?" \
  --lmstudio-endpoint http://127.0.0.1:1234/v1 \
  --lmstudio-model openai/gpt-oss-20b
```

## 项目结构与流程

- `pipeline/structured_builder.py`：统一构建器。遍历 `data-dir` 下的 `.txt/.md/.jsonl/.json` 文件，分块并调用生成器，写出 `notes.jsonl` 与索引。
  - 现已支持高效并发：线程池保持“满载”提交（不再按 `batch_size` 限制并行度），通过 `config.vllm.concurrency.max_workers` 控制并发线程数；可配置多端点轮询以充分利用多服务实例。
- `doc/chunker.py`：句级滑窗分块，输出 `{doc_id, chunk_id, text}`。分块参数来自 `config.chunk`（`n_sent`、`overlap`）。
- `generator/note_generator.py`：构造严格 JSON 提示词，调用 vLLM，使用 `generator/note_parsing.py` 纠错解析，再用 `validators/note_validator.py` 校验与归一化输出。
- `indexer/index_builder.py`：从 `notes.jsonl` 构建：
  - `entity_to_notes.json` / `predicate_to_notes.json`
  - `graph_edges.jsonl` / `inverse_edges.jsonl`
  - `type_edge_index.json` / `domain_index.json`
  - `field_index.json` / `entity_alias_index.json`
  - `manifest.json`
- `retriever/`：将问题解析为约束 IR，执行 BIND/EXPAND 等关系算子，得分与兜底策略见 `retriever/pipeline.py`。
- `generator/answerer.py`：把结构化检索得到的证据传给 LM Studio 生成最终自然语言答案。
- `main.py`：CLI 入口，含 `process` 与 `query` 两条子命令。

## 数据输入与分块

- 支持扩展名：`.txt`、`.md`、`.jsonl`、`.json`
- 对于 `.jsonl`：逐行读取对象，优先取 `text` 或 `content` 字段作为原文。
- 分块输出示例：
  - `{"doc_id": "doc1", "chunk_id": "c0000", "text": "..."}`
- 构建阶段会同时写出 `notes/chunks.jsonl`，便于复核分块质量。

## 笔记与验证

- 生成器提示词要求每条笔记含键：`subj`, `pred`, `obj`, `subj_type`, `obj_type`, `evidence`, `meta`。
- 类型限定：`PERSON | WORK | ORG | PLACE | EVENT | CONCEPT | TIME`。
- `validators/note_validator.py` 会：
  - 归一谓词（含同义映射与职业/title规约），为 `meta.attribute` 填充值集合。
  - 规范 `subject_profile`/`object_profile`，生成质量分与 `final_conf`。
  - 输出 `note_id`、规范化 `subj/obj` 与置信度字段。

## 查询与答案

- `query/query_processor.py` 会校验索引完整性，执行结构化检索并汇总 `evidence`。
- 若提供 LM Studio 配置，则调用其接口生成 `answer`；否则仅返回结构化检索结果。
- 返回结构：`{"structured": {...}, "answer": "..."}`。

## 配置说明

- `config/config_loader.py` 提供默认值，可被 `config.yaml` 覆盖：
  - `vllm.endpoint`, `vllm.model`, `vllm.temperature`, `vllm.max_tokens`
  - `vllm.concurrency.max_workers`：并发工作线程数（默认 `8`）。线程池会持续饱和，建议根据 vLLM 吞吐调整。
  - `vllm.concurrency.batch_size`：批量提交大小（默认 `1`）。仅用于少量初始预热或控制调度粒度，但并不限制并发度（线程池会维持满载）。
  - `vllm.concurrency.endpoints`：可选，多端点列表（覆盖 `endpoint`，用于轮询均衡）
  - `vllm.concurrency.timeout_sec`：HTTP 超时秒数（默认 `60`）
  - `vllm.concurrency.retry_backoff`：重试退避秒数序列（默认 `[1,2,4]`）
  - `lmstudio.endpoint`, `lmstudio.model`
  - `notes.out_path`, `notes.indexes_dir`
  - `chunk.n_sent`, `chunk.overlap`, `chunk.max_tokens`
  - `parsing.*` 与 `schema_guard.*` 用于解析与类型守卫

## 高级用法

- 直接使用适配器与分片构建：

```bash
python main_build_notes.py \
  --dataset mirage \
  --data_dir data/mirage_sample \
  --out notes/notes.mirage.jsonl \
  --indexes_dir indexes/ \
  --vllm_endpoint http://127.0.0.1:8000/v1 \
  --vllm_model qwen2.5-7b-instruct \
  --shard-idx 0 --shard-cnt 1
```

- 双卡脚本（自动拉起 vLLM 并并行分片）：`scripts/mirage/build_notes.sh`。
  - 若启用多端点，可在 `config.yaml` 中添加 `vllm.concurrency.endpoints: [http://127.0.0.1:8001/v1, http://127.0.0.1:8002/v1]`，并相应提高 `max_workers` 以压满吞吐；同时建议设置 `retry_backoff`，在端点短时失败时快速切换与退避。

### 并发调优建议

- 连接池：每个线程拥有复用的 `requests.Session`，连接池大小按 `max_workers` 自动扩展，无需手动设置。
- 饱和策略：我们在 `StructuredBuilder` 与 `main_build_notes` 中采用“完成即补位”的饱和提交策略，确保线程数始终保持在上限附近。
- vLLM 参数配合：留意 `--max-num-batched-tokens`、`--gpu-memory-utilization` 等；当响应超时或队列过长时，适当降低 `max_workers` 或提高上述阈值。
- 容错与回退：启用 `retry_backoff` 可降低短时错误的影响；在多端点场景下，轮询策略会在失败后切换端点。

#### 自适应并发（可选）

- 在 `config.yaml` 打开：

```yaml
vllm:
  adaptive:
    enabled: true
    min_workers: 4
    max_workers: 32
    target_p50_ms: 1200
    target_p95_ms: 3500
    step_up: 2
    step_down: 2
    window_size: 50
    cool_down_sec: 5.0
```

- 机制：生成器在每次 vLLM 调用后记录延迟，构建器每隔 `cool_down_sec` 读取建议并动态调整目标并发。线程池上限固定为 `adaptive.max_workers`，实际 `inflight` 会在建议值附近波动，趋近 GPU 的可承载吞吐。
- 什么时候有用：
  - 端点吞吐随时间波动（队列长度变化、临时降速）。
  - 多端点轮询时，整体延迟特征变化明显。
- 注意：如 vLLM 已通过 `--max-num-batched-tokens`/`--gpu-memory-utilization` 等充分调优，开启自适应并发通常仍能微调队列压力，但也可能引入轻微波动；保守做法是设置较小的 `step_up/step_down` 与适当的 `cool_down_sec`。

## 依赖

- 见 `requirements.txt`。需准备可用的 vLLM 与 LM Studio 服务。

## 开发建议

- 为新领域扩充 `schema/aliases.json` 与 `schema/vocab.json`，提升规范化与别名召回。
- 在 `retriever/operators.py` 中增加领域图算子与重排策略。
- 结合 `meta.final_conf` 与 `quality_score` 设计可回答性策略与拒答。
