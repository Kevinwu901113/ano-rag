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

## 依赖

- 见 `requirements.txt`。需准备可用的 vLLM 与 LM Studio 服务。

## 开发建议

- 为新领域扩充 `schema/aliases.json` 与 `schema/vocab.json`，提升规范化与别名召回。
- 在 `retriever/operators.py` 中增加领域图算子与重排策略。
- 结合 `meta.final_conf` 与 `quality_score` 设计可回答性策略与拒答。
