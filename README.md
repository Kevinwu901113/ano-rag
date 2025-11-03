# Ano-RAG（结构化版）概览

该版本的 Ano-RAG 以**三元组原子笔记**为核心，不再依赖向量索引或混合检索栈。主流程分为两条入口：

- `python main.py process`：读取原始文档 → 分块 → 通过 vLLM 抽取三元组笔记 → 写出 `notes/notes.jsonl` 与倒排索引。
- `python main.py query`：加载倒排索引 → 执行结构化检索（基于关系算子） → 将证据交给 LM Studio 生成最终答案。

## 模块组成

| 模块 | 作用 |
| --- | --- |
| `doc/chunker.py` | 文档分块，输出 `{doc_id, chunk_id, text}` |
| `generator/note_generator.py` | 调用 vLLM，按提示词抽取三元组笔记并用 `validators/note_validator.py` 归一化 |
| `indexer/index_builder.py` | 从 `notes.jsonl` 建倒排索引和有向边缓存 |
| `retriever/` | 解析问题为约束 IR，执行 BIND/EXPAND 算子并评分路径 |
| `generator/answerer.py` | 以证据句为输入向 LM Studio 询问最终答案 |
| `pipeline/structured_builder.py` | 整合分块、抽取、索引的一体化构建器 |
| `main.py` | CLI 入口：`process`（构建）与 `query`（问答） |

所有遗留向量、BM25、混合检索代码已经移除；旧入口（例如 `main_mirage.py`）会直接抛出 `ImportError`。

## 使用示例

```bash
# 1. 预备：确保 vLLM 与 LM Studio 服务已运行

# 2. 构建笔记与索引
python main.py process \
  --data-dir data/sample \
  --vllm-endpoint http://127.0.0.1:8000/v1 \
  --vllm-model qwen2.5-7b-instruct

# 生成结果：
#   notes/notes.jsonl
#   indexes/{entity_to_notes.json, graph_edges.jsonl, ...}

# 3. 查询
python main.py query \
  "Who is the spouse of the Green performer?" \
  --lmstudio-endpoint http://127.0.0.1:1234/v1 \
  --lmstudio-model openai/gpt-oss-20b
```

## 配置

`config.yaml` 和 `config/config_loader.py` 只保留必要键：

- `vllm.endpoint` / `vllm.model`
- `lmstudio.endpoint` / `lmstudio.model`
- `notes.out_path` / `notes.indexes_dir`
- `chunk` 的分块参数（句数、重叠、token 上限）

如果需定制更多流程，可在上述模块扩展；无需再维护旧的向量检索或混合融合配置。

## 目录清理

- `vector_store/`、`retrieval/` 仅保留占位；旧实现位于 Git 历史中。
- `tests/` 与依赖向量检索的脚本已删除，如需新测试请针对结构化链路编写。

## 下一步

1. 根据场景丰富 `retriever/parser.py` 中的模板与算符。
2. 在 `retriever/operators.py` 中加入反向边匹配、领域过滤等增强。
3. 针对常见问答构造新的集成测试，保证索引与查询链路持续稳定。
