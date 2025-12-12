# 离线标准化脚本使用指南

`utils/offline_normalization_script.py` 用于对已有的 `atomic_notes` 进行离线补全与归一，避免每次都重新调用 vLLM 生成笔记。

它会：

- 从 `raw_span` / `raw_span_evidence` / `content` 中基于规则抽取实体与关系信号。
- 调用 `EntityPredicateNormalizer` 将实体/谓词归一到统一词表。
- 写回 `normalized_entities` / `normalized_predicates` 字段（默认只补全缺失值，除非显式覆盖）。
- 输出处理统计，便于审计抽取与归一质量。

## 安装

```bash
pip install -r requirements.txt
```

## 快速使用

```bash
python utils/offline_normalization_script.py input_notes.json output_notes.json
```

## 参数

- `input_file`：输入 atomic_notes 的 JSON 文件路径（必须是数组）。
- `output_file`：输出 JSON 路径。
- `--config <path>`：可选 JSON 配置文件（见下节）。
- `--force-update`：强制重算并覆盖已有 `normalized_*` 字段。
- `--log-level {DEBUG,INFO,WARNING,ERROR}`：日志级别（默认 INFO）。

示例：

```bash
python utils/offline_normalization_script.py \
  notes/notes.json \
  notes/notes.normalized.json \
  --config config/offline_normalization.json \
  --log-level INFO
```

## 输入格式

输入文件为 note 数组，每条 note 至少包含文本字段之一：

```json
[
  {
    "content": "John Smith founded Acme Corp in 2001.",
    "raw_span": "John Smith founded Acme Corp",
    "raw_span_evidence": "…",
    "entities": ["John Smith"],
    "predicates": ["founded"],
    "normalized_entities": [],
    "normalized_predicates": []
  }
]
```

脚本会合并三段文本进行抽取；若 `entities`/`predicates` 已存在，将作为候选一起归一。  
当前实现不会主动回写新的 `entities`/`predicates` 列表，只写 `normalized_*`。

## 输出格式

输出仍为 note 数组，新增或更新字段：

```json
[
  {
    "...": "...",
    "normalized_entities": ["John Smith", "Acme Corporation"],
    "normalized_predicates": ["founded"]
  }
]
```

## 配置文件（JSON）

`--config` 接受 JSON 文件；不提供时使用脚本默认配置。  
目前仅读取 `normalizer` 相关配置，结构与 `create_entity_predicate_normalizer(...)` 一致：

```json
{
  "normalizer": {
    "entity_normalizer": {
      "fuzzy_threshold": 0.8,
      "enable_fuzzy_matching": true,
      "case_sensitive": false
    },
    "predicate_normalizer": {
      "fuzzy_threshold": 0.8,
      "enable_fuzzy_matching": true,
      "case_sensitive": false
    }
  }
}
```

## 统计与调试

脚本结束后会打印类似统计：

```
Normalization Statistics:
  Processed notes: 1000
  Entities extracted: 2500
  Predicates extracted: 800
  Entities normalized: 2300
  Predicates normalized: 750
  Errors: 5
```

若需要定位具体抽取/归一问题，建议用 `--log-level DEBUG` 运行并抽样检查输出。

## 扩展开发

规则抽取是启发式的，如需适配新领域，可按需扩展：

- **实体模式**：在 `OfflineNormalizer.entity_patterns` 里追加正则，例如：

```python
self.entity_patterns.append(r"\\b[A-Z][a-z]+\\s+[0-9]+\\b")
```

- **谓词模式**：在 `OfflineNormalizer.predicate_patterns` 里追加 `(pattern, predicate)` 对：

```python
self.predicate_patterns.append(
    (r"(\\w+)\\s+acquired\\s+(\\w+)", "acquired")
)
```

## 注意事项

- `--force-update` 会覆盖已有 `normalized_*` 字段；如只想补全缺失值，请不要开启该开关。
- 抽取与归一的阈值（如 0.5 置信度）可在脚本内调整，以匹配不同数据分布。
