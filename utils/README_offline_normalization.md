# 离线标准化脚本使用指南

## 概述

离线标准化脚本用于批量处理atomic_notes数据，基于`raw_span`和`raw_span_evidence`字段使用正则表达式提取和标准化实体与谓词，填充`normalized_entities`和`normalized_predicates`字段。

## 功能特性

- **实体提取**: 从文本中提取人名、组织、地点、产品等实体
- **谓词提取**: 识别实体间的关系，如founded、located_in、works_for等
- **标准化处理**: 使用EntityPredicateNormalizer进行实体和谓词的标准化
- **批量处理**: 支持大规模数据的批量处理
- **增量更新**: 支持只更新缺失字段或强制更新所有字段
- **统计报告**: 提供详细的处理统计信息

## 安装依赖

```bash
# 确保已安装项目依赖
pip install -r requirements.txt

# 可选：安装fuzzywuzzy用于模糊匹配
pip install fuzzywuzzy python-levenshtein
```

## 使用方法

### 基本用法

```bash
# 处理atomic_notes文件
python utils/offline_normalization_script.py input_notes.json output_notes.json
```

### 高级用法

```bash
# 使用自定义配置文件
python utils/offline_normalization_script.py input_notes.json output_notes.json --config config/offline_normalization_config.yaml

# 强制更新已有的标准化字段
python utils/offline_normalization_script.py input_notes.json output_notes.json --force-update

# 设置日志级别
python utils/offline_normalization_script.py input_notes.json output_notes.json --log-level DEBUG
```

### 参数说明

- `input_file`: 输入的atomic_notes JSON文件路径
- `output_file`: 输出的标准化JSON文件路径
- `--config`: 配置文件路径（可选）
- `--force-update`: 强制更新已有的标准化字段（可选）
- `--log-level`: 日志级别，可选值：DEBUG, INFO, WARNING, ERROR（默认：INFO）

## 输入数据格式

输入文件应为包含atomic_notes列表的JSON文件，每个note应包含以下字段：

```json
[
  {
    "content": "笔记内容",
    "raw_span": "原始文本片段",
    "raw_span_evidence": "证据文本",
    "entities": ["现有实体1", "现有实体2"],
    "predicates": ["现有谓词1", "现有谓词2"],
    "normalized_entities": [],  // 将被填充或更新
    "normalized_predicates": []  // 将被填充或更新
  }
]
```

## 输出数据格式

输出文件将包含标准化后的atomic_notes，新增或更新的字段：

```json
[
  {
    "content": "笔记内容",
    "raw_span": "原始文本片段",
    "raw_span_evidence": "证据文本",
    "entities": ["现有实体1", "现有实体2", "提取的实体3"],
    "predicates": ["现有谓词1", "现有谓词2"],
    "normalized_entities": ["标准化实体1", "标准化实体2", "标准化实体3"],
    "normalized_predicates": ["标准化谓词1", "标准化谓词2"]
  }
]
```

## 配置文件

配置文件使用YAML格式，主要配置项包括：

### 标准化器配置

```yaml
normalizer:
  entity_normalizer:
    fuzzy_threshold: 0.8          # 模糊匹配阈值
    enable_fuzzy_matching: true   # 启用模糊匹配
    case_sensitive: false         # 是否区分大小写
  
  predicate_normalizer:
    fuzzy_threshold: 0.8
    enable_fuzzy_matching: true
    case_sensitive: false
```

### 提取配置

```yaml
entity_extraction:
  enable_ner: true               # 启用命名实体识别
  enable_regex: true             # 启用正则表达式提取
  min_confidence: 0.5            # 最小置信度阈值

predicate_extraction:
  enable_relation_extraction: true
  enable_regex: true
  min_confidence: 0.5
```

## 实体提取模式

脚本使用以下正则表达式模式提取实体：

### 人名模式
- `John Smith` (名 姓)
- `John A. Smith` (名 中间名缩写 姓)
- `John Michael Smith` (名 中间名 姓)

### 组织机构模式
- 包含Inc, Corp, Ltd, LLC, Company, Corporation等后缀
- 包含Foundation, Association, Society, Group等后缀
- University, College, Institute等教育机构

### 地点模式
- 包含City, State, Country, Province等后缀
- 知名城市和国家名称

### 产品/品牌模式
- 包含™或®商标符号
- 知名科技品牌和产品

## 谓词提取模式

脚本识别以下类型的关系：

### 创建关系
- `founded`, `established`, `created`
- `co-founded`, `cofounded`

### 位置关系
- `located in`, `based in`, `situated in`

### 工作关系
- `works for`, `employed by`, `works at`

### 成员关系
- `member of`, `belongs to`

### 部分关系
- `part of`, `component of`

### 实例关系
- `instance of`, `type of`, `kind of`

## 统计信息

处理完成后，脚本会输出详细的统计信息：

```
Normalization Statistics:
  Processed notes: 1000
  Entities extracted: 2500
  Predicates extracted: 800
  Entities normalized: 2300
  Predicates normalized: 750
  Errors: 5
```

## 错误处理

- 脚本会记录处理过程中的错误，但不会中断整个处理流程
- 错误的笔记会保持原样输出
- 所有错误都会记录在日志中

## 性能优化建议

1. **批量处理**: 对于大型数据集，建议分批处理
2. **内存管理**: 监控内存使用，必要时调整批处理大小
3. **并行处理**: 未来版本将支持多进程并行处理
4. **增量更新**: 使用增量模式避免重复处理已标准化的数据

## 故障排除

### 常见问题

1. **ImportError**: 确保所有依赖已正确安装
2. **FileNotFoundError**: 检查输入文件路径是否正确
3. **JSONDecodeError**: 确保输入文件为有效的JSON格式
4. **MemoryError**: 减少批处理大小或增加系统内存

### 调试技巧

1. 使用`--log-level DEBUG`获取详细日志
2. 检查配置文件格式是否正确
3. 验证输入数据格式是否符合要求
4. 查看错误统计确定问题范围

## 示例

### 处理示例数据

```bash
# 创建示例输入文件
echo '[
  {
    "content": "John Smith founded Apple Inc in California.",
    "raw_span": "John Smith founded Apple Inc",
    "raw_span_evidence": "John Smith founded Apple Inc",
    "entities": ["John Smith", "Apple Inc"],
    "predicates": [],
    "normalized_entities": [],
    "normalized_predicates": []
  }
]' > example_input.json

# 运行标准化脚本
python utils/offline_normalization_script.py example_input.json example_output.json

# 查看结果
cat example_output.json
```

预期输出：
```json
[
  {
    "content": "John Smith founded Apple Inc in California.",
    "raw_span": "John Smith founded Apple Inc",
    "raw_span_evidence": "John Smith founded Apple Inc",
    "entities": ["John Smith", "Apple Inc", "California"],
    "predicates": [],
    "normalized_entities": ["John Smith", "Apple Inc", "California"],
    "normalized_predicates": ["founded", "located_in"]
  }
]
```

## 扩展开发

### 添加新的实体模式

在`OfflineNormalizer`类的`entity_patterns`列表中添加新的正则表达式：

```python
self.entity_patterns.append(r'\b[A-Z][a-z]+\s+[0-9]+\b')  # 产品型号模式
```

### 添加新的谓词模式

在`predicate_patterns`列表中添加新的模式：

```python
self.predicate_patterns.append(
    (r'(\w+)\s+acquired\s+(\w+)', 'acquired')
)
```

### 自定义标准化规则

通过配置文件或直接修改EntityPredicateNormalizer来添加自定义标准化规则。

## 版本历史

- v1.0.0: 初始版本，支持基本的实体和谓词提取与标准化

## 许可证

本脚本遵循项目的整体许可证。