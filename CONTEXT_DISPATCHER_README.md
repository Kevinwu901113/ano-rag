# ContextDispatcher - 结构增强的上下文调度器

## 概述

`ContextDispatcher` 是一个新的结构增强上下文调度器，用于替换原有的基于纯嵌入相似度的检索与上下文拼接逻辑。它实现了一个三阶段的检索流程，结合语义召回、图谱扩展和智能上下文调度，提供更精准和结构化的上下文生成。

## 核心特性

### 三阶段检索流程

1. **阶段1：语义召回**
   - 使用嵌入向量余弦相似度召回 top-n 原子笔记
   - 支持多查询合并和去重
   - 可配置相似度阈值

2. **阶段2：图谱扩展召回**
   - 对 top-p (p < n) 个召回项进行 k-hop 图谱扩展
   - 获取邻居节点集合，发现相关但语义距离较远的内容
   - 自动过滤已在语义结果中的重复项

3. **阶段3：上下文调度**
   - 从语义和图谱结果中分别选取前 x 和前 y 个笔记
   - 使用固定模板格式化最终上下文
   - 支持自定义上下文模板

## 配置参数

在 `config.yaml` 中配置 ContextDispatcher：

```yaml
context_dispatcher:
  enabled: true                    # 启用新的结构增强调度器
  
  # 阶段1：语义召回参数
  semantic_top_n: 50              # n: 语义召回的top-n数量
  semantic_threshold: 0.1         # 最小相似度阈值
  
  # 阶段2：图谱扩展参数
  graph_expand_top_p: 20          # p: 用于图谱扩展的top-p数量 (p < n)
  k_hop: 2                        # k: k-hop图谱扩展
  graph_threshold: 0.0            # 最小图谱分数阈值
  
  # 阶段3：上下文调度参数
  final_semantic_count: 8         # x: 最终选择的语义结果数量
  final_graph_count: 5            # y: 最终选择的图谱结果数量
  
  # 上下文模板
  context_template: "Note {note_id}: {content}\n"
```

### 参数调优指南

- **semantic_top_n (n)**: 控制初始召回的广度，建议 30-100
- **graph_expand_top_p (p)**: 控制图谱扩展的种子数量，建议 n/2 到 n/3
- **k_hop (k)**: 控制图谱扩展的深度，建议 1-3
- **final_semantic_count (x)**: 控制最终语义结果数量，建议 5-15
- **final_graph_count (y)**: 控制最终图谱结果数量，建议 3-10
- **总上下文长度**: x + y，建议控制在 10-20 以内

## 使用方法

### 基本使用

```python
from utils.context_dispatcher import ContextDispatcher
from vector_store import VectorRetriever
from graph.graph_retriever import GraphRetriever

# 初始化组件
vector_retriever = VectorRetriever()
graph_retriever = GraphRetriever(graph_index, k_hop=2)

# 创建调度器
dispatcher = ContextDispatcher(vector_retriever, graph_retriever)

# 执行三阶段调度
result = dispatcher.dispatch(query, rewritten_queries)

# 获取结果
context = result['context']
selected_notes = result['selected_notes']
stage_info = result['stage_info']
```

### 与 QueryProcessor 集成

`ContextDispatcher` 已集成到 `QueryProcessor` 中，通过配置自动启用：

```python
from query import QueryProcessor

# 创建查询处理器（自动根据配置选择调度器）
processor = QueryProcessor(atomic_notes, embeddings, graph_file)

# 处理查询
result = processor.process(query)

# 结果包含调度信息
if 'dispatch_info' in result:
    print(f"Semantic: {result['dispatch_info']['semantic_count']}")
    print(f"Graph: {result['dispatch_info']['graph_count']}")
    print(f"Final: {result['dispatch_info']['final_count']}")
```

## 兼容性

### 接口兼容性

- ✅ 保持与现有 `QueryProcessor` 的接口兼容
- ✅ 输出格式与原系统一致
- ✅ 支持新旧调度器切换
- ✅ 保留所有原有功能

### 数据结构兼容性

- ✅ 复用现有嵌入模型和向量索引
- ✅ 复用现有图谱数据结构
- ✅ 保持 note_id 和 paragraph_idxs 结构
- ✅ 兼容现有日志和评估系统

### 配置兼容性

- ✅ 通过 `context_dispatcher.enabled` 控制启用状态
- ✅ 保留原有 `context_scheduler` 配置
- ✅ 支持运行时参数调整

## 输出结构

### dispatch() 方法返回

```python
{
    'context': str,                    # 格式化的最终上下文
    'selected_notes': List[Dict],      # 选中的笔记列表
    'semantic_results': List[Dict],    # 语义召回结果
    'graph_results': List[Dict],       # 图谱扩展结果
    'stage_info': {
        'semantic_count': int,         # 语义召回数量
        'graph_count': int,           # 图谱扩展数量
        'final_count': int            # 最终选择数量
    }
}
```

### 笔记对象结构

每个笔记对象包含以下字段：

```python
{
    'note_id': str,                   # 笔记ID
    'content': str,                   # 笔记内容
    'retrieval_stage': str,           # 'semantic' 或 'graph_expansion'
    'stage_rank': int,                # 在该阶段的排名
    'selection_reason': str,          # 'semantic_top' 或 'graph_expansion'
    'retrieval_info': {               # 检索信息
        'similarity': float           # 相似度分数
    },
    'graph_score': float,             # 图谱分数（图谱结果）
    'paragraph_idxs': List[int],      # 段落索引
    # ... 其他原有字段
}
```

## 日志和调试

### 日志输出示例

```
[INFO] ContextDispatcher initialized with params: n=50, p=20, k=2, x=8, y=5
[INFO] Starting context dispatch for query: 什么是人工智能？
[INFO] Stage 1 - Semantic recall: 45 notes
[INFO] Graph expansion from 20 seed notes
[INFO] Stage 2 - Graph expansion: 12 notes
[INFO] Selected 8 semantic + 5 graph notes
[INFO] Stage 3 - Context scheduling: 13 final notes
```

### 调试信息

- 每个阶段的处理数量
- 去重和过滤统计
- 最终选择的笔记ID列表
- 配置参数摘要

## 性能优化

### 批处理优化

- 向量检索使用批处理
- 图谱扩展并行处理
- 内存高效的去重算法

### 缓存机制

- 复用现有向量索引缓存
- 图谱结构缓存
- 查询结果缓存（可选）

## 故障排除

### 常见问题

1. **ImportError**: 确保 `utils/__init__.py` 包含 `ContextDispatcher` 导入
2. **配置错误**: 检查 `config.yaml` 中的参数设置
3. **空结果**: 检查向量索引和图谱是否正确加载
4. **性能问题**: 调整批处理大小和参数范围

### 调试步骤

1. 运行 `example_context_dispatcher.py` 验证配置
2. 检查日志输出中的阶段统计
3. 验证向量检索和图谱检索的独立功能
4. 逐步调整参数观察效果

## 迁移指南

### 从原系统迁移

1. **备份配置**: 保存现有 `config.yaml`
2. **添加配置**: 添加 `context_dispatcher` 配置段
3. **测试运行**: 设置 `enabled: true` 并测试
4. **参数调优**: 根据效果调整参数
5. **性能对比**: 对比新旧系统的效果

### 回滚方案

如需回滚到原系统：

```yaml
context_dispatcher:
  enabled: false  # 禁用新调度器
```

## 扩展开发

### 自定义模板

```python
# 自定义上下文模板
custom_template = "[{note_id}] {content}\n---\n"
dispatcher.context_template = custom_template
```

### 动态参数调整

```python
# 运行时调整参数
dispatcher.update_config(
    semantic_top_n=60,
    final_semantic_count=10
)
```

### 扩展检索策略

可以继承 `ContextDispatcher` 类实现自定义检索策略：

```python
class CustomContextDispatcher(ContextDispatcher):
    def _semantic_recall(self, queries):
        # 自定义语义召回逻辑
        pass
    
    def _graph_expansion_recall(self, semantic_results):
        # 自定义图谱扩展逻辑
        pass
```

## 总结

`ContextDispatcher` 提供了一个结构化、可配置、高性能的上下文调度解决方案，在保持完全向后兼容的同时，显著提升了检索的精度和上下文的质量。通过三阶段的设计，它能够更好地平衡语义相关性和结构化知识，为下游的答案生成提供更丰富和准确的上下文信息。