# EFSA (Entity-Focused Score Aggregation) 实体聚合答案生成

## 概述

EFSA是一种新的答案生成方法，通过对最终入围的原子笔记中的实体进行聚合打分，选择证据分最高的实体作为短答案。该方法不依赖关系抽取或谓词匹配，而是基于实体在候选笔记中的出现频率、重要性和一致性来进行答案选择。

## 核心思想

在最终入围的原子笔记小集合里，不去猜"关系/谓词"，而是把出现的实体当作候选答案来聚合打分；排除桥接实体，选证据分最高的另一个实体作为短答案。

## 实现架构

### 文件结构
```
answer/
├── efsa_answer.py          # EFSA核心算法实现
├── __init__.py
query/
├── query_processor.py      # 集成EFSA到查询处理流程
test_efsa.py               # EFSA单元测试
test_integration.py        # EFSA集成测试
```

### 核心函数

#### 1. `compute_cov_cons(note, path_entities)`
计算笔记的覆盖率和一致性：
- **覆盖率 (Coverage)**: `path_entities ∩ note.entities` 占比
- **一致性 (Consistency)**: 候选文本是否提到路径实体名 (0/1)

#### 2. `efsa_answer(candidates, query, bridge_entity=None, path_entities=None, topN=20)`
EFSA核心算法：
- 对每个实体累计证据分数
- 应用路径衰减和实体优势
- 添加文档多样性奖励
- 排除桥接实体
- 选择最高分实体作为答案

#### 3. `efsa_answer_with_fallback(...)`
带回退机制的EFSA：
- 首先尝试EFSA生成实体答案
- 如果失败则返回None，由调用方处理回退逻辑

## 打分机制

### 基础权重计算
```python
w_note = note.final_score * (hop_decay**(hop_no-1)) * (1 + β*cov + γ*cons)
```

其中：
- `note.final_score`: 重排后的综合分
- `hop_decay`: 路径衰减因子 (默认0.85)
- `hop_no`: 跳数信息
- `β`: 覆盖率权重 (默认0.10)
- `γ`: 一致性权重 (默认0.05)

### 文档多样性奖励
```python
score[e] *= (1 + ε * min(max(#docs-1, 0), 3))
```

其中：
- `ε`: 多样性奖励因子 (默认0.03)
- `#docs`: 实体出现的文档数量

## 集成位置

EFSA集成在**"重排+过滤完成 → 答案生成"**之间：

1. 使用现有的检索、多跳、重排、过滤产生 `final_candidates`
2. 调用 `efsa_answer_with_fallback(final_candidates, query, bridge_entity, path_entities)`
3. 若有返回实体答案：直接作为 `predicted_answer`，并设置 `support_idxs`
4. 若没有：回退到现有的句子型答案逻辑

## 使用示例

### 基本使用
```python
from answer.efsa_answer import efsa_answer_with_fallback

# 候选笔记数据
candidates = [
    {
        'entities': ['Steve Hillage', 'Miquette Giraudy'],
        'final_score': 0.85,
        'hop_no': 1,
        'doc_id': 'doc1',
        'paragraph_idxs': [1],
        'content': '...'
    },
    # ... 更多候选
]

# 生成答案
answer, support_idxs = efsa_answer_with_fallback(
    candidates=candidates,
    query="Who is Steve Hillage's partner?",
    bridge_entity="Steve Hillage",
    path_entities=["Steve Hillage", "System 7"],
    topN=20
)

if answer:
    print(f"EFSA Answer: {answer}")
    print(f"Support: {support_idxs}")
else:
    print("EFSA failed, use fallback method")
```

### 在QueryProcessor中的集成
```python
# 在query_processor.py的process方法中
from answer.efsa_answer import efsa_answer_with_fallback

# 提取桥接实体和路径实体
bridge_entities = [n.get('bridge_entity') for n in selected_notes if n.get('bridge_entity')]
path_entities = [e for n in selected_notes for e in n.get('bridge_path', [])]

# 尝试EFSA
efsa_answer, efsa_support_idxs = efsa_answer_with_fallback(
    candidates=selected_notes,
    query=query,
    bridge_entity=bridge_entities[0] if bridge_entities else None,
    path_entities=list(set(path_entities))[-2:],
    topN=20
)

if efsa_answer:
    # 使用EFSA结果
    answer = efsa_answer
    predicted_support_idxs = efsa_support_idxs
else:
    # 回退到LLM生成
    # ... 原有逻辑
```

## 测试

### 运行单元测试
```bash
python test_efsa.py
```

### 运行集成测试
```bash
python test_integration.py
```

## 配置参数

可以通过修改 `efsa_answer.py` 中的常量来调整算法参数：

```python
# 路径衰减因子
HOP_DECAY = 0.85

# 覆盖率权重
COV_WEIGHT = 0.10

# 一致性权重
CONS_WEIGHT = 0.05

# 多样性奖励因子
DIVERSITY_REWARD = 0.03
```

## 优势特点

1. **无关系依赖**: 不需要关系抽取或谓词匹配
2. **实体聚合**: 基于实体在多个笔记中的证据进行聚合打分
3. **桥接过滤**: 自动排除桥接实体，避免输出中间节点
4. **多样性奖励**: 考虑实体在不同文档中的出现情况
5. **回退机制**: 与现有LLM答案生成无缝集成
6. **高效计算**: 复用现有的覆盖率和一致性计算逻辑

## 适用场景

EFSA特别适合以下类型的问题：
- 实体关系查询 (如 "Who is X's partner?")
- 实体属性查询 (如 "What is X's profession?")
- 实体关联查询 (如 "Who collaborated with X?")

对于需要复杂推理或句子级答案的问题，会自动回退到LLM生成。