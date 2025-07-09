# 多跳问题处理分析与改进方案

## 当前系统问题分析

### 1. 关系提取的局限性

**问题识别：**
- 当前关系提取主要依赖于简单的模式匹配和统计方法
- 缺乏对复杂语义关系的深度理解
- 关系权重设计不够精细，无法有效支持多跳推理

**具体问题：**
1. **引用关系提取过于简单**：只基于`[note_id]`、`@note_id`等模式，无法捕获隐式引用
2. **实体共存关系缺乏语义理解**：仅基于实体重叠，忽略了实体间的语义关系类型
3. **语义相似性阈值过高**：0.7的阈值可能错过重要的弱关联
4. **缺乏因果关系和时序关系**：多跳推理中至关重要的逻辑关系类型缺失

### 2. 图谱检索策略不足

**问题识别：**
- 简单的k-hop搜索无法有效处理复杂的多跳推理路径
- 缺乏路径质量评估和推理链构建
- 图谱中心性计算过于简单

### 3. 查询处理缺乏多跳意识

**问题识别：**
- 查询重写没有针对多跳问题的特殊处理
- 缺乏推理步骤的分解和规划
- 上下文调度没有考虑推理链的完整性

## 改进方案

### 1. 增强关系提取器

#### 1.1 添加新的关系类型

```python
# 新增关系类型
RELATION_TYPES = {
    'causal': '因果关系',
    'temporal': '时序关系', 
    'comparison': '比较关系',
    'definition': '定义关系',
    'part_of': '部分-整体关系',
    'instance_of': '实例关系',
    'contradiction': '矛盾关系',
    'support': '支持关系'
}
```

#### 1.2 基于LLM的关系提取

增加智能关系提取方法，使用LLM识别复杂的语义关系：

```python
def extract_semantic_relations_with_llm(self, note_pairs: List[Tuple[Dict, Dict]]) -> List[Dict]:
    """使用LLM提取语义关系"""
    relations = []
    
    for note1, note2 in note_pairs:
        prompt = f"""
        分析以下两个文本片段之间的关系类型：
        
        文本1：{note1['content']}
        文本2：{note2['content']}
        
        请识别它们之间的关系类型（因果、时序、比较、定义、部分-整体、实例、矛盾、支持等）
        并给出关系强度评分(0-1)。
        """
        
        # 调用LLM分析关系
        relation_info = self.llm.analyze_relation(prompt)
        if relation_info['confidence'] > 0.6:
            relations.append({
                'source_id': note1['note_id'],
                'target_id': note2['note_id'],
                'relation_type': relation_info['type'],
                'weight': relation_info['strength'],
                'metadata': {
                    'reasoning': relation_info['reasoning'],
                    'confidence': relation_info['confidence']
                }
            })
    
    return relations
```

#### 1.3 推理路径权重计算

```python
def calculate_reasoning_path_weight(self, path: List[str]) -> float:
    """计算推理路径的权重"""
    if len(path) < 2:
        return 0.0
    
    total_weight = 1.0
    for i in range(len(path) - 1):
        edge_data = self.graph.get_edge_data(path[i], path[i+1])
        if edge_data:
            relation_type = edge_data.get('relation_type', 'unknown')
            base_weight = edge_data.get('weight', 0.5)
            
            # 根据关系类型调整权重
            type_multiplier = {
                'causal': 1.2,      # 因果关系对推理很重要
                'temporal': 1.1,    # 时序关系有助于推理
                'definition': 1.3,  # 定义关系很重要
                'reference': 1.0,   # 引用关系标准权重
                'semantic_similarity': 0.8  # 语义相似性权重较低
            }.get(relation_type, 0.7)
            
            total_weight *= (base_weight * type_multiplier)
        else:
            total_weight *= 0.1  # 无连接的惩罚
    
    # 路径长度惩罚
    length_penalty = 0.9 ** (len(path) - 2)
    return total_weight * length_penalty
```

### 2. 改进图谱检索器

#### 2.1 多跳推理路径搜索

```python
class MultiHopGraphRetriever(GraphRetriever):
    def __init__(self, graph_index: GraphIndex, max_hops: int = 3):
        super().__init__(graph_index, max_hops)
        self.reasoning_path_finder = ReasoningPathFinder()
    
    def retrieve_with_reasoning_paths(self, seed_note_ids: List[str], 
                                    query_type: str = 'factual') -> Dict[str, Any]:
        """检索时构建推理路径"""
        G = self.index.graph
        
        # 找到所有可能的推理路径
        reasoning_paths = []
        for seed in seed_note_ids:
            paths = self.reasoning_path_finder.find_reasoning_paths(
                G, seed, max_length=self.k_hop, query_type=query_type
            )
            reasoning_paths.extend(paths)
        
        # 评估路径质量
        scored_paths = []
        for path in reasoning_paths:
            score = self.calculate_reasoning_path_weight(path['nodes'])
            path['reasoning_score'] = score
            scored_paths.append(path)
        
        # 选择最佳路径
        scored_paths.sort(key=lambda x: x['reasoning_score'], reverse=True)
        
        # 收集路径中的所有节点
        result_nodes = set()
        selected_paths = scored_paths[:10]  # 选择前10条路径
        
        for path in selected_paths:
            result_nodes.update(path['nodes'])
        
        # 构建结果
        results = []
        for node_id in result_nodes:
            if node_id in G:
                data = G.nodes[node_id].copy()
                data['reasoning_paths'] = [
                    p for p in selected_paths if node_id in p['nodes']
                ]
                results.append(data)
        
        return {
            'notes': results,
            'reasoning_paths': selected_paths,
            'path_count': len(selected_paths)
        }
```

#### 2.2 推理路径发现器

```python
class ReasoningPathFinder:
    def find_reasoning_paths(self, graph: nx.Graph, start_node: str, 
                           max_length: int = 3, query_type: str = 'factual') -> List[Dict]:
        """发现推理路径"""
        paths = []
        
        # 使用DFS搜索路径
        def dfs(current_node, path, visited):
            if len(path) > max_length:
                return
            
            if len(path) > 1:  # 至少包含2个节点的路径
                path_info = {
                    'nodes': path.copy(),
                    'relations': self._extract_path_relations(graph, path),
                    'path_type': self._classify_path_type(graph, path)
                }
                paths.append(path_info)
            
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    edge_data = graph.get_edge_data(current_node, neighbor)
                    if self._is_valid_reasoning_edge(edge_data, query_type):
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, path, visited)
                        path.pop()
                        visited.remove(neighbor)
        
        visited = {start_node}
        dfs(start_node, [start_node], visited)
        
        return paths
    
    def _is_valid_reasoning_edge(self, edge_data: Dict, query_type: str) -> bool:
        """判断边是否适合推理"""
        if not edge_data:
            return False
        
        relation_type = edge_data.get('relation_type', '')
        weight = edge_data.get('weight', 0)
        
        # 根据查询类型过滤关系
        valid_relations = {
            'factual': ['reference', 'entity_coexistence', 'definition', 'instance_of'],
            'causal': ['causal', 'temporal', 'reference'],
            'comparative': ['comparison', 'semantic_similarity', 'entity_coexistence']
        }
        
        return (relation_type in valid_relations.get(query_type, []) and 
                weight > 0.3)
```

### 3. 增强查询处理

#### 3.1 多跳查询识别和分解

```python
def analyze_multi_hop_query(self, query: str) -> Dict[str, Any]:
    """分析多跳查询"""
    prompt = f"""
    分析以下查询是否需要多跳推理：
    
    查询：{query}
    
    请判断：
    1. 是否为多跳问题（需要通过多个信息片段推理得出答案）
    2. 推理步骤分解
    3. 关键实体和概念
    4. 推理类型（因果推理、比较推理、定义推理等）
    
    返回JSON格式：
    {{
        "is_multi_hop": true/false,
        "reasoning_steps": ["步骤1", "步骤2"],
        "key_entities": ["实体1", "实体2"],
        "reasoning_type": "causal/comparative/definitional",
        "sub_queries": ["子查询1", "子查询2"]
    }}
    """
    
    response = self.llm.generate(prompt)
    return self._parse_multi_hop_analysis(response)

def decompose_multi_hop_query(self, query: str) -> List[str]:
    """分解多跳查询为子查询"""
    analysis = self.analyze_multi_hop_query(query)
    
    if not analysis['is_multi_hop']:
        return [query]
    
    # 生成子查询
    sub_queries = []
    for step in analysis['reasoning_steps']:
        sub_query = f"关于{step}的信息"
        sub_queries.append(sub_query)
    
    # 添加原始查询
    sub_queries.append(query)
    
    return sub_queries
```

#### 3.2 改进上下文调度器

```python
class MultiHopContextScheduler(ContextScheduler):
    def schedule_for_multi_hop(self, candidate_notes: List[Dict[str, Any]], 
                              reasoning_paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为多跳推理调度上下文"""
        # 为每个笔记计算推理路径得分
        path_scores = self._calculate_path_scores(candidate_notes, reasoning_paths)
        
        scored_notes = []
        for note in candidate_notes:
            note_id = note.get('note_id')
            
            # 原有得分
            base_score = self._calculate_base_score(note)
            
            # 推理路径得分
            path_score = path_scores.get(note_id, 0)
            
            # 推理完整性得分
            completeness_score = self._calculate_completeness_score(
                note, reasoning_paths
            )
            
            # 综合得分
            total_score = (
                0.3 * base_score +
                0.4 * path_score +
                0.3 * completeness_score
            )
            
            note['multi_hop_score'] = total_score
            scored_notes.append(note)
        
        # 确保推理链的完整性
        selected_notes = self._ensure_reasoning_chain_completeness(
            scored_notes, reasoning_paths
        )
        
        return selected_notes[:self.top_n]
    
    def _ensure_reasoning_chain_completeness(self, notes: List[Dict], 
                                           paths: List[Dict]) -> List[Dict]:
        """确保推理链的完整性"""
        selected = []
        covered_paths = set()
        
        # 按得分排序
        notes.sort(key=lambda x: x.get('multi_hop_score', 0), reverse=True)
        
        for note in notes:
            note_id = note.get('note_id')
            
            # 检查这个笔记是否能完善推理链
            relevant_paths = [
                p for p in paths if note_id in p.get('nodes', [])
            ]
            
            if relevant_paths:
                # 检查是否有新的路径被覆盖
                new_paths = [
                    p for p in relevant_paths 
                    if tuple(p.get('nodes', [])) not in covered_paths
                ]
                
                if new_paths or len(selected) < 3:  # 确保最少3个笔记
                    selected.append(note)
                    for path in relevant_paths:
                        covered_paths.add(tuple(path.get('nodes', [])))
            
            if len(selected) >= self.top_n:
                break
        
        return selected
```

### 4. 实施建议

#### 4.1 优先级排序
1. **高优先级**：增强关系提取器，添加基于LLM的语义关系识别
2. **中优先级**：改进图谱检索器，实现推理路径搜索
3. **低优先级**：优化查询处理和上下文调度

#### 4.2 配置参数调整
```yaml
# 在config.yaml中添加多跳推理配置
multi_hop:
  enabled: true
  max_reasoning_hops: 3
  min_path_confidence: 0.6
  relation_types:
    causal:
      weight: 1.2
      confidence_threshold: 0.7
    temporal:
      weight: 1.1
      confidence_threshold: 0.6
    definition:
      weight: 1.3
      confidence_threshold: 0.8
  
  # LLM关系提取配置
  llm_relation_extraction:
    enabled: true
    batch_size: 16
    max_pairs_per_batch: 50
```

#### 4.3 测试和评估
1. 使用MuSiQue数据集进行多跳推理测试
2. 设计推理路径质量评估指标
3. 对比改进前后的多跳问题回答准确率

这些改进将显著提升系统处理多跳问题的能力，通过更智能的关系提取、路径搜索和上下文调度来支持复杂的推理任务。