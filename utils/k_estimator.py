import re
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, deque
from loguru import logger
from utils.text_utils import TextUtils


class KEstimator:
    """基于实体图的K值估计器
    
    在不读取题目id的前提下，估计当前样本所需证据数K ∈ [2,4]
    """
    
    def __init__(self):
        """初始化K值估计器"""
        self.entity_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # 大写短语
            r'\b\d{4}\b',  # 年份
            r'\b\d+(?:\.\d+)?\b',  # 数字
            r'\b[A-Z]{2,}\b',  # 缩写
        ]
    
    def extract_entities_from_text(self, text: str) -> Set[str]:
        """从文本中抽取实体（大写短语、年份、数字等）
        
        Args:
            text: 输入文本
            
        Returns:
            实体集合
        """
        entities = set()
        
        # 使用多种模式提取实体
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text)
            entities.update(matches)
        
        # 使用TextUtils的实体提取功能
        text_entities = TextUtils.extract_entities(text, confidence_threshold=0.3)
        entities.update(text_entities)
        
        # 过滤和清理
        filtered_entities = set()
        for entity in entities:
            entity = entity.strip()
            if len(entity) >= 2 and not entity.lower() in {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}:
                filtered_entities.add(entity)
        
        return filtered_entities
    
    def build_passage_graph(self, passages_by_idx: Dict[int, str]) -> Dict[int, Set[int]]:
        """构建段落图：若两段的实体集合有交集，则连边
        
        Args:
            passages_by_idx: 段落索引到内容的映射
            
        Returns:
            邻接表表示的图
        """
        # 提取每个段落的实体
        passage_entities = {}
        for idx, passage in passages_by_idx.items():
            entities = self.extract_entities_from_text(passage)
            passage_entities[idx] = entities
        
        # 构建图
        graph = defaultdict(set)
        passage_indices = list(passages_by_idx.keys())
        
        for i in range(len(passage_indices)):
            for j in range(i + 1, len(passage_indices)):
                idx1, idx2 = passage_indices[i], passage_indices[j]
                entities1 = passage_entities[idx1]
                entities2 = passage_entities[idx2]
                
                # 如果两段落的实体集合有交集，则连边
                if entities1 & entities2:  # 集合交集
                    graph[idx1].add(idx2)
                    graph[idx2].add(idx1)
        
        return dict(graph)
    
    def find_question_anchor_passage(self, question: str, passages_by_idx: Dict[int, str], 
                                   packed_order: List[int]) -> Optional[int]:
        """选择问题锚点段（与问题实体重合最多且靠前）
        
        Args:
            question: 问题文本
            passages_by_idx: 段落索引到内容的映射
            packed_order: 段落在prompt中的顺序
            
        Returns:
            锚点段落的索引，如果没找到返回None
        """
        question_entities = self.extract_entities_from_text(question)
        
        if not question_entities:
            # 如果问题中没有实体，选择第一个段落作为锚点
            return packed_order[0] if packed_order else None
        
        best_passage_idx = None
        max_overlap = 0
        best_position = float('inf')
        
        for position, passage_idx in enumerate(packed_order):
            if passage_idx not in passages_by_idx:
                continue
                
            passage = passages_by_idx[passage_idx]
            passage_entities = self.extract_entities_from_text(passage)
            
            # 计算实体重合数
            overlap = len(question_entities & passage_entities)
            
            # 选择重合最多且位置靠前的段落
            if overlap > max_overlap or (overlap == max_overlap and position < best_position):
                max_overlap = overlap
                best_position = position
                best_passage_idx = passage_idx
        
        return best_passage_idx
    
    def find_answer_passage(self, answer: str, passages_by_idx: Dict[int, str]) -> Optional[int]:
        """找到答案段（含最终答案子串的段落）
        
        Args:
            answer: 答案文本
            passages_by_idx: 段落索引到内容的映射
            
        Returns:
            答案段落的索引，如果没找到返回None
        """
        if not answer or not answer.strip():
            return None
        
        answer_clean = answer.strip().lower()
        
        # 尝试精确匹配
        for idx, passage in passages_by_idx.items():
            if answer_clean in passage.lower():
                return idx
        
        # 尝试部分匹配（答案的关键词）
        answer_words = set(re.findall(r'\b\w+\b', answer_clean))
        answer_words = {w for w in answer_words if len(w) > 2}  # 过滤短词
        
        if not answer_words:
            return None
        
        best_passage_idx = None
        max_match_ratio = 0
        
        for idx, passage in passages_by_idx.items():
            passage_words = set(re.findall(r'\b\w+\b', passage.lower()))
            matched_words = answer_words & passage_words
            match_ratio = len(matched_words) / len(answer_words) if answer_words else 0
            
            if match_ratio > max_match_ratio and match_ratio > 0.5:  # 至少50%匹配
                max_match_ratio = match_ratio
                best_passage_idx = idx
        
        return best_passage_idx
    
    def find_shortest_path(self, graph: Dict[int, Set[int]], start: int, end: int) -> Optional[List[int]]:
        """使用BFS找到两个节点之间的最短路径
        
        Args:
            graph: 邻接表表示的图
            start: 起始节点
            end: 终止节点
            
        Returns:
            最短路径的节点列表，如果没有路径返回None
        """
        if start == end:
            return [start]
        
        if start not in graph:
            return None
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in graph.get(current, set()):
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def estimate_required_k(self, question: str, answer: str, passages_by_idx: Dict[int, str], 
                          packed_order: List[int], k_min: int = 2, k_max: int = 4) -> int:
        """估计当前样本所需证据数K
        
        Args:
            question: 问题文本
            answer: 答案文本
            passages_by_idx: 段落索引到内容的映射
            packed_order: 段落在prompt中的顺序
            k_min: K的最小值
            k_max: K的最大值
            
        Returns:
            估计的K值
        """
        try:
            # 1. 构建段落图
            graph = self.build_passage_graph(passages_by_idx)
            
            # 2. 找到问题锚点段
            anchor_idx = self.find_question_anchor_passage(question, passages_by_idx, packed_order)
            if anchor_idx is None:
                logger.warning("Could not find question anchor passage, using conservative K=2")
                return k_min
            
            # 3. 找到答案段
            answer_idx = self.find_answer_passage(answer, passages_by_idx)
            if answer_idx is None:
                logger.warning("Could not find answer passage, using conservative K=2")
                return k_min
            
            # 4. 如果锚点段就是答案段，返回最小K
            if anchor_idx == answer_idx:
                return k_min
            
            # 5. 找到最短路径
            shortest_path = self.find_shortest_path(graph, anchor_idx, answer_idx)
            if shortest_path is None:
                logger.warning(f"No path found between anchor {anchor_idx} and answer {answer_idx}, using conservative K=2")
                return k_min
            
            # 6. 计算K = 边数 + 1
            edges = len(shortest_path) - 1
            estimated_k = edges + 1
            
            # 7. 限制在[k_min, k_max]范围内
            estimated_k = max(k_min, min(k_max, estimated_k))
            
            logger.info(f"Estimated K={estimated_k} (path length={len(shortest_path)}, edges={edges})")
            return estimated_k
            
        except Exception as e:
            logger.error(f"Error in K estimation: {e}")
            return k_min  # 失败时保守回退


# 便捷函数
def estimate_required_k(question: str, answer: str, passages_by_idx: Dict[int, str], 
                       packed_order: List[int], k_min: int = 2, k_max: int = 4) -> int:
    """估计当前样本所需证据数K的便捷函数
    
    Args:
        question: 问题文本
        answer: 答案文本  
        passages_by_idx: 段落索引到内容的映射
        packed_order: 段落在prompt中的顺序
        k_min: K的最小值
        k_max: K的最大值
        
    Returns:
        估计的K值
    """
    estimator = KEstimator()
    return estimator.estimate_required_k(question, answer, passages_by_idx, packed_order, k_min, k_max)