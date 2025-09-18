"""
MMR (Maximal Marginal Relevance) selection algorithm
"""

from typing import List, Dict, Any, Callable


def mmr_select(items: List[Dict[str, Any]], k: int, lambda_: float = 0.7, 
               sim: Callable[[Dict[str, Any], Dict[str, Any]], float] = None) -> List[Dict[str, Any]]:
    """
    使用MMR算法选择多样化的项目
    
    Args:
        items: 候选项目列表
        k: 选择的项目数量
        lambda_: 相关性和多样性的权衡参数 (0-1)
        sim: 相似度计算函数
    
    Returns:
        选中的项目列表
    """
    if not items or k <= 0:
        return []
    
    if len(items) <= k:
        return items[:]
    
    # 如果没有提供相似度函数，使用默认的向量相似度
    if sim is None:
        def default_sim(a, b):
            vec_a = a.get("vec", [])
            vec_b = b.get("vec", [])
            if not vec_a or not vec_b or len(vec_a) != len(vec_b):
                return 0.0
            return sum(x * y for x, y in zip(vec_a, vec_b))
        sim = default_sim
    
    # 按分数排序
    sorted_items = sorted(items, key=lambda x: x.get("score", 0), reverse=True)
    
    # 选择第一个项目（分数最高）
    selected = [sorted_items[0]]
    remaining = sorted_items[1:]
    
    # 迭代选择剩余项目
    while len(selected) < k and remaining:
        best_item = None
        best_score = float('-inf')
        
        for item in remaining:
            # 计算相关性分数
            relevance = item.get("score", 0)
            
            # 计算与已选项目的最大相似度
            max_similarity = 0.0
            for selected_item in selected:
                similarity = sim(item, selected_item)
                max_similarity = max(max_similarity, similarity)
            
            # MMR分数 = λ * 相关性 - (1-λ) * 最大相似度
            mmr_score = lambda_ * relevance - (1 - lambda_) * max_similarity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_item = item
        
        if best_item:
            selected.append(best_item)
            remaining.remove(best_item)
        else:
            break
    
    return selected