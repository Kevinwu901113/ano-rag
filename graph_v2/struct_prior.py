"""
Structural prior for graph building
"""

from typing import Dict, Any, List


def struct_score(center_id: int, target_id: int, weak_graph: Dict[str, Any]) -> float:
    """
    计算结构化先验分数
    """
    # 简化实现：返回基于ID的伪随机分数
    return 0.1 + 0.2 * (hash(f"{center_id}_{target_id}") % 100) / 100


def build_weak_graph(paragraphs: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    构建弱图结构
    """
    # 简化实现：返回空图结构
    return {
        "nodes": list(paragraphs.keys()),
        "edges": []
    }