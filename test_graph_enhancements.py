#!/usr/bin/env python3
"""
测试GraphBuilder增强功能的集成效果
包括：时间衰减权重、反hub惩罚、metapath白名单和路径偏好
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import networkx as nx
from datetime import datetime, timedelta
from graph.graph_builder import GraphBuilder
from config import config
from loguru import logger

def create_test_atomic_notes():
    """创建测试用的原子笔记数据"""
    base_time = datetime.now()
    
    test_notes = [
        {
            "note_id": "note_1",
            "content": "Apple Inc. was founded by Steve Jobs in 1976.",
            "summary": "Apple founding",
            "entities": ["Apple Inc.", "Steve Jobs"],
            "normalized_entities": ["Apple Inc.", "Steve Jobs"],
            "predicate": "founded_by",
            "timestamp": (base_time - timedelta(days=30)).isoformat(),
            "importance_score": 0.9,
            "sentence_index": 0
        },
        {
            "note_id": "note_2",
            "content": "Steve Jobs was the CEO of Apple Inc.",
            "summary": "Jobs as CEO",
            "entities": ["Steve Jobs", "Apple Inc."],
            "normalized_entities": ["Steve Jobs", "Apple Inc."],
            "predicate": "is_ceo_of",
            "timestamp": (base_time - timedelta(days=10)).isoformat(),
            "importance_score": 0.8,
            "sentence_index": 1
        },
        {
            "note_id": "note_3",
            "content": "Apple Inc. develops innovative technology products.",
            "summary": "Apple products",
            "entities": ["Apple Inc.", "technology products"],
            "normalized_entities": ["Apple Inc.", "technology products"],
            "predicate": "develops",
            "timestamp": (base_time - timedelta(days=5)).isoformat(),
            "importance_score": 0.7,
            "sentence_index": 2
        },
        {
            "note_id": "note_4",
            "content": "Microsoft Corporation is a competitor of Apple Inc.",
            "summary": "Microsoft vs Apple",
            "entities": ["Microsoft Corporation", "Apple Inc."],
            "normalized_entities": ["Microsoft Corporation", "Apple Inc."],
            "predicate": "competes_with",
            "timestamp": (base_time - timedelta(days=1)).isoformat(),
            "importance_score": 0.6,
            "sentence_index": 3
        }
    ]
    
    return test_notes

def test_graph_builder_enhancements():
    """测试GraphBuilder的增强功能"""
    logger.info("开始测试GraphBuilder增强功能")
    
    # 创建测试数据
    test_notes = create_test_atomic_notes()
    
    # 创建GraphBuilder实例
    builder = GraphBuilder()
    
    # 打印配置信息
    logger.info(f"时间衰减配置: {builder.time_decay_config}")
    logger.info(f"反hub惩罚配置: {builder.hub_penalty_config}")
    logger.info(f"metapath配置: {builder.metapath_config}")
    
    # 构建图
    graph = builder.build_graph(test_notes)
    
    # 分析图结构
    logger.info(f"图节点数: {graph.number_of_nodes()}")
    logger.info(f"图边数: {graph.number_of_edges()}")
    
    # 检查时间衰减权重
    logger.info("\n=== 时间衰减权重分析 ===")
    time_decay_edges = 0
    for u, v, data in graph.edges(data=True):
        if data.get('time_decay_applied', False):
            time_decay_edges += 1
            logger.info(f"边 {u}->{v}: 原权重={data.get('original_weight', 'N/A')}, "
                       f"衰减后权重={data.get('weight', 'N/A')}, "
                       f"衰减因子={data.get('time_decay_factor', 'N/A')}")
    
    logger.info(f"应用时间衰减的边数: {time_decay_edges}")
    
    # 检查反hub惩罚
    logger.info("\n=== 反hub惩罚分析 ===")
    hub_penalty_edges = 0
    hub_nodes = 0
    for node, data in graph.nodes(data=True):
        if data.get('enhanced_hub_penalty', False):
            hub_nodes += 1
            logger.info(f"Hub节点 {node}: 惩罚因子={data.get('penalty_factor', 'N/A')}")
    
    for u, v, data in graph.edges(data=True):
        if data.get('enhanced_hub_penalty_applied', False):
            hub_penalty_edges += 1
    
    logger.info(f"Hub节点数: {hub_nodes}")
    logger.info(f"应用反hub惩罚的边数: {hub_penalty_edges}")
    
    # 检查metapath过滤
    logger.info("\n=== Metapath过滤分析 ===")
    metapath_edges = 0
    for u, v, data in graph.edges(data=True):
        if data.get('metapath_preference_applied', False):
            metapath_edges += 1
            logger.info(f"边 {u}->{v}: 关系类型={data.get('relation_type', 'N/A')}, "
                       f"偏好权重={data.get('preference_weight', 'N/A')}")
    
    logger.info(f"应用metapath偏好的边数: {metapath_edges}")
    
    # 输出图的基本统计信息
    logger.info("\n=== 图统计信息 ===")
    if graph.number_of_nodes() > 0:
        degrees = dict(graph.degree())
        avg_degree = sum(degrees.values()) / len(degrees)
        max_degree = max(degrees.values()) if degrees else 0
        min_degree = min(degrees.values()) if degrees else 0
        
        logger.info(f"平均度数: {avg_degree:.2f}")
        logger.info(f"最大度数: {max_degree}")
        logger.info(f"最小度数: {min_degree}")
        
        # 输出度数分布
        degree_dist = {}
        for degree in degrees.values():
            degree_dist[degree] = degree_dist.get(degree, 0) + 1
        logger.info(f"度数分布: {degree_dist}")
    
    # 检查边的权重分布
    logger.info("\n=== 边权重分布 ===")
    weights = [data.get('weight', 0) for u, v, data in graph.edges(data=True)]
    if weights:
        avg_weight = sum(weights) / len(weights)
        max_weight = max(weights)
        min_weight = min(weights)
        
        logger.info(f"平均权重: {avg_weight:.4f}")
        logger.info(f"最大权重: {max_weight:.4f}")
        logger.info(f"最小权重: {min_weight:.4f}")
    
    logger.info("GraphBuilder增强功能测试完成")
    return graph

def main():
    """主函数"""
    try:
        graph = test_graph_builder_enhancements()
        logger.info("测试成功完成")
        return True
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)