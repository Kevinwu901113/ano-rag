#!/usr/bin/env python3
"""
示例脚本：演示新的ContextDispatcher的使用
展示三阶段结构增强的上下文调度流程
"""

import os
import sys
from typing import List, Dict, Any
from loguru import logger

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from utils.context_dispatcher import ContextDispatcher
from vector_store import VectorRetriever
from graph.graph_index import GraphIndex
from graph.graph_retriever import GraphRetriever
from query import QueryProcessor

def demo_context_dispatcher():
    """演示ContextDispatcher的基本使用"""
    logger.info("=== ContextDispatcher Demo ===")
    
    # 模拟一些原子笔记数据
    sample_notes = [
        {
            'note_id': 'note_001',
            'content': '人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。',
            'paragraph_idxs': [1, 2]
        },
        {
            'note_id': 'note_002', 
            'content': '机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习模式。',
            'paragraph_idxs': [3, 4]
        },
        {
            'note_id': 'note_003',
            'content': '深度学习使用神经网络来模拟人脑的学习过程，在图像识别等领域取得突破。',
            'paragraph_idxs': [5, 6]
        }
    ]
    
    try:
        # 初始化组件（这里使用模拟数据）
        logger.info("Initializing components...")
        
        # 注意：在实际使用中，这些组件需要正确初始化
        # 这里仅作为演示，实际使用时需要加载真实的向量索引和图谱
        
        # 显示配置信息
        dispatcher_config = config.get('context_dispatcher', {})
        logger.info(f"ContextDispatcher configuration: {dispatcher_config}")
        
        # 显示参数说明
        logger.info("\n=== Three-Stage Process Parameters ===")
        logger.info(f"Stage 1 - Semantic Recall: top_n = {dispatcher_config.get('semantic_top_n', 50)}")
        logger.info(f"Stage 2 - Graph Expansion: top_p = {dispatcher_config.get('graph_expand_top_p', 20)}, k_hop = {dispatcher_config.get('k_hop', 2)}")
        logger.info(f"Stage 3 - Context Scheduling: semantic_count = {dispatcher_config.get('final_semantic_count', 8)}, graph_count = {dispatcher_config.get('final_graph_count', 5)}")
        
        logger.info("\n=== Context Template ===")
        template = dispatcher_config.get('context_template', 'Note {note_id}: {content}\n')
        logger.info(f"Template: {repr(template)}")
        
        # 演示模板格式化
        logger.info("\n=== Template Formatting Example ===")
        for note in sample_notes[:2]:
            formatted = template.format(
                note_id=note['note_id'],
                content=note['content']
            )
            logger.info(f"Formatted: {repr(formatted)}")
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return False
        
    return True

def demo_query_processor_integration():
    """演示QueryProcessor与ContextDispatcher的集成"""
    logger.info("\n=== QueryProcessor Integration Demo ===")
    
    try:
        # 检查配置
        use_dispatcher = config.get('context_dispatcher.enabled', True)
        logger.info(f"ContextDispatcher enabled: {use_dispatcher}")
        
        if use_dispatcher:
            logger.info("✓ New structure-enhanced context dispatching is enabled")
            logger.info("  - Stage 1: Semantic recall using embedding similarity")
            logger.info("  - Stage 2: Graph expansion using k-hop traversal")
            logger.info("  - Stage 3: Context scheduling with template formatting")
        else:
            logger.info("✓ Legacy context scheduling is enabled")
            logger.info("  - Traditional vector + graph retrieval")
            logger.info("  - Original context scheduler")
            
        # 显示兼容性信息
        logger.info("\n=== Compatibility Information ===")
        logger.info("✓ Maintains interface compatibility with existing QueryProcessor")
        logger.info("✓ Preserves output format for downstream components")
        logger.info("✓ Supports both new and legacy scheduling modes")
        logger.info("✓ All parameters configurable via config.yaml")
        
    except Exception as e:
        logger.error(f"Integration demo failed: {e}")
        return False
        
    return True

def show_configuration_guide():
    """显示配置指南"""
    logger.info("\n=== Configuration Guide ===")
    
    config_example = """
# 在 config.yaml 中配置 ContextDispatcher:
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
"""
    
    logger.info(config_example)
    
    logger.info("\n=== Parameter Tuning Guidelines ===")
    logger.info("• semantic_top_n (n): 控制初始召回的广度，建议 30-100")
    logger.info("• graph_expand_top_p (p): 控制图谱扩展的种子数量，建议 n/2 到 n/3")
    logger.info("• k_hop (k): 控制图谱扩展的深度，建议 1-3")
    logger.info("• final_semantic_count (x): 控制最终语义结果数量，建议 5-15")
    logger.info("• final_graph_count (y): 控制最终图谱结果数量，建议 3-10")
    logger.info("• 总上下文长度 = x + y，建议控制在 10-20 以内")

def main():
    """主函数"""
    logger.info("Starting ContextDispatcher demonstration...")
    
    # 运行演示
    success = True
    success &= demo_context_dispatcher()
    success &= demo_query_processor_integration()
    
    # 显示配置指南
    show_configuration_guide()
    
    if success:
        logger.info("\n✓ All demonstrations completed successfully!")
        logger.info("\n=== Next Steps ===")
        logger.info("1. 确保向量索引和图谱已正确构建")
        logger.info("2. 根据需要调整 config.yaml 中的参数")
        logger.info("3. 运行 main.py 或 main_musique.py 测试完整流程")
        logger.info("4. 监控日志输出，观察三阶段调度的效果")
    else:
        logger.error("Some demonstrations failed. Please check the logs.")
        
    return success

if __name__ == "__main__":
    main()