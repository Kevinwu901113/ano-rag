#!/usr/bin/env python3
"""
混合检索系统使用示例

本示例展示如何使用 BM25 + 向量相似度的混合检索功能
"""

import os
import sys
from typing import List, Dict, Any
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from query.query_processor import QueryProcessor
from config import config

def create_sample_notes() -> List[Dict[str, Any]]:
    """创建示例笔记数据"""
    return [
        {
            'note_id': 'ai_001',
            'content': 'Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.',
            'title': 'What is AI?',
            'source': 'AI Encyclopedia'
        },
        {
            'note_id': 'ml_001',
            'content': 'Machine learning is a method of data analysis that automates analytical model building using algorithms that iteratively learn from data.',
            'title': 'Machine Learning Basics',
            'source': 'ML Handbook'
        },
        {
            'note_id': 'dl_001',
            'content': 'Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.',
            'title': 'Deep Learning Introduction',
            'source': 'Deep Learning Guide'
        },
        {
            'note_id': 'nlp_001',
            'content': 'Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with interactions between computers and human language.',
            'title': 'NLP Overview',
            'source': 'NLP Textbook'
        },
        {
            'note_id': 'cv_001',
            'content': 'Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.',
            'title': 'Computer Vision Fundamentals',
            'source': 'CV Research'
        }
    ]

def demonstrate_hybrid_search():
    """演示混合检索功能"""
    logger.info("=== 混合检索系统演示 ===")
    
    # 1. 准备数据
    notes = create_sample_notes()
    logger.info(f"准备了 {len(notes)} 个示例笔记")
    
    # 2. 配置混合检索
    config.update_config({
        'hybrid_search': {
            'enabled': True,
            'bm25_weight': 0.6,
            'vector_weight': 1.0
        }
    })
    
    # 3. 初始化查询处理器
    logger.info("初始化查询处理器...")
    processor = QueryProcessor(notes)
    
    # 4. 测试查询
    test_queries = [
        "machine learning algorithms",
        "artificial intelligence applications", 
        "neural networks deep learning",
        "computer vision image processing",
        "natural language understanding"
    ]
    
    for query in test_queries:
        logger.info(f"\n查询: '{query}'")
        
        # 使用混合检索
        results = processor._hybrid_search(query, notes, top_k=3)
        
        logger.info("混合检索结果:")
        for i, result in enumerate(results, 1):
            hybrid_score = result.get('hybrid_score', 0)
            vector_score = result.get('vector_score', 0)
            bm25_score = result.get('bm25_score', 0)
            
            logger.info(f"  {i}. {result['title']}")
            logger.info(f"     混合分数: {hybrid_score:.4f} (向量: {vector_score:.4f}, BM25: {bm25_score:.4f})")
            logger.info(f"     内容: {result['content'][:80]}...")

def compare_search_methods():
    """比较不同检索方法的效果"""
    logger.info("\n=== 检索方法对比 ===")
    
    notes = create_sample_notes()
    
    # 测试查询
    query = "machine learning neural networks"
    logger.info(f"测试查询: '{query}'")
    
    # 1. 纯向量检索
    config.update_config({'hybrid_search': {'enabled': False}})
    processor_vector = QueryProcessor(notes)
    vector_results = processor_vector._fallback_vector_search(query, notes, top_k=3)
    
    logger.info("\n纯向量检索结果:")
    for i, result in enumerate(vector_results, 1):
        score = result.get('vector_score', 0)
        logger.info(f"  {i}. {result['title']} (分数: {score:.4f})")
    
    # 2. 混合检索
    config.update_config({
        'hybrid_search': {
            'enabled': True,
            'bm25_weight': 0.6,
            'vector_weight': 1.0
        }
    })
    processor_hybrid = QueryProcessor(notes)
    hybrid_results = processor_hybrid._hybrid_search(query, notes, top_k=3)
    
    logger.info("\n混合检索结果:")
    for i, result in enumerate(hybrid_results, 1):
        hybrid_score = result.get('hybrid_score', 0)
        vector_score = result.get('vector_score', 0)
        bm25_score = result.get('bm25_score', 0)
        logger.info(f"  {i}. {result['title']} (混合: {hybrid_score:.4f}, 向量: {vector_score:.4f}, BM25: {bm25_score:.4f})")

def performance_benchmark():
    """性能基准测试"""
    logger.info("\n=== 性能基准测试 ===")
    
    import time
    
    # 创建较大的数据集
    base_notes = create_sample_notes()
    large_notes = []
    
    for i in range(200):  # 1000 个笔记
        for j, note in enumerate(base_notes):
            new_note = note.copy()
            new_note['note_id'] = f"note_{i}_{j}"
            new_note['content'] = f"[Variant {i}] {note['content']}"
            large_notes.append(new_note)
    
    logger.info(f"使用 {len(large_notes)} 个笔记进行性能测试")
    
    # 配置混合检索
    config.update_config({
        'hybrid_search': {
            'enabled': True,
            'bm25_weight': 0.6,
            'vector_weight': 1.0
        }
    })
    
    # 初始化处理器（包含索引构建时间）
    start_time = time.time()
    processor = QueryProcessor(large_notes)
    init_time = time.time() - start_time
    logger.info(f"初始化时间: {init_time:.3f} 秒")
    
    # 测试查询性能
    test_query = "machine learning artificial intelligence neural networks"
    
    # 预热
    processor._hybrid_search(test_query, large_notes[:100], top_k=10)
    
    # 正式测试
    start_time = time.time()
    results = processor._hybrid_search(test_query, large_notes, top_k=10)
    query_time = time.time() - start_time
    
    logger.info(f"查询时间: {query_time:.3f} 秒")
    logger.info(f"查询速度: {len(large_notes)/query_time:.0f} 文档/秒")
    logger.info(f"返回结果: {len(results)} 个")
    
    if results:
        top_result = results[0]
        logger.info(f"最佳匹配: {top_result['title']} (分数: {top_result.get('hybrid_score', 0):.4f})")

def main():
    """主函数"""
    logger.info("BM25 混合检索系统使用示例")
    
    try:
        # 基本演示
        demonstrate_hybrid_search()
        
        # 方法对比
        compare_search_methods()
        
        # 性能测试
        performance_benchmark()
        
        logger.info("\n=== 演示完成 ===")
        
    except Exception as e:
        logger.error(f"演示过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())