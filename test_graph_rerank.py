#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图感知两阶段rerank系统测试脚本

测试新实现的图感知rerank功能，包括：
1. 配置参数加载
2. GraphAwareRetrieval类初始化
3. ContextDispatcher图感知模式
4. 回退机制验证
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from loguru import logger

def test_config_loading():
    """测试图感知rerank配置参数加载"""
    logger.info("Testing graph-aware rerank configuration loading...")
    
    # 检查基本配置参数
    use_graph_rerank = config.get('context_dispatcher.graph_aware_rerank.use_graph_rerank', False)
    seeds_semantic = config.get('context_dispatcher.graph_aware_rerank.seeds_semantic', 5)
    seeds_bm25 = config.get('context_dispatcher.graph_aware_rerank.seeds_bm25', 3)
    subgraph_radius = config.get('context_dispatcher.graph_aware_rerank.subgraph_radius', 2)
    
    logger.info(f"use_graph_rerank: {use_graph_rerank}")
    logger.info(f"seeds_semantic: {seeds_semantic}")
    logger.info(f"seeds_bm25: {seeds_bm25}")
    logger.info(f"subgraph_radius: {subgraph_radius}")
    
    # 检查路径生成参数
    k_paths = config.get('context_dispatcher.graph_aware_rerank.k_paths', 10)
    pick_paths = config.get('context_dispatcher.graph_aware_rerank.pick_paths', 3)
    token_budget = config.get('context_dispatcher.graph_aware_rerank.token_budget', 2000)
    
    logger.info(f"k_paths: {k_paths}")
    logger.info(f"pick_paths: {pick_paths}")
    logger.info(f"token_budget: {token_budget}")
    
    return True

def test_graph_retrieval_import():
    """测试GraphAwareRetrieval类导入"""
    logger.info("Testing GraphAwareRetrieval import...")
    
    try:
        from graph.graph_retrieval import GraphAwareRetrieval, create_graph_aware_retrieval
        logger.info("✓ GraphAwareRetrieval imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import GraphAwareRetrieval: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error importing GraphAwareRetrieval: {e}")
        return False

def test_context_dispatcher_import():
    """测试ContextDispatcher导入和初始化"""
    logger.info("Testing ContextDispatcher import...")
    
    try:
        from utils.context_dispatcher import ContextDispatcher
        logger.info("✓ ContextDispatcher imported successfully")
        
        # 测试初始化（不需要实际的graph_index和vector_retriever）
        dispatcher = ContextDispatcher(config, graph_index=None, vector_retriever=None)
        logger.info("✓ ContextDispatcher initialized successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import ContextDispatcher: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error with ContextDispatcher: {e}")
        return False

def test_fallback_mechanism():
    """测试回退机制"""
    logger.info("Testing fallback mechanism...")
    
    try:
        from utils.context_dispatcher import ContextDispatcher
        
        # 测试在没有graph_index时的回退
        dispatcher = ContextDispatcher(config, graph_index=None, vector_retriever=None)
        
        # 模拟候选结果
        mock_candidates = [
            {'note_id': '1', 'content': 'test content 1', 'final_similarity': 0.9},
            {'note_id': '2', 'content': 'test content 2', 'final_similarity': 0.8},
            {'note_id': '3', 'content': 'test content 3', 'final_similarity': 0.7}
        ]
        
        # 测试dispatch方法（应该回退到legacy模式）
        result = dispatcher.dispatch(mock_candidates, query="test query")
        logger.info(f"✓ Fallback mechanism works, returned {len(result)} candidates")
        return True
    except Exception as e:
        logger.error(f"✗ Fallback mechanism failed: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("Starting graph-aware rerank system tests...")
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("GraphAwareRetrieval Import", test_graph_retrieval_import),
        ("ContextDispatcher Import", test_context_dispatcher_import),
        ("Fallback Mechanism", test_fallback_mechanism)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                logger.info(f"✓ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Graph-aware rerank system is ready.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)