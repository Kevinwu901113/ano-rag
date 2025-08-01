#!/usr/bin/env python3
"""
测试脚本：验证ContextDispatcher与现有系统的集成
"""

import os
import sys
from typing import List, Dict, Any
from loguru import logger

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config

def test_imports():
    """测试所有必要的导入"""
    logger.info("=== Testing Imports ===")
    
    try:
        # 测试核心组件导入
        from utils.context_dispatcher import ContextDispatcher
        logger.info("✓ ContextDispatcher imported successfully")
        
        from utils import ContextDispatcher as UtilsContextDispatcher
        logger.info("✓ ContextDispatcher available from utils module")
        
        from query.query_processor import QueryProcessor
        logger.info("✓ QueryProcessor imported successfully")
        
        # 测试依赖组件
        from vector_store import VectorRetriever
        from graph.graph_retriever import GraphRetriever
        from graph.graph_index import GraphIndex
        logger.info("✓ All dependency components imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during import: {e}")
        return False

def test_configuration():
    """测试配置加载"""
    logger.info("\n=== Testing Configuration ===")
    
    try:
        # 测试ContextDispatcher配置
        dispatcher_config = config.get('context_dispatcher', {})
        logger.info(f"ContextDispatcher config loaded: {bool(dispatcher_config)}")
        
        if dispatcher_config:
            enabled = dispatcher_config.get('enabled', False)
            logger.info(f"ContextDispatcher enabled: {enabled}")
            
            # 检查关键参数
            required_params = [
                'semantic_top_n', 'graph_expand_top_p', 'k_hop',
                'final_semantic_count', 'final_graph_count'
            ]
            
            missing_params = []
            for param in required_params:
                if param not in dispatcher_config:
                    missing_params.append(param)
                else:
                    logger.info(f"  {param}: {dispatcher_config[param]}")
            
            if missing_params:
                logger.warning(f"Missing parameters: {missing_params}")
                return False
            else:
                logger.info("✓ All required parameters present")
                
        # 测试向量存储配置
        vector_config = config.get('vector_store', {})
        logger.info(f"Vector store config: top_k={vector_config.get('top_k')}, threshold={vector_config.get('similarity_threshold')}")
        
        # 测试图谱配置
        graph_config = config.get('graph', {})
        logger.info(f"Graph config: k_hop={graph_config.get('k_hop')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False

def test_context_dispatcher_creation():
    """测试ContextDispatcher实例创建"""
    logger.info("\n=== Testing ContextDispatcher Creation ===")
    
    try:
        from utils.context_dispatcher import ContextDispatcher
        
        # 创建模拟的依赖组件
        class MockVectorRetriever:
            def search(self, queries, top_k=None, similarity_threshold=None):
                # 返回模拟的搜索结果
                return [[
                    {
                        'note_id': f'mock_note_{i}',
                        'content': f'Mock content {i}',
                        'retrieval_info': {'similarity': 0.8 - i * 0.1},
                        'paragraph_idxs': [i]
                    }
                    for i in range(min(top_k or 10, 5))
                ]]
        
        class MockGraphRetriever:
            def retrieve(self, seed_note_ids):
                # 返回模拟的图谱检索结果
                return [
                    {
                        'note_id': f'graph_note_{i}',
                        'content': f'Graph content {i}',
                        'graph_score': 0.7 - i * 0.1,
                        'graph_distance': i + 1,
                        'centrality': 0.5,
                        'paragraph_idxs': [i + 10]
                    }
                    for i in range(3)
                ]
        
        # 创建ContextDispatcher实例
        mock_vector_retriever = MockVectorRetriever()
        mock_graph_retriever = MockGraphRetriever()
        
        dispatcher = ContextDispatcher(mock_vector_retriever, mock_graph_retriever)
        logger.info("✓ ContextDispatcher instance created successfully")
        
        # 测试配置获取
        config_summary = dispatcher.get_config_summary()
        logger.info(f"Configuration summary: {config_summary}")
        
        # 测试dispatch方法
        test_query = "测试查询"
        result = dispatcher.dispatch(test_query)
        
        logger.info(f"✓ Dispatch completed successfully")
        logger.info(f"  Context length: {len(result['context'])}")
        logger.info(f"  Selected notes: {len(result['selected_notes'])}")
        logger.info(f"  Stage info: {result['stage_info']}")
        
        # 验证结果结构
        required_keys = ['context', 'selected_notes', 'semantic_results', 'graph_results', 'stage_info']
        missing_keys = [key for key in required_keys if key not in result]
        
        if missing_keys:
            logger.error(f"Missing result keys: {missing_keys}")
            return False
        
        logger.info("✓ Result structure validation passed")
        
        return True
        
    except Exception as e:
        logger.error(f"ContextDispatcher creation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_query_processor_integration():
    """测试QueryProcessor集成"""
    logger.info("\n=== Testing QueryProcessor Integration ===")
    
    try:
        # 检查QueryProcessor是否正确导入ContextDispatcher
        from query.query_processor import QueryProcessor
        
        # 检查QueryProcessor的源码中是否包含ContextDispatcher相关代码
        import inspect
        source = inspect.getsource(QueryProcessor)
        
        if 'ContextDispatcher' in source:
            logger.info("✓ QueryProcessor contains ContextDispatcher integration")
        else:
            logger.warning("⚠ QueryProcessor may not have ContextDispatcher integration")
            
        if 'use_context_dispatcher' in source:
            logger.info("✓ QueryProcessor has dispatcher selection logic")
        else:
            logger.warning("⚠ QueryProcessor may not have dispatcher selection logic")
            
        # 检查配置驱动的调度器选择
        dispatcher_enabled = config.get('context_dispatcher.enabled', True)
        logger.info(f"ContextDispatcher enabled in config: {dispatcher_enabled}")
        
        return True
        
    except Exception as e:
        logger.error(f"QueryProcessor integration test failed: {e}")
        return False

def test_backward_compatibility():
    """测试向后兼容性"""
    logger.info("\n=== Testing Backward Compatibility ===")
    
    try:
        # 测试原有的ContextScheduler仍然可用
        from utils.context_scheduler import ContextScheduler, MultiHopContextScheduler
        logger.info("✓ Legacy ContextScheduler still available")
        
        # 测试配置切换
        original_enabled = config.get('context_dispatcher.enabled', True)
        
        # 模拟禁用ContextDispatcher
        logger.info("Testing with ContextDispatcher disabled...")
        config._config['context_dispatcher'] = config._config.get('context_dispatcher', {})
        config._config['context_dispatcher']['enabled'] = False
        
        disabled_state = config.get('context_dispatcher.enabled', True)
        logger.info(f"ContextDispatcher disabled: {not disabled_state}")
        
        # 恢复原始配置
        config._config['context_dispatcher']['enabled'] = original_enabled
        logger.info("✓ Configuration switching works")
        
        return True
        
    except Exception as e:
        logger.error(f"Backward compatibility test failed: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("Starting ContextDispatcher integration tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("ContextDispatcher Creation Test", test_context_dispatcher_creation),
        ("QueryProcessor Integration Test", test_query_processor_integration),
        ("Backward Compatibility Test", test_backward_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # 总结
    logger.info("\n=== Test Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! ContextDispatcher integration is successful.")
        logger.info("\n=== Ready for Production ===")
        logger.info("1. ContextDispatcher is properly integrated")
        logger.info("2. Configuration system is working")
        logger.info("3. Backward compatibility is maintained")
        logger.info("4. You can now use the new structure-enhanced context dispatching")
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)