#!/usr/bin/env python3
"""
测试EmbeddingManager单例模式修复
"""

import threading
import time
import sys
import os
from loguru import logger

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置环境变量避免网络请求
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

def test_embedding_manager_thread(thread_id):
    """在线程中测试EmbeddingManager初始化"""
    logger.info(f"Thread {thread_id}: Starting EmbeddingManager initialization")
    try:
        # 直接导入避免循环依赖
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "embedding_manager", 
            "/home/wjk/workplace/anorag/vector_store/embedding_manager.py"
        )
        embedding_module = importlib.util.module_from_spec(spec)
        
        # 手动设置必要的依赖
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import torch
        
        # 模拟config模块
        class MockConfig:
            @staticmethod
            def get(key, default=None):
                config_map = {
                    'embedding.model_name': 'BAAI/bge-m3',
                    'embedding.batch_size': 32,
                    'embedding.max_length': 512,
                    'embedding.normalize': True,
                    'performance.use_gpu': True,
                    'storage.embedding_cache_path': None,
                    'storage.work_dir': None
                }
                return config_map.get(key, default)
        
        # 模拟其他依赖
        class MockGPUUtils:
            @staticmethod
            def get_device():
                return 'cuda' if torch.cuda.is_available() else 'cpu'
        
        class MockBatchProcessor:
            def __init__(self, batch_size, use_gpu):
                self.batch_size = batch_size
                self.use_gpu = use_gpu
        
        class MockFileUtils:
            @staticmethod
            def ensure_dir(path):
                os.makedirs(path, exist_ok=True)
        
        # 设置模块的依赖
        sys.modules['config'] = type('MockModule', (), {'config': MockConfig()})()
        sys.modules['utils'] = type('MockModule', (), {
            'GPUUtils': MockGPUUtils,
            'BatchProcessor': MockBatchProcessor,
            'FileUtils': MockFileUtils
        })()
        
        spec.loader.exec_module(embedding_module)
        
        EmbeddingManager = embedding_module.EmbeddingManager
        embedding_manager = EmbeddingManager()
        
        logger.info(f"Thread {thread_id}: EmbeddingManager initialized successfully")
        logger.info(f"Thread {thread_id}: Model name: {embedding_manager.model_name}")
        logger.info(f"Thread {thread_id}: Embedding dim: {embedding_manager.embedding_dim}")
        return True
    except Exception as e:
        logger.error(f"Thread {thread_id}: Failed to initialize EmbeddingManager: {e}")
        import traceback
        logger.error(f"Thread {thread_id}: Traceback: {traceback.format_exc()}")
        return False

def test_concurrent_initialization():
    """测试并发初始化"""
    logger.info("Testing concurrent initialization...")
    
    threads = []
    results = []
    
    def thread_wrapper(thread_id):
        result = test_embedding_manager_thread(thread_id)
        results.append(result)
    
    # 创建多个线程同时初始化EmbeddingManager
    for i in range(3):
        thread = threading.Thread(target=thread_wrapper, args=(i,))
        threads.append(thread)
    
    # 启动所有线程
    for thread in threads:
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 检查结果
    if all(results):
        logger.info("✓ All threads initialized EmbeddingManager successfully")
        return True
    else:
        logger.error("✗ Some threads failed to initialize EmbeddingManager")
        return False

if __name__ == "__main__":
    logger.info("Starting EmbeddingManager fix test...")
    
    # 测试并发初始化
    if not test_concurrent_initialization():
        exit(1)
    
    logger.info("Test completed! Check the logs above for results.")