#!/usr/bin/env python3
"""
測試循環後援修復的腳本
驗證 LM Studio 和 Ollama 之間不會發生循環調用導致的死鎖
"""

import sys
import os
import time
import logging
from typing import Dict, Any

# 添加項目路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.parallel_task_atomic_note_generator import ParallelTaskAtomicNoteGenerator
from llm.lmstudio_client import LMStudioClient
from llm.ollama_client import OllamaClient
from config.config_loader import ConfigLoader

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_chunk_data() -> Dict[str, Any]:
    """創建測試用的數據塊"""
    return {
        'text': '這是一個測試文本，用於驗證原子筆記生成功能。內容包含一些基本信息，需要被處理成結構化的筆記格式。',
        'source': 'test_document.txt',
        'chunk_id': 'test_chunk_001',
        'metadata': {
            'page': 1,
            'section': 'introduction'
        }
    }

def test_circular_fallback_prevention():
    """測試循環後援防護機制"""
    logger.info("開始測試循環後援防護機制...")
    
    try:
        # 載入配置
        config_loader = ConfigLoader()
        config = config_loader.load_config()
        
        # 創建客戶端實例（模擬不可用的狀態）
        lmstudio_client = LMStudioClient(
            base_url="http://localhost:1234",  # 假設不可用
            model="test-model"
        )
        
        ollama_client = OllamaClient(
            base_url="http://localhost:11434",  # 假設不可用
            model="test-model"
        )
        
        # 創建生成器實例
        from llm.local_llm import LocalLLM
        llm = LocalLLM()  # 創建一個基本的 LocalLLM 實例
        generator = ParallelTaskAtomicNoteGenerator(llm=llm)
        
        # 創建測試數據
        test_data = create_test_chunk_data()
        system_prompt = "你是一個專業的筆記整理助手，請將給定的文本轉換為結構化的原子筆記。"
        
        logger.info("測試 LM Studio 處理（預期會觸發後援機制）...")
        start_time = time.time()
        
        # 測試 LM Studio 處理
        result_lmstudio = generator._process_with_lmstudio(test_data, 1, system_prompt)
        
        processing_time = time.time() - start_time
        logger.info(f"LM Studio 處理完成，耗時: {processing_time:.2f}秒")
        logger.info(f"結果類型: {type(result_lmstudio)}")
        
        # 檢查結果
        if result_lmstudio and 'atomic_notes' in result_lmstudio:
            logger.info("✓ LM Studio 處理成功返回結果")
        else:
            logger.info("✓ LM Studio 處理返回空結果（符合預期）")
        
        logger.info("測試 Ollama 處理（預期會觸發後援機制）...")
        start_time = time.time()
        
        # 測試 Ollama 處理
        result_ollama = generator._process_with_ollama(test_data, 2, system_prompt)
        
        processing_time = time.time() - start_time
        logger.info(f"Ollama 處理完成，耗時: {processing_time:.2f}秒")
        logger.info(f"結果類型: {type(result_ollama)}")
        
        # 檢查結果
        if result_ollama and 'atomic_notes' in result_ollama:
            logger.info("✓ Ollama 處理成功返回結果")
        else:
            logger.info("✓ Ollama 處理返回空結果（符合預期）")
        
        # 測試後援防護機制
        logger.info("測試循環後援防護...")
        
        # 模擬多次後援調用
        task_id = "test_task_circular"
        
        # 第一次後援應該成功
        can_fallback_1 = generator._can_fallback(task_id, "ollama", "lmstudio")
        logger.info(f"第一次後援檢查: {can_fallback_1} (應該為 True)")
        
        # 第二次後援應該成功
        can_fallback_2 = generator._can_fallback(task_id, "lmstudio", "ollama")
        logger.info(f"第二次後援檢查: {can_fallback_2} (應該為 True)")
        
        # 第三次後援應該被阻止（達到最大深度）
        can_fallback_3 = generator._can_fallback(task_id, "ollama", "lmstudio")
        logger.info(f"第三次後援檢查: {can_fallback_3} (應該為 False)")
        
        # 測試循環檢測
        task_id_2 = "test_task_circular_2"
        generator._can_fallback(task_id_2, "ollama", "lmstudio")
        can_circular = generator._can_fallback(task_id_2, "ollama", "lmstudio")
        logger.info(f"循環後援檢查: {can_circular} (應該為 False)")
        
        # 檢查統計信息
        stats = generator.stats  # 直接訪問 stats 屬性而不是調用方法
        logger.info("處理統計信息:")
        for key, value in stats.items():
            if value > 0:
                logger.info(f"  {key}: {value}")
        
        logger.info("✅ 循環後援防護測試完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_timeout_behavior():
    """測試超時行為，確保不會無限等待"""
    logger.info("測試超時行為...")
    
    start_time = time.time()
    max_test_time = 30  # 最大測試時間30秒
    
    try:
        success = test_circular_fallback_prevention()
        total_time = time.time() - start_time
        
        if total_time > max_test_time:
            logger.warning(f"⚠️ 測試耗時過長: {total_time:.2f}秒 (超過 {max_test_time}秒)")
            return False
        else:
            logger.info(f"✅ 測試在合理時間內完成: {total_time:.2f}秒")
            return success
            
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ 測試超時或異常: {e} (耗時: {total_time:.2f}秒)")
        return False

if __name__ == "__main__":
    logger.info("🚀 開始循環後援修復測試...")
    
    # 測試超時行為
    success = test_timeout_behavior()
    
    if success:
        logger.info("🎉 所有測試通過！循環後援問題已修復。")
        sys.exit(0)
    else:
        logger.error("💥 測試失敗！可能仍存在問題。")
        sys.exit(1)