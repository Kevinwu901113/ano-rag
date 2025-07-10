#!/usr/bin/env python3
"""
验证修复后的增强组件是否正常工作
"""

import sys
import os
from typing import List, Dict, Any
from loguru import logger

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_relation_extractor():
    """测试增强关系提取器的基本功能"""
    try:
        from graph.enhanced_relation_extractor import EnhancedRelationExtractor
        from utils.text_utils import TextUtils
        
        # 创建测试数据
        test_notes = [
            {
                'note_id': 'test_1',
                'content': 'Krusty the Clown是《辛普森一家》中的角色。',
                'entities': ['Krusty the Clown', '辛普森一家'],
                'keywords': ['角色', '动画']
            },
            {
                'note_id': 'test_2',
                'content': 'Dan Castellaneta为Krusty the Clown配音。',
                'entities': ['Dan Castellaneta', 'Krusty the Clown'],
                'keywords': ['配音', '演员']
            }
        ]
        
        # 初始化关系提取器
        extractor = EnhancedRelationExtractor()
        
        # 测试关系提取
        relations = extractor.extract_all_relations(test_notes)
        
        logger.info(f"✅ 关系提取测试通过，提取到 {len(relations)} 个关系")
        
        # 测试文本相似度计算
        similarity = TextUtils.calculate_similarity_keywords("测试文本一", "测试文本二")
        logger.info(f"✅ 文本相似度计算测试通过，相似度: {similarity:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 增强关系提取器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_utils():
    """测试TextUtils的各种方法"""
    try:
        from utils.text_utils import TextUtils
        
        # 测试文本分块
        text = "这是一个测试文本。它包含多个句子。用于测试文本处理功能。"
        chunks = TextUtils.chunk_text(text, chunk_size=20)
        logger.info(f"✅ 文本分块测试通过，生成 {len(chunks)} 个块")
        
        # 测试实体提取
        entities = TextUtils.extract_entities("Apple Inc. 是一家位于 California 的公司")
        logger.info(f"✅ 实体提取测试通过，提取到实体: {entities}")
        
        # 测试关键词相似度
        similarity = TextUtils.calculate_similarity_keywords(
            "人工智能和机器学习", 
            "AI和深度学习技术"
        )
        logger.info(f"✅ 关键词相似度测试通过，相似度: {similarity:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ TextUtils测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    logger.info("开始验证修复后的组件功能")
    
    test_results = {
        'TextUtils功能': test_text_utils(),
        '增强关系提取器': test_enhanced_relation_extractor()
    }
    
    logger.info("\n=== 验证结果汇总 ===")
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(test_results.values())
    if all_passed:
        logger.info("\n🎉 所有验证测试通过！修复成功。")
    else:
        logger.error("\n⚠️ 部分验证测试失败，需要进一步检查。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)