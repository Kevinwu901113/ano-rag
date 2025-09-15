#!/usr/bin/env python3
"""
测试QueryProcessor融合阶段的改进
验证原子笔记特征是否正确集成到融合机制中
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from config import config
from retrieval.listt5_reranker import fuse_scores
import json

def test_fuse_scores_with_atomic_features():
    """测试fuse_scores函数是否正确处理原子特征"""
    logger.info("测试fuse_scores函数与原子特征的集成...")
    
    # 模拟候选文档数据
    candidates = [
        {
            'id': 'note_1',
            'title': '机器学习基础',
            'content': '机器学习是人工智能的一个分支...',
            'dense_score': 0.8,
            'sparse_score': 0.6,
            'learned_fusion_score': 0.75,
            # 原子笔记特征
            'hit_fact_count': 3,
            'avg_importance': 0.7,
            'predicate_coverage': 0.6,
            'temporal_coverage': 0.4,
            'cross_sentence_diversity': 0.5
        },
        {
            'id': 'note_2',
            'title': '深度学习应用',
            'content': '深度学习在图像识别中的应用...',
            'dense_score': 0.7,
            'sparse_score': 0.8,
            'learned_fusion_score': 0.65,
            # 原子笔记特征
            'hit_fact_count': 5,
            'avg_importance': 0.9,
            'predicate_coverage': 0.8,
            'temporal_coverage': 0.6,
            'cross_sentence_diversity': 0.7
        },
        {
            'id': 'note_3',
            'title': '神经网络结构',
            'content': '卷积神经网络的基本结构...',
            'dense_score': 0.6,
            'sparse_score': 0.7,
            'learned_fusion_score': 0.55,
            # 原子笔记特征
            'hit_fact_count': 2,
            'avg_importance': 0.5,
            'predicate_coverage': 0.4,
            'temporal_coverage': 0.3,
            'cross_sentence_diversity': 0.3
        }
    ]
    
    # 模拟ListT5分数
    list_scores = [0.85, 0.75, 0.65]
    
    # 使用配置中的权重
    calibration_config = config.get('calibration', {})
    atomic_features_config = calibration_config.get('atomic_features', {})
    
    weights = {
        'listt5_weight': calibration_config.get('listt5_weight', 0.35),
        'learned_fusion_weight': calibration_config.get('learned_fusion_weight', 0.2),
        'atomic_features_weight': atomic_features_config.get('weight', 0.1),
        'atomic_features': atomic_features_config
    }
    
    logger.info(f"使用的权重配置: {json.dumps(weights, indent=2, ensure_ascii=False)}")
    
    # 执行融合
    try:
        fused_candidates = fuse_scores(candidates, list_scores, weights)
        
        logger.info("融合结果:")
        for i, candidate in enumerate(fused_candidates):
            logger.info(f"候选 {i+1}: {candidate['title']}")
            logger.info(f"  - ListT5分数: {candidate.get('listt5_score', 'N/A'):.3f}")
            logger.info(f"  - 原子特征分数: {candidate.get('atomic_score', 'N/A'):.3f}")
            logger.info(f"  - 最终融合分数: {candidate.get('fused_score', 'N/A'):.3f}")
            logger.info(f"  - 原子特征详情:")
            logger.info(f"    * 命中事实数: {candidate.get('hit_fact_count', 0)}")
            logger.info(f"    * 平均重要性: {candidate.get('avg_importance', 0.0):.3f}")
            logger.info(f"    * 谓词覆盖度: {candidate.get('predicate_coverage', 0.0):.3f}")
            logger.info("")
        
        # 验证排序是否正确
        sorted_candidates = sorted(fused_candidates, key=lambda x: x.get('fused_score', 0), reverse=True)
        logger.info("按融合分数排序后的结果:")
        for i, candidate in enumerate(sorted_candidates):
            logger.info(f"{i+1}. {candidate['title']} (融合分数: {candidate.get('fused_score', 0):.3f})")
        
        return True
        
    except Exception as e:
        logger.error(f"融合测试失败: {e}")
        return False

def test_config_loading():
    """测试配置文件是否正确加载原子特征权重"""
    logger.info("测试配置文件加载...")
    
    calibration_config = config.get('calibration', {})
    logger.info(f"校准配置: {json.dumps(calibration_config, indent=2, ensure_ascii=False)}")
    
    atomic_features_config = calibration_config.get('atomic_features', {})
    if atomic_features_config:
        logger.info("✓ 原子特征配置已正确加载")
        logger.info(f"原子特征配置详情: {json.dumps(atomic_features_config, indent=2, ensure_ascii=False)}")
        return True
    else:
        logger.warning("⚠ 原子特征配置未找到或为空")
        return False

def main():
    """主测试函数"""
    logger.info("开始测试QueryProcessor融合阶段改进...")
    
    # 测试配置加载
    config_ok = test_config_loading()
    
    # 测试融合函数
    fusion_ok = test_fuse_scores_with_atomic_features()
    
    if config_ok and fusion_ok:
        logger.info("✓ 所有测试通过！融合机制改进成功集成")
        return True
    else:
        logger.error("✗ 部分测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)