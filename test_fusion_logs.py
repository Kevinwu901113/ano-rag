#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试融合权重日志输出
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from config import config
from retrieval.listt5_reranker import fuse_scores
import json

# 设置日志级别为 DEBUG
logger.remove()
logger.add(sys.stderr, level="DEBUG")

# 同时设置标准 logging 模块的日志级别
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('retrieval.listt5_reranker').setLevel(logging.DEBUG)

def test_fusion_logging():
    """测试融合权重的日志输出"""
    logger.info("开始测试融合权重日志输出...")
    
    # 模拟候选结果
    candidates = [
        {
            'title': '测试文档1',
            'content': '这是一个测试文档',
            'hit_fact_count': 3,
            'avg_importance': 0.8,
            'predicate_coverage': 0.7,
            'temporal_coverage': 0.6,
            'cross_sentence_diversity': 0.5
        },
        {
            'title': '测试文档2', 
            'content': '这是另一个测试文档',
            'hit_fact_count': 2,
            'avg_importance': 0.6,
            'predicate_coverage': 0.5,
            'temporal_coverage': 0.4,
            'cross_sentence_diversity': 0.3
        }
    ]
    
    # 模拟ListT5分数
    list_scores = [0.8, 0.6]
    
    # 使用配置中的权重
    calibration_config = config.get('calibration', {})
    atomic_features_config = calibration_config.get('atomic_features', {})
    
    weights = {
        'listt5_weight': calibration_config.get('listt5_reranker_weight', 0.35),
        'learned_fusion_weight': calibration_config.get('learned_fusion_weight', 0.2),
        'atomic_features_weight': atomic_features_config.get('total_weight', 0.1),
        'atomic_features': {
            'hit_fact_count_weight': atomic_features_config.get('fact_count_weight', 0.3),
            'avg_importance_weight': atomic_features_config.get('avg_importance_weight', 0.25),
            'predicate_coverage_weight': atomic_features_config.get('coverage_weight', 0.25),
            'temporal_coverage_weight': atomic_features_config.get('temporal_coverage_weight', 0.15),
            'cross_sentence_diversity_weight': atomic_features_config.get('diversity_weight', 0.1)
        }
    }
    
    logger.info(f"测试权重配置: {json.dumps(weights, indent=2, ensure_ascii=False)}")
    
    # 执行融合 - 这里应该会触发我们添加的日志
    try:
        fused_candidates = fuse_scores(candidates, list_scores, weights)
        
        logger.info("融合测试完成")
        logger.info(f"处理了 {len(fused_candidates)} 个候选结果")
        
        return True
        
    except Exception as e:
        logger.error(f"融合测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_fusion_logging()
    if success:
        logger.info("✓ 融合权重日志测试通过")
    else:
        logger.error("✗ 融合权重日志测试失败")
    sys.exit(0 if success else 1)