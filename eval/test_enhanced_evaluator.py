#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强评测器测试脚本

用于测试和演示新的评测维度：
1. 路径召回率
2. 谓词一致率
3. 实体覆盖率
"""

import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入增强评测器
from enhanced_evaluator import EnhancedEvaluator, evaluate_with_enhanced_metrics

def create_sample_test_data() -> List[Dict[str, Any]]:
    """
    创建示例测试数据
    
    Returns:
        测试数据列表
    """
    test_data = [
        {
            "id": "query_001",
            "query": "什么是人工智能的发展历程？",
            "ground_truth": {
                "relevant_docs": ["doc_001", "doc_002", "doc_003"],
                "expected_entities": [
                    "人工智能", "机器学习", "深度学习", "神经网络", 
                    "图灵", "麦卡锡", "感知机", "专家系统"
                ],
                "expected_predicates": [
                    "发展", "创建", "提出", "改进", "应用", "影响"
                ],
                "expected_paths": [
                    {
                        "entities": ["图灵", "图灵测试"],
                        "relations": ["提出", "创建"]
                    },
                    {
                        "entities": ["麦卡锡", "人工智能"],
                        "relations": ["提出", "定义"]
                    },
                    {
                        "entities": ["感知机", "神经网络"],
                        "relations": ["发展", "演化"]
                    }
                ]
            }
        },
        {
            "id": "query_002",
            "query": "区块链技术的应用场景有哪些？",
            "ground_truth": {
                "relevant_docs": ["doc_004", "doc_005"],
                "expected_entities": [
                    "区块链", "比特币", "以太坊", "智能合约", 
                    "去中心化", "加密货币", "数字货币", "金融科技"
                ],
                "expected_predicates": [
                    "应用", "使用", "实现", "支持", "提供", "解决"
                ],
                "expected_paths": [
                    {
                        "entities": ["区块链", "比特币"],
                        "relations": ["支持", "实现"]
                    },
                    {
                        "entities": ["智能合约", "以太坊"],
                        "relations": ["运行", "执行"]
                    }
                ]
            }
        },
        {
            "id": "query_003",
            "query": "量子计算的基本原理是什么？",
            "ground_truth": {
                "relevant_docs": ["doc_006", "doc_007", "doc_008"],
                "expected_entities": [
                    "量子计算", "量子比特", "叠加态", "纠缠", 
                    "量子门", "量子算法", "薛定谔", "海森堡"
                ],
                "expected_predicates": [
                    "基于", "利用", "实现", "表示", "操作", "测量"
                ],
                "expected_paths": [
                    {
                        "entities": ["量子比特", "叠加态"],
                        "relations": ["处于", "表示"]
                    },
                    {
                        "entities": ["量子门", "量子操作"],
                        "relations": ["执行", "实现"]
                    }
                ]
            }
        }
    ]
    
    return test_data

def create_sample_atomic_notes() -> List[Dict]:
    """
    创建示例原子笔记数据
    
    Returns:
        原子笔记列表
    """
    atomic_notes = [
        {
            "id": "doc_001",
            "title": "人工智能发展史",
            "content": "人工智能的发展可以追溯到1950年代，当时图灵提出了著名的图灵测试。1956年，约翰·麦卡锡首次提出了'人工智能'这个术语。",
            "raw_span": "人工智能的发展可以追溯到1950年代，当时图灵提出了著名的图灵测试。1956年，约翰·麦卡锡首次提出了'人工智能'这个术语。",
            "namespace": "ai_history",
            "metadata": {"source": "AI教科书", "chapter": "第一章"}
        },
        {
            "id": "doc_002",
            "title": "神经网络发展",
            "content": "感知机是最早的神经网络模型之一，由罗森布拉特在1957年提出。后来发展出了多层感知机和深度学习网络。",
            "raw_span": "感知机是最早的神经网络模型之一，由罗森布拉特在1957年提出。后来发展出了多层感知机和深度学习网络。",
            "namespace": "ai_history",
            "metadata": {"source": "神经网络原理", "chapter": "第二章"}
        },
        {
            "id": "doc_003",
            "title": "机器学习里程碑",
            "content": "机器学习经历了多个重要阶段：专家系统时代、统计学习方法兴起、深度学习革命。每个阶段都有其代表性的算法和应用。",
            "raw_span": "机器学习经历了多个重要阶段：专家系统时代、统计学习方法兴起、深度学习革命。每个阶段都有其代表性的算法和应用。",
            "namespace": "ai_history",
            "metadata": {"source": "机器学习概论", "chapter": "第三章"}
        },
        {
            "id": "doc_004",
            "title": "区块链应用",
            "content": "区块链技术在金融领域有广泛应用，包括数字货币、智能合约、去中心化金融等。比特币是第一个成功的区块链应用。",
            "raw_span": "区块链技术在金融领域有广泛应用，包括数字货币、智能合约、去中心化金融等。比特币是第一个成功的区块链应用。",
            "namespace": "blockchain",
            "metadata": {"source": "区块链技术指南", "chapter": "第四章"}
        },
        {
            "id": "doc_005",
            "title": "以太坊平台",
            "content": "以太坊是一个支持智能合约的区块链平台，它扩展了区块链的应用范围，使得去中心化应用成为可能。",
            "raw_span": "以太坊是一个支持智能合约的区块链平台，它扩展了区块链的应用范围，使得去中心化应用成为可能。",
            "namespace": "blockchain",
            "metadata": {"source": "以太坊白皮书", "chapter": "第一章"}
        },
        {
            "id": "doc_006",
            "title": "量子计算原理",
            "content": "量子计算基于量子力学原理，利用量子比特的叠加态和纠缠特性进行计算。量子比特可以同时处于0和1的叠加态。",
            "raw_span": "量子计算基于量子力学原理，利用量子比特的叠加态和纠缠特性进行计算。量子比特可以同时处于0和1的叠加态。",
            "namespace": "quantum",
            "metadata": {"source": "量子计算导论", "chapter": "第一章"}
        },
        {
            "id": "doc_007",
            "title": "量子门操作",
            "content": "量子门是量子计算中的基本操作单元，通过量子门可以对量子比特进行各种操作，实现量子算法。",
            "raw_span": "量子门是量子计算中的基本操作单元，通过量子门可以对量子比特进行各种操作，实现量子算法。",
            "namespace": "quantum",
            "metadata": {"source": "量子算法", "chapter": "第二章"}
        },
        {
            "id": "doc_008",
            "title": "量子纠缠现象",
            "content": "量子纠缠是量子力学中的一种现象，两个或多个粒子之间存在非局域关联，这种特性被用于量子通信和量子计算。",
            "raw_span": "量子纠缠是量子力学中的一种现象，两个或多个粒子之间存在非局域关联，这种特性被用于量子通信和量子计算。",
            "namespace": "quantum",
            "metadata": {"source": "量子物理学", "chapter": "第三章"}
        }
    ]
    
    return atomic_notes

def mock_query_processor_results() -> Dict[str, Any]:
    """
    模拟查询处理器的结果
    
    Returns:
        模拟的查询结果
    """
    return {
        "results": [
            {
                "id": "doc_001",
                "note_id": "doc_001",
                "content": "人工智能的发展可以追溯到1950年代，当时图灵提出了著名的图灵测试。1956年，约翰·麦卡锡首次提出了'人工智能'这个术语。",
                "similarity_score": 0.85,
                "path_score": 0.75,
                "metadata": {"source": "AI教科书"}
            },
            {
                "id": "doc_002",
                "note_id": "doc_002",
                "content": "感知机是最早的神经网络模型之一，由罗森布拉特在1957年提出。后来发展出了多层感知机和深度学习网络。",
                "similarity_score": 0.78,
                "path_score": 0.65,
                "metadata": {"source": "神经网络原理"}
            },
            {
                "id": "doc_003",
                "note_id": "doc_003",
                "content": "机器学习经历了多个重要阶段：专家系统时代、统计学习方法兴起、深度学习革命。每个阶段都有其代表性的算法和应用。",
                "similarity_score": 0.72,
                "path_score": 0.55,
                "metadata": {"source": "机器学习概论"}
            }
        ],
        "query_info": {
            "original_query": "什么是人工智能的发展历程？",
            "rewritten_queries": ["人工智能发展历史", "AI发展过程"]
        },
        "metadata": {
            "total_candidates": 50,
            "final_count": 3,
            "execution_time": 0.245
        }
    }

class MockQueryProcessor:
    """
    模拟查询处理器，用于测试
    """
    
    def __init__(self, atomic_notes, embeddings=None, llm=None):
        self.atomic_notes = atomic_notes
        self.embeddings = embeddings
        self.llm = llm
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        模拟处理查询
        
        Args:
            query: 查询文本
            
        Returns:
            模拟的查询结果
        """
        # 根据查询内容返回不同的模拟结果
        if "人工智能" in query:
            return {
                "results": [
                    {
                        "id": "doc_001",
                        "content": "人工智能的发展可以追溯到1950年代，当时图灵提出了著名的图灵测试。1956年，约翰·麦卡锡首次提出了'人工智能'这个术语。",
                        "similarity_score": 0.85,
                        "path_score": 0.75
                    },
                    {
                        "id": "doc_002",
                        "content": "感知机是最早的神经网络模型之一，由罗森布拉特在1957年提出。后来发展出了多层感知机和深度学习网络。",
                        "similarity_score": 0.78,
                        "path_score": 0.65
                    }
                ]
            }
        elif "区块链" in query:
            return {
                "results": [
                    {
                        "id": "doc_004",
                        "content": "区块链技术在金融领域有广泛应用，包括数字货币、智能合约、去中心化金融等。比特币是第一个成功的区块链应用。",
                        "similarity_score": 0.82,
                        "path_score": 0.70
                    },
                    {
                        "id": "doc_005",
                        "content": "以太坊是一个支持智能合约的区块链平台，它扩展了区块链的应用范围，使得去中心化应用成为可能。",
                        "similarity_score": 0.75,
                        "path_score": 0.60
                    }
                ]
            }
        elif "量子" in query:
            return {
                "results": [
                    {
                        "id": "doc_006",
                        "content": "量子计算基于量子力学原理，利用量子比特的叠加态和纠缠特性进行计算。量子比特可以同时处于0和1的叠加态。",
                        "similarity_score": 0.88,
                        "path_score": 0.80
                    },
                    {
                        "id": "doc_007",
                        "content": "量子门是量子计算中的基本操作单元，通过量子门可以对量子比特进行各种操作，实现量子算法。",
                        "similarity_score": 0.76,
                        "path_score": 0.68
                    }
                ]
            }
        else:
            return {"results": []}

class MockEnhancedEvaluator(EnhancedEvaluator):
    """
    增强评测器的模拟版本，用于测试
    """
    
    def __init__(self, atomic_notes, embeddings=None, llm=None):
        # 使用模拟的查询处理器
        self.processor = MockQueryProcessor(atomic_notes, embeddings, llm)
        self.atomic_notes = atomic_notes
        
        # 配置参数
        self.batch_size = 16
        self.metrics = ["precision", "recall", "f1", "path_recall", "predicate_consistency", "entity_coverage"]
        
        # 初始化标准化器（简化版）
        self.entity_normalizer = None
        self.predicate_normalizer = None
        self.ner = None
        
        # 统计信息
        self.stats = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'avg_execution_time': 0.0
        }
        
        logger.info(f"Mock enhanced evaluator initialized with metrics: {self.metrics}")
    
    def _extract_entities_from_content(self, content: str) -> List[str]:
        """简化的实体提取"""
        # 简单的关键词匹配
        entities = []
        keywords = [
            "人工智能", "机器学习", "深度学习", "神经网络", "图灵", "麦卡锡", "感知机",
            "区块链", "比特币", "以太坊", "智能合约", "去中心化",
            "量子计算", "量子比特", "叠加态", "纠缠", "量子门"
        ]
        
        for keyword in keywords:
            if keyword in content:
                entities.append(keyword)
        
        return entities
    
    def _extract_relations_from_content(self, content: str) -> List[str]:
        """简化的关系提取"""
        relations = []
        relation_words = ["提出", "创建", "发展", "应用", "支持", "实现", "基于", "利用"]
        
        for word in relation_words:
            if word in content:
                relations.append(word)
        
        return relations

def run_test():
    """
    运行测试
    """
    logger.info("开始增强评测器测试")
    
    # 创建测试数据
    test_data = create_sample_test_data()
    atomic_notes = create_sample_atomic_notes()
    
    logger.info(f"创建了 {len(test_data)} 个测试查询")
    logger.info(f"创建了 {len(atomic_notes)} 个原子笔记")
    
    # 创建模拟评测器
    evaluator = MockEnhancedEvaluator(atomic_notes)
    
    # 执行评测
    logger.info("开始执行批量评测...")
    summary = evaluator.evaluate_batch(test_data)
    
    # 打印结果
    evaluator.print_summary(summary)
    
    # 保存结果
    output_path = "/home/wjk/workplace/anorag/eval/test_results.json"
    evaluator.save_results(summary, output_path)
    
    logger.info(f"测试完成，结果已保存到 {output_path}")
    
    return summary

def demonstrate_new_metrics():
    """
    演示新的评测指标
    """
    print("\n" + "="*60)
    print("新增评测指标演示")
    print("="*60)
    
    print("\n1. 路径召回率 (Path Recall Rate)")
    print("   - 衡量系统是否能够召回预期的知识路径")
    print("   - 基于实体-关系路径的匹配度计算")
    print("   - 适用于评估知识图谱相关的检索任务")
    
    print("\n2. 谓词一致率 (Predicate Consistency Rate)")
    print("   - 衡量检索结果中谓词/关系的一致性")
    print("   - 通过标准化谓词进行匹配")
    print("   - 有助于评估语义理解的准确性")
    
    print("\n3. 实体覆盖率 (Entity Coverage Rate)")
    print("   - 衡量检索结果对关键实体的覆盖程度")
    print("   - 通过实体标准化和匹配计算")
    print("   - 评估信息完整性的重要指标")
    
    print("\n这些指标与传统的精确率、召回率、F1分数互补，")
    print("提供了更全面的检索质量评估维度。")
    print("="*60)

if __name__ == "__main__":
    # 演示新指标
    demonstrate_new_metrics()
    
    # 运行测试
    try:
        summary = run_test()
        
        print("\n测试总结:")
        print(f"- 成功评测了 {summary.total_queries} 个查询")
        print(f"- 平均路径召回率: {summary.avg_path_recall_rate:.3f}")
        print(f"- 平均谓词一致率: {summary.avg_predicate_consistency_rate:.3f}")
        print(f"- 平均实体覆盖率: {summary.avg_entity_coverage_rate:.3f}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()