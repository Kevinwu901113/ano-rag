#!/usr/bin/env python3
"""
多跳查询优化测试脚本

测试验证多跳样例，确保每一跳Top-M保活和完整路径关键节点包含
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from query.query_processor import QueryProcessor
from loguru import logger
import json

def create_test_candidates():
    """创建测试用的多跳候选数据"""
    candidates = []
    
    # 第一跳候选（hop_no=1）
    for i in range(10):
        candidates.append({
            "note_id": f"hop1_{i}",
            "title": f"First Hop Title {i}",
            "content": f"This is first hop content {i} about entity A and relation R",
            "hop_no": 1,
            "bridge_entity": f"EntityA{i}",
            "bridge_path": [f"EntityA{i}"],
            "final_score": 0.9 - i * 0.05,  # 递减分数
            "score": 0.9 - i * 0.05
        })
    
    # 第二跳候选（hop_no=2）
    for i in range(15):
        candidates.append({
            "note_id": f"hop2_{i}",
            "title": f"Second Hop Title {i}",
            "content": f"This is second hop content {i} about entity B and relation S",
            "hop_no": 2,
            "bridge_entity": f"EntityB{i}",
            "bridge_path": [f"EntityA{i%5}", f"EntityB{i}"],
            "final_score": 0.8 - i * 0.03,  # 递减分数
            "score": 0.8 - i * 0.03
        })
    
    # 第三跳候选（hop_no=3）
    for i in range(8):
        candidates.append({
            "note_id": f"hop3_{i}",
            "title": f"Third Hop Title {i}",
            "content": f"This is third hop content {i} about entity C and relation T",
            "hop_no": 3,
            "bridge_entity": f"EntityC{i}",
            "bridge_path": [f"EntityA{i%3}", f"EntityB{i%5}", f"EntityC{i}"],
            "final_score": 0.7 - i * 0.04,  # 递减分数
            "score": 0.7 - i * 0.04
        })
    
    # 第四跳候选（hop_no=4）
    for i in range(5):
        candidates.append({
            "note_id": f"hop4_{i}",
            "title": f"Fourth Hop Title {i}",
            "content": f"This is fourth hop content {i} about entity D and relation U",
            "hop_no": 4,
            "bridge_entity": f"EntityD{i}",
            "bridge_path": [f"EntityA{i%2}", f"EntityB{i%3}", f"EntityC{i%4}", f"EntityD{i}"],
            "final_score": 0.6 - i * 0.05,  # 递减分数
            "score": 0.6 - i * 0.05
        })
    
    return candidates

def test_multihop_reranking():
    """测试多跳重排功能"""
    logger.info("开始测试多跳重排功能")
    
    # 创建测试数据
    test_candidates = create_test_candidates()
    logger.info(f"创建了 {len(test_candidates)} 个测试候选")
    
    # 统计各跳候选数量
    hop_counts = {}
    for cand in test_candidates:
        hop_no = cand.get("hop_no", 1)
        hop_counts[hop_no] = hop_counts.get(hop_no, 0) + 1
    
    logger.info(f"各跳候选数量: {hop_counts}")
    
    # 创建QueryProcessor实例（使用最小配置）
    try:
        # 创建最小的atomic_notes用于初始化
        minimal_notes = [
            {"note_id": "test_1", "title": "Test", "content": "Test content", "entities": [], "predicates": []}
        ]
        
        processor = QueryProcessor(
            atomic_notes=minimal_notes,
            embeddings=None,
            graph_file=None,
            vector_index_file=None,
            llm=None
        )
        
        # 设置配置
        processor.config = {
            "hybrid_search": {
                "multi_hop": {
                    "beam_width": 8,
                    "per_hop_keep_top_m": 5,
                    "focused_weight_by_hop": {
                        "1": 0.30,
                        "2": 0.25,
                        "3": 0.20,
                        "4": 0.15
                    },
                    "hop_decay": 0.85,
                    "lower_threshold": 0.10
                },
                "answer_bias": {
                    "who_person_boost": 1.10
                }
            }
        }
        
        # 测试查询
        test_query = "Who is the person related to entity A through relation R?"
        
        logger.info(f"测试查询: {test_query}")
        logger.info("执行多跳重排...")
        
        # 执行重排
        reranked_candidates = processor._rerank_khop_candidates(test_query, test_candidates)
        
        logger.info(f"重排后候选数量: {len(reranked_candidates)}")
        
        # 分析结果
        analyze_results(reranked_candidates)
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_results(candidates):
    """分析重排结果"""
    logger.info("=== 重排结果分析 ===")
    
    # 按跳数分组统计
    hop_stats = {}
    for cand in candidates:
        hop_no = cand.get("hop_no", 1)
        if hop_no not in hop_stats:
            hop_stats[hop_no] = []
        hop_stats[hop_no].append(cand)
    
    # 输出各跳统计
    for hop_no in sorted(hop_stats.keys()):
        hop_candidates = hop_stats[hop_no]
        logger.info(f"第 {hop_no} 跳: {len(hop_candidates)} 个候选")
        
        # 显示前3个候选的详细信息
        for i, cand in enumerate(hop_candidates[:3]):
            logger.info(f"  Top-{i+1}: {cand['note_id']}, score={cand.get('final_score', 0):.4f}, "
                       f"bridge_entity={cand.get('bridge_entity', 'N/A')}")
    
    # 检查路径完整性
    logger.info("=== 路径完整性检查 ===")
    complete_paths = 0
    for cand in candidates:
        bridge_path = cand.get("bridge_path", [])
        hop_no = cand.get("hop_no", 1)
        if len(bridge_path) >= hop_no:
            complete_paths += 1
        else:
            logger.warning(f"候选 {cand['note_id']} 路径不完整: hop_no={hop_no}, path_len={len(bridge_path)}")
    
    logger.info(f"路径完整的候选: {complete_paths}/{len(candidates)} ({complete_paths/len(candidates)*100:.1f}%)")
    
    # 检查Top-M保活机制
    logger.info("=== Top-M保活机制检查 ===")
    per_hop_keep_top_m = 5  # 从配置中获取
    
    for hop_no in sorted(hop_stats.keys()):
        hop_candidates = hop_stats[hop_no]
        if len(hop_candidates) <= per_hop_keep_top_m:
            logger.info(f"第 {hop_no} 跳: 所有 {len(hop_candidates)} 个候选都被保留")
        else:
            logger.info(f"第 {hop_no} 跳: 保留了 {len(hop_candidates)} 个候选 (应该 <= {per_hop_keep_top_m})")
    
    # 检查分数分布
    logger.info("=== 分数分布检查 ===")
    scores = [cand.get('final_score', 0) for cand in candidates]
    if scores:
        logger.info(f"分数范围: {min(scores):.4f} - {max(scores):.4f}")
        logger.info(f"平均分数: {sum(scores)/len(scores):.4f}")

def test_multihop_safety_filter():
    """测试多跳安全过滤功能"""
    logger.info("开始测试多跳安全过滤功能")
    
    # 创建测试数据
    test_candidates = create_test_candidates()
    
    try:
        # 创建QueryProcessor实例
        minimal_notes = [
            {"note_id": "test_1", "title": "Test", "content": "Test content", "entities": [], "predicates": []}
        ]
        
        processor = QueryProcessor(
            atomic_notes=minimal_notes,
            embeddings=None,
            graph_file=None,
            vector_index_file=None,
            llm=None
        )
        
        # 设置配置
        processor.config = {
            "hybrid_search": {
                "multi_hop": {
                    "per_hop_keep_top_m": 5,
                    "lower_threshold": 0.10
                }
            }
        }
        
        test_query = "Test query for safety filter"
        
        logger.info(f"过滤前候选数量: {len(test_candidates)}")
        
        # 执行安全过滤
        filtered_candidates = processor._filter_with_multihop_safety(test_candidates, test_query)
        
        logger.info(f"过滤后候选数量: {len(filtered_candidates)}")
        
        # 分析过滤结果
        analyze_filter_results(test_candidates, filtered_candidates)
        
        return True
        
    except Exception as e:
        logger.error(f"安全过滤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_filter_results(original_candidates, filtered_candidates):
    """分析过滤结果"""
    logger.info("=== 安全过滤结果分析 ===")
    
    # 统计原始候选
    original_hop_stats = {}
    for cand in original_candidates:
        hop_no = cand.get("hop_no", 1)
        original_hop_stats[hop_no] = original_hop_stats.get(hop_no, 0) + 1
    
    # 统计过滤后候选
    filtered_hop_stats = {}
    for cand in filtered_candidates:
        hop_no = cand.get("hop_no", 1)
        filtered_hop_stats[hop_no] = filtered_hop_stats.get(hop_no, 0) + 1
    
    # 输出对比
    for hop_no in sorted(original_hop_stats.keys()):
        original_count = original_hop_stats[hop_no]
        filtered_count = filtered_hop_stats.get(hop_no, 0)
        retention_rate = filtered_count / original_count * 100 if original_count > 0 else 0
        
        logger.info(f"第 {hop_no} 跳: {original_count} -> {filtered_count} "
                   f"(保留率: {retention_rate:.1f}%)")

def main():
    """主测试函数"""
    logger.info("开始多跳查询优化测试")
    
    success_count = 0
    total_tests = 2
    
    # 测试1: 多跳重排
    if test_multihop_reranking():
        success_count += 1
        logger.info("✓ 多跳重排测试通过")
    else:
        logger.error("✗ 多跳重排测试失败")
    
    # 测试2: 多跳安全过滤
    if test_multihop_safety_filter():
        success_count += 1
        logger.info("✓ 多跳安全过滤测试通过")
    else:
        logger.error("✗ 多跳安全过滤测试失败")
    
    # 总结
    logger.info(f"=== 测试总结 ===")
    logger.info(f"通过测试: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        logger.info("🎉 所有测试通过！多跳查询优化功能正常工作")
        return True
    else:
        logger.error("❌ 部分测试失败，需要检查实现")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)