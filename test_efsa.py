#!/usr/bin/env python3
"""
EFSA (Entity-Focused Score Aggregation) 测试脚本
测试实体聚合答案生成功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from answer.efsa_answer import efsa_answer, efsa_answer_with_fallback, compute_cov_cons

def test_compute_cov_cons():
    """测试覆盖率和一致性计算"""
    print("=== 测试 compute_cov_cons 函数 ===")
    
    # 模拟笔记数据
    note = {
        'entities': ['Steve Hillage', 'Miquette Giraudy', 'System 7'],
        'title': 'Steve Hillage discography',
        'content': 'Steve Hillage worked with Miquette Giraudy on various projects including System 7.'
    }
    
    path_entities = ['Steve Hillage', 'System 7']
    
    cov, cons = compute_cov_cons(note, path_entities)
    print(f"Coverage: {cov:.3f}, Consistency: {cons}")
    print(f"Note entities: {note['entities']}")
    print(f"Path entities: {path_entities}")
    print()

def test_efsa_answer():
    """测试EFSA答案生成"""
    print("=== 测试 EFSA 答案生成 ===")
    
    # 模拟候选笔记数据
    candidates = [
        {
            'entities': ['Steve Hillage', 'Miquette Giraudy', 'System 7'],
            'final_score': 0.85,
            'hop_no': 1,
            'doc_id': 'doc1',
            'paragraph_idxs': [1],
            'title': 'Steve Hillage biography',
            'content': 'Steve Hillage is a guitarist who collaborated with Miquette Giraudy.'
        },
        {
            'entities': ['Miquette Giraudy', 'System 7', 'electronic music'],
            'final_score': 0.78,
            'hop_no': 2,
            'doc_id': 'doc2',
            'paragraph_idxs': [2],
            'title': 'Miquette Giraudy profile',
            'content': 'Miquette Giraudy is known for her work in electronic music and System 7.'
        },
        {
            'entities': ['Steve Hillage', 'guitar', 'progressive rock'],
            'final_score': 0.72,
            'hop_no': 1,
            'doc_id': 'doc3',
            'paragraph_idxs': [3],
            'title': 'Progressive rock artists',
            'content': 'Steve Hillage is a prominent figure in progressive rock music.'
        }
    ]
    
    query = "Who is Steve Hillage's partner?"
    bridge_entity = "Steve Hillage"
    path_entities = ["Steve Hillage", "System 7"]
    
    print(f"Query: {query}")
    print(f"Bridge entity: {bridge_entity}")
    print(f"Path entities: {path_entities}")
    print(f"Candidates: {len(candidates)} notes")
    print()
    
    # 测试EFSA答案生成
    answer, support_idxs = efsa_answer(
        candidates=candidates,
        query=query,
        bridge_entity=bridge_entity,
        path_entities=path_entities,
        topN=20
    )
    
    print(f"EFSA Answer: {answer}")
    print(f"Support indices: {support_idxs}")
    print()
    
    # 测试带回退的EFSA
    print("=== 测试带回退的 EFSA ===")
    answer_fb, support_fb = efsa_answer_with_fallback(
        candidates=candidates,
        query=query,
        bridge_entity=bridge_entity,
        path_entities=path_entities,
        topN=20
    )
    
    print(f"EFSA with fallback Answer: {answer_fb}")
    print(f"Support indices: {support_fb}")
    print()

def test_edge_cases():
    """测试边界情况"""
    print("=== 测试边界情况 ===")
    
    # 空候选列表
    print("1. 空候选列表:")
    answer, support = efsa_answer([], "test query")
    print(f"   Answer: {answer}, Support: {support}")
    
    # 没有实体的候选
    print("2. 没有实体的候选:")
    candidates_no_entities = [
        {
            'entities': [],
            'final_score': 0.5,
            'hop_no': 1,
            'doc_id': 'doc1',
            'paragraph_idxs': [1]
        }
    ]
    answer, support = efsa_answer(candidates_no_entities, "test query")
    print(f"   Answer: {answer}, Support: {support}")
    
    # 所有实体都是桥接实体
    print("3. 所有实体都是桥接实体:")
    candidates_all_bridge = [
        {
            'entities': ['Steve Hillage'],
            'final_score': 0.8,
            'hop_no': 1,
            'doc_id': 'doc1',
            'paragraph_idxs': [1]
        }
    ]
    answer, support = efsa_answer(
        candidates_all_bridge, 
        "test query", 
        bridge_entity="Steve Hillage"
    )
    print(f"   Answer: {answer}, Support: {support}")
    print()

if __name__ == "__main__":
    print("EFSA (Entity-Focused Score Aggregation) 测试")
    print("=" * 50)
    
    try:
        test_compute_cov_cons()
        test_efsa_answer()
        test_edge_cases()
        
        print("✅ 所有测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()