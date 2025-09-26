#!/usr/bin/env python3
"""
EFSA集成测试脚本
测试EFSA在完整查询处理流程中的集成效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from typing import Dict, Any, List

def create_mock_query_processor():
    """创建模拟的QueryProcessor用于测试"""
    
    class MockOllama:
        def generate_final_answer(self, prompt):
            return '{"answer": "Mock LLM Answer", "support_idxs": [1, 2]}'
        
        def evaluate_answer(self, query, context, answer):
            return {"relevance": 0.8, "confidence": 0.7}
    
    class MockQueryProcessor:
        def __init__(self):
            self.ollama = MockOllama()
        
        def _create_test_candidates(self) -> List[Dict[str, Any]]:
            """创建测试用的候选笔记"""
            return [
                {
                    'note_id': 'note1',
                    'entities': ['Steve Hillage', 'Miquette Giraudy', 'System 7'],
                    'final_score': 0.85,
                    'hop_no': 1,
                    'doc_id': 'doc1',
                    'paragraph_idxs': [1],
                    'title': 'Steve Hillage biography',
                    'content': 'Steve Hillage is a guitarist who collaborated with Miquette Giraudy on System 7.',
                    'bridge_entity': None,
                    'bridge_path': []
                },
                {
                    'note_id': 'note2', 
                    'entities': ['Miquette Giraudy', 'System 7', 'electronic music'],
                    'final_score': 0.78,
                    'hop_no': 2,
                    'doc_id': 'doc2',
                    'paragraph_idxs': [2],
                    'title': 'Miquette Giraudy profile',
                    'content': 'Miquette Giraudy is known for her work in electronic music and System 7.',
                    'bridge_entity': 'Steve Hillage',
                    'bridge_path': ['Steve Hillage', 'System 7']
                },
                {
                    'note_id': 'note3',
                    'entities': ['Steve Hillage', 'guitar', 'progressive rock'],
                    'final_score': 0.72,
                    'hop_no': 1,
                    'doc_id': 'doc3',
                    'paragraph_idxs': [3],
                    'title': 'Progressive rock artists',
                    'content': 'Steve Hillage is a prominent figure in progressive rock music.',
                    'bridge_entity': None,
                    'bridge_path': []
                },
                {
                    'note_id': 'note4',
                    'entities': ['Miquette Giraudy', 'partner', 'collaboration'],
                    'final_score': 0.90,
                    'hop_no': 2,
                    'doc_id': 'doc4',
                    'paragraph_idxs': [4],
                    'title': 'Musical partnerships',
                    'content': 'Miquette Giraudy has been Steve Hillage\'s long-time partner and collaborator.',
                    'bridge_entity': 'Steve Hillage',
                    'bridge_path': ['Steve Hillage', 'collaboration']
                }
            ]
        
        def test_efsa_integration(self, query: str) -> Dict[str, Any]:
            """测试EFSA集成的答案生成流程"""
            print(f"Processing query: {query}")
            
            # 模拟候选笔记选择过程
            selected_notes = self._create_test_candidates()
            print(f"Selected {len(selected_notes)} candidate notes")
            
            # EFSA实体聚合答案生成（模拟集成后的逻辑）
            from answer.efsa_answer import efsa_answer_with_fallback
            
            # 提取桥接实体和路径实体信息
            bridge_entities = []
            path_entities = []
            for note in selected_notes:
                if note.get('bridge_entity'):
                    bridge_entities.append(note['bridge_entity'])
                if note.get('bridge_path'):
                    path_entities.extend(note['bridge_path'])
            
            # 去重并取最后几个路径实体
            bridge_entities = list(set(bridge_entities))
            path_entities = list(set(path_entities))[-2:] if path_entities else []
            
            print(f"Bridge entities: {bridge_entities}")
            print(f"Path entities: {path_entities}")
            
            # 尝试EFSA实体答案生成
            efsa_answer, efsa_support_idxs = efsa_answer_with_fallback(
                candidates=selected_notes,
                query=query,
                bridge_entity=bridge_entities[0] if bridge_entities else None,
                path_entities=path_entities,
                topN=20
            )
            
            if efsa_answer:
                # EFSA成功生成实体答案
                print(f"✅ EFSA generated entity answer: {efsa_answer}")
                answer = efsa_answer
                predicted_support_idxs = efsa_support_idxs
                
                # 为了保持一致性，仍然生成评分
                context = "\n".join(n.get('content','') for n in selected_notes)
                scores = self.ollama.evaluate_answer(query, context, answer)
                
                return {
                    'query': query,
                    'answer': answer,
                    'predicted_support_idxs': predicted_support_idxs,
                    'scores': scores,
                    'notes': selected_notes,
                    'method': 'EFSA',
                    'bridge_entities': bridge_entities,
                    'path_entities': path_entities
                }
            else:
                # EFSA未找到实体答案，回退到原有的LLM句子型答案生成
                print("⚠️  EFSA did not find entity answer, falling back to LLM-based answer generation")
                
                # 模拟LLM答案生成
                answer = "Mock LLM generated answer"
                predicted_support_idxs = [1, 2]
                
                context = "\n".join(n.get('content','') for n in selected_notes)
                scores = self.ollama.evaluate_answer(query, context, answer)
                
                return {
                    'query': query,
                    'answer': answer,
                    'predicted_support_idxs': predicted_support_idxs,
                    'scores': scores,
                    'notes': selected_notes,
                    'method': 'LLM_Fallback',
                    'bridge_entities': bridge_entities,
                    'path_entities': path_entities
                }
    
    return MockQueryProcessor()

def test_various_queries():
    """测试不同类型的查询"""
    processor = create_mock_query_processor()
    
    test_queries = [
        "Who is Steve Hillage's partner?",
        "What is System 7?", 
        "Who collaborated with Steve Hillage?",
        "What genre is Steve Hillage associated with?"
    ]
    
    results = []
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: {query}")
        print('='*60)
        
        try:
            result = processor.test_efsa_integration(query)
            results.append(result)
            
            print(f"\n📊 Result Summary:")
            print(f"   Method: {result['method']}")
            print(f"   Answer: {result['answer']}")
            print(f"   Support indices: {result['predicted_support_idxs']}")
            print(f"   Scores: {result['scores']}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def analyze_results(results: List[Dict[str, Any]]):
    """分析测试结果"""
    print(f"\n{'='*60}")
    print("📈 EFSA集成测试结果分析")
    print('='*60)
    
    efsa_count = sum(1 for r in results if r['method'] == 'EFSA')
    llm_count = sum(1 for r in results if r['method'] == 'LLM_Fallback')
    
    print(f"总查询数: {len(results)}")
    print(f"EFSA成功: {efsa_count} ({efsa_count/len(results)*100:.1f}%)")
    print(f"LLM回退: {llm_count} ({llm_count/len(results)*100:.1f}%)")
    
    print(f"\n📋 详细结果:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Query: {result['query']}")
        print(f"   Method: {result['method']}")
        print(f"   Answer: {result['answer']}")
        print(f"   Relevance: {result['scores'].get('relevance', 'N/A')}")
        print()

if __name__ == "__main__":
    print("EFSA集成测试")
    print("=" * 60)
    
    try:
        # 运行测试
        results = test_various_queries()
        
        # 分析结果
        analyze_results(results)
        
        print("✅ 集成测试完成!")
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()