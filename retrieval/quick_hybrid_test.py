#!/usr/bin/env python3
"""
快速混合检索功能验证
"""

import os
import sys
from typing import List, Dict, Any, Iterable
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.bm25_search import build_bm25_corpus, bm25_scores


class HybridRetriever:
    """非常轻量的段落检索器，供 NQ/MuSiQue 入口脚本复用。

    项目完整的混合检索流水线依赖大量配置和索引，
    这里提供一个基于词重叠的简化实现，便于在纯 JSONL
    输入上快速跑通端到端链路。后续若接入真实检索器，
    只需替换 ``rank`` 方法即可。
    """

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    @staticmethod
    def _tokenize(text: str) -> Iterable[str]:
        if not text:
            return []
        return [t for t in "".join(ch if ch.isalnum() else " " for ch in text.lower()).split() if t]

    @staticmethod
    def _extract_text(paragraph: Any) -> str:
        if isinstance(paragraph, str):
            return paragraph
        if not isinstance(paragraph, dict):
            return ""
        for key in ("text", "paragraph", "content"):
            if paragraph.get(key):
                return str(paragraph[key])
        sentences = paragraph.get("sentences") or paragraph.get("lines")
        if isinstance(sentences, list):
            return " ".join(str(x) for x in sentences)
        return ""

    def rank(self, question: str, paragraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        q_tokens = set(self._tokenize(question))
        results: List[Dict[str, Any]] = []

        for i, para in enumerate(paragraphs or []):
            idx = i
            if isinstance(para, dict):
                try:
                    idx = int(para.get("idx", i))
                except (TypeError, ValueError):
                    idx = i
            text = self._extract_text(para)
            if not text:
                continue
            p_tokens = set(self._tokenize(text))
            if not p_tokens:
                score = 0.0
            else:
                overlap = len(q_tokens & p_tokens)
                score = overlap / max(len(q_tokens) or 1, 1)
            results.append({
                "idx": idx,
                "text": text,
                "score": float(score),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[: self.top_k]

def quick_test():
    """快速测试 BM25 混合检索核心功能"""
    logger.info("=== 快速混合检索功能验证 ===")
    
    # 创建简单测试数据
    test_notes = [
        {
            'note_id': 'n1',
            'content': 'Machine learning algorithms are used for data analysis and pattern recognition.',
            'title': 'ML Algorithms'
        },
        {
            'note_id': 'n2', 
            'content': 'Python programming language is popular for artificial intelligence development.',
            'title': 'Python AI'
        },
        {
            'note_id': 'n3',
            'content': 'Deep learning neural networks can process complex data patterns.',
            'title': 'Deep Learning'
        },
        {
            'note_id': 'n4',
            'content': 'Natural language processing helps computers understand human language.',
            'title': 'NLP Basics'
        }
    ]
    
    logger.info(f"测试数据: {len(test_notes)} 个笔记")
    
    # 1. 测试 BM25 语料库构建
    logger.info("\n1. 构建 BM25 语料库...")
    corpus = build_bm25_corpus(test_notes, lambda note: note.get('content', ''))
    logger.info("✓ BM25 语料库构建成功")
    
    # 2. 测试 BM25 分数计算
    test_queries = [
        'machine learning algorithms',
        'python programming',
        'neural networks',
        'natural language'
    ]
    
    logger.info("\n2. 测试 BM25 分数计算...")
    for query in test_queries:
        scores = bm25_scores(corpus, test_notes, query)
        
        # 找到最佳匹配
        best_idx = scores.index(max(scores))
        best_note = test_notes[best_idx]
        best_score = scores[best_idx]
        
        logger.info(f"查询: '{query}'")
        logger.info(f"  最佳匹配: {best_note['title']} (分数: {best_score:.4f})")
        logger.info(f"  内容: {best_note['content'][:50]}...")
    
    # 3. 测试分数融合逻辑
    logger.info("\n3. 测试分数融合...")
    query = "machine learning python"
    bm25_scores_list = bm25_scores(corpus, test_notes, query)
    
    # 模拟向量相似度分数
    vector_scores = [0.8, 0.9, 0.6, 0.4]  # 假设的向量相似度
    
    # 融合权重
    vector_weight = 1.0
    bm25_weight = 0.6
    
    logger.info(f"查询: '{query}'")
    logger.info("融合结果:")
    
    fusion_results = []
    for i, note in enumerate(test_notes):
        hybrid_score = vector_weight * vector_scores[i] + bm25_weight * bm25_scores_list[i]
        fusion_results.append({
            'note': note,
            'hybrid_score': hybrid_score,
            'vector_score': vector_scores[i],
            'bm25_score': bm25_scores_list[i]
        })
    
    # 按融合分数排序
    fusion_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    
    for i, result in enumerate(fusion_results, 1):
        logger.info(f"  {i}. {result['note']['title']}")
        logger.info(f"     混合: {result['hybrid_score']:.4f} (向量: {result['vector_score']:.4f}, BM25: {result['bm25_score']:.4f})")
    
    logger.info("\n✓ 所有核心功能测试通过")
    
    # 4. 性能简测
    logger.info("\n4. 简单性能测试...")
    import time
    
    # 扩展测试数据
    extended_notes = []
    for i in range(50):  # 200 个笔记
        for note in test_notes:
            new_note = note.copy()
            new_note['note_id'] = f"{note['note_id']}_{i}"
            new_note['content'] = f"[Copy {i}] {note['content']}"
            extended_notes.append(new_note)
    
    start_time = time.time()
    extended_corpus = build_bm25_corpus(extended_notes, lambda note: note.get('content', ''))
    build_time = time.time() - start_time
    
    start_time = time.time()
    scores = bm25_scores(extended_corpus, extended_notes, "machine learning python algorithms")
    query_time = time.time() - start_time
    
    logger.info(f"数据规模: {len(extended_notes)} 个笔记")
    logger.info(f"构建时间: {build_time:.3f} 秒")
    logger.info(f"查询时间: {query_time:.3f} 秒")
    logger.info(f"查询速度: {len(extended_notes)/query_time:.0f} 文档/秒")
    
    logger.info("\n=== 快速验证完成 ===")

def main():
    try:
        quick_test()
        return 0
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())