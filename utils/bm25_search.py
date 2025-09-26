from typing import List, Dict, Any, Optional, Callable
import math
import re
from collections import Counter, defaultdict
from loguru import logger

try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
    logger.info("Using rank_bm25 library for BM25 implementation")
except ImportError:
    RANK_BM25_AVAILABLE = False
    logger.warning("rank_bm25 not available, using fallback TF-IDF implementation")


class SimpleBM25:
    """Simplified BM25 implementation as fallback when rank_bm25 is not available."""
    
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_count = len(corpus)
        
        # Calculate document frequencies
        for doc in corpus:
            freq = Counter(doc)
            self.doc_freqs.append(freq)
            
        # Calculate IDF for all terms
        all_terms = set()
        for doc in corpus:
            all_terms.update(doc)
            
        for term in all_terms:
            containing_docs = sum(1 for freq in self.doc_freqs if term in freq)
            self.idf[term] = math.log((self.doc_count - containing_docs + 0.5) / (containing_docs + 0.5) + 1.0)
    
    def get_scores(self, query: List[str]) -> List[float]:
        """Calculate BM25 scores for query against all documents."""
        scores = []
        
        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_len[i]
            
            for term in query:
                if term in doc_freq:
                    tf = doc_freq[term]
                    idf = self.idf.get(term, 0)
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    score += idf * (numerator / denominator)
            
            scores.append(score)
        
        return scores


class FieldWeightedBM25:
    """支持字段权重的BM25实现，用于title、entities、content的加权计算"""
    
    def __init__(self, corpus: List[Dict[str, List[str]]], field_weights: Dict[str, float] = None, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            corpus: 文档列表，每个文档是字段到token列表的映射，如 {'title': [...], 'entities': [...], 'content': [...]}
            field_weights: 字段权重，如 {'title': 2.0, 'entities': 1.5, 'content': 1.0}
            k1, b: BM25参数
        """
        self.corpus = corpus
        self.field_weights = field_weights or {'title': 2.0, 'entities': 1.5, 'content': 1.0}
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus)
        
        # 为每个字段计算统计信息
        self.field_stats = {}
        self.field_doc_freqs = {}
        self.field_idf = {}
        
        for field in self.field_weights.keys():
            # 计算字段的文档长度和平均长度
            field_doc_lens = []
            field_doc_freqs = []
            
            for doc in corpus:
                field_tokens = doc.get(field, [])
                field_doc_lens.append(len(field_tokens))
                field_doc_freqs.append(Counter(field_tokens))
            
            avgdl = sum(field_doc_lens) / len(field_doc_lens) if field_doc_lens else 0
            
            self.field_stats[field] = {
                'doc_lens': field_doc_lens,
                'avgdl': avgdl
            }
            self.field_doc_freqs[field] = field_doc_freqs
            
            # 计算字段的IDF
            all_terms = set()
            for doc_freq in field_doc_freqs:
                all_terms.update(doc_freq.keys())
            
            field_idf = {}
            for term in all_terms:
                containing_docs = sum(1 for doc_freq in field_doc_freqs if term in doc_freq)
                field_idf[term] = math.log((self.doc_count - containing_docs + 0.5) / (containing_docs + 0.5) + 1.0)
            
            self.field_idf[field] = field_idf
    
    def get_scores(self, query: List[str]) -> List[float]:
        """计算查询对所有文档的BM25分数"""
        scores = []
        
        for doc_idx in range(self.doc_count):
            total_score = 0.0
            
            # 对每个字段计算BM25分数并加权
            for field, weight in self.field_weights.items():
                field_score = 0.0
                doc_freq = self.field_doc_freqs[field][doc_idx]
                doc_len = self.field_stats[field]['doc_lens'][doc_idx]
                avgdl = self.field_stats[field]['avgdl']
                
                for term in query:
                    if term in doc_freq:
                        tf = doc_freq[term]
                        idf = self.field_idf[field].get(term, 0)
                        
                        # BM25公式
                        numerator = tf * (self.k1 + 1)
                        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avgdl)) if avgdl > 0 else tf + self.k1
                        field_score += idf * (numerator / denominator)
                
                # 应用字段权重
                total_score += weight * field_score
            
            scores.append(total_score)
        
        return scores


def build_field_weighted_bm25_corpus(notes: List[Dict[str, Any]], field_weights: Dict[str, float] = None) -> FieldWeightedBM25:
    """
    构建支持字段权重的BM25语料库
    
    Args:
        notes: 笔记列表
        field_weights: 字段权重，默认为 {'title': 2.0, 'entities': 1.5, 'content': 1.0}
    
    Returns:
        FieldWeightedBM25对象
    """
    if field_weights is None:
        field_weights = {'title': 2.0, 'entities': 1.5, 'content': 1.0}
    
    # 构建字段化的语料库
    field_corpus = []
    
    for note in notes:
        doc_fields = {}
        
        # 提取title字段
        title = note.get('title', '') or ''
        doc_fields['title'] = tokenize_text(title)
        
        # 提取entities字段
        entities = note.get('entities', []) or []
        if isinstance(entities, list):
            entities_text = ' '.join(entities)
        else:
            entities_text = str(entities)
        doc_fields['entities'] = tokenize_text(entities_text)
        
        # 提取content字段
        content = note.get('content', '') or ''
        doc_fields['content'] = tokenize_text(content)
        
        field_corpus.append(doc_fields)
    
    corpus = FieldWeightedBM25(field_corpus, field_weights)
    logger.debug(f"Built field-weighted BM25 corpus with {len(field_corpus)} documents, weights: {field_weights}")
    return corpus


def field_weighted_bm25_scores(corpus: FieldWeightedBM25, docs: List[Dict[str, Any]], query: str) -> List[float]:
    """
    使用字段权重BM25计算相关性分数
    
    Args:
        corpus: FieldWeightedBM25语料库对象
        docs: 文档列表（用于验证长度匹配）
        query: 查询字符串
    
    Returns:
        BM25分数列表
    """
    try:
        # 分词查询
        query_tokens = tokenize_text(query)
        
        if not query_tokens:
            logger.warning("Empty query tokens, returning zero scores")
            return [0.0] * len(docs)
        
        # 获取分数
        scores = corpus.get_scores(query_tokens)
        
        # 确保分数长度匹配文档长度
        if len(scores) != len(docs):
            logger.warning(f"Score length mismatch: {len(scores)} scores vs {len(docs)} docs")
            if len(scores) < len(docs):
                scores.extend([0.0] * (len(docs) - len(scores)))
            else:
                scores = scores[:len(docs)]
        
        # 归一化分数到[0, 1]范围
        if scores:
            max_score = max(scores)
            if max_score > 0:
                scores = [score / max_score for score in scores]
        
        logger.debug(f"Calculated field-weighted BM25 scores for {len(docs)} documents, max score: {max(scores) if scores else 0}")
        return scores
        
    except Exception as e:
        logger.error(f"Error calculating field-weighted BM25 scores: {e}")
        return [0.0] * len(docs)


def tokenize_text(text: str) -> List[str]:
    """Simple tokenization function."""
    # Convert to lowercase and split on non-alphanumeric characters
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def build_bm25_corpus(notes: List[Dict[str, Any]], text_fn: Callable[[Dict[str, Any]], str]) -> Any:
    """
    构建 BM25 语料库
    
    Args:
        notes: 笔记列表
        text_fn: 从笔记中提取文本的函数
    
    Returns:
        BM25 语料库对象
    """
    # Extract and tokenize text from notes
    tokenized_corpus = []
    
    for note in notes:
        try:
            text = text_fn(note)
            if text:
                tokens = tokenize_text(text)
                tokenized_corpus.append(tokens)
            else:
                tokenized_corpus.append([])
        except Exception as e:
            logger.warning(f"Error extracting text from note: {e}")
            tokenized_corpus.append([])
    
    # Build BM25 index
    if RANK_BM25_AVAILABLE:
        try:
            corpus = BM25Okapi(tokenized_corpus)
            logger.debug(f"Built BM25 corpus with {len(tokenized_corpus)} documents using rank_bm25")
            return corpus
        except Exception as e:
            logger.error(f"Error building BM25 corpus with rank_bm25: {e}")
            # Fall back to simple implementation
    
    # Use fallback implementation
    corpus = SimpleBM25(tokenized_corpus)
    logger.debug(f"Built BM25 corpus with {len(tokenized_corpus)} documents using fallback implementation")
    return corpus


def bm25_scores(corpus: Any, docs: List[Dict[str, Any]], query: str) -> List[float]:
    """
    计算 BM25 相关性分数
    
    Args:
        corpus: BM25 语料库对象
        docs: 文档列表（用于验证长度匹配）
        query: 查询字符串
    
    Returns:
        BM25 分数列表
    """
    try:
        # Tokenize query
        query_tokens = tokenize_text(query)
        
        if not query_tokens:
            logger.warning("Empty query tokens, returning zero scores")
            return [0.0] * len(docs)
        
        # Get scores from corpus
        if RANK_BM25_AVAILABLE and hasattr(corpus, 'get_scores'):
            # Using rank_bm25
            scores = corpus.get_scores(query_tokens)
            # Convert numpy array to list if needed
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
        elif hasattr(corpus, 'get_scores'):
            # Using our fallback implementation
            scores = corpus.get_scores(query_tokens)
        else:
            logger.error("Invalid corpus object")
            return [0.0] * len(docs)
        
        # Ensure scores length matches docs length
        if len(scores) != len(docs):
            logger.warning(f"Score length mismatch: {len(scores)} scores vs {len(docs)} docs")
            # Pad or truncate as needed
            if len(scores) < len(docs):
                scores.extend([0.0] * (len(docs) - len(scores)))
            else:
                scores = scores[:len(docs)]
        
        # Normalize scores to [0, 1] range for better fusion
        if scores:
            max_score = max(scores)
            if max_score > 0:
                scores = [score / max_score for score in scores]
        
        logger.debug(f"Calculated BM25 scores for {len(docs)} documents, max score: {max(scores) if scores else 0}")
        return scores
        
    except Exception as e:
        logger.error(f"Error calculating BM25 scores: {e}")
        return [0.0] * len(docs)


def test_bm25_implementation():
    """Test function for BM25 implementation."""
    # Sample notes for testing
    test_notes = [
        {'content': 'The quick brown fox jumps over the lazy dog'},
        {'content': 'Machine learning is a subset of artificial intelligence'},
        {'content': 'Python is a popular programming language for data science'},
        {'content': 'Natural language processing involves understanding human language'}
    ]
    
    # Text extraction function
    def extract_content(note):
        return note.get('content', '')
    
    # Build corpus
    corpus = build_bm25_corpus(test_notes, extract_content)
    
    # Test queries
    test_queries = [
        'machine learning',
        'python programming',
        'natural language',
        'quick fox'
    ]
    
    for query in test_queries:
        scores = bm25_scores(corpus, test_notes, query)
        print(f"Query: '{query}'")
        for i, (note, score) in enumerate(zip(test_notes, scores)):
            print(f"  Doc {i}: {score:.4f} - {note['content'][:50]}...")
        print()


if __name__ == "__main__":
    test_bm25_implementation()