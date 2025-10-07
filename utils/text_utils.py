import re
import string
from typing import List, Dict, Any
from loguru import logger

class TextUtils:
    """文本处理工具类"""
    
    @staticmethod
    def split_by_sentence(text: str) -> List[str]:
        """按句号分割文本，避免直接截断"""
        # 使用正则表达式分割句子，保留句号
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        # 过滤空字符串
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """将文本分块，先按句子分割再合并到指定大小"""
        sentences = TextUtils.split_by_sentence(text)
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # 如果当前块加上新句子超过chunk_size，保存当前块
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'length': current_length
                })
                
                # 处理重叠
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + " " + sentence
                    current_length = len(overlap_text) + sentence_length + 1
                else:
                    current_chunk = sentence
                    current_length = sentence_length
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                    current_length += sentence_length + 1
                else:
                    current_chunk = sentence
                    current_length = sentence_length
        
        # 添加最后一个块
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'length': current_length
            })
        
        return chunks
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本，去除多余的空白字符"""
        # 去除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 去除首尾空白
        text = text.strip()
        return text
    
    STOP_WORDS = {"The", "Created", "After"}

    @staticmethod
    def _estimate_confidence(entity: str) -> float:
        """基于长度的简单置信度估计"""
        letters = re.sub(r'[^A-Za-z]', '', entity)
        if not letters:
            return 0.0
        return min(1.0, len(letters) / 10)

    @staticmethod
    def extract_entities(
        text: str,
        *,
        confidence_threshold: float = 0.5,
        stop_words: List[str] | None = None,
    ) -> List[str]:
        """简单的实体提取（可以后续用NER模型替换）"""
        if stop_words is None:
            stop_words = list(TextUtils.STOP_WORDS)

        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)

        filtered = []
        for ent in set(entities):
            words = ent.split()
            # 移除位于开头或结尾的停用词
            while words and words[0] in stop_words:
                words = words[1:]
            while words and words[-1] in stop_words:
                words = words[:-1]
            if not words:
                continue
            if any(w in stop_words for w in words):
                continue
            ent_clean = " ".join(words)
            confidence = TextUtils._estimate_confidence(ent_clean)
            if confidence >= confidence_threshold:
                filtered.append(ent_clean)

        return filtered

    @staticmethod
    def extract_entities_fallback(
        text: str,
        *,
        min_len: int = 2,
        allow_types: List[str] | None = None,
    ) -> List[str]:
        """轻量级实体回补，基于启发式正则匹配"""
        if not text:
            return []

        # 使用现有的英文专有名词启发式
        candidates = TextUtils.extract_entities(text, confidence_threshold=0.2)

        seen = set()
        entities: List[str] = []
        for candidate in candidates:
            normalized = candidate.strip()
            if len(normalized) < max(1, min_len):
                continue
            if normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            entities.append(normalized)

        # 对中文大写字母不敏感，补充对专名的简单匹配
        chinese_candidates = re.findall(r'[\u4e00-\u9fff]{%d,}' % max(1, min_len), text)
        for candidate in chinese_candidates:
            normalized = candidate.strip()
            key = normalized.lower()
            if key not in seen:
                seen.add(key)
                entities.append(normalized)

        return entities
    
    @staticmethod
    def calculate_similarity_keywords(text1: str, text2: str) -> float:
        """计算两个文本的关键词相似度"""
        # 简单的关键词提取和相似度计算
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        # 去除停用词（简化版）
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """标准化文本"""
        # 转换为小写
        text = text.lower()
        # 去除标点符号
        text = text.translate(str.maketrans('', '', string.punctuation))
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        return text