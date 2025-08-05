from typing import List, Dict, Any, Optional
import re
import numpy as np
from loguru import logger
from collections import defaultdict

class EnhancedNoiseFilter:
    """增强的去噪机制，使用复合评分系统判定笔记质量"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 默认配置
        self.config = config or {}
        
        # 评分权重
        self.weights = {
            'importance_score': self.config.get('weights.importance_score', 0.4),
            'content_length_score': self.config.get('weights.content_length_score', 0.3),
            'verified_entity_ratio': self.config.get('weights.verified_entity_ratio', 0.3)
        }
        
        # 阈值设置
        self.usefulness_threshold = self.config.get('usefulness_threshold', 0.65)
        self.min_content_length = self.config.get('min_content_length', 20)
        self.max_content_length_for_score = self.config.get('max_content_length_for_score', 100)
        
        # 质量指标权重
        self.quality_weights = {
            'content_completeness': 0.25,
            'entity_relevance': 0.25,
            'information_density': 0.20,
            'linguistic_quality': 0.15,
            'factual_consistency': 0.15
        }
        
        # 噪声模式识别
        self.noise_patterns = [
            r'^\s*$',  # 空内容
            r'^\s*\.\.\.$',  # 只有省略号
            r'^\s*[^a-zA-Z0-9\u4e00-\u9fff]*$',  # 只有标点符号
            r'^\s*(?:the|a|an|and|or|but)\s*$',  # 只有停用词
            r'^\s*\d+\s*$',  # 只有数字
        ]
        
        # 高质量内容指示词
        self.quality_indicators = {
            'factual': ['born', 'died', 'created', 'founded', 'established', 'married', 'divorced'],
            'descriptive': ['known for', 'famous for', 'characterized by', 'described as'],
            'relational': ['son of', 'daughter of', 'married to', 'worked with', 'collaborated'],
            'temporal': ['in', 'during', 'before', 'after', 'since', 'until'],
            'quantitative': ['first', 'last', 'most', 'least', 'many', 'few', 'several']
        }
    
    def calculate_usefulness_score(self, note: Dict[str, Any]) -> float:
        """计算笔记的有用性评分"""
        # 获取基础信息
        content = note.get('content', '')
        entities = note.get('entities', [])
        importance_score = note.get('importance_score', 0.5)
        
        # 1. 重要性评分（原有分数）
        w1 = self.weights['importance_score']
        score1 = float(importance_score)
        
        # 2. 内容长度评分
        w2 = self.weights['content_length_score']
        content_length = len(content)
        score2 = min(content_length / self.max_content_length_for_score, 1.0)
        
        # 3. 验证实体比例
        w3 = self.weights['verified_entity_ratio']
        verified_entities = self._count_verified_entities(content, entities)
        total_entities = len(entities) if entities else 1
        score3 = verified_entities / max(1, total_entities)
        
        # 计算复合评分
        usefulness_score = w1 * score1 + w2 * score2 + w3 * score3
        
        return min(max(usefulness_score, 0.0), 1.0)
    
    def assess_note_quality(self, note: Dict[str, Any]) -> Dict[str, Any]:
        """全面评估笔记质量"""
        content = note.get('content', '')
        entities = note.get('entities', [])
        
        quality_scores = {}
        
        # 1. 内容完整性
        quality_scores['content_completeness'] = self._assess_content_completeness(content)
        
        # 2. 实体相关性
        quality_scores['entity_relevance'] = self._assess_entity_relevance(content, entities)
        
        # 3. 信息密度
        quality_scores['information_density'] = self._assess_information_density(content)
        
        # 4. 语言质量
        quality_scores['linguistic_quality'] = self._assess_linguistic_quality(content)
        
        # 5. 事实一致性
        quality_scores['factual_consistency'] = self._assess_factual_consistency(content)
        
        # 计算综合质量分数
        overall_quality = sum(
            self.quality_weights[metric] * score 
            for metric, score in quality_scores.items()
        )
        
        return {
            'quality_scores': quality_scores,
            'overall_quality': overall_quality,
            'quality_level': self._get_quality_level(overall_quality)
        }
    
    def filter_noise_notes(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤噪声笔记，使用增强的评分机制"""
        logger.info(f"Filtering noise from {len(atomic_notes)} atomic notes")
        
        filtered_notes = []
        noise_stats = {
            'total_notes': len(atomic_notes),
            'noise_notes': 0,
            'quality_notes': 0,
            'forced_quality': 0,  # 强制设为非噪声的数量
            'noise_reasons': defaultdict(int)
        }
        
        for note in atomic_notes:
            # 计算有用性评分
            usefulness_score = self.calculate_usefulness_score(note)
            
            # 评估质量
            quality_assessment = self.assess_note_quality(note)
            
            # 检查是否为明显噪声
            is_obvious_noise = self._is_obvious_noise(note)
            
            # 更新笔记信息
            enhanced_note = note.copy()
            enhanced_note['usefulness_score'] = usefulness_score
            enhanced_note['quality_assessment'] = quality_assessment
            
            # 判定是否为噪声
            if is_obvious_noise:
                enhanced_note['is_noise'] = True
                enhanced_note['noise_reason'] = 'obvious_noise'
                noise_stats['noise_notes'] += 1
                noise_stats['noise_reasons']['obvious_noise'] += 1
            elif usefulness_score > self.usefulness_threshold:
                # 强制设为非噪声
                enhanced_note['is_noise'] = False
                enhanced_note['noise_reason'] = None
                noise_stats['quality_notes'] += 1
                if note.get('is_noise', False):  # 原来被标记为噪声
                    noise_stats['forced_quality'] += 1
            else:
                # 基于质量评估决定
                overall_quality = quality_assessment['overall_quality']
                if overall_quality < 0.4:
                    enhanced_note['is_noise'] = True
                    enhanced_note['noise_reason'] = 'low_quality'
                    noise_stats['noise_notes'] += 1
                    noise_stats['noise_reasons']['low_quality'] += 1
                else:
                    enhanced_note['is_noise'] = False
                    enhanced_note['noise_reason'] = None
                    noise_stats['quality_notes'] += 1
            
            filtered_notes.append(enhanced_note)
        
        # 记录统计信息
        logger.info(f"Noise filtering completed: {noise_stats['quality_notes']} quality notes, "
                   f"{noise_stats['noise_notes']} noise notes, {noise_stats['forced_quality']} forced quality")
        
        return filtered_notes
    
    def _count_verified_entities(self, content: str, entities: List[str]) -> int:
        """计算在内容中得到验证的实体数量"""
        if not entities:
            return 0
        
        verified_count = 0
        content_lower = content.lower()
        
        for entity in entities:
            if entity and entity.lower() in content_lower:
                verified_count += 1
        
        return verified_count
    
    def _assess_content_completeness(self, content: str) -> float:
        """评估内容完整性"""
        if not content or len(content.strip()) < self.min_content_length:
            return 0.0
        
        # 检查句子完整性
        sentences = re.split(r'[.!?]+', content)
        complete_sentences = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.endswith('...'):
                complete_sentences += 1
        
        if not sentences:
            return 0.0
        
        completeness_ratio = complete_sentences / len(sentences)
        
        # 长度奖励
        length_bonus = min(len(content) / 200, 1.0)
        
        return (completeness_ratio + length_bonus) / 2
    
    def _assess_entity_relevance(self, content: str, entities: List[str]) -> float:
        """评估实体相关性"""
        if not entities:
            return 0.5  # 中性分数
        
        content_lower = content.lower()
        relevant_entities = 0
        
        for entity in entities:
            if entity and entity.lower() in content_lower:
                # 检查实体在内容中的上下文
                entity_contexts = self._get_entity_contexts(content, entity)
                if any(self._is_meaningful_context(ctx) for ctx in entity_contexts):
                    relevant_entities += 1
        
        return relevant_entities / len(entities)
    
    def _assess_information_density(self, content: str) -> float:
        """评估信息密度"""
        if not content:
            return 0.0
        
        words = content.split()
        if len(words) < 5:
            return 0.0
        
        # 计算信息词汇比例
        info_words = 0
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        for word in words:
            word_clean = re.sub(r'[^a-zA-Z]', '', word.lower())
            if word_clean and word_clean not in stop_words and len(word_clean) > 2:
                info_words += 1
        
        density = info_words / len(words)
        
        # 检查质量指示词
        quality_bonus = 0
        for category, indicators in self.quality_indicators.items():
            for indicator in indicators:
                if indicator in content.lower():
                    quality_bonus += 0.1
                    break
        
        return min(density + quality_bonus, 1.0)
    
    def _assess_linguistic_quality(self, content: str) -> float:
        """评估语言质量"""
        if not content:
            return 0.0
        
        score = 0.5  # 基础分数
        
        # 检查语法结构
        if re.search(r'[A-Z][^.!?]*[.!?]', content):  # 包含完整句子
            score += 0.2
        
        # 检查标点符号使用
        if re.search(r'[.!?]', content):  # 包含句号等
            score += 0.1
        
        # 检查大小写使用
        if re.search(r'[A-Z]', content) and re.search(r'[a-z]', content):  # 包含大小写
            score += 0.1
        
        # 检查词汇多样性
        words = re.findall(r'\b\w+\b', content.lower())
        if words:
            unique_words = len(set(words))
            diversity = unique_words / len(words)
            score += diversity * 0.1
        
        return min(score, 1.0)
    
    def _assess_factual_consistency(self, content: str) -> float:
        """评估事实一致性"""
        if not content:
            return 0.0
        
        # 基于内容质量评估一致性
        # 检查内容中是否有关键实体和数字
        content_entities = set(re.findall(r'\b[A-Z][a-z]+\b', content))
        content_numbers = set(re.findall(r'\b\d+\b', content))
        
        # 基于内容的丰富程度评估
        entity_score = min(len(content_entities) / 5, 1.0)  # 假设5个实体为满分
        number_score = min(len(content_numbers) / 3, 1.0)   # 假设3个数字为满分
        
        return (entity_score + number_score) / 2
    
    def _is_obvious_noise(self, note: Dict[str, Any]) -> bool:
        """检查是否为明显的噪声"""
        content = note.get('content', '')
        
        # 检查噪声模式
        for pattern in self.noise_patterns:
            if re.match(pattern, content, re.IGNORECASE):
                return True
        
        # 检查内容长度
        if len(content.strip()) < self.min_content_length:
            return True
        
        # 检查是否只包含重复字符
        if len(set(content.replace(' ', ''))) < 3:
            return True
        
        return False
    
    def _get_entity_contexts(self, content: str, entity: str, window: int = 20) -> List[str]:
        """获取实体在内容中的上下文"""
        contexts = []
        entity_lower = entity.lower()
        content_lower = content.lower()
        
        start = 0
        while True:
            pos = content_lower.find(entity_lower, start)
            if pos == -1:
                break
            
            # 提取上下文
            context_start = max(0, pos - window)
            context_end = min(len(content), pos + len(entity) + window)
            context = content[context_start:context_end]
            contexts.append(context)
            
            start = pos + 1
        
        return contexts
    
    def _is_meaningful_context(self, context: str) -> bool:
        """判断上下文是否有意义"""
        # 检查是否包含动词或描述性词汇
        meaningful_words = {
            'is', 'was', 'are', 'were', 'has', 'had', 'have', 'does', 'did', 'do',
            'created', 'made', 'born', 'died', 'married', 'worked', 'lived', 'known',
            'famous', 'voice', 'actor', 'character', 'role', 'played', 'portrayed'
        }
        
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        return bool(context_words.intersection(meaningful_words))
    
    def _get_quality_level(self, score: float) -> str:
        """根据分数获取质量等级"""
        if score >= 0.8:
            return 'high'
        elif score >= 0.6:
            return 'medium'
        elif score >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def get_filtering_statistics(self, original_notes: List[Dict[str, Any]], 
                               filtered_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取过滤统计信息"""
        original_noise = sum(1 for note in original_notes if note.get('is_noise', False))
        filtered_noise = sum(1 for note in filtered_notes if note.get('is_noise', False))
        
        # 统计质量分布
        quality_distribution = defaultdict(int)
        usefulness_scores = []
        
        for note in filtered_notes:
            if 'quality_assessment' in note:
                level = note['quality_assessment']['quality_level']
                quality_distribution[level] += 1
            
            if 'usefulness_score' in note:
                usefulness_scores.append(note['usefulness_score'])
        
        stats = {
            'total_notes': len(original_notes),
            'original_noise_count': original_noise,
            'filtered_noise_count': filtered_noise,
            'noise_reduction': original_noise - filtered_noise,
            'quality_distribution': dict(quality_distribution),
            'average_usefulness_score': np.mean(usefulness_scores) if usefulness_scores else 0.0,
            'usefulness_score_std': np.std(usefulness_scores) if usefulness_scores else 0.0
        }
        
        return stats