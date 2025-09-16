from typing import List, Dict, Any, Set, Optional, Tuple
from loguru import logger
import re
import yaml
import os
import hashlib
import difflib
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NoteValidator:
    """原子笔记验证器，用于实体唯一性校验和源文档实体回溯约束"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        # P2-8: 优先从传入的config dict读取配置，而非文件路径
        if config is not None:
            # 使用传入的配置字典
            self.config = config.get('note_validator', {})
            logger.info("NoteValidator: Using provided config dict")
        else:
            # 回退到从文件路径加载配置
            self.config = self._load_config(config_path)
            logger.info("NoteValidator: Loaded config from file path")
        
        # 从配置加载人名模式
        self.person_name_patterns = self.config.get('person_name_patterns', [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # 标准英文姓名
            r'\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b',  # 中间名缩写
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # 三个词的姓名
        ])
        
        # 从配置加载已知的演员和角色映射
        self.known_voice_actors = self.config.get('known_voice_actors', {
            'Dan Castellaneta': ['Krusty the Clown', 'Homer Simpson'],
            'Adriana Caselotti': ['Snow White'],
            'Florian St. Pierre': []
        })
        
        # 加载验证配置
        self.entity_uniqueness_config = self.config.get('entity_uniqueness', {
            'enabled': True,
            'auto_fix': True,
            'selection_strategy': 'known_actors'
        })
        
        self.source_traceability_config = self.config.get('source_traceability', {
            'enabled': True,
            'strict_mode': False,
            'similarity_threshold': 0.8
        })
        
        # 加载人名指示词
        self.person_indicators = self.config.get('person_indicators', ['actor', 'actress', 'voice', 'performer', 'artist'])
        
        # P2-8: 组件初始化后打印结构化日志，公布生效参数
        self._log_effective_config()
    
    def _log_effective_config(self):
        """打印生效的配置参数"""
        effective_config = {
            'entity_uniqueness': {
                'enabled': self.entity_uniqueness_config.get('enabled', True),
                'auto_fix': self.entity_uniqueness_config.get('auto_fix', True),
                'selection_strategy': self.entity_uniqueness_config.get('selection_strategy', 'known_actors')
            },
            'source_traceability': {
                'enabled': self.source_traceability_config.get('enabled', True),
                'strict_mode': self.source_traceability_config.get('strict_mode', False),
                'similarity_threshold': self.source_traceability_config.get('similarity_threshold', 0.8)
            },
            'person_name_patterns_count': len(self.person_name_patterns),
            'known_voice_actors_count': len(self.known_voice_actors),
            'person_indicators_count': len(self.person_indicators)
        }
        logger.info(f"NoteValidator effective config: {effective_config}")
    
    # ==================== 内容质量验证辅助方法 ====================
    
    def _assess_content_completeness(self, content: str, entities: List[str]) -> float:
        """评估内容完整性"""
        if not content.strip():
            return 0.0
        
        # 检查内容长度合理性
        length_score = min(len(content) / 200, 1.0)  # 200字符为满分
        
        # 检查是否包含实体信息
        entity_coverage = 0.0
        if entities:
            covered_entities = sum(1 for entity in entities if entity.lower() in content.lower())
            entity_coverage = covered_entities / len(entities)
        
        # 检查句子完整性
        sentence_completeness = self._check_sentence_completeness(content)
        
        return (length_score + entity_coverage + sentence_completeness) / 3
    
    def _assess_content_relevance(self, content: str, entities: List[str]) -> float:
        """评估内容相关性"""
        if not content.strip():
            return 0.0
        
        # 检查实体密度
        entity_density = 0.0
        if entities:
            total_entity_mentions = sum(content.lower().count(entity.lower()) for entity in entities)
            words = len(content.split())
            entity_density = min(total_entity_mentions / max(words, 1), 1.0)
        
        # 检查关键词密度
        keyword_density = self._calculate_keyword_density(content)
        
        # 检查主题一致性
        topic_consistency = self._check_topic_consistency(content, entities)
        
        return (entity_density + keyword_density + topic_consistency) / 3
    
    def _assess_content_accuracy(self, content: str, paragraph_idxs: List[int], 
                               source_paragraphs: Dict[int, str]) -> float:
        """评估内容准确性"""
        if not paragraph_idxs or not source_paragraphs:
            return 0.5  # 无法验证时给中等分数
        
        # 获取源文本
        source_texts = [source_paragraphs.get(idx, '') for idx in paragraph_idxs]
        combined_source = ' '.join(source_texts)
        
        if not combined_source.strip():
            return 0.0
        
        # 计算与源文本的相似度
        similarity = self._calculate_text_similarity(content, combined_source)
        
        # 检查事实一致性
        fact_consistency = self._check_fact_consistency(content, combined_source)
        
        return (similarity + fact_consistency) / 2
    
    def _check_sentence_completeness(self, content: str) -> float:
        """检查句子完整性"""
        sentences = re.split(r'[.!?]+', content)
        if not sentences:
            return 0.0
        
        complete_sentences = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.endswith('...'):
                complete_sentences += 1
        
        return complete_sentences / max(len(sentences), 1)
    
    def _calculate_keyword_density(self, content: str) -> float:
        """计算关键词密度"""
        # 简单的关键词识别：大写字母开头的词
        words = content.split()
        keywords = [word for word in words if word and word[0].isupper()]
        
        if not words:
            return 0.0
        
        return min(len(keywords) / len(words), 0.3) / 0.3  # 30%为满分
    
    def _check_topic_consistency(self, content: str, entities: List[str]) -> float:
        """检查主题一致性"""
        if not entities:
            return 0.5
        
        # 检查内容是否围绕主要实体展开
        main_entity_mentions = 0
        for entity in entities[:3]:  # 只检查前3个主要实体
            main_entity_mentions += content.lower().count(entity.lower())
        
        words = len(content.split())
        consistency_score = min(main_entity_mentions / max(words * 0.1, 1), 1.0)
        
        return consistency_score
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            # 回退到简单的词汇重叠计算
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / max(len(union), 1)
    
    def _check_fact_consistency(self, content: str, source_text: str) -> float:
        """检查事实一致性"""
        # 提取数字、日期等关键事实
        content_facts = self._extract_facts(content)
        source_facts = self._extract_facts(source_text)
        
        if not content_facts:
            return 1.0  # 没有具体事实，认为一致
        
        consistent_facts = 0
        for fact in content_facts:
            if any(self._facts_match(fact, source_fact) for source_fact in source_facts):
                consistent_facts += 1
        
        return consistent_facts / len(content_facts)
    
    def _extract_facts(self, text: str) -> List[str]:
        """提取文本中的事实信息"""
        facts = []
        
        # 提取数字
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        facts.extend(numbers)
        
        # 提取日期
        dates = re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b', text)
        facts.extend(dates)
        
        # 提取专有名词
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        facts.extend(proper_nouns)
        
        return facts
    
    def _facts_match(self, fact1: str, fact2: str) -> bool:
        """检查两个事实是否匹配"""
        return fact1.lower() == fact2.lower() or fact1 in fact2 or fact2 in fact1
    
    def _identify_quality_issues(self, content: str, entities: List[str]) -> List[str]:
        """识别质量问题"""
        issues = []
        
        if len(content.strip()) < 20:
            issues.append("content_too_short")
        
        if not entities:
            issues.append("no_entities_identified")
        
        if content.count('...') > 2:
            issues.append("incomplete_content")
        
        if not re.search(r'[.!?]$', content.strip()):
            issues.append("incomplete_sentence")
        
        return issues
    
    # ==================== 实体一致性验证辅助方法 ====================
    
    def _build_entity_registry(self, atomic_notes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """构建全局实体注册表"""
        entity_registry = defaultdict(lambda: {
            'variants': set(),
            'contexts': [],
            'frequency': 0,
            'notes': set()
        })
        
        for note in atomic_notes:
            note_id = note.get('note_id', '')
            entities = note.get('entities', [])
            content = note.get('content', '')
            
            for entity in entities:
                canonical_entity = self._get_canonical_entity_name(entity)
                entity_registry[canonical_entity]['variants'].add(entity)
                entity_registry[canonical_entity]['contexts'].append(content)
                entity_registry[canonical_entity]['frequency'] += content.lower().count(entity.lower())
                entity_registry[canonical_entity]['notes'].add(note_id)
        
        return dict(entity_registry)
    
    def _get_canonical_entity_name(self, entity: str) -> str:
        """获取实体的标准名称"""
        # 移除多余空格和标点
        canonical = re.sub(r'\s+', ' ', entity.strip())
        canonical = re.sub(r'[^\w\s]', '', canonical)
        return canonical.title()
    
    def _detect_entity_confusion(self, entities: List[str], content: str, 
                               entity_registry: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测实体混淆"""
        confusion_issues = []
        
        for entity in entities:
            canonical = self._get_canonical_entity_name(entity)
            
            # 检查是否有相似的实体可能被混淆
            similar_entities = self._find_similar_entities(canonical, entity_registry)
            
            for similar_entity, similarity in similar_entities:
                if similarity > 0.8 and canonical != similar_entity:
                    confusion_issues.append({
                        'entity': entity,
                        'confused_with': similar_entity,
                        'similarity': similarity,
                        'type': 'name_similarity'
                    })
        
        return confusion_issues
    
    def _detect_entity_fabrication(self, entities: List[str], content: str) -> List[Dict[str, Any]]:
        """检测虚构实体"""
        fabrication_issues = []
        
        for entity in entities:
            # 检查实体是否在内容中出现
            if entity.lower() not in content.lower():
                fabrication_issues.append({
                    'entity': entity,
                    'type': 'not_in_content',
                    'description': f"Entity '{entity}' not found in content"
                })
            
            # 检查是否为明显的虚构名称
            if self._is_likely_fabricated(entity):
                fabrication_issues.append({
                    'entity': entity,
                    'type': 'likely_fabricated',
                    'description': f"Entity '{entity}' appears to be fabricated"
                })
        
        return fabrication_issues
    
    def _find_similar_entities(self, entity: str, entity_registry: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float]]:
        """查找相似实体"""
        similar_entities = []
        
        for registered_entity in entity_registry.keys():
            if entity != registered_entity:
                similarity = difflib.SequenceMatcher(None, entity.lower(), registered_entity.lower()).ratio()
                if similarity > 0.7:
                    similar_entities.append((registered_entity, similarity))
        
        return sorted(similar_entities, key=lambda x: x[1], reverse=True)
    
    def _is_likely_fabricated(self, entity: str) -> bool:
        """判断实体是否可能是虚构的"""
        # 检查是否包含明显的虚构模式
        fabrication_patterns = [
            r'\b(fake|fictional|made-up|invented)\b',
            r'\b[A-Z][a-z]*X{2,}\b',  # 包含多个X的名称
            r'\b[A-Z][a-z]*\d{3,}\b',  # 包含多个数字的名称
        ]
        
        for pattern in fabrication_patterns:
            if re.search(pattern, entity, re.IGNORECASE):
                return True
        
        return False
    
    def _standardize_entity_names(self, entities: List[str], 
                                entity_registry: Dict[str, Dict[str, Any]]) -> List[str]:
        """标准化实体名称"""
        standardized = []
        
        for entity in entities:
            canonical = self._get_canonical_entity_name(entity)
            
            # 查找最佳匹配的标准名称
            best_match = canonical
            best_frequency = entity_registry.get(canonical, {}).get('frequency', 0)
            
            for registered_entity, info in entity_registry.items():
                if entity.lower() in [v.lower() for v in info['variants']]:
                    if info['frequency'] > best_frequency:
                        best_match = registered_entity
                        best_frequency = info['frequency']
            
            standardized.append(best_match)
        
        return standardized
    
    # ==================== 重复检测辅助方法 ====================
    
    def _find_exact_duplicates(self, atomic_notes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """查找精确重复"""
        content_hash_to_notes = defaultdict(list)
        
        for note in atomic_notes:
            content = note.get('content', '')
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            content_hash_to_notes[content_hash].append(note.get('note_id', ''))
        
        # 只返回有重复的组
        return {h: notes for h, notes in content_hash_to_notes.items() if len(notes) > 1}
    
    def _find_semantic_duplicates(self, atomic_notes: List[Dict[str, Any]], 
                                threshold: float = 0.85) -> List[Tuple[str, str, float]]:
        """查找语义重复"""
        duplicates = []
        
        # 提取所有内容
        contents = [note.get('content', '') for note in atomic_notes]
        note_ids = [note.get('note_id', '') for note in atomic_notes]
        
        if len(contents) < 2:
            return duplicates
        
        try:
            # 使用TF-IDF计算相似度
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(contents)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # 查找高相似度对
            for i in range(len(contents)):
                for j in range(i + 1, len(contents)):
                    similarity = similarity_matrix[i][j]
                    if similarity >= threshold:
                        duplicates.append((note_ids[i], note_ids[j], similarity))
        
        except Exception as e:
            logger.warning(f"Failed to compute semantic similarity: {e}")
            # 回退到简单的词汇重叠
            duplicates = self._find_lexical_duplicates(atomic_notes, threshold)
        
        return duplicates
    
    def _find_lexical_duplicates(self, atomic_notes: List[Dict[str, Any]], 
                               threshold: float = 0.85) -> List[Tuple[str, str, float]]:
        """基于词汇重叠查找重复"""
        duplicates = []
        
        for i, note1 in enumerate(atomic_notes):
            for j, note2 in enumerate(atomic_notes[i + 1:], i + 1):
                content1 = note1.get('content', '')
                content2 = note2.get('content', '')
                
                words1 = set(content1.lower().split())
                words2 = set(content2.lower().split())
                
                if not words1 or not words2:
                    continue
                
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                similarity = len(intersection) / len(union)
                
                if similarity >= threshold:
                    duplicates.append((note1.get('note_id', ''), note2.get('note_id', ''), similarity))
        
        return duplicates
    
    def _find_entity_duplicates(self, atomic_notes: List[Dict[str, Any]], 
                              threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """基于实体重叠查找重复"""
        duplicates = []
        
        for i, note1 in enumerate(atomic_notes):
            for j, note2 in enumerate(atomic_notes[i + 1:], i + 1):
                entities1 = set(note1.get('entities', []))
                entities2 = set(note2.get('entities', []))
                
                if not entities1 or not entities2:
                    continue
                
                intersection = entities1.intersection(entities2)
                union = entities1.union(entities2)
                similarity = len(intersection) / len(union)
                
                if similarity >= threshold:
                    duplicates.append((note1.get('note_id', ''), note2.get('note_id', ''), similarity))
        
        return duplicates
    
    def _merge_duplicate_results(self, exact_duplicates: Dict[str, List[str]], 
                               semantic_duplicates: List[Tuple[str, str, float]], 
                               entity_duplicates: List[Tuple[str, str, float]]) -> Dict[str, Set[str]]:
        """合并重复检测结果"""
        all_duplicates = defaultdict(set)
        
        # 处理精确重复
        for hash_val, note_ids in exact_duplicates.items():
            if len(note_ids) > 1:
                primary = note_ids[0]
                for duplicate in note_ids[1:]:
                    all_duplicates[primary].add(duplicate)
        
        # 处理语义重复
        for note1, note2, similarity in semantic_duplicates:
            all_duplicates[note1].add(note2)
        
        # 处理实体重复
        for note1, note2, similarity in entity_duplicates:
            all_duplicates[note1].add(note2)
        
        return dict(all_duplicates)
    
    def _deduplicate_notes(self, atomic_notes: List[Dict[str, Any]], 
                         duplicates: Dict[str, Set[str]]) -> List[Dict[str, Any]]:
        """去重处理"""
        notes_to_remove = set()
        for primary, dups in duplicates.items():
            notes_to_remove.update(dups)
        
        deduplicated = []
        for note in atomic_notes:
            note_id = note.get('note_id', '')
            if note_id not in notes_to_remove:
                # 如果这是主要note，添加重复信息
                if note_id in duplicates:
                    note['duplicates_removed'] = list(duplicates[note_id])
                deduplicated.append(note)
        
        return deduplicated
    
    # ==================== 构建质量验证辅助方法 ====================
    
    def _detect_content_splicing_issues(self, content: str, paragraph_idxs: List[int], 
                                       source_paragraphs: Dict[int, str]) -> List[Dict[str, Any]]:
        """检测内容拼接问题"""
        issues = []
        
        if not paragraph_idxs or len(paragraph_idxs) < 2:
            return issues
        
        # 检查段落间的连贯性
        for i in range(len(paragraph_idxs) - 1):
            idx1, idx2 = paragraph_idxs[i], paragraph_idxs[i + 1]
            
            if idx1 in source_paragraphs and idx2 in source_paragraphs:
                para1 = source_paragraphs[idx1]
                para2 = source_paragraphs[idx2]
                
                # 检查段落是否相邻
                if abs(idx1 - idx2) > 1:
                    # 检查内容连贯性
                    coherence_score = self._calculate_coherence(para1, para2)
                    if coherence_score < 0.3:
                        issues.append({
                            'type': 'poor_coherence',
                            'paragraphs': [idx1, idx2],
                            'coherence_score': coherence_score,
                            'description': f"Poor coherence between paragraphs {idx1} and {idx2}"
                        })
        
        return issues
    
    def _detect_context_breaks(self, content: str, paragraph_idxs: List[int], 
                             source_paragraphs: Dict[int, str]) -> List[Dict[str, Any]]:
        """检测上下文断裂"""
        issues = []
        
        # 检查内容中的突兀转换
        sentences = re.split(r'[.!?]+', content)
        
        for i in range(len(sentences) - 1):
            sent1 = sentences[i].strip()
            sent2 = sentences[i + 1].strip()
            
            if sent1 and sent2:
                # 检查主题转换是否突兀
                topic_shift = self._detect_abrupt_topic_shift(sent1, sent2)
                if topic_shift:
                    issues.append({
                        'type': 'abrupt_topic_shift',
                        'sentences': [sent1[:50] + '...', sent2[:50] + '...'],
                        'description': "Abrupt topic shift detected"
                    })
        
        return issues
    
    def _detect_information_completeness_issues(self, content: str, paragraph_idxs: List[int], 
                                              source_paragraphs: Dict[int, str]) -> List[Dict[str, Any]]:
        """检测信息完整性问题"""
        issues = []
        
        # 检查是否有未完成的句子或想法
        if content.endswith('...'):
            issues.append({
                'type': 'incomplete_content',
                'description': "Content appears to be incomplete (ends with ...)"
            })
        
        # 检查是否有悬空的引用
        dangling_refs = re.findall(r'\b(he|she|it|they|this|that)\b', content.lower())
        if len(dangling_refs) > len(content.split()) * 0.1:  # 超过10%的代词
            issues.append({
                'type': 'excessive_pronouns',
                'pronoun_count': len(dangling_refs),
                'description': "Excessive use of pronouns may indicate missing context"
            })
        
        return issues
    
    def _calculate_coherence(self, text1: str, text2: str) -> float:
        """计算两段文本的连贯性"""
        # 简单的连贯性计算：基于共同词汇和主题词
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # 计算词汇重叠
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        lexical_overlap = len(intersection) / len(union)
        
        # 检查连接词的存在
        connectives = {'however', 'therefore', 'moreover', 'furthermore', 'additionally', 'also'}
        has_connectives = bool(connectives.intersection(words2))
        
        coherence_score = lexical_overlap
        if has_connectives:
            coherence_score += 0.2
        
        return min(coherence_score, 1.0)
    
    def _detect_abrupt_topic_shift(self, sent1: str, sent2: str) -> bool:
        """检测突兀的主题转换"""
        # 提取主要名词
        nouns1 = re.findall(r'\b[A-Z][a-z]+\b', sent1)
        nouns2 = re.findall(r'\b[A-Z][a-z]+\b', sent2)
        
        if not nouns1 or not nouns2:
            return False
        
        # 检查是否有共同的主题词
        common_nouns = set(nouns1).intersection(set(nouns2))
        
        # 如果没有共同主题词且句子都较长，可能是突兀转换
        return len(common_nouns) == 0 and len(sent1.split()) > 5 and len(sent2.split()) > 5
    
    # ==================== 图谱关系验证辅助方法 ====================
    
    def _analyze_entity_cooccurrence(self, atomic_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析实体共现模式"""
        entity_pairs = defaultdict(int)
        entity_contexts = defaultdict(list)
        
        for note in atomic_notes:
            entities = note.get('entities', [])
            content = note.get('content', '')
            
            # 记录实体上下文
            for entity in entities:
                entity_contexts[entity].append(content)
            
            # 记录实体对共现
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i + 1:]:
                    pair = tuple(sorted([entity1, entity2]))
                    entity_pairs[pair] += 1
        
        return {
            'entity_pairs': dict(entity_pairs),
            'entity_contexts': dict(entity_contexts),
            'total_entities': len(entity_contexts),
            'total_pairs': len(entity_pairs)
        }
    
    def _detect_missing_relations(self, atomic_notes: List[Dict[str, Any]], 
                                entity_cooccurrence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测缺失的关系"""
        missing_relations = []
        entity_pairs = entity_cooccurrence['entity_pairs']
        
        # 查找高频共现但缺少明确关系的实体对
        for (entity1, entity2), frequency in entity_pairs.items():
            if frequency >= 3:  # 共现3次以上
                # 检查是否已有明确的关系描述
                has_explicit_relation = self._check_explicit_relation(entity1, entity2, atomic_notes)
                
                if not has_explicit_relation:
                    missing_relations.append({
                        'entity1': entity1,
                        'entity2': entity2,
                        'cooccurrence_frequency': frequency,
                        'relation_type': 'missing_explicit_relation',
                        'priority': 'high' if frequency >= 5 else 'medium'
                    })
        
        return missing_relations
    
    def _detect_weak_relations(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测弱关系"""
        weak_relations = []
        
        # 分析每个note中的实体关系强度
        for note in atomic_notes:
            entities = note.get('entities', [])
            content = note.get('content', '')
            
            if len(entities) >= 2:
                for i, entity1 in enumerate(entities):
                    for entity2 in entities[i + 1:]:
                        relation_strength = self._calculate_relation_strength(entity1, entity2, content)
                        
                        if relation_strength < 0.3:  # 弱关系阈值
                            weak_relations.append({
                                'entity1': entity1,
                                'entity2': entity2,
                                'note_id': note.get('note_id', ''),
                                'relation_strength': relation_strength,
                                'issue': 'weak_relation_evidence'
                            })
        
        return weak_relations
    
    def _recommend_new_relations(self, atomic_notes: List[Dict[str, Any]], 
                               entity_cooccurrence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """推荐新的关系"""
        recommendations = []
        entity_contexts = entity_cooccurrence['entity_contexts']
        
        # 基于上下文相似性推荐关系
        entities = list(entity_contexts.keys())
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1:]:
                # 计算上下文相似性
                contexts1 = entity_contexts[entity1]
                contexts2 = entity_contexts[entity2]
                
                similarity = self._calculate_context_similarity(contexts1, contexts2)
                
                if similarity > 0.6:  # 高相似性阈值
                    relation_type = self._infer_relation_type(entity1, entity2, contexts1 + contexts2)
                    
                    recommendations.append({
                        'entity1': entity1,
                        'entity2': entity2,
                        'context_similarity': similarity,
                        'recommended_relation_type': relation_type,
                        'confidence': 'high' if similarity > 0.8 else 'medium'
                    })
        
        return recommendations
    
    def _check_explicit_relation(self, entity1: str, entity2: str, 
                               atomic_notes: List[Dict[str, Any]]) -> bool:
        """检查是否存在明确的关系描述"""
        relation_indicators = [
            'is', 'was', 'are', 'were', 'played', 'voiced', 'married', 'related',
            'worked', 'collaborated', 'appeared', 'starred', 'directed', 'produced'
        ]
        
        for note in atomic_notes:
            content = note.get('content', '').lower()
            entities = note.get('entities', [])
            
            if entity1 in entities and entity2 in entities:
                # 检查是否包含关系指示词
                for indicator in relation_indicators:
                    if indicator in content:
                        # 检查实体是否在关系指示词附近
                        if self._entities_near_indicator(entity1, entity2, indicator, content):
                            return True
        
        return False
    
    def _calculate_relation_strength(self, entity1: str, entity2: str, content: str) -> float:
        """计算关系强度"""
        content_lower = content.lower()
        entity1_lower = entity1.lower()
        entity2_lower = entity2.lower()
        
        # 检查实体在文本中的距离
        entity1_positions = [m.start() for m in re.finditer(re.escape(entity1_lower), content_lower)]
        entity2_positions = [m.start() for m in re.finditer(re.escape(entity2_lower), content_lower)]
        
        if not entity1_positions or not entity2_positions:
            return 0.0
        
        # 计算最小距离
        min_distance = min(abs(pos1 - pos2) for pos1 in entity1_positions for pos2 in entity2_positions)
        
        # 距离越近，关系越强
        distance_score = max(0, 1 - min_distance / 100)  # 100字符内为强关系
        
        # 检查关系指示词
        relation_indicators = ['and', 'with', 'by', 'of', 'in', 'as']
        indicator_score = 0
        for indicator in relation_indicators:
            if indicator in content_lower:
                indicator_score += 0.1
        
        return min(distance_score + indicator_score, 1.0)
    
    def _calculate_context_similarity(self, contexts1: List[str], contexts2: List[str]) -> float:
        """计算上下文相似性"""
        if not contexts1 or not contexts2:
            return 0.0
        
        # 合并上下文
        combined1 = ' '.join(contexts1)
        combined2 = ' '.join(contexts2)
        
        return self._calculate_text_similarity(combined1, combined2)
    
    def _infer_relation_type(self, entity1: str, entity2: str, contexts: List[str]) -> str:
        """推断关系类型"""
        combined_context = ' '.join(contexts).lower()
        
        # 基于关键词推断关系类型
        if any(word in combined_context for word in ['voice', 'played', 'character']):
            return 'voice_actor_character'
        elif any(word in combined_context for word in ['married', 'spouse', 'wife', 'husband']):
            return 'marriage'
        elif any(word in combined_context for word in ['work', 'collaborate', 'project']):
            return 'collaboration'
        elif any(word in combined_context for word in ['appear', 'show', 'episode']):
            return 'appearance'
        else:
            return 'related'
    
    def _entities_near_indicator(self, entity1: str, entity2: str, indicator: str, content: str) -> bool:
        """检查实体是否在关系指示词附近"""
        content_lower = content.lower()
        entity1_lower = entity1.lower()
        entity2_lower = entity2.lower()
        
        # 查找指示词位置
        indicator_positions = [m.start() for m in re.finditer(re.escape(indicator), content_lower)]
        
        for pos in indicator_positions:
            # 检查实体是否在指示词前后50字符内
            window_start = max(0, pos - 50)
            window_end = min(len(content), pos + 50)
            window = content_lower[window_start:window_end]
            
            if entity1_lower in window and entity2_lower in window:
                return True
        
        return False
    
    def validate_entity_uniqueness(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证实体唯一性：每个note应只允许一个主实体（人物）"""
        if not self.entity_uniqueness_config.get('enabled', True):
            logger.info("Entity uniqueness validation is disabled")
            return atomic_notes
            
        logger.info("Starting entity uniqueness validation")
        validated_notes = []
        validation_errors = []
        
        for note in atomic_notes:
            note_id = note.get('note_id', 'unknown')
            entities = note.get('entities', [])
            content = note.get('content', '')
            
            # 提取人名实体
            person_entities = self._extract_person_entities(entities, content)
            
            if len(person_entities) <= 1:
                # 通过验证：没有人名或只有一个人名
                note['validation_status'] = 'passed'
                note['primary_entity'] = person_entities[0] if person_entities else None
                validated_notes.append(note)
            else:
                # 失败：包含多个人名实体
                error_info = {
                    'note_id': note_id,
                    'error_type': 'multiple_person_entities',
                    'detected_entities': person_entities,
                    'content_preview': content[:100] + '...' if len(content) > 100 else content
                }
                validation_errors.append(error_info)
                
                # 根据配置决定是否自动修复
                if self.entity_uniqueness_config.get('auto_fix', True):
                    # 尝试修复：选择最重要的实体作为主实体
                    primary_entity = self._select_primary_entity(person_entities, content)
                    if primary_entity:
                        # 创建修复后的note
                        fixed_note = note.copy()
                        fixed_note['validation_status'] = 'fixed'
                        fixed_note['primary_entity'] = primary_entity
                        fixed_note['validation_errors'] = [error_info]
                        fixed_note['entities'] = [e for e in entities if e == primary_entity or not self._is_person_name(e)]
                        validated_notes.append(fixed_note)
                    else:
                        # 无法修复，标记为失败
                        note['validation_status'] = 'failed'
                        note['validation_errors'] = [error_info]
                        validated_notes.append(note)
                else:
                    # 不自动修复，直接标记为失败
                    note['validation_status'] = 'failed'
                    note['validation_errors'] = [error_info]
                    validated_notes.append(note)
        
        logger.info(f"Entity uniqueness validation completed. {len(validation_errors)} errors found")
        if validation_errors:
            logger.warning(f"Validation errors: {validation_errors}")
        
        return validated_notes
    
    def validate_source_entity_traceability(self, atomic_notes: List[Dict[str, Any]], 
                                          source_paragraphs: Dict[int, str]) -> List[Dict[str, Any]]:
        """验证源文档实体回溯约束"""
        if not self.source_traceability_config.get('enabled', True):
            logger.info("Source entity traceability validation is disabled")
            return atomic_notes
            
        logger.info("Starting source entity traceability validation")
        validated_notes = []
        
        for note in atomic_notes:
            note_id = note.get('note_id', 'unknown')
            paragraph_idxs = note.get('paragraph_idxs', [])
            entities = note.get('entities', [])
            content = note.get('content', '')
            
            # 获取源段落文本
            source_texts = []
            for idx in paragraph_idxs:
                if idx in source_paragraphs:
                    source_texts.append(source_paragraphs[idx])
            
            # 验证实体来源
            trace_info = self._trace_entity_sources(entities, content, source_texts, paragraph_idxs)
            
            # 添加trace字段
            note['entity_trace'] = trace_info
            
            # 检查是否有拼接错误
            splicing_errors = self._detect_splicing_errors(entities, content, source_texts, paragraph_idxs)
            
            if splicing_errors:
                note['validation_status'] = note.get('validation_status', 'passed') + '_with_splicing_errors'
                note['splicing_errors'] = splicing_errors
                logger.warning(f"Splicing errors detected in note {note_id}: {splicing_errors}")
            
            validated_notes.append(note)
        
        logger.info("Source entity traceability validation completed")
        return validated_notes
    
    def validate_content_quality(self, atomic_notes: List[Dict[str, Any]], 
                               source_paragraphs: Dict[int, str] = None) -> Dict[str, Any]:
        """验证召回内容质量，包括完整性、相关性和准确性评估"""
        try:
            quality_scores = []
            all_quality_issues = []
            
            for note in atomic_notes:
                content = note.get('content', '')
                entities = note.get('entities', [])
                paragraph_idxs = note.get('paragraph_idxs', [])
                
                # 评估内容完整性
                completeness_score = self._assess_content_completeness(content, entities)
                
                # 评估内容相关性
                relevance_score = self._assess_content_relevance(content, entities)
                
                # 评估内容准确性
                accuracy_score = self._assess_content_accuracy(content, paragraph_idxs, source_paragraphs or {})
                
                # 计算综合质量分数
                overall_score = (completeness_score + relevance_score + accuracy_score) / 3
                
                # 识别质量问题
                quality_issues = self._identify_quality_issues(content, entities)
                
                note_quality = {
                    'note_id': note.get('note_id', ''),
                    'completeness_score': completeness_score,
                    'relevance_score': relevance_score,
                    'accuracy_score': accuracy_score,
                    'overall_score': overall_score,
                    'issues': quality_issues
                }
                
                quality_scores.append(note_quality)
                all_quality_issues.extend([{**{'note_id': note.get('note_id', ''), 'issue_type': issue}} for issue in quality_issues])
            
            # 计算整体质量分数
            overall_quality = sum(score['overall_score'] for score in quality_scores) / max(len(quality_scores), 1)
            
            # 确定验证状态
            status = 'passed'
            if overall_quality < 0.6:
                status = 'failed'
            elif overall_quality < 0.8:
                status = 'warning'
            
            return {
                'status': status,
                'quality_scores': quality_scores,
                'quality_issues': all_quality_issues,
                'overall_quality': overall_quality,
                'total_notes': len(atomic_notes),
                'low_quality_notes': len([s for s in quality_scores if s['overall_score'] < 0.7])
            }
            
        except Exception as e:
            logger.error(f"Content quality validation failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'quality_scores': [],
                'quality_issues': [],
                'overall_quality': 0.0
            }

    def validate_entity_consistency(self, atomic_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证实体一致性，解决实体混淆和虚构内容问题"""
        try:
            # 构建全局实体注册表
            entity_registry = self._build_entity_registry(atomic_notes)
            
            all_confusion_issues = []
            all_fabrication_issues = []
            standardized_notes = []
            
            for note in atomic_notes:
                entities = note.get('entities', [])
                content = note.get('content', '')
                
                # 检测实体混淆
                confusion_issues = self._detect_entity_confusion(entities, content, entity_registry)
                all_confusion_issues.extend([{**issue, 'note_id': note.get('note_id', '')} for issue in confusion_issues])
                
                # 检测虚构实体
                fabrication_issues = self._detect_entity_fabrication(entities, content)
                all_fabrication_issues.extend([{**issue, 'note_id': note.get('note_id', '')} for issue in fabrication_issues])
                
                # 标准化实体名称
                standardized_entities = self._standardize_entity_names(entities, entity_registry)
                
                # 创建标准化的note
                standardized_note = note.copy()
                standardized_note['entities'] = standardized_entities
                standardized_note['original_entities'] = entities  # 保留原始实体
                standardized_notes.append(standardized_note)
            
            # 计算一致性分数
            total_entities = sum(len(note.get('entities', [])) for note in atomic_notes)
            problematic_entities = len(all_confusion_issues) + len(all_fabrication_issues)
            consistency_score = 1.0 - (problematic_entities / max(total_entities, 1))
            
            # 确定验证状态
            status = 'passed'
            if consistency_score < 0.7:
                status = 'failed'
            elif consistency_score < 0.9:
                status = 'warning'
            
            return {
                'status': status,
                'confusion_issues': all_confusion_issues,
                'fabrication_issues': all_fabrication_issues,
                'standardized_notes': standardized_notes,
                'consistency_score': consistency_score,
                'entity_registry_size': len(entity_registry)
            }
            
        except Exception as e:
            logger.error(f"Entity consistency validation failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'confusion_issues': [],
                'fabrication_issues': [],
                'standardized_notes': atomic_notes
            }

    def detect_duplicate_notes(self, atomic_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检测和处理重复的原子笔记"""
        try:
            # 查找精确重复
            exact_duplicates = self._find_exact_duplicates(atomic_notes)
            
            # 查找语义重复
            semantic_duplicates = self._find_semantic_duplicates(atomic_notes)
            
            # 查找实体重复
            entity_duplicates = self._find_entity_duplicates(atomic_notes)
            
            # 合并重复检测结果
            all_duplicates = self._merge_duplicate_results(exact_duplicates, semantic_duplicates, entity_duplicates)
            
            # 执行去重
            deduplicated_notes = self._deduplicate_notes(atomic_notes, all_duplicates)
            
            # 计算重复率
            total_duplicates = sum(len(dups) for dups in all_duplicates.values())
            duplicate_rate = total_duplicates / max(len(atomic_notes), 1)
            
            # 确定验证状态
            status = 'passed'
            if duplicate_rate > 0.3:
                status = 'failed'
            elif duplicate_rate > 0.1:
                status = 'warning'
            
            return {
                'status': status,
                'exact_duplicates': exact_duplicates,
                'semantic_duplicates': semantic_duplicates,
                'entity_duplicates': entity_duplicates,
                'deduplicated_notes': deduplicated_notes,
                'duplicate_rate': duplicate_rate,
                'removed_count': total_duplicates,
                'remaining_count': len(deduplicated_notes)
            }
            
        except Exception as e:
            logger.error(f"Duplicate detection failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'exact_duplicates': {},
                'semantic_duplicates': [],
                'entity_duplicates': [],
                'deduplicated_notes': atomic_notes
            }

    def validate_note_construction(self, atomic_notes: List[Dict[str, Any]], 
                                 source_paragraphs: Dict[int, str] = None) -> Dict[str, Any]:
        """验证笔记构建质量，检测拼接污染、上下文断裂和信息完整性问题"""
        try:
            all_splicing_issues = []
            all_context_breaks = []
            all_completeness_issues = []
            total_score = 0.0
            
            for note in atomic_notes:
                content = note.get('content', '')
                paragraph_idxs = note.get('paragraph_idxs', [])
                
                # 检测拼接问题
                splicing_issues = self._detect_content_splicing_issues(content, paragraph_idxs, source_paragraphs or {})
                all_splicing_issues.extend([{**issue, 'note_id': note.get('note_id', '')} for issue in splicing_issues])
                
                # 检测上下文断裂
                context_breaks = self._detect_context_breaks(content, paragraph_idxs, source_paragraphs or {})
                all_context_breaks.extend([{**issue, 'note_id': note.get('note_id', '')} for issue in context_breaks])
                
                # 检测信息完整性问题
                completeness_issues = self._detect_information_completeness_issues(content, paragraph_idxs, source_paragraphs or {})
                all_completeness_issues.extend([{**issue, 'note_id': note.get('note_id', '')} for issue in completeness_issues])
                
                # 计算单个note的构建分数
                note_score = 1.0
                note_score -= len(splicing_issues) * 0.2
                note_score -= len(context_breaks) * 0.15
                note_score -= len(completeness_issues) * 0.1
                total_score += max(note_score, 0.0)
            
            # 计算平均构建分数
            construction_score = total_score / max(len(atomic_notes), 1)
            
            # 确定验证状态
            status = 'passed'
            total_issues = len(all_splicing_issues) + len(all_context_breaks) + len(all_completeness_issues)
            
            if total_issues > len(atomic_notes) * 0.3 or construction_score < 0.6:
                status = 'failed'
            elif total_issues > len(atomic_notes) * 0.1 or construction_score < 0.8:
                status = 'warning'
            
            return {
                'status': status,
                'splicing_issues': all_splicing_issues,
                'context_breaks': all_context_breaks,
                'completeness_issues': all_completeness_issues,
                'construction_score': construction_score,
                'total_issues': total_issues
            }
            
        except Exception as e:
            logger.error(f"Note construction validation failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'splicing_issues': [],
                'context_breaks': [],
                'completeness_issues': [],
                'construction_score': 0.0
            }

    def validate_graph_relations(self, atomic_notes: List[Dict[str, Any]], 
                                existing_relations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """验证图谱关系完整性，识别缺失的关键实体关系边"""
        try:
            # 分析实体共现模式
            entity_cooccurrence = self._analyze_entity_cooccurrence(atomic_notes)
            
            # 检测缺失的关系
            missing_relations = self._detect_missing_relations(atomic_notes, entity_cooccurrence)
            
            # 检测弱关系
            weak_relations = self._detect_weak_relations(atomic_notes)
            
            # 推荐新的关系
            recommended_relations = self._recommend_new_relations(atomic_notes, entity_cooccurrence)
            
            # 计算关系覆盖率
            total_possible_relations = len(entity_cooccurrence['entity_pairs'])
            explicit_relations = sum(1 for rel in missing_relations if rel['relation_type'] != 'missing_explicit_relation')
            relation_coverage = explicit_relations / max(total_possible_relations, 1)
            
            # 确定验证状态
            status = 'passed'
            if len(missing_relations) > 5 or relation_coverage < 0.5:
                status = 'failed'
            elif len(missing_relations) > 2 or relation_coverage < 0.7:
                status = 'warning'
            
            return {
                'status': status,
                'missing_relations': missing_relations,
                'weak_relations': weak_relations,
                'recommended_relations': recommended_relations,
                'relation_coverage': relation_coverage,
                'entity_cooccurrence_stats': {
                    'total_entities': entity_cooccurrence['total_entities'],
                    'total_pairs': entity_cooccurrence['total_pairs']
                }
            }
            
        except Exception as e:
            logger.error(f"Graph relation validation failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'missing_relations': [],
                'weak_relations': [],
                'recommended_relations': [],
                'relation_coverage': 0.0
            }

    def _extract_person_entities(self, entities: List[str], content: str) -> List[str]:
        """从实体列表和内容中提取人名实体"""
        person_entities = []
        
        # 检查已知的实体列表
        for entity in entities:
            if self._is_person_name(entity):
                person_entities.append(entity)
        
        # 从内容中提取额外的人名
        for pattern in self.person_name_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if match not in person_entities and self._is_person_name(match):
                    person_entities.append(match)
        
        # 检查已知的配音演员
        for actor in self.known_voice_actors.keys():
            if actor in content and actor not in person_entities:
                person_entities.append(actor)
        
        return list(set(person_entities))  # 去重
    
    def _is_person_name(self, entity: str) -> bool:
        """判断实体是否为人名"""
        # 检查是否在已知配音演员列表中
        if entity in self.known_voice_actors:
            return True
        
        # 检查是否符合人名模式
        for pattern in self.person_name_patterns:
            if re.match(pattern, entity):
                return True
        
        # 使用配置中的人名指示词进行启发式判断
        entity_lower = entity.lower()
        return any(indicator in entity_lower for indicator in self.person_indicators)
    
    def _select_primary_entity(self, person_entities: List[str], content: str) -> Optional[str]:
        """从多个人名实体中选择主要实体"""
        if not person_entities:
            return None
        
        strategy = self.entity_uniqueness_config.get('selection_strategy', 'known_actors')
        
        if strategy == 'known_actors':
            # 优先选择已知的配音演员
            for entity in person_entities:
                if entity in self.known_voice_actors:
                    return entity
            # 如果没有已知演员，回退到频率策略
            return self._select_by_frequency(person_entities, content)
        
        elif strategy == 'frequency':
            # 选择在内容中出现频率最高的实体
            return self._select_by_frequency(person_entities, content)
        
        elif strategy == 'first':
            # 选择第一个实体
            return person_entities[0]
        
        else:
            # 默认策略：已知演员优先
            logger.warning(f"Unknown selection strategy: {strategy}. Using 'known_actors' strategy.")
            return self._select_primary_entity(person_entities, content)
    
    def _select_by_frequency(self, person_entities: List[str], content: str) -> Optional[str]:
        """根据频率选择实体"""
        entity_counts = {}
        for entity in person_entities:
            entity_counts[entity] = content.lower().count(entity.lower())
        
        return max(entity_counts, key=entity_counts.get) if entity_counts else None
    
    def _trace_entity_sources(self, entities: List[str], content: str, 
                            source_texts: List[str], paragraph_idxs: List[int]) -> Dict[str, Any]:
        """追踪实体来源"""
        trace_info = {
            'total_entities': len(entities),
            'traced_entities': {},
            'untraced_entities': [],
            'source_paragraphs': paragraph_idxs
        }
        
        combined_source_text = ' '.join(source_texts).lower()
        
        for entity in entities:
            entity_lower = entity.lower()
            found_in_sources = []
            
            # 检查每个源段落
            for i, source_text in enumerate(source_texts):
                if entity_lower in source_text.lower():
                    found_in_sources.append(paragraph_idxs[i] if i < len(paragraph_idxs) else i)
            
            if found_in_sources:
                trace_info['traced_entities'][entity] = {
                    'found_in_paragraphs': found_in_sources,
                    'verification_status': 'verified'
                }
            else:
                trace_info['untraced_entities'].append(entity)
                trace_info['traced_entities'][entity] = {
                    'found_in_paragraphs': [],
                    'verification_status': 'unverified'
                }
        
        return trace_info
    
    def _detect_splicing_errors(self, entities: List[str], content: str, 
                              source_texts: List[str], paragraph_idxs: List[int]) -> List[Dict[str, Any]]:
        """检测拼接错误"""
        errors = []
        
        if not source_texts or not paragraph_idxs:
            return errors
        
        combined_source_text = ' '.join(source_texts).lower()
        
        # 检查内容中的人名是否都来自指定的源段落
        content_person_entities = self._extract_person_entities(entities, content)
        
        for entity in content_person_entities:
            entity_lower = entity.lower()
            
            # 检查实体是否在源文本中
            if entity_lower not in combined_source_text:
                # 进一步检查是否在其他段落中（这表明可能的拼接错误）
                error = {
                    'entity': entity,
                    'error_type': 'entity_not_in_source',
                    'expected_paragraphs': paragraph_idxs,
                    'description': f"Entity '{entity}' appears in note content but not found in source paragraphs {paragraph_idxs}"
                }
                errors.append(error)
        
        return errors
    
    def generate_validation_report(self, validated_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成验证报告"""
        report = {
            'total_notes': len(validated_notes),
            'passed_notes': 0,
            'fixed_notes': 0,
            'failed_notes': 0,
            'notes_with_splicing_errors': 0,
            'entity_uniqueness_errors': [],
            'splicing_errors': [],
            'summary': {}
        }
        
        for note in validated_notes:
            status = note.get('validation_status', 'unknown')
            
            if status == 'passed':
                report['passed_notes'] += 1
            elif status == 'fixed':
                report['fixed_notes'] += 1
            elif status == 'failed':
                report['failed_notes'] += 1
            
            if 'splicing_errors' in note:
                report['notes_with_splicing_errors'] += 1
                report['splicing_errors'].extend(note['splicing_errors'])
            
            if 'validation_errors' in note:
                report['entity_uniqueness_errors'].extend(note['validation_errors'])
        
        # 生成摘要
        report['summary'] = {
            'success_rate': (report['passed_notes'] + report['fixed_notes']) / report['total_notes'] * 100,
            'entity_uniqueness_error_rate': len(report['entity_uniqueness_errors']) / report['total_notes'] * 100,
            'splicing_error_rate': report['notes_with_splicing_errors'] / report['total_notes'] * 100
        }
        
        return report
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载验证配置"""
        if config_path is None:
            # 默认配置文件路径
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'validation_config.yaml')
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    full_config = yaml.safe_load(f)
                    return full_config.get('validation', {})
            else:
                logger.warning(f"Validation config file not found: {config_path}. Using default settings.")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load validation config: {e}. Using default settings.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'enabled': True,
            'entity_uniqueness': {
                'enabled': True,
                'auto_fix': True,
                'selection_strategy': 'known_actors'
            },
            'source_traceability': {
                'enabled': True,
                'strict_mode': False,
                'similarity_threshold': 0.8
            },
            'known_voice_actors': {
                'Dan Castellaneta': ['Krusty the Clown', 'Homer Simpson'],
                'Adriana Caselotti': ['Snow White'],
                'Florian St. Pierre': []
            },
            'person_name_patterns': [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                r'\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b',
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'
            ],
            'person_indicators': ['actor', 'actress', 'voice', 'performer', 'artist'],
            'error_handling': {
                'on_entity_uniqueness_error': 'warn',
                'on_source_traceability_error': 'warn',
                'continue_on_error': True
            },
            'reporting': {
                'detailed_report': True,
                'save_logs': True,
                'log_file': 'validation_logs.json'
            }
        }