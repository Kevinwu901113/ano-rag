from typing import List, Dict, Any, Set, Optional
import re
import difflib
from loguru import logger
from collections import defaultdict

class EnhancedNER:
    """增强的命名实体识别模块，专门针对人物识别和实体归一化优化"""
    
    def __init__(self):
        # 人物识别模式
        self.person_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # 标准英文姓名
            r'\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b',  # 中间名缩写
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # 三个词的姓名
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # 四个词的姓名
        ]
        
        # 非正式称谓和修饰词过滤
        self.title_prefixes = {
            'his majesty', 'her majesty', 'the sole', 'the great', 'the magnificent',
            'lord', 'lady', 'sir', 'dame', 'dr', 'professor', 'mr', 'mrs', 'ms',
            'the honorable', 'the right honorable', 'his excellency', 'her excellency'
        }
        
        # 非人类实体标识词
        self.non_person_indicators = {
            'company', 'corporation', 'inc', 'ltd', 'llc', 'organization', 'foundation',
            'university', 'college', 'school', 'hospital', 'museum', 'library',
            'government', 'department', 'ministry', 'agency', 'bureau', 'office',
            'city', 'town', 'village', 'country', 'nation', 'state', 'province',
            'building', 'tower', 'center', 'mall', 'park', 'street', 'avenue',
            'movie', 'film', 'book', 'novel', 'series', 'show', 'program',
            'band', 'group', 'team', 'club', 'party', 'union'
        }
        
        # 实体别名映射缓存
        self.entity_aliases = defaultdict(set)
        
    def extract_entities(self, text: str, filter_non_persons: bool = True) -> List[str]:
        """提取实体，支持人物过滤"""
        entities = []
        
        # 使用多种模式提取实体
        for pattern in self.person_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        # 提取其他大写开头的实体
        general_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(general_entities)
        
        # 去重
        entities = list(set(entities))
        
        # 清理和过滤
        cleaned_entities = []
        for entity in entities:
            cleaned_entity = self._clean_entity(entity)
            if cleaned_entity and self._is_valid_entity(cleaned_entity, filter_non_persons):
                cleaned_entities.append(cleaned_entity)
        
        return list(set(cleaned_entities))
    
    def _clean_entity(self, entity: str) -> str:
        """清理实体，去除非正式称谓和修饰词"""
        entity = entity.strip()
        
        # 转换为小写进行匹配
        entity_lower = entity.lower()
        
        # 去除称谓前缀
        for prefix in self.title_prefixes:
            if entity_lower.startswith(prefix + ' '):
                entity = entity[len(prefix):].strip()
                break
        
        # 去除"The"前缀（如果不是专有名词的一部分）
        if entity.lower().startswith('the ') and len(entity.split()) > 2:
            # 检查是否是"The + 形容词 + 名词"的模式
            words = entity.split()
            if len(words) >= 3 and words[1].lower() in ['great', 'magnificent', 'sole', 'only']:
                entity = ' '.join(words[2:])
            elif not self._is_proper_noun_with_the(entity):
                entity = entity[4:].strip()
        
        return entity
    
    def _is_proper_noun_with_the(self, entity: str) -> bool:
        """判断是否是以The开头的专有名词"""
        proper_nouns_with_the = {
            'the simpsons', 'the beatles', 'the rolling stones', 'the who',
            'the united states', 'the united kingdom', 'the netherlands',
            'the scarecrow', 'the wizard', 'the lion'
        }
        return entity.lower() in proper_nouns_with_the
    
    def _is_valid_entity(self, entity: str, filter_non_persons: bool) -> bool:
        """验证实体是否有效"""
        if len(entity) < 2:
            return False
        
        # 过滤明显的非实体
        if entity.lower() in {'and', 'or', 'but', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}:
            return False
        
        if filter_non_persons:
            return self._is_likely_person(entity)
        
        return True
    
    def _is_likely_person(self, entity: str) -> bool:
        """判断实体是否可能是人物"""
        entity_lower = entity.lower()
        
        # 检查是否包含非人类实体指示词
        for indicator in self.non_person_indicators:
            if indicator in entity_lower:
                return False
        
        # 检查姓名模式
        words = entity.split()
        
        # 单个词的情况
        if len(words) == 1:
            # 如果是常见的人名，认为是人物
            common_names = {
                'john', 'mary', 'james', 'patricia', 'robert', 'jennifer',
                'michael', 'linda', 'william', 'elizabeth', 'david', 'barbara',
                'richard', 'susan', 'joseph', 'jessica', 'thomas', 'sarah',
                'charles', 'karen', 'christopher', 'nancy', 'daniel', 'lisa',
                'matthew', 'betty', 'anthony', 'helen', 'mark', 'sandra',
                'donald', 'donna', 'steven', 'carol', 'paul', 'ruth',
                'andrew', 'sharon', 'joshua', 'michelle', 'kenneth', 'laura',
                'kevin', 'sarah', 'brian', 'kimberly', 'george', 'deborah',
                'edward', 'dorothy', 'ronald', 'lisa', 'timothy', 'nancy',
                'jason', 'karen', 'jeffrey', 'betty', 'ryan', 'helen',
                'jacob', 'sandra', 'gary', 'donna', 'nicholas', 'carol',
                'eric', 'ruth', 'jonathan', 'sharon', 'stephen', 'michelle',
                'larry', 'laura', 'justin', 'sarah', 'scott', 'kimberly',
                'brandon', 'deborah', 'benjamin', 'dorothy', 'samuel', 'lisa',
                'frank', 'nancy', 'raymond', 'karen', 'alexander', 'betty',
                'patrick', 'helen', 'jack', 'sandra', 'dennis', 'donna',
                'jerry', 'carol', 'tyler', 'ruth', 'aaron', 'sharon',
                'henry', 'michelle', 'douglas', 'laura', 'nathaniel', 'sarah',
                'peter', 'kimberly', 'zachary', 'deborah', 'kyle', 'dorothy'
            }
            return entity_lower in common_names
        
        # 两个或更多词的情况
        elif len(words) >= 2:
            # 检查是否符合姓名模式
            for word in words:
                if not word[0].isupper() or not word[1:].islower():
                    return False
            return True
        
        return False
    
    def normalize_entities(self, entities: List[str], similarity_threshold: float = 0.8) -> Dict[str, List[str]]:
        """实体别名归一化"""
        if not entities:
            return {}
        
        # 构建实体组
        entity_groups = defaultdict(list)
        processed = set()
        
        for i, entity1 in enumerate(entities):
            if entity1 in processed:
                continue
            
            # 创建新组
            group_key = entity1
            entity_groups[group_key].append(entity1)
            processed.add(entity1)
            
            # 查找相似实体
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if entity2 in processed:
                    continue
                
                if self._are_similar_entities(entity1, entity2, similarity_threshold):
                    entity_groups[group_key].append(entity2)
                    processed.add(entity2)
        
        # 选择每组的代表实体
        normalized_mapping = {}
        for group_key, group_entities in entity_groups.items():
            # 选择最短且最常见的形式作为代表
            representative = self._select_representative_entity(group_entities)
            for entity in group_entities:
                normalized_mapping[entity] = representative
        
        return normalized_mapping
    
    def _are_similar_entities(self, entity1: str, entity2: str, threshold: float) -> bool:
        """判断两个实体是否相似"""
        # 完全匹配
        if entity1 == entity2:
            return True
        
        # 忽略大小写匹配
        if entity1.lower() == entity2.lower():
            return True
        
        # 检查包含关系（处理"The Scarecrow Oz" vs "Scarecrow Oz"的情况）
        if entity1.lower() in entity2.lower() or entity2.lower() in entity1.lower():
            # 确保不是偶然的子串匹配
            words1 = set(entity1.lower().split())
            words2 = set(entity2.lower().split())
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            if overlap / total >= 0.7:
                return True
        
        # 使用序列匹配器计算相似度
        similarity = difflib.SequenceMatcher(None, entity1.lower(), entity2.lower()).ratio()
        return similarity >= threshold
    
    def _select_representative_entity(self, entities: List[str]) -> str:
        """选择代表实体"""
        if len(entities) == 1:
            return entities[0]
        
        # 优先选择不包含"The"前缀的版本
        non_the_entities = [e for e in entities if not e.lower().startswith('the ')]
        if non_the_entities:
            # 在不包含"The"的实体中选择最短的
            return min(non_the_entities, key=len)
        
        # 如果都包含"The"，选择最短的
        return min(entities, key=len)
    
    def extract_and_normalize_entities(self, text: str, filter_non_persons: bool = True) -> Dict[str, Any]:
        """提取并归一化实体的完整流程"""
        # 提取实体
        raw_entities = self.extract_entities(text, filter_non_persons)
        
        # 归一化
        normalization_mapping = self.normalize_entities(raw_entities)
        
        # 获取最终的唯一实体列表
        unique_entities = list(set(normalization_mapping.values()))
        
        return {
            'raw_entities': raw_entities,
            'normalized_entities': unique_entities,
            'normalization_mapping': normalization_mapping,
            'entity_count': len(unique_entities)
        }
    
    def enhance_entity_tracing(self, atomic_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """增强实体追踪，提高traced_entities的数量"""
        logger.info("Enhancing entity tracing for atomic notes")
        
        # 构建全局实体映射
        global_entities = set()
        for note in atomic_notes:
            entities = note.get('entities', [])
            global_entities.update(entities)
        
        # 归一化全局实体
        global_normalization = self.normalize_entities(list(global_entities))
        
        enhanced_notes = []
        for note in atomic_notes:
            enhanced_note = note.copy()
            
            # 重新提取和归一化实体
            content = note.get('content', '')
            entity_result = self.extract_and_normalize_entities(content)
            
            # 更新实体信息
            enhanced_note['entities'] = entity_result['normalized_entities']
            enhanced_note['raw_entities'] = entity_result['raw_entities']
            enhanced_note['entity_normalization'] = entity_result['normalization_mapping']
            
            # 增强entity_trace
            if 'entity_trace' not in enhanced_note:
                enhanced_note['entity_trace'] = {}
            
            # 提高traced_entities的覆盖率
            traced_entities = []
            for entity in enhanced_note['entities']:
                # 在原文中查找实体的所有出现位置
                entity_positions = self._find_entity_positions(content, entity)
                if entity_positions:
                    traced_entities.append({
                        'entity': entity,
                        'positions': entity_positions,
                        'confidence': 0.9,  # 高置信度
                        'source': 'enhanced_ner'
                    })
            
            enhanced_note['entity_trace']['traced_entities'] = traced_entities
            enhanced_note['entity_trace']['trace_coverage'] = len(traced_entities) / max(len(enhanced_note['entities']), 1)
            
            enhanced_notes.append(enhanced_note)
        
        logger.info(f"Enhanced entity tracing for {len(enhanced_notes)} notes")
        return enhanced_notes
    
    def _find_entity_positions(self, text: str, entity: str) -> List[Dict[str, Any]]:
        """查找实体在文本中的位置"""
        positions = []
        entity_lower = entity.lower()
        text_lower = text.lower()
        
        start = 0
        while True:
            pos = text_lower.find(entity_lower, start)
            if pos == -1:
                break
            
            positions.append({
                'start': pos,
                'end': pos + len(entity),
                'text': text[pos:pos + len(entity)]
            })
            start = pos + 1
        
        return positions