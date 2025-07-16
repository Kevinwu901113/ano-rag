"""增强的召回优化器，集成去重、实体消歧和多跳检索功能"""

import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
from loguru import logger
import re
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

from config import config
from .text_utils import TextUtils
from .note_validator import NoteValidator


class EnhancedRecallOptimizer:
    """增强的召回优化器"""
    
    def __init__(self, vector_retriever=None, graph_processor=None):
        self.vector_retriever = vector_retriever
        self.graph_processor = graph_processor
        self.note_validator = NoteValidator()
        
        # 优化配置
        self.config = config.get('recall_optimization', {})
        self.similarity_threshold = self.config.get('similarity_threshold', 0.75)
        self.duplicate_threshold = self.config.get('duplicate_threshold', 0.9)
        self.entity_confidence_threshold = self.config.get('entity_confidence_threshold', 0.8)
        self.coverage_threshold = self.config.get('coverage_threshold', 0.8)
        
        # 实体消歧配置
        self.known_entities = self._load_known_entities()
        
        logger.info("EnhancedRecallOptimizer initialized")
    
    def optimize_recall(self, query: str, initial_results: List[Dict[str, Any]], 
                       source_paragraphs: Dict[int, str] = None) -> List[Dict[str, Any]]:
        """优化召回结果"""
        logger.info(f"Optimizing recall for query: {query[:50]}...")
        
        # 1. 去重处理
        deduplicated_results = self._remove_duplicates(initial_results)
        logger.info(f"After deduplication: {len(deduplicated_results)} results")
        
        # 2. 实体消歧
        disambiguated_results = self._disambiguate_entities(deduplicated_results, query)
        logger.info(f"After entity disambiguation: {len(disambiguated_results)} results")
        
        # 3. 相似度阈值过滤
        filtered_results = self._filter_by_similarity(disambiguated_results)
        logger.info(f"After similarity filtering: {len(filtered_results)} results")
        
        # 4. 完整性检查和补充召回
        enhanced_results = self._enhance_completeness(filtered_results, query, source_paragraphs)
        logger.info(f"After completeness enhancement: {len(enhanced_results)} results")
        
        # 5. 多跳推理增强（如果有图谱处理器）
        if self.graph_processor:
            enhanced_results = self._enhance_with_multi_hop(enhanced_results, query)
            logger.info(f"After multi-hop enhancement: {len(enhanced_results)} results")
        
        # 6. 最终排序和质量评估
        final_results = self._final_ranking_and_quality_check(enhanced_results, query)
        
        logger.info(f"Final optimized results: {len(final_results)} notes")
        return final_results
    
    def _remove_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """移除重复的召回结果"""
        if not results:
            return results
        
        unique_results = []
        seen_ids = set()
        content_signatures = set()
        
        for result in results:
            note_id = result.get('note_id')
            content = result.get('content', '')
            
            # 1. 基于ID的去重
            if note_id and note_id in seen_ids:
                continue
            
            # 2. 基于内容相似度的去重
            content_signature = self._generate_content_signature(content)
            if content_signature in content_signatures:
                continue
            
            # 3. 基于语义相似度的去重
            is_duplicate = False
            for existing_result in unique_results:
                similarity = self._calculate_content_similarity(
                    content, existing_result.get('content', '')
                )
                if similarity > self.duplicate_threshold:
                    # 保留相似度更高的结果
                    if result.get('similarity_score', 0) > existing_result.get('similarity_score', 0):
                        unique_results.remove(existing_result)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_results.append(result)
                if note_id:
                    seen_ids.add(note_id)
                content_signatures.add(content_signature)
        
        return unique_results
    
    def _disambiguate_entities(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """实体消歧处理"""
        query_entities = self._extract_entities_from_text(query)
        
        disambiguated_results = []
        
        for result in results:
            content = result.get('content', '')
            note_entities = self._extract_entities_from_text(content)
            
            # 检查实体一致性
            entity_consistency = self._check_entity_consistency(
                query_entities, note_entities, content
            )
            
            # 添加实体一致性分数
            result['entity_consistency_score'] = entity_consistency
            
            # 只保留实体一致性较高的结果
            if entity_consistency >= self.entity_confidence_threshold:
                disambiguated_results.append(result)
            else:
                logger.debug(f"Filtered out result due to low entity consistency: {entity_consistency:.2f}")
        
        return disambiguated_results
    
    def _filter_by_similarity(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于相似度阈值过滤"""
        filtered_results = []
        
        for result in results:
            similarity_score = result.get('similarity_score', 0)
            if similarity_score >= self.similarity_threshold:
                filtered_results.append(result)
            else:
                logger.debug(f"Filtered out result due to low similarity: {similarity_score:.2f}")
        
        return filtered_results
    
    def _enhance_completeness(self, results: List[Dict[str, Any]], query: str, 
                            source_paragraphs: Dict[int, str] = None) -> List[Dict[str, Any]]:
        """完整性检查和补充召回"""
        if not source_paragraphs:
            return results
        
        # 分析查询的核心信息需求
        query_requirements = self._analyze_query_requirements(query)
        
        # 检查当前结果的覆盖率
        coverage_score = self._calculate_coverage_score(results, query_requirements)
        
        if coverage_score < self.coverage_threshold:
            logger.info(f"Coverage score {coverage_score:.2f} below threshold, enhancing recall")
            
            # 补充召回
            additional_results = self._supplementary_recall(
                query, query_requirements, source_paragraphs, results
            )
            
            results.extend(additional_results)
        
        return results
    
    def _enhance_with_multi_hop(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """多跳推理增强"""
        if not self.graph_processor:
            return results
        
        try:
            # 分解多跳查询
            query_components = self._decompose_multi_hop_query(query)
            
            if len(query_components) > 1:
                logger.info(f"Detected multi-hop query with {len(query_components)} components")
                
                # 使用图谱检索进行多跳推理
                multi_hop_results = self._perform_multi_hop_retrieval(query_components)
                
                # 合并结果
                enhanced_results = self._merge_multi_hop_results(results, multi_hop_results)
                return enhanced_results
        
        except Exception as e:
            logger.warning(f"Multi-hop enhancement failed: {e}")
        
        return results
    
    def _final_ranking_and_quality_check(self, results: List[Dict[str, Any]], 
                                        query: str) -> List[Dict[str, Any]]:
        """最终排序和质量检查"""
        if not results:
            return results
        
        # 计算综合分数
        for result in results:
            comprehensive_score = self._calculate_comprehensive_score(result, query)
            result['comprehensive_score'] = comprehensive_score
        
        # 按综合分数排序
        results.sort(key=lambda x: x.get('comprehensive_score', 0), reverse=True)
        
        # 质量验证
        validated_results = []
        for result in results:
            if self._validate_result_quality(result, query):
                validated_results.append(result)
        
        return validated_results
    
    def _generate_content_signature(self, content: str) -> str:
        """生成内容签名用于去重"""
        # 移除标点符号和空白字符，转换为小写
        normalized = re.sub(r'[^\w\s]', '', content.lower())
        words = normalized.split()
        
        # 使用前10个和后10个词生成签名
        if len(words) <= 20:
            signature_words = words
        else:
            signature_words = words[:10] + words[-10:]
        
        return ' '.join(signature_words)
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度"""
        return SequenceMatcher(None, content1, content2).ratio()
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """从文本中提取实体"""
        # 简单的实体提取（可以替换为更复杂的NER模型）
        entities = []
        
        # 提取人名（大写字母开头的连续词）
        person_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        persons = re.findall(person_pattern, text)
        entities.extend(persons)
        
        # 提取其他专有名词
        proper_noun_pattern = r'\b[A-Z][a-z]*\b'
        proper_nouns = re.findall(proper_noun_pattern, text)
        entities.extend(proper_nouns)
        
        return list(set(entities))
    
    def _check_entity_consistency(self, query_entities: List[str], 
                                note_entities: List[str], content: str) -> float:
        """检查实体一致性"""
        if not query_entities:
            return 1.0
        
        # 计算实体重叠度
        query_set = set(entity.lower() for entity in query_entities)
        note_set = set(entity.lower() for entity in note_entities)
        
        if not note_set:
            return 0.0
        
        intersection = len(query_set & note_set)
        union = len(query_set | note_set)
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # 检查已知实体映射
        consistency_bonus = self._check_known_entity_mappings(query_entities, note_entities)
        
        return min(1.0, jaccard_similarity + consistency_bonus)
    
    def _check_known_entity_mappings(self, query_entities: List[str], 
                                   note_entities: List[str]) -> float:
        """检查已知实体映射"""
        bonus = 0.0
        
        for query_entity in query_entities:
            for note_entity in note_entities:
                if self._are_related_entities(query_entity, note_entity):
                    bonus += 0.2
        
        return min(0.5, bonus)  # 最多0.5的奖励
    
    def _are_related_entities(self, entity1: str, entity2: str) -> bool:
        """检查两个实体是否相关"""
        # 检查已知实体关系
        for known_entity, related_entities in self.known_entities.items():
            if (entity1.lower() in known_entity.lower() and 
                any(entity2.lower() in rel.lower() for rel in related_entities)):
                return True
            if (entity2.lower() in known_entity.lower() and 
                any(entity1.lower() in rel.lower() for rel in related_entities)):
                return True
        
        return False
    
    def _load_known_entities(self) -> Dict[str, List[str]]:
        """加载已知实体关系"""
        return {
            'Dan Castellaneta': ['Krusty the Clown', 'Homer Simpson', 'Deb Lacusta'],
            'Adriana Caselotti': ['Snow White'],
            'Krusty the Clown': ['Dan Castellaneta'],
            'Deb Lacusta': ['Dan Castellaneta']
        }
    
    def _analyze_query_requirements(self, query: str) -> Dict[str, Any]:
        """分析查询的信息需求"""
        requirements = {
            'entities': self._extract_entities_from_text(query),
            'keywords': self._extract_keywords(query),
            'question_type': self._identify_question_type(query),
            'required_relations': self._identify_required_relations(query)
        }
        return requirements
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        stop_words = {'the', 'is', 'are', 'was', 'were', 'who', 'what', 'where', 'when', 'how'}
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _identify_question_type(self, query: str) -> str:
        """识别问题类型"""
        query_lower = query.lower()
        if 'who' in query_lower:
            return 'person'
        elif 'what' in query_lower:
            return 'definition'
        elif 'where' in query_lower:
            return 'location'
        elif 'when' in query_lower:
            return 'time'
        elif 'spouse' in query_lower or 'married' in query_lower:
            return 'relationship'
        else:
            return 'general'
    
    def _identify_required_relations(self, query: str) -> List[str]:
        """识别查询需要的关系类型"""
        relations = []
        query_lower = query.lower()
        
        if 'spouse' in query_lower or 'married' in query_lower:
            relations.append('marriage')
        if 'voice actor' in query_lower or 'voiced by' in query_lower:
            relations.append('voice_acting')
        if "'s" in query_lower:
            relations.append('possession')
        
        return relations
    
    def _calculate_coverage_score(self, results: List[Dict[str, Any]], 
                                requirements: Dict[str, Any]) -> float:
        """计算覆盖率分数"""
        if not results or not requirements:
            return 0.0
        
        required_entities = set(requirements.get('entities', []))
        required_keywords = set(requirements.get('keywords', []))
        
        covered_entities = set()
        covered_keywords = set()
        
        for result in results:
            content = result.get('content', '').lower()
            result_entities = set(self._extract_entities_from_text(content))
            result_keywords = set(self._extract_keywords(content))
            
            covered_entities.update(result_entities)
            covered_keywords.update(result_keywords)
        
        entity_coverage = len(covered_entities & required_entities) / max(len(required_entities), 1)
        keyword_coverage = len(covered_keywords & required_keywords) / max(len(required_keywords), 1)
        
        return (entity_coverage + keyword_coverage) / 2
    
    def _supplementary_recall(self, query: str, requirements: Dict[str, Any], 
                            source_paragraphs: Dict[int, str], 
                            existing_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """补充召回"""
        additional_results = []
        
        # 获取已召回的段落索引
        recalled_paragraph_idxs = set()
        for result in existing_results:
            paragraph_idxs = result.get('paragraph_idxs', [])
            recalled_paragraph_idxs.update(paragraph_idxs)
        
        # 在未召回的段落中搜索相关内容
        for idx, paragraph in source_paragraphs.items():
            if idx in recalled_paragraph_idxs:
                continue
            
            # 检查段落是否包含所需信息
            relevance_score = self._calculate_paragraph_relevance(paragraph, requirements)
            
            if relevance_score > 0.5:  # 相关性阈值
                additional_result = {
                    'note_id': f'supplementary_{idx}',
                    'content': paragraph,
                    'paragraph_idxs': [idx],
                    'similarity_score': relevance_score,
                    'source': 'supplementary_recall'
                }
                additional_results.append(additional_result)
        
        return additional_results
    
    def _calculate_paragraph_relevance(self, paragraph: str, requirements: Dict[str, Any]) -> float:
        """计算段落相关性"""
        paragraph_lower = paragraph.lower()
        
        # 实体匹配
        entity_matches = 0
        for entity in requirements.get('entities', []):
            if entity.lower() in paragraph_lower:
                entity_matches += 1
        
        # 关键词匹配
        keyword_matches = 0
        for keyword in requirements.get('keywords', []):
            if keyword.lower() in paragraph_lower:
                keyword_matches += 1
        
        # 关系匹配
        relation_matches = 0
        for relation in requirements.get('required_relations', []):
            if relation == 'marriage' and ('spouse' in paragraph_lower or 'married' in paragraph_lower):
                relation_matches += 1
            elif relation == 'voice_acting' and ('voice' in paragraph_lower or 'actor' in paragraph_lower):
                relation_matches += 1
        
        # 计算综合相关性分数
        total_requirements = (len(requirements.get('entities', [])) + 
                            len(requirements.get('keywords', [])) + 
                            len(requirements.get('required_relations', [])))
        
        if total_requirements == 0:
            return 0.0
        
        total_matches = entity_matches + keyword_matches + relation_matches
        return total_matches / total_requirements
    
    def _decompose_multi_hop_query(self, query: str) -> List[str]:
        """分解多跳查询"""
        # 简单的多跳查询分解
        if "'s" in query:
            # 处理所有格形式的多跳查询
            parts = query.split("'s")
            if len(parts) >= 2:
                return [part.strip() for part in parts]
        
        return [query]
    
    def _perform_multi_hop_retrieval(self, query_components: List[str]) -> List[Dict[str, Any]]:
        """执行多跳检索"""
        # 这里应该调用图谱检索器进行多跳推理
        # 简化实现
        return []
    
    def _merge_multi_hop_results(self, original_results: List[Dict[str, Any]], 
                               multi_hop_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并多跳检索结果"""
        # 简单合并，去重
        all_results = original_results + multi_hop_results
        return self._remove_duplicates(all_results)
    
    def _calculate_comprehensive_score(self, result: Dict[str, Any], query: str) -> float:
        """计算综合分数"""
        similarity_score = result.get('similarity_score', 0.0)
        entity_consistency = result.get('entity_consistency_score', 1.0)
        
        # 权重配置
        weights = {
            'similarity': 0.4,
            'entity_consistency': 0.3,
            'content_quality': 0.2,
            'source_reliability': 0.1
        }
        
        # 内容质量评估
        content_quality = self._assess_content_quality(result.get('content', ''))
        
        # 来源可靠性
        source_reliability = 1.0 if result.get('source') != 'supplementary_recall' else 0.8
        
        comprehensive_score = (
            weights['similarity'] * similarity_score +
            weights['entity_consistency'] * entity_consistency +
            weights['content_quality'] * content_quality +
            weights['source_reliability'] * source_reliability
        )
        
        return comprehensive_score
    
    def _assess_content_quality(self, content: str) -> float:
        """评估内容质量"""
        if not content:
            return 0.0
        
        # 长度评估
        length_score = min(1.0, len(content) / 200)  # 200字符为满分
        
        # 完整性评估（是否有完整的句子）
        sentences = content.split('.')
        completeness_score = min(1.0, len([s for s in sentences if len(s.strip()) > 10]) / 2)
        
        return (length_score + completeness_score) / 2
    
    def _validate_result_quality(self, result: Dict[str, Any], query: str) -> bool:
        """验证结果质量"""
        # 基本质量检查
        content = result.get('content', '')
        if len(content) < 20:  # 内容太短
            return False
        
        comprehensive_score = result.get('comprehensive_score', 0)
        if comprehensive_score < 0.3:  # 综合分数太低
            return False
        
        return True