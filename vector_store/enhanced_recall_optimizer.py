#!/usr/bin/env python3
"""
增强召回优化器
实现去重、实体消歧、相似度过滤、完整性增强和多跳检索功能
"""

import re
import hashlib
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
from loguru import logger
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import config

try:
    # Optional import used for isinstance checks
    from graph.multi_hop_query_processor import MultiHopQueryProcessor
except Exception:  # pragma: no cover - optional dependency
    MultiHopQueryProcessor = None

class EnhancedRecallOptimizer:
    """增强召回优化器"""
    
    def __init__(self, vector_retriever=None, graph_retriever=None):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        
        # 从配置加载参数
        self.config = config.get('vector_store.recall_optimization', {})
        
        # 去重配置
        self.dedup_config = self.config.get('deduplication', {})
        self.content_similarity_threshold = self.dedup_config.get('content_similarity_threshold', 0.85)
        self.signature_similarity_threshold = self.dedup_config.get('signature_similarity_threshold', 0.9)
        
        # 实体消歧配置
        self.entity_config = self.config.get('entity_disambiguation', {})
        self.entity_similarity_threshold = self.entity_config.get('entity_similarity_threshold', 0.8)
        self.known_entities = self.entity_config.get('known_entities', {})
        
        # 相似度过滤配置
        self.similarity_config = self.config.get('similarity_filtering', {})
        self.min_similarity_threshold = self.similarity_config.get('min_similarity_threshold', 0.3)
        self.adaptive_threshold = self.similarity_config.get('adaptive_threshold', True)
        
        # 完整性增强配置
        self.completeness_config = self.config.get('completeness_enhancement', {})
        self.coverage_threshold = self.completeness_config.get('coverage_threshold', 0.7)
        self.max_additional_notes = self.completeness_config.get('max_additional_notes', 10)
        
        # 多跳检索配置
        self.multi_hop_config = self.config.get('multi_hop_retrieval', {})
        self.max_hops = self.multi_hop_config.get('max_hops', 2)
        self.hop_similarity_threshold = self.multi_hop_config.get('hop_similarity_threshold', 0.6)
        
        # 初始化TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
    def optimize_recall(self, initial_results: List[Dict[str, Any]], 
                       query: str, 
                       query_embedding: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """优化召回结果"""
        if not initial_results:
            return initial_results
            
        logger.info(f"开始优化召回结果，初始数量: {len(initial_results)}")
        
        # 1. 去重
        if self.dedup_config.get('enabled', True):
            results = self._remove_duplicates(initial_results)
            logger.info(f"去重后数量: {len(results)}")
        else:
            results = initial_results
            
        # 2. 实体消歧
        if self.entity_config.get('enabled', True):
            results = self._disambiguate_entities(results, query)
            logger.info(f"实体消歧后数量: {len(results)}")
            
        # 3. 相似度过滤
        if self.similarity_config.get('enabled', True) and query_embedding is not None:
            results = self._filter_by_similarity(results, query_embedding)
            logger.info(f"相似度过滤后数量: {len(results)}")
            
        # 4. 完整性增强
        if self.completeness_config.get('enabled', True):
            results = self._enhance_completeness(results, query, query_embedding)
            logger.info(f"完整性增强后数量: {len(results)}")
            
        # 5. 多跳检索增强
        if self.multi_hop_config.get('enabled', True) and self.graph_retriever:
            results = self._enhance_with_multi_hop(results, query)
            logger.info(f"多跳增强后数量: {len(results)}")
            
        # 6. 最终排序和质量检查
        results = self._final_ranking_and_quality_check(results, query, query_embedding)
        
        logger.info(f"优化完成，最终数量: {len(results)}")
        return results
        
    def _remove_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去除重复结果"""
        seen_ids = set()
        seen_signatures = set()
        unique_results = []
        
        for result in results:
            note_id = result.get('note_id')
            content = result.get('content', '')
            
            # 基于ID的去重
            if note_id in seen_ids:
                continue
                
            # 基于内容签名的去重
            content_signature = self._generate_content_signature(content)
            
            # 检查是否与已有内容过于相似
            is_duplicate = False
            for existing_sig in seen_signatures:
                similarity = self._calculate_signature_similarity(content_signature, existing_sig)
                if similarity > self.signature_similarity_threshold:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                seen_ids.add(note_id)
                seen_signatures.add(content_signature)
                result['optimization_info'] = result.get('optimization_info', {}) 
                result['optimization_info']['deduplication'] = 'kept'
                unique_results.append(result)
            else:
                logger.debug(f"移除重复内容: {note_id}")
                
        return unique_results
        
    def _generate_content_signature(self, content: str) -> str:
        """生成内容签名"""
        # 标准化文本
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        # 移除标点符号
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # 生成哈希
        return hashlib.md5(normalized.encode()).hexdigest()
        
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """计算签名相似度"""
        if sig1 == sig2:
            return 1.0
        # 简单的字符级相似度
        common_chars = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return common_chars / max(len(sig1), len(sig2))
        
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度"""
        try:
            # 使用TF-IDF计算相似度
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([content1, content2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception as e:
            logger.warning(f"计算内容相似度失败: {e}")
            return 0.0
            
    def _disambiguate_entities(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """实体消歧"""
        # 提取查询中的实体
        query_entities = self._extract_entities(query)
        
        disambiguated_results = []
        entity_groups = defaultdict(list)
        
        # 按实体分组
        for result in results:
            content = result.get('content', '')
            note_entities = self._extract_entities(content)
            
            # 检查实体一致性
            is_consistent = self._check_entity_consistency(note_entities, query_entities)
            
            if is_consistent:
                # 映射到已知实体
                mapped_entities = self._map_to_known_entities(note_entities)
                entity_key = tuple(sorted(mapped_entities))
                entity_groups[entity_key].append(result)
            else:
                logger.debug(f"实体不一致，移除笔记: {result.get('note_id')}")
                
        # 从每个实体组中选择最佳结果
        for entity_key, group_results in entity_groups.items():
            if len(group_results) == 1:
                result = group_results[0]
                result['optimization_info'] = result.get('optimization_info', {})
                result['optimization_info']['entity_disambiguation'] = 'single_match'
                disambiguated_results.append(result)
            else:
                # 选择最相关的结果
                best_result = max(group_results, key=lambda x: x.get('similarity_score', 0))
                best_result['optimization_info'] = best_result.get('optimization_info', {})
                best_result['optimization_info']['entity_disambiguation'] = f'best_of_{len(group_results)}'
                disambiguated_results.append(best_result)
                
        return disambiguated_results
        
    def _extract_entities(self, text: str) -> Set[str]:
        """提取实体"""
        entities = set()
        
        # 简单的人名提取（大写字母开头的连续单词）
        person_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        persons = re.findall(person_pattern, text)
        entities.update(persons)
        
        # 提取已知实体
        for known_entity in self.known_entities.keys():
            if known_entity.lower() in text.lower():
                entities.add(known_entity)
                
        return entities
        
    def _check_entity_consistency(self, note_entities: Set[str], query_entities: Set[str]) -> bool:
        """检查实体一致性"""
        if not query_entities:
            return True
            
        # 检查是否有共同实体或相关实体
        for note_entity in note_entities:
            for query_entity in query_entities:
                if self._are_entities_related(note_entity, query_entity):
                    return True
                    
        return len(note_entities & query_entities) > 0
        
    def _are_entities_related(self, entity1: str, entity2: str) -> bool:
        """检查两个实体是否相关"""
        # 检查已知关系
        for known_entity, related_entities in self.known_entities.items():
            if (entity1 == known_entity and entity2 in related_entities) or \
               (entity2 == known_entity and entity1 in related_entities):
                return True
                
        # 简单的字符串相似度检查
        similarity = self._calculate_string_similarity(entity1.lower(), entity2.lower())
        return similarity > self.entity_similarity_threshold
        
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似度"""
        if str1 == str2:
            return 1.0
            
        # 简单的编辑距离相似度
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
            
        # 计算公共子序列长度
        common_chars = sum(1 for a, b in zip(str1, str2) if a == b)
        return common_chars / max_len
        
    def _map_to_known_entities(self, entities: Set[str]) -> Set[str]:
        """映射到已知实体"""
        mapped = set()
        
        for entity in entities:
            # 检查是否是已知实体
            if entity in self.known_entities:
                mapped.add(entity)
            else:
                # 尝试映射到最相似的已知实体
                best_match = None
                best_similarity = 0
                
                for known_entity in self.known_entities.keys():
                    similarity = self._calculate_string_similarity(entity.lower(), known_entity.lower())
                    if similarity > best_similarity and similarity > self.entity_similarity_threshold:
                        best_similarity = similarity
                        best_match = known_entity
                        
                if best_match:
                    mapped.add(best_match)
                else:
                    mapped.add(entity)
                    
        return mapped
        
    def _filter_by_similarity(self, results: List[Dict[str, Any]], 
                             query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """基于相似度过滤"""
        if not hasattr(self.vector_retriever, 'embedding_manager'):
            return results
            
        filtered_results = []
        similarities = []
        
        # 计算所有结果的相似度
        for result in results:
            similarity = result.get('similarity_score', 0)
            similarities.append(similarity)
            
        # 自适应阈值
        if self.adaptive_threshold and similarities:
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            adaptive_threshold = max(self.min_similarity_threshold, 
                                   mean_similarity - 0.5 * std_similarity)
        else:
            adaptive_threshold = self.min_similarity_threshold
            
        # 过滤结果
        for result, similarity in zip(results, similarities):
            if similarity >= adaptive_threshold:
                result['optimization_info'] = result.get('optimization_info', {})
                result['optimization_info']['similarity_filter'] = f'passed_{similarity:.3f}'
                filtered_results.append(result)
            else:
                logger.debug(f"相似度过低，移除笔记: {result.get('note_id')} (similarity: {similarity:.3f})")
                
        return filtered_results
        
    def _enhance_completeness(self, results: List[Dict[str, Any]], 
                             query: str, 
                             query_embedding: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        """增强完整性"""
        if not self.vector_retriever:
            return results
            
        # 分析查询需求
        query_requirements = self._analyze_query_requirements(query)
        
        # 计算当前覆盖率
        current_coverage = self._calculate_coverage(results, query_requirements)
        
        if current_coverage >= self.coverage_threshold:
            logger.info(f"当前覆盖率已满足要求: {current_coverage:.3f}")
            return results
            
        # 补充召回
        additional_results = self._supplement_recall(results, query, query_requirements)
        
        # 合并结果
        enhanced_results = results + additional_results[:self.max_additional_notes]
        
        return enhanced_results
        
    def _analyze_query_requirements(self, query: str) -> Dict[str, Any]:
        """分析查询需求"""
        requirements = {
            'entities': self._extract_entities(query),
            'keywords': self._extract_keywords(query),
            'question_type': self._identify_question_type(query)
        }
        return requirements
        
    def _extract_keywords(self, text: str) -> Set[str]:
        """提取关键词"""
        # 简单的关键词提取
        words = re.findall(r'\b\w+\b', text.lower())
        # 过滤停用词
        stop_words = {'the', 'is', 'at', 'which', 'on', 'who', 'what', 'where', 'when', 'how', 'of', 'a', 'an'}
        keywords = {word for word in words if word not in stop_words and len(word) > 2}
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
        elif 'how' in query_lower:
            return 'method'
        else:
            return 'general'
            
    def _calculate_coverage(self, results: List[Dict[str, Any]], 
                           requirements: Dict[str, Any]) -> float:
        """计算覆盖率"""
        if not requirements['entities'] and not requirements['keywords']:
            return 1.0
            
        total_requirements = len(requirements['entities']) + len(requirements['keywords'])
        covered_requirements = 0
        
        # 收集所有结果的内容
        all_content = ' '.join(result.get('content', '') for result in results).lower()
        
        # 检查实体覆盖
        for entity in requirements['entities']:
            if entity.lower() in all_content:
                covered_requirements += 1
                
        # 检查关键词覆盖
        for keyword in requirements['keywords']:
            if keyword in all_content:
                covered_requirements += 1
                
        return covered_requirements / total_requirements if total_requirements > 0 else 1.0
        
    def _supplement_recall(self, existing_results: List[Dict[str, Any]], 
                          query: str, 
                          requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """补充召回"""
        if not self.vector_retriever:
            return []
            
        # 构建补充查询
        supplement_queries = self._build_supplement_queries(requirements)
        
        additional_results = []
        existing_ids = {result.get('note_id') for result in existing_results}
        
        for supp_query in supplement_queries:
            try:
                supp_results = self.vector_retriever.search([supp_query], top_k=5)
                for result in supp_results[0] if supp_results else []:
                    if result.get('note_id') not in existing_ids:
                        result['optimization_info'] = result.get('optimization_info', {})
                        result['optimization_info']['supplement_recall'] = supp_query
                        additional_results.append(result)
                        existing_ids.add(result.get('note_id'))
            except Exception as e:
                logger.warning(f"补充召回失败: {e}")
                
        return additional_results
        
    def _build_supplement_queries(self, requirements: Dict[str, Any]) -> List[str]:
        """构建补充查询"""
        queries = []
        
        # 基于实体的查询
        for entity in requirements['entities']:
            queries.append(entity)
            
        # 基于关键词的查询
        if len(requirements['keywords']) > 1:
            keyword_combinations = list(requirements['keywords'])[:3]  # 限制数量
            queries.extend(keyword_combinations)
            
        return queries[:5]  # 限制查询数量
        
    def _enhance_with_multi_hop(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """多跳检索增强"""
        if not self.graph_retriever:
            return results
            
        # 多跳查询分解
        hop_queries = self._decompose_multi_hop_query(query)
        
        enhanced_results = results.copy()
        existing_ids = {result.get('note_id') for result in results}
        
        # 执行多跳检索
        for hop_query in hop_queries:
            hop_results = self._execute_multi_hop_retrieval(hop_query, existing_ids)
            for result in hop_results:
                if result.get('note_id') not in existing_ids:
                    result['optimization_info'] = result.get('optimization_info', {})
                    result['optimization_info']['multi_hop'] = hop_query
                    enhanced_results.append(result)
                    existing_ids.add(result.get('note_id'))
                    
        return enhanced_results
        
    def _decompose_multi_hop_query(self, query: str) -> List[str]:
        """分解多跳查询"""
        # 简单的查询分解策略
        entities = self._extract_entities(query)
        keywords = self._extract_keywords(query)
        
        hop_queries = []
        
        # 基于实体的跳跃
        for entity in entities:
            hop_queries.append(f"related to {entity}")
            
        # 基于关系的跳跃
        if 'spouse' in query.lower():
            hop_queries.append("marriage relationship")
            hop_queries.append("family connection")
            
        return hop_queries[:self.max_hops]
        
    def _execute_multi_hop_retrieval(self, hop_query: str, existing_ids: Set[str]) -> List[Dict[str, Any]]:
        """执行多跳检索"""
        try:
            hop_results: List[Dict[str, Any]] = []

            hop_embedding = None
            if hasattr(self.vector_retriever, "embedding_manager"):
                try:
                    embeddings = self.vector_retriever.embedding_manager.encode_queries([hop_query])
                    if isinstance(embeddings, list):
                        hop_embedding = embeddings[0]
                    elif getattr(embeddings, "shape", None) is not None:
                        hop_embedding = embeddings[0]
                except Exception as exc:  # pragma: no cover - encoding failure
                    logger.warning(f"生成多跳查询嵌入失败: {exc}")

            if self.graph_retriever and hop_embedding is not None:
                if MultiHopQueryProcessor and isinstance(self.graph_retriever, MultiHopQueryProcessor):
                    try:
                        result = self.graph_retriever.retrieve(hop_embedding)
                        hop_results = result.get("notes", []) if isinstance(result, dict) else []
                    except Exception as exc:
                        logger.warning(f"图检索器retrieve调用失败: {exc}")
                elif hasattr(self.graph_retriever, "retrieve_with_reasoning_paths"):
                    try:
                        hop_results = self.graph_retriever.retrieve_with_reasoning_paths(hop_embedding)
                    except Exception as exc:
                        logger.warning(f"图检索器retrieve_with_reasoning_paths调用失败: {exc}")

            if not hop_results:
                # 回退到向量检索
                hop_results = self.vector_retriever.search([hop_query], top_k=3)
                hop_results = hop_results[0] if hop_results else []

            # 过滤已存在的结果
            filtered_results: List[Dict[str, Any]] = []
            for result in hop_results:
                note_id = result.get("note_id")
                if note_id and note_id not in existing_ids:
                    similarity = result.get(
                        "similarity_score",
                        result.get("retrieval_info", {}).get("similarity", 0),
                    )
                    if similarity >= self.hop_similarity_threshold:
                        filtered_results.append(result)

            return filtered_results

        except Exception as e:  # pragma: no cover - unexpected failure
            logger.warning(f"多跳检索失败: {e}")
            return []
            
    def _final_ranking_and_quality_check(self, results: List[Dict[str, Any]], 
                                         query: str, 
                                         query_embedding: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        """最终排序和质量检查"""
        # 计算综合分数
        for result in results:
            score = self._calculate_comprehensive_score(result, query, query_embedding)
            result['comprehensive_score'] = score
            
        # 按综合分数排序
        results.sort(key=lambda x: x.get('comprehensive_score', 0), reverse=True)
        
        # 质量检查
        quality_results = []
        for result in results:
            if self._passes_quality_check(result, query):
                quality_results.append(result)
            else:
                logger.debug(f"质量检查未通过: {result.get('note_id')}")
                
        return quality_results
        
    def _calculate_comprehensive_score(self, result: Dict[str, Any], 
                                      query: str, 
                                      query_embedding: Optional[np.ndarray]) -> float:
        """计算综合分数"""
        # 基础相似度分数
        base_score = result.get('similarity_score', 0)
        
        # 内容质量分数
        content = result.get('content', '')
        quality_score = self._assess_content_quality(content, query)
        
        # 优化信息奖励
        optimization_bonus = 0
        opt_info = result.get('optimization_info', {})
        if 'supplement_recall' in opt_info:
            optimization_bonus += 0.1
        if 'multi_hop' in opt_info:
            optimization_bonus += 0.15
            
        # 综合分数
        comprehensive_score = base_score * 0.6 + quality_score * 0.3 + optimization_bonus
        
        return comprehensive_score
        
    def _assess_content_quality(self, content: str, query: str) -> float:
        """评估内容质量"""
        if not content:
            return 0.0
            
        # 长度分数
        length_score = min(len(content) / 200, 1.0)  # 200字符为满分
        
        # 关键词匹配分数
        query_keywords = self._extract_keywords(query)
        content_lower = content.lower()
        matched_keywords = sum(1 for kw in query_keywords if kw in content_lower)
        keyword_score = matched_keywords / len(query_keywords) if query_keywords else 0
        
        # 实体匹配分数
        query_entities = self._extract_entities(query)
        matched_entities = sum(1 for entity in query_entities if entity.lower() in content_lower)
        entity_score = matched_entities / len(query_entities) if query_entities else 0
        
        # 综合质量分数
        quality_score = (length_score * 0.3 + keyword_score * 0.4 + entity_score * 0.3)
        
        return quality_score
        
    def _passes_quality_check(self, result: Dict[str, Any], query: str) -> bool:
        """质量检查"""
        content = result.get('content', '')
        
        # 最小长度检查
        if len(content.strip()) < 5:
            return False
            
        # 相似度检查 - 放宽阈值
        similarity = result.get('similarity_score', 0)
        if similarity == 0:
            # 尝试从retrieval_info中获取相似度
            retrieval_info = result.get('retrieval_info', {})
            similarity = retrieval_info.get('similarity', 0)
        if similarity < 0.05:  # 降低最低相似度阈值
            return False
            
        # 综合分数检查 - 放宽阈值
        comprehensive_score = result.get('comprehensive_score', 0)
        if comprehensive_score < 0.1:  # 降低最低综合分数阈值
            return False
            
        return True