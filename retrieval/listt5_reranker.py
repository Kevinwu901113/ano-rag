#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ListT5 重排序器 - 基于T5的列表式重排序模型

使用预训练的T5模型对候选文档进行重排序，支持批量处理和温度缩放。
"""

import logging
import torch
from typing import List, Dict, Any, Optional, Tuple
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from torch.nn.functional import softmax

logger = logging.getLogger(__name__)

class ListT5Reranker:
    """
    基于T5的列表式重排序器
    
    使用预训练的T5模型对候选文档进行重排序，支持:
    - 批量处理
    - 温度缩放
    - GPU加速
    - 文本截断和优化
    """
    
    def __init__(self, 
                 model_name: str = "castorini/doc2query-t5-large-list",
                 max_seq_len: int = 2048,
                 temperature: float = 1.0,
                 batch_size: int = 4,
                 device: Optional[str] = None):
        """
        初始化ListT5重排序器
        
        Args:
            model_name: T5模型名称
            max_seq_len: 最大序列长度
            temperature: 温度缩放参数
            batch_size: 批处理大小
            device: 设备类型 (cuda/cpu)
        """
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.batch_size = batch_size
        
        # 设备选择
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"初始化ListT5重排序器: {model_name}, 设备: {self.device}")
        
        # 加载模型和分词器
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"成功加载ListT5模型: {model_name}")
        except Exception as e:
            logger.error(f"加载ListT5模型失败: {e}")
            raise
    
    def _prepare_input_text(self, question: str, candidates: List[Dict[str, Any]]) -> str:
        """
        准备输入文本，格式化为ListT5期望的格式
        
        Args:
            question: 查询问题
            candidates: 候选文档列表
            
        Returns:
            格式化的输入文本
        """
        # 构建输入格式: Query: {question} Documents: [1] {title} {text} [2] {title} {text} ...
        input_parts = [f"Query: {question}", "Documents:"]
        
        for i, candidate in enumerate(candidates, 1):
            title = candidate.get('title', '').strip()
            text = candidate.get('text', '').strip()
            
            # 文本截断 - 取标题和第一句话
            if text:
                # 简单的句子分割
                sentences = text.split('. ')
                first_sentence = sentences[0] if sentences else text[:200]
            else:
                first_sentence = ""
            
            # 组合标题和首句
            doc_text = f"{title} {first_sentence}".strip()
            
            # 进一步截断以控制长度
            if len(doc_text) > 300:
                doc_text = doc_text[:300] + "..."
            
            input_parts.append(f"[{i}] {doc_text}")
        
        return " ".join(input_parts)
    
    def _extract_scores(self, generated_text: str, num_candidates: int) -> List[float]:
        """
        从生成的文本中提取排序分数
        
        Args:
            generated_text: 模型生成的文本
            num_candidates: 候选文档数量
            
        Returns:
            排序分数列表
        """
        try:
            # 简单的分数提取逻辑 - 这里需要根据具体的ListT5模型输出格式调整
            # 假设模型输出类似 "[1] [3] [2] ..." 的排序
            scores = [0.0] * num_candidates
            
            # 解析排序结果
            import re
            matches = re.findall(r'\[(\d+)\]', generated_text)
            
            if matches:
                # 根据排序位置分配分数
                for rank, doc_id in enumerate(matches):
                    try:
                        idx = int(doc_id) - 1  # 转换为0-based索引
                        if 0 <= idx < num_candidates:
                            # 分数递减，排在前面的分数更高
                            scores[idx] = (num_candidates - rank) / num_candidates
                    except (ValueError, IndexError):
                        continue
            else:
                # 如果解析失败，返回均匀分数
                logger.warning(f"无法解析ListT5输出: {generated_text}")
                scores = [1.0 / num_candidates] * num_candidates
            
            return scores
            
        except Exception as e:
            logger.error(f"提取分数失败: {e}")
            # 返回均匀分数作为fallback
            return [1.0 / num_candidates] * num_candidates
    
    def score(self, question: str, candidates: List[Dict[str, Any]]) -> List[float]:
        """
        对候选文档进行评分
        
        Args:
            question: 查询问题
            candidates: 候选文档列表
            
        Returns:
            每个候选文档的分数列表
        """
        if not candidates:
            return []
        
        try:
            # 准备输入文本
            input_text = self._prepare_input_text(question, candidates)
            
            # 分词
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_seq_len,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # 生成输出
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=100,  # 输出长度限制
                    num_beams=1,
                    do_sample=False,
                    temperature=self.temperature
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"ListT5输出: {generated_text}")
            
            # 提取分数
            scores = self._extract_scores(generated_text, len(candidates))
            
            # 温度缩放和归一化
            if self.temperature != 1.0:
                scores = np.array(scores)
                scores = scores / self.temperature
                scores = softmax(torch.tensor(scores), dim=0).numpy().tolist()
            
            logger.debug(f"ListT5分数: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"ListT5评分失败: {e}")
            # 返回均匀分数作为fallback
            return [1.0 / len(candidates)] * len(candidates)
    
    def batch_score(self, questions: List[str], candidates_list: List[List[Dict[str, Any]]]) -> List[List[float]]:
        """
        批量评分
        
        Args:
            questions: 查询问题列表
            candidates_list: 候选文档列表的列表
            
        Returns:
            每个查询的分数列表
        """
        results = []
        
        for i in range(0, len(questions), self.batch_size):
            batch_questions = questions[i:i + self.batch_size]
            batch_candidates = candidates_list[i:i + self.batch_size]
            
            batch_results = []
            for question, candidates in zip(batch_questions, batch_candidates):
                scores = self.score(question, candidates)
                batch_results.append(scores)
            
            results.extend(batch_results)
        
        return results


def create_listt5_reranker(config) -> ListT5Reranker:
    """
    创建ListT5重排序器实例
    
    Args:
        config: 配置对象
        
    Returns:
        ListT5Reranker实例
    """
    rerank_config = getattr(config, 'rerank', {})
    calibration_config = getattr(config, 'calibration', {})
    
    return ListT5Reranker(
        model_name=getattr(rerank_config, 'listt5_model', "castorini/doc2query-t5-large-list"),
        max_seq_len=getattr(rerank_config, 'max_seq_len', 2048),
        temperature=getattr(calibration_config, 'listt5_temperature', 1.0),
        batch_size=getattr(rerank_config, 'batch_size', 4)
    )


def fuse_scores(candidates: List[Dict[str, Any]], 
                list_scores: List[float], 
                weights: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    融合ListT5分数与现有分数
    
    Args:
        candidates: 候选文档列表
        list_scores: ListT5分数
        weights: 融合权重配置
        
    Returns:
        融合后的候选文档列表
    """
    if len(candidates) != len(list_scores):
        logger.warning(f"候选数量({len(candidates)})与分数数量({len(list_scores)})不匹配")
        return candidates
    
    # 获取融合权重
    listt5_weight = weights.get('listt5_weight', 0.35)
    learned_fusion_weight = weights.get('learned_fusion_weight', 0.0)
    atomic_features_weight = weights.get('atomic_features_weight', 0.1)
    
    # 获取原子特征的详细权重配置
    atomic_features_config = weights.get('atomic_features', {})
    fact_count_weight = atomic_features_config.get('hit_fact_count_weight', 0.3)
    importance_weight = atomic_features_config.get('avg_importance_weight', 0.25)
    predicate_coverage_weight = atomic_features_config.get('predicate_coverage_weight', 0.2)
    temporal_coverage_weight = atomic_features_config.get('temporal_coverage_weight', 0.15)
    diversity_weight = atomic_features_config.get('cross_sentence_diversity_weight', 0.1)
    
    # 记录融合权重详情，便于调试和A/B测试
    logger.debug(f"Score fusion details - ListT5: {listt5_weight:.3f}, "
                f"Learned Fusion: {learned_fusion_weight:.3f}, "
                f"Atomic Features: {atomic_features_weight:.3f}")
    logger.debug(f"Atomic sub-weights - Facts: {fact_count_weight:.2f}, "
                f"Importance: {importance_weight:.2f}, Predicates: {predicate_coverage_weight:.2f}, "
                f"Temporal: {temporal_coverage_weight:.2f}, Diversity: {diversity_weight:.2f}")
    
    # 归一化现有分数
    existing_scores = []
    learned_fusion_scores = []
    atomic_scores = []
    
    for candidate in candidates:
        # 优先使用cross-encoder分数，否则使用fusion_score
        if 'ce_score' in candidate:
            existing_scores.append(candidate['ce_score'])
        elif 'fusion_score' in candidate:
            existing_scores.append(candidate['fusion_score'])
        else:
            # fallback到dense分数
            existing_scores.append(candidate.get('dense', 0.0))
        
        # 获取 learned_fusion 分数
        learned_fusion_scores.append(candidate.get('learned_fusion_score', 0.0))
        
        # 计算原子笔记特征综合分数（直接从候选对象获取）
        hit_fact_count = candidate.get('hit_fact_count', 0)
        avg_importance = candidate.get('avg_importance', 0.0)
        predicate_coverage = candidate.get('predicate_coverage', 0.0)
        temporal_coverage = candidate.get('temporal_coverage', 0.0)
        cross_sentence_diversity = candidate.get('cross_sentence_diversity', 0.0)
        
        # 综合原子特征分数 (使用配置的权重)
        atomic_score = (fact_count_weight * min(hit_fact_count / 5.0, 1.0) +  # 命中事实数归一化
                       importance_weight * avg_importance +  # 平均重要性
                       predicate_coverage_weight * predicate_coverage +  # 谓词覆盖度
                       temporal_coverage_weight * temporal_coverage +  # 时间覆盖度
                       diversity_weight * cross_sentence_diversity)  # 跨句多样性
        atomic_scores.append(atomic_score)
    
    # Z-score归一化
    def normalize_scores(scores):
        scores = np.array(scores)
        if np.std(scores) > 0:
            return ((scores - np.mean(scores)) / np.std(scores)).tolist()
        else:
            return scores.tolist()
    
    norm_existing = normalize_scores(existing_scores)
    norm_listt5 = normalize_scores(list_scores)
    norm_learned_fusion = normalize_scores(learned_fusion_scores)
    norm_atomic = normalize_scores(atomic_scores)
    
    # 融合分数
    fused_candidates = []
    for i, candidate in enumerate(candidates):
        fused_candidate = candidate.copy()
        
        # 计算基础融合分数
        base_fused_score = (listt5_weight * norm_listt5[i] + 
                           (1 - listt5_weight) * norm_existing[i])
        
        # 集成原子笔记特征
        if atomic_features_weight > 0:
            base_fused_score = ((1 - atomic_features_weight) * base_fused_score + 
                               atomic_features_weight * norm_atomic[i])
        
        # 如果有 learned_fusion 权重，进一步融合
        if learned_fusion_weight > 0:
            fused_score = ((1 - learned_fusion_weight) * base_fused_score + 
                          learned_fusion_weight * norm_learned_fusion[i])
        else:
            fused_score = base_fused_score
        
        fused_candidate['listt5_score'] = list_scores[i]
        fused_candidate['atomic_score'] = atomic_scores[i]
        fused_candidate['fused_score'] = fused_score
        fused_candidates.append(fused_candidate)
    
    return fused_candidates


def sort_desc(candidates: List[Dict[str, Any]], score_key: str = 'fused_score') -> List[Dict[str, Any]]:
    """
    按分数降序排序候选文档
    
    Args:
        candidates: 候选文档列表
        score_key: 用于排序的分数键
        
    Returns:
        排序后的候选文档列表
    """
    return sorted(candidates, key=lambda x: x.get(score_key, 0.0), reverse=True)