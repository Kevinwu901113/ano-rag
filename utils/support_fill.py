import re
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
from loguru import logger
from utils.k_estimator import estimate_required_k
from utils.text_utils import TextUtils


class SupportFiller:
    """支持段落补齐器
    
    在写盘前对LLM输出的support_idxs做"只修idx、不改答案"的结构化补齐/纠偏
    """
    
    def __init__(self):
        """初始化支持段落补齐器"""
        pass
    
    def extract_entities_from_text(self, text: str) -> Set[str]:
        """从文本中抽取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体集合
        """
        entities = set()
        
        # 使用正则提取实体
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # 大写短语
            r'\b\d{4}\b',  # 年份
            r'\b\d+(?:\.\d+)?\b',  # 数字
            r'\b[A-Z]{2,}\b',  # 缩写
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.update(matches)
        
        # 使用TextUtils的实体提取功能
        text_entities = TextUtils.extract_entities(text, confidence_threshold=0.3)
        entities.update(text_entities)
        
        # 过滤和清理
        filtered_entities = set()
        for entity in entities:
            entity = entity.strip()
            if len(entity) >= 2 and not entity.lower() in {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}:
                filtered_entities.add(entity)
        
        return filtered_entities
    
    def find_answer_containing_passages(self, answer: str, passages_by_idx: Dict[int, str]) -> List[int]:
        """找到包含答案子串的段落
        
        Args:
            answer: 答案文本
            passages_by_idx: 段落索引到内容的映射
            
        Returns:
            包含答案的段落索引列表
        """
        if not answer or not answer.strip():
            return []
        
        answer_clean = answer.strip().lower()
        containing_passages = []
        
        # 精确匹配
        for idx, passage in passages_by_idx.items():
            if answer_clean in passage.lower():
                containing_passages.append(idx)
        
        # 如果没有精确匹配，尝试部分匹配
        if not containing_passages:
            answer_words = set(re.findall(r'\b\w+\b', answer_clean))
            answer_words = {w for w in answer_words if len(w) > 2}  # 过滤短词
            
            if answer_words:
                candidates = []
                for idx, passage in passages_by_idx.items():
                    passage_words = set(re.findall(r'\b\w+\b', passage.lower()))
                    matched_words = answer_words & passage_words
                    match_ratio = len(matched_words) / len(answer_words) if answer_words else 0
                    
                    if match_ratio > 0.5:  # 至少50%匹配
                        candidates.append((idx, match_ratio))
                
                # 按匹配度排序，取最好的
                candidates.sort(key=lambda x: x[1], reverse=True)
                containing_passages = [idx for idx, _ in candidates[:3]]  # 最多取3个
        
        return containing_passages
    
    def calculate_entity_overlap(self, text1: str, text2: str) -> float:
        """计算两个文本的实体重合度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            重合度分数 [0, 1]
        """
        entities1 = self.extract_entities_from_text(text1)
        entities2 = self.extract_entities_from_text(text2)
        
        if not entities1 and not entities2:
            return 0.0
        
        if not entities1 or not entities2:
            return 0.0
        
        intersection = entities1 & entities2
        union = entities1 | entities2
        
        return len(intersection) / len(union) if union else 0.0
    
    def find_bridging_passages(self, question: str, answer_passage_content: str, 
                             passages_by_idx: Dict[int, str], packed_order: List[int],
                             exclude_idxs: Set[int], target_count: int) -> List[int]:
        """找到桥接段落
        
        Args:
            question: 问题文本
            answer_passage_content: 答案段落内容
            passages_by_idx: 段落索引到内容的映射
            packed_order: 段落在prompt中的顺序
            exclude_idxs: 要排除的段落索引
            target_count: 目标桥接段落数量
            
        Returns:
            桥接段落索引列表
        """
        candidates = []
        
        for idx in packed_order:
            if idx in exclude_idxs or idx not in passages_by_idx:
                continue
            
            passage_content = passages_by_idx[idx]
            
            # 计算与问题的实体重合度
            question_overlap = self.calculate_entity_overlap(question, passage_content)
            
            # 计算与答案段的实体重合度
            answer_overlap = self.calculate_entity_overlap(answer_passage_content, passage_content)
            
            # 综合评分：实体重合度 + 位置权重
            position_in_order = packed_order.index(idx) if idx in packed_order else len(packed_order)
            position_weight = 1.0 / (position_in_order + 1)  # 位置越靠前权重越高
            
            combined_score = (question_overlap + answer_overlap) * 0.8 + position_weight * 0.2
            
            candidates.append((idx, combined_score, question_overlap, answer_overlap))
        
        # 按综合评分排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前target_count个
        selected = [idx for idx, _, _, _ in candidates[:target_count]]
        
        logger.debug(f"Selected {len(selected)} bridging passages from {len(candidates)} candidates")
        return selected
    
    def fill_support_idxs_noid(self, question: str, answer: str, raw_support_idxs: List[int],
                              passages_by_idx: Dict[int, str], packed_order: List[int]) -> List[int]:
        """对LLM输出的support_idxs做结构化补齐/纠偏
        
        Args:
            question: 问题文本
            answer: 答案文本
            raw_support_idxs: LLM原始输出的支持段落索引
            passages_by_idx: 段落索引到内容的映射
            packed_order: 段落在prompt中的顺序
            
        Returns:
            补齐后的支持段落索引列表
        """
        # 空值短路处理
        if not passages_by_idx:
            logger.warning("Empty passages_by_idx, returning empty support list", 
                         extra={"empty_input_fallback": True, "reason": "no_passages"})
            return []
        
        if not answer or not isinstance(answer, str) or not answer.strip():
            logger.warning("Empty or invalid answer, returning first 2 passages", 
                         extra={"empty_input_fallback": True, "reason": "invalid_answer"})
            return packed_order[:2] if len(packed_order) >= 2 else packed_order
        
        try:
            # 1. 估计所需的K值
            target_k = estimate_required_k(question, answer, passages_by_idx, packed_order)
            logger.info(f"Target K estimated as: {target_k}")
            
            # 2. 找到包含答案的段落
            answer_containing_passages = self.find_answer_containing_passages(answer, passages_by_idx)
            
            # 3. 确保第一个位置是包含答案的段落
            final_support_idxs = []
            used_idxs = set()
            
            # 优先使用原始输出中包含答案的段落
            answer_idx_from_raw = None
            if raw_support_idxs:
                for idx in raw_support_idxs:
                    if idx in answer_containing_passages:
                        answer_idx_from_raw = idx
                        break
            
            # 如果原始输出中没有包含答案的段落，从候选中选择第一个
            if answer_idx_from_raw is None and answer_containing_passages:
                answer_idx_from_raw = answer_containing_passages[0]
            
            # 如果还是没有找到，使用原始输出的第一个（如果存在）
            if answer_idx_from_raw is None and raw_support_idxs:
                answer_idx_from_raw = raw_support_idxs[0]
                logger.warning(f"Could not find answer-containing passage, using first raw support: {answer_idx_from_raw}")
            
            # 如果完全没有，从packed_order中选择第一个
            if answer_idx_from_raw is None and packed_order:
                answer_idx_from_raw = packed_order[0]
                logger.warning(f"No raw support found, using first passage from packed_order: {answer_idx_from_raw}")
            
            if answer_idx_from_raw is not None:
                final_support_idxs.append(answer_idx_from_raw)
                used_idxs.add(answer_idx_from_raw)
            
            # 4. 添加原始输出中的其他有效段落（去重）
            if raw_support_idxs:
                for idx in raw_support_idxs:
                    if idx not in used_idxs and idx in passages_by_idx:
                        final_support_idxs.append(idx)
                        used_idxs.add(idx)
                        if len(final_support_idxs) >= target_k:
                            break
            
            # 5. 如果还需要更多段落，用桥接段补齐
            if len(final_support_idxs) < target_k:
                needed_count = target_k - len(final_support_idxs)
                
                # 获取答案段落内容用于桥接计算
                answer_passage_content = ""
                if final_support_idxs:
                    answer_passage_content = passages_by_idx.get(final_support_idxs[0], "")
                
                bridging_passages = self.find_bridging_passages(
                    question, answer_passage_content, passages_by_idx, 
                    packed_order, used_idxs, needed_count
                )
                
                final_support_idxs.extend(bridging_passages)
            
            # 6. 去重并截断到target_k
            seen = set()
            deduplicated = []
            for idx in final_support_idxs:
                if idx not in seen:
                    seen.add(idx)
                    deduplicated.append(idx)
                    if len(deduplicated) >= target_k:
                        break
            
            final_support_idxs = deduplicated
            
            logger.info(f"Final support_idxs: {final_support_idxs} (length: {len(final_support_idxs)})")
            return final_support_idxs
            
        except Exception as e:
            logger.error(f"Error in support filling: {e}")
            # 失败时返回原始输出（去重并限制长度）
            if raw_support_idxs:
                seen = set()
                fallback = []
                for idx in raw_support_idxs:
                    if idx not in seen and idx in passages_by_idx:
                        seen.add(idx)
                        fallback.append(idx)
                        if len(fallback) >= 4:  # 最多4个
                            break
                return fallback
            else:
                # 如果连原始输出都没有，返回前2个段落
                return packed_order[:2] if len(packed_order) >= 2 else packed_order


# 便捷函数
def fill_support_idxs_noid(question: str, answer: str, raw_support_idxs: List[int],
                          passages_by_idx: Dict[int, str], packed_order: List[int]) -> List[int]:
    """对LLM输出的support_idxs做结构化补齐/纠偏的便捷函数
    
    Args:
        question: 问题文本
        answer: 答案文本
        raw_support_idxs: LLM原始输出的支持段落索引
        passages_by_idx: 段落索引到内容的映射
        packed_order: 段落在prompt中的顺序
        
    Returns:
        补齐后的支持段落索引列表
    """
    # 空值短路处理
    if not passages_by_idx:
        logger.warning("Empty passages_by_idx, returning empty support list", 
                     extra={"empty_input_fallback": True, "reason": "no_passages"})
        return []
    
    if not answer or not isinstance(answer, str) or not answer.strip():
        logger.warning("Empty or invalid answer, returning first 2 passages", 
                     extra={"empty_input_fallback": True, "reason": "invalid_answer"})
        return packed_order[:2] if len(packed_order) >= 2 else packed_order
    
    filler = SupportFiller()
    return filler.fill_support_idxs_noid(question, answer, raw_support_idxs, passages_by_idx, packed_order)