"""
原子笔记质量过滤器
实现长度、显著性、质量标志、去重、限额等过滤逻辑
"""

import hashlib
from collections import defaultdict
from typing import List, Dict, Any, Set, Optional
from loguru import logger
from config import config

from utils.note_completeness import is_complete_sentence
from utils.notes_parser import enrich_note_keys
from utils.text_utils import TextUtils


class NotesQualityFilter:
    """原子笔记质量过滤器"""
    
    def __init__(self, question: Optional[str] = None):
        """
        初始化质量过滤器
        
        Args:
            question: 当前查询问题，用于调整salience阈值
        """
        self.question = question
        
        # 从配置加载过滤参数
        notes_cfg = config.get('notes_llm', {}) or {}
        quality_cfg = config.get('quality_filter', {}) or {}
        limit_cfg = notes_cfg.get('limit') if isinstance(notes_cfg.get('limit'), dict) else {}

        self.min_chars = int(quality_cfg.get('min_chars', notes_cfg.get('min_chars', 20)))
        self.max_chars = int(notes_cfg.get('max_chars', 400))
        self.min_salience = float(quality_cfg.get('min_salience', notes_cfg.get('min_salience', 0.3)))
        self.max_notes_per_chunk = int(notes_cfg.get('max_notes_per_chunk', 12))
        self.require_entities = bool(quality_cfg.get('require_entities', True))
        self.limit_strategy = (limit_cfg or {}).get('strategy', 'top_n')
        bucket_cfg = (limit_cfg or {}).get('bucket') if isinstance((limit_cfg or {}).get('bucket'), dict) else {}
        self.bucket_by = (bucket_cfg or {}).get('by', 'paragraph_idx')
        self.bucket_quota = int((bucket_cfg or {}).get('quota_per_bucket', 1))
        self.entities_fallback_cfg = (notes_cfg.get('entities_fallback', {}) or {}).copy()
        
        # 无question时降低salience阈值
        if not question:
            self.min_salience = max(0.2, self.min_salience - 0.1)
            logger.debug(f"No question provided, lowered salience threshold to {self.min_salience}")
        
        # 质量标志配置
        self.required_flags = {"OK"}
        self.forbidden_flags = {"DUPLICATE", "LOW_INFO", "TOO_LONG"}
        
        # 去重配置
        self.jaccard_threshold = 0.9
        
        # 统计信息
        self.stats = {
            'total_input': 0,
            'filtered_by_completeness': 0,
            'filtered_by_length': 0,
            'filtered_by_sent_count': 0,
            'filtered_by_salience': 0,
            'filtered_by_quality_flags': 0,
            'filtered_by_duplicate': 0,
            'filtered_by_limit': 0,
            'final_output': 0
        }
    
    def filter_notes(self, notes: List[Dict[str, Any]], accumulated_notes: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        对笔记列表进行质量过滤
        
        Args:
            notes: 待过滤的笔记列表
            accumulated_notes: 已累积的笔记列表，用于去重
            
        Returns:
            List[Dict]: 过滤后的笔记列表
        """
        if not notes:
            return []
        
        self.stats['total_input'] = len(notes)
        logger.debug(f"Starting quality filtering for {len(notes)} notes")

        # 补充结构化键值，后续过滤不会因缺失 slot 直接拒绝
        notes = [enrich_note_keys(note) for note in notes]

        # 0. 完整性过滤（仅句式层面）
        notes = self._filter_by_completeness(notes)

        # 1. 长度过滤
        notes = self._filter_by_length(notes)
        
        # 2. 句子数量过滤
        notes = self._filter_by_sent_count(notes)
        
        # 3. 显著性过滤
        notes = self._filter_by_salience(notes)

        # 4. 质量标志过滤
        notes = self._filter_by_quality_flags(notes)

        # 4.5 实体回补与实体需求
        notes = self._enrich_entities(notes)
        notes = self._enforce_entity_requirement(notes)
        
        # 5. 去重过滤
        notes = self._filter_duplicates(notes, accumulated_notes)
        
        # 6. 限额过滤（按配置策略）
        notes = self._apply_limit(notes)
        
        self.stats['final_output'] = len(notes)
        logger.info(f"Quality filtering completed: {self.stats['total_input']} -> {self.stats['final_output']} notes")
        
        return notes

    def _filter_by_completeness(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按完整命题要求过滤"""

        filtered: List[Dict[str, Any]] = []
        for note in notes:
            text = str(note.get('text') or '').strip()
            if text and is_complete_sentence(text, None):
                filtered.append(note)
            else:
                self.stats['filtered_by_completeness'] += 1
                logger.debug(f"Filtered note by completeness: {text}")

        return filtered

    def _enrich_entities(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """缺失实体时尝试回补"""

        if not notes or not self.require_entities:
            return notes

        if not self.entities_fallback_cfg.get('enabled', True):
            return notes

        min_len = int(self.entities_fallback_cfg.get('min_len', 2))
        allow_types = self.entities_fallback_cfg.get(
            'types', ['PERSON', 'ORG', 'GPE', 'WORK_OF_ART', 'EVENT']
        )

        for note in notes:
            entities = note.get('entities')
            if isinstance(entities, list) and any(str(e).strip() for e in entities):
                continue

            text = str(note.get('text') or '').strip()
            if not text:
                continue

            try:
                extracted = TextUtils.extract_entities_fallback(
                    text,
                    min_len=min_len,
                    allow_types=allow_types,
                ) or []
            except Exception as err:
                logger.debug(f"Entity fallback failed for note '{text[:50]}...': {err}")
                extracted = []

            if extracted:
                deduped = [str(e) for e in dict.fromkeys(extracted) if str(e).strip()]
                if deduped:
                    note['entities'] = deduped

        return notes

    def _enforce_entity_requirement(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """根据配置过滤缺失实体的笔记"""

        if not self.require_entities:
            return notes

        filtered: List[Dict[str, Any]] = []
        for note in notes:
            entities = note.get('entities', [])
            if isinstance(entities, list) and any(str(e).strip() for e in entities):
                filtered.append(note)
            else:
                self.stats['filtered_by_completeness'] += 1
                logger.debug(f"Filtered note for missing entities: {note.get('text', '')}")

        return filtered
    
    def _filter_by_length(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按文本长度过滤"""
        filtered = []
        for note in notes:
            text = note.get('text', '')
            text_len = len(text)
            
            if self.min_chars <= text_len <= self.max_chars:
                filtered.append(note)
            else:
                self.stats['filtered_by_length'] += 1
                logger.debug(f"Filtered note by length: {text_len} chars (range: {self.min_chars}-{self.max_chars})")
        
        return filtered
    
    def _filter_by_sent_count(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按句子数量过滤"""
        filtered = []
        for note in notes:
            sent_count = note.get('sent_count', 1)
            
            if 1 <= sent_count <= 3:
                filtered.append(note)
            else:
                self.stats['filtered_by_sent_count'] += 1
                logger.debug(f"Filtered note by sent_count: {sent_count} (range: 1-3)")
        
        return filtered
    
    def _filter_by_salience(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按显著性过滤"""
        filtered = []
        for note in notes:
            salience = note.get('salience', 0.0)
            
            if salience >= self.min_salience:
                filtered.append(note)
            else:
                self.stats['filtered_by_salience'] += 1
                logger.debug(f"Filtered note by salience: {salience} < {self.min_salience}")
        
        return filtered
    
    def _filter_by_quality_flags(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按质量标志过滤"""
        filtered = []
        for note in notes:
            quality_flags = set(note.get('quality_flags', []))
            
            # 必须包含OK标志
            if not self.required_flags.issubset(quality_flags):
                self.stats['filtered_by_quality_flags'] += 1
                logger.debug(f"Filtered note by missing required flags: {quality_flags}")
                continue
            
            # 不能包含禁止标志
            if quality_flags.intersection(self.forbidden_flags):
                self.stats['filtered_by_quality_flags'] += 1
                logger.debug(f"Filtered note by forbidden flags: {quality_flags}")
                continue
            
            filtered.append(note)
        
        return filtered
    
    def _filter_duplicates(self, notes: List[Dict[str, Any]], accumulated_notes: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """去重过滤"""
        if not notes:
            return notes
        
        # 同chunk内去重（基于text的hash）
        seen_hashes = set()
        chunk_deduped = []
        
        for note in notes:
            text = note.get('text', '')
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                chunk_deduped.append(note)
            else:
                self.stats['filtered_by_duplicate'] += 1
                logger.debug(f"Filtered duplicate note by hash: {text[:50]}...")
        
        # 与已累积笔记去重（基于Jaccard相似度）
        if accumulated_notes:
            final_filtered = []
            for note in chunk_deduped:
                if not self._is_duplicate_by_jaccard(note, accumulated_notes):
                    final_filtered.append(note)
                else:
                    self.stats['filtered_by_duplicate'] += 1
                    logger.debug(f"Filtered duplicate note by Jaccard similarity")
            return final_filtered
        
        return chunk_deduped
    
    def _is_duplicate_by_jaccard(self, note: Dict[str, Any], accumulated_notes: List[Dict[str, Any]]) -> bool:
        """检查笔记是否与已累积笔记重复（基于Jaccard相似度）"""
        note_text = note.get('text', '').lower()
        note_words = set(note_text.split())
        
        if not note_words:
            return False
        
        for acc_note in accumulated_notes:
            acc_text = acc_note.get('text', '').lower()
            acc_words = set(acc_text.split())
            
            if not acc_words:
                continue
            
            # 计算Jaccard相似度
            intersection = len(note_words.intersection(acc_words))
            union = len(note_words.union(acc_words))
            
            if union > 0:
                jaccard_similarity = intersection / union
                if jaccard_similarity > self.jaccard_threshold:
                    return True
        
        return False
    
    def _apply_limit(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用数量限制，支持桶策略"""
        if len(notes) <= self.max_notes_per_chunk:
            return notes

        sorted_notes = sorted(notes, key=lambda x: x.get('salience', 0.0), reverse=True)

        if self.limit_strategy != 'bucketed':
            limited_notes = sorted_notes[:self.max_notes_per_chunk]
        else:
            buckets: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
            for note in sorted_notes:
                bucket_key = self._resolve_bucket_key(note)
                buckets[bucket_key].append(note)

            quota = max(1, self.bucket_quota)
            limited_notes: List[Dict[str, Any]] = []
            selected_ids: Set[int] = set()
            overflow: List[Dict[str, Any]] = []

            for bucket_notes in buckets.values():
                bucket_sorted = sorted(bucket_notes, key=lambda x: x.get('salience', 0.0), reverse=True)
                primary = bucket_sorted[:quota]
                extra = bucket_sorted[quota:]

                for note in primary:
                    note_id = id(note)
                    if note_id in selected_ids:
                        continue
                    limited_notes.append(note)
                    selected_ids.add(note_id)

                overflow.extend(extra)

            if len(limited_notes) > self.max_notes_per_chunk:
                limited_notes = sorted(
                    limited_notes,
                    key=lambda x: x.get('salience', 0.0),
                    reverse=True,
                )[: self.max_notes_per_chunk]
                selected_ids = {id(note) for note in limited_notes}
            elif len(limited_notes) < self.max_notes_per_chunk:
                overflow_sorted = sorted(overflow, key=lambda x: x.get('salience', 0.0), reverse=True)
                for note in overflow_sorted:
                    if len(limited_notes) >= self.max_notes_per_chunk:
                        break
                    note_id = id(note)
                    if note_id in selected_ids:
                        continue
                    limited_notes.append(note)
                    selected_ids.add(note_id)

                if len(limited_notes) < self.max_notes_per_chunk:
                    for note in sorted_notes:
                        if len(limited_notes) >= self.max_notes_per_chunk:
                            break
                        note_id = id(note)
                        if note_id in selected_ids:
                            continue
                        limited_notes.append(note)
                        selected_ids.add(note_id)

        filtered_count = max(0, len(notes) - len(limited_notes))
        self.stats['filtered_by_limit'] += filtered_count
        if filtered_count > 0:
            logger.debug(
                f"Applied {self.limit_strategy} limit: kept {len(limited_notes)} of {len(notes)} notes"
            )

        return limited_notes

    def _resolve_bucket_key(self, note: Dict[str, Any]) -> Any:
        if self.bucket_by == 'paragraph_idx':
            paragraph_idxs = note.get('paragraph_idxs') or []
            if isinstance(paragraph_idxs, list) and paragraph_idxs:
                return paragraph_idxs[0]
            return f"chunk_{note.get('chunk_index')}"
        return note.get(self.bucket_by)
    
    def get_stats(self) -> Dict[str, int]:
        """获取过滤统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        for key in self.stats:
            self.stats[key] = 0


def apply_quality_filter(
    notes: List[Dict[str, Any]],
    question: Optional[str] = None,
    accumulated_notes: Optional[List[Dict[str, Any]]] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """便捷函数：对笔记应用质量过滤"""

    filter_instance = NotesQualityFilter(question=question)
    filtered_notes = filter_instance.filter_notes(notes, accumulated_notes)
    stats = filter_instance.get_stats()

    return filtered_notes, stats
