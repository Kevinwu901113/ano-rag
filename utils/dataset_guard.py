from typing import List, Dict, Any, Optional, Callable
from loguru import logger
import re
import os
import time
from collections import defaultdict


class DatasetNamespaceError(Exception):
    """命名空间验证错误异常"""
    pass


class NamespaceMismatchError(DatasetNamespaceError):
    """命名空间不匹配错误"""
    pass


class EmptyRecallError(DatasetNamespaceError):
    """空召回错误"""
    pass


class DatasetGuard:
    """数据集命名空间守卫，用于验证和过滤笔记的命名空间归属"""
    
    def __init__(self, bm25_fallback_fn: Optional[Callable] = None):
        self.logger = logger
        self.bm25_fallback_fn = bm25_fallback_fn
        self.namespace_stats = defaultdict(lambda: {'hits': 0, 'misses': 0, 'fallbacks': 0})
        self.session_start_time = time.time()
    
    def note_belongs(self, note: Dict[str, Any], dataset: str, qid: str) -> bool:
        """
        验证笔记是否属于指定的 dataset/qid 命名空间
        
        Args:
            note: 原子笔记对象
            dataset: 数据集名称
            qid: 问题ID
            
        Returns:
            bool: 是否属于指定命名空间
        """
        try:
            # 获取笔记的源信息
            source_info = note.get('source_info', {})
            
            # 检查 file_path
            file_path = source_info.get('file_path', '')
            if file_path:
                return self._validate_file_path(file_path, dataset, qid)
            
            # 检查 file_name
            file_name = source_info.get('file_name', '')
            if file_name:
                return self._validate_file_name(file_name, dataset, qid)
            
            # 检查其他可能的路径字段
            for path_field in ['source_file', 'document_path', 'origin_file']:
                path_value = source_info.get(path_field, '')
                if path_value:
                    return self._validate_file_path(path_value, dataset, qid)
            
            # 如果没有找到任何路径信息，记录警告
            self.logger.warning(f"Note {note.get('note_id', 'unknown')} has no source path information")
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating note namespace: {e}")
            return False
    
    def _validate_file_path(self, file_path: str, dataset: str, qid: str) -> bool:
        """
        验证文件路径是否包含指定的 dataset 和 qid
        
        Args:
            file_path: 文件路径
            dataset: 数据集名称
            qid: 问题ID
            
        Returns:
            bool: 是否匹配命名空间
        """
        if not file_path or not dataset or not qid:
            return False
        
        # 标准化路径分隔符
        normalized_path = file_path.replace('\\', '/').lower()
        dataset_lower = dataset.lower()
        qid_lower = qid.lower()
        
        # 检查路径中是否同时包含 dataset 和 qid
        contains_dataset = dataset_lower in normalized_path
        contains_qid = qid_lower in normalized_path
        
        # 更严格的匹配：检查是否作为路径组件存在
        path_components = normalized_path.split('/')
        dataset_in_components = any(dataset_lower in component for component in path_components)
        qid_in_components = any(qid_lower in component for component in path_components)
        
        return (contains_dataset and contains_qid) or (dataset_in_components and qid_in_components)
    
    def _validate_file_name(self, file_name: str, dataset: str, qid: str) -> bool:
        """
        验证文件名是否包含指定的 dataset 和 qid
        
        Args:
            file_name: 文件名
            dataset: 数据集名称
            qid: 问题ID
            
        Returns:
            bool: 是否匹配命名空间
        """
        if not file_name or not dataset or not qid:
            return False
        
        file_name_lower = file_name.lower()
        dataset_lower = dataset.lower()
        qid_lower = qid.lower()
        
        # 检查文件名中是否同时包含 dataset 和 qid
        return dataset_lower in file_name_lower and qid_lower in file_name_lower
    
    def filter_notes_by_namespace(self, notes: List[Dict[str, Any]], dataset: str, qid: str) -> List[Dict[str, Any]]:
        """
        过滤出符合指定命名空间要求的笔记
        
        Args:
            notes: 笔记列表
            dataset: 数据集名称
            qid: 问题ID
            
        Returns:
            List[Dict[str, Any]]: 过滤后的笔记列表
        """
        if not notes:
            return []
        
        filtered_notes = []
        for note in notes:
            if self.note_belongs(note, dataset, qid):
                filtered_notes.append(note)
        
        self.logger.info(
            f"Namespace filtering: {len(filtered_notes)}/{len(notes)} notes belong to {dataset}/{qid}"
        )
        
        return filtered_notes
    
    def assert_namespace_or_raise(self, notes: List[Dict[str, Any]], dataset: str, qid: str, 
                                 index_version: Optional[str] = None) -> None:
        """
        命名空间断言检查，如果没有符合条件的笔记则抛出异常
        
        Args:
            notes: 笔记列表
            dataset: 数据集名称
            qid: 问题ID
            index_version: 索引版本号（可选）
            
        Raises:
            DatasetNamespaceError: 当没有符合命名空间要求的笔记时
        """
        filtered_notes = self.filter_notes_by_namespace(notes, dataset, qid)
        
        if not filtered_notes:
            # 记录错误日志
            error_msg = f"dataset/qid mismatch: No notes found in namespace {dataset}/{qid}. Total notes checked: {len(notes)}"
            self.logger.error(error_msg)
            raise DatasetNamespaceError(error_msg)
        
        # 校验成功，记录DEBUG级别的详细信息
        candidate_files = set()
        for note in filtered_notes:
            source_info = note.get('source_info', {})
            # 收集候选文件名
            for path_field in ['file_path', 'file_name', 'source_file', 'document_path', 'origin_file']:
                path_value = source_info.get(path_field, '')
                if path_value:
                    # 提取文件名
                    file_name = os.path.basename(path_value) if '/' in path_value or '\\' in path_value else path_value
                    if file_name:
                        candidate_files.add(file_name)
        
        # 记录DEBUG级别日志
        debug_info = {
            'dataset': dataset,
            'qid': qid,
            'index_version': index_version or 'unknown',
            'candidate_files': sorted(list(candidate_files)),
            'matched_notes_count': len(filtered_notes)
        }
        
        self.logger.debug(
            f"Namespace validation passed - Dataset: {dataset}, QID: {qid}, "
            f"Index Version: {index_version or 'unknown'}, "
            f"Candidate Files: {', '.join(sorted(list(candidate_files)))}, "
            f"Matched Notes: {len(filtered_notes)}"
        )
        
        self.logger.info(f"Namespace assertion passed: {len(filtered_notes)} notes in {dataset}/{qid}")
    
    def namespace_aware_retrieve(self, query: str, dataset: str, qid: str, 
                                retrieval_fn: Callable, 
                                top_k: int = 20,
                                min_namespace_ratio: float = 0.3,
                                enable_fallback: bool = True,
                                **retrieval_kwargs) -> List[Dict[str, Any]]:
        """
        命名空间感知的检索方法，带有BM25回退机制
        
        Args:
            query: 查询字符串
            dataset: 数据集名称
            qid: 问题ID
            retrieval_fn: 主检索函数
            top_k: 返回结果数量
            min_namespace_ratio: 最小命名空间命中比例阈值
            enable_fallback: 是否启用回退机制
            **retrieval_kwargs: 传递给检索函数的其他参数
            
        Returns:
            List[Dict[str, Any]]: 检索结果列表
            
        Raises:
            NamespaceMismatchError: 命名空间不匹配且回退失败
            EmptyRecallError: 检索结果为空
        """
        namespace_key = f"{dataset}/{qid}"
        
        try:
            # 第一阶段：主检索
            self.logger.info(f"Starting namespace-aware retrieval for {namespace_key}")
            candidates = retrieval_fn(query, top_k=top_k * 2, **retrieval_kwargs)
            
            if not candidates:
                self.namespace_stats[namespace_key]['misses'] += 1
                raise EmptyRecallError(f"No candidates found for query: {query[:50]}...")
            
            # 第二阶段：命名空间过滤和验证
            namespace_candidates = self.filter_notes_by_namespace(candidates, dataset, qid)
            namespace_ratio = len(namespace_candidates) / len(candidates) if candidates else 0.0
            
            self.logger.info(
                f"Namespace filtering: {len(namespace_candidates)}/{len(candidates)} "
                f"candidates belong to {namespace_key} (ratio: {namespace_ratio:.2f})"
            )
            
            # 检查命名空间命中率
            if namespace_ratio >= min_namespace_ratio and len(namespace_candidates) >= top_k:
                # 命名空间命中率足够，直接返回
                self.namespace_stats[namespace_key]['hits'] += 1
                result = namespace_candidates[:top_k]
                
                # 记录成功的详细信息
                sample_files = self._extract_sample_files(result, max_samples=3)
                self.logger.info(
                    f"Namespace retrieval successful - Dataset: {dataset}, QID: {qid}, "
                    f"Hit ratio: {namespace_ratio:.2f}, Sample files: {', '.join(sample_files)}"
                )
                return result
            
            # 第三阶段：BM25混合检索回退
            if enable_fallback and self.bm25_fallback_fn:
                self.logger.warning(
                    f"Namespace hit ratio {namespace_ratio:.2f} below threshold {min_namespace_ratio}, "
                    f"triggering BM25 fallback for {namespace_key}"
                )
                
                try:
                    fallback_candidates = self.bm25_fallback_fn(
                        query=query, 
                        dataset=dataset, 
                        qid=qid, 
                        top_k=top_k,
                        **retrieval_kwargs
                    )
                    
                    if fallback_candidates:
                        # 验证回退结果的命名空间
                        fallback_namespace_candidates = self.filter_notes_by_namespace(
                            fallback_candidates, dataset, qid
                        )
                        
                        if fallback_namespace_candidates:
                            self.namespace_stats[namespace_key]['fallbacks'] += 1
                            
                            # 记录回退成功信息
                            fallback_ratio = len(fallback_namespace_candidates) / len(fallback_candidates)
                            sample_files = self._extract_sample_files(fallback_namespace_candidates, max_samples=3)
                            
                            self.logger.warning(
                                f"BM25 fallback successful - Dataset: {dataset}, QID: {qid}, "
                                f"Fallback ratio: {fallback_ratio:.2f}, Sample files: {', '.join(sample_files)}"
                            )
                            
                            return fallback_namespace_candidates[:top_k]
                        
                except Exception as e:
                    self.logger.error(f"BM25 fallback failed for {namespace_key}: {e}")
            
            # 第四阶段：最终错误处理
            self.namespace_stats[namespace_key]['misses'] += 1
            
            # 记录详细的错误信息
            error_details = {
                'namespace': namespace_key,
                'query': query[:100],
                'total_candidates': len(candidates),
                'namespace_candidates': len(namespace_candidates),
                'namespace_ratio': namespace_ratio,
                'min_required_ratio': min_namespace_ratio,
                'fallback_enabled': enable_fallback,
                'fallback_available': self.bm25_fallback_fn is not None
            }
            
            self.logger.error(
                f"Namespace retrieval failed - {error_details}"
            )
            
            raise NamespaceMismatchError(
                f"Insufficient namespace matches for {namespace_key}. "
                f"Found {len(namespace_candidates)}/{len(candidates)} candidates "
                f"(ratio: {namespace_ratio:.2f}, required: {min_namespace_ratio:.2f}). "
                f"Fallback {'failed' if enable_fallback else 'disabled'}."
            )
            
        except (NamespaceMismatchError, EmptyRecallError):
            raise
        except Exception as e:
            self.namespace_stats[namespace_key]['misses'] += 1
            self.logger.error(f"Unexpected error in namespace-aware retrieval for {namespace_key}: {e}")
            raise DatasetNamespaceError(f"Retrieval failed for {namespace_key}: {str(e)}")
    
    def get_namespace_stats(self, notes: List[Dict[str, Any]], dataset: str, qid: str) -> Dict[str, Any]:
        """
        获取命名空间统计信息
        
        Args:
            notes: 笔记列表
            dataset: 数据集名称
            qid: 问题ID
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not notes:
            return {
                'total_notes': 0,
                'matching_notes': 0,
                'match_rate': 0.0,
                'namespace': f"{dataset}/{qid}"
            }
        
        matching_notes = self.filter_notes_by_namespace(notes, dataset, qid)
        match_rate = len(matching_notes) / len(notes) if notes else 0.0
        
        return {
            'total_notes': len(notes),
            'matching_notes': len(matching_notes),
            'match_rate': match_rate,
            'namespace': f"{dataset}/{qid}",
            'has_matches': len(matching_notes) > 0
        }
    
    def _extract_sample_files(self, notes: List[Dict[str, Any]], max_samples: int = 3) -> List[str]:
        """
        从笔记列表中提取样本文件名
        
        Args:
            notes: 笔记列表
            max_samples: 最大样本数量
            
        Returns:
            List[str]: 样本文件名列表
        """
        sample_files = set()
        
        for note in notes[:max_samples * 2]:  # 多取一些以防重复
            source_info = note.get('source_info', {})
            
            # 尝试多个可能的路径字段
            for path_field in ['file_path', 'file_name', 'source_file', 'document_path', 'origin_file']:
                path_value = source_info.get(path_field, '')
                if path_value:
                    # 提取文件名
                    file_name = os.path.basename(path_value) if ('/' in path_value or '\\' in path_value) else path_value
                    if file_name:
                        sample_files.add(file_name)
                        break
            
            if len(sample_files) >= max_samples:
                break
        
        return sorted(list(sample_files))
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        获取会话统计信息
        
        Returns:
            Dict[str, Any]: 会话统计信息
        """
        total_hits = sum(stats['hits'] for stats in self.namespace_stats.values())
        total_misses = sum(stats['misses'] for stats in self.namespace_stats.values())
        total_fallbacks = sum(stats['fallbacks'] for stats in self.namespace_stats.values())
        total_requests = total_hits + total_misses
        
        session_duration = time.time() - self.session_start_time
        
        return {
            'session_duration_seconds': session_duration,
            'total_requests': total_requests,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'total_fallbacks': total_fallbacks,
            'hit_rate': total_hits / total_requests if total_requests > 0 else 0.0,
            'fallback_rate': total_fallbacks / total_requests if total_requests > 0 else 0.0,
            'namespace_details': dict(self.namespace_stats)
        }
    
    def log_session_summary(self) -> None:
        """
        记录会话摘要统计信息
        """
        stats = self.get_session_stats()
        
        self.logger.info(
            f"Namespace Guard Session Summary - "
            f"Duration: {stats['session_duration_seconds']:.1f}s, "
            f"Requests: {stats['total_requests']}, "
            f"Hit Rate: {stats['hit_rate']:.2f}, "
            f"Fallback Rate: {stats['fallback_rate']:.2f}"
        )
        
        # 记录每个命名空间的详细统计
        for namespace, details in stats['namespace_details'].items():
            total = details['hits'] + details['misses']
            if total > 0:
                hit_rate = details['hits'] / total
                fallback_rate = details['fallbacks'] / total
                self.logger.debug(
                    f"Namespace {namespace}: {total} requests, "
                    f"hit rate: {hit_rate:.2f}, fallback rate: {fallback_rate:.2f}"
                )
    
    def set_bm25_fallback(self, fallback_fn: Callable) -> None:
        """
        设置BM25回退函数
        
        Args:
            fallback_fn: BM25回退函数
        """
        self.bm25_fallback_fn = fallback_fn
        self.logger.info("BM25 fallback function registered")
    
    def validate_and_route(self, candidates: List[Dict[str, Any]], 
                          dataset: str, qid: str,
                          strict_mode: bool = False) -> List[Dict[str, Any]]:
        """
        验证候选结果并进行命名空间路由
        
        Args:
            candidates: 候选结果列表
            dataset: 数据集名称
            qid: 问题ID
            strict_mode: 是否启用严格模式（严格模式下不符合命名空间的结果会被完全过滤）
            
        Returns:
            List[Dict[str, Any]]: 路由后的结果列表
        """
        if not candidates:
            return []
        
        namespace_candidates = self.filter_notes_by_namespace(candidates, dataset, qid)
        
        if strict_mode:
            # 严格模式：只返回符合命名空间的候选
            return namespace_candidates
        else:
            # 宽松模式：优先返回符合命名空间的候选，不足时补充其他候选
            other_candidates = [
                candidate for candidate in candidates 
                if not self.note_belongs(candidate, dataset, qid)
            ]
            
            # 合并结果：命名空间候选优先
            return namespace_candidates + other_candidates


# 全局实例
dataset_guard = DatasetGuard()


# 便捷函数
def note_belongs(note: Dict[str, Any], dataset: str, qid: str) -> bool:
    """验证笔记是否属于指定命名空间"""
    return dataset_guard.note_belongs(note, dataset, qid)


def filter_notes_by_namespace(notes: List[Dict[str, Any]], dataset: str, qid: str) -> List[Dict[str, Any]]:
    """过滤出符合命名空间要求的笔记"""
    return dataset_guard.filter_notes_by_namespace(notes, dataset, qid)


def assert_namespace_or_raise(notes: List[Dict[str, Any]], dataset: str, qid: str, 
                             index_version: Optional[str] = None) -> None:
    """命名空间断言检查"""
    return dataset_guard.assert_namespace_or_raise(notes, dataset, qid, index_version)


def namespace_aware_retrieve(query: str, dataset: str, qid: str, 
                           retrieval_fn: Callable, 
                           top_k: int = 20,
                           min_namespace_ratio: float = 0.3,
                           enable_fallback: bool = True,
                           **retrieval_kwargs) -> List[Dict[str, Any]]:
    """命名空间感知的检索方法"""
    return dataset_guard.namespace_aware_retrieve(
        query, dataset, qid, retrieval_fn, top_k, 
        min_namespace_ratio, enable_fallback, **retrieval_kwargs
    )


def set_bm25_fallback(fallback_fn: Callable) -> None:
    """设置BM25回退函数"""
    return dataset_guard.set_bm25_fallback(fallback_fn)


def get_namespace_stats() -> Dict[str, Any]:
    """获取命名空间统计信息"""
    return dataset_guard.get_session_stats()


def validate_and_route(candidates: List[Dict[str, Any]], 
                      dataset: str, qid: str,
                      strict_mode: bool = False) -> List[Dict[str, Any]]:
    """验证候选结果并进行命名空间路由"""
    return dataset_guard.validate_and_route(candidates, dataset, qid, strict_mode)