import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from tqdm import tqdm
import networkx as nx
from networkx.readwrite import json_graph
from collections import defaultdict
import hashlib
import json
from .chunker import DocumentChunker
from .clustering import TopicClustering
from .incremental_processor import IncrementalProcessor
from llm import AtomicNoteGenerator, LocalLLM
from llm.parallel_task_atomic_note_generator import ParallelTaskAtomicNoteGenerator
from utils import BatchProcessor, FileUtils, JSONLProgressTracker
from utils.enhanced_ner import EnhancedNER
from utils.consistency_checker import ConsistencyChecker
from config import config
from vector_store import EmbeddingManager
from graph import GraphBuilder

class DocumentProcessor:
    """文档处理器主类，整合所有文档处理功能"""

    def __init__(self, output_dir: Optional[str] = None, llm: Optional[LocalLLM] = None):
        # 初始化组件
        self.chunker = DocumentChunker()
        self.clustering = TopicClustering()
        # 增量处理缓存目录放在工作目录下
        cache_dir = None
        if output_dir:
            cache_dir = os.path.join(output_dir, 'cache')
        else:
            work_dir = config.get('storage.work_dir')
            if work_dir:
                cache_dir = os.path.join(work_dir, 'cache')
            else:
                # 默认缓存目录
                cache_dir = './cache'
        self.incremental_processor = IncrementalProcessor(cache_dir=cache_dir)
        if llm is None:
            raise ValueError("DocumentProcessor requires a LocalLLM instance to be passed")
        self.llm = llm
        
        # 选择原子笔记生成器：优先使用并行任务分配生成器
        parallel_config = config.get('atomic_note_generation', {})
        if (parallel_config.get('parallel_enabled', False) and 
            parallel_config.get('parallel_strategy') == 'task_division'):
            logger.info("Using ParallelTaskAtomicNoteGenerator for atomic note generation")
            self.atomic_note_generator = ParallelTaskAtomicNoteGenerator(self.llm)
        else:
            logger.info("Using standard AtomicNoteGenerator for atomic note generation")
            self.atomic_note_generator = AtomicNoteGenerator(self.llm)
        self.batch_processor = BatchProcessor(
            batch_size=config.get('document.batch_size', 32),
            use_gpu=config.get('performance.use_gpu', True)
        )
        self.embedding_manager = EmbeddingManager()
        self.graph_builder = GraphBuilder(llm=self.llm)
        
        # 存储路径，默认使用配置中的工作目录
        self.processed_docs_path = output_dir or config.get('storage.work_dir') or config.get('storage.processed_docs_path') or './data/processed'
        FileUtils.ensure_dir(self.processed_docs_path)
        
    def process_documents(self, file_paths: List[str], force_reprocess: bool = False, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """处理文档的主要入口点"""
        logger.info(f"Starting document processing for {len(file_paths)} files")
        
        # 检查是否有JSONL文件，如果有则创建进度跟踪器
        jsonl_files = [f for f in file_paths if f.endswith('.jsonl')]
        progress_tracker = None
        if jsonl_files:
            # 使用第一个JSONL文件创建进度跟踪器
            progress_tracker = JSONLProgressTracker(jsonl_files[0], "Processing JSONL data")
            progress_tracker.start()

        if output_dir:
            self.processed_docs_path = output_dir
            FileUtils.ensure_dir(self.processed_docs_path)
        
        # 获取处理计划
        processing_plan = self.incremental_processor.get_processing_plan(file_paths)

        if not force_reprocess and processing_plan['can_skip_processing']:
            logger.info("No files need processing, loading cached results")
            return self._load_cached_results(file_paths)

        files_to_process = processing_plan['files_to_process']
        unchanged_files = processing_plan.get('unchanged_files', [])
        files_to_clean = processing_plan.get('files_to_clean', [])

        if force_reprocess:
            files_to_process = file_paths
            unchanged_files = []
            files_to_clean = []

        logger.info(f"Processing {len(files_to_process)} files")

        chunk_file = os.path.join(self.processed_docs_path, "chunks.jsonl")
        existing_chunks = []
        if os.path.exists(chunk_file):
            try:
                existing_chunks = FileUtils.read_jsonl(chunk_file)
            except Exception as e:
                logger.warning(f"Failed to load existing chunks: {e}")

        # 过滤出未变更文件的旧分块
        cached_chunks = [
            c for c in existing_chunks
            if c.get('source_info', {}).get('file_path') in unchanged_files
        ]

        # 移除需要清理的文件对应的分块
        cached_chunks = [
            c for c in cached_chunks
            if c.get('source_info', {}).get('file_path') not in files_to_clean
        ]

        chunks_updated = False
        new_chunks = []
        if files_to_process:
            logger.info("Step 1: Document chunking")
            new_chunks = self._chunk_documents(files_to_process)
            chunks_updated = True
        if files_to_clean:
            chunks_updated = True

        all_chunks = cached_chunks + new_chunks
        chunk_count = len(all_chunks)

        if chunks_updated:
            FileUtils.write_jsonl(all_chunks, chunk_file)
        
        if not all_chunks:
            logger.warning("No chunks created from documents")
            return {'atomic_notes': [], 'topic_pools': [], 'processing_stats': {}}
        
        atomic_file = os.path.join(self.processed_docs_path, "atomic_notes.json")
        if (not force_reprocess and not chunks_updated
                and os.path.exists(atomic_file)):
            logger.info(f"Loading atomic notes from {atomic_file}")
            atomic_notes = FileUtils.read_json(atomic_file)
        else:
            logger.info("Step 2: Generating atomic notes")
            atomic_notes = self._generate_atomic_notes(all_chunks, progress_tracker)
            FileUtils.write_json(atomic_notes, atomic_file)
            
            # 保存被标记为需要重写的摘要（如果启用了摘要校验器）
            summary_auditor_enabled = config.get('summary_auditor.enabled', False)
            if summary_auditor_enabled:
                try:
                    from utils.summary_auditor import SummaryAuditor
                    auditor = SummaryAuditor(llm=self.llm)
                    auditor.save_flagged_summaries(atomic_notes, self.processed_docs_path)
                except ImportError as e:
                    logger.warning(f"Failed to import SummaryAuditor: {e}")
                except Exception as e:
                    logger.error(f"Failed to save flagged summaries: {e}")
        note_count = len(atomic_notes)
        
        if not atomic_notes:
            logger.warning("No atomic notes generated")
            return {'atomic_notes': [], 'topic_pools': [], 'processing_stats': {}}
        
        embed_file = os.path.join(self.processed_docs_path, "embeddings.npy")
        if (not force_reprocess and not chunks_updated
                and os.path.exists(embed_file)):
            logger.info(f"Loading embeddings from {embed_file}")
            embeddings = np.load(embed_file)
        else:
            logger.info("Step 3: Creating embeddings")
            embeddings = self.embedding_manager.encode_atomic_notes(atomic_notes)
            np.save(embed_file, embeddings)
        
        cluster_file = os.path.join(self.processed_docs_path, "clustering.json")
        if (not force_reprocess and not chunks_updated
                and os.path.exists(cluster_file)):
            logger.info(f"Loading clustering from {cluster_file}")
            clustering_result = FileUtils.read_json(cluster_file)
        else:
            logger.info("Step 4: Topic clustering")
            clustering_result = self.clustering.cluster_notes(atomic_notes, embeddings)
            FileUtils.write_json(clustering_result, cluster_file)
        
        graph_file = os.path.join(self.processed_docs_path, "graph.json")
        graphml_file = os.path.join(self.processed_docs_path, "graph.graphml")
        if (not force_reprocess and not chunks_updated
                and os.path.exists(graph_file)):
            logger.info(f"Loading graph from {graph_file}")
            graph_data = FileUtils.read_json(graph_file)
        else:
            logger.info("Step 5: Building graph relationships")
            graph = self.graph_builder.build_graph(
                clustering_result['clustered_notes'], embeddings
            )
            graph_data = nx.node_link_data(graph)
            FileUtils.write_json(graph_data, graph_file)
            
            # 导出GraphML格式
            try:
                from graph.graphml_exporter import GraphMLExporter
                exporter = GraphMLExporter()
                exporter.export_graph(graph, graphml_file)
                logger.info(f"Graph exported to GraphML format: {graphml_file}")
            except Exception as e:
                logger.warning(f"Failed to export GraphML: {e}")
        
        # 一致性检查（在数据写入前）
        consistency_result = None
        if config.get('consistency_check.enabled', True):
            logger.info("Performing consistency check before data persistence")
            checker = ConsistencyChecker()
            consistency_result = checker.check_consistency(
                clustering_result['clustered_notes'], 
                graph_data
            )
            
            # 如果有严重错误且启用了严格模式，停止处理
            if not consistency_result['is_consistent'] and config.get('consistency_check.strict_mode', False):
                error_msg = f"Consistency check failed with {len(consistency_result['errors'])} errors"
                logger.error(error_msg)
                
                # 导出错误报告
                error_report_file = os.path.join(self.processed_docs_path, "consistency_errors.json")
                checker.export_report(error_report_file)
                
                raise RuntimeError(f"{error_msg}. See report: {error_report_file}")
            
            # 导出一致性报告
            consistency_report_file = os.path.join(self.processed_docs_path, "consistency_report.json")
            checker.export_report(consistency_report_file)
            
            if consistency_result['errors']:
                logger.warning(f"Consistency check found {len(consistency_result['errors'])} errors and {len(consistency_result['warnings'])} warnings")
            else:
                logger.info(f"Consistency check passed with {len(consistency_result['warnings'])} warnings")
        
        # 保存处理结果
        result = {
            'atomic_notes': clustering_result['clustered_notes'],
            'topic_pools': clustering_result['topic_pools'],
            'cluster_info': clustering_result['cluster_info'],
            'graph_data': graph_data,
            'consistency_check': consistency_result,
            'processing_stats': self._calculate_processing_stats(
                files_to_process,
                atomic_notes,
                clustering_result,
                chunk_count,
                note_count,
            )
        }
        
        # 更新缓存
        self._update_processing_cache(files_to_process, result)

        result_file = os.path.join(self.processed_docs_path, "result.json")
        FileUtils.write_json(result, result_file)

        # 关闭进度跟踪器
        if progress_tracker:
            progress_tracker.close()
            
        logger.info("Document processing completed successfully")
        return result
    
    def _chunk_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """文档分块处理"""
        all_chunks = []
        for file_path in tqdm(file_paths, desc="Chunking documents"):
            try:
                # 创建source_info
                source_info = {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'file_hash': FileUtils.get_file_hash(file_path) if hasattr(FileUtils, 'get_file_hash') else 'unknown'
                }
                chunks = self.chunker.chunk_document(file_path, source_info)
                all_chunks.extend(chunks)
                logger.debug(f"Created {len(chunks)} chunks from {file_path}")
            except Exception as e:
                logger.error(f"Failed to chunk document {file_path}: {e}")
        
        # 验证分块结果
        valid_chunks = self.chunker.validate_chunks(all_chunks)
        logger.info(f"Created {len(valid_chunks)} valid chunks from {len(file_paths)} documents")
        
        return valid_chunks
    
    def _generate_atomic_notes(self, chunks: List[Dict[str, Any]], progress_tracker: Optional[JSONLProgressTracker] = None) -> List[Dict[str, Any]]:
        """生成原子笔记，支持多条笔记批量处理和全局去重"""
        try:
            logger.info(f"开始处理 {len(chunks)} 个文档块，生成原子笔记")
            
            # 使用批处理生成原子笔记，支持多条笔记输出
            raw_atomic_notes = self.atomic_note_generator.generate_atomic_notes(chunks, progress_tracker)
            logger.info(f"原始生成 {len(raw_atomic_notes)} 条原子笔记")
            
            # 验证原子笔记质量，过滤低质量笔记
            valid_notes = self.atomic_note_generator.validate_atomic_notes(raw_atomic_notes)
            logger.info(f"质量验证后保留 {len(valid_notes)} 条笔记")

            # 实体归一化和追踪增强
            valid_notes = EnhancedNER().enhance_entity_tracing(valid_notes)
            logger.info(f"实体增强完成，处理 {len(valid_notes)} 条笔记")

            # 增强笔记关系
            enhanced_notes = self.atomic_note_generator.enhance_notes_with_relations(valid_notes)
            logger.info(f"关系增强完成，处理 {len(enhanced_notes)} 条笔记")
            
            # 扁平化多条笔记并进行全局去重
            flattened_notes = self._flatten_and_deduplicate_notes(enhanced_notes)
            logger.info(f"扁平化和去重完成，最终保留 {len(flattened_notes)} 条笔记")
            
            # 计算和记录详细统计指标
            stats = self._calculate_note_generation_stats(chunks, raw_atomic_notes, valid_notes, enhanced_notes, flattened_notes)
            self._log_processing_stats(stats)
            
            logger.info(f"原子笔记生成完成：原始 {len(raw_atomic_notes)} → 最终 {len(flattened_notes)} 条（去重率: {((len(raw_atomic_notes) - len(flattened_notes)) / max(len(raw_atomic_notes), 1) * 100):.1f}%）")
            return flattened_notes
            
        except Exception as e:
            logger.error(f"原子笔记生成失败: {e}")
            return []
    
    def _flatten_and_deduplicate_notes(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """扁平化多条笔记并进行全局去重，确保数据层面干净和可管控"""
        if not notes:
            return []
            
        logger.info(f"开始扁平化和去重处理，输入 {len(notes)} 条笔记")
        flattened_notes = []
        multi_fact_count = 0
        single_fact_count = 0
        
        # 扁平化：统一处理多条笔记，将多事实笔记拆分为单独的笔记
        for note_idx, note in enumerate(notes):
            if isinstance(note.get('content'), list) and len(note['content']) > 1:
                # 多事实笔记，拆分为单个事实
                multi_fact_count += 1
                for fact_idx, fact in enumerate(note['content']):
                    single_note = note.copy()
                    single_note['content'] = fact
                    single_note['fact_index'] = fact_idx
                    single_note['original_note_id'] = note.get('id', f'note_{note_idx}')
                    single_note['is_multi_fact_split'] = True
                    single_note['total_facts_in_original'] = len(note['content'])
                    flattened_notes.append(single_note)
            else:
                # 单事实笔记，直接添加
                single_fact_count += 1
                if isinstance(note.get('content'), list) and len(note['content']) == 1:
                    note['content'] = note['content'][0]  # 统一格式
                note['is_multi_fact_split'] = False
                flattened_notes.append(note)
        
        logger.info(f"扁平化完成：多事实笔记 {multi_fact_count} 条，单事实笔记 {single_fact_count} 条，总计 {len(flattened_notes)} 条")
        
        # 全局去重：基于文档、跨度和文本内容进行精确去重
        deduplicated_notes = self._deduplicate_notes_by_content(flattened_notes)
        
        duplicate_count = len(flattened_notes) - len(deduplicated_notes)
        logger.info(f"去重完成：移除 {duplicate_count} 条重复笔记，保留 {len(deduplicated_notes)} 条唯一笔记")
        
        return deduplicated_notes
    
    def _deduplicate_notes_by_content(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于内容进行去重，同文档同跨度/同文本的事实合并为一条，保留最高重要性分或合并元数据"""
        if not notes:
            return []
            
        # 创建去重键到笔记列表的映射
        dedup_groups = defaultdict(list)
        content_only_groups = defaultdict(list)  # 纯内容去重组
        
        for note in notes:
            # 生成去重键：文档路径 + 跨度 + 文本内容的哈希
            source_info = note.get('source_info', {})
            file_path = source_info.get('file_path', '')
            span_start = source_info.get('span_start', 0)
            span_end = source_info.get('span_end', 0)
            content = str(note.get('content', '')).strip()
            
            # 创建精确去重键（文档+位置+内容）
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            dedup_key = f"{file_path}:{span_start}-{span_end}:{content_hash}"
            dedup_groups[dedup_key].append(note)
            
            # 创建内容去重键（仅内容，用于跨文档去重）
            content_only_groups[content_hash].append(note)
        
        # 统计去重信息
        exact_duplicates = sum(1 for group in dedup_groups.values() if len(group) > 1)
        content_duplicates = sum(1 for group in content_only_groups.values() if len(group) > 1)
        
        logger.info(f"去重分析：精确重复组 {exact_duplicates} 个，内容重复组 {content_duplicates} 个")
        
        # 对每个去重组进行合并
        deduplicated_notes = []
        merged_count = 0
        
        for dedup_key, group_notes in dedup_groups.items():
            if len(group_notes) == 1:
                deduplicated_notes.append(group_notes[0])
            else:
                # 合并重复笔记，保留最高重要性分或合并元数据
                merged_note = self._merge_duplicate_notes(group_notes)
                deduplicated_notes.append(merged_note)
                merged_count += 1
        
        logger.info(f"去重合并：处理 {merged_count} 个重复组，最终保留 {len(deduplicated_notes)} 条唯一笔记")
        return deduplicated_notes
    
    def _merge_duplicate_notes(self, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并重复的笔记，保留最高重要性分或合并元数据"""
        if not notes:
            return {}
        
        # 按重要性分数排序，选择最高分的作为基础
        sorted_notes = sorted(notes, key=lambda x: x.get('importance_score', 0), reverse=True)
        base_note = sorted_notes[0].copy()
        
        # 合并元数据
        all_ids = []
        all_sources = []
        total_confidence = 0
        count = 0
        
        for note in notes:
            if note.get('id'):
                all_ids.append(note['id'])
            if note.get('source_info'):
                all_sources.append(note['source_info'])
            if 'confidence' in note:
                total_confidence += note['confidence']
                count += 1
        
        # 更新合并后的元数据
        base_note['merged_from_ids'] = all_ids
        base_note['merged_sources'] = all_sources
        base_note['duplicate_count'] = len(notes)
        
        if count > 0:
            base_note['average_confidence'] = total_confidence / count
        
        return base_note
    
    def _calculate_note_generation_stats(self, chunks: List[Dict[str, Any]], 
                                       raw_notes: List[Dict[str, Any]], 
                                       valid_notes: List[Dict[str, Any]],
                                       enhanced_notes: List[Dict[str, Any]],
                                       final_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算笔记生成的详细统计指标，记录每千token产出笔记数、平均每句事实数、被过滤比例"""
        # 计算总token数和句子数
        total_tokens = sum(len(chunk.get('content', '').split()) for chunk in chunks)
        total_chars = sum(len(chunk.get('content', '')) for chunk in chunks)
        total_sentences = sum(chunk.get('content', '').count('.') + chunk.get('content', '').count('!') + chunk.get('content', '').count('?') for chunk in chunks)
        
        # 计算每千token产出笔记数
        notes_per_1k_tokens = (len(final_notes) / max(total_tokens, 1)) * 1000
        
        # 计算平均每句事实数
        avg_facts_per_sentence = len(final_notes) / max(total_sentences, 1)
        
        # 计算各阶段过滤统计
        quality_filtered = len(raw_notes) - len(valid_notes)  # 质量验证过滤
        enhancement_change = len(enhanced_notes) - len(valid_notes)  # 增强阶段变化
        duplicate_filtered = len(enhanced_notes) - len(final_notes)  # 去重过滤
        
        # 计算过滤比例
        total_raw = max(len(raw_notes), 1)
        quality_filter_ratio = quality_filtered / total_raw
        duplicate_filter_ratio = duplicate_filtered / max(len(enhanced_notes), 1)
        total_retention_ratio = len(final_notes) / total_raw
        
        # 计算内容质量统计
        length_filtered = sum(1 for note in raw_notes if len(str(note.get('content', ''))) < 10)
        score_filtered = sum(1 for note in raw_notes if note.get('importance_score', 0) < 0.3)
        
        # 计算多事实笔记统计
        multi_fact_notes = sum(1 for note in enhanced_notes if isinstance(note.get('content'), list) and len(note['content']) > 1)
        single_fact_notes = len(enhanced_notes) - multi_fact_notes
        
        # 计算平均笔记长度
        avg_note_length = sum(len(str(note.get('content', ''))) for note in final_notes) / max(len(final_notes), 1)
        
        # 计算实体和关键词统计
        total_entities = sum(len(note.get('entities', [])) for note in final_notes)
        total_keywords = sum(len(note.get('keywords', [])) for note in final_notes)
        unique_entities = len(set(entity for note in final_notes for entity in note.get('entities', [])))
        unique_keywords = len(set(keyword for note in final_notes for keyword in note.get('keywords', [])))
        
        return {
            # 基础统计
            'total_tokens': total_tokens,
            'total_chars': total_chars,
            'total_sentences': total_sentences,
            'chunk_count': len(chunks),
            
            # 各阶段笔记数量
            'raw_notes_count': len(raw_notes),
            'valid_notes_count': len(valid_notes),
            'enhanced_notes_count': len(enhanced_notes),
            'final_notes_count': len(final_notes),
            
            # 核心指标
            'notes_per_1k_tokens': round(notes_per_1k_tokens, 2),
            'avg_facts_per_sentence': round(avg_facts_per_sentence, 3),
            'avg_note_length': round(avg_note_length, 1),
            
            # 过滤统计
            'quality_filtered': quality_filtered,
            'duplicate_filtered': duplicate_filtered,
            'length_filtered': length_filtered,
            'score_filtered': score_filtered,
            
            # 过滤比例
            'quality_filter_ratio': round(quality_filter_ratio, 3),
            'duplicate_filter_ratio': round(duplicate_filter_ratio, 3),
            'total_retention_ratio': round(total_retention_ratio, 3),
            
            # 多事实统计
            'multi_fact_notes': multi_fact_notes,
            'single_fact_notes': single_fact_notes,
            'multi_fact_ratio': round(multi_fact_notes / max(len(enhanced_notes), 1), 3),
            
            # 实体和关键词统计
            'total_entities': total_entities,
            'total_keywords': total_keywords,
            'unique_entities': unique_entities,
            'unique_keywords': unique_keywords,
            'avg_entities_per_note': round(total_entities / max(len(final_notes), 1), 2),
            'avg_keywords_per_note': round(total_keywords / max(len(final_notes), 1), 2),
            
            # 处理效率
            'processing_efficiency': round(len(final_notes) / max(total_chars / 1000, 1), 2),  # 每千字符产出笔记数
            'timestamp': self._get_timestamp()
        }
    
    def _log_processing_stats(self, stats: Dict[str, Any]):
        """将统计指标写入日志，便于A/B测试分析，提供详细的多事实处理统计"""
        logger.info("=== 原子笔记生成统计指标 ===")
        
        # 基础处理统计
        logger.info(f"📊 基础统计:")
        logger.info(f"  - 处理文档块: {stats['chunk_count']} 个")
        logger.info(f"  - 总token数: {stats['total_tokens']:,}")
        logger.info(f"  - 总字符数: {stats['total_chars']:,}")
        logger.info(f"  - 总句子数: {stats['total_sentences']:,}")
        
        # 各阶段笔记数量
        logger.info(f"📝 笔记处理流程:")
        logger.info(f"  - 原始生成: {stats['raw_notes_count']} 条")
        logger.info(f"  - 质量验证: {stats['valid_notes_count']} 条 (过滤 {stats['quality_filtered']} 条)")
        logger.info(f"  - 关系增强: {stats['enhanced_notes_count']} 条")
        logger.info(f"  - 去重后: {stats['final_notes_count']} 条 (去重 {stats['duplicate_filtered']} 条)")
        
        # 核心效率指标
        logger.info(f"⚡ 核心指标:")
        logger.info(f"  - 每千token产出笔记数: {stats['notes_per_1k_tokens']}")
        logger.info(f"  - 平均每句事实数: {stats['avg_facts_per_sentence']}")
        logger.info(f"  - 处理效率: {stats['processing_efficiency']} 笔记/千字符")
        logger.info(f"  - 平均笔记长度: {stats['avg_note_length']} 字符")
        
        # 过滤和质量控制
        logger.info(f"🔍 质量控制:")
        logger.info(f"  - 质量过滤率: {stats['quality_filter_ratio']*100:.1f}% ({stats['quality_filtered']} 条)")
        logger.info(f"  - 去重过滤率: {stats['duplicate_filter_ratio']*100:.1f}% ({stats['duplicate_filtered']} 条)")
        logger.info(f"  - 总保留率: {stats['total_retention_ratio']*100:.1f}%")
        logger.info(f"  - 长度过滤: {stats['length_filtered']} 条")
        logger.info(f"  - 分数过滤: {stats['score_filtered']} 条")
        
        # 多事实处理统计
        logger.info(f"📋 多事实处理:")
        logger.info(f"  - 多事实笔记: {stats['multi_fact_notes']} 条 ({stats['multi_fact_ratio']*100:.1f}%)")
        logger.info(f"  - 单事实笔记: {stats['single_fact_notes']} 条")
        
        # 实体和关键词统计
        logger.info(f"🏷️ 实体和关键词:")
        logger.info(f"  - 总实体数: {stats['total_entities']} (唯一: {stats['unique_entities']})")
        logger.info(f"  - 总关键词数: {stats['total_keywords']} (唯一: {stats['unique_keywords']})")
        logger.info(f"  - 平均每笔记实体数: {stats['avg_entities_per_note']}")
        logger.info(f"  - 平均每笔记关键词数: {stats['avg_keywords_per_note']}")
        
        logger.info(f"⏰ 处理时间: {stats['timestamp']}")
        logger.info("=== 统计指标结束 ===")
        
        # 结构化格式记录，便于A/B测试和自动化分析
        logger.info(f"ATOMIC_NOTE_STATS_JSON: {json.dumps(stats, ensure_ascii=False)}")
        
        # 关键指标摘要，便于快速对比
        summary_stats = {
            'notes_per_1k_tokens': stats['notes_per_1k_tokens'],
            'avg_facts_per_sentence': stats['avg_facts_per_sentence'],
            'total_retention_ratio': stats['total_retention_ratio'],
            'duplicate_filter_ratio': stats['duplicate_filter_ratio'],
            'multi_fact_ratio': stats['multi_fact_ratio'],
            'processing_efficiency': stats['processing_efficiency'],
            'timestamp': stats['timestamp']
        }
        logger.info(f"ATOMIC_NOTE_SUMMARY: {json.dumps(summary_stats, ensure_ascii=False)}")
    
    
    def _calculate_processing_stats(self, file_paths: List[str],
                                   atomic_notes: List[Dict[str, Any]],
                                   clustering_result: Dict[str, Any],
                                   chunk_count: int,
                                   note_count: int) -> Dict[str, Any]:
        """计算处理统计信息"""
        stats = {
            'files_processed': len(file_paths),
            'chunks_created': chunk_count,
            'atomic_notes_created': note_count,
            'topic_pools_created': len(clustering_result.get('topic_pools', [])),
            'clusters_found': clustering_result.get('cluster_info', {}).get('n_clusters', 0),
            'noise_notes': clustering_result.get('cluster_info', {}).get('n_noise', 0),
            'silhouette_score': clustering_result.get('cluster_info', {}).get('silhouette_score', 0.0),
            'processing_time': self._get_timestamp()
        }
        
        # 计算平均笔记长度
        if atomic_notes:
            avg_note_length = sum(len(note.get('content', '')) for note in atomic_notes) / len(atomic_notes)
            stats['avg_note_length'] = avg_note_length
        
        # 计算关键词和实体统计
        all_keywords = []
        all_entities = []
        for note in atomic_notes:
            all_keywords.extend(note.get('keywords', []))
            all_entities.extend(note.get('entities', []))
        
        stats['unique_keywords'] = len(set(all_keywords))
        stats['unique_entities'] = len(set(all_entities))
        
        return stats
    
    def _update_processing_cache(self, file_paths: List[str], result: Dict[str, Any]):
        """更新处理缓存"""
        for file_path in tqdm(file_paths, desc="Updating cache"):
            self.incremental_processor.update_file_cache(file_path, {
                'atomic_notes_count': len([n for n in result['atomic_notes']
                                         if n.get('source_info', {}).get('file_path') == file_path]),
                'processing_stats': result['processing_stats']
            })
    
    def _load_cached_results(self, file_paths: List[str]) -> Dict[str, Any]:
        """加载缓存的处理结果"""
        logger.info("Loading cached results from disk")

        atomic_file = os.path.join(self.processed_docs_path, "atomic_notes.json")
        cluster_file = os.path.join(self.processed_docs_path, "clustering.json")
        embed_file = os.path.join(self.processed_docs_path, "embeddings.npy")
        graph_file = os.path.join(self.processed_docs_path, "graph.json")
        result_file = os.path.join(self.processed_docs_path, "result.json")

        atomic_notes = []
        topic_pools = []
        cluster_info = {}
        graph_data = {}
        embeddings = None
        processing_stats = None

        if os.path.exists(atomic_file):
            try:
                atomic_notes = FileUtils.read_json(atomic_file)
            except Exception as e:
                logger.warning(f"Failed to load atomic notes: {e}")

        if os.path.exists(cluster_file):
            try:
                cluster_data = FileUtils.read_json(cluster_file)
                topic_pools = cluster_data.get('topic_pools', [])
                cluster_info = cluster_data.get('cluster_info', {})
                atomic_notes = cluster_data.get('clustered_notes', atomic_notes)
            except Exception as e:
                logger.warning(f"Failed to load clustering result: {e}")

        if os.path.exists(embed_file):
            try:
                embeddings = np.load(embed_file)
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")

        if os.path.exists(graph_file):
            try:
                graph_data = FileUtils.read_json(graph_file)
            except Exception as e:
                logger.warning(f"Failed to load graph data: {e}")

        if os.path.exists(result_file):
            try:
                processing_stats = FileUtils.read_json(result_file).get('processing_stats')
            except Exception as e:
                logger.warning(f"Failed to load processing stats: {e}")

        if not processing_stats:
            processing_stats = {
                'files_processed': len(file_paths),
                'loaded_from_cache': True
            }

        return {
            'atomic_notes': atomic_notes,
            'topic_pools': topic_pools,
            'cluster_info': cluster_info,
            'graph_data': graph_data,
            'embeddings': embeddings,
            'processing_stats': processing_stats
        }
    
    def process_single_document(self, file_path: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """处理单个文档"""
        return self.process_documents([file_path], force_reprocess)
    
    def get_processing_status(self, file_paths: List[str]) -> Dict[str, Any]:
        """获取文档处理状态"""
        processing_plan = self.incremental_processor.get_processing_plan(file_paths)
        cache_stats = self.incremental_processor.get_cache_statistics()
        
        return {
            'processing_plan': processing_plan,
            'cache_statistics': cache_stats,
            'can_skip_processing': processing_plan['can_skip_processing']
        }
    
    def clear_processing_cache(self, file_patterns: List[str] = None):
        """清理处理缓存"""
        self.incremental_processor.clear_cache(file_patterns)
        logger.info("Processing cache cleared")
    
    def validate_and_repair_cache(self):
        """验证和修复缓存"""
        validation_result = self.incremental_processor.validate_cache_integrity()
        
        issues_found = any(validation_result.values())
        if issues_found:
            logger.warning("Cache integrity issues found, repairing...")
            self.incremental_processor.repair_cache(validation_result)
            logger.info("Cache repair completed")
        else:
            logger.info("Cache integrity check passed")
        
        return validation_result
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self.llm, 'cleanup'):
            self.llm.cleanup()
        if hasattr(self.embedding_manager, 'cleanup'):
            self.embedding_manager.cleanup()
        logger.info("Document processor cleanup completed")