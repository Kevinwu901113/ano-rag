import os
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
from tqdm import tqdm
import networkx as nx
from networkx.readwrite import json_graph
from .chunker import DocumentChunker
from .clustering import TopicClustering
from .incremental_processor import IncrementalProcessor
from llm import AtomicNoteGenerator, LocalLLM
from utils import BatchProcessor, FileUtils
from utils.enhanced_ner import EnhancedNER
from config import config
from vector_store import EmbeddingManager
from graph import GraphBuilder

class DocumentProcessor:
    """文档处理器主类，整合所有文档处理功能"""
    
    def __init__(self, output_dir: Optional[str] = None):
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
        self.incremental_processor = IncrementalProcessor(cache_dir=cache_dir)
        self.llm = LocalLLM()
        self.atomic_note_generator = AtomicNoteGenerator(self.llm)
        self.batch_processor = BatchProcessor(
            batch_size=config.get('document.batch_size', 32),
            use_gpu=config.get('performance.use_gpu', True)
        )
        self.embedding_manager = EmbeddingManager()
        self.graph_builder = GraphBuilder()
        
        # 存储路径，默认使用配置中的工作目录
        self.processed_docs_path = output_dir or config.get('storage.work_dir') or config.get('storage.processed_docs_path', './data/processed')
        FileUtils.ensure_dir(self.processed_docs_path)
        
    def process_documents(self, file_paths: List[str], force_reprocess: bool = False, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """处理文档的主要入口点"""
        logger.info(f"Starting document processing for {len(file_paths)} files")

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
            atomic_notes = self._generate_atomic_notes(all_chunks)
            FileUtils.write_json(atomic_notes, atomic_file)
            
            # 保存被标记为需要重写的摘要（如果启用了摘要校验器）
            summary_auditor_enabled = config.get('summary_auditor.enabled', False)
            if summary_auditor_enabled:
                try:
                    from utils.summary_auditor import SummaryAuditor
                    auditor = SummaryAuditor()
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
        
        # 保存处理结果
        result = {
            'atomic_notes': clustering_result['clustered_notes'],
            'topic_pools': clustering_result['topic_pools'],
            'cluster_info': clustering_result['cluster_info'],
            'graph_data': graph_data,
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
    
    def _generate_atomic_notes(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成原子笔记"""
        try:
            # 使用批处理生成原子笔记
            atomic_notes = self.atomic_note_generator.generate_atomic_notes(chunks)
            
            # 验证原子笔记质量
            valid_notes = self.atomic_note_generator.validate_atomic_notes(atomic_notes)

            # 实体归一化和追踪增强
            valid_notes = EnhancedNER().enhance_entity_tracing(valid_notes)

            # 增强笔记关系
            enhanced_notes = self.atomic_note_generator.enhance_notes_with_relations(valid_notes)
            
            logger.info(f"Generated {len(enhanced_notes)} atomic notes")
            return enhanced_notes
            
        except Exception as e:
            logger.error(f"Failed to generate atomic notes: {e}")
            return []
    
    
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