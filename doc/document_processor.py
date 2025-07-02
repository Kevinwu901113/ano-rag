import os
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
from .chunker import DocumentChunker
from .clustering import TopicClustering
from .incremental_processor import IncrementalProcessor
from llm import AtomicNoteGenerator, LocalLLM
from utils import BatchProcessor, FileUtils
from config import config

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
        
        # 确定需要处理的文件
        files_to_process = processing_plan['files_to_process']
        if force_reprocess:
            files_to_process = file_paths
        
        logger.info(f"Processing {len(files_to_process)} files")
        
        chunk_file = os.path.join(self.processed_docs_path, "chunks.jsonl")
        if not force_reprocess and os.path.exists(chunk_file):
            logger.info(f"Loading chunks from {chunk_file}")
            all_chunks = FileUtils.read_jsonl(chunk_file)
        else:
            logger.info("Step 1: Document chunking")
            all_chunks = self._chunk_documents(files_to_process)
            FileUtils.write_jsonl(all_chunks, chunk_file)
        
        if not all_chunks:
            logger.warning("No chunks created from documents")
            return {'atomic_notes': [], 'topic_pools': [], 'processing_stats': {}}
        
        atomic_file = os.path.join(self.processed_docs_path, "atomic_notes.json")
        if not force_reprocess and os.path.exists(atomic_file):
            logger.info(f"Loading atomic notes from {atomic_file}")
            atomic_notes = FileUtils.read_json(atomic_file)
        else:
            logger.info("Step 2: Generating atomic notes")
            atomic_notes = self._generate_atomic_notes(all_chunks)
            FileUtils.write_json(atomic_notes, atomic_file)
        
        if not atomic_notes:
            logger.warning("No atomic notes generated")
            return {'atomic_notes': [], 'topic_pools': [], 'processing_stats': {}}
        
        embed_file = os.path.join(self.processed_docs_path, "embeddings.npy")
        if not force_reprocess and os.path.exists(embed_file):
            logger.info(f"Loading embeddings from {embed_file}")
            embeddings = np.load(embed_file)
        else:
            logger.info("Step 3: Creating embeddings (placeholder)")
            embeddings = self._create_embeddings_placeholder(atomic_notes)
            np.save(embed_file, embeddings)
        
        cluster_file = os.path.join(self.processed_docs_path, "clustering.json")
        if not force_reprocess and os.path.exists(cluster_file):
            logger.info(f"Loading clustering from {cluster_file}")
            clustering_result = FileUtils.read_json(cluster_file)
        else:
            logger.info("Step 4: Topic clustering")
            clustering_result = self.clustering.cluster_notes(atomic_notes, embeddings)
            FileUtils.write_json(clustering_result, cluster_file)
        
        graph_file = os.path.join(self.processed_docs_path, "graph.json")
        if not force_reprocess and os.path.exists(graph_file):
            logger.info(f"Loading graph from {graph_file}")
            graph_data = FileUtils.read_json(graph_file)
        else:
            logger.info("Step 5: Building graph relationships (placeholder)")
            graph_data = self._build_graph_placeholder(clustering_result['clustered_notes'])
            FileUtils.write_json(graph_data, graph_file)
        
        # 保存处理结果
        result = {
            'atomic_notes': clustering_result['clustered_notes'],
            'topic_pools': clustering_result['topic_pools'],
            'cluster_info': clustering_result['cluster_info'],
            'graph_data': graph_data,
            'processing_stats': self._calculate_processing_stats(files_to_process, atomic_notes, clustering_result)
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
        
        for file_path in file_paths:
            try:
                chunks = self.chunker.chunk_document(file_path)
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
            
            # 增强笔记关系
            enhanced_notes = self.atomic_note_generator.enhance_notes_with_relations(valid_notes)
            
            logger.info(f"Generated {len(enhanced_notes)} atomic notes")
            return enhanced_notes
            
        except Exception as e:
            logger.error(f"Failed to generate atomic notes: {e}")
            return []
    
    def _create_embeddings_placeholder(self, atomic_notes: List[Dict[str, Any]]) -> 'np.ndarray':
        """创建向量嵌入的占位符实现"""
        import numpy as np
        
        # 这里是占位符实现，实际的嵌入生成将在vector_store模块中实现
        # 为了聚类能够工作，我们创建随机嵌入
        n_notes = len(atomic_notes)
        embedding_dim = 768  # 常见的嵌入维度
        
        # 基于文本内容创建简单的特征向量
        embeddings = []
        for note in atomic_notes:
            # 简单的特征提取：基于关键词和实体
            features = np.zeros(embedding_dim)
            
            # 基于文本长度
            text_length = len(note.get('content', ''))
            features[0] = min(text_length / 1000, 1.0)
            
            # 基于关键词数量
            keywords_count = len(note.get('keywords', []))
            features[1] = min(keywords_count / 10, 1.0)
            
            # 基于实体数量
            entities_count = len(note.get('entities', []))
            features[2] = min(entities_count / 10, 1.0)
            
            # 添加一些随机噪声以模拟真实嵌入
            features[3:] = np.random.normal(0, 0.1, embedding_dim - 3)
            
            embeddings.append(features)
        
        return np.array(embeddings)
    
    def _build_graph_placeholder(self, atomic_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建图谱的占位符实现"""
        # 这里是占位符实现，实际的图谱构建将在graph模块中实现
        nodes = []
        edges = []
        
        # 创建节点
        for note in atomic_notes:
            node = {
                'id': note.get('note_id'),
                'type': 'atomic_note',
                'content': note.get('content', ''),
                'cluster_id': note.get('cluster_id'),
                'keywords': note.get('keywords', []),
                'entities': note.get('entities', [])
            }
            nodes.append(node)
        
        # 创建边（基于现有的关系信息）
        for note in atomic_notes:
            note_id = note.get('note_id')
            
            # 相关笔记关系
            for related in note.get('related_notes', []):
                edge = {
                    'source': note_id,
                    'target': related.get('note_id'),
                    'type': related.get('relation_type', 'related'),
                    'weight': related.get('similarity', 0.5)
                }
                edges.append(edge)
            
            # 实体共现关系
            for entity_rel in note.get('entity_relations', []):
                edge = {
                    'source': note_id,
                    'target': entity_rel.get('target_note_id'),
                    'type': 'entity_coexistence',
                    'weight': len(entity_rel.get('common_entities', [])) * 0.1
                }
                edges.append(edge)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'node_count': len(nodes),
            'edge_count': len(edges)
        }
    
    def _calculate_processing_stats(self, file_paths: List[str], 
                                   atomic_notes: List[Dict[str, Any]], 
                                   clustering_result: Dict[str, Any]) -> Dict[str, Any]:
        """计算处理统计信息"""
        stats = {
            'files_processed': len(file_paths),
            'atomic_notes_created': len(atomic_notes),
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
        for file_path in file_paths:
            self.incremental_processor.update_file_cache(file_path, {
                'atomic_notes_count': len([n for n in result['atomic_notes'] 
                                         if n.get('source_info', {}).get('file_path') == file_path]),
                'processing_stats': result['processing_stats']
            })
    
    def _load_cached_results(self, file_paths: List[str]) -> Dict[str, Any]:
        """加载缓存的处理结果"""
        # 这里应该从存储中加载已处理的结果
        # 目前返回空结果，实际实现需要从vector_store和graph模块加载
        logger.info("Loading cached results (placeholder implementation)")
        
        return {
            'atomic_notes': [],
            'topic_pools': [],
            'processing_stats': {
                'files_processed': len(file_paths),
                'loaded_from_cache': True
            }
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
        logger.info("Document processor cleanup completed")