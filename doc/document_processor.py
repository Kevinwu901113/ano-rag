import asyncio
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import networkx as nx
from contextlib import nullcontext
from networkx.readwrite import json_graph
from .chunker import DocumentChunker
from .clustering import TopicClustering
from .incremental_processor import IncrementalProcessor
from llm import LocalLLM
from llm.vllm_atomic_note_generator import VllmAtomicNoteGenerator
from utils import BatchProcessor, FileUtils, JSONLProgressTracker
from utils.enhanced_ner import EnhancedNER
from utils.consistency_checker import ConsistencyChecker
from utils.note_jsonl_writer import get_global_note_writer
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
        
        # vLLM 原子笔记生成器在真正需要时再初始化，以便根据实际可用端点配置
        logger.info("Preparing VllmAtomicNoteGenerator for atomic note generation (lazy init)")
        self.atomic_note_generator: Optional[VllmAtomicNoteGenerator] = None
        self._env_vllm_endpoints: List[str] = []
        env_eps = os.getenv("ANO_RAG_VLLM_ENDPOINTS")
        if env_eps:
            parsed = [ep.strip() for ep in env_eps.split(",") if ep.strip()]
            if parsed:
                self._env_vllm_endpoints = parsed
                logger.info(f"Detected vLLM endpoints from environment: {self._env_vllm_endpoints}")
        self._active_vllm_endpoints: Optional[List[str]] = list(self._env_vllm_endpoints)
        self.batch_processor = BatchProcessor(
            batch_size=config.get('document.batch_size', 32),
            use_gpu=config.get('performance.use_gpu', True)
        )
        self.embedding_manager = EmbeddingManager()
        self.graph_builder = GraphBuilder(llm=self.llm)
        
        # 存储路径，默认使用配置中的工作目录
        self.processed_docs_path = output_dir or config.get('storage.work_dir') or config.get('storage.processed_docs_path') or './data/processed'
        FileUtils.ensure_dir(self.processed_docs_path)

        self._vllm_autostart_cfg = config.get('llm.note_generator.autostart', {})
        self._vllm_autostart_enabled = bool(self._vllm_autostart_cfg.get('enabled', False))
        disable_env = os.getenv("ANO_RAG_DISABLE_VLLM_AUTOSTART")
        if disable_env and disable_env.strip().lower() not in {"0", "false", "no"}:
            logger.info("Environment requests disabling vLLM autostart")
            self._vllm_autostart_enabled = False
        self._vllm_manager_context = None
        self._shard_name = os.getenv("ANO_RAG_SHARD_NAME")
        self._step2_barrier_dir = os.getenv("ANO_RAG_STEP2_BARRIER_DIR") or ""
        self._step2_total = self._safe_int(os.getenv("ANO_RAG_STEP2_TOTAL"), default=0)
        self._step2_timeout = self._safe_int(os.getenv("ANO_RAG_STEP2_BARRIER_TIMEOUT"), default=1800)
        poll_default = self._safe_int(os.getenv("ANO_RAG_STEP2_BARRIER_POLL"), default=5)
        self._step2_poll_interval = max(1, poll_default)
        self._step2_abort_marker = Path(self._step2_barrier_dir, "ABORT") if self._step2_barrier_dir else None
        if self._step2_barrier_dir and self._step2_total > 1 and self._shard_name:
            logger.info(
                f"Step 2 barrier configured for shard {self._shard_name} "
                f"(dir={self._step2_barrier_dir}, total={self._step2_total}, "
                f"timeout={self._step2_timeout}s, poll={self._step2_poll_interval}s)"
            )
        
    @staticmethod
    def _safe_int(value: Optional[str], default: int = 0) -> int:
        try:
            return int(value) if value is not None else default
        except (TypeError, ValueError):
            return default
    
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
            reset_partial = force_reprocess or chunks_updated
            self._active_vllm_endpoints = list(self._env_vllm_endpoints) if self._env_vllm_endpoints else None
            context_mgr = self._vllm_note_generation_context()
            try:
                if context_mgr:
                    with context_mgr as manager:
                        if manager and hasattr(manager, "get_ready_endpoints"):
                            ready_eps = manager.get_ready_endpoints()
                            if ready_eps:
                                logger.info(f"vLLM autostart ready endpoints: {ready_eps}")
                                self._active_vllm_endpoints = ready_eps
                            else:
                                failed = getattr(manager, "get_failed_servers", lambda: [])()
                                if failed:
                                    logger.warning(f"vLLM autostart reported startup failures: {failed}")
                        atomic_notes = self._generate_atomic_notes(
                            all_chunks,
                            progress_tracker,
                            reset_partial=reset_partial
                        )
                else:
                    atomic_notes = self._generate_atomic_notes(
                        all_chunks,
                        progress_tracker,
                        reset_partial=reset_partial
                    )
            finally:
                self._active_vllm_endpoints = list(self._env_vllm_endpoints) if self._env_vllm_endpoints else None
                self._shutdown_atomic_note_generator()
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
        self._step2_barrier_wait(note_count)
        self._record_step3_marker(note_count)
        
        if not atomic_notes:
            logger.warning("No atomic notes generated")
            return {'atomic_notes': [], 'topic_pools': [], 'processing_stats': {}}
        
        embed_file = os.path.join(self.processed_docs_path, "embeddings.npy")
        if (not force_reprocess and not chunks_updated
                and os.path.exists(embed_file)):
            logger.info(f"Loading embeddings from {embed_file}")
            embeddings = np.load(embed_file)
        else:
            logger.info("Stage checkpoint: Step 2 completed; preparing embeddings")
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

    def _shutdown_atomic_note_generator(self):
        """Try to close existing vLLM client before reconfiguration."""
        if not self.atomic_note_generator:
            return
        client = getattr(self.atomic_note_generator, "vllm_client", None)
        if not client:
            return
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(client.close())
            finally:
                loop.close()
        except Exception as exc:  # noqa: BLE001 - best effort cleanup
            logger.debug(f"Failed to close existing vLLM client cleanly: {exc}")
        finally:
            self.atomic_note_generator = None

    def _step2_barrier_wait(self, note_count: int):
        """在进入Step3前等待其他分片完成Step2，防止提前卸载vLLM。"""
        if (
            not self._step2_barrier_dir
            or not self._shard_name
            or self._step2_total <= 1
        ):
            return

        barrier_path = Path(self._step2_barrier_dir)
        try:
            barrier_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(f"Failed to prepare Step 2 barrier dir {barrier_path}: {exc}; continuing without sync")
            return

        marker_file = barrier_path / f"{self._shard_name}.step2"
        try:
            marker_file.write_text(f"{time.time()}\tnotes={note_count}\n")
        except Exception as exc:
            logger.warning(f"Failed to write Step 2 barrier marker for {self._shard_name}: {exc}")

        expected = max(1, self._step2_total)
        poll_interval = max(1, self._step2_poll_interval)
        start_time = time.time()
        deadline = start_time + self._step2_timeout if self._step2_timeout > 0 else None
        next_status_log = start_time
        status_interval = max(30, poll_interval)
        initial_reached = len(list(barrier_path.glob("*.step2")))

        logger.info(
            f"Shard {self._shard_name} completed Step 2 (notes={note_count}); "
            f"waiting for remaining shards ({initial_reached}/{expected} ready)"
        )

        abort_file = self._step2_abort_marker if self._step2_abort_marker else barrier_path / "ABORT"

        while True:
            markers = list(barrier_path.glob("*.step2"))
            reached = len(markers)
            if reached >= expected:
                logger.info(
                    f"All shards reached Step 2 barrier ({reached}/{expected}). Proceeding to Step 3."
                )
                return

            if abort_file.exists():
                logger.warning(
                    f"Detected Step 2 barrier abort signal at {abort_file}; "
                    f"proceeding without waiting for remaining shards ({reached}/{expected})."
                )
                return

            now = time.time()
            if deadline and now >= deadline:
                logger.warning(
                    f"Step 2 barrier timeout after {self._step2_timeout}s; "
                    f"continuing with Step 3 ({reached}/{expected} shards reached)."
                )
                return

            if now >= next_status_log:
                remaining = max(0, expected - reached)
                logger.info(
                    f"Waiting for {remaining} remaining shard(s) to finish Step 2 "
                    f"({reached}/{expected} ready)..."
                )
                next_status_log = now + status_interval

            time.sleep(poll_interval)

    def _record_step3_marker(self, note_count: int) -> None:
        """Persist a Step 3 readiness marker for barrier monitoring."""
        if (
            not self._step2_barrier_dir
            or not self._shard_name
            or self._step2_total <= 1
        ):
            return

        marker = Path(self._step2_barrier_dir, f"{self._shard_name}.step3")
        try:
            marker.write_text(f"{time.time()}\tnotes={note_count}\n")
        except Exception as exc:  # noqa: BLE001 - best effort marker
            logger.warning(
                "Failed to write Step 3 barrier marker for %s: %s",
                self._shard_name,
                exc,
            )

    def _ensure_atomic_note_generator(self):
        """Initialize or reconfigure the vLLM note generator as needed."""
        endpoints = self._active_vllm_endpoints or []
        current = self.atomic_note_generator

        if endpoints:
            configured = getattr(current, "configured_endpoints", [])
            if current is None or configured != endpoints:
                if current is not None:
                    self._shutdown_atomic_note_generator()
                logger.info(f"Initializing VllmAtomicNoteGenerator with endpoints: {endpoints}")
                self.atomic_note_generator = VllmAtomicNoteGenerator(self.llm, endpoints=endpoints)
        else:
            if current is None:
                logger.info("Initializing VllmAtomicNoteGenerator with default configuration")
                self.atomic_note_generator = VllmAtomicNoteGenerator(self.llm)

    def _generate_atomic_notes(
        self,
        chunks: List[Dict[str, Any]],
        progress_tracker: Optional[JSONLProgressTracker] = None,
        reset_partial: bool = False
    ) -> List[Dict[str, Any]]:
        """生成原子笔记"""
        self._ensure_atomic_note_generator()
        try:
            partial_atomic_file = os.path.join(self.processed_docs_path, "atomic_notes.partial.jsonl")
            if reset_partial or not os.path.exists(partial_atomic_file):
                FileUtils.write_file(partial_atomic_file, "")
                logger.info(f"Initialized partial atomic notes file: {partial_atomic_file}")
                resume_partial = False
            else:
                logger.info(f"Appending to existing partial atomic notes file: {partial_atomic_file}")
                resume_partial = True

            def _append_partial_notes(notes: List[Dict[str, Any]]):
                if not notes:
                    return
                for note in notes:
                    FileUtils.append_jsonl_atomic(partial_atomic_file, note)

            # 使用批处理生成原子笔记
            atomic_notes = self.atomic_note_generator.generate_atomic_notes(
                chunks,
                progress_tracker,
                on_notes_batch=_append_partial_notes
            )
            
            # 验证原子笔记质量
            valid_notes = self.atomic_note_generator.validate_atomic_notes(atomic_notes)

            # 实体归一化和追踪增强
            valid_notes = EnhancedNER().enhance_entity_tracing(valid_notes)

            # 增强笔记关系
            enhanced_notes = self.atomic_note_generator.enhance_notes_with_relations(valid_notes)
            
            # 写入JSONL文件
            try:
                # 使用当前处理目录作为JSONL文件的输出目录，与atomic_notes.json等文件保持一致
                work_dir = self.processed_docs_path or config.get('storage.work_dir')
                jsonl_writer = get_global_note_writer(
                    work_dir,
                    reset=reset_partial or not resume_partial
                )
                for note in enhanced_notes:
                    # 从source_info中获取问题ID
                    source_info = note.get('source_info', {})
                    question_id = (source_info.get('file_name', '') or 
                                 source_info.get('document_id', '') or 
                                 source_info.get('file_path', ''))
                    
                    # 直接使用笔记中已设置的idx字段
                    idx = note.get('idx', note.get('chunk_index', 0))
                    
                    jsonl_writer.write_note(note, question_id, idx)
                
                logger.info(f"Written {len(enhanced_notes)} notes to note.jsonl")
            except Exception as e:
                logger.error(f"Failed to write notes to JSONL: {e}")

            try:
                FileUtils.write_jsonl(enhanced_notes, partial_atomic_file)
                logger.info(f"Persisted {len(enhanced_notes)} atomic notes to partial JSONL: {partial_atomic_file}")
            except Exception as e:
                logger.error(f"Failed to persist partial atomic notes: {e}")

            logger.info(f"Generated {len(enhanced_notes)} atomic notes")
            return enhanced_notes
            
        except Exception as e:
            logger.error(f"Failed to generate atomic notes: {e}")
            return []
        finally:
            # ensure the vLLM client is released after note generation
            self._shutdown_atomic_note_generator()
    
    
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

    def _vllm_note_generation_context(self):
        """Context manager to optionally autostart/teardown vLLM servers."""
        if not self._vllm_autostart_enabled:
            return nullcontext()

        try:
            from utils.vllm_server_manager import VLLMServerManager
            return VLLMServerManager(self._vllm_autostart_cfg)
        except Exception as exc:
            logger.error(f"Failed to initialize vLLM server manager, falling back to manual mode: {exc}")
            return nullcontext()
    
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
