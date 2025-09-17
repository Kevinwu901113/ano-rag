#!/usr/bin/env python3
"""
并行处理统一接口

该模块提供了标准化的并行处理接口，包含针对不同任务类型的具体处理器实现。
支持文档处理、查询处理、Musique数据集处理等多种场景。
"""

import os
import json
import shutil
from typing import List, Dict, Any, Optional, Union, Callable
from loguru import logger

from .parallel_engine import (
    ParallelEngine, TaskProcessor, ParallelTask, ParallelResult, 
    ParallelStrategy, ProcessingMode
)
from llm import LocalLLM


class ParallelInterface:
    """统一的并行处理接口"""
    
    def __init__(self, 
                 max_workers: int = 4,
                 processing_mode: ProcessingMode = ProcessingMode.AUTO,
                 default_strategy: ParallelStrategy = ParallelStrategy.HYBRID,
                 debug: bool = False):
        
        self.engine = ParallelEngine(
            max_workers=max_workers,
            processing_mode=processing_mode,
            default_strategy=default_strategy,
            enable_stats=True,
            debug=debug
        )
        self.debug = debug
        
        logger.info(f"ParallelInterface initialized with {max_workers} workers")
    
    def process_documents(self, 
                         documents: List[Dict[str, Any]], 
                         output_dir: str,
                         force_reprocess: bool = False,
                         strategy: Optional[ParallelStrategy] = None) -> List[Dict[str, Any]]:
        """并行处理文档"""
        
        # 创建文档处理任务
        tasks = []
        for i, doc in enumerate(documents):
            task = ParallelTask(
                task_id=f"doc_{i}",
                data=doc,
                task_type="document",
                metadata={
                    'output_dir': output_dir,
                    'force_reprocess': force_reprocess
                }
            )
            tasks.append(task)
        
        # 使用文档任务处理器
        processor = DocumentTaskProcessor(output_dir, self.debug)
        results = self.engine.process_tasks(tasks, processor, strategy)
        
        # 提取处理结果
        processed_docs = []
        for result in results:
            if result.success:
                processed_docs.append(result.aggregated_result)
            else:
                logger.error(f"Document processing failed: {result.error_message}")
        
        return processed_docs
    
    def process_queries(self, 
                       queries: List[Dict[str, Any]], 
                       knowledge_base: Dict[str, Any],
                       strategy: Optional[ParallelStrategy] = None) -> List[Dict[str, Any]]:
        """并行处理查询"""
        
        # 创建查询处理任务
        tasks = []
        for i, query in enumerate(queries):
            task = ParallelTask(
                task_id=f"query_{i}",
                data=query,
                task_type="query",
                metadata={'knowledge_base': knowledge_base}
            )
            tasks.append(task)
        
        # 使用查询任务处理器
        processor = QueryTaskProcessor(knowledge_base, self.debug)
        results = self.engine.process_tasks(tasks, processor, strategy)
        
        # 提取查询结果
        query_results = []
        for result in results:
            if result.success:
                query_results.append(result.aggregated_result)
            else:
                logger.error(f"Query processing failed: {result.error_message}")
        
        return query_results
    
    def process_musique_dataset(self, 
                               items: List[Dict[str, Any]], 
                               base_work_dir: str,
                               strategy: Optional[ParallelStrategy] = None) -> List[Dict[str, Any]]:
        """并行处理Musique数据集"""
        
        # 创建Musique处理任务
        tasks = []
        for item in items:
            item_id = item.get('id', f'item_{len(tasks)}')
            task = ParallelTask(
                task_id=item_id,
                data=item,
                task_type="musique",
                metadata={'base_work_dir': base_work_dir}
            )
            tasks.append(task)
        
        # 使用Musique任务处理器
        processor = MusiqueTaskProcessor(base_work_dir, self.debug)
        results = self.engine.process_tasks(tasks, processor, strategy)
        
        # 提取处理结果
        musique_results = []
        for result in results:
            if result.success:
                musique_results.append(result.aggregated_result)
            else:
                logger.error(f"Musique processing failed: {result.error_message}")
                # 添加错误结果
                error_result = {
                    'id': result.task_id,
                    'predicted_answer': 'Processing failed',
                    'predicted_support_idxs': [],
                    'predicted_answerable': True,
                    'error': result.error_message
                }
                musique_results.append(error_result)
        
        return musique_results
    
    def get_performance_stats(self) -> Optional[Dict[str, Any]]:
        """获取性能统计信息"""
        stats = self.engine.get_stats()
        if stats:
            return {
                'total_tasks': stats.total_tasks,
                'successful_tasks': stats.successful_tasks,
                'failed_tasks': stats.failed_tasks,
                'success_rate': stats.successful_tasks / stats.total_tasks if stats.total_tasks > 0 else 0,
                'avg_processing_time': stats.avg_processing_time,
                'total_processing_time': stats.total_processing_time,
                'throughput': stats.throughput,
                'strategy_usage': stats.strategy_usage
            }
        return None
    
    def cleanup(self):
        """清理资源"""
        self.engine.cleanup()


class DocumentTaskProcessor(TaskProcessor):
    """文档任务处理器"""
    
    def __init__(self, output_dir: str, debug: bool = False):
        self.output_dir = output_dir
        self.debug = debug
    
    def process_single_task(self, task: ParallelTask, llm: LocalLLM, **kwargs) -> Dict[str, Any]:
        """处理单个文档任务"""
        from doc import DocumentProcessor
        
        document = task.data
        task_output_dir = os.path.join(self.output_dir, f"task_{task.task_id}")
        os.makedirs(task_output_dir, exist_ok=True)
        
        # 创建文档处理器
        processor = DocumentProcessor(output_dir=task_output_dir, llm=llm)
        
        # 如果文档是文件路径列表
        if isinstance(document, list) and all(isinstance(item, str) for item in document):
            result = processor.process_documents(document, 
                                               force_reprocess=task.metadata.get('force_reprocess', False),
                                               output_dir=task_output_dir)
        else:
            # 如果是文档内容，需要先保存为文件
            doc_file = os.path.join(task_output_dir, f"{task.task_id}.json")
            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False)
            
            result = processor.process_documents([doc_file], 
                                               force_reprocess=task.metadata.get('force_reprocess', False),
                                               output_dir=task_output_dir)
        
        # 添加任务信息
        result['task_id'] = task.task_id
        result['output_dir'] = task_output_dir
        
        return result
    
    def aggregate_results(self, results: List[Dict[str, Any]], task: ParallelTask) -> Dict[str, Any]:
        """聚合文档处理结果"""
        if not results:
            return {'task_id': task.task_id, 'atomic_notes': [], 'error': 'No results to aggregate'}
        
        # 选择原子笔记最多的结果
        best_result = max(results, key=lambda r: len(r.get('atomic_notes', [])))
        
        # 合并所有原子笔记（去重）
        all_notes = []
        seen_note_ids = set()
        
        for result in results:
            for note in result.get('atomic_notes', []):
                note_id = note.get('note_id', '')
                if note_id and note_id not in seen_note_ids:
                    seen_note_ids.add(note_id)
                    all_notes.append(note)
        
        aggregated = best_result.copy()
        aggregated['atomic_notes'] = all_notes
        aggregated['aggregated_from'] = len(results)
        
        return aggregated


class QueryTaskProcessor(TaskProcessor):
    """查询任务处理器"""
    
    def __init__(self, knowledge_base: Dict[str, Any], debug: bool = False):
        self.knowledge_base = knowledge_base
        self.debug = debug
    
    def process_single_task(self, task: ParallelTask, llm: LocalLLM, **kwargs) -> Dict[str, Any]:
        """处理单个查询任务"""
        from query import QueryProcessor
        
        query_data = task.data
        question = query_data.get('question', query_data.get('query', ''))
        
        # 从知识库获取必要信息
        atomic_notes = self.knowledge_base.get('atomic_notes', [])
        embeddings = self.knowledge_base.get('embeddings')
        graph_file = self.knowledge_base.get('graph_file')
        
        # 创建查询处理器
        query_processor = QueryProcessor(
            atomic_notes=atomic_notes,
            embeddings=embeddings,
            graph_file=graph_file,
            llm=llm
        )
        
        # 执行查询
        result = query_processor.process(question)
        
        # 添加任务信息
        result['task_id'] = task.task_id
        result['question'] = question
        
        return result
    
    def aggregate_results(self, results: List[Dict[str, Any]], task: ParallelTask) -> Dict[str, Any]:
        """聚合查询结果"""
        if not results:
            return {'task_id': task.task_id, 'answer': 'No results to aggregate'}
        
        # 选择最佳答案（优先非错误、非空答案）
        best_result = None
        for result in results:
            answer = result.get('answer', '')
            if answer and 'error' not in answer.lower() and 'no answer' not in answer.lower():
                if not best_result or len(answer) > len(best_result.get('answer', '')):
                    best_result = result
        
        if not best_result:
            best_result = results[0]  # 如果没有好的答案，使用第一个
        
        # 合并召回的笔记
        all_notes = []
        seen_note_ids = set()
        
        for result in results:
            for note in result.get('notes', []):
                note_id = note.get('note_id', '')
                if note_id and note_id not in seen_note_ids:
                    seen_note_ids.add(note_id)
                    all_notes.append(note)
        
        aggregated = best_result.copy()
        aggregated['notes'] = all_notes
        aggregated['aggregated_from'] = len(results)
        
        return aggregated


class MusiqueTaskProcessor(TaskProcessor):
    """Musique任务处理器"""
    
    def __init__(self, base_work_dir: str, debug: bool = False):
        self.base_work_dir = base_work_dir
        self.debug = debug
    
    def process_single_task(self, task: ParallelTask, llm: LocalLLM, **kwargs) -> Dict[str, Any]:
        """处理单个Musique任务"""
        from doc import DocumentProcessor
        from query import QueryProcessor
        
        item = task.data
        item_id = item.get('id', task.task_id)
        question = item.get('question', '')
        paragraphs = item.get('paragraphs', [])
        
        # 创建任务工作目录
        debug_dir = os.path.join(self.base_work_dir, "debug")
        task_work_dir = os.path.join(debug_dir, f"task_{item_id}")
        os.makedirs(task_work_dir, exist_ok=True)
        
        try:
            # 1. 创建段落文件
            paragraph_files = self._create_paragraph_files(item, task_work_dir)
            
            # 2. 处理文档（构建知识库）
            processor = DocumentProcessor(output_dir=task_work_dir, llm=llm)
            process_result = processor.process_documents(paragraph_files, 
                                                       force_reprocess=True, 
                                                       output_dir=task_work_dir)
            
            if not process_result.get('atomic_notes'):
                logger.warning(f"No atomic notes generated for item {item_id}")
                return {
                    'id': item_id,
                    'predicted_answer': 'No answer found',
                    'predicted_support_idxs': [],
                    'predicted_answerable': True
                }
            
            # 3. 查询处理
            atomic_notes = process_result['atomic_notes']
            
            # 加载必要的文件
            graph_file = os.path.join(task_work_dir, 'graph.json')
            embed_file = os.path.join(task_work_dir, 'embeddings.npy')
            
            embeddings = None
            if os.path.exists(embed_file):
                import numpy as np
                try:
                    embeddings = np.load(embed_file)
                except Exception as e:
                    logger.warning(f'Failed to load embeddings for {item_id}: {e}')
            
            query_processor = QueryProcessor(
                atomic_notes,
                embeddings,
                graph_file=graph_file if os.path.exists(graph_file) else None,
                vector_index_file=None,
                llm=llm
            )
            
            # 4. 执行查询
            query_result = query_processor.process(question)
            
            # 5. 提取结果
            predicted_answer = query_result.get('answer', 'No answer found')
            predicted_support_idxs = query_result.get('predicted_support_idxs', [])
            
            # 清理临时文件
            if not self.debug:
                for p in paragraph_files:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            
            result = {
                'id': item_id,
                'predicted_answer': predicted_answer,
                'predicted_support_idxs': predicted_support_idxs,
                'predicted_answerable': True,
                'task_id': task.task_id
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process Musique item {item_id}: {e}")
            return {
                'id': item_id,
                'predicted_answer': 'Error occurred during processing',
                'predicted_support_idxs': [],
                'predicted_answerable': True,
                'error': str(e),
                'task_id': task.task_id
            }
        finally:
            # 清理工作目录
            if not self.debug:
                try:
                    shutil.rmtree(task_work_dir)
                except:
                    pass
    
    def _create_paragraph_files(self, item: Dict[str, Any], work_dir: str) -> List[str]:
        """创建段落文件"""
        item_id = item.get('id', 'unknown')
        paragraphs = item.get('paragraphs', [])
        
        file_paths = []
        for i, para in enumerate(paragraphs):
            idx = para.get('idx', i)
            file_name = f"{item_id}_para_{idx}.json"
            file_path = os.path.join(work_dir, file_name)
            
            # 只保存段落信息，不包含question
            para_data = {
                'id': f"{item_id}_para_{idx}",
                'paragraphs': [para]
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(para_data, f, ensure_ascii=False)
            file_paths.append(file_path)
        
        return file_paths
    
    def aggregate_results(self, results: List[Dict[str, Any]], task: ParallelTask) -> Dict[str, Any]:
        """聚合Musique结果"""
        if not results:
            return {
                'id': task.task_id,
                'predicted_answer': 'No results to aggregate',
                'predicted_support_idxs': [],
                'predicted_answerable': True
            }
        
        # 选择最佳答案
        best_result = None
        for result in results:
            answer = result.get('predicted_answer', '')
            if answer and 'Error' not in answer and 'No answer found' not in answer:
                if not best_result or len(answer) > len(best_result.get('predicted_answer', '')):
                    best_result = result
        
        if not best_result:
            best_result = results[0]
        
        # 合并支持段落索引
        all_support_idxs = set()
        for result in results:
            all_support_idxs.update(result.get('predicted_support_idxs', []))
        
        aggregated = best_result.copy()
        aggregated['predicted_support_idxs'] = list(all_support_idxs)
        aggregated['aggregated_from'] = len(results)
        
        return aggregated


def create_parallel_interface(max_workers: int = 4, 
                             processing_mode: ProcessingMode = ProcessingMode.AUTO,
                             strategy: ParallelStrategy = ParallelStrategy.HYBRID,
                             debug: bool = False) -> ParallelInterface:
    """创建并行处理接口的工厂函数"""
    return ParallelInterface(
        max_workers=max_workers,
        processing_mode=processing_mode,
        default_strategy=strategy,
        debug=debug
    )