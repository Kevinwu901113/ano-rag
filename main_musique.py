#!/usr/bin/env python3
"""
Musique数据集批量处理脚本

该脚本用于批量处理musique测试集，对每个测试样本：
1. 提取id、paragraphs、question
2. 将每个段落视为独立文档，使用paragraphs构建知识库（process阶段）
   - 注意：question字段不参与文档构建过程，确保原子笔记只来源于段落内容
3. 使用question进行查询（query阶段）
4. 输出包含predicted_answer、predicted_support_idxs等信息的结果

处理方式符合项目标准：每个段落一个文件，用于构建笔记和图谱。
"""

import argparse
import json
import os
import shutil
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger

from doc import DocumentProcessor
from query import QueryProcessor
from config import config
from utils import FileUtils, setup_logging
from llm import LocalLLM


RESULT_ROOT = config.get('storage.result_root', 'result')


def get_latest_workdir() -> str:
    """获取最新的工作目录"""
    os.makedirs(RESULT_ROOT, exist_ok=True)
    subdirs = [d for d in os.listdir(RESULT_ROOT) if os.path.isdir(os.path.join(RESULT_ROOT, d))]
    if not subdirs:
        return create_new_workdir()
    latest = sorted(subdirs)[-1]
    return os.path.join(RESULT_ROOT, latest)


def create_new_workdir() -> str:
    """创建新的工作目录"""
    os.makedirs(RESULT_ROOT, exist_ok=True)
    existing = [int(d) for d in os.listdir(RESULT_ROOT) if d.isdigit()]
    next_idx = max(existing) + 1 if existing else 1
    work_dir = os.path.join(RESULT_ROOT, str(next_idx))
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


def create_item_workdir(base_work_dir: str, item_id: str) -> str:
    """为单个item创建工作目录"""
    item_work_dir = os.path.join(base_work_dir, f"item_{item_id}")
    os.makedirs(item_work_dir, exist_ok=True)
    return item_work_dir


class MusiqueProcessor:
    """Musique数据集处理器"""

    def __init__(self, max_workers: int = 4, debug: bool = False, work_dir: str = None, llm: Optional[LocalLLM] = None):
        self.max_workers = max_workers
        self.debug = debug  # 调试模式，不清理中间文件
        self.base_work_dir = work_dir or create_new_workdir()
        self.llm = llm or LocalLLM()
        logger.info(f"Using base work directory: {self.base_work_dir}")

    def _create_paragraph_files(self, item: Dict[str, Any], work_dir: str) -> List[str]:
        """将item的每个段落保存为独立的JSON文件并返回文件路径列表
        
        注意：只保存段落内容，不包含question字段，确保question不参与文档构建过程。
        每个段落被视为一个独立的文档，用于构建原子笔记和知识图谱。
        """
        item_id = item.get('id', 'unknown')
        paragraphs = item.get('paragraphs', [])

        file_paths = []
        for i, para in enumerate(paragraphs):
            idx = para.get('idx', i)
            file_name = f"{item_id}_para_{idx}.json"
            file_path = os.path.join(work_dir, file_name)
            # 只保存段落信息，不包含question，确保与项目标准一致
            para_data = {
                'id': f"{item_id}_para_{idx}",
                'paragraphs': [para]  # 保持与项目标准一致的结构
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(para_data, f, ensure_ascii=False)
            file_paths.append(file_path)

        return file_paths
        
    def process_single_item(self, item: Dict[str, Any], work_dir: str) -> Dict[str, Any]:
        """处理单个musique测试项"""
        item_id = item.get('id', 'unknown')
        question = item.get('question', '')
        
        logger.info(f"Processing item {item_id}")
        
        try:
            # 1. 为每个段落创建独立的文件
            paragraph_files = self._create_paragraph_files(item, work_dir)

            # 2. 处理文档（构建知识库）
            processor = DocumentProcessor(output_dir=work_dir, llm=self.llm)
            process_result = processor.process_documents(paragraph_files, force_reprocess=True, output_dir=work_dir)
            
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
            graph_file = os.path.join(work_dir, 'graph.json')
            embed_file = os.path.join(work_dir, 'embeddings.npy')
            
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
                vector_index_file=None,  # 不使用预构建的向量索引
                llm=self.llm
            )
            
            # 4. 执行查询
            query_result = query_processor.process(question)
            
            # 5. 提取结果
            predicted_answer = query_result.get('answer', 'No answer found')
            predicted_support_idxs = query_result.get('predicted_support_idxs', [])
            
            # 调试模式下保留临时文件
            if not self.debug:
                for p in paragraph_files:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            else:
                logger.info(f"Debug mode: keeping temp files {paragraph_files}")
            
            # 6. 收集召回的原子文档信息
            recalled_notes = query_result.get('notes', [])
            atomic_notes_info = {
                'id': item_id,
                'question': question,
                'recalled_atomic_notes': []
            }
            
            for note in recalled_notes:
                note_info = {
                    'note_id': note.get('note_id', ''),
                    'content': note.get('content', ''),
                    'paragraph_idxs': note.get('paragraph_idxs', []),
                    'similarity_score': note.get('retrieval_info', {}).get('similarity', 0.0)
                }
                atomic_notes_info['recalled_atomic_notes'].append(note_info)
            
            result = {
                'id': item_id,
                'predicted_answer': predicted_answer,
                'predicted_support_idxs': predicted_support_idxs,
                'predicted_answerable': True
            }
            
            logger.info(f"Completed processing item {item_id}")
            return result, atomic_notes_info
            
        except Exception as e:
            logger.error(f"Failed to process item {item_id}: {e}")
            error_result = {
                'id': item_id,
                'predicted_answer': 'Error occurred during processing',
                'predicted_support_idxs': [],
                'predicted_answerable': True
            }
            error_atomic_notes = {
                'id': item_id,
                'question': question,
                'recalled_atomic_notes': [],
                'error': str(e)
            }
            return error_result, error_atomic_notes
    
    def process_dataset(self, input_file: str, output_file: str, atomic_notes_file: str = None, parallel: bool = True) -> None:
        """批量处理musique数据集"""
        logger.info(f"Starting batch processing of {input_file}")
        
        # 预先初始化EmbeddingManager以避免并行处理时的模型加载冲突
        if parallel:
            logger.info("Pre-initializing EmbeddingManager for parallel processing...")
            from vector_store.embedding_manager import EmbeddingManager
            embedding_manager = EmbeddingManager()
            logger.info("EmbeddingManager pre-initialization completed")
        
        # 读取输入数据
        if input_file.endswith('.jsonl'):
            with open(input_file, 'r', encoding='utf-8') as f:
                items = [json.loads(line.strip()) for line in f if line.strip()]
        else:
            items = FileUtils.read_json(input_file)
            if not isinstance(items, list):
                items = [items]
        
        logger.info(f"Loaded {len(items)} items from {input_file}")
        
        results = []
        atomic_notes_records = []  # 用于保存召回的原子文档信息
        
        if parallel and len(items) > 1:
            # 并行处理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 为每个item创建独立的工作目录
                futures = []
                for i, item in enumerate(items):
                    item_id = item.get('id', f'item_{i}')
                    work_dir = create_item_workdir(self.base_work_dir, item_id)
                    future = executor.submit(self.process_single_item, item, work_dir)
                    futures.append((future, work_dir, item_id))
                
                # 收集结果
                for future, work_dir, item_id in tqdm(futures, desc="Processing items"):
                    try:
                        result, atomic_notes_info = future.result()
                        results.append(result)
                        atomic_notes_records.append(atomic_notes_info)
                    except Exception as e:
                        logger.error(f"Failed to get result for item {item_id}: {e}")
                        results.append({
                            'id': item_id,
                            'predicted_answer': 'Processing failed',
                            'predicted_support_idxs': [],
                            'predicted_answerable': True
                        })
                        atomic_notes_records.append({
                            'id': item_id,
                            'question': '',
                            'recalled_atomic_notes': [],
                            'error': str(e)
                        })
                    finally:
                        # 调试模式下保留工作目录
                        if not self.debug:
                            try:
                                shutil.rmtree(work_dir)
                            except:
                                pass
                        else:
                            logger.info(f"Debug mode: keeping work directory {work_dir}")
        else:
            # 串行处理
            for i, item in enumerate(tqdm(items, desc="Processing items")):
                item_id = item.get('id', f'item_{i}')
                work_dir = create_item_workdir(self.base_work_dir, item_id)
                try:
                    result, atomic_notes_info = self.process_single_item(item, work_dir)
                    results.append(result)
                    atomic_notes_records.append(atomic_notes_info)
                except Exception as e:
                    logger.error(f"Failed to process item {item_id}: {e}")
                    results.append({
                        'id': item_id,
                        'predicted_answer': 'Processing failed',
                        'predicted_support_idxs': [],
                        'predicted_answerable': True
                    })
                    atomic_notes_records.append({
                        'id': item_id,
                        'question': item.get('question', ''),
                        'recalled_atomic_notes': [],
                        'error': str(e)
                    })
                finally:
                    # 调试模式下保留工作目录
                    if not self.debug:
                        try:
                            shutil.rmtree(work_dir)
                        except:
                            pass
                    else:
                        logger.info(f"Debug mode: keeping work directory {work_dir}")
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"Batch processing completed. Results saved to {output_file}")
        
        # 保存召回的原子文档信息
        if atomic_notes_file:
            with open(atomic_notes_file, 'w', encoding='utf-8') as f:
                for atomic_notes_info in atomic_notes_records:
                    f.write(json.dumps(atomic_notes_info, ensure_ascii=False) + '\n')
            logger.info(f"Atomic notes recall information saved to {atomic_notes_file}")
        
        # 打印统计信息
        total_items = len(results)
        answered_items = sum(1 for r in results if r['predicted_answer'] != 'No answer found' and 'Error' not in r['predicted_answer'])
        avg_support_idxs = sum(len(r['predicted_support_idxs']) for r in results) / total_items if total_items > 0 else 0
        
        logger.info(f"Statistics:")
        logger.info(f"  Total items: {total_items}")
        logger.info(f"  Successfully answered: {answered_items} ({answered_items/total_items*100:.1f}%)")
        logger.info(f"  Average support paragraphs: {avg_support_idxs:.1f}")
        
        if self.debug:
            logger.info(f"Debug mode: All intermediate files preserved in {self.base_work_dir}")
            logger.info(f"  - Item work directories: {self.base_work_dir}/item_<id>/")
            logger.info(f"  - Each item directory contains: atomic_notes.json, graph.json, embeddings.npy, etc.")


def main():
    parser = argparse.ArgumentParser(description='Musique数据集批量处理工具')
    parser.add_argument('input_file', nargs='?', default='data/1.jsonl', help='输入的musique数据文件（.json或.jsonl格式），默认：data/1.jsonl')
    parser.add_argument('output_file', nargs='?', default='musique_results.jsonl', help='输出结果文件（.jsonl格式），默认：musique_results.jsonl')
    parser.add_argument('--workers', type=int, default=4, help='并行处理的工作线程数（默认：4）')
    parser.add_argument('--serial', action='store_true', help='使用串行处理而非并行处理')
    parser.add_argument('--log-file', help='日志文件路径')
    parser.add_argument('--atomic-notes-file', default='atomic_notes_recall.jsonl', help='保存召回原子文档的文件路径，默认：atomic_notes_recall.jsonl')
    parser.add_argument('--debug', action='store_true', help='调试模式，保留所有中间文件和工作目录')
    parser.add_argument('--work-dir', help='指定工作目录，如果不指定则自动创建新目录')
    parser.add_argument('--new', action='store_true', help='强制创建新的工作目录')
    parser.add_argument('--test-auditor', action='store_true', help='测试摘要校验器功能')
    parser.add_argument('--audit-file', help='指定要审核的原子笔记文件路径')
    
    args = parser.parse_args()
    
    # 确定工作目录
    if args.new:
        work_dir = create_new_workdir()
    elif args.work_dir:
        work_dir = args.work_dir
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = get_latest_workdir()  # 使用最新的工作目录继续任务
    
    # 将所有输出文件路径调整到工作目录内
    if not os.path.isabs(args.output_file):
        output_file = os.path.join(work_dir, args.output_file)
    else:
        output_file = args.output_file
    
    if not os.path.isabs(args.atomic_notes_file):
        atomic_notes_file = os.path.join(work_dir, args.atomic_notes_file)
    else:
        atomic_notes_file = args.atomic_notes_file
    
    # 设置日志文件路径到工作目录内
    if args.log_file:
        if not os.path.isabs(args.log_file):
            log_file = os.path.join(work_dir, args.log_file)
        else:
            log_file = args.log_file
    else:
        log_file = os.path.join(work_dir, 'musique_processing.log')
    
    setup_logging(log_file)
    
    # 测试摘要校验器功能
    if args.test_auditor:
        if not args.audit_file:
            logger.error("测试摘要校验器需要指定 --audit-file 参数")
            return
        
        logger.info(f"开始测试摘要校验器，审核文件: {args.audit_file}")
        try:
            from utils.summary_auditor import SummaryAuditor
            # 为测试功能创建 LocalLLM 实例
            test_llm = LocalLLM()
            auditor = SummaryAuditor(llm=test_llm)
        except ImportError as e:
            logger.error(f"Failed to import SummaryAuditor: {e}")
            return
        except Exception as e:
            logger.error(f"Failed to initialize SummaryAuditor: {e}")
            return
        
        # 读取原子笔记文件
        try:
            atomic_notes = FileUtils.read_json(args.audit_file)
            logger.info(f"加载了 {len(atomic_notes)} 个原子笔记")
            
            # 执行批量审核
            audit_results = auditor.batch_audit_summaries(atomic_notes)
            
            # 输出审核结果统计
            total_notes = len(atomic_notes)
            flagged_count = sum(1 for note in atomic_notes if note.get('audit_result', {}).get('needs_rewrite', False))
            
            logger.info(f"审核完成: 总计 {total_notes} 个笔记，标记需要重写 {flagged_count} 个")
            logger.info(f"标记率: {flagged_count/total_notes*100:.2f}%")
            
            # 保存审核结果
            output_dir = os.path.dirname(args.audit_file)
            auditor.save_flagged_summaries(atomic_notes, output_dir)
            
        except Exception as e:
            logger.error(f"测试摘要校验器时出错: {e}")
        
        return
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return
    
    logger.info(f"Work directory: {work_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Atomic notes file: {atomic_notes_file}")
    logger.info(f"Log file: {log_file}")
    
    # 创建共享的 LocalLLM 实例
    logger.info("Initializing shared LocalLLM instance...")
    shared_llm = LocalLLM()
    logger.info("LocalLLM instance initialized successfully")
    
    # 创建处理器并开始处理
    processor = MusiqueProcessor(
        max_workers=args.workers,
        debug=args.debug,
        work_dir=work_dir,
        llm=shared_llm
    )
    processor.process_dataset(
        args.input_file, 
        output_file, 
        atomic_notes_file=atomic_notes_file,
        parallel=not args.serial
    )


if __name__ == '__main__':
    main()
