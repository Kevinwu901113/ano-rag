import argparse
import os
from datetime import datetime
from glob import glob
import numpy as np
from doc import DocumentProcessor
from config import config
from query import QueryProcessor
from utils import FileUtils, setup_logging
from loguru import logger
from llm import LocalLLM
from parallel import create_parallel_interface, ProcessingMode, ParallelStrategy


RESULT_ROOT = config.get('storage.result_root', 'result')


def get_latest_workdir() -> str:
    os.makedirs(RESULT_ROOT, exist_ok=True)
    subdirs = [d for d in os.listdir(RESULT_ROOT) if os.path.isdir(os.path.join(RESULT_ROOT, d))]
    if not subdirs:
        return create_new_workdir()
    latest = sorted(subdirs)[-1]
    return os.path.join(RESULT_ROOT, latest)


def create_new_workdir() -> str:
    os.makedirs(RESULT_ROOT, exist_ok=True)
    existing = [int(d) for d in os.listdir(RESULT_ROOT) if d.isdigit()]
    next_idx = max(existing) + 1 if existing else 1
    work_dir = os.path.join(RESULT_ROOT, str(next_idx))
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


def process_docs(args):
    work_dir = create_new_workdir() if args.new else get_latest_workdir()
    # 更新配置中的工作目录和所有存储路径
    cfg = config.load_config()
    storage = cfg.setdefault('storage', {})
    storage['work_dir'] = work_dir
    # 设置所有存储路径到工作目录下
    storage['vector_db_path'] = os.path.join(work_dir, 'vector_store')
    storage['graph_db_path'] = os.path.join(work_dir, 'graph_store')
    storage['processed_docs_path'] = os.path.join(work_dir, 'processed')
    storage['cache_path'] = os.path.join(work_dir, 'cache')
    storage['vector_index_path'] = os.path.join(work_dir, 'vector_index')
    storage['vector_store_path'] = os.path.join(work_dir, 'vector_store')
    storage['embedding_cache_path'] = os.path.join(work_dir, 'embeddings')
    # 设置评估数据集路径
    cfg.setdefault('eval', {})['datasets_path'] = os.path.join(work_dir, 'eval_datasets')
    setup_logging(os.path.join(work_dir, 'ano-rag.log'))
    logger.info(f"Using work dir: {work_dir}")

    extensions = [".json", ".jsonl", ".docx"]
    src_dir = cfg.get('storage', {}).get('source_docs_dir', 'data')
    files = FileUtils.list_files(src_dir, extensions)

    # 检查是否使用并行处理
    if getattr(args, 'parallel', False):
        process_docs_parallel(args, work_dir, files)
    else:
        llm = LocalLLM()
        processor = DocumentProcessor(output_dir=work_dir, llm=llm)
        result = processor.process_documents(files, force_reprocess=args.force, output_dir=work_dir)
        FileUtils.write_json(result.get('atomic_notes', []), os.path.join(work_dir, 'atomic_notes.json'))
        stats = result.get('processing_stats', {})
        logger.info(
            f"Processed {stats.get('files_processed')} files, "
            f"created {stats.get('chunks_created', 0)} chunks and "
            f"{stats.get('atomic_notes_created', 0)} atomic notes."
        )


def query_mode(args):
    work_dir = args.work_dir or config.get('storage.work_dir') or get_latest_workdir()
    # 确保配置中的工作目录和所有存储路径一致
    cfg = config.load_config()
    storage = cfg.setdefault('storage', {})
    storage['work_dir'] = work_dir
    # 设置所有存储路径到工作目录下
    storage['vector_db_path'] = os.path.join(work_dir, 'vector_store')
    storage['graph_db_path'] = os.path.join(work_dir, 'graph_store')
    storage['processed_docs_path'] = os.path.join(work_dir, 'processed')
    storage['cache_path'] = os.path.join(work_dir, 'cache')
    storage['vector_index_path'] = os.path.join(work_dir, 'vector_index')
    storage['vector_store_path'] = os.path.join(work_dir, 'vector_store')
    storage['embedding_cache_path'] = os.path.join(work_dir, 'embeddings')
    # 设置评估数据集路径
    cfg.setdefault('eval', {})['datasets_path'] = os.path.join(work_dir, 'eval_datasets')
    setup_logging(os.path.join(work_dir, 'ano-rag.log'))
    notes_file = os.path.join(work_dir, 'atomic_notes.json')
    if not os.path.exists(notes_file):
        logger.error(f'Notes file not found: {notes_file}')
        return
    notes = FileUtils.read_json(notes_file)

    graph_file = os.path.join(work_dir, 'graph.json')
    faiss_files = glob(os.path.join(work_dir, '*.faiss'))
    vector_index_file = faiss_files[0] if faiss_files else None
    embed_file = os.path.join(work_dir, 'embeddings.npy')
    embeddings = None
    if os.path.exists(embed_file):
        try:
            embeddings = np.load(embed_file)
        except Exception as e:
            logger.warning(f'Failed to load embeddings: {e}')

    llm = LocalLLM()
    processor = QueryProcessor(
        notes,
        embeddings,
        graph_file=graph_file if os.path.exists(graph_file) else None,
        vector_index_file=vector_index_file if vector_index_file and os.path.exists(vector_index_file) else None,
        llm=llm,
        cfg=cfg,
    )
    output = processor.process(args.query)
    print(output['answer'])


def process_docs_parallel(args, work_dir: str, files: list):
    """并行处理文档"""
    logger.info(f"Starting parallel document processing with {len(files)} files")
    
    # 创建并行接口
    max_workers = getattr(args, 'workers', 4)
    strategy = getattr(args, 'strategy', 'hybrid')
    
    strategy_map = {
        'copy': ParallelStrategy.DATA_COPY,
        'split': ParallelStrategy.DATA_SPLIT,
        'dispatch': ParallelStrategy.TASK_DISPATCH,
        'hybrid': ParallelStrategy.HYBRID
    }
    
    parallel_interface = create_parallel_interface(
        max_workers=max_workers,
        processing_mode=ProcessingMode.AUTO,
        strategy=strategy_map.get(strategy, ParallelStrategy.HYBRID),
        debug=getattr(args, 'debug', False)
    )
    
    try:
        # 准备文档数据
        documents = []
        for file_path in files:
            documents.append({
                'file_path': file_path,
                'force_reprocess': args.force
            })
        
        # 并行处理文档
        results = parallel_interface.process_documents(
            documents=documents,
            output_dir=work_dir,
            force_reprocess=args.force
        )
        
        # 合并结果
        all_atomic_notes = []
        total_files = 0
        total_chunks = 0
        
        for result in results:
            if result and 'atomic_notes' in result:
                all_atomic_notes.extend(result['atomic_notes'])
                stats = result.get('processing_stats', {})
                total_files += stats.get('files_processed', 0)
                total_chunks += stats.get('chunks_created', 0)
        
        # 保存合并后的原子笔记
        FileUtils.write_json(all_atomic_notes, os.path.join(work_dir, 'atomic_notes.json'))
        
        # 获取性能统计
        perf_stats = parallel_interface.get_performance_stats()
        if perf_stats:
            logger.info(f"Parallel processing stats: {perf_stats}")
        
        logger.info(
            f"Parallel processed {total_files} files, "
            f"created {total_chunks} chunks and "
            f"{len(all_atomic_notes)} atomic notes."
        )
        
    finally:
        parallel_interface.cleanup()


def query_parallel(args):
    """并行查询处理"""
    work_dir = args.work_dir or config.get('storage.work_dir') or get_latest_workdir()
    # 确保配置中的工作目录和所有存储路径一致
    cfg = config.load_config()
    storage = cfg.setdefault('storage', {})
    storage['work_dir'] = work_dir
    # 设置所有存储路径到工作目录下
    storage['vector_db_path'] = os.path.join(work_dir, 'vector_store')
    storage['graph_db_path'] = os.path.join(work_dir, 'graph_store')
    storage['processed_docs_path'] = os.path.join(work_dir, 'processed')
    storage['cache_path'] = os.path.join(work_dir, 'cache')
    storage['vector_index_path'] = os.path.join(work_dir, 'vector_index')
    storage['vector_store_path'] = os.path.join(work_dir, 'vector_store')
    storage['embedding_cache_path'] = os.path.join(work_dir, 'embeddings')
    # 设置评估数据集路径
    cfg.setdefault('eval', {})['datasets_path'] = os.path.join(work_dir, 'eval_datasets')
    setup_logging(os.path.join(work_dir, 'ano-rag.log'))
    
    notes_file = os.path.join(work_dir, 'atomic_notes.json')
    if not os.path.exists(notes_file):
        logger.error(f'Notes file not found: {notes_file}')
        return
    notes = FileUtils.read_json(notes_file)

    graph_file = os.path.join(work_dir, 'graph.json')
    faiss_files = glob(os.path.join(work_dir, '*.faiss'))
    vector_index_file = faiss_files[0] if faiss_files else None
    embed_file = os.path.join(work_dir, 'embeddings.npy')
    embeddings = None
    if os.path.exists(embed_file):
        try:
            embeddings = np.load(embed_file)
        except Exception as e:
            logger.warning(f'Failed to load embeddings: {e}')
    
    # 准备知识库
    knowledge_base = {
        'atomic_notes': notes,
        'embeddings': embeddings,
        'graph_file': graph_file if os.path.exists(graph_file) else None,
        'vector_index_file': vector_index_file if vector_index_file and os.path.exists(vector_index_file) else None
    }
    
    # 创建并行接口
    max_workers = getattr(args, 'workers', 4)
    strategy = getattr(args, 'strategy', 'hybrid')
    
    strategy_map = {
        'copy': ParallelStrategy.DATA_COPY,
        'split': ParallelStrategy.DATA_SPLIT,
        'dispatch': ParallelStrategy.TASK_DISPATCH,
        'hybrid': ParallelStrategy.HYBRID
    }
    
    parallel_interface = create_parallel_interface(
        max_workers=max_workers,
        processing_mode=ProcessingMode.AUTO,
        strategy=strategy_map.get(strategy, ParallelStrategy.HYBRID),
        debug=getattr(args, 'debug', False)
    )
    
    try:
        # 准备查询数据
        queries = [{'question': args.query}]
        
        # 并行处理查询
        results = parallel_interface.process_queries(
            queries=queries,
            knowledge_base=knowledge_base
        )
        
        if results:
            print(results[0].get('answer', 'No answer found'))
        else:
            print('No results returned')
            
        # 获取性能统计
        perf_stats = parallel_interface.get_performance_stats()
        if perf_stats:
            logger.info(f"Parallel query stats: {perf_stats}")
            
    finally:
        parallel_interface.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Ano-RAG system')
    sub = parser.add_subparsers(dest='cmd')

    # 文档处理命令
    proc = sub.add_parser('process', help='Process documents')
    proc.add_argument('--new', action='store_true', help='Create new work directory')
    proc.add_argument('--force', action='store_true', help='Force reprocess')
    proc.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    proc.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    proc.add_argument('--strategy', choices=['copy', 'split', 'dispatch', 'hybrid'], default='hybrid',
                     help='Parallel processing strategy (default: hybrid)')
    proc.add_argument('--debug', action='store_true', help='Enable debug mode')

    # 查询命令
    q = sub.add_parser('query', help='Query the knowledge base')
    q.add_argument('query', help='Query string')
    q.add_argument('--work-dir', help='Specify work directory')
    q.add_argument('--parallel', action='store_true', help='Enable parallel query processing')
    q.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    q.add_argument('--strategy', choices=['copy', 'split', 'dispatch', 'hybrid'], default='hybrid',
                     help='Parallel processing strategy (default: hybrid)')
    q.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()
    if args.cmd == 'process':
        process_docs(args)
    elif args.cmd == 'query':
        if getattr(args, 'parallel', False):
            query_parallel(args)
        else:
            query_mode(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
