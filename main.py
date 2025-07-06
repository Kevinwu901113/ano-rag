import argparse
import os
from datetime import datetime
from glob import glob
from doc import DocumentProcessor
from config import config
from query import QueryProcessor
from utils import FileUtils, setup_logging
from loguru import logger


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

    processor = DocumentProcessor(output_dir=work_dir)

    extensions = [".json", ".jsonl", ".docx"]
    src_dir = cfg.get('storage', {}).get('source_docs_dir', 'data')
    files = FileUtils.list_files(src_dir, extensions)

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

    processor = QueryProcessor(
        notes,
        graph_file=graph_file if os.path.exists(graph_file) else None,
        vector_index_file=vector_index_file if vector_index_file and os.path.exists(vector_index_file) else None,
    )
    output = processor.process(args.query)
    print(output['answer'])


def main():
    parser = argparse.ArgumentParser(description='Ano-RAG system')
    sub = parser.add_subparsers(dest='cmd')

    proc = sub.add_parser('process')
    proc.add_argument('--new', action='store_true', help='Create new work directory')
    proc.add_argument('--force', action='store_true', help='Force reprocess')

    q = sub.add_parser('query')
    q.add_argument('query', help='Query string')
    q.add_argument('--work-dir', help='Specify work directory')

    args = parser.parse_args()
    if args.cmd == 'process':
        process_docs(args)
    elif args.cmd == 'query':
        query_mode(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
