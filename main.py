import argparse
import os
from datetime import datetime
from glob import glob
from doc import DocumentProcessor
from config import config
from query import QueryProcessor
from utils import FileUtils
from loguru import logger


RESULT_ROOT = "result"


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
    logger.info(f"Using work dir: {work_dir}")

    processor = DocumentProcessor(output_dir=work_dir)

    extensions = [".json", ".jsonl", ".docx"]
    files = FileUtils.list_files("data", extensions)

    result = processor.process_documents(files, force_reprocess=args.force, output_dir=work_dir)
    FileUtils.write_json(result.get('atomic_notes', []), os.path.join(work_dir, 'atomic_notes.json'))
    logger.info('Documents processed')


def query_mode(args):
    work_dir = args.work_dir or get_latest_workdir()
    notes_file = os.path.join(work_dir, 'atomic_notes.json')
    if not os.path.exists(notes_file):
        logger.error(f'Notes file not found: {notes_file}')
        return
    notes = FileUtils.read_json(notes_file)
    processor = QueryProcessor(notes)
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
