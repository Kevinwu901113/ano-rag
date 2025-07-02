import argparse
from doc import DocumentProcessor
from config import config
from query import QueryProcessor
from utils import FileUtils
from loguru import logger


def process_docs(args):
    processor = DocumentProcessor()
    result = processor.process_documents(args.files)
    FileUtils.write_json(result['atomic_notes'], 'processed_notes.json')
    logger.info('Documents processed')


def query_mode(args):
    notes = FileUtils.read_json('processed_notes.json')
    processor = QueryProcessor(notes)
    output = processor.process(args.query)
    print(output['answer'])


def main():
    parser = argparse.ArgumentParser(description='Ano-RAG system')
    sub = parser.add_subparsers(dest='cmd')

    proc = sub.add_parser('process')
    proc.add_argument('files', nargs='+', help='Files to process')

    q = sub.add_parser('query')
    q.add_argument('query', help='Query string')

    args = parser.parse_args()
    if args.cmd == 'process':
        process_docs(args)
    elif args.cmd == 'query':
        query_mode(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
