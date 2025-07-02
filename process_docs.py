#!/usr/bin/env python
import argparse

from src.doc_processing.batch import process_files


def main():
    parser = argparse.ArgumentParser(description="Process documents and generate atomic notes")
    parser.add_argument('files', nargs='+', help='Files to process')
    parser.add_argument('-o', '--output', default='output', help='Directory to write results')
    parser.add_argument('--cpu', action='store_true', help='Force CPU (pandas) processing')
    args = parser.parse_args()

    out_path = process_files(args.files, args.output, use_gpu=not args.cpu)
    print(f"Saved notes to {out_path}")


if __name__ == '__main__':
    main()
