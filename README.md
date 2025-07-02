# ano-rag

This repository provides tools for parsing documents and generating "atomic notes" for further stages in an RAG (Retrieval Augmented Generation) pipeline.

## Features

- Parsers for JSON, JSONL and Word (`.docx`) files.
- Sentence-based chunking that avoids splitting in the middle of sentences.
- Batch processing script that can process multiple files at once.
- Uses GPU-accelerated cudf DataFrames when available, falling back to pandas otherwise.
- Atomic notes are stored in Parquet format for downstream consumption.

## Usage

```
./process_docs.py path/to/file1.json path/to/file2.docx -o output_dir
```

Use `--cpu` to disable cudf usage even if installed.
