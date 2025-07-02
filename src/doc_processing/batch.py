import importlib
import json
import os
from typing import List, Callable, Dict

import pandas as pd

from .json_parser import parse_json
from .jsonl_parser import parse_jsonl
from .word_parser import parse_word
from .notes import AtomicNote


PARSERS: Dict[str, Callable[[str], List[AtomicNote]]] = {
    '.json': parse_json,
    '.jsonl': parse_jsonl,
    '.jl': parse_jsonl,
    '.docx': parse_word,
}


def get_dataframe_module(use_gpu: bool = True):
    """Return cudf or pandas depending on availability and use_gpu flag."""
    if use_gpu:
        spec = importlib.util.find_spec('cudf')
        if spec is not None:
            cudf = importlib.import_module('cudf')
            return cudf
    return pd


def parse_file(path: str) -> List[AtomicNote]:
    ext = os.path.splitext(path)[1].lower()
    parser = PARSERS.get(ext)
    if not parser:
        raise ValueError(f"Unsupported file type: {ext}")
    return parser(path)


def process_files(paths: List[str], output_dir: str, use_gpu: bool = True):
    """Process multiple files and save atomic notes to output_dir."""
    df_module = get_dataframe_module(use_gpu)
    all_notes = []
    for path in paths:
        notes = parse_file(path)
        for note in notes:
            record = note.to_dict()
            # Parquet does not support nested dicts by default
            record['metadata'] = json.dumps(record['metadata'])
            all_notes.append(record)
    df = df_module.DataFrame(all_notes)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'atomic_notes.parquet')
    df.to_parquet(out_path)
    return out_path
