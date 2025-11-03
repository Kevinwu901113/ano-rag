import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


class FileUtils:
    @staticmethod
    def list_files(dir_path: str, exts: List[str]) -> List[str]:
        outputs: List[str] = []
        for root, _, files in os.walk(dir_path):
            for name in files:
                if any(name.lower().endswith(ext) for ext in exts):
                    outputs.append(os.path.join(root, name))
        return sorted(outputs)

    @staticmethod
    def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
