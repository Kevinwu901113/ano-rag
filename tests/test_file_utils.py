import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.file_utils import FileUtils


def test_get_latest_run_dir(tmp_path):
    (tmp_path / "mirage_run_20240101").mkdir()
    (tmp_path / "mirage_run_20240102").mkdir()
    latest_dir = tmp_path / "mirage_run_20240103"
    latest_dir.mkdir()
    (tmp_path / "random_dir").mkdir()

    base_time = int(time.time()) - 10
    for idx, directory in enumerate(sorted(tmp_path.iterdir())):
        os.utime(directory, (base_time + idx, base_time + idx))

    latest = FileUtils.get_latest_run_dir(str(tmp_path), "mirage_run_")
    assert latest is not None
    assert latest.name == "mirage_run_20240103"


def test_append_jsonl_atomic(tmp_path):
    target = tmp_path / "predictions" / "predictions.jsonl"
    records = [{"id": i, "value": f"data-{i}"} for i in range(32)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        for record in records:
            executor.submit(FileUtils.append_jsonl_atomic, str(target), record)

    with open(target, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == len(records)
    parsed = [json.loads(line) for line in lines]
    assert {item["id"] for item in parsed} == {record["id"] for record in records}


def test_write_manifest(tmp_path):
    manifest_path = tmp_path / "runs" / "manifest.json"
    manifest_data = {"run_id": "mirage_run_1", "status": "completed"}

    FileUtils.write_manifest(str(manifest_path), manifest_data)

    with open(manifest_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    assert saved == manifest_data
