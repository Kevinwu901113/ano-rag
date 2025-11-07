from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import uuid
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from loguru import logger


class NotesCache:
    """Idempotent KV store mapping paragraph hash -> notes list, backed by parquet.

    Schema:
      columns: ["hash", "notes_json"]
      - "hash": str (paragraph content hash)
      - "notes_json": str (JSON array of notes or JSONL string)

    Operations are idempotent: writing the same hash multiple times will
    overwrite with the latest content without error.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._df: Optional[pd.DataFrame] = None
        # Guard concurrent writers (producer threads) to avoid tmp-file races
        self._lock = threading.RLock()

    def _load(self) -> None:
        if self._df is not None:
            return
        if self.path.exists():
            try:
                self._df = pd.read_parquet(self.path)
            except Exception as exc:
                logger.warning("Failed to read parquet at {}: {} (initializing empty)", self.path, exc)
                self._df = pd.DataFrame(columns=["hash", "notes_json"])
        else:
            self._df = pd.DataFrame(columns=["hash", "notes_json"])

    def get(self, hash_key: str) -> Optional[List[Dict[str, Any]]]:
        self._load()
        assert self._df is not None
        rows = self._df[self._df["hash"] == hash_key]
        if rows.empty:
            return None
        value = rows.iloc[0]["notes_json"]
        try:
            if isinstance(value, str) and value:
                data = json.loads(value)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
        return None

    def get_many(self, hash_keys: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
        self._load()
        assert self._df is not None
        keys = list(hash_keys)
        if not keys:
            return {}
        rows = self._df[self._df["hash"].isin(keys)]
        out: Dict[str, List[Dict[str, Any]]] = {}
        for _, row in rows.iterrows():
            key = row["hash"]
            val = row["notes_json"]
            try:
                data = json.loads(val)
                if isinstance(data, list):
                    out[key] = data
            except Exception:
                continue
        return out

    def put(self, hash_key: str, notes: List[Dict[str, Any]]) -> None:
        """Idempotent write: upsert row by hash."""
        self._load()
        assert self._df is not None
        serialized = json.dumps(notes, ensure_ascii=False)
        with self._lock:
            df = self._df
            idxs = df.index[df["hash"] == hash_key].tolist()
            if idxs:
                # overwrite
                df.loc[idxs[0], "notes_json"] = serialized
            else:
                df = pd.concat([df, pd.DataFrame({"hash": [hash_key], "notes_json": [serialized]})], ignore_index=True)
            self._df = df
            self._persist()

    def put_many(self, items: Iterable[Tuple[str, List[Dict[str, Any]]]]) -> None:
        self._load()
        assert self._df is not None
        with self._lock:
            df = self._df
            for key, notes in items:
                serialized = json.dumps(notes, ensure_ascii=False)
                idxs = df.index[df["hash"] == key].tolist()
                if idxs:
                    df.loc[idxs[0], "notes_json"] = serialized
                else:
                    df = pd.concat([df, pd.DataFrame({"hash": [key], "notes_json": [serialized]})], ignore_index=True)
            self._df = df
            self._persist()

    def contains(self, hash_key: str) -> bool:
        self._load()
        assert self._df is not None
        # Read path does not strictly need locking; keep simple
        return not self._df[self._df["hash"] == hash_key].empty

    def keys(self) -> List[str]:
        self._load()
        assert self._df is not None
        return list(self._df["hash"].unique())

    def _persist(self) -> None:
        assert self._df is not None
        # Serialize writes to avoid concurrent rename races
        with self._lock:
            try:
                # Write atomically using a unique temp file per write
                unique = f".tmp.{uuid.uuid4().hex}.{os.getpid()}.{threading.get_ident()}"
                tmp = self.path.parent / f"{self.path.name}{unique}"
                # Ensure parent exists
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._df.to_parquet(tmp, index=False)
                # Robustness: verify file exists before replace
                if not tmp.exists():
                    raise FileNotFoundError(f"Temp parquet not found: {tmp}")
                os.replace(tmp, self.path)
            except Exception as exc:
                logger.error("Failed to persist notes cache '{}' : {}", self.path, exc)