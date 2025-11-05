import yaml
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "system": {"project_name": "ano-rag", "device": "cuda"},
    "chunk": {"n_sent": 2, "overlap": 0, "max_tokens": 500},
    "vllm": {
        "endpoint": "http://127.0.0.1:8000/v1",
        "model": "qwen2.5-7b-instruct",
        "temperature": 0.0,
        "max_tokens": 700,
        "concurrency": {
            "max_workers": 8,
            "batch_size": 1,
            "endpoints": [],  # optional multi-endpoint pool, overrides endpoint
            "retry_backoff": [1, 2, 4],
            "timeout_sec": 60
        },
    },
    "lmstudio": {
        "endpoint": "http://127.0.0.1:1234/v1",
        "model": "openai/gpt-oss-20b",
        "temperature": 0.2,
        "max_tokens": 64,
    },
    "notes": {"out_path": "notes/notes.jsonl", "indexes_dir": "indexes/"},
    "parsing": {
        "allow_jsonl": True,
        "enable_array_packer": True,
        "enable_bare_key_fix": True,
        "enable_loose_extractor": True,
        "loose_split_key": "subj",
        "max_tokens": 1024,
        "stop": ['"]\n', "\n]", "\n\nEND", "END_JSON"],
    },
    "schema_guard": {
        "min_evidence_len": 4,
        "max_evidence_len": 512,
        "type_map": {
            "GROUP": "ORG",
            "OBJECT": "CONCEPT",
            "NUMBER": "CONCEPT",
            "QUANTITY": "CONCEPT",
            "PERCENT": "CONCEPT",
            "DATE": "TIME",
            "YEAR": "TIME",
        },
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class ConfigLoader:
    """Minimal configuration loader for the structured RAG pipeline."""

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] | None = None

    def load_config(self) -> Dict[str, Any]:
        if self._config is None:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as handle:
                    user_config = yaml.safe_load(handle) or {}
            else:
                user_config = {}
            self._config = _deep_merge(DEFAULT_CONFIG, user_config)
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        config = self.load_config()
        current: Any = config
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, key: str, value: Any) -> None:
        config = self.load_config()
        parts = key.split(".")
        cursor = config
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
        self._config = config

    def save_config(self) -> None:
        if self._config is None:
            return
        with open(self.config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self._config, handle, allow_unicode=True)


config = ConfigLoader()
