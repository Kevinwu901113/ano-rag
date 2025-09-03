import yaml
from pathlib import Path
from typing import Any, Dict, Tuple

# Default configuration skeleton
DEFAULT_CONFIG: Dict[str, Any] = {
    "system": {"project_name": "ano-rag", "seed": 42, "device": "cpu"},
    "document": {"chunk_size": 256, "overlap": 32},
    "embedding": {
        "model_name": "BAAI/bge-m3",
        "batch_size": 64,
        "max_length": 512,
        "normalize": True,
    },
    "retrieval": {
        "candidate_pool": 50,
        "hybrid": {
            "enabled": True,
            "fusion_method": "linear",
            "weights": {"dense": 1.0, "bm25": 0.5, "graph": 0.5, "path": 0.1},
            "rrf_k": 60,
        },
        "bm25": {"k1": 1.2, "b": 0.75, "text_field": "title_raw_span"},
        "graph": {"enabled": True, "k_hop": 2, "expand_top_m": 20},
        "multi_hop": {
            "enabled": True,
            "strategy": "hybrid",
            "top_k_seed": {
                "enabled": False,
                "seed_count": 5,
                "fallback_to_entity": True
            },
            "entity_extraction": {
                "enabled": True,
                "max_entities": 10
            },
            "hybrid_mode": {
                "primary_strategy": "entity_extraction",
                "fallback_strategy": "top_k_seed",
                "switch_threshold": 3
            }
        },
    },
    "path_aware": {"enabled": True, "min_path_score": 0.3},
    "dispatcher": {
        "final_semantic_count": 8,
        "final_graph_count": 5,
        "bridge_policy": "keepalive",
        "bridge_boost_epsilon": 0.02,
        "debug_log": True,
    },
    "llm": {
        "provider": "openai", 
        "model": "gpt-3.5-turbo", 
        "temperature": 0.7, 
        "max_output_tokens": 512,
        "hybrid_llm": {
            "mode": "task_division",
            "light_tasks": {
                "provider": "ollama",
                "model": "qwen2.5:latest",
                "base_url": "http://localhost:11434",
                "timeout": 30
            },
            "heavy_tasks": {
                "provider": "lmstudio",
                "model": "openai/gpt-oss-20b",
                "base_url": "http://localhost:1234/v1",
                "instances": 2,
                "timeout": 60
            }
        }
    },
    "guardrail": {"enabled": True, "min_results": 1, "min_score": 0.0, "timeout_seconds": 30},
    "vector_store": {
        "top_k": 20,
        "similarity_threshold": 0.5,
        "batch_size": 32,
        "dimension": 1024,
        "index_type": "IVFFlat",
        "similarity_metric": "cosine"
    },
}


def _merge_with_defaults(data: Dict[str, Any], defaults: Dict[str, Any], path: str = "") -> Tuple[Dict[str, Any], list]:
    """Merge user config with defaults and collect deprecated keys."""
    result: Dict[str, Any] = {}
    deprecated: list = []

    for key, default_value in defaults.items():
        if isinstance(default_value, dict):
            sub_data = data.get(key, {}) if isinstance(data, dict) else {}
            merged, dep = _merge_with_defaults(sub_data, default_value, f"{path}{key}.")
            result[key] = merged
            deprecated.extend(dep)
        else:
            if isinstance(data, dict) and key in data:
                result[key] = data[key]
            else:
                result[key] = default_value

    if isinstance(data, dict):
        for extra_key in data.keys():
            if extra_key not in defaults:
                deprecated.append(f"{path}{extra_key}".rstrip("."))
    return result, deprecated


class ConfigLoader:
    """Load and access configuration with strict schema."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] | None = None
        self.deprecated_keys: list[str] = []

    def load_config(self) -> Dict[str, Any]:
        if self._config is None:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
            else:
                raw = {}
            merged, deprecated = _merge_with_defaults(raw, DEFAULT_CONFIG)
            self._config = merged
            self.deprecated_keys = deprecated
            for key in deprecated:
                print(f"[Config] deprecated field ignored: {key}")
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        config = self.load_config()
        value: Any = config
        for part in key.split('.'):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """Set a configuration value using dot notation."""
        config = self.load_config()
        keys = key.split('.')
        current = config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final key
        current[keys[-1]] = value
        self._config = config
    
    def update_config(self, updates: Dict[str, Any]):
        config = self.load_config()
        # shallow update only for existing keys
        for k, v in updates.items():
            if k in config:
                config[k] = v
        self._config = config

    def save_config(self):
        if self._config is not None:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)


config = ConfigLoader()
