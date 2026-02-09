import os
import re
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


CONFIG_ENV_VAR = "ANO_RAG_CONFIG"
DEFAULT_EMBED_MODEL = "qwen3-embedding"
DEFAULT_EMBED_DEVICE = "cuda"
DEFAULT_EMBED_ENDPOINT = "http://127.0.0.1:8001/v1"


DEFAULT_CONFIG: Dict[str, Any] = {
    "system": {"project_name": "ano-rag", "device": "cuda"},
    "policies": {
        "strict_definition_extraction": True,
        "doc_entity_isolation": True,
        "allow_cross_doc": False,
        "answer_only_label": True,
        "enable_evidence_canonical": True,
    },
    "chunk": {"n_sent": 3, "overlap": 1, "max_tokens": 768},
    "vllm": {
        "endpoint": "http://127.0.0.1:8000/v1",
        "model": "qwen3-30b-a3b",
        "temperature": 0.0,
        "max_tokens": 256,
        "max_new_tokens": 256,
        "concurrency": {
            "max_workers": 16,
            "batch_size": 1,
            "endpoints": [],  # optional multi-endpoint pool, overrides endpoint
            "connect_timeout_sec": 3.05,
            "read_timeout_sec": 20.0,
            "retry_max_attempts": 2,
            "retry_total_cap_sec": 30.0,
            "retry_backoff_base": 1.0,
            "retry_backoff_max_sec": 6.0,
            "retry_jitter_frac": 0.5,
            "pause_on_timeout_rate": 0.3,
            "pause_sec": 7.0,
            # Endpoint health controls
            "blacklist_duration_sec": 15.0,
            # Log endpoint selection every N successful calls (0=disabled)
            "endpoint_log_every": 0,
            "buckets": {
                "0_256": {"workers": 8},
                "256_512": {"workers": 6},
                "512_1024": {"workers": 4},
                "1024_plus": {"workers": 2},
            },
            "refill_factor": 1.5,
        },
        "adaptive": {
            "enabled": False,
            "min_workers": 2,
            "max_workers": 16,
            "target_p50_ms": 1200,
            "target_p95_ms": 3500,
            "step_up": 2,
            "step_down": 2,
            "window_size": 50,
            "cool_down_sec": 5.0
        },
        "json_mode": {
            "use_guided_json": False,
            "use_response_format": False,
            "schema_name": "ano-note",
        },
    },
    "openai": {
        "enabled": True,
        "model": "gpt-4",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.7,
        "max_tokens": 8192,
        "base_url": "https://api.openai.com/v1",
        "timeout_sec": 60.0,
        "max_retries": 5,
        "retry_backoff_sec": 2.0,
        "retry_backoff_max_sec": 60.0,
        "system_prompt_name": "system_prompt.txt",
    },
    "datasets": {},
    "llm_profiles": {
        "extract": {"temperature": 0.0, "max_tokens": 256, "thinking": False},
        "generate": {"temperature": 0.2, "max_tokens": 128, "thinking": None},
    },
    "structrag": {
        "router": "llm",
        "supported_types": ["chunk", "graph"],
        "top_k": 10,
        "llm_model": "qwen3-30b-a3b",
    },
    "routing": {
        "token_budget_hint": 320000,
    },
    "llm": {
        "max_context_len": 8192,
        "safety_margin_tokens": 256,
    },
    "rerank": {
        "max_candidates": 12,
        "max_candidate_tokens": 256,
    },
    "answer": {
        "max_evidence_items": 12,
        "max_evidence_tokens": 512,
    },
    "retriever": {
        "structured": {
            "enabled": True,
            "walk_enabled": True,
            "multihop_rescue_enabled": True,
            "fanout": 8,
            "entity_match_threshold": 0.5,
            "path_consistency_threshold": 0.9,
            "vector_fallback_enabled": True,
        },
        "embedding": {
            "enabled": True,
            "provider": "vllm",
            "model": DEFAULT_EMBED_MODEL,
            "endpoint": DEFAULT_EMBED_ENDPOINT,
            "api_key": "sk-no-key-required",
            "model_path_override": None,
            "cache_dir": None,
            "download_dir": None,
            "device": DEFAULT_EMBED_DEVICE,
            "dtype": None,
            "auto_build": False,
            "offline_index_path": "indexes/faiss/notes.faiss",
            "meta_path": "indexes/faiss/notes.meta.parquet",
            "max_len_note": 256,
            "topn": 200,
            "faiss": {"kind": "HNSW32", "nprobe": 16, "efSearch": 128},
            "normalize": True,
        },
        "hybrid": {
            "enabled": True,
            "agreement_threshold": 2,
            "weights": {
                "bm25": 1.0,
                "embedding": 1.0,
                "structured": 1.5,
                "subject_match": 2.0,
                "source_agree": 1.0,
            },
        },
        "bm25": {
            "enabled": True,
            "backend": "rank_bm25",
            "store_path": "indexes/bm25/notes",
            "k1": 0.9,
            "b": 0.4,
            "ngram": [1, 2],
            "field_weights": {"subj": 2.0, "pred": 1.6, "obj": 1.2, "ctx": 1.0},
            "topn": 200,
        },
        "fusion": {
            "method": "rrf",
            "rrf_k": 60,
            "weights": {"struct": 0.5, "emb": 0.3, "bm25": 0.2},
            "pre_topM": 128,
            "final_topK": 64,
        },
    },
    "reranker": {
        "enabled": True,
        "type": "llm",
        "llm": {"endpoint": "${vllm.endpoint}", "model": "${vllm.model}", "batch": 16, "timeout_s": 60},
        "final_weights": {"pre": 0.3, "rerank": 0.5, "struct": 0.2},
    },
    "notes": {"out_path": "notes/notes.jsonl", "indexes_dir": "indexes/"},
    "hotpot_entry": {
        "data": "data/hotpot_dev_distractor_5_jsonl.jsonl",
        "cache_dir": "result/cache",
        "output_dir": "result",
        "top_k": 10,
        "query_title_promotion_enabled": True,
        "query_title_promotion_window": 5,
        "limit": 0,
        "workers": 1,
        "debug_dir": "result/debug",
        "debug_max_notes": 50,
        "stall_warn_sec": 600.0,
        "stall_abort_sec": 1800.0,
        "force_build": False,
    },
    "musique_entry": {
        "data": "data/musique_full_v1.0_dev_500.jsonl",
        "cache_dir": "result/musique/cache",
        "output_dir": "result/musique",
        "top_k": 10,
        "limit": 0,
        "workers": 1,
        "debug_dir": "result/musique/debug",
        "debug_max_notes": 50,
        "stall_warn_sec": 300.0,
        "stall_abort_sec": 900.0,
        "force_build": False,
        "use_structured_answer": False,
        "include_decomposition_sp": True,
        "unanswerable_token": "Insufficient evidence",
        "official_template": "sample/sample_dev_pred.json",
    },
    "narrativeqa_entry": {
        "qaps": "narrativeqa/qaps.csv",
        "summaries": "narrativeqa/third_party/wikipedia/summaries.csv",
        "stories_dir": "narrativeqa/tmp",
        "cache_dir": "result/narrativeqa/cache",
        "output_dir": "result/narrativeqa",
        "context_mode": "summary",
        "top_k": 10,
        "limit": 0,
        "workers": 1,
        "stall_warn_sec": 300.0,
        "stall_abort_sec": 900.0,
        "force_build": False,
    },
    "ultradomain": {
        "protocol": {
            "version": "v1",
            "output_root": "result/ultradomain/lightrag_protocol_v1_mix_legal",
        },
        "dataset": {
            "dataset_id": "TBD_ULTRADOMAIN_DATASET_ID",
            "split": "train",
            "config_name": None,
            "cache_dir": None,
            "limit": 0,
            "domain": "all",
        },
        "llm": {
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "qwen3-30b-a3b",
            "api_key_env": "OPENAI_API_KEY",
        },
        "tokenizer": {
            "local_path": None,
            "model": "${vllm.model}",
            "fallback_model": "deepseek-ai/DeepSeek-V3.2",
            "local_only": True,
            "trust_remote_code": True,
        },
        "chunking": {
            "chunk_size": 1200,
            "overlap": 100,
        },
        "retrieval": {
            "prefilter_top_k": 40,
            "budget_tokens": 12000,
        },
        "answer": {
            "temperature": 0.2,
            "top_p": 1.0,
            "max_output_tokens": 1024,
        },
        "judge": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 1024,
        },
        "indexing": {
            "skip_relrag": False,
            "skip_chunk": False,
            "llm_provider": "vllm",
            "llm_endpoint": "${vllm.endpoint}",
            "llm_model": "${vllm.model}",
            "llm_api_key": None,
            "llm_temperature": 0.0,
            "llm_max_tokens": None,
            "embed_provider": "${retriever.embedding.provider}",
            "embed_model": "${retriever.embedding.model}",
            "embed_endpoint": "${retriever.embedding.endpoint}",
            "embed_api_key": "${retriever.embedding.api_key}",
        },
        "pipeline": {
            "resume": False,
            "skip_validate": False,
        },
        "validation": {
            "expected_questions_per_domain": 125,
            "strict": True,
        },
    },
    "parsing": {
        "allow_jsonl": True,
        "enable_array_packer": True,
        "enable_bare_key_fix": True,
        "enable_loose_extractor": True,
        "loose_split_key": "subj",
        "max_tokens": 768,
        "parse_retry": 1,
        "validation_retry": 1,
        "stop": ['"]\n', "\n]", "\n\nEND", "END_JSON", "\n\n"],
        "assume_valid_json": False,
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
    "retrieval": {
        "weights": {
            # 规则权重：score = base_score * w
            "corefers": 1.0,                    # 命中 COREFERS_TO：×1.0
            "mentions": 0.9,                    # 命中 MENTIONS：×0.9
            "unresolved_pronoun": 0.8,         # has_unresolved_pronoun=true：×0.8
            "neighbor_bonus": 0.9,             # 邻域带出：额外×0.9
            "alias_penalty": 0.95,             # 命中 alias 而非 canonical：×0.95
            "fallback_contains_entity_boost": 1.05  # 全文回退：包含规范实体或别名加权
        }
    },
}

ENV_OVERRIDES = {
    "retriever.embedding.cache_dir": "EMB_CACHE_DIR",
    "retriever.embedding.model_path_override": "EMB_MODEL_PATH",
    "retriever.embedding.download_dir": "EMB_DOWNLOAD_DIR",
    "retriever.embedding.device": "EMB_DEVICE",
    "retriever.embedding.dtype": "EMB_DTYPE",
    "retriever.embedding.endpoint": "EMB_ENDPOINT",
    "retriever.embedding.api_key": "EMB_API_KEY",
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _set_nested(config: Dict[str, Any], path: str, value: Any) -> None:
    cursor = config
    parts = path.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _apply_env_overrides(config: Dict[str, Any]) -> None:
    for dotted_key, env_var in ENV_OVERRIDES.items():
        env_val = os.environ.get(env_var)
        if env_val:
            _set_nested(config, dotted_key, env_val)


def _finalize_config(config: Dict[str, Any]) -> None:
    retriever_cfg = config.get("retriever") or {}
    embedding_cfg = retriever_cfg.get("embedding")
    system_device = (config.get("system") or {}).get("device")
    if isinstance(embedding_cfg, dict) and system_device and not embedding_cfg.get("device"):
        embedding_cfg["device"] = system_device


def _merge_models_config(config: Dict[str, Any], models_cfg: Dict[str, Any]) -> None:
    for key in ("vllm", "openai"):
        model_cfg = models_cfg.get(key)
        if not isinstance(model_cfg, dict):
            continue
        base = config.get(key) if isinstance(config.get(key), dict) else {}
        config[key] = _deep_merge(base, model_cfg)


def _get_nested(config: Dict[str, Any], path: str) -> Any:
    cursor: Any = config
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _is_url_like(value: str) -> bool:
    lowered = value.strip().lower()
    return "://" in lowered


def _as_repo_relative(base_dir: Path, value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    # STRATEGY: Resolve relative paths against the repo root
    return str((base_dir / candidate).resolve())


def _inject_runtime_paths(config: Dict[str, Any], *, repo_root: Path, config_path: Path) -> None:
    paths = config.setdefault("paths", {})
    if isinstance(paths, dict):
        paths["repo_root"] = str(repo_root)
        paths["config_path"] = str(config_path)
        paths["config_dir"] = str(config_path.parent)


def _normalize_path_fields(config: Dict[str, Any], *, base_dir: Path) -> None:
    """
    Normalize known path fields to be absolute, relative to base_dir (repo root).
    """
    dotted_keys = [
        "notes.out_path",
        "notes.indexes_dir",
        "retriever.embedding.cache_dir",
        "retriever.embedding.download_dir",
        "retriever.embedding.model_path_override",
        "retriever.embedding.offline_index_path",
        "retriever.embedding.meta_path",
        "retriever.bm25.store_path",
        "hotpot_entry.data",
        "hotpot_entry.cache_dir",
        "hotpot_entry.output_dir",
        "hotpot_entry.debug_dir",
        "musique_entry.data",
        "musique_entry.cache_dir",
        "musique_entry.output_dir",
        "musique_entry.debug_dir",
        "musique_entry.official_template",
        "narrativeqa_entry.qaps",
        "narrativeqa_entry.summaries",
        "narrativeqa_entry.stories_dir",
        "narrativeqa_entry.cache_dir",
        "narrativeqa_entry.output_dir",
        "ultradomain.protocol.output_root",
        "ultradomain.dataset.cache_dir",
        "ultradomain.tokenizer.local_path",
    ]
    for key in dotted_keys:
        current = _get_nested(config, key)
        if current is None:
            continue
        if not isinstance(current, str):
            continue
        raw = current.strip()
        if not raw:
            continue
        if raw.startswith("${") or _is_url_like(raw):
            continue
        _set_nested(config, key, _as_repo_relative(base_dir, raw))


def _resolve_placeholders(config: Dict[str, Any]) -> Dict[str, Any]:
    def _resolve(value: Any) -> Any:
        if isinstance(value, dict):
            for key, val in value.items():
                value[key] = _resolve(val)
            return value
        if isinstance(value, list):
            return [_resolve(item) for item in value]
        if isinstance(value, str):
            def _replace(match: re.Match[str]) -> str:
                ref = match.group(1)
                replacement = _get_nested(config, ref)
                return str(replacement) if replacement is not None else match.group(0)
            return PLACEHOLDER_PATTERN.sub(_replace, value)
        return value

    return _resolve(config)


class ConfigLoader:
    """
    Minimal configuration loader for the structured RAG pipeline.
    
    Priority:
    1. --config argument (passed via init)
    2. ANO_RAG_CONFIG environment variable
    3. relrag/config/config.yaml (new default)
    4. relrag/config.yaml (legacy fallback)
    """

    def __init__(self, config_path: str | None = None):
        self.repo_root = Path(__file__).resolve().parents[2]
        default_v2 = Path(__file__).resolve().parent / "config.yaml"
        default_v1 = Path(__file__).resolve().parent.parent / "config.yaml"
        if config_path is None:
            env_path = os.environ.get(CONFIG_ENV_VAR)
            if env_path:
                config_path = Path(env_path)
            else:
                config_path = default_v2 if default_v2.exists() else default_v1
        self.config_path = Path(config_path).expanduser()
        self._config: Dict[str, Any] | None = None

    def load_config(self) -> Dict[str, Any]:
        if self._config is None:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as handle:
                    user_config = yaml.safe_load(handle) or {}
            else:
                user_config = {}
            self._config = _deep_merge(DEFAULT_CONFIG, user_config)

            if isinstance(user_config.get("models"), dict):
                _merge_models_config(self._config, user_config.get("models") or {})

            # RAG_EMBED_MODEL overrides retriever.embedding.model
            env_embed_model = os.environ.get("RAG_EMBED_MODEL")
            if env_embed_model:
                self._config.setdefault("retriever", {}).setdefault("embedding", {})["model"] = env_embed_model

            _apply_env_overrides(self._config)
            _finalize_config(self._config)
            self._config = _resolve_placeholders(self._config)
            _inject_runtime_paths(self._config, repo_root=self.repo_root, config_path=self.config_path)
            # STRATEGY: All relative paths are resolved against the REPO ROOT
            _normalize_path_fields(self._config, base_dir=self.repo_root)
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
PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")
