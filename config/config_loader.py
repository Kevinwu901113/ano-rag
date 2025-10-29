import yaml
from copy import deepcopy
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
    "note_keys": {
        "rel_lexicon": {
            "performed_by": ["performed by", "the performer is", "由", "演奏", "演出"],
            "spouse_of": ["spouse", "partner", "married to", "配偶", "伴侣"],
            "born_in": ["born in", "出生于", "出生在"],
            "released_in": ["released in", "发行于", "发行在"],
            "member_of": ["member of", "成员", "属于"],
        },
        "type_hints": {
            "album": ["(album)"],
            "song": ["(song)"],
            "film": ["(film)"],
            "person": ["(person)", "先生", "女士", "Dr."],
        },
        "relation_type_map": {
            "performed_by": {"head": "song", "tail": "person"},
            "released_in": {"head": "album", "tail": "year"},
            "born_in": {"head": "person", "tail": "place"},
            "spouse_of": {"head": "person", "tail": "person"},
            "member_of": {"head": "person", "tail": "organization"},
        },
        "fallback_splitters": [
            " is ",
            " was ",
            " were ",
            " by ",
            " in ",
            " of ",
            "于",
            "在",
            "是",
        ],
        "default_rel": "related_to",
        "normalize": {"strip_quotes": True, "collapse_space": True, "lower": False},
    },
    "graph": {
        "edge": {
            "base_weight": 0.0,
            "key_match_weight": 1.5,
            "type_compat_weight": 1.0,
            "same_paragraph_bonus": 0.3,
            "same_title_bonus": 0.2,
        },
        "semantic_similarity": {
            "use_lsh": True,
            "num_planes": 32,
            "bands": 8,
            "max_candidates": 64,
            "min_candidates": 10,
            "random_seed": 42
        }
    },
    "multi_hop": {
        "max_hops": 4,
        "beam_size": 8,
        "branch_factor": 6,
    },
    "answering": {
        "rel_chains": [["performed_by", "spouse_of"]],
        "relax_last_hop": ["spouse_of|partner_of"],
        "strict_person": {"enabled": True},
        "efsa_hint": {
            "enabled": True,
            "threshold": 0.70,
            "multi_candidate": 2,
        },
        "final_evidence_first": True,
        "require_verbatim_spans": True,
        "force_insufficient_if_no_spans": True,
    },
    "retry": {"max_times": 1},
    "validator": {"allow_partial": True},
    "answer_selector": {
        "enabled": True,
        "anchor_top_k": 5,
        "use_candidate_pool": True,
        "apply_before_llm": True,
    },
    # 添加存储配置默认值
    "storage": {
        "work_dir": "./result/work",
        "vector_index_path": "./result/vector_index",
        "embedding_cache_path": "./result/embedding_cache",
        "vector_store_path": "./result/vector_store",
        "processed_docs_path": "./result/processed_docs",
        "result_root": "./result"
    },
    # 添加模型路径配置
    "model_path": "./models/default_model",
    # 添加并行策略配置
    "parallel_strategy": "thread",
    # 性能配置：默认禁用FAISS GPU，加速稳定性
    "performance": {
        "use_gpu": False
    },
    # 添加 LLM 配置的缺失键
    "llm": {
        "provider": "openai", 
        "model": "gpt-3.5-turbo", 
        "temperature": 0.7, 
        "max_output_tokens": 512,
        "openai": {
            "api_key": "",
            "base_url": "https://api.openai.com/v1"
        },
        "lmstudio": {
            "base_url": "http://localhost:1234/v1"
        },
        "local_model": {
            "temperature": 0.7,
            "max_tokens": 512
        },
        # 已移除 hybrid_llm 中的 ollama 配置；如需多提供者请直接使用 LLMFactory 并在具体模块配置中声明
        # 新增：笔记生成器（vLLM）默认配置，避免自定义键被忽略
        "note_generator": {
            "provider": "vllm-openai",
            "endpoints": [],
            "model": "qwen2.5:7b",
            "max_tokens": 96,
            "temperature": 0.2,
            "top_p": 0.9,
            "timeout": 15,
            "concurrency_per_endpoint": 32,
            "bucket_edges": [64, 128, 256, 512, 1024],
            "retry": {
                "max_attempts": 3,
                "backoff_base_ms": 200
            }
        }
    },
    # 添加各种配置文件路径
    "diversity_scheduler_config_file": "./config/diversity_scheduler.yaml",
    "path_aware_ranker_config_file": "./config/path_aware_ranker.yaml",
    "retrieval_guardrail_config_file": "./config/retrieval_guardrail.yaml",
    "entity_predicate_normalizer_config_file": "./config/entity_predicate_normalizer.yaml",
    "embedding_strategy_config_file": "./config/embedding_strategy.yaml",
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
            "max_hops": 3,
            "max_paths": 10,
            "min_path_score": 0.3,
            "min_path_score_floor": 0.1,
            "min_path_score_step": 0.05,
            "path_diversity_threshold": 0.7,
            "max_initial_candidates": 20,
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
    "hybrid_search": {
        "enabled": True,
        "fusion_method": "linear",
        "prf_bridge": {
            "enabled": True,
            "first_hop_topk": 2,
            "prf_topk": 20,
        },
        "linear": {
            "vector_weight": 1.0,
            "bm25_weight": 0.5,
            "path_weight": 0.1,
        },
        "rrf": {
            "k": 60,
            "vector_weight": 1.0,
            "bm25_weight": 1.0,
            "path_weight": 1.0,
        },
        "weights": {"dense": 1.0, "bm25": 0.5, "graph": 0.5, "path": 0.1},
        "bm25": {"k1": 1.2, "b": 0.75, "corpus_field": "title_raw_span"},
        "path_aware": {"enabled": True},
        "retrieval_guardrail": {
            "enabled": True,
            "must_have_terms": {},
            "boost_entities": {},
            "boost_predicates": {},
            "predicate_mappings": {},
        },
        "fallback": {
            "enabled": True,
            "sparse_boost_factor": 1.5,
            "query_rewrite_enabled": True,
            "max_retries": 2,
        },
        "two_hop_expansion": {
            "enabled": True,
            "top_m_candidates": 20,
            "entity_extraction_method": "rule_based",
            "target_predicates": [
                "founded_by",
                "located_in",
                "member_of",
                "works_for",
                "part_of",
                "instance_of",
            ],
            "max_second_hop_candidates": 15,
            "merge_strategy": "weighted",
        },
        "section_filtering": {
            "enabled": True,
            "filter_rule": "main_entity_related",
            "fallback_to_lexical": True,
        },
        "lexical_fallback": {
            "enabled": True,
            "must_have_terms_sources": ["main_entity", "predicate_stems"],
            "miss_penalty": 0.6,
            "blacklist_penalty": 0.5,
            "noise_threshold": 0.20,
        },
        "namespace_filtering": {
            "enabled": True,
            "stages": [
                "initial_recall",
                "post_fusion",
                "post_two_hop",
                "final_scheduling",
            ],
            "same_namespace_bm25_fallback": True,
            "strict_mode": True,
        },
        "multi_hop": {
            "max_hops": 4,
            "beam_width": 8,
            "per_hop_keep_top_m": 5,
            "focused_weight_by_hop": {1: 0.30, 2: 0.25, 3: 0.20, 4: 0.15},
            "hop_decay": 0.85,
            "lower_threshold": 0.10,
        },
        "answer_bias": {"who_person_boost": 1.10},
    },
    "dispatcher": {
        "final_semantic_count": 8,
        "final_graph_count": 5,
        "bridge_policy": "keepalive",
        "bridge_boost_epsilon": 0.02,
        "debug_log": True,
        "enabled": True,
        "k_hop": 2,
    },
    "context_dispatcher": {
        "enabled": True,
        "k_hop": 2,
        "final_semantic_count": 8,
        "final_graph_count": 5,
        "bridge_policy": "keepalive",
        "bridge_boost_epsilon": 0.02,
        "debug_log": True,
    },
    "guardrail": {"enabled": True, "min_results": 1, "min_score": 0.0, "timeout_seconds": 30},
    "atomic_note_generation": {
        "parallel_enabled": False,
        "parallel_strategy": "task_division",
        "task_division": {
            "enabled": True,
            "allocation_method": "round_robin",
            "enable_fallback": True,
            "fallback_timeout": 10,
            "client_timeouts": {
                "ollama": 30,
                "lmstudio": 90,
            },
            "client_retries": {
                "ollama": 2,
                "lmstudio": 3,
            }
        },
        "ollama": {
            "model": "qwen2.5:latest",
            "base_url": "http://localhost:11434",
            "timeout": 30,
            "temperature": 0.1
        },
        "lmstudio": {
            "model": "qwen2.5-7b-instruct",
            "base_url": "http://localhost:1234/v1",
            "timeout": 60,
            "temperature": 0.1
        },
        "monitoring": {
            "enabled": True,
            "log_stats": True,
            "export_metrics": False
        }
    },
    "notes_llm": {
        "use_v2_schema": True,
        "stream_early_stop": True,
        "sentinel_char": "~",
        "enable_fast_path": True,
        "retry_once_on_parse_error": True,
        "shorten_on_retry_chars": 1000,
        "min_chars": 20,
        "max_chars": 400,
        "min_salience": 0.3,
        "max_notes_per_chunk": 12,
        "max_note_chars": 200,
        "enable_rule_fallback": True,
        "entities_fallback": {
            "enabled": True,
            "min_len": 2,
            "types": ["PERSON", "ORG", "GPE", "WORK_OF_ART", "EVENT"],
        },
        "limit": {
            "strategy": "bucketed",
            "bucket": {
                "by": "paragraph_idx",
                "quota_per_bucket": 1,
            },
        },
        "llm_params": {
            "temperature": 0,
            "top_p": 0,
            "max_tokens": 128,
            "stop": ["\n\n", "~"]
        }
    },
    "quality_filter": {
        "require_entities": False,
        "min_chars": 20,
        "min_salience": 0.3,
    },
    "note_completeness": {
        "require_sentence_terminal": True,
        "allowed_sentence_terminals": ["。", ".", "!", "?"],
        "min_word_count_en": 4,
        "min_char_count_zh": 8,
        "verb_patterns_en": [],
        "verb_patterns_zh": [],
        "bad_starts_en": [],
        "bad_starts_zh": [],
        "require_entities": False,
    },
    "evidence_rerank": {
        "enable": True,
        "w_album": 0.5,
        "w_song": -0.3,
        "w_supporting": 0.4,
        "w_q_performer_album": 0.3,
        "album_tokens": ["(album)", " album"],
        "song_tokens": ["(song)", " single", "(film)"],
        "support_flag_keys": ["is_supporting", "supporting"],
        "query_performer_terms": ["performer", "singer", "vocalist"],
        "query_album_terms": ["album", "record", "ep"],
    },
    "vector_store": {
        "top_k": 20,
        "similarity_threshold": 0.5,
        "batch_size": 32,
        "dimension": 1024,
        "index_type": "IVFFlat",
        "similarity_metric": "cosine"
    },
    "safety": {
        "per_hop_keep_top_m": 5,
        "lower_threshold": 0.1,
        "cluster": {
            "enabled": False,
            "cos_threshold": 0.85,
            "keep_per_cluster": 3
        }
    },
    "context": {
        "max_notes_for_llm": 20,
        "max_tokens": None
    },
    "ranking": {
        "dense_weight": 0.7,
        "bm25_weight": 0.3,
        "hop_decay": 0.8
    }
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


def _get_path(data: Dict[str, Any] | None, path: Tuple[str, ...]):
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _set_path(data: Dict[str, Any], path: Tuple[str, ...], value: Any):
    cur = data
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def _deep_merge(base: Any, updates: Any):
    if updates is None:
        return deepcopy(base)
    if not isinstance(base, dict) or not isinstance(updates, dict):
        return deepcopy(updates)
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict):
            merged[key] = _deep_merge(merged.get(key, {}), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _normalize_hybrid_branch(config_data: Dict[str, Any]):
    hybrid = _get_path(config_data, ("hybrid_search",))
    if not isinstance(hybrid, dict):
        return

    normalized = deepcopy(hybrid)
    linear = normalized.get("linear", {}) or {}
    weights = normalized.get("weights", {}) or {}

    if not linear and weights:
        linear = {
            "vector_weight": weights.get("dense", 1.0),
            "bm25_weight": weights.get("bm25", 0.5),
            "path_weight": weights.get("path", 0.1),
        }
    else:
        weights = dict(weights)
        if "vector_weight" in linear:
            weights["dense"] = linear["vector_weight"]
        if "bm25_weight" in linear:
            weights["bm25"] = linear["bm25_weight"]
        if "path_weight" in linear:
            weights["path"] = linear["path_weight"]
        if "graph" not in weights:
            weights["graph"] = DEFAULT_CONFIG["hybrid_search"]["weights"].get("graph", 0.5)

    normalized["linear"] = linear
    normalized["weights"] = weights

    rrf = normalized.get("rrf", {}) or {}
    if "k" not in rrf:
        rrf["k"] = normalized.get("rrf_k", DEFAULT_CONFIG["hybrid_search"]["rrf"]["k"])
    normalized["rrf"] = rrf
    normalized["rrf_k"] = rrf.get("k")

    _set_path(config_data, ("hybrid_search",), deepcopy(normalized))
    _set_path(config_data, ("retrieval", "hybrid"), deepcopy(normalized))


def _synchronize_aliases(config_data: Dict[str, Any], raw_data: Dict[str, Any] | None):
    raw_data = raw_data or {}

    # Context dispatcher mirrors dispatcher settings
    context_default = _get_path(DEFAULT_CONFIG, ("context_dispatcher",)) or {}
    dispatcher_override = _get_path(raw_data, ("dispatcher",))
    context_override = _get_path(raw_data, ("context_dispatcher",))
    merged_context = _deep_merge(context_default, dispatcher_override)
    merged_context = _deep_merge(merged_context, context_override)
    if not isinstance(merged_context, dict):
        merged_context = {"enabled": bool(merged_context)}
    _set_path(config_data, ("context_dispatcher",), deepcopy(merged_context))
    _set_path(config_data, ("dispatcher",), deepcopy(merged_context))

    # Hybrid search mirrors retrieval.hybrid
    hybrid_default = _get_path(DEFAULT_CONFIG, ("hybrid_search",)) or {}
    legacy_hybrid_override = _get_path(raw_data, ("retrieval", "hybrid"))
    hybrid_override = _get_path(raw_data, ("hybrid_search",))
    merged_hybrid = _deep_merge(hybrid_default, legacy_hybrid_override)
    merged_hybrid = _deep_merge(merged_hybrid, hybrid_override)
    if not isinstance(merged_hybrid, dict):
        merged_hybrid = {"enabled": bool(merged_hybrid)}
    _set_path(config_data, ("hybrid_search",), deepcopy(merged_hybrid))

    # BM25 parameters shared between schemas
    bm25_default = _get_path(DEFAULT_CONFIG, ("hybrid_search", "bm25")) or {}
    legacy_bm25_override = _get_path(raw_data, ("retrieval", "bm25"))
    hybrid_bm25_override = _get_path(raw_data, ("hybrid_search", "bm25"))
    merged_bm25 = _deep_merge(bm25_default, legacy_bm25_override)
    merged_bm25 = _deep_merge(merged_bm25, hybrid_bm25_override)
    _set_path(config_data, ("hybrid_search", "bm25"), deepcopy(merged_bm25))
    _set_path(config_data, ("retrieval", "bm25"), deepcopy(merged_bm25))

    # Multi-hop configuration shared between legacy and retrieval-scoped keys
    multi_hop_default = _get_path(DEFAULT_CONFIG, ("retrieval", "multi_hop")) or {}
    legacy_multi_hop_override = _get_path(raw_data, ("multi_hop",))
    retrieval_multi_hop_override = _get_path(raw_data, ("retrieval", "multi_hop"))
    merged_multi_hop = _deep_merge(multi_hop_default, legacy_multi_hop_override)
    merged_multi_hop = _deep_merge(merged_multi_hop, retrieval_multi_hop_override)
    if not isinstance(merged_multi_hop, dict):
        merged_multi_hop = {"enabled": bool(merged_multi_hop)}
    _set_path(config_data, ("retrieval", "multi_hop"), deepcopy(merged_multi_hop))
    _set_path(config_data, ("multi_hop",), deepcopy(merged_multi_hop))

    # 旧键到新键的映射 - 影响条数的配置项
    # hybrid_search.multi_hop.per_hop_keep_top_m → safety.per_hop_keep_top_m
    legacy_per_hop_keep = _get_path(raw_data, ("hybrid_search", "multi_hop", "per_hop_keep_top_m"))
    if legacy_per_hop_keep is not None:
        _set_path(config_data, ("safety", "per_hop_keep_top_m"), legacy_per_hop_keep)
    
    # hybrid_search.multi_hop.lower_threshold → safety.lower_threshold  
    legacy_lower_threshold = _get_path(raw_data, ("hybrid_search", "multi_hop", "lower_threshold"))
    if legacy_lower_threshold is not None:
        _set_path(config_data, ("safety", "lower_threshold"), legacy_lower_threshold)
    
    # hybrid_search.safety.* → safety.*
    legacy_safety = _get_path(raw_data, ("hybrid_search", "safety"))
    if legacy_safety is not None:
        current_safety = _get_path(config_data, ("safety",)) or {}
        merged_safety = _deep_merge(current_safety, legacy_safety)
        _set_path(config_data, ("safety",), merged_safety)
    
    # retrieval.vector.top_k ↔ vector_store.top_k (双向映射)
    legacy_vector_top_k = _get_path(raw_data, ("retrieval", "vector", "top_k"))
    new_vector_top_k = _get_path(raw_data, ("vector_store", "top_k"))
    
    if legacy_vector_top_k is not None:
        _set_path(config_data, ("vector_store", "top_k"), legacy_vector_top_k)
    if new_vector_top_k is not None:
        _set_path(config_data, ("retrieval", "vector", "top_k"), new_vector_top_k)
    
    # retrieval.bm25.top_k → retrieval.bm25.top_k (保持一致性)
    legacy_bm25_top_k = _get_path(raw_data, ("retrieval", "bm25", "top_k"))
    if legacy_bm25_top_k is not None:
        _set_path(config_data, ("retrieval", "bm25", "top_k"), legacy_bm25_top_k)

    _normalize_hybrid_branch(config_data)


class ConfigLoader:
    """Load and access configuration with strict schema."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] | None = None
        self._raw_config: Dict[str, Any] | None = None
        self.deprecated_keys: list[str] = []

    def load_config(self) -> Dict[str, Any]:
        if self._config is None:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
            else:
                raw = {}
            self._raw_config = raw
            merged, deprecated = _merge_with_defaults(raw, DEFAULT_CONFIG)
            self._config = merged
            _synchronize_aliases(self._config, self._raw_config)
            self.deprecated_keys = deprecated
            
            # Print unknown/ignored keys for configuration cleanup
            if deprecated:
                print(f"[CONFIG][IGNORED] {sorted(list(deprecated))[:50]}{'...' if len(deprecated) > 50 else ''}")
            
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
        if self._raw_config is None:
            self._raw_config = {}
        raw_current = self._raw_config
        for k in keys[:-1]:
            raw_current = raw_current.setdefault(k, {})
        raw_current[keys[-1]] = value
        _synchronize_aliases(self._config, self._raw_config)

    def update_config(self, updates: Dict[str, Any]):
        config = self.load_config()
        # shallow update only for existing keys
        for k, v in updates.items():
            if k in config:
                config[k] = v
        self._config = config
        _synchronize_aliases(self._config, self._raw_config)

    def save_config(self):
        if self._config is not None:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)


config = ConfigLoader()
