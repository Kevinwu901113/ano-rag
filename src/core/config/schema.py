from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict
from pathlib import Path

import yaml

from pydantic import BaseModel, ConfigDict, Field, conint

DEFAULT_CONFIG: Dict[str, Any] = {
    "system": {"project_name": "ano-rag", "seed": 42, "device": "cpu"},
    "document": {"chunk_size": 256, "overlap": 32},
    "embedding": {
        "model_name": "BAAI/bge-m3",
        "batch_size": 64,
        "max_length": 512,
        "normalize": True,
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
        "fusion": {
            "dense_weight": 1.0,
            "bm25_weight": 0.6,
            "focused_weight_hop2": 0.30,
            "rrf_lambda": 0.2,
        },
        "features": {
            "cov_weight": 0.10,
            "cons_weight": 0.05,
            "hop_decay": 0.85,
        },
        "cluster_suppression": {
            "enabled": True,
            "cos_threshold": 0.90,
            "keep_per_cluster": 2,
        },
        "safety": {
            "per_hop_keep_top_m": 6,
            "lower_threshold": 0.10,
            "keep_one_per_doc": True,
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
        "rerank": {
            "entity_overlap_weight": 0.4,
            "path_consistency_weight": 0.3,
            "semantic_weight": 0.35,
            "soft_penalty_no_entity": 0.7,
            "rrf_k": 60,
        },
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
            "fallback_timeout": 10
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
        "stream_early_stop": True,
        "sentinel_char": "~",
        "enable_fast_path": True,
        "retry_once_on_parse_error": True,
        "shorten_on_retry_chars": 1000,
        "min_chars": 25,
        "max_chars": 400,
        "min_salience": 0.3,
        "max_notes_per_chunk": 6,
        "enable_rule_fallback": True,
        "llm_params": {
            "temperature": 0,
            "top_p": 0,
            "max_tokens": 128,
            "stop": ["\n\n", "~"]
        }
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

def _merge_defaults(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _merge_defaults(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


config_yaml = Path(__file__).resolve().parents[3] / "config.yaml"
if config_yaml.exists():
    yaml_defaults = yaml.safe_load(config_yaml.read_text(encoding="utf-8")) or {}
    DEFAULT_CONFIG = _merge_defaults(deepcopy(DEFAULT_CONFIG), yaml_defaults)
else:
    DEFAULT_CONFIG = deepcopy(DEFAULT_CONFIG)

if "multi_hop" not in DEFAULT_CONFIG:
    DEFAULT_CONFIG["multi_hop"] = deepcopy(DEFAULT_CONFIG.get("retrieval", {}).get("multi_hop", {}))


class RootConfig(BaseModel):
    """Typed representation of the configuration tree backed by DEFAULT_CONFIG."""

    model_config = ConfigDict(extra='allow')

    config_version: conint(ge=1) = DEFAULT_CONFIG.get('config_version', 1)
    system: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('system', {})))
    document: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('document', {})))
    embedding: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('embedding', {})))
    vector_store: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('vector_store', {})))
    retrieval: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('retrieval', {})))
    path_aware: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('path_aware', {})))
    hybrid_search: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('hybrid_search', {})))
    dispatcher: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('dispatcher', {})))
    context_dispatcher: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('context_dispatcher', {})))
    graph: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('graph', {})))
    llm: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('llm', {})))
    notes_llm: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('notes_llm', {})))
    calibration: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('calibration', {})))
    safety: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('safety', {})))
    scheduler: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('scheduler', {})))
    multi_hop: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('multi_hop', {})))
    storage: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('storage', {})))
    guardrail: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('guardrail', {})))
    atomic_note_generation: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('atomic_note_generation', {})))
    enhanced_relation_extraction: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('enhanced_relation_extraction', {})))
    feature_switches: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('feature_switches', {})))
    rerank: Dict[str, Any] = Field(default_factory=lambda: deepcopy(DEFAULT_CONFIG.get('rerank', {})))

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        for field_name in self.__class__.model_fields:
            if field_name == 'config_version':
                continue
            value = getattr(self, field_name)
            if isinstance(value, dict):
                setattr(self, field_name, deepcopy(value))
