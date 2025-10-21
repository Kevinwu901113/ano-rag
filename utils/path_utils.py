"""Utility helpers for resolving configured storage paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from config import config
from config.config_loader import DEFAULT_CONFIG


_STORAGE_KEY_DEFAULT_SUBDIR: Dict[str, str] = {
    "vector_db_path": "vector_store",
    "graph_db_path": "graph_store",
    "processed_docs_path": "processed_docs",
    "cache_path": "cache",
    "vector_index_path": "vector_index",
    "vector_store_path": "vector_store",
    "embedding_cache_path": "embedding_cache",
}


def _resolve_storage_value(
    value: Optional[str],
    work_dir: Optional[str],
    fallback_name: str,
) -> str:
    """Resolve a storage path value against the configured work directory."""

    work_dir_path = Path(work_dir) if work_dir else None

    if value:
        path_obj = Path(value)
        if path_obj.is_absolute():
            return str(path_obj)
        if work_dir_path:
            return str(work_dir_path / path_obj.name)
        return str(path_obj)

    if work_dir_path:
        return str(work_dir_path / fallback_name)

    result_root = DEFAULT_CONFIG.get("storage", {}).get("result_root", "./result")
    result_root_path = Path(result_root)
    return str(result_root_path / fallback_name)


def setup_storage_paths(work_dir: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Update configuration with storage paths rooted at ``work_dir``."""

    cfg = cfg or config.load_config()
    storage_cfg = cfg.setdefault("storage", {})
    storage_cfg["work_dir"] = work_dir

    defaults = DEFAULT_CONFIG.get("storage", {})

    # Ensure result root exists in the configuration
    result_root = storage_cfg.get("result_root") or defaults.get("result_root")
    if result_root:
        storage_cfg["result_root"] = str(Path(result_root))

    for key, fallback_dir in _STORAGE_KEY_DEFAULT_SUBDIR.items():
        configured_value = storage_cfg.get(key) or defaults.get(key)
        storage_cfg[key] = _resolve_storage_value(configured_value, work_dir, fallback_dir)

    # Keep evaluation datasets co-located with the work directory
    cfg.setdefault("eval", {})["datasets_path"] = str(Path(work_dir) / "eval_datasets")

    return cfg


def get_storage_path(
    key: str,
    fallback_subdir: str,
    cfg: Optional[Dict[str, Any]] = None,
    work_dir: Optional[str] = None,
) -> str:
    """Return a resolved storage path for the given configuration key."""

    cfg = cfg or config.load_config()
    storage_cfg = cfg.get("storage", {})
    work_dir = work_dir or storage_cfg.get("work_dir")

    defaults = DEFAULT_CONFIG.get("storage", {})
    configured_value = storage_cfg.get(key) or defaults.get(key)

    return _resolve_storage_value(configured_value, work_dir, fallback_subdir)


def resolve_cache_dir(
    preferred_base: Optional[str] = None,
    default_subdir: str = "cache",
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve a cache directory location with consistent fallbacks."""

    if preferred_base:
        return str(Path(preferred_base) / default_subdir)

    return get_storage_path("cache_path", default_subdir, cfg=cfg)

