from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from typing import Any, Dict, Iterable, Tuple

import os
import re

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Deeply merge two dictionaries returning a new dict."""
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def expand_env(data: Any) -> Any:
    """Recursively expand ${VAR} placeholders using environment variables."""
    if isinstance(data, str):
        def _replace(match: re.Match[str]) -> str:
            var = match.group(1)
            return os.environ.get(var, "")

        return _ENV_PATTERN.sub(_replace, data)
    if isinstance(data, list):
        return [expand_env(item) for item in data]
    if isinstance(data, dict):
        return {key: expand_env(value) for key, value in data.items()}
    return data


def get_path(data: Mapping[str, Any], path: Tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def set_path(data: MutableMapping[str, Any], path: Tuple[str, ...], value: Any) -> None:
    current: MutableMapping[str, Any] = data
    for key in path[:-1]:
        next_val = current.get(key)
        if not isinstance(next_val, MutableMapping):
            next_val = {}
            current[key] = next_val
        current = next_val  # type: ignore[assignment]
    current[path[-1]] = value


def delete_path(data: MutableMapping[str, Any], path: Tuple[str, ...]) -> None:
    parents: list[MutableMapping[str, Any]] = []
    keys: list[str] = []
    current: Any = data
    for key in path[:-1]:
        if not isinstance(current, MutableMapping) or key not in current:
            return
        parents.append(current)
        keys.append(key)
        current = current[key]
    if isinstance(current, MutableMapping) and path[-1] in current:
        del current[path[-1]]
        # prune empty parents
        for parent, key in reversed(list(zip(parents, keys))):
            child = parent[key]
            if isinstance(child, MutableMapping) and not child:
                del parent[key]
            else:
                break


def iter_paths(data: Mapping[str, Any], prefix: Tuple[str, ...] = ()) -> Iterable[Tuple[str, ...]]:
    for key, value in data.items():
        current = prefix + (key,)
        yield current
        if isinstance(value, Mapping):
            yield from iter_paths(value, current)
