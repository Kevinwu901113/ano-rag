from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

from .schema import RootConfig


class FrozenConfig(Mapping[str, Any]):
    """Immutable mapping wrapper around the validated configuration."""

    def __init__(self, data: Dict[str, Any]):
        self._data = {key: _freeze(value) for key, value in data.items()}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"FrozenConfig({self._data!r})"

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data
        for part in key.split('.'):
            if isinstance(value, Mapping) and part in value:
                value = value[part]
            else:
                return default
        return value

    def to_dict(self) -> Dict[str, Any]:
        def _unfreeze(item: Any) -> Any:
            if isinstance(item, FrozenConfig):
                return {k: _unfreeze(v) for k, v in item.items()}
            if isinstance(item, FrozenMapping):
                return {k: _unfreeze(v) for k, v in item.items()}
            if isinstance(item, tuple):
                return tuple(_unfreeze(v) for v in item)
            return item

        return {key: _unfreeze(value) for key, value in self._data.items()}


class FrozenMapping(Mapping[str, Any]):
    def __init__(self, data: Dict[str, Any]):
        self._data = {key: _freeze(value) for key, value in data.items()}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data
        for part in key.split('.'):
            if isinstance(value, Mapping) and part in value:
                value = value[part]
            else:
                return default
        return value

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"FrozenMapping({self._data!r})"


def _freeze(value: Any) -> Any:
    if isinstance(value, FrozenConfig | FrozenMapping):
        return value
    if isinstance(value, dict):
        return FrozenMapping(value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def freeze(model: RootConfig) -> FrozenConfig:
    return FrozenConfig(model.model_dump())
