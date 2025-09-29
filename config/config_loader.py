from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from core.config.loader import load_config
from core.config.printer import format_diagnostics
from core.config.freeze import FrozenConfig
from core.config.utils import deep_merge, set_path


class ConfigLoader:
    """Load and access configuration using the SSOT pipeline."""

    def __init__(
        self,
        config_path: str | None = None,
        profile: str | None = None,
        *,
        strict_unknowns: bool | None = None,
    ):
        self.profile = profile
        # 如果未指定 config_path，则默认读取仓库根目录下的 config.yaml
        self._config_path = (
            Path(config_path)
            if config_path
            else Path(__file__).resolve().parent.parent / "config.yaml"
        )
        self._overrides: Dict[str, Any] = {}
        self._frozen: Optional[FrozenConfig] = None
        self._normalized: Optional[Dict[str, Any]] = None
        self._layers: list[str] = []
        self.deprecated_keys: list[str] = []
        # 支持通过 strict_unknowns 控制对未知配置键的处理方式
        self._strict_unknowns = strict_unknowns

    def load_config(self) -> FrozenConfig:
        if self._frozen is None:
            extra_layers = self._resolve_extra_layers()
            # 将 strict_unknowns 传递给 load_config，以便控制未知键校验
            frozen, model, normalized, layers = load_config(
                profile=self.profile,
                extra_layers=extra_layers,
                inline_overrides=self._overrides,
                strict_unknowns=self._strict_unknowns,
            )
            self._frozen = frozen
            self._normalized = normalized
            self._layers = layers
            diagnostics = normalized.get("_diagnostics", {})
            self.deprecated_keys = diagnostics.get("deprecated_paths", [])
        return self._frozen

    def _resolve_extra_layers(self) -> Iterable[str]:
        layers = []
        if self._config_path and self._config_path.exists():
            layers.append(str(self._config_path))
        return layers

    def get(self, key: str, default: Any = None) -> Any:
        config = self.load_config()
        return config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        set_path(self._overrides, tuple(key.split(".")), deepcopy(value))
        self._frozen = None

    def update_config(self, updates: Dict[str, Any]) -> None:
        self._overrides = deep_merge(self._overrides, updates)
        self._frozen = None

    def to_dict(self) -> Dict[str, Any]:
        config = self.load_config()
        return config.to_dict()

    def save_config(self) -> None:
        if not self._config_path:
            return
        data = deepcopy(self._overrides) or {}
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with self._config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)

    def diagnostics(self) -> str:
        self.load_config()
        diagnostics = (
            self._normalized.get("_diagnostics", {}) if self._normalized else {}
        )
        return format_diagnostics(diagnostics)


# 默认实例，用于在其他模块中直接导入和使用
config = ConfigLoader()
