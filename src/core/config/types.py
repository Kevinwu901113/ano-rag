from __future__ import annotations

from dataclasses import dataclass

from .freeze import FrozenConfig


@dataclass(frozen=True)
class ModuleConfig:
    """Strongly typed slices of the frozen config for dependency injection."""

    rerank: FrozenConfig
    retrieval: FrozenConfig
    hybrid_search: FrozenConfig
    multi_hop: FrozenConfig
    dispatcher: FrozenConfig
    context_dispatcher: FrozenConfig
    scheduler: FrozenConfig
    storage: FrozenConfig

    @classmethod
    def from_root(cls, config: FrozenConfig) -> "ModuleConfig":
        return cls(
            rerank=config["rerank"],
            retrieval=config["retrieval"],
            hybrid_search=config["hybrid_search"],
            multi_hop=config["multi_hop"],
            dispatcher=config["dispatcher"],
            context_dispatcher=config["context_dispatcher"],
            scheduler=config.get("scheduler", {}),
            storage=config.get("storage", {}),
        )
