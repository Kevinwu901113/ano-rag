"""Core configuration package providing the SSOT config pipeline."""

from .schema import RootConfig, DEFAULT_CONFIG  # noqa: F401
from .loader import load_config, PRIORITY  # noqa: F401
from .types import ModuleConfig  # noqa: F401
from .freeze import FrozenConfig  # noqa: F401
