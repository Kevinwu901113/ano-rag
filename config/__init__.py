from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if SRC_ROOT.exists():
    src_str = str(SRC_ROOT)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from .config_loader import ConfigLoader, config  # noqa: E402

__all__ = ["ConfigLoader", "config"]
