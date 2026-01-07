import os
import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_path: str | None = None):
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    if log_path:
        Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
        logger.add(log_path, rotation="5 MB", retention=5, level="INFO")
    return logger
