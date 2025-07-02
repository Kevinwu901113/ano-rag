from loguru import logger
import os
import sys


def setup_logging(log_file: str):
    """Configure loguru to log to the given file with rotation."""
    os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        log_file,
        rotation="10 MB",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    return logger
