import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("memorymesh")
    logger.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # stderr handler (safe for MCP)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    # file handler with rotation
    file_handler = RotatingFileHandler(
        "memorymesh.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger