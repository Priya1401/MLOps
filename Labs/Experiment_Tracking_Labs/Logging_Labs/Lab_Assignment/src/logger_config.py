# logger_config.py
import logging
import logging.config
from pathlib import Path

LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
        },
        "simple": {
            "format": "%(levelname)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "INFO",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "level": "DEBUG",
            "filename": LOG_DIR / "app.log",
            "maxBytes": 5_000_000,
            "backupCount": 3,
            "encoding": "utf8",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "DEBUG",
    },
}

logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)
