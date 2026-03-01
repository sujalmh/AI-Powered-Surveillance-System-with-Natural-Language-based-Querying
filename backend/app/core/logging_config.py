"""
Loguru-based logging for the backend.
- Console: INFO + ERROR only (clean output).
- File: ALL levels (DEBUG+) for full diagnostics.
- Log files written to data/logs/ with daily rotation.
"""
import logging
import os
import sys
from pathlib import Path

from loguru import logger


class _InterceptHandler(logging.Handler):
    """
    Redirect all stdlib logging into Loguru so that third-party libs
    (uvicorn, pymongo, langchain …) also go through loguru sinks.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Find caller from where the logged message originated
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _console_filter(record: dict) -> bool:
    """Only allow INFO and ERROR (not DEBUG, WARNING, CRITICAL) to console."""
    return record["level"].name in ("INFO", "ERROR")


def configure_logging(
    log_level: str = "DEBUG",
    log_json: bool = False,
) -> None:
    """
    Configure loguru sinks.  Call once at app startup.
    - Console sink: INFO + ERROR only, human-readable.
    - File sink: ALL levels (DEBUG+), rotated daily, kept 30 days.
    """
    # Remove loguru's default stderr sink
    logger.remove()

    # ── Console sink: INFO + ERROR only ──────────────────────────────
    logger.add(
        sys.stdout,
        level="DEBUG",              # accept everything, filter below
        filter=_console_filter,     # only pass INFO and ERROR
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>",
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    # ── File sink: everything (DEBUG+) ───────────────────────────────
    log_dir = Path(os.getenv("STORAGE_ROOT", "data")).resolve() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"

    if log_json or os.getenv("LOG_JSON", "false").lower() == "true":
        logger.add(
            str(log_dir / "app_{time:YYYY-MM-DD}.log"),
            level="DEBUG",
            format=log_format,
            rotation="00:00",       # new file every midnight
            retention="30 days",
            compression="gz",
            serialize=True,         # JSON lines
            backtrace=True,
            diagnose=True,
            encoding="utf-8",
        )
    else:
        logger.add(
            str(log_dir / "app_{time:YYYY-MM-DD}.log"),
            level="DEBUG",
            format=log_format,
            rotation="00:00",
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=True,
            encoding="utf-8",
        )

    # ── Intercept stdlib logging → loguru ────────────────────────────
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Silence noisy third-party loggers
    for noisy in ("uvicorn", "uvicorn.access", "uvicorn.error",
                   "pymongo", "httpcore", "httpx", "openai",
                   "langchain", "langchain_core", "langchain_openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("Loguru logging configured — console: INFO+ERROR | file: ALL → {}", log_dir)


def get_logger(name: str):
    """Return a loguru logger bound to the given module name."""
    return logger.bind(name=name)
