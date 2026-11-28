"""Logging, with loguru.

Two places, each line with the time and the file:line that logged it:
- the terminal Loopy was started from: the conversation ("You:   ...",
  "Loopy: ...") and anything that goes wrong;
- a log file (loopy.log in Loopy's data folder, rotated at 5 MB, last 5
  kept), with the date and full tracebacks too, for figuring out what
  happened after the fact.

    from loopy.log import logger, setup_logging
    setup_logging()                 # once, at startup
    logger.info("You:   hello")
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from loguru import logger

from loopy.models import DATA_DIR

LOG_FILE = DATA_DIR / "logs" / "loopy.log"


def default_log_file() -> Path | None:
    """LOG_FILE, unless LOOPY_LOG_FILE says otherwise ("" = no file; the tests use that)."""
    env = os.environ.get("LOOPY_LOG_FILE")
    if env is None:
        return LOG_FILE
    return Path(env) if env else None


_DEFAULT = Path("<default>")
# Every line says when it was logged and which file and line logged it.
_TERMINAL_FORMAT = "{time:HH:mm:ss.SSS} | {level: <7} | {file}:{line: <4} | {message}"
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {file}:{line} ({function}) | {message}"


def setup_logging(log_file: Path | None = _DEFAULT, terminal=None) -> None:
    """Send logs to the terminal and (unless log_file is None) to the log
    file, and log crashes that would otherwise vanish: an exception on the
    main thread or in any background thread."""
    if log_file == _DEFAULT:
        log_file = default_log_file()
    logger.remove()
    out = terminal or sys.stdout
    if hasattr(out, "reconfigure"):
        out.reconfigure(errors="replace")  # the Windows console can't show every character
    logger.add(out, format=_TERMINAL_FORMAT, level="INFO", colorize=False)
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            logger.add(log_file, format=_FILE_FORMAT, level="DEBUG", rotation="5 MB", retention=5,
                       encoding="utf-8", enqueue=True, backtrace=True,
                       diagnose=False)  # no variable values in the file: they can hold what you said
        except OSError as exc:
            logger.warning(f"[note] can't write the log file ({exc}); logging to the terminal only")

    def on_crash(kind, value, tb):
        logger.opt(exception=(kind, value, tb)).critical("Loopy crashed")

    def on_thread_crash(args: threading.ExceptHookArgs):
        name = args.thread.name if args.thread else "?"
        logger.opt(exception=(args.exc_type, args.exc_value, args.exc_traceback)).error(
            f"[error in background thread {name}]")

    sys.excepthook = on_crash
    threading.excepthook = on_thread_crash


__all__ = ["LOG_FILE", "logger", "setup_logging"]
