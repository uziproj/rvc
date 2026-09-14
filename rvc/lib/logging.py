"""RVC centralized logging system.

Provides a single, configurable logger shared across all RVC modules.
Supports multiple levels (DEBUG / INFO / WARNING / ERROR / CRITICAL),
colored terminal output, and avoids the duplicate-handler bug that
previously caused every log line to appear twice.

Usage from user code:

    from rvc import Config, set_log_level

    # Set level via Config (recommended)
    config = Config(log_level="debug")

    # Or change at runtime
    set_log_level("warning")

    # Or via RVClass
    from rvc import RVClass
    rvc = RVClass(config=config, pth_path="model.pth", log_level="info")

Usage from internal modules:

    from rvc.lib.logging import get_logger
    logger = get_logger("utils")
    logger.info("Doing something")
    logger.debug("Detailed info")
    logger.warning("Heads up")
    logger.error("Something broke")
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
LOG_LEVELS = {
    "debug": logging.DEBUG,      # 10
    "info": logging.INFO,        # 20
    "warning": logging.WARNING,  # 30
    "warn": logging.WARNING,     # alias
    "error": logging.ERROR,      # 40
    "critical": logging.CRITICAL,  # 50
    "fatal": logging.CRITICAL,    # alias
}

DEFAULT_LEVEL = "info"
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# ANSI color codes (auto-disabled when stderr is not a TTY)
_COLORS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET":    "\033[0m",
}


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------
class ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to the log level name."""

    def __init__(self, use_color: bool = True,
                 fmt: str = DEFAULT_FORMAT,
                 datefmt: str = DEFAULT_DATEFMT):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color and record.levelname in _COLORS:
            # Color a copy of the levelname so we don't mutate the original
            record.levelname = (
                f"{_COLORS[record.levelname]}{record.levelname}"
                f"{_COLORS['RESET']}"
            )
        return super().format(record)


# ---------------------------------------------------------------------------
# Internal state (module-level singletons)
# ---------------------------------------------------------------------------
_logger_configured: bool = False
_current_level: int = logging.INFO
_current_use_color: Optional[bool] = None


def _resolve_level(level: Union[str, int, None]) -> int:
    """Resolve a level string/int/None into a numeric logging level."""
    if level is None:
        return _current_level
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return LOG_LEVELS.get(level.lower().strip(), logging.INFO)
    raise TypeError(f"Unsupported log level type: {type(level).__name__}")


def _configure_root_logger(level: int = logging.INFO,
                           use_color: Optional[bool] = None) -> None:
    """Configure the root 'rvc' logger.

    Idempotent: removes any existing handlers before adding a fresh one,
    so calling this multiple times never produces duplicate output.
    """
    global _logger_configured, _current_level, _current_use_color

    rvc_logger = logging.getLogger("rvc")
    rvc_logger.setLevel(level)
    # IMPORTANT: don't propagate to the root logger — that's what caused
    # the duplicate-output bug where every line appeared twice with two
    # different formats.
    rvc_logger.propagate = False

    # Remove all existing handlers (prevents duplicate handlers on re-config)
    for h in list(rvc_logger.handlers):
        rvc_logger.removeHandler(h)

    if use_color is None:
        use_color = sys.stderr.isatty() and sys.stderr.isatty()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColorFormatter(use_color=use_color))
    rvc_logger.addHandler(handler)

    _logger_configured = True
    _current_level = level
    _current_use_color = use_color


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_logger(name: str = "rvc") -> logging.Logger:
    """Return a logger under the 'rvc' namespace.

    All RVC modules should use this instead of ``logging.getLogger(__name__)``
    so that the configured formatter and propagation settings are respected.

    Args:
        name: Sub-logger name. Will be prefixed with 'rvc.' if it doesn't
            already start with 'rvc.'. Pass 'rvc' (or omit) to get the
            top-level RVC logger.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    if not name.startswith("rvc"):
        name = f"rvc.{name}"
    # Ensure the root 'rvc' logger is configured at least once
    if not _logger_configured:
        _configure_root_logger(_current_level, use_color=_current_use_color)
    return logging.getLogger(name)


def set_log_level(level: Union[str, int] = DEFAULT_LEVEL,
                  use_color: Optional[bool] = None) -> None:
    """Set the global log level for ALL RVC loggers.

    Args:
        level: One of 'debug', 'info', 'warning', 'error', 'critical'
               (case-insensitive), or a ``logging`` level integer like
               ``logging.DEBUG``.
        use_color: Force-enable or force-disable ANSI colors. If ``None``,
                   auto-detects based on whether stderr is a TTY.
    """
    resolved = _resolve_level(level)
    # Reset the configured flag so _configure_root_logger actually re-runs
    global _logger_configured
    _logger_configured = False
    _configure_root_logger(resolved, use_color=use_color)


def get_log_level() -> str:
    """Return the current log level as a lowercase string."""
    for name, value in LOG_LEVELS.items():
        if value == _current_level:
            return name
    return "info"


# ---------------------------------------------------------------------------
# Convenience: configure on import with a sane default
# ---------------------------------------------------------------------------
# Don't configure here — defer until first get_logger() / set_log_level() call
# so that Config(log_level=...) is the single source of truth.
