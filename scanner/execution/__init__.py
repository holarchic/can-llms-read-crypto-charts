"""Auto-execution subpackage — trade placement for CONFIRM signals."""

from .config import ExecutionConfig, load_execution_config

__all__ = ["ExecutionConfig", "load_execution_config"]
