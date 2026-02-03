from .core import init_core
from .config import LLMConfig
from .safe_agent import SafeAgent

__all__ = ["SafeAgent", "init_agent"]


def init_agent() -> SafeAgent:
    LLMConfig.load()
    init_core()

    return SafeAgent()
