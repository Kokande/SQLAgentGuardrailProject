from config import AppConfig


class Agent:
    ...


class SafeAgent:
    ...


def init_agent(cfg: AppConfig) -> SafeAgent:
    return SafeAgent()
