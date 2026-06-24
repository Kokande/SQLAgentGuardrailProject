from .config import LLMConfig

import os
import contextvars
from typing import Type
from asyncio import Semaphore
from abc import ABC, abstractmethod

from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage


LLM_SEMAPHORE: Semaphore = Semaphore(int(os.getenv("LLM_STREAMS", "1")))
CURRENT_THREAD_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "thread_id", default=None
)


def get_model(cfg: Type[LLMConfig]) -> GigaChat:
    return GigaChat(
        base_url=cfg.base_url,
        auth_url=cfg.auth_url,
        model=cfg.model,
        scope=cfg.scope,
        credentials=cfg.token,
        verify_ssl_certs=cfg.verify_ssl,
    )


class Agent(ABC):
    @property
    @abstractmethod
    def _system_prompt(self) -> SystemMessage:
        pass

    @abstractmethod
    def _build_agent(self) -> None:
        pass

    @abstractmethod
    async def ainvoke(self, message: str, configurable: dict) -> str:
        pass


def init_core():
    global LLM_SEMAPHORE

    LLM_SEMAPHORE = Semaphore(LLMConfig.num_streams)
