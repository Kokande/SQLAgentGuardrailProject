from .config import LLMConfig

from asyncio import Semaphore
from typing import Type
from abc import ABC, abstractmethod

from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage


LLM_SEMAPHORE: Semaphore = Semaphore(0)


def get_model(cfg: Type[LLMConfig]) -> GigaChat:
    return GigaChat(
        base_url=cfg.base_url,
        auth_url=cfg.auth_url,
        model=cfg.model,
        scope=cfg.scope,
        credentials=cfg.token,
        verify_ssl_certs=False,
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
    async def ainvoke(self, message: str) -> str:
        pass


def init_core():
    global LLM_SEMAPHORE

    LLM_SEMAPHORE = Semaphore(LLMConfig.num_streams)
