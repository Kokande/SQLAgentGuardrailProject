from typing import Sequence
from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage


class Agent(ABC):
    @property
    @abstractmethod
    def _tools(self) -> Sequence[BaseTool]:
        pass

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
