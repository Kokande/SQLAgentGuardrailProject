from abc import ABC, abstractmethod
from pydantic import BaseModel


class GuardrailResponse(BaseModel):
    blocked: bool
    commentary: str


class BaseGuardrail(ABC):
    @abstractmethod
    async def preprocess(self, message: str) -> GuardrailResponse:
        pass

    @abstractmethod
    async def postprocess(self, message: str) -> GuardrailResponse:
        pass
