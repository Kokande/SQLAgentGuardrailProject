from .core import BaseGuardrail, GuardrailResponse

from asyncio import Semaphore


class LLMGuardrail(BaseGuardrail):
    _semaphore: Semaphore

    def __init__(self, semaphore: Semaphore = None):
        self._semaphore = semaphore or Semaphore()

    def __repr__(self) -> str:
        return "LLMGuardrail"

    async def preprocess(self, message: str) -> GuardrailResponse:
        return GuardrailResponse(blocked=False, commentary="LLMPreprocessPassed;")

    async def postprocess(self, message: str) -> GuardrailResponse:
        return GuardrailResponse(blocked=False, commentary="LLMPostprocessPassed;")
