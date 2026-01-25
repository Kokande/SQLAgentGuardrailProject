from .core import BaseGuardrail, GuardrailResponse


class MLGuardrail(BaseGuardrail):
    async def preprocess(self, message: str) -> GuardrailResponse:
        return GuardrailResponse(blocked=False, commentary="MLPreprocessPassed;")

    async def postprocess(self, message: str) -> GuardrailResponse:
        return GuardrailResponse(blocked=False, commentary="MLPostprocessPassed;")
