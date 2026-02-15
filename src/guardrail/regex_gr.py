from .core import BaseGuardrail, GuardrailResponse


class RegexGuardrail(BaseGuardrail):
    def __repr__(self) -> str:
        return "RegexGuardrail"

    async def preprocess(self, message: str) -> GuardrailResponse:
        return GuardrailResponse(blocked=False, commentary="RegexPreprocessPassed;")

    async def postprocess(self, message: str) -> GuardrailResponse:
        return GuardrailResponse(blocked=False, commentary="RegexPostprocessPassed;")
