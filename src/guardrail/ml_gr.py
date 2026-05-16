from .core import BaseGuardrail, GuardrailResponse

# Weighted keyword list for a simple bag-of-words toxicity heuristic.
# Positive score = more likely unsafe. Threshold at 1.0 blocks the message.
_TOXIC_KEYWORDS: dict[str, float] = {
    # Prompt-injection signals
    "ignore": 0.6,
    "forget": 0.6,
    "pretend": 0.5,
    "игнорируй": 0.7,
    "забудь": 0.7,
    "притворись": 0.6,
    "инструкции": 0.4,
    "instructions": 0.4,
    # Destructive SQL vocabulary
    "drop": 0.8,
    "delete": 0.7,
    "truncate": 0.9,
    "update": 0.5,
    "insert": 0.5,
    "alter": 0.7,
    "удали": 0.8,
    "удалить": 0.8,
    "очисти": 0.7,
    # Role-override signals
    "ты теперь": 0.7,
    "you are now": 0.7,
    "new persona": 0.6,
    "новый бот": 0.6,
    "другой бот": 0.6,
}

_BLOCK_THRESHOLD = 1.0


def _score(message: str) -> float:
    lower = message.lower()
    return sum(weight for kw, weight in _TOXIC_KEYWORDS.items() if kw in lower)


class MLGuardrail(BaseGuardrail):
    def __repr__(self) -> str:
        return "MLGuardrail"

    async def preprocess(self, message: str) -> GuardrailResponse:
        if _score(message) >= _BLOCK_THRESHOLD:
            return GuardrailResponse(
                blocked=True,
                commentary="Сообщение помечено как потенциально опасное.",
            )
        return GuardrailResponse(blocked=False, commentary="MLPassed;")

    async def postprocess(self, message: str) -> GuardrailResponse:
        if _score(message) >= _BLOCK_THRESHOLD:
            return GuardrailResponse(
                blocked=True,
                commentary="Ответ помечен как потенциально опасный.",
            )
        return GuardrailResponse(blocked=False, commentary="MLPassed;")
