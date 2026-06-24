import re

from .core import BaseGuardrail, GuardrailResponse

# Patterns that indicate prompt-injection or attempts to break the agent's role.
_INJECTION_PATTERNS = [
    r"игнорируй\s+(предыдущие|все)\s+инструкции",
    r"ignore\s+(previous|all)\s+instructions",
    r"forget\s+(everything|your\s+instructions)",
    r"забудь\s+(всё|все\s+инструкции)",
    r"ты\s+теперь\s+другой\s+бот",
    r"притворись",
    r"pretend\s+(you\s+are|to\s+be)",
    r"you\s+are\s+now\s+(?!a\s+helpful)",
]

# Patterns that indicate an attempt to get the LLM to generate destructive SQL.
_DESTRUCTIVE_SQL_PATTERNS = [
    r"\bdrop\s+table\b",
    r"\btruncate\s+table\b",
    r"\bdelete\s+from\b",
    r"\bupdate\s+\w+\s+set\b",
    r"\bdrop\s+database\b",
    r"\balter\s+table\b",
    r"\binsert\s+into\b",
]

# Patterns that match personal data items in responses (phone numbers and emails).
# Used only in postprocess to detect bulk personal data dumps.
_PERSONAL_DATA_PATTERNS = [
    r"\+7[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}",  # +7 (xxx) xxx-xx-xx
    r"\b8[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b",  # 8 (xxx) xxx-xx-xx
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",  # email address
]
# Block the response when more than this many personal data items appear (bulk dump guard).
_PERSONAL_DATA_THRESHOLD = 4

_MAX_MESSAGE_LEN = 2000

_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.UNICODE)
_DESTRUCTIVE_RE = re.compile("|".join(_DESTRUCTIVE_SQL_PATTERNS), re.IGNORECASE)
_PERSONAL_DATA_RE = re.compile("|".join(_PERSONAL_DATA_PATTERNS), re.IGNORECASE)


def _check_input(message: str) -> GuardrailResponse:
    if len(message) > _MAX_MESSAGE_LEN:
        return GuardrailResponse(
            blocked=True,
            commentary="Сообщение слишком длинное. Пожалуйста, сократите запрос.",
        )
    if _INJECTION_RE.search(message):
        return GuardrailResponse(
            blocked=True,
            commentary="Запрос не соответствует допустимым темам.",
        )
    if _DESTRUCTIVE_RE.search(message):
        return GuardrailResponse(
            blocked=True,
            commentary="Запросы на изменение или удаление данных не разрешены.",
        )
    return GuardrailResponse(blocked=False, commentary="RegexPassed;")


def _check_output(message: str) -> GuardrailResponse:
    result = _check_input(message)
    if result.blocked:
        return result
    hits = len(_PERSONAL_DATA_RE.findall(message))
    if hits > _PERSONAL_DATA_THRESHOLD:
        return GuardrailResponse(
            blocked=True,
            commentary="Ответ содержит слишком много персональных данных и был заблокирован в целях безопасности.",
        )
    return GuardrailResponse(blocked=False, commentary="RegexPassed;")


class RegexGuardrail(BaseGuardrail):
    def __repr__(self) -> str:
        return "RegexGuardrail"

    async def preprocess(self, message: str) -> GuardrailResponse:
        return _check_input(message)

    async def postprocess(self, message: str) -> GuardrailResponse:
        return _check_output(message)
