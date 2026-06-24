import guardrail
from .sql_agent import SQLAgent
from .core import LLM_SEMAPHORE

import json as _json
import logging
from datetime import datetime, timezone
from typing import Dict, List

_guardrail_logger = logging.getLogger("agent.guardrail_reactions")


def _log_guardrail(
    thread: str, name: str, phase: str, analyzed: str, blocked: bool, details: str
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread": thread,
        "guardrail_name": name,
        "phase": phase,
        "analyzed_message": analyzed,
        "verdict": "BLOCKED" if blocked else "PASSED",
        "details": details,
    }
    _guardrail_logger.info(_json.dumps(entry, ensure_ascii=False))


class SafeAgent:
    agent: SQLAgent
    logger: logging.Logger
    guardrails: Dict[str, List[guardrail.BaseGuardrail]]
    exception_reply: str = (
        "Извините, кажется что-то пошло не так. Попробуйте написать мне ещё раз. "
    )

    def __init__(self):
        self.agent = SQLAgent()
        self.logger = logging.getLogger("agent.guardrail")
        self.init_guardrails()

    def init_guardrails(self):
        self.guardrails = {
            "disable": [],
            "regular_expressions": [guardrail.RegexGuardrail()],
            "machine_learning": [guardrail.MLGuardrail()],
            "large_language_model": [guardrail.LLMGuardrail(LLM_SEMAPHORE)],
            "hybrid": [
                guardrail.RegexGuardrail(),
                guardrail.MLGuardrail(),
                guardrail.LLMGuardrail(LLM_SEMAPHORE),
            ],
        }

    async def ainvoke(
        self, message: str, session_id: str, guardrail_type: str = None
    ) -> str:
        if guardrail_type is None:
            self.logger.warning("guardrail_type not provided; defaulting to 'disable'.")
            guardrail_type = "disable"

        response = ""
        if guardrail_type not in self.guardrails:
            self.logger.warning(
                f"No such guardrail: '{guardrail_type}'. Using 'disable'."
            )
            response = await self.agent.ainvoke(
                message, configurable={"configurable": {"thread_id": session_id}}
            )
        else:
            for mechanism in self.guardrails[guardrail_type]:
                try:
                    preprocess = await mechanism.preprocess(message)
                    _log_guardrail(
                        session_id,
                        type(mechanism).__name__,
                        "preprocess",
                        message,
                        preprocess.blocked,
                        preprocess.commentary,
                    )

                    self.logger.info(
                        f"{mechanism} pre-check passed for message '{message}'"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed the {guardrail_type} check: {e}. Returning the prepared message"
                    )

                    return self.exception_reply

                if preprocess.blocked:
                    self.logger.info(
                        f"Blocked message: {message}. "
                        f"Caused by {mechanism} "
                        f"with message: {preprocess.commentary}"
                    )

                    return preprocess.commentary
                else:
                    # response += preprocess.commentary + "\n"
                    pass

            agent_response = await self.agent.ainvoke(
                message, configurable={"configurable": {"thread_id": session_id}}
            )
            response += agent_response

            for mechanism in self.guardrails[guardrail_type]:
                postprocess = await mechanism.postprocess(agent_response)
                _log_guardrail(
                    session_id,
                    type(mechanism).__name__,
                    "postprocess",
                    agent_response,
                    postprocess.blocked,
                    postprocess.commentary,
                )
                if postprocess.blocked:
                    self.logger.info(
                        f"Blocked message: {message}. "
                        f"Caused by {mechanism} "
                        f"with message: {postprocess.commentary}"
                    )

                    return postprocess.commentary
                else:
                    # response += postprocess.commentary
                    self.logger.info(
                        f"{mechanism} post-check passed for message '{response}'"
                    )

        return response
