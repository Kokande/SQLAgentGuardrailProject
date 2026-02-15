import guardrail
from .sql_agent import SQLAgent
from .core import LLM_SEMAPHORE

import logging
from typing import Dict, List


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
            raise NotImplementedError(
                "In-runtime guardrail selection not implemented :("
            )

        response = ""
        if guardrail_type not in self.guardrails:
            self.logger.warning(
                f"No such guardrail: '{guardrail_type}'. Using 'disable'."
            )
            response = await self.agent.ainvoke(message)
        else:
            for mechanism in self.guardrails[guardrail_type]:
                try:
                    preprocess = await mechanism.preprocess(message)

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
                postprocess = await mechanism.preprocess(agent_response)
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
