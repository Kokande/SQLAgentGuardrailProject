import guardrail
from config import AppConfig
from agent.memory.checkpointer import get_sql_checkpointer

import logging
from asyncio import Semaphore
from typing import TypedDict, Sequence, Dict, List

from pydantic import BaseModel
from langchain_gigachat import GigaChat
from langgraph.graph.message import BaseMessage
from langchain.chat_models.base import BaseChatModel
from langgraph.prebuilt import create_react_agent
from langchain_gigachat.tools.giga_tool import BaseTool
from langgraph.graph.state import StateGraph, CompiledStateGraph


def get_model(cfg: AppConfig):
    return GigaChat(
        base_url=cfg.llm.base_url,
        auth_url=cfg.llm.auth_url,
        model=cfg.llm.model,
        scope=cfg.llm.scope,
        credentials=cfg.llm.token,
        verify=False
    )


class Agent:
    llm: BaseChatModel
    semaphore: Semaphore
    recursion_limit: int
    agent: CompiledStateGraph
    logger: logging.Logger

    def __init__(self, cfg: AppConfig):
        self.semaphore = Semaphore(cfg.llm.num_streams)
        self.recursion_limit = 25
        self.llm = get_model(cfg)
        self.build_agent(cfg)
        self.logger = logging.getLogger("agent")

    def build_agent(self, cfg: AppConfig) -> None:
        class AgentState(TypedDict):
            messages: Sequence[BaseMessage]
            remaining_steps: int

        async def llm_node(
            state: AgentState
        ):
            async with self.semaphore:
                return

        workflow = StateGraph(
            state_schema=...,
            context_schema=...
        )
        # self.agent = workflow.compile(
        #     checkpointer=await get_sql_checkpointer()
        # )

    async def ainvoke(self, message: str) -> str:
        response = "agent_response;"

        return response


class SafeAgent:
    agent: Agent
    logger: logging.Logger
    guardrails: Dict[str, List[guardrail.BaseGuardrail]]

    def __init__(self, cfg):
        self.agent = Agent(cfg)
        self.logger = logging.getLogger("guardrailed_agent")
        self.init_guardrails()

    def init_guardrails(self):
        self.guardrails = {
            "disable": [],
            "regular_expressions": [guardrail.RegexGuardrail()],
            "machine_learning": [guardrail.MLGuardrail()],
            "large_language_model": [guardrail.LLMGuardrail(self.agent.semaphore)],
            "hybrid": [
                guardrail.RegexGuardrail(),
                guardrail.MLGuardrail(),
                guardrail.LLMGuardrail(self.agent.semaphore)
            ]
        }

    async def ainvoke(
            self,
            message: str,
            guardrail_type: str
    ) -> str:
        response = ""
        if guardrail_type not in self.guardrails:
            self.logger.warning(f"No such guardrail: '{guardrail_type}'. Using 'disable'.")
            response = await self.agent.ainvoke(message)
        else:
            for mechanism in self.guardrails[guardrail_type]:
                preprocess = await mechanism.preprocess(message)
                if preprocess.blocked:
                    self.logger.info(f"Blocked message: {message}. "
                                     f"Caused by {mechanism} "
                                     f"with message: {preprocess.commentary}"
                    )
                    return preprocess.commentary
                else:
                    response += preprocess.commentary

            agent_response = await self.agent.ainvoke(message)
            response += agent_response

            for mechanism in self.guardrails[guardrail_type]:
                postprocess = await mechanism.preprocess(agent_response)
                if postprocess.blocked:
                    self.logger.info(f"Blocked message: {message}. "
                                     f"Caused by {mechanism} "
                                     f"with message: {postprocess.commentary}"
                    )
                    return postprocess.commentary
                else:
                    response += postprocess.commentary

        return response


def init_agent(cfg: AppConfig) -> SafeAgent:
    return SafeAgent(cfg)
