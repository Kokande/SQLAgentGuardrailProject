import guardrail
from .core import Agent
from .config import LLMConfig
from agent.memory.checkpointer import get_sql_checkpointer

import logging
from asyncio import Semaphore
from typing import TypedDict, Sequence, Dict, List, ClassVar, Self, Type

from langgraph.graph import END, START
from langchain_gigachat import GigaChat
from langchain_core.tools import BaseTool
from langgraph.graph.message import BaseMessage
from langchain.chat_models.base import BaseChatModel
from langgraph.graph.state import StateGraph, CompiledStateGraph
from langchain_core.messages import HumanMessage, SystemMessage


LLM_SEMAPHORE: Semaphore = Semaphore(0)


def get_model(cfg: Type[LLMConfig]) -> GigaChat:
    return GigaChat(
        base_url=cfg.base_url,
        auth_url=cfg.auth_url,
        model=cfg.model,
        scope=cfg.scope,
        credentials=cfg.token,
        verify=False,
    )


class SQLAgent(Agent):
    """
    SQL agent singletone
    """

    _instance: ClassVar[Self | None] = None

    llm: BaseChatModel
    recursion_limit: int
    agent: CompiledStateGraph
    logger: logging.Logger

    @property
    def _system_prompt(self) -> SystemMessage:
        return SystemMessage(
            content="""
            Ты агент по работе с базой данных салона для домашних животных
            ...
            """  # todo Make a proper prompt
        )

    @property
    def _tools(self) -> Sequence[BaseTool]:
        return [
            ...  # todo Make tools
        ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize_instance()
        return cls._instance

    def __init__(self):
        pass

    def initialize_instance(self):
        self.logger = logging.getLogger("agent")
        self.recursion_limit = 25
        self.llm = get_model(LLMConfig)
        self._build_agent()

    def _build_agent(self) -> None:
        class AgentState(TypedDict):
            message: str
            messages: List[BaseMessage]
            remaining_steps: int
            reply: str

        class AgentInputSchema(TypedDict):
            message: str

        class AgentOutputSchema(TypedDict):
            reply: str

        async def preprocess(state: AgentState) -> dict:
            """
            Node made for initializing agent interaction
            :param state: current state of the agent
            :return: post-init state
            """
            return {"messages": [HumanMessage(content=state["message"])]}

        async def postprocess(state: AgentState) -> dict:
            """
            Nod made for separating agents output message
            :param state: current state of the agent
            :return: final state
            """
            return {"reply": state["messages"][-1].content}

        async def llm_node(state: AgentState) -> dict:
            """
            LLM communication node
            :param state: current state of the agent
            :return: message from LLM
            """
            llm_model = get_model(LLMConfig).bind_tools(self._tools)
            async with LLM_SEMAPHORE:
                llm_response = await llm_model.ainvoke(
                    [self._system_prompt] + state["messages"]
                )
            return {"messages": llm_response}

        workflow = StateGraph(
            state_schema=AgentState,
            input_schema=AgentInputSchema,
            output_schema=AgentOutputSchema,
        )
        workflow.add_node("preprocess", preprocess)
        workflow.add_node("llm_node", llm_node)
        workflow.add_node("postprocess", postprocess)

        workflow.add_edge(START, "preprocess")

        workflow.add_edge("preprocess", "llm_node")
        workflow.add_edge("llm_node", "postprocess")

        workflow.add_edge("postprocess", END)
        # todo Finish the graph

        self.agent = workflow.compile(checkpointer=get_sql_checkpointer())

    async def ainvoke(self, message: str) -> str:
        # response = await self.agent.ainvoke(message)
        response = {"reply": "agentmessage;"}

        return response["reply"]


class SafeAgent:
    agent: SQLAgent
    logger: logging.Logger
    guardrails: Dict[str, List[guardrail.BaseGuardrail]]

    def __init__(self):
        self.agent = SQLAgent()
        self.logger = logging.getLogger("guardrailed_agent")
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

    async def ainvoke(self, message: str, guardrail_type: str) -> str:
        response = ""
        if guardrail_type not in self.guardrails:
            self.logger.warning(
                f"No such guardrail: '{guardrail_type}'. Using 'disable'."
            )
            response = await self.agent.ainvoke(message)
        else:
            for mechanism in self.guardrails[guardrail_type]:
                preprocess = await mechanism.preprocess(message)
                if preprocess.blocked:
                    self.logger.info(
                        f"Blocked message: {message}. "
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
                    self.logger.info(
                        f"Blocked message: {message}. "
                        f"Caused by {mechanism} "
                        f"with message: {postprocess.commentary}"
                    )
                    return postprocess.commentary
                else:
                    response += postprocess.commentary

        return response


def init_agent() -> SafeAgent:
    global LLM_SEMAPHORE

    LLMConfig.load()
    LLM_SEMAPHORE = Semaphore(LLMConfig.num_streams)

    return SafeAgent()
