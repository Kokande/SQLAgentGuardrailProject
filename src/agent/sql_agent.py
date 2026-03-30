from .config import LLMConfig
from .core import Agent, LLM_SEMAPHORE, get_model
from agent.tools.sql_tool import SqlTool
from agent.memory.checkpointer import get_sql_checkpointer

import logging
from typing import TypedDict, Annotated, ClassVar, Self, List

from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import BaseMessage
from langgraph.graph.message import add_messages
from langchain.chat_models.base import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import StateGraph, CompiledStateGraph
from langchain_core.messages import HumanMessage, SystemMessage


class SQLAgent(Agent):
    """
    SQL agent singletone
    """

    _instance: ClassVar[Self | None] = None
    _tools: List = [SqlTool()]

    llm: BaseChatModel
    recursion_limit: int
    agent: CompiledStateGraph

    tool_node: ToolNode
    checkpointer: BaseCheckpointSaver

    logger: logging.Logger

    @property
    def _system_prompt(self) -> SystemMessage:
        return SystemMessage(
            content="""
            Ты - агент чат-бот по работе с базой данных салона для домашних животных.
            Помогай пользователю в работе с БД.

            Пользователь - простой человек, старайся не общаться с ним кодом.

            Не используй markdown-форматирование.
            Вместо - используй форматирование социальной сети Telegram:
            **жирный**
            __курсив__
            `код`
            ~~перечеркнутый~~
            ```блок кода```
            ||скрытый текст||

            Вот миграции БД, чтобы ты знал её состав:

            CREATE TABLE IF NOT EXISTS Chats (
                userId TEXT NOT NULL PRIMARY KEY,
                sessionId TEXT NOT NULL,
                guardrailType TEXT NOT NULL
            )
            CREATE TABLE IF NOT EXISTS Clients (
                clientId TEXT PRIMARY KEY NOT NULL,
                fullName TEXT NOT NULL,
                phoneNumber TEXT NOT NULL,
                email TEXT,
                address TEXT
            );
            CREATE TABLE IF NOT EXISTS Pets (
                petId TEXT PRIMARY KEY NOT NULL,
                clientId TEXT NOT NULL,
                name TEXT,
                nice INTEGER NOT NULL,
                mof TEXT NOT NULL,
                description TEXT,
                FOREIGN KEY (clientId) REFERENCES Clients(clientId)
            );
            CREATE TABLE IF NOT EXISTS Stays (
                stayId TEXT PRIMARY KEY NOT NULL,
                clientId TEXT NOT NULL,
                petId TEXT NOT NULL,
                visit TEXT NOT NULL,
                leave TEXT NOT NULL,
                plan TEXT NOT NULL,
                FOREIGN KEY (clientId) REFERENCES Clients(clientId),
                FOREIGN KEY (petId) REFERENCES Pets(petId)
            );
            """
        )

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
        self.tool_node = ToolNode(self._tools)
        self.checkpointer = get_sql_checkpointer()
        self._build_agent()

    def _build_agent(self) -> None:
        class AgentState(TypedDict):
            message: str
            messages: Annotated[List[BaseMessage], add_messages]
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

        async def llm_node(state: AgentState) -> dict:
            """
            LLM communication node
            :param state: current state of the agent
            :return: message from LLM
            """
            self.logger.debug(f"Entered llm_node with state: {state}")

            llm_model = get_model(LLMConfig).bind_tools(self._tools)
            async with LLM_SEMAPHORE:
                llm_response = await llm_model.ainvoke(
                    [self._system_prompt] + state["messages"]
                )

            self.logger.info(f"LLM response: {llm_response}")

            return {"messages": llm_response}

        async def postprocess(state: AgentState) -> dict:
            """
            Nod made for separating agents output message
            :param state: current state of the agent
            :return: final state
            """
            return {"reply": state["messages"][-1].content}

        async def should_continue(state: AgentState):
            """
            Conditional node
            """
            if state["messages"][-1].tool_calls:
                return "tools"
            return "postprocess"

        workflow = StateGraph(
            state_schema=AgentState,
            input_schema=AgentInputSchema,
            output_schema=AgentOutputSchema,
        )
        workflow.add_node("preprocess", preprocess)
        workflow.add_node("llm_node", llm_node)
        workflow.add_node("postprocess", postprocess)
        workflow.add_node("tools", self.tool_node)

        workflow.add_edge(START, "preprocess")
        workflow.add_edge("preprocess", "llm_node")
        workflow.add_conditional_edges(
            "llm_node", should_continue, ["tools", "postprocess"]
        )
        workflow.add_edge("tools", "llm_node")
        workflow.add_edge("postprocess", END)

        self.agent = workflow.compile(checkpointer=self.checkpointer)

    async def ainvoke(self, message: str, configurable: dict) -> str:
        response = await self.agent.ainvoke({"message": message}, config=configurable)

        return response["reply"]
