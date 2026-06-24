from .config import LLMConfig
from .core import Agent, LLM_SEMAPHORE, CURRENT_THREAD_ID, get_model
from agent.tools.sql_tool import SqlTool
from agent.memory.checkpointer import get_sql_checkpointer

import json as _json
import logging
from datetime import datetime, timezone
from typing import TypedDict, Annotated, ClassVar, Self, List

from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import BaseMessage
from langgraph.graph.message import add_messages
from langchain.chat_models.base import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import StateGraph, CompiledStateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

_comm_logger = logging.getLogger("agent.communication")


def _log_comm(thread: str, msg_type: str, **fields) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread": thread,
        "type": msg_type,
        **fields,
    }
    _comm_logger.info(_json.dumps(entry, ensure_ascii=False))


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
            Ты — дружелюбный помощник зоосалона для домашних животных.

            ВАЖНЫЕ ПРАВИЛА:
            - Никогда не упоминай базы данных, SQL, таблицы, запросы или технические детали.
            - Говори как заботливый администратор салона, а не как программа.
            - Если для поиска нужны данные клиента (имя или телефон) — вежливо попроси их уточнить.

            ФОРМАТИРОВАНИЕ (для Telegram, без стандартного markdown):
            - Используй **жирный** для имён и дат.
            - Списки оформляй через "•" или нумерацию, по одному пункту на строку.
            - Если результатов много (5+) — сначала дай краткую сводку ("У вас 3 предстоящих визита:"),
              затем перечисли самые важные. Не выводи длинные списки одним блоком.
            - Не показывай технические идентификаторы (clientId, stayId и т.д.).
            - Отвечай кратко и понятно — сообщение должно хорошо читаться на экране телефона.
            - Не используй стандартный markdown. Форматирование Telegram:
              **жирный**, __курсив__, ~~зачёркнутый~~, ||скрытый||

            СПРАВОЧНИК (только для твоего понимания структуры, пользователю не показывай):
            CREATE TABLE IF NOT EXISTS Clients (
                clientId TEXT PRIMARY KEY,  -- внутренний ID, не показывай
                fullName TEXT,              -- полное имя клиента
                phoneNumber TEXT,           -- номер телефона
                email TEXT,
                address TEXT
            );
            CREATE TABLE IF NOT EXISTS Pets (
                petId TEXT PRIMARY KEY,     -- внутренний ID, не показывай
                clientId TEXT,
                name TEXT,                  -- кличка питомца
                nice INTEGER,               -- 1 = дружелюбный, 0 = агрессивный
                mof TEXT,                   -- пол: male / female
                description TEXT,           -- описание, особенности
                FOREIGN KEY (clientId) REFERENCES Clients(clientId)
            );
            CREATE TABLE IF NOT EXISTS Stays (
                stayId TEXT PRIMARY KEY,    -- внутренний ID, не показывай
                clientId TEXT,
                petId TEXT,
                visit TEXT,                 -- дата заезда (начало визита)
                leave TEXT,                 -- дата выезда (конец визита)
                plan TEXT,                  -- запланированные процедуры / услуги
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

        async def preprocess(state: AgentState, config: RunnableConfig) -> dict:
            """
            Node made for initializing agent interaction
            :param state: current state of the agent
            :return: post-init state
            """
            thread_id = config.get("configurable", {}).get("thread_id", "unknown")
            CURRENT_THREAD_ID.set(thread_id)
            _log_comm(thread_id, "user_message", content=state["message"])
            return {"messages": [HumanMessage(content=state["message"])]}

        async def llm_node(state: AgentState) -> dict:
            """
            LLM communication node
            :param state: current state of the agent
            :return: message from LLM
            """
            thread_id = CURRENT_THREAD_ID.get("unknown")
            self.logger.debug(f"Entered llm_node with state: {state}")

            llm_model = self.llm.bind_tools(self._tools)
            async with LLM_SEMAPHORE:
                llm_response = await llm_model.ainvoke(
                    [self._system_prompt] + state["messages"]
                )

            self.logger.info(f"LLM response: {llm_response}")

            if llm_response.tool_calls:
                for tc in llm_response.tool_calls:
                    _log_comm(
                        thread_id,
                        "tool_call",
                        tool_name=tc["name"],
                        tool_input=tc["args"],
                    )
            else:
                _log_comm(thread_id, "ai_message", content=llm_response.content)

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
