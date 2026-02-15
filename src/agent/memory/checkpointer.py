import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


def get_sql_checkpointer() -> AsyncSqliteSaver:
    return AsyncSqliteSaver(aiosqlite.connect("agent_memory.db"))
