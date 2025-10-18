from db.core import get_db_connection

from langgraph.checkpoint.sqlite import SqliteSaver


async def get_sql_checkpointer() -> SqliteSaver:
    return SqliteSaver(await get_db_connection())
