from db.core import get_db_connection

from langgraph.checkpoint.sqlite import SqliteSaver


def get_sql_checkpointer() -> SqliteSaver:
    return SqliteSaver(get_db_connection())
