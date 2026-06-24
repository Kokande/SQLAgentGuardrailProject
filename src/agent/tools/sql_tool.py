from db.core import connect
from agent.core import CURRENT_THREAD_ID

import json
import logging
from datetime import datetime, timezone
from typing import Type

from pydantic import BaseModel
from langchain_gigachat.tools.giga_tool import GigaBaseTool


__all__ = ["SqlTool"]
logger = logging.getLogger("agent.tools")
_comm_logger = logging.getLogger("agent.communication")


def _log_tool_result(result: str) -> None:
    thread_id = CURRENT_THREAD_ID.get("unknown")
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread": thread_id,
        "type": "tool_result",
        "content": result,
    }
    _comm_logger.info(json.dumps(entry, ensure_ascii=False))


_MAX_ROWS = 100


class SqlInput(BaseModel):
    query: str


class SqlOutput(BaseModel):
    result: str
    error: bool = False


class SqlTool(GigaBaseTool):
    name: str = "sql_tool"
    description: str = """
        Делает запрос к базе данных SQLite3
        """
    args_schema: Type[BaseModel] = SqlInput
    return_schema: Type[BaseModel] = SqlOutput

    def _run(self, query: str) -> SqlOutput:
        raise NotImplementedError("Use async _arun instead.")

    async def _arun(self, query: str) -> SqlOutput:
        logger.info(f"Agents sql query: {query}")

        normalized = query.strip().upper().lstrip("(")
        if not normalized.startswith("SELECT"):
            _log_tool_result("Разрешены только SELECT-запросы.")
            return SqlOutput(
                result="Разрешены только SELECT-запросы.",
                error=True,
            )

        try:
            async with connect() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(query)
                    cols = (
                        [d[0] for d in cursor.description] if cursor.description else []
                    )
                    rows = await cursor.fetchmany(_MAX_ROWS)

            result_data = [dict(zip(cols, row)) for row in rows]
            result = json.dumps(result_data, ensure_ascii=False)
            if len(rows) == _MAX_ROWS:
                result += f"\n[Показано первые {_MAX_ROWS} строк]"

            logger.info(f"Got sql result ({len(rows)} rows)")
            _log_tool_result(result)
            return SqlOutput(result=result)
        except Exception as e:
            logger.error(f"Exception in the SQLTool's runtime: {e}")
            _log_tool_result(f"ОШИБКА: {e}")
            return SqlOutput(result=f"ОШИБКА: {e}", error=True)
