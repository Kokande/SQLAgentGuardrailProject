from db.core import connect

import asyncio
import logging
from typing import Type

from pydantic import BaseModel
from langchain_gigachat.tools.giga_tool import GigaBaseTool


__all__ = ["SqlTool"]
logger = logging.getLogger("agent.tools")


class SqlInput(BaseModel):
    query: str


class SqlOutput(BaseModel):
    result: str


class SqlTool(GigaBaseTool):
    name: str = "sql_tool"
    description: str = """
        Делает запрос к базе данных SQLite3
        """
    args_schema: Type[BaseModel] = SqlInput
    return_schema: Type[BaseModel] = SqlOutput

    def _run(self, query: str) -> SqlOutput:
        return asyncio.run(self._arun(query))

    async def _arun(self, query: str) -> SqlOutput:
        logger.info(f"Agents sql query: {query}")
        try:
            async with connect() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(query)
                    result = str(await cursor.fetchall())
                await conn.commit()

            logger.info(f"Got sql result: {result}")
        except Exception as e:
            logger.error(f"Exception in the SQLTool's runtime: {e}")

            result = str(e)

        return SqlOutput(result=result)
