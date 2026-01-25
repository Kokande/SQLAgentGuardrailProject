import config
from bot import bot
from db.core import init_db
from agent.SQLAgent import SafeAgent, init_agent

import asyncio
import logging
from typing import ClassVar, Type

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher


def init_env() -> None:
    load_dotenv("configs/service.properties")
    load_dotenv("configs/secret.properties")


class App:
    """
    Contains main App objects
    """

    _initialized: ClassVar[bool] = False
    bot: ClassVar[Bot | None] = None
    dispatcher: ClassVar[Dispatcher | None] = None
    cfg: ClassVar[Type[config.AppConfig] | None] = None
    agent: ClassVar[SafeAgent | None] = None

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return

        cls.cfg = config.init_cfg()
        cls.bot = bot.init_bot(cls.cfg)
        cls.agent = init_agent()
        cls.dispatcher = bot.init_dispatcher(cls.agent)


async def main() -> None:
    init_env()
    await init_db()

    App.initialize()
    logging.basicConfig(
        level=App.cfg.log_level,
        format="[%(asctime)s] (%(levelname)s) %(name)s - %(message)s",
    )

    await App.dispatcher.start_polling(App.bot)


if __name__ == "__main__":
    asyncio.run(main())
