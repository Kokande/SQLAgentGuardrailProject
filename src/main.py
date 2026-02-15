import config
from bot import bot
from db.core import init_db
from agent import init_agent, SafeAgent
from utils.log_cfg import LOGGING_CONFIG

import asyncio
import logging
import logging.config
from typing import ClassVar, Type

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher


logger = logging.getLogger("service")


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

    logging.config.dictConfig(LOGGING_CONFIG)

    await init_db()
    logger.info("DB initialized")

    App.initialize()

    logger.info("App initialized")
    logging.basicConfig(level=App.cfg.log_level)

    await App.dispatcher.start_polling(App.bot)


if __name__ == "__main__":
    asyncio.run(main())
