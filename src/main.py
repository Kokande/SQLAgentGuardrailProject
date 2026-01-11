import config
from bot import bot
from db.core import init_db
from agent.SQLAgent import SafeAgent, init_agent

import asyncio
import logging
from typing import ClassVar, Self

from aiogram import Bot, Dispatcher


class App:
    _instance: ClassVar[Self | None] = None
    bot: ClassVar[Bot | None] = None
    dispatcher: ClassVar[Dispatcher | None] = None
    cfg: ClassVar[config.AppConfig | None] = None
    agent: ClassVar[SafeAgent | None] = None

    def __new__(cls):
        if cls._instance:
            return cls._instance
        else:
            cls._instance = super(App, cls).__new__(cls)
            cls.cfg = config.init_cfg()
            cls.bot = bot.init_bot(cls.cfg)
            cls.agent = init_agent(cls.cfg)
            cls.dispatcher = bot.init_dispatcher(cls.agent)

            return cls._instance


async def main() -> None:
    await init_db()

    app = App()
    logging.basicConfig(level=app.cfg.log_level, format='[%(asctime)s] (%(levelname)s) %(name)s - %(message)s')

    await app.dispatcher.start_polling(app.bot)


if __name__ == "__main__":
    asyncio.run(main())
