import config
from bot import bot
from db.core import init_db
from agent.SQLAgent import SafeAgent, init_agent

import asyncio
import logging

from aiogram import Bot, Dispatcher

__all__ = ["app"]


class App:
    bot: Bot
    dp: Dispatcher
    cfg: config.AppConfig
    agent: SafeAgent


app: App


def init_app() -> None:
    global app

    app = App

    app.cfg = config.init_cfg()
    app.agent = init_agent(app.cfg)
    app.bot = bot.init_bot(app.cfg)
    app.dp = bot.init_dispatcher(app.agent)

    logging.basicConfig(level=app.cfg.log_level, format='[%(asctime)s] (%(levelname)s) %(name)s - %(message)s')


async def main() -> None:
    init_app()
    await init_db()

    await app.dp.start_polling(app.bot)


if __name__ == "__main__":
    asyncio.run(main())
