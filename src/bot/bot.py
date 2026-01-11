import config
from agent.SQLAgent import SafeAgent
from db.core import create_session, change_guardrail, get_guardrail

import logging
from typing import Callable, Any, Dict, Awaitable

from aiogram import BaseMiddleware
from aiogram.filters import Command
from aiogram.dispatcher.dispatcher import Dispatcher, Bot
from aiogram.types import (
    Message, CallbackQuery,
    ReplyKeyboardMarkup, KeyboardButton,
    InlineKeyboardMarkup, InlineKeyboardButton
)


def init_bot(cfg: config.AppConfig) -> Bot:
    return Bot(cfg.bot_token)


def init_dispatcher(agent: SafeAgent) -> Dispatcher:
    dp = Dispatcher()

    main_keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="Обновить сессию"),
                KeyboardButton(text="Сменить метод защиты")
            ]
        ]
    )
    guardrail_keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Отключить", callback_data="disable"),
                InlineKeyboardButton(text="RE", callback_data="regular_expressions"),
                InlineKeyboardButton(text="ML", callback_data="machine_learning"),
                InlineKeyboardButton(text="LLM", callback_data="large_language_model"),
                InlineKeyboardButton(text="Гибридный", callback_data="hybrid")
            ]
        ]
    )

    @dp.message(Command("start"))
    async def command_start_handler(message: Message) -> None:
        await create_session(message.from_user.id)
        await message.answer(
            "Здравствуйте! Я ИИ-агент, позволяющий проводить анализ по прекрасным пушистым посетителям нашего спа-салона. ",
            reply_markup=main_keyboard
        )

    @dp.message(lambda message: message.text == "Обновить сессию")
    async def refresh_session_handler(message: Message) -> None:
        await create_session(message.from_user.id)
        await message.answer(
            "Моя голова чиста и я готов продолжить работу!",
            reply_markup=main_keyboard
        )

    @dp.message(lambda message: message.text == "Сменить метод защиты")
    async def change_guardrail_type(message: Message) -> None:
        await message.answer(
            "Выберите новый метод защиты",
            reply_markup=guardrail_keyboard
        )

    @dp.callback_query(lambda c: c.data in [
                                     "disable",
                                     "regular_expressions",
                                     "machine_learning",
                                     "large_language_model",
                                     "hybrid"
                                 ]
                       )
    async def gr_change(callback_query: CallbackQuery):
        await change_guardrail(callback_query.from_user.id, callback_query.data)
        await callback_query.answer(f"Метод защиты успешно изменён на {callback_query.data}. Можем продолжить работу!")

    @dp.message()
    async def message(message: Message) -> None:
        answer = await agent.ainvoke(message.text, guardrail_type=await get_guardrail(message.from_user.id))
        await message.answer(answer, reply_markup=main_keyboard)

    class LoggingMiddleware(BaseMiddleware):
        logger = logging.getLogger(__name__)

        async def __call__(
                self,
                handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
                event: Message,
                data: Dict[str, Any]
        ) -> Any:
            self.logger.debug(f"Got msg from user {event.from_user.id}: {event.text}")
            return await handler(event, data)

    dp.message.middleware(LoggingMiddleware())

    return dp
