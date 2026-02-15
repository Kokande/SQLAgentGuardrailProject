import os

from typing import Type, ClassVar


class AppConfig:
    log_level: ClassVar[str] = "DEBUG"
    bot_token: ClassVar[str] = ""
    _loaded: ClassVar[bool] = False

    @classmethod
    def load(cls):
        if cls._loaded:
            return

        cls.log_level = os.getenv("LOG_LEVEL", cls.log_level)
        cls.bot_token = os.getenv("TELEGRAM_API_KEY", cls.bot_token)
        cls._loaded = True

        cls.validate()

    @classmethod
    def validate(cls):
        if not cls.bot_token:
            raise Exception(
                "Could not retrieve TELEGRAM_API_KEY from environment variables"
            )


def init_cfg() -> Type[AppConfig]:
    """
    Initializes app configuration variables
    :return: AppConfig
    """
    AppConfig.load()

    return AppConfig
