import os

from dotenv import load_dotenv
from pydantic import BaseModel


class AppConfig(BaseModel):
    log_level: str
    bot_token: str
    gigachat_token: str


def init_cfg() -> AppConfig:
    load_dotenv("appconfig.properties")
    load_dotenv("secret.properties")

    return AppConfig(
        log_level=os.getenv("LOG_LEVEL"),
        bot_token=os.getenv("TELEGRAM_API_KEY"),
        gigachat_token=os.getenv("GIGACHAT_API_KEY")
    )