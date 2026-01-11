import os

from dotenv import load_dotenv
from pydantic import BaseModel


class LLMConfig(BaseModel):
    token: str
    base_url: str
    auth_url: str
    model: str
    scope: str
    num_streams: int


class AppConfig(BaseModel):
    log_level: str
    bot_token: str
    llm: LLMConfig


def init_cfg() -> AppConfig:
    load_dotenv("configs/service.properties")
    load_dotenv("configs/secret.properties")

    return AppConfig(
        log_level=os.getenv("LOG_LEVEL"),
        bot_token=os.getenv("TELEGRAM_API_KEY"),
        llm=LLMConfig(
            token=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            auth_url=os.getenv("LLM_AUTH_URL"),
            model=os.getenv("LLM_MODEL"),
            scope=os.getenv("LLM_SCOPE"),
            num_streams=os.getenv("LLM_STREAMS")
        )
    )