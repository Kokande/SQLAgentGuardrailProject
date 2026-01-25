import os
from typing import ClassVar


class LLMConfig:
    _loaded: ClassVar[bool] = False
    token: ClassVar[str] = ""
    base_url: ClassVar[str] = "https://gigachat.devices.sberbank.ru/api/v1"
    auth_url: ClassVar[str] = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    model: ClassVar[str] = "GIGACHAT-2-MAX"
    scope: ClassVar[str] = "GIGACHAT_API_PERS"
    num_streams: ClassVar[int] = 1

    @classmethod
    def load(cls):
        if cls._loaded:
            return

        cls.token = os.getenv("LLM_API_KEY")
        cls.base_url = os.getenv("LLM_BASE_URL", cls.base_url)
        cls.auth_url = os.getenv("LLM_AUTH_URL", cls.auth_url)
        cls.model = os.getenv("LLM_MODEL", cls.model)
        cls.scope = os.getenv("LLM_SCOPE", cls.scope)
        cls.num_streams = int(os.getenv("LLM_STREAMS", cls.num_streams))
        cls._loaded = True

        cls.validate()

    @classmethod
    def validate(cls):
        if not cls.token:
            raise Exception("Could not retrieve LLM_API_KEY from environment variables")
