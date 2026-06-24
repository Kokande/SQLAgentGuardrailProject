import os
import logging

default_logging_setup = {"handlers": ["console"], "level": "DEBUG", "propagate": False}

loggers = {
    "service": default_logging_setup,
    "agent": default_logging_setup,
    "agent.tools": default_logging_setup,
    "database": default_logging_setup,
    "aiogram": default_logging_setup,
    "agent.guardrail": default_logging_setup,
}


def setup_file_logging(log_dir: str = "logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    plain = logging.Formatter("%(message)s")

    for logger_name, filename in (
        ("agent.communication", "agent_communication.jsonl"),
        ("agent.guardrail_reactions", "guardrail.jsonl"),
    ):
        lg = logging.getLogger(logger_name)
        lg.setLevel(logging.INFO)
        lg.propagate = False
        fh = logging.FileHandler(os.path.join(log_dir, filename), encoding="utf-8")
        fh.setFormatter(plain)
        lg.addHandler(fh)


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "[%(asctime)s] (%(levelname)s)\t %(name)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": loggers,
}
