default_logging_setup = {"handlers": ["console"], "level": "DEBUG", "propagate": False}

loggers = {
    "service": default_logging_setup,
    "agent": default_logging_setup,
    "agent.tools": default_logging_setup,
    "database": default_logging_setup,
    "aiogram": default_logging_setup,
    "agent.guardrail": default_logging_setup,
}

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
