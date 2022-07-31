import logging
import os

from pythonjsonlogger import jsonlogger


def setup_logger():
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    fmt = (
        "%(levelname)s %(asctime)s "
        "[%(filename)s:%(funcName)s:%(lineno)d] "
        "%(message)s"
    )
    formatter = jsonlogger.JsonFormatter(fmt)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    log_level = os.environ.get("LOG_LEVEL", logging.INFO)

    logger.setLevel(log_level)
    return logger


logger = setup_logger()
