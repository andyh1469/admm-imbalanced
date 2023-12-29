import logging
import os


def get_logger(timestamp):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Create and configure handlers, formatters, etc.
    file_handler = logging.FileHandler(f"logs/log {timestamp}.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
