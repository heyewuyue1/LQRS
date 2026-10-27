import sys
import logging
import datetime
import os
from config import LoggingConfig

log_dir = f"/home/xxxxxxxx/LQRS_release/logs/{datetime.datetime.now().strftime('%Y-%m-%d')}"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%H-%M-%S')}.log")

formatter = logging.Formatter(fmt="%(asctime)s [%(name)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG) 

screen_handler = logging.StreamHandler(stream=sys.stdout)
screen_handler.setFormatter(formatter)
screen_handler.setLevel(logging.INFO)

def setup_custom_logger(name):
    custom_logger = logging.getLogger(name)
    if not custom_logger.handlers:
        custom_logger.setLevel(LoggingConfig.log_level)
        custom_logger.addHandler(file_handler)
        custom_logger.addHandler(screen_handler)
    return custom_logger
