import logging
import const
from src.utils import Time

def clear_logging_file():
    with open(const.LOGGING_FILE, "w+") as out:
        out.write("")

def setup_logger():
    logger = logging.getLogger("custom_logger")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(const.LOGGING_FILE, encoding="ascii")
    formatter = logging.Formatter("%(message)s")  # Custom formatting
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

clear_logging_file()
loggingCurrentTime = Time()  # Will be updated by the simulator class
currentTimeStr = loggingCurrentTime.to_str()
logger = setup_logger()

def Log(description: str, *args) -> None:
    global logger, loggingCurrentTime
    log_entry = f"{loggingCurrentTime.to_str()}\t{description}\t" + "\t".join(map(str, args))
    logger.info(log_entry)

def update_logging_file():
    global logger
    for handler in logger.handlers:
        handler.flush()

def update_logging_time(time: Time):
    global loggingCurrentTime, currentTimeStr
    loggingCurrentTime = time.copy()
    currentTimeStr = loggingCurrentTime.to_str()

def get_logging_time():
    global loggingCurrentTime
    return loggingCurrentTime.copy()

def get_logging_time_no_copy():
    global loggingCurrentTime
    return loggingCurrentTime

def close_logging_file():
    global logger
    for handler in logger.handlers:
        handler.close()
    logging.shutdown()
