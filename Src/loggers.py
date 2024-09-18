import logging
import os
from datetime import datetime

# Define the log file name based on the current date
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Create a logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Full path for the log file
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure the logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def get_logger(logger_name: str) -> logging.Logger:
    """
    Creates a logger instance with the given name.

    Args:
        logger_name (str): Name of the logger (typically __name__ of the calling module).

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(logger_name)
    return logger
