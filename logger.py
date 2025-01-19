import logging
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure the root logger for general logs
logging.basicConfig(
    filename='logs/general.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_logger(name, log_file=None):
    """
    Returns a logger instance with an optional file handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler for model-specific logs
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
