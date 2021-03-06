import yaml
import os
import logging
import logging.config
import time


def read_config(config_path):
    """
    Args:
        config_path: Path to the config file
    Returns:
        content: Returns content of config file
    """
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)
    return content


def get_logger(logs_dir, logger_name):
    """
    Args:
        logs_dir: Directory to save logs
        logger_name: Logger name to create logger object
    Returns:
        logger: Returns logger object for logging
    """
    # create logger with 'spam_application'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # create log file with log dir
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, logger_name + ".log")
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_unique_name():
    """
    Returns:
        Returns unique name with current time stamp
    """
    return time.strftime('%Y_%m_%d_%H_%M_%S')
    