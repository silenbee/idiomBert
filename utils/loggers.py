import time
import logging

logger = logging.getLogger()


def init_logger():
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    time_ = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    file_handler = logging.FileHandler(time_ + '.log', encoding='UTF-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)