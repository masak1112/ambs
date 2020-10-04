import logging
import time
from functools import wraps

def logDecorator(fn,verbose=False):
    @wraps(fn)
    def wrapper(*args,**kwargs):
        print("inside wrapper of log decorator function")
        logger = logging.getLogger(fn.__name__)
        # create a file handler
        handler = logging.FileHandler("log.log")
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        #create a console handler
        ch = logging.StreamHandler()
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(ch)
        logger.info("Logging 1")
        start = time.time()
        results = fn(*args,**kwargs)
        end = time.time()
        logger.info("{} ran in {}s".format(fn.__name__, round(end - start, 2)))
        return results
    return wrapper


#logger = logging.getLogger(__name__)
# def set_logger(verbose=False):
#     # Remove all handlers associated with the root logger object.
#     for handler in logging.root.handlers[:]:
#         logging.root.removeHandler(handler)
#     logger = logging.getLogger(__name__)
#     logger.propagate = False
#
#
#     if not logger.handlers:
#         logger.setLevel(logging.DEBUG if verbose else logging.INFO)
#         formatter = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#
#
#
#         #再创建一个handler，用于输出到控制台
#         console_handler = logging.StreamHandler()
#         console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
#         console_handler.setFormatter(formatter)
#         logger.handlers = []
#         logger.addHandler(console_handler)
#
#     return logger