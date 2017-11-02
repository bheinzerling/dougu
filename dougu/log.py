import logging


def get_logger(file=None, fmt=None, datefmt=None):
    if not fmt:
        fmt = '%(asctime)s| %(message)s'
    if not datefmt:
        datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    log = logging.getLogger(__name__)
    logging.root.handlers[0].formatter = formatter
    if file:
        add_log_filehandler(log, file)
    return log


def add_log_filehandler(log, file):
    fhandler = logging.FileHandler(file)
    log.addHandler(fhandler)
