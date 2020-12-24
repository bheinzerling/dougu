import logging


def get_formatter(fmt=None, datefmt=None):
    if not fmt:
        fmt = '%(asctime)s| %(message)s'
    if not datefmt:
        datefmt = "%Y-%m-%d %H:%M:%S"
    return logging.Formatter(fmt, datefmt=datefmt)


def get_logger(file=None, fmt=None, datefmt=None):
    log = logging.getLogger(__name__)
    formatter = get_formatter(fmt, datefmt)
    if not logging.root.handlers:
        logging.root.addHandler(logging.StreamHandler())
    logging.root.handlers[0].formatter = formatter
    if file:
        add_log_filehandler(log, file)
    return log


def add_log_filehandler(log, file):
    fhandler = logging.FileHandler(file)
    log.addHandler(fhandler)


class WithLog:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = get_logger().info
