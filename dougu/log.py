import logging
from contextlib import contextmanager


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


# https://gist.github.com/simon-weber/7853144
@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)
