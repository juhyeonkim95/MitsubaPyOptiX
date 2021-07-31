"""
A simple execution time logger implemented as a python decorator.
Available under the terms of the MIT license.
"""

import logging
import time

from functools import wraps
import contextlib
from datetime import timedelta
import errno
import os
import signal

logger = logging.getLogger(__name__)


# Misc logger setup so a debug log statement gets printed on stdout.
logger.setLevel("DEBUG")
logger.propagate = False
handler = logging.StreamHandler()
log_format = "%(asctime)s %(levelname)s -- %(message)s"
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)
logger.addHandler(handler)


def timed(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug("{} ran in {}ms".format(func.__name__, round((end - start) * 1000, 3)))
        return result

    return wrapper


@contextlib.contextmanager
def time_measure(ident, _logger=logger, show_started=True, time_unit='sec'):
    if show_started:
        _logger.info("%s Started" % ident)
    start_time = time.time()
    yield
    elapsed_time = str(timedelta(seconds=time.time() - start_time))
    _logger.info("%s Finished in %s " % (ident, elapsed_time))


@contextlib.contextmanager
def record_elapsed_time(ident, time_sequence, _logger=logger, debug=True):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    time_sequence.append(elapsed_time)
    if debug:
        _logger.debug("%s Finished in %s " % (ident, str(timedelta(seconds=elapsed_time))))
    else:
        _logger.info("%s Finished in %s " % (ident, str(timedelta(seconds=elapsed_time))))


DEFAULT_TIMEOUT_MESSAGE = os.strerror(errno.ETIME)


class timeout(contextlib.ContextDecorator):
    def __init__(self, seconds, *, timeout_message=DEFAULT_TIMEOUT_MESSAGE, suppress_timeout_errors=False):
        self.seconds = int(seconds)
        self.timeout_message = timeout_message
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        if self.seconds > 0:
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
        if self.suppress and exc_type is TimeoutError:
            return True
