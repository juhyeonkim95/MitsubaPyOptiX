"""
A simple execution time logger implemented as a python decorator.
Available under the terms of the MIT license.
"""

import logging
import time

from functools import wraps

logger = logging.getLogger(__name__)


# Misc logger setup so a debug log statement gets printed on stdout.
logger.setLevel("DEBUG")
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
        logger.debug("{} ran in {}s".format(func.__name__, round(end - start, 2)))
        return result

    return wrapper


@timed
def slow_function():
    """This is a slow-running function used as an example."""
    print("running a slow function...", end="")
    time.sleep(3.2)
    print("done")


if __name__ == "__main__":
    slow_function()
