import contextlib
import time
import logging

logger = logging.getLogger()

@contextlib.contextmanager
def time_measure(ident):
    tstart = time.time()
    yield
    elapsed = time.time() - tstart
    print(ident, elapsed)
    #logger.debug("{0}: {1} ms".format(ident, elapsed))


with time_measure('test_method:sum1'):
    a = 1
    b = 100
    sum1 = sum(range(a, b))
    sum2 = sum(range(a, b))
