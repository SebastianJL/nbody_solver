import time


class Timed(object):
    """Context manager for printing runtime of enclosed code."""

    def __init__(self, msg):
        self.msg = msg
        self._start = time.perf_counter()
        self._duration = None

    @property
    def duration(self):
        if self._duration is None:
            raise "Cannot read duration before context manager is left."
        else:
            return self._duration

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._duration = time.perf_counter() - self._start
        print(f'{self.msg}: {self._duration:g}s')