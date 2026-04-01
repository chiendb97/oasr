"""
Simple timer for profiling.
"""


class Timer:
    """Simple timer for profiling."""

    def __init__(self, name: str = "", cuda_sync: bool = True):
        self.name = name
        self.cuda_sync = cuda_sync
        self._start_time = None
        self._elapsed = 0.0

    def __enter__(self):
        import time
        if self.cuda_sync:
            from oasr import synchronize
            synchronize()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        import time
        if self.cuda_sync:
            from oasr import synchronize
            synchronize()
        self._elapsed = time.perf_counter() - self._start_time
        if self.name:
            print(f"{self.name}: {self._elapsed * 1000:.2f} ms")

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return self._elapsed

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self._elapsed * 1000
