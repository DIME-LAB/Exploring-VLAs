# Copyright 2026 Exploring-VLAs contributors. Licensed under Apache-2.0.
"""Track discrete FJT moves without putting futures on the streaming path."""
import threading
import time


def wait_future(future, deadline):
    event = threading.Event()
    future.add_done_callback(lambda _: event.set())
    if not event.wait(max(0, deadline - time.monotonic())):
        raise TimeoutError("Controller response timed out")
    return future.result()


class Move:
    def __init__(self, futures):
        self.futures = futures
        self._lock = threading.RLock()
        self._cancelled = False
        self._handles = []
        for future in futures:
            future.add_done_callback(self._accepted)

    def _accepted(self, future):
        try:
            handle = future.result()
        except Exception:
            return  # wait() reports the original error
        with self._lock:
            self._handles.append(handle)
            if self._cancelled and handle.accepted:
                handle.cancel_goal_async()

    def cancel(self):
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            for handle in self._handles:
                if handle.accepted:
                    handle.cancel_goal_async()

    def wait(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        try:
            handles = [wait_future(future, deadline) for future in self.futures]
            if not all(handle.accepted for handle in handles):
                raise RuntimeError("Controller declined a move")
            results = [wait_future(handle.get_result_async(), deadline) for handle in handles]
            for result in results:
                if result.status != 4 or result.result.error_code != 0:
                    raise RuntimeError(f"Controller move failed: status={result.status}, code={result.result.error_code}")
            return results
        except BaseException:
            self.cancel()
            raise
