# Copyright 2026 Exploring-VLAs contributors. Licensed under Apache-2.0.
"""Reference-counted ROS resources. No ROS import or initialization on import."""
from __future__ import annotations

import threading
import os
import uuid

_lock = threading.RLock()
_sessions = {}


class _Session:
    def __init__(self, context, executor):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        self.context = context
        self.default_context = context is rclpy.get_default_context()
        self.owns_context = not context.ok()
        self.owns_executor = executor is None
        self.refs = 0
        self.error = None
        self.stop = threading.Event()
        self.thread = None
        self.node = None
        self.executor = executor
        try:
            if self.owns_context:
                rclpy.init(context=context)
            if executor is None:
                self.executor = SingleThreadedExecutor(context=context)
            self.node = rclpy.create_node("lerobot_ros2_" + uuid.uuid4().hex[:12], context=context)
            self.executor.add_node(self.node)
            if self.owns_executor:
                self.thread = threading.Thread(target=self._spin, daemon=True, name="lerobot_ros2_spin")
                self.thread.start()
        except BaseException:
            self.close()
            raise

    def _spin(self):
        try:
            while not self.stop.is_set() and self.context.ok():
                self.executor.spin_once(timeout_sec=0.05)
        except Exception as exc:
            if not self.stop.is_set():
                self.error = exc

    def check(self):
        if self.error is not None:
            raise RuntimeError("ROS executor callback failed") from self.error
        if not self.context.ok():
            raise RuntimeError("ROS context is stopped")

    def close(self):
        self.stop.set()
        if self.owns_executor and self.executor is not None:
            self.executor.wake()
            if self.thread is not None:
                self.thread.join()
        try:
            if self.node is not None:
                self.executor.remove_node(self.node)
                if self.owns_executor and self.executor is not None:
                    # Release the stopped spin generator/wait set before native node teardown.
                    self.executor.shutdown()
                self.node.destroy_node()
            elif self.owns_executor and self.executor is not None:
                self.executor.shutdown()
        finally:
            if self.owns_context:
                import rclpy
                if self.default_context and rclpy.get_default_context() is self.context:
                    rclpy.try_shutdown()
                elif self.context.ok():
                    self.context.shutdown()


class Lease:
    def __init__(self, key, session):
        self._key = key
        self._session = session
        self._closed = False

    @property
    def node(self):
        # The handle remains accessible after ROS stops so teardown can finish.
        if self._closed:
            raise RuntimeError("ROS lease is closed")
        return self._session.node

    def check(self):
        if self._closed:
            raise RuntimeError("ROS lease is closed")
        self._session.check()

    def close(self):
        with _lock:
            if self._closed:
                return
            self._closed = True
            self._session.refs -= 1
            if self._session.refs == 0:
                del _sessions[self._key]
                self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def acquire(*, context=None, executor=None):
    """Acquire the shared node. An external executor is spun by its caller."""
    import rclpy

    if rclpy.get_rmw_implementation_identifier() == "rmw_cyclonedds_cpp" and not os.environ.get("CYCLONEDDS_URI"):
        raise RuntimeError(
            "CycloneDDS requires an explicit CYCLONEDDS_URI. For Exploring-VLAs set "
            "CYCLONEDDS_URI=file:///absolute/path/to/Exploring-VLAs/linux-env/cyclonedds.xml "
            "(loopback transport). No ROS context was started."
        )

    if context is None:
        context = executor.context if executor is not None else rclpy.get_default_context()
    if executor is not None and executor.context is not context:
        raise ValueError("executor and context must refer to the same ROS context")
    key = id(context)
    with _lock:
        session = _sessions.get(key)
        if session is not None:
            if executor is not None and executor is not session.executor:
                raise ValueError("context already attached to a different executor")
            session.check()
        else:
            session = _Session(context, executor)
            _sessions[key] = session
        session.refs += 1
        return Lease(key, session)
