"""Track scene service requests until their remote completion is known."""
from collections import deque
from concurrent.futures import Future
import threading
import time


class GuiSceneFuture(Future):
    """Keep rclpy's nonblocking result() behavior for queued GUI requests."""
    def result(self, timeout=None):
        if not self.done() and timeout is None:
            return None
        return super().result(timeout=timeout)

    def cancel(self):
        return False  # Cancelling a local future cannot recall a remote mutation.


class GuiSceneQueue:
    """Serialize GUI transactions and their RPCs without blocking ROS callbacks.

    A slow remote response suspends scene updates, not the GUI executor. The
    worker retains it through completion; later updates cannot overtake it.
    """
    def __init__(self, client, warn):
        self.client, self.warn = client, warn
        self.condition = threading.Condition(threading.RLock())
        self.requests, self.jobs = deque(), deque()
        self.active, self.worker = None, None

    def has_work(self):
        with self.condition:
            return bool(self.active or self.requests or self.jobs or self.worker)

    def call_async(self, request):
        future = GuiSceneFuture()
        with self.condition:
            self.requests.append((request, future))
            self._start_next()
        return future

    def _start_next(self):
        if self.active is not None or not self.requests:
            return
        request, future = self.requests.popleft()
        self.active = future
        try:
            raw = self.client.call_async(request)
        except Exception as exc:
            self._finished(future, error=exc)
        else:
            def done(response):
                try:
                    value = response.result()
                except Exception as exc:
                    self._finished(future, error=exc)
                else:
                    self._finished(future, value=value)
            raw.add_done_callback(done)

    def _finished(self, future, value=None, error=None):
        with self.condition:
            if self.active is not future:
                return
            self.active = None
            self._start_next()
            self.condition.notify_all()
        if error is None:
            future.set_result(value)
        else:
            future.set_exception(error)

    def _wait_idle(self):
        started, warned = time.monotonic(), False
        with self.condition:
            while self.active is not None or self.requests:
                self.condition.wait(timeout=.25)
                if not warned and time.monotonic()-started > 5:
                    warned = True
                    self.warn('GUI scene response pending; later scene updates wait for completion')

    def submit_update(self, callback):
        future = GuiSceneFuture()
        with self.condition:
            self.jobs.append((callback, future))
            if self.worker is None:
                self.worker = threading.Thread(target=self._run, daemon=True)
                self.worker.start()
        return future

    def _run(self):
        while True:
            with self.condition:
                if not self.jobs:
                    self.worker = None
                    return
                callback, future = self.jobs.popleft()
            self._wait_idle()
            try:
                value = callback()
            except Exception as exc:
                self.warn('GUI scene update failed: '+repr(exc))
                future.set_exception(exc)
            else:
                future.set_result(value)
            finally:
                self._wait_idle()


class SceneMutationIncomplete(RuntimeError):
    """A submitted scene request has no conclusive completion; stop the run."""


class SceneFuture:
    def __init__(self, future, owner, label):
        self.future, self.owner, self.label = future, owner, label

    def done(self):
        return self.future.done()

    def result(self):
        if not self.future.done():
            self.owner.incomplete(self.label, 'wait ended before service completion')
        try:
            response = self.future.result()
        except Exception as exc:
            self.owner.incomplete(self.label, 'service completion failed: '+repr(exc))
        if response is None:
            self.owner.incomplete(self.label, 'service returned no conclusive response')
        self.owner.pending.pop(id(self), None)
        return response

    def __getattr__(self, name):
        return getattr(self.future, name)


class SceneMutationClient:
    """One entrance for attach, world ADD/REMOVE and ACM mutation lifetimes.

    An expired client wait never cancels the server's request. Keep unresolved
    futures for diagnosis and refuse further mutation after the terminal hook.
    """
    def __init__(self, client, guard, terminal, collector=lambda:True, warn=lambda message:None):
        self.client, self.guard, self.terminal = client, guard, terminal
        self.collector, self.warn = collector, warn
        self.gui = GuiSceneQueue(client, warn)
        self.pending = {}
        self.failure = None

    def incomplete(self, label, reason):
        if not self.collector():
            message = f'GUI scene request incomplete ({label}): {reason}'
            self.warn(message)
            raise SceneMutationIncomplete(message)
        self.failure = f'Unresolved scene mutation ({label}): {reason}; restart this instance control stack'
        self.terminal(SceneMutationIncomplete(self.failure))
        raise SceneMutationIncomplete(self.failure)

    def require_idle(self):
        if not self.collector():
            return not self.gui.has_work()
        if self.gui.has_work():
            self.incomplete('GUI scene queue', 'collector activated before GUI work completed')
        if self.failure:
            raise SceneMutationIncomplete(self.failure)
        for tracked in list(self.pending.values()):
            if not tracked.done():
                self.incomplete(tracked.label, 'another scene operation started before completion')
            tracked.result()  # Deliberately drain an already completed request.

    def call_async(self, request):
        if not self.collector():
            return self.gui.call_async(request)
        with self.guard():
            if self.failure:
                raise SceneMutationIncomplete(self.failure)
            self.require_idle()
            scene = request.scene
            world = [f'{obj.id}:{obj.operation}' for obj in scene.world.collision_objects]
            attached = [f'{obj.object.id}:{obj.object.operation}' for obj in scene.robot_state.attached_collision_objects]
            acm = bool(scene.allowed_collision_matrix.entry_names or scene.allowed_collision_matrix.default_entry_names)
            label = f'world={world}, attached={attached}, acm={acm}'
            try:
                future = self.client.call_async(request)
            except Exception as exc:
                self.incomplete(label, 'submission completion unknown: '+repr(exc))
            tracked = SceneFuture(future, self, label)
            self.pending[id(tracked)] = tracked
            return tracked

    def submit_gui_update(self, callback):
        if self.collector():
            raise RuntimeError('Collector scene updates must use their generation guard')
        return self.gui.submit_update(callback)

    def __getattr__(self, name):
        return getattr(self.client, name)
