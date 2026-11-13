"""Own complete scene transactions and track remote completion lifetimes."""
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from functools import wraps
import threading
import time


@dataclass(frozen=True)
class SceneExecution:
    """Construction-time policy; collection requires the complete hook bundle."""
    mode: str = 'gui'
    effect: object = None
    scene_failure: object = None
    callback_failure: object = None

    def __post_init__(self):
        hooks = (self.effect, self.scene_failure, self.callback_failure)
        if self.mode not in ('gui', 'collector'):
            raise ValueError('Scene mode must be gui or collector')
        if self.mode == 'collector' and not all(callable(h) for h in hooks):
            raise ValueError('Collector requires effect, scene failure and callback failure hooks')
        if self.mode == 'gui' and any(h is not None for h in hooks):
            raise ValueError('GUI scene policy cannot contain collector hooks')

    @property
    def collector(self):
        return self.mode == 'collector'


class GuiSceneNeedsResync(RuntimeError):
    """The GUI remains usable, but scene mutations need authoritative repair."""


def scene_transaction(method):
    """Synchronous GUI operations share the same owner as queued startup work."""
    @wraps(method)
    def run(self, *args, **kwargs):
        client = getattr(self, '_apply_scene_client', None)
        if client is not None and not client.collector:
            return client.gui.invoke(lambda: method(self, *args, **kwargs))
        return method(self, *args, **kwargs)
    return run


class GuiSceneFuture(Future):
    def result(self, timeout=None):
        if not self.done() and timeout is None:
            return None
        return super().result(timeout=timeout)

    def cancel(self):
        return False  # A local cancellation cannot recall a remote mutation.


class GuiSceneQueue:
    """One owner for the complete GUI transaction, including every ACM read/write.

    A normal wait expiry marks the scene for repair and continues waiting for
    a bounded grace period. A late response inside that period resumes the
    transaction's remaining steps. Otherwise later jobs first drain outstanding
    responses and repair from MoveIt; a silent server refuses those jobs while
    the ROS executor and GUI remain alive.
    """
    def __init__(self, client, warn, repair=None):
        self.client, self.warn, self.repair = client, warn, repair
        self.condition = threading.Condition(threading.RLock())
        self.jobs, self.pending = deque(), []
        self.worker = None
        self.needs_resync = False
        self.grace_seconds = 30.
        self.wait_scale = 1.  # Tests can exercise the same deadlines quickly.

    def owns_current(self):
        return threading.current_thread() is self.worker

    def has_work(self):
        with self.condition:
            return bool(self.jobs or self.worker or any(not f.done() for f in self.pending))

    def _require_owner(self):
        if not self.owns_current():
            raise RuntimeError('GUI scene RPC requires ownership of its complete transaction')

    def call_async(self, request):
        self._require_owner()
        future = self.client.call_async(request)
        self.pending.append(future)
        return future

    def wait(self, future, timeout):
        self._require_owner()
        if future not in self.pending:
            self.pending.append(future)
        deadline = time.monotonic() + timeout*self.wait_scale
        while not future.done() and time.monotonic() < deadline:
            threading.Event().wait(.005)
        if not future.done():
            self.needs_resync = True
            self.warn('GUI scene needs resync: response exceeded its wait; retaining transaction ownership')
            deadline = time.monotonic() + self.grace_seconds
            while not future.done() and time.monotonic() < deadline:
                threading.Event().wait(.005)
        if not future.done():
            raise GuiSceneNeedsResync('GUI scene response still pending; repair required before another transaction')
        try:
            value = future.result()
            if value is None:
                raise GuiSceneNeedsResync('GUI scene service returned no response')
            return value
        except Exception:
            self.needs_resync = True
            raise

    def _ready(self):
        while self.pending:
            future = self.pending[0]
            try:
                self.wait(future, 0.)
            except Exception as exc:
                if not future.done():
                    raise  # A still-running remote request cannot be overtaken.
                self.needs_resync = True
                self.warn('GUI scene completed with an error; rereading MoveIt: '+repr(exc))
            self.pending.pop(0)
        if self.needs_resync:
            if self.repair is None:
                raise GuiSceneNeedsResync('GUI scene needs resync; no authoritative repair configured')
            self.repair()
            self.pending.clear()
            self.needs_resync = False
            self.warn('GUI scene resync verified from MoveIt')

    def submit_update(self, callback):
        future = GuiSceneFuture()
        with self.condition:
            self.jobs.append((callback, future))
            if self.worker is None:
                self.worker = threading.Thread(target=self._run, daemon=True)
                self.worker.start()
        return future

    def invoke(self, callback):
        if self.owns_current():
            return callback()  # Nested helpers retain the whole transaction.
        return self.submit_update(callback).result(timeout=120.)

    def _run(self):
        while True:
            with self.condition:
                if not self.jobs:
                    self.worker = None
                    return
                callback, future = self.jobs.popleft()
            try:
                self._ready()
                value = callback()
                self._ready()  # Also repair a late response before reporting success.
            except Exception as exc:
                self.needs_resync = True
                self.warn('GUI scene transaction needs resync: '+repr(exc))
                future.set_exception(exc)
            else:
                future.set_result(value)


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
    def __init__(self, client, guard, terminal, collector=True, warn=lambda message:None, repair=None):
        self.client, self.guard, self.terminal = client, guard, terminal
        if type(collector) is not bool:
            raise TypeError('Scene mode must be an explicit bool')
        self._collector, self.warn = collector, warn
        self.gui = GuiSceneQueue(client, warn, repair)
        self.pending = {}
        self.failure = None

    @property
    def collector(self):
        return self._collector

    def incomplete(self, label, reason):
        if not self.collector:
            message = f'GUI scene request incomplete ({label}): {reason}'
            self.warn(message)
            raise SceneMutationIncomplete(message)
        self.failure = f'Unresolved scene mutation ({label}): {reason}; restart this instance control stack'
        self.terminal(SceneMutationIncomplete(self.failure))
        raise SceneMutationIncomplete(self.failure)

    def require_idle(self):
        if not self.collector:
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
        if not self.collector:
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
        if self.collector:
            raise RuntimeError('Collector scene updates must use their generation guard')
        return self.gui.submit_update(callback)

    def __getattr__(self, name):
        return getattr(self.client, name)
