"""Explicit adapter modes and strict collector scene-request lifetimes."""
from dataclasses import dataclass


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
    def __init__(self, client, guard, terminal):
        self.client, self.guard, self.terminal = client, guard, terminal
        self.pending = {}
        self.failure = None

    def incomplete(self, label, reason):
        self.failure = f'Unresolved scene mutation ({label}): {reason}; restart this instance control stack'
        self.terminal(SceneMutationIncomplete(self.failure))
        raise SceneMutationIncomplete(self.failure)

    def require_idle(self):
        if self.failure:
            raise SceneMutationIncomplete(self.failure)
        for tracked in list(self.pending.values()):
            if not tracked.done():
                self.incomplete(tracked.label, 'another scene operation started before completion')
            tracked.result()  # Deliberately drain an already completed request.

    def call_async(self, request):
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

    def __getattr__(self, name):
        return getattr(self.client, name)
