"""Exactly-once commands and cancellation ownership for a held sim clock."""
import threading
from contextlib import contextmanager


class CancelledGeneration(RuntimeError):
    """Work belongs to a retired motion/reset generation."""


class OwnershipStalled(TimeoutError):
    """A transport/scene owner did not retire; the run must abort."""


class StepGeneration:
    """Serialize capture, dispatch and reset; retain the observation's identity.

    The lock covers computation through ACK, recording and completion delivery.
    A reset acquires the same lock before touching the scene. Re-entrant
    cancellation is permitted, but invalidates the token checked before dispatch.
    """
    def __init__(self):
        self.lock = threading.RLock()
        self._identity_lock = threading.RLock()
        self._identity = (None, -1)
        self.last_epoch = -1
        self.retirement_timeout = 15.
        self.stalled = None
        self.on_stall = None

    @property
    def identity(self):
        return self._identity

    @property
    def epoch(self):
        return self._identity[0]

    @property
    def floor_ns(self):
        return self._identity[1]

    @contextmanager
    def observe_identity(self):
        # Separate from long motion ownership: sensor callbacks must still run
        # while capture owns a generation and waits for its images.
        with self._identity_lock:
            yield self._identity

    def _invalidate(self):
        with self._identity_lock:self._identity=(None,self._identity[1])

    @contextmanager
    def own(self, label='motion'):
        if not self.lock.acquire(timeout=self.retirement_timeout):
            self.stalled = f'Stalled generation owner during {label}: {self.lock!r}; retirement exceeded {self.retirement_timeout}s; abort required'
            self._invalidate()
            if self.on_stall is not None:
                self.on_stall(OwnershipStalled(self.stalled), label)
            raise OwnershipStalled(self.stalled)
        try:
            if self.stalled:
                raise OwnershipStalled(self.stalled)
            yield
        finally:
            self.lock.release()

    def poison(self, error):
        if self.stalled is None:
            self.stalled = str(error)
        self._invalidate()

    def fail(self, error, requester):
        """An unresolved external mutation is terminal, like owner retirement."""
        self.poison(error)
        if self.on_stall is not None:
            self.on_stall(OwnershipStalled(self.stalled), requester)

    def install(self, result):
        with self.own('install'):
            newest=max(self.last_epoch,self.epoch if self.epoch is not None else -1)
            if result['epoch']<=newest or result['invalid_through_ns']<self.floor_ns:
                raise CancelledGeneration('Cancellation epoch or native clock did not advance monotonically')
            self.last_epoch=result['epoch']
            with self._identity_lock:
                self._identity = (result['epoch'],result['invalid_through_ns'])

    def require(self, epoch, stamp=None):
        active,floor=self.identity
        if self.stalled or active is None or epoch != active or (stamp is not None and stamp <= floor):
            raise CancelledGeneration('Cancelled motion/observation generation')

    def cancel(self, native_cancel, retire):
        with self.own('cancel/reset'):
            # Failure leaves the local generation unusable. The collector's
            # common terminal handler owns setter-free emergency cancellation.
            if self.epoch is not None:self.last_epoch=max(self.last_epoch,self.epoch)
            self._invalidate()
            retire()
            result = native_cancel()
            self.install(result)
            return result


class CommandLedger:
    def __init__(self):
        self.last = None
        self.ack = None

    def process(self, payload, apply, advance):
        if self.last is not None:
            if payload['observation_stamp_ns'] < self.last['observation_stamp_ns']:
                raise ValueError('Stale command observation')
            if payload['observation_stamp_ns'] == self.last['observation_stamp_ns']:
                if payload != self.last:
                    raise ValueError('Conflicting command for an already-applied tick')
                return self.ack
        ack = dict(observation_stamp_ns=payload['observation_stamp_ns'], epoch=payload['epoch'],
                   **apply(payload['command'], payload['observation_stamp_ns']))
        if payload['advance']:
            advance()
        self.last = dict(payload, command=list(payload['command']))
        self.ack = dict(ack, applied_command=list(ack['applied_command']))
        return self.ack
