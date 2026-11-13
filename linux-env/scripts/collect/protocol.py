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
        self.epoch = None
        self.floor_ns = -1
        self.retirement_timeout = 15.
        self.stalled = None
        self.on_stall = None

    @contextmanager
    def own(self, label='motion'):
        if not self.lock.acquire(timeout=self.retirement_timeout):
            self.stalled = f'Stalled generation owner during {label}: {self.lock!r}; retirement exceeded {self.retirement_timeout}s; abort required'
            self.epoch = None
            if self.on_stall is not None:
                self.on_stall(OwnershipStalled(self.stalled), label)
            raise OwnershipStalled(self.stalled)
        try:
            if self.stalled:
                raise OwnershipStalled(self.stalled)
            yield
        finally:
            self.lock.release()

    def install(self, result):
        with self.own('install'):
            self.epoch = result['epoch']
            self.floor_ns = result['invalid_through_ns']

    def require(self, epoch, stamp=None):
        if self.epoch is None or epoch != self.epoch or (stamp is not None and stamp <= self.floor_ns):
            raise CancelledGeneration('Cancelled motion/observation generation')

    def cancel(self, native_cancel, retire):
        with self.own('cancel/reset'):
            # Failure leaves the local generation unusable. The collector's
            # common terminal handler owns setter-free emergency cancellation.
            self.epoch = None
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
