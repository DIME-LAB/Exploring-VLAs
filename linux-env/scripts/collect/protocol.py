"""Exactly-once commands and cancellation ownership for a held sim clock."""
import threading


class CancelledGeneration(RuntimeError):
    """Work belongs to a retired motion/reset generation."""


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

    def install(self, result):
        with self.lock:
            self.epoch = result['epoch']
            self.floor_ns = result['invalid_through_ns']

    def require(self, epoch, stamp=None):
        if self.epoch is None or epoch != self.epoch or (stamp is not None and stamp <= self.floor_ns):
            raise CancelledGeneration('Cancelled motion/observation generation')

    def cancel(self, native_cancel, retire):
        with self.lock:
            # Retire even if the native RPC fails: no new command may escape.
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
