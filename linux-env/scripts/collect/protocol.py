"""Exactly-once command application for a held simulation clock."""


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
