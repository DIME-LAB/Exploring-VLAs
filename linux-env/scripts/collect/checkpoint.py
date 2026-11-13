"""Validate the ordered receipt ledger before resuming an interrupted run."""
from contract import COLORS, parameters


def progress(receipts, seed):
    accepted = 0
    for trial, receipt in enumerate(receipts):
        expected = dict(parameters(seed + trial, COLORS[accepted % 3]), trial=trial)
        if receipt['params'] != expected:
            raise ValueError(f'Resume seed/colour/parameters differ at attempt {trial}')
        if receipt['accepted']:
            if not receipt.get('timing', {}).get('passed'):
                raise ValueError(f'Accepted attempt {trial} lacks a passing audit')
            accepted += 1
    return len(receipts), accepted
