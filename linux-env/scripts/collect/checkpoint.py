"""Validate the ordered receipt ledger before resuming an interrupted run."""
from contract import COLORS, parameters
from recovery import regrasp_slot, receipt_error


def progress(receipts, seed, regrasp_only=False, row_loader=lambda r:None):
    accepted = 0
    for trial, receipt in enumerate(receipts):
        expected = dict(parameters(seed + trial,COLORS[accepted % 3],regrasp=regrasp_only or regrasp_slot(accepted)), trial=trial)
        if receipt['params'] != expected:
            raise ValueError(f'Resume seed/colour/parameters differ at attempt {trial}')
        if receipt['accepted']:
            if receipt_error(receipt,row_loader(receipt)):raise ValueError('Accepted recovery lacks complete measured miss evidence')
            if not receipt.get('timing', {}).get('passed'):
                raise ValueError(f'Accepted attempt {trial} lacks a passing audit')
            accepted += 1
    return len(receipts), accepted
