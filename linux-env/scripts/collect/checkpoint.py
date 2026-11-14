"""Validate the ordered receipt ledger before resuming an interrupted run."""
from contract import COLORS, parameters
from recovery import regrasp_slot, receipt_error


def recovery_mode(regrasp_only=False,normal_only=False):
    if regrasp_only and normal_only:raise ValueError('Normal-only and regrasp-only are mutually exclusive')
    return 'normal_only' if normal_only else 'regrasp_only' if regrasp_only else 'mixed'


def require_recovery_mode(previous,regrasp_only=False,normal_only=False):
    mode=recovery_mode(regrasp_only,normal_only)
    if previous is not None:
        old=recovery_mode(previous.get('regrasp_only',False),previous.get('normal_only',False))
        if previous.get('recovery_mode',old)!=old:raise ValueError('Stored recovery mode is inconsistent')
        if old!=mode:raise ValueError('Resume recovery mode differs')
    return mode


def episode_regrasp(accepted,regrasp_only=False,normal_only=False):
    mode=recovery_mode(regrasp_only,normal_only)
    return mode=='regrasp_only' or (mode=='mixed' and regrasp_slot(accepted))


def progress(receipts, seed, regrasp_only=False, row_loader=lambda r:None,normal_only=False):
    accepted = 0
    for trial, receipt in enumerate(receipts):
        expected = dict(parameters(seed + trial,COLORS[accepted % 3],regrasp=episode_regrasp(accepted,regrasp_only,normal_only)), trial=trial)
        if receipt['params'] != expected:
            raise ValueError(f'Resume seed/colour/parameters differ at attempt {trial}')
        if receipt['accepted']:
            if receipt_error(receipt,row_loader(receipt)):raise ValueError('Accepted recovery lacks complete measured miss evidence')
            if not receipt.get('timing', {}).get('passed'):
                raise ValueError(f'Accepted attempt {trial} lacks a passing audit')
            accepted += 1
    return len(receipts), accepted
