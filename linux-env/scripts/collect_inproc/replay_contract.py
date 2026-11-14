"""Exact frame identities for physics-state capture and deferred rendering."""
import copy
import math
import shared
from tick_contract import digest, STEP_NS


def packet(snapshot, transforms, cameras, index):
    if not transforms or set(cameras) != {'wrist', 'top'}:
        raise ValueError('Every moving prim and both camera transforms are required')
    for matrix in [*transforms.values(), *cameras.values()]:
        if len(matrix) != 4 or any(len(r) != 4 for r in matrix):
            raise ValueError('Replay transforms must be 4x4 matrices')
        if not all(math.isfinite(v) for row in matrix for v in row):
            raise ValueError('Non-finite replay transform')
    value = dict(version=1, index=index, epoch=snapshot['epoch'],
        sim_ns=snapshot['sim_ns'], physics_time_s=snapshot['physics_time_s'],
        state=copy.deepcopy(snapshot['state']), bodies=copy.deepcopy(snapshot['bodies']),
        transforms=copy.deepcopy(transforms), cameras=copy.deepcopy(cameras))
    return dict(payload=value, sha256=digest(value))


def verify(value, previous=None, paths=None):
    p = value['payload']
    if digest(p) != value['sha256']:
        raise ValueError('Replay state digest differs')
    if paths is not None and set(p['transforms']) != set(paths):
        raise ValueError('Replay prim inventory differs')
    if previous is not None:
        old = previous['payload']
        if (p['epoch'] != old['epoch'] or p['index'] != old['index']+1 or
            abs(p['sim_ns']-old['sim_ns']-STEP_NS) > 10):
            raise ValueError('Replay frame gap or epoch mismatch')
    return p
