"""Versioned fixed-base target rule for frozen collector-cut evaluation.

The bf7336d physics cut predates collect.contract.visible_target. Keep this
pure selector independent of the live collector, and test parity against it.
"""
import math
from pose_math import rotate

RULE = 'nearest_base_xy_min_y_5mm_v1'
TIE_M = .005


def select_target(snapshot, color, refuse_ambiguous=False):
    if color not in ('red', 'green', 'blue'):
        raise ValueError('Unknown task colour')
    base = snapshot['base_world_pose']
    origin, q = base['position'], base['orientation']
    if len(origin) != 3 or len(q) != 4 or not all(math.isfinite(x) for x in origin+q):
        raise ValueError('Invalid fixed base pose')
    inverse = [q[0]]+[-x for x in q[1:]]
    candidates = []
    for index in range(2):
        name = f'{color}_2x3_{index}'
        world = snapshot['bodies'][name]['position']
        if len(world) != 3 or not all(math.isfinite(x) for x in world):
            raise ValueError('Invalid target centre')
        position = rotate(inverse, [x-y for x, y in zip(world, origin)])
        candidates.append(dict(object=name, base_position_m=position,
                               distance_xy_m=math.hypot(*position[:2])))
    ranked = sorted(candidates, key=lambda c: (c['distance_xy_m'], c['object']))
    margin = ranked[1]['distance_xy_m']-ranked[0]['distance_xy_m']
    ambiguous = margin <= TIE_M
    chosen = min(candidates, key=lambda c: (c['base_position_m'][1], c['object'])) if ambiguous else ranked[0]
    result = dict(rule=RULE, frame='/World/base',
                  base_world_pose=dict(position=list(origin), orientation=list(q)),
                  candidates=candidates, nearest_distance_object=ranked[0]['object'],
                  object=chosen['object'], margin_m=margin, tie_threshold_m=TIE_M,
                  ambiguous=ambiguous, chosen_by='smaller_base_y_then_object_name' if ambiguous else 'nearest_xy',
                  ambiguous_policy='refuse' if refuse_ambiguous else 'smaller_base_y_then_object_name')
    if refuse_ambiguous and ambiguous:
        raise ValueError('Visible target distance margin is <=5 mm: '+str(result))
    return result
