"""Whole-block containment in the cup frame, using synchronized physical poses."""
import math
from pose_math import rotate
from cup_geometry import CUP,PROFILE,WALL_REST_TOLERANCE_M,wall_checks
from tick_contract import snapshot_contracts

# Body origin and primitives are cup-base coordinates, from the shared asset.
PHYSICAL_FLOOR_M = CUP['floor_top_m']
FLOOR_REST_TOLERANCE_M = PROFILE['floor_rest_tolerance_m']
INTERIOR_FLOOR_M = PHYSICAL_FLOOR_M - FLOOR_REST_TOLERANCE_M
INTERIOR_RIM_M = CUP['rim_m'] - PROFILE['rim_clearance_m']
SETTLED_TICKS = 15


def initial_arm(snapshots, target):
    """Verify measured home/stability, then the transport resets initial velocities.

    PhysX joint velocity samples can include constraint impulses even when
    sampled positions are stable; retain those samples rather than relabel them.
    """
    contract=snapshot_contracts(snapshots)
    if not contract['passed']:return dict(passed=False,contract_passed=False,contract=contract,reason='Home verification contract failed')
    if len(snapshots) < 6 or any(s.get('state_source') != 'isaac_articulation' for s in snapshots):
        return dict(passed=False,contract_passed=True,contract=contract, reason='Six physical home ticks required')
    states = [s['state'][:5] for s in snapshots]
    if not all(math.isfinite(v) for row in states for v in row):
        return dict(passed=False,contract_passed=True,contract=contract, reason='Non-finite physical home state')
    error = max(abs(q-r) for q,r in zip(states[-1],target))
    spread = max(max(row[i] for row in states)-min(row[i] for row in states) for i in range(5))
    return dict(contract_passed=True,contract=contract,passed=error <= .01 and spread <= .002, home_error_rad=error,
                position_spread_rad=spread, sampled_velocity=snapshots[-1]['velocity'])



def contained(snapshot, object_name, color):
    block = snapshot['bodies'][object_name]
    cup = snapshot['bodies']['cup_' + color]
    inverse = [cup['orientation'][0]] + [-v for v in cup['orientation'][1:]]
    local=snapshot['local_corners'][object_name]
    if len(local)!=8 or any(len(c)!=3 or not all(math.isfinite(v) for v in c) for c in local):
        raise ValueError('Invalid physical block corners')
    corners = []
    for corner in snapshot['local_corners'][object_name]:
        offset = rotate(block['orientation'], corner)
        relative = [block['position'][i]+offset[i]-cup['position'][i] for i in range(3)]
        corners.append(rotate(inverse, relative))
    radius = max(math.hypot(c[0], c[1]) for c in corners)
    walls=wall_checks(corners)
    low, high = min(c[2] for c in corners), max(c[2] for c in corners)
    velocity = max(math.sqrt(sum(v*v for v in body['velocity'][:3])) for body in (block, cup))
    angular = max(math.sqrt(sum(v*v for v in body['velocity'][3:])) for body in (block, cup))
    upright = rotate(cup['orientation'], [0, 0, 1])[2] > math.cos(math.radians(5))
    ok = (all(c['passed'] for c in walls) and low >= INTERIOR_FLOOR_M and high <= INTERIOR_RIM_M
          and velocity <= .015 and angular <= .15 and upright)
    return dict(passed=ok, max_corner_radius_m=radius, lowest_corner_m=low,
                highest_corner_m=high, linear_speed_m_s=velocity, angular_speed_rad_s=angular,
                upright=upright, min_wall_clearance_m=min(c['clearance_m'] for c in walls),
                wall_checks=walls,geometry_source_sha256=PROFILE['source_sha256'],
                floor_tolerance_m=FLOOR_REST_TOLERANCE_M,wall_tolerance_m=WALL_REST_TOLERANCE_M)


def settled_containment(snapshots,object_name,color):
    contract=snapshot_contracts(snapshots)
    if not contract['passed']:return dict(passed=False,contract_passed=False,contract=contract,checks=[])
    checks=[contained(s,object_name,color) for s in snapshots]
    physical=len(checks)>=SETTLED_TICKS and all(c['passed'] for c in checks)
    return dict(passed=physical,physical_passed=physical,contract_passed=True,contract=contract,
                required_ticks=SETTLED_TICKS,checks=checks,sim_stamps_ns=[s['sim_ns'] for s in snapshots])
