"""Whole-block containment in the cup frame, using synchronized physical poses."""
import math
from cup_profile import radial_limit

# Cup body coordinates have their origin at the base; rim height is 96.52 mm.
# Authored floor is 1.2 mm. The 0.6 mm floor allowance is geometric;
# effective block/cup rest offsets are zero and do not authorize overlap.
PHYSICAL_FLOOR_M = .0012
FLOOR_REST_TOLERANCE_M = .0006
INTERIOR_FLOOR_M = PHYSICAL_FLOOR_M - FLOOR_REST_TOLERANCE_M
INTERIOR_RIM_M = .094
SETTLED_TICKS = 15


def initial_arm(snapshots, target):
    """Verify measured home/stability, then the transport resets initial velocities.

    PhysX joint velocity samples can include constraint impulses even when
    sampled positions are stable; retain those samples rather than relabel them.
    """
    if len(snapshots) < 6 or any(s.get('state_source') != 'isaac_articulation' for s in snapshots):
        return dict(passed=False, reason='Six physical home ticks required')
    states = [s['state'][:5] for s in snapshots]
    if not all(math.isfinite(v) for row in states for v in row):
        return dict(passed=False, reason='Non-finite physical home state')
    error = max(abs(q-r) for q,r in zip(states[-1],target))
    spread = max(max(row[i] for row in states)-min(row[i] for row in states) for i in range(5))
    return dict(passed=error <= .01 and spread <= .002, home_error_rad=error,
                position_spread_rad=spread, sampled_velocity=snapshots[-1]['velocity'])


def rotate(q, v):
    w, x, y, z = q
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    if norm < 1e-10:
        raise ValueError('Invalid pose quaternion')
    w, x, y, z = (t/norm for t in q)
    uv = (y*v[2]-z*v[1], z*v[0]-x*v[2], x*v[1]-y*v[0])
    uuv = (y*uv[2]-z*uv[1], z*uv[0]-x*uv[2], x*uv[1]-y*uv[0])
    return [v[i]+2*(w*uv[i]+uuv[i]) for i in range(3)]


def contained(snapshot, object_name, color):
    block = snapshot['bodies'][object_name]
    cup = snapshot['bodies']['cup_' + color]
    inverse = [cup['orientation'][0]] + [-v for v in cup['orientation'][1:]]
    corners = []
    for corner in snapshot['local_corners'][object_name]:
        offset = rotate(block['orientation'], corner)
        relative = [block['position'][i]+offset[i]-cup['position'][i] for i in range(3)]
        corners.append(rotate(inverse, relative))
    radius = max(math.hypot(c[0], c[1]) for c in corners)
    radial_checks = [dict(corner=index, height_m=c[2], radius_m=math.hypot(c[0],c[1]),
                          allowed_radius_m=radial_limit(c[2]),
                          clearance_m=radial_limit(c[2])-math.hypot(c[0],c[1]))
                     for index,c in enumerate(corners)]
    low, high = min(c[2] for c in corners), max(c[2] for c in corners)
    velocity = max(math.sqrt(sum(v*v for v in body['velocity'][:3])) for body in (block, cup))
    angular = max(math.sqrt(sum(v*v for v in body['velocity'][3:])) for body in (block, cup))
    upright = rotate(cup['orientation'], [0, 0, 1])[2] > math.cos(math.radians(5))
    ok = (all(c['clearance_m'] >= 0 for c in radial_checks) and low >= INTERIOR_FLOOR_M and high <= INTERIOR_RIM_M
          and velocity <= .015 and angular <= .15 and upright)
    return dict(passed=ok, max_corner_radius_m=radius, lowest_corner_m=low,
                highest_corner_m=high, linear_speed_m_s=velocity, angular_speed_rad_s=angular,
                upright=upright, min_radial_clearance_m=min(c['clearance_m'] for c in radial_checks),
                radial_checks=radial_checks)


def settled_containment(snapshots, object_name, color):
    checks = [contained(s, object_name, color) for s in snapshots]
    gaps = [b['sim_ns']-a['sim_ns'] for a,b in zip(snapshots, snapshots[1:])]
    passed = (len(checks) >= SETTLED_TICKS and all(c['passed'] for c in checks)
              and all(abs(dt-round(1e9/30)) <= 10 for dt in gaps))
    return dict(passed=passed, required_ticks=SETTLED_TICKS, checks=checks,
                sim_stamps_ns=[s['sim_ns'] for s in snapshots])
