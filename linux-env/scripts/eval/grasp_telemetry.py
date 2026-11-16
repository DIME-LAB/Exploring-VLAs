"""Opt-in measured pad geometry and interval contact telemetry for native eval.

Imported outside Kit for CPU geometry tests; install() requires the held collector.
It changes contact-report subscription/threshold only, never drive/collision physics.
"""
import itertools
import hashlib
import math
from pathlib import Path
import time
import numpy as np


def pose_matrix(position, quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=float)
    if not np.isclose(w*w+x*x+y*y+z*z, 1, atol=1e-5):
        raise ValueError('Invalid native quaternion')
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = position
    return result


def line_box_interval(origin, direction, matrix, half_size):
    """Signed distances along a world line inside a finite affine box."""
    inverse = np.linalg.inv(matrix)
    p = (inverse @ np.r_[origin, 1])[:3]
    d = (inverse @ np.r_[direction, 0])[:3]
    low, high = -math.inf, math.inf
    for o, v, h in zip(p, d, half_size):
        if abs(v) < 1e-12:
            if abs(o) > h + 1e-12: return None
        else:
            a, b = sorted(((-h-o)/v, (h-o)/v))
            low, high = max(low, a), min(high, b)
            if low > high: return None
    return [float(low), float(high)]


def aperture_at_depth(origin, axis, fixed, moving):
    """Finite pad-face gap on the opening-axis line through a block centre.

    A line missing either jaw's finite pads is explicitly unavailable; no
    angle-to-gap fit or infinite-plane extrapolation is used. Negative gap
    denotes overlap of the pad intervals, not physical interpenetration proof.
    """
    direction = np.asarray(axis, dtype=float)
    if not np.isclose(np.linalg.norm(direction), 1): raise ValueError('Opening axis must be unit length')
    intervals = []
    for pads in (fixed, moving):
        hits = [line_box_interval(origin, direction, np.asarray(p['world_matrix']), p['half_size_m']) for p in pads]
        intervals.append([hit for hit in hits if hit is not None])
    if not all(intervals):
        return dict(status='unavailable-outside-finite-pad-depth', signed_separation_m=None,
                    fixed_intervals_m=intervals[0], moving_intervals_m=intervals[1])
    left = max(hit[1] for hit in intervals[0]); right = min(hit[0] for hit in intervals[1])
    return dict(status='measured-finite-pad-line', signed_separation_m=right-left,
                fixed_intervals_m=intervals[0], moving_intervals_m=intervals[1])


class ContactInterval:
    """Copy all callbacks until a new snapshot stamp, without I/O or locks."""
    def __init__(self):
        self.steps = []; self.events = []; self.error = None
        self.previous = None; self.cached = None

    def step(self, dt, sim_ns):
        self.steps.append(dict(dt_s=float(dt), sim_ns=int(sim_ns), wall_ns=time.monotonic_ns()))

    def add(self, event):
        self.events.append(event)

    def drain(self, epoch, sim_ns):
        stamp = (int(epoch), int(sim_ns))
        if stamp == self.previous: return self.cached
        previous = self.previous
        steps, events = self.steps, self.events
        self.steps, self.events = [], []
        status = 'unavailable' if self.error or previous is None or previous[0] != stamp[0] or not steps else 'contacts' if any(e['points'] or e['friction_anchors'] for e in events) else 'zero-contact'
        value = dict(status=status, error=self.error, start_epoch=previous[0] if previous else None,
                     start_sim_ns=previous[1] if previous else None, end_epoch=stamp[0], end_sim_ns=stamp[1],
                     physics_substep_count=len(steps), physics_dt_sum_s=sum(s['dt_s'] for s in steps),
                     physics_steps=steps, events=events,
                     contact_impulse_vector_sum_Ns=np.sum([p['impulse_Ns'] for e in events for p in e['points']], axis=0).tolist() if any(e['points'] for e in events) else [0., 0., 0.],
                     friction_anchor_impulse_vector_sum_Ns=np.sum([p['impulse_Ns'] for e in events for p in e['friction_anchors']], axis=0).tolist() if any(e['friction_anchors'] for e in events) else [0., 0., 0.])
        self.previous = stamp; self.cached = value
        return value


def guarded_snapshot(original, augment, interval):
    """A reporting failure must never poison the collector's physics runtime."""
    def snapshot():
        result = original()
        try:
            augment(result)
        except Exception as error:
            interval.error = repr(error)
            result['grasp_telemetry'] = dict(version=1, status='unavailable', error=repr(error),
                                           contacts=interval.drain(result['epoch'], result['sim_ns']))
        return result
    return snapshot


def validate_intervals(snapshots):
    """Acceptance applies to every physics interval after the baseline."""
    counts = dict(contact_ticks=0, zero_contact_ticks=0, physics_substeps=0)
    for current in snapshots[1:]:
        details = current['grasp_telemetry']; interval = details['contacts']
        if details.get('status') == 'unavailable' or interval['status'] not in ('contacts', 'zero-contact') or interval['physics_substep_count'] <= 0 or abs(interval['physics_dt_sum_s']-1/30) >= 1e-6:
            raise ValueError('Unavailable or incomplete measured-contact interval')
        counts['contact_ticks' if interval['status'] == 'contacts' else 'zero_contact_ticks'] += 1
        counts['physics_substeps'] += interval['physics_substep_count']
    return counts


def install(runtime, *, enabled=False):
    """Install only in the owned, paused native simulation, before warmup."""
    if enabled is not True or runtime.mode != 'collect' or runtime.timeline.is_playing():
        raise RuntimeError('Measured grasp telemetry requires opt-in held simulation')
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if hasattr(runtime, '_eval_grasp_telemetry'):
        import copy
        owner = runtime._eval_grasp_telemetry
        if owner['source_sha256'] != source_sha or owner['stage'] is not runtime.stage or runtime.snapshot is not owner['wrapper'] or owner['interval'].error or any(owner.get(k) is None for k in ('step_subscription', 'contact_subscription')):
            raise RuntimeError('Existing grasp telemetry does not match the healthy owned installation')
        return dict(copy.deepcopy(owner['proof']), reused=True)
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools, Gf
    from isaacsim.core.prims import RigidPrim
    from omni.physx import get_physx_interface, get_physx_simulation_interface
    import importlib.util
    repo = Path(__file__).resolve().parents[3]
    path = repo/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/compute_workspace.py'
    spec = importlib.util.spec_from_file_location('eval_grasp_fk', path)
    fk = importlib.util.module_from_spec(spec); spec.loader.exec_module(fk)
    stage = runtime.stage
    links = {}; geometry = {}; thresholds = []
    for role, name in (('fixed', 'gripper_link'), ('moving', 'moving_jaw_so101_v1_link')):
        found = [p for p in Usd.PrimRange(stage.GetPrimAtPath('/World/SO_ARM101')) if p.GetName() == name and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(found) != 1: raise RuntimeError('Cannot uniquely identify physical jaw: '+name)
        body = found[0]; body_path = str(body.GetPath())
        view = RigidPrim(body_path, name='eval_measured_'+role+'_jaw', reset_xform_properties=False, prepare_contact_sensors=False); view.initialize()
        links[role] = (body_path, view); pads = []
        for prim in Usd.PrimRange(body):
            if prim.GetName().startswith('pad_') and prim.HasAPI(UsdPhysics.CollisionAPI):
                if not prim.IsA(UsdGeom.Cube): raise RuntimeError('Expected finite primitive pad cube')
                matrix, _ = UsdGeom.XformCache(Usd.TimeCode.Default()).ComputeRelativeTransform(prim, body)
                size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
                # USD uses row-vector matrices; telemetry matrices use column vectors.
                pads.append(dict(collider_path=str(prim.GetPath()), body_path=body_path,
                                 body_matrix=np.asarray(matrix, dtype=float).T.tolist(), half_size_m=[size/2]*3))
        if len(pads) != 3: raise RuntimeError('Expected exactly three finite collision pads on '+name)
        geometry[role] = pads
        api = PhysxSchema.PhysxContactReportAPI.Apply(body)
        attr = api.CreateThresholdAttr(); old = attr.Get(); attr.Set(0.0)
        thresholds.append(dict(body_path=body_path, old=old, new=0.0))
    interval = ContactInterval()
    finger_paths = {v[0] for v in links.values()}
    block_paths = {path for name, path in runtime.body_paths.items() if any(name.startswith(c) for c in ('red', 'green', 'blue')) and '2x' in name}
    if len(block_paths) < 3: raise RuntimeError('Cannot identify coloured block rigid bodies')

    def on_step(dt):
        try: interval.step(dt, runtime.stamp())
        except Exception as error: interval.error = repr(error)

    def on_contact(headers, data, anchors):
        try:
            for header in headers:
                actor0, actor1 = (str(PhysicsSchemaTools.intToSdfPath(getattr(header, k))) for k in ('actor0', 'actor1'))
                if not ((actor0 in finger_paths and actor1 in block_paths) or (actor1 in finger_paths and actor0 in block_paths)): continue
                event = dict(actor0=actor0, actor1=actor1, collider0=str(PhysicsSchemaTools.intToSdfPath(header.collider0)), collider1=str(PhysicsSchemaTools.intToSdfPath(header.collider1)), event_type=str(header.type), sim_ns=runtime.stamp(), wall_ns=time.monotonic_ns(), points=[], friction_anchors=[])
                for i in range(header.contact_data_offset, header.contact_data_offset+header.num_contact_data):
                    item = data[i]; impulse = [float(x) for x in item.impulse]; normal = [float(x) for x in item.normal]
                    projection = float(np.dot(impulse, normal))
                    event['points'].append(dict(position_m=[float(x) for x in item.position], normal=normal, impulse_Ns=impulse, normal_projection_Ns=projection, tangential_residual_Ns=(np.asarray(impulse)-projection*np.asarray(normal)).tolist(), separation_m=float(item.separation)))
                for i in range(header.friction_anchors_offset, header.friction_anchors_offset+header.num_friction_anchors_data):
                    item = anchors[i]
                    event['friction_anchors'].append(dict(position_m=[float(x) for x in item.position], impulse_Ns=[float(x) for x in item.impulse]))
                interval.add(event)
        except Exception as error: interval.error = repr(error)

    owner = dict(interval=interval, links=links, geometry=geometry, stage=stage, source_sha256=source_sha)
    owner['step_subscription'] = get_physx_interface().subscribe_physics_step_events(on_step)
    owner['contact_subscription'] = get_physx_simulation_interface().subscribe_full_contact_report_events(on_contact)
    original = runtime.snapshot

    def augment(result):
        measured = {}; pads = {}
        for role, (path, view) in links.items():
            positions, orientations = view.get_world_poses()
            transform = pose_matrix(positions[0], orientations[0])
            measured[role] = dict(body_path=path, position_m=positions[0].tolist(), quaternion_wxyz=orientations[0].tolist(), world_matrix=transform.tolist())
            pads[role] = []
            for pad in geometry[role]:
                matrix = transform @ np.asarray(pad['body_matrix'])
                corners = [(matrix @ np.r_[np.asarray(sign)*pad['half_size_m'], 1])[:3].tolist() for sign in itertools.product((-1, 1), repeat=3)]
                pads[role].append(dict(**pad, world_matrix=matrix.tolist(), world_corners_m=corners))
        fixed_matrix = np.asarray(measured['fixed']['world_matrix'])
        direction = fixed_matrix[:3, 0]
        fixed_centre = np.mean([np.asarray(p['world_matrix'])[:3, 3] for p in pads['fixed']], axis=0)
        moving_centre = np.mean([np.asarray(p['world_matrix'])[:3, 3] for p in pads['moving']], axis=0)
        # Declare an axis pointing from the fixed pad toward the moving pad.
        if np.dot(direction, moving_centre-fixed_centre) < 0: direction = -direction
        blocks = {}
        for name, body in result['bodies'].items():
            if runtime.body_paths[name] not in block_paths: continue
            transform = pose_matrix(body['position'], body['orientation'])
            corners = [(transform @ np.r_[point, 1])[:3].tolist() for point in result['local_corners'][name]]
            blocks[name] = dict(body_path=runtime.body_paths[name], pose=body, world_corners_m=corners,
                                aperture_at_block_depth=aperture_at_depth(body['position'], direction, pads['fixed'], pads['moving']))
        result['grasp_telemetry'] = dict(version=1, measured_links=measured, finite_pads=pads, blocks=blocks,
                                       gripper_joint_rad=float(result['state'][5]), opening_axis_world=direction.tolist(),
                                       opening_axis_definition='fixed jaw local X, signed toward moving pad centres',
                                       tcp_frame='fixed-jaw tip; compute_workspace.TCP_TRANSFORM',
                                       tcp_world_matrix=(fixed_matrix @ fk.TCP_TRANSFORM).tolist(),
                                       contacts=interval.drain(result['epoch'], result['sim_ns']))
    wrapper = guarded_snapshot(original, augment, interval)
    proof = dict(version=1, enabled=True, reused=False, source_sha256=source_sha, links={k:v[0] for k,v in links.items()}, finite_pads=geometry,
                block_paths=sorted(block_paths), contact_threshold_changes=thresholds,
                contact_api='subscribe_full_contact_report_events', physics_api='subscribe_physics_step_events',
                impulse_units='N s; raw contact vector/projection and separate raw friction anchors; no force conversion')
    owner.update(wrapper=wrapper, proof=proof)
    runtime.snapshot = wrapper; runtime._eval_grasp_telemetry = owner
    return proof
