"""Seeded physical scene bank for paired policy evaluations (executed in Kit)."""
from pathlib import Path
import hashlib
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
import json
import time
import os
from collections import deque

_VIEWS = {}
_FRAME_SAMPLES = deque(maxlen=20000)
_FRAME_SUB = None


def start_probe():
    """Timestamp Kit updates in memory; no publisher, file I/O or scene mutation."""
    global _FRAME_SUB
    import omni.kit.app
    from isaacsim.core.api import World
    if _FRAME_SUB is None:
        world = World.instance()
        _FRAME_SUB = omni.kit.app.get_app().get_post_update_event_stream().create_subscription_to_pop(
            lambda _: _FRAME_SAMPLES.append((time.time(), world.current_time)), name='eval_frame_probe')
    _FRAME_SAMPLES.clear()
    return dict(started_wall_time=time.time(), pid=os.getpid())


def probe():
    samples = list(_FRAME_SAMPLES)
    gaps = [b[0]-a[0] for a,b in zip(samples, samples[1:])]
    return dict(samples=samples, max_wall_gap_s=max(gaps, default=None))


def prepare(seed, bank, colors=('blue',), generate=True):
    if not generate:
        from phase3 import validate_bank
        validate_bank(bank)
    import numpy as np
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdPhysics
    from isaacsim.core.prims import RigidPrim
    from so_arm101_dt.extension import get_instance

    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    playing = timeline.is_playing()
    reset = {}
    if len(colors) == 3:
        import collection_runtime
        reset = collection_runtime.runtime().cancel()
    timeline.pause()
    try:
        if not _VIEWS:
            for root in ('/World/Objects', '/World/Containers'):
                for child in stage.GetPrimAtPath(root).GetChildren():
                    body = next((p for p in Usd.PrimRange(child) if p.HasAPI(UsdPhysics.RigidBodyAPI)), None)
                    if body:
                        view = RigidPrim(str(body.GetPath()), name='eval_' + child.GetName(),
                                         reset_xform_properties=False, prepare_contact_sensors=False)
                        view.initialize()
                        _VIEWS[child.GetName()] = view
        if not {'cup_blue', 'cup_green', 'cup_red'} <= _VIEWS.keys():
            raise RuntimeError('Evaluation requires all three physical cups')
        bank = Path(bank)
        bank.mkdir(parents=True, exist_ok=True)

        def snapshot():
            out = {}
            for name, view in _VIEWS.items():
                pos, quat = view.get_world_poses()
                out[name] = dict(position=list(map(float, pos[0])), orientation=list(map(float, quat[0])))
            return out

        def restore(bodies):
            if set(bodies) != set(_VIEWS):
                raise RuntimeError('Scene bank body names differ from the live scene')
            for name, body in bodies.items():
                view = _VIEWS[name]
                view.set_world_poses(np.asarray([body['position']], dtype=np.float32),
                                     np.asarray([body['orientation']], dtype=np.float32))
                view.set_velocities(np.zeros((1, 6), dtype=np.float32))
            actual = snapshot()
            error = max(abs(a-b) for name in bodies for a,b in zip(bodies[name]['position'], actual[name]['position']))
            qerror = max(min(max(abs(a-b) for a,b in zip(bodies[name]['orientation'], actual[name]['orientation'])),
                             max(abs(a+b) for a,b in zip(bodies[name]['orientation'], actual[name]['orientation']))) for name in bodies)
            if error > 1e-6 or qerror > 1e-6:
                raise RuntimeError(f'Scene restore error: {error} m / {qerror} quaternion')
            velocity = max(abs(float(v)) for view in _VIEWS.values() for v in view.get_velocities()[0])
            if velocity > 1e-6:
                raise RuntimeError(f'Scene velocity reset error: {velocity}')
            return error, qerror

        baseline = bank / 'baseline.json'
        if not baseline.exists():
            if not generate:raise FileNotFoundError('Frozen baseline file missing')
            baseline.write_text(json.dumps(snapshot(), indent=2) + '\n')
        target = bank / f'seed-{seed}.json'
        ext = get_instance()
        if not target.exists():
            if not generate:
                raise FileNotFoundError('Frozen seed file missing: '+str(target))
            restore(json.loads(baseline.read_text()))
            rng = np.random.get_state()
            try:
                np.random.seed(seed)
                for attempt in range(64):
                    restore(json.loads(baseline.read_text()))
                    result = ext._cmd_randomize_object_poses(required_visible_color=colors[0], required_visible_count=1)
                    if result.get('status') != 'success':
                        raise RuntimeError(str(result))
                    bodies = snapshot()
                    if all(any(name.startswith(color) and ext._is_in_camera_view(body['position'], margin_px=30)
                               for name,body in bodies.items()) for color in colors):
                        break
                else:
                    raise RuntimeError(f'Seed {seed} cannot produce all-colour visibility')
            finally:
                np.random.set_state(rng)
            target.write_text(json.dumps(bodies, indent=2) + '\n')
        bodies = json.loads(target.read_text())
        error, qerror = restore(bodies)
        visible = {color: [name for name, body in bodies.items() if name.startswith(color)
                   and ext._is_in_camera_view(body['position'], margin_px=30)] for color in colors}
        if any(not names for names in visible.values()):
            raise RuntimeError(f'Seed {seed} lacks requested colour visibility: {visible}')
        return dict(seed=seed, scene_file=str(target), bodies=bodies, visible_blue=visible.get('blue', []), visible_colors=visible,
                    reset_epoch=reset.get('epoch'), reset_stamp_ns=reset.get('invalid_through_ns'),
                    max_position_error_m=error, max_orientation_error=qerror)
    finally:
        if playing:
            timeline.play()


def freeze(bank):
    """Generate every held-out scene before scoring and seal its physical bytes."""
    import hashlib
    bank = Path(bank)
    if (bank/'bank-manifest.json').exists():
        raise FileExistsError('Bank already frozen')
    receipts = [prepare(seed, bank, ('red','green','blue')) for seed in range(900000,900030)]
    files = [bank/'baseline.json']+[bank/f'seed-{seed}.json' for seed in range(900000,900030)]
    manifest = dict(seeds=list(range(900000,900030)),colors=['red','green','blue'],
                    files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    visible={r['seed']:r['visible_colors'] for r in receipts})
    (bank/'bank-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
