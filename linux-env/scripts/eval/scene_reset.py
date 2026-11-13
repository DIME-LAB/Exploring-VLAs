"""Seeded physical scene bank for paired policy evaluations (executed in Kit)."""
from pathlib import Path
import json

_VIEWS = {}


def prepare(seed, bank):
    import numpy as np
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdPhysics
    from isaacsim.core.prims import RigidPrim
    from so_arm101_dt.extension import get_instance

    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    playing = timeline.is_playing()
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
            baseline.write_text(json.dumps(snapshot(), indent=2) + '\n')
        target = bank / f'seed-{seed}.json'
        ext = get_instance()
        if not target.exists():
            restore(json.loads(baseline.read_text()))
            rng = np.random.get_state()
            try:
                np.random.seed(seed)
                result = ext._cmd_randomize_object_poses(required_visible_color='blue', required_visible_count=1)
                if result.get('status') != 'success':
                    raise RuntimeError(str(result))
                bodies = snapshot()
            finally:
                np.random.set_state(rng)
            target.write_text(json.dumps(bodies, indent=2) + '\n')
        bodies = json.loads(target.read_text())
        error, qerror = restore(bodies)
        visible = [name for name, body in bodies.items() if name.startswith('blue')
                   and ext._is_in_camera_view(body['position'], margin_px=30)]
        if not visible:
            raise RuntimeError(f'Seed {seed} has no visible blue lego')
        return dict(seed=seed, scene_file=str(target), bodies=bodies, visible_blue=visible,
                    max_position_error_m=error, max_orientation_error=qerror)
    finally:
        if playing:
            timeline.play()
