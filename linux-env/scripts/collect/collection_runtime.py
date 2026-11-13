"""Isaac main-thread transport: measured feedback and one-frame command handshake.

Loaded by the assigned Kit bridge, never by editing the running extension.
Collection disconnects only this scene's ROS follower executor. Commands write
its existing PhysX drive targets while paused, then exactly one render/physics
frame advances. Deployment restores the ROS follower and keeps measured feedback.
"""
import json
import hashlib
import math
import os
from pathlib import Path
import time
import numpy as np
import omni.graph.core as og
import omni.kit.app
import omni.timeline
import omni.usd
import rclpy
from rclpy.context import Context
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim
from contract import JOINTS

_RUNTIME = None
_LOADED_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class Runtime:
    def __init__(self):
        import torch
        torch.cuda.set_device(int(os.environ.get('ISAACSIM_GPU', '0')))
        self.world = World.instance()
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.pause()
        self.busy = False
        self.failure = None
        self.mode = 'collect'
        self.start_time = self.world.current_time
        self.last_snapshot = None
        self.art = Articulation('/World/SO_ARM101', name='collection_measured_arm', reset_xform_properties=False)
        self.art.initialize()
        self.indices = [self.art.dof_names.index(name) for name in JOINTS]
        self.stage = omni.usd.get_context().get_stage()
        self.body_paths = {}
        for root in ('/World/Objects', '/World/Containers'):
            parent = self.stage.GetPrimAtPath(root)
            for child in parent.GetChildren():
                if child.GetName() == 'PhysicsMaterial' or child.GetName().startswith('drop_'):
                    continue
                body = next((p for p in Usd.PrimRange(child) if p.HasAPI(UsdPhysics.RigidBodyAPI)), None)
                if body:
                    self.body_paths[child.GetName()] = str(body.GetPath())
        self.require_scene()
        self.bodies = {}
        self.local_corners = {}
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for name, path in self.body_paths.items():
            view = RigidPrim(path, name='collection_' + name, reset_xform_properties=False,
                             prepare_contact_sensors=False)
            view.initialize()
            self.bodies[name] = view
            if not name.startswith('cup_'):
                # Preserve an oriented bbox in the physical body's coordinate frame.
                box = cache.ComputeWorldBound(self.stage.GetPrimAtPath(path))
                low, high = box.GetRange().GetMin(), box.GetRange().GetMax()
                pos, quat = view.get_world_poses()
                rotation = Gf.Rotation(Gf.Quatd(float(quat[0][0]), Gf.Vec3d(*map(float, quat[0][1:]))))
                inverse = rotation.GetInverse()
                corners = []
                for x in (low[0], high[0]):
                    for y in (low[1], high[1]):
                        for z in (low[2], high[2]):
                            world = box.GetMatrix().Transform(Gf.Vec3d(x, y, z))
                            corners.append(list(inverse.TransformDir(world - Gf.Vec3d(*map(float, pos[0])))))
                self.local_corners[name] = corners
        self.context = Context()
        rclpy.init(args=[], context=self.context)
        self.node = rclpy.create_node('collection_physical_feedback', context=self.context)
        self.joint_pub = self.node.create_publisher(JointState, '/joint_states_measured', 64)
        self.tick_pub = self.node.create_publisher(String, '/collection_tick', 64)
        self.subscription = omni.kit.app.get_app().get_post_update_event_stream().create_subscription_to_pop(
            self.after_update, name='collection_one_frame')
        self.set_mode('collect')

    def require_scene(self):
        missing = [name for name in ('cup_red', 'cup_green', 'cup_blue')
                   if name not in self.body_paths or not self.stage.GetPrimAtPath(self.body_paths[name]).IsValid()]
        if missing:
            raise RuntimeError('Fresh scene required: missing existing rigid cups ' + repr(missing))
        for name, path in self.body_paths.items():
            if not self.stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError('Fresh scene required: missing existing body ' + name)

    def set_mode(self, mode):
        if self.busy:
            raise RuntimeError('Cannot change mode while a frame is running')
        self.timeline.pause()
        source = '/Graph/ActionGraph_SO_ARM101/on_playback_tick.outputs:tick'
        target = '/Graph/ActionGraph_SO_ARM101/articulation_controller.inputs:execIn'
        if mode == 'collect':
            og.Controller.disconnect(source, target)
        elif mode == 'deploy':
            og.Controller.connect(source, target)
        else:
            raise ValueError(mode)
        self.mode = mode
        return dict(mode=mode, measured_topic='/joint_states_measured', domain=os.environ.get('ROS_DOMAIN_ID'))

    def stamp(self):
        # This is the same native simulation clock wired to ROS /clock and
        # ROS2CameraHelper. World.current_time has float integration drift.
        return round(float(og.Controller.attribute(
            '/Graph/ActionGraph_SO_ARM101/isaac_read_simulation_time.outputs:simulationTime').get()) * 1e9)

    def snapshot(self):
        self.require_scene()
        positions = self.art.get_joint_positions()[0, self.indices]
        velocities = self.art.get_joint_velocities()[0, self.indices]
        bodies = {}
        for name, view in self.bodies.items():
            pos, quat = view.get_world_poses()
            velocity = view.get_velocities()[0]
            bodies[name] = dict(position=list(map(float, pos[0])), orientation=list(map(float, quat[0])),
                                velocity=list(map(float, velocity)))
        return dict(sim_ns=self.stamp(), physics_time_s=self.world.current_time,
                    state=list(map(float, positions)), velocity=list(map(float, velocities)),
                    bodies=bodies, local_corners=self.local_corners, state_source='isaac_articulation')

    def publish(self):
        snapshot = self.snapshot()
        msg = JointState()
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(snapshot['sim_ns'], 1000000000)
        msg.name = list(JOINTS);msg.position = snapshot['state'];msg.velocity = snapshot['velocity']
        self.joint_pub.publish(msg)
        self.tick_pub.publish(String(data=json.dumps(snapshot)))
        self.last_snapshot = snapshot
        return snapshot

    def apply(self, command, observation_ns):
        if self.busy or self.timeline.is_playing():
            raise RuntimeError('Command application requires a held simulation')
        if abs(self.stamp() - observation_ns) > 10:
            raise RuntimeError(f'Stale command observation: {observation_ns} != {self.stamp()}')
        values = np.asarray(command, dtype=np.float32)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError('Invalid six-joint command')
        self.art.set_joint_position_targets(values.reshape(1, 6), joint_indices=self.indices)
        self.art.set_joint_velocity_targets(np.zeros((1, 6), dtype=np.float32), joint_indices=self.indices)
        applied = self.art.get_applied_actions().joint_positions[0, self.indices]
        if not np.allclose(applied, values, atol=1e-6, rtol=0):
            raise RuntimeError('PhysX did not accept the executed targets')
        return dict(application_stamp_ns=self.stamp(), applied_command=list(map(float, applied)))

    def advance(self):
        if self.failure:
            raise RuntimeError(self.failure)
        if self.mode != 'collect' or self.busy:
            raise RuntimeError('An advancing frame is already pending or mode is not collect')
        self.start_time = self.world.current_time
        self.busy = True
        self.timeline.play()
        return dict(start_ns=round(self.start_time * 1e9), target_ns=round((self.start_time + 1/30) * 1e9))

    def cancel(self):
        self.timeline.pause()
        self.busy = False
        return dict(held=True, failure=self.failure)

    def after_update(self, event):
        try:
            if self.mode == 'collect':
                if not self.busy or self.world.current_time <= self.start_time + 1e-8:
                    return
                self.timeline.pause()
                delta = self.world.current_time - self.start_time
                self.busy = False
                if abs(delta - 1/30) > 1e-7:
                    raise RuntimeError(f'One-frame step advanced {delta} sim seconds')
                self.publish()
            elif self.timeline.is_playing():
                self.publish()
        except Exception as exc:
            self.timeline.pause();self.busy = False;self.failure = repr(exc)

    def reset_velocities(self):
        for view in self.bodies.values():
            view.set_velocities(np.zeros((1, 6), dtype=np.float32))
        return True

    def restore(self, snapshot, arm=False):
        if self.busy or self.timeline.is_playing():
            raise RuntimeError('Restore requires a held simulation')
        self.require_scene()
        for name, body in snapshot['bodies'].items():
            view = self.bodies[name]
            view.set_world_poses(np.asarray([body['position']], dtype=np.float32),
                                 np.asarray([body['orientation']], dtype=np.float32))
            view.set_velocities(np.zeros((1, 6), dtype=np.float32))
        if arm:
            self.art.set_joint_positions(np.asarray([snapshot['state']], dtype=np.float32), joint_indices=self.indices)
            self.art.set_joint_velocities(np.zeros((1, 6), dtype=np.float32), joint_indices=self.indices)
            self.apply(snapshot['state'], self.stamp())
        actual = self.snapshot()
        error = max(abs(x-y) for name in snapshot['bodies']
                    for x,y in zip(snapshot['bodies'][name]['position'], actual['bodies'][name]['position']))
        orientation_error = max(min(max(abs(x-y) for x,y in zip(snapshot['bodies'][name]['orientation'], actual['bodies'][name]['orientation'])),
                                    max(abs(x+y) for x,y in zip(snapshot['bodies'][name]['orientation'], actual['bodies'][name]['orientation'])))
                                for name in snapshot['bodies'])
        if error > 1e-6:
            raise RuntimeError(f'Initial pose restore differs by {error} m')
        if orientation_error > 1e-6 or any(max(abs(v) for v in body['velocity']) > 1e-6 for body in actual['bodies'].values()):
            raise RuntimeError('Initial orientation/zero velocity restore failed')
        return dict(max_position_error_m=error, max_orientation_error=orientation_error, snapshot=actual)


def install(mode='collect'):
    global _RUNTIME
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _LOADED_SHA256:
        raise RuntimeError('Runtime source changed; relaunch this instance before collection')
    if _RUNTIME is None:
        _RUNTIME = Runtime()
    _RUNTIME.set_mode(mode)
    import torch
    import carb
    return dict(mode=mode, joint_names=_RUNTIME.art.dof_names,
                loaded_sha256=_LOADED_SHA256,
                cuda_device=torch.cuda.current_device(), renderer_gpu=carb.settings.get_settings().get('/renderer/activeGpu'),
                physics_gpu=carb.settings.get_settings().get('/physics/cudaDevice'),
                measured_topic='/joint_states_measured', local_corners=_RUNTIME.local_corners)


def runtime():
    if _RUNTIME is None:
        raise RuntimeError('Collection runtime is not installed')
    return _RUNTIME
