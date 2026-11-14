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
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim
from comparisons import matches
from tick_contract import world_step_errors
from contract import JOINTS
from protocol import CommandLedger

_RUNTIME = None
_LOADED_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_LOADED_DEPENDENCIES = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ('contract.py', 'protocol.py','comparisons.py','tick_contract.py')}


class Runtime:
    def __init__(self):
        import torch
        if torch.cuda.device_count()!=1:raise RuntimeError('Collection requires exactly one visible CUDA device')
        torch.cuda.set_device(0)
        self.world = World.instance()
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.pause()
        self.busy = False
        self.failure = None
        self.mode = 'collect'
        self.start_time = self.world.current_time
        self.last_snapshot = None
        self.epoch = 0
        self.invalid_through_ns = -1
        self.ledger = CommandLedger()
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
        self.node = rclpy.create_node('collection_physical_feedback', context=self.context,
            parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.joint_pub = self.node.create_publisher(JointState, '/joint_states_measured', 64)
        self.tick_pub = self.node.create_publisher(String, '/collection_tick', 64)
        self.ack_pub = self.node.create_publisher(String, '/collection_command_ack', 64)
        self.command_sub = self.node.create_subscription(String, '/collection_command', self.command_received, 64)
        self.io_executor = SingleThreadedExecutor(context=self.context)
        self.io_executor.add_node(self.node)
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
        graph = '/Graph/ActionGraph_SO_ARM101/'
        connections = [(graph+'on_playback_tick.outputs:tick', graph+'articulation_controller.inputs:execIn')]
        connections += [(graph+'ros2_subscribe_joint_state.outputs:'+kind+'Command',
                         graph+'articulation_controller.inputs:'+kind+'Command')
                        for kind in ('position','velocity','effort')]
        if mode == 'collect':
            for source,target in connections:
                og.Controller.disconnect(source, target)
        elif mode == 'deploy':
            for source,target in connections:
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

    def resolved_shape_offsets(self):
        """Read effective PhysX shape values instead of USD default sentinels."""
        result = {}
        for name, view in self.bodies.items():
            contact = view._physics_view.get_contact_offsets().tolist()
            rest = view._physics_view.get_rest_offsets().tolist()
            if len(contact)!=1 or len(rest)!=1 or not contact[0] or len(contact[0])!=len(rest[0]):
                raise RuntimeError('Incomplete effective shape offsets: '+name)
            if not all(math.isfinite(v) for v in contact[0]+rest[0]):
                raise RuntimeError('Unresolved effective shape offsets: '+name)
            result[name] = dict(contact_m=contact[0],rest_m=rest[0])
        return result

    def snapshot(self):
        state_read_start=time.monotonic_ns()
        self.require_scene()
        positions = self.art.get_joint_positions()[0, self.indices]
        velocities = self.art.get_joint_velocities()[0, self.indices]
        targets = self.art.get_applied_actions().joint_positions[0, self.indices]
        if not all(np.all(np.isfinite(values)) for values in (positions, velocities, targets)):
            raise RuntimeError('Non-finite physical articulation feedback')
        bodies = {}
        for name, view in self.bodies.items():
            pos, quat = view.get_world_poses()
            velocity = view.get_velocities()[0]
            if not all(np.all(np.isfinite(value)) for value in (pos, quat, velocity)):
                raise RuntimeError('Non-finite physical body feedback: '+name)
            bodies[name] = dict(position=list(map(float, pos[0])), orientation=list(map(float, quat[0])),
                                velocity=list(map(float, velocity)))
        base_prim = self.stage.GetPrimAtPath('/World/base')
        if not base_prim.IsValid():
            raise RuntimeError('Physical base frame /World/base is missing')
        base = UsdGeom.Xformable(base_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rotation = base.ExtractRotationQuat()
        base_pose = dict(position=list(base.ExtractTranslation()),
                         orientation=[rotation.GetReal()]+list(rotation.GetImaginary()))
        return dict(wall_profile_native=dict(state_read_ms=(time.monotonic_ns()-state_read_start)/1e6),base_world_pose=base_pose, epoch=self.epoch, sim_ns=self.stamp(), physics_time_s=self.world.current_time,
                    state=list(map(float, positions)), velocity=list(map(float, velocities)),
                    drive_targets=list(map(float, targets)),
                    bodies=bodies, local_corners=self.local_corners, state_source='isaac_articulation')

    def publish(self):
        snapshot = self.snapshot()
        snapshot['wall_profile_native'].update(step_start_ns=getattr(self,'step_start_ns',None),
            step_end_ns=getattr(self,'step_end_ns',None),snapshot_ready_ns=time.monotonic_ns())
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
        if not matches(self.stamp(),observation_ns,'stamp_ns'):
            raise RuntimeError(f'Stale command observation: {observation_ns} != {self.stamp()}')
        values = np.asarray(command, dtype=np.float32)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError('Invalid six-joint command')
        self.art.set_joint_position_targets(values.reshape(1, 6), joint_indices=self.indices)
        self.art.set_joint_velocity_targets(np.zeros((1, 6), dtype=np.float32), joint_indices=self.indices)
        applied = self.art.get_applied_actions().joint_positions[0, self.indices]
        if not matches(list(map(float,applied)),list(map(float,values)),'target_rad'):
            raise RuntimeError('PhysX did not accept the executed targets')
        return dict(application_stamp_ns=self.stamp(), applied_command=list(map(float, applied)))

    def advance(self):
        if self.failure:
            raise RuntimeError(self.failure)
        if self.mode != 'collect' or self.busy:
            raise RuntimeError('An advancing frame is already pending or mode is not collect')
        self.start_time = self.world.current_time
        self.busy = True
        self.step_start_ns=time.monotonic_ns()
        self.timeline.play()
        return dict(start_ns=round(self.start_time * 1e9), target_ns=round((self.start_time + 1/30) * 1e9))

    def cancel(self):
        self.timeline.pause()
        self.busy = False
        self.epoch += 1
        self.invalid_through_ns = self.stamp()
        self.ledger = CommandLedger()
        return dict(held=True,failure=self.failure,epoch=self.epoch,invalid_through_ns=self.invalid_through_ns,
                    clock_baseline=dict(epoch=self.epoch,sim_ns=self.invalid_through_ns,physics_time_s=self.world.current_time))

    def command_received(self, message):
        payload = json.loads(message.data)
        try:
            if payload['epoch'] != self.epoch:
                raise ValueError('Cancelled command epoch')
            if payload['observation_stamp_ns'] <= self.invalid_through_ns:
                raise ValueError('Observation predates cancellation/reset barrier')
            if self.mode != 'collect':
                raise RuntimeError('Collection command refused in deployment mode')
            ack = self.ledger.process(payload, self.apply, self.advance)
        except Exception as exc:
            ack = dict(epoch=payload.get('epoch'), observation_stamp_ns=payload.get('observation_stamp_ns'), error=repr(exc))
            if payload.get('epoch') == self.epoch:
                self.failure = repr(exc)
                self.timeline.pause();self.busy = False
        self.ack_pub.publish(String(data=json.dumps(ack)))

    def after_update(self, event):
        try:
            if self.mode == 'collect':
                if self.busy and self.world.current_time > self.start_time and not matches(self.world.current_time,self.start_time,'world_s'):
                    self.step_end_ns=time.monotonic_ns()
                    self.timeline.pause()
                    delta = self.world.current_time - self.start_time
                    self.busy = False
                    if world_step_errors(self.world.current_time,self.start_time):
                        raise RuntimeError(f'One-frame step advanced {delta} sim seconds')
                    self.publish()
                # Drain clock/command callbacks in the same held frame. A short
                # wall deadline lets the expert respond without another render.
                deadline = time.monotonic() + (.005 if not self.busy else 0.)
                self.io_executor.spin_once(timeout_sec=0.)
                while not self.busy and time.monotonic() < deadline:
                    self.io_executor.spin_once(timeout_sec=max(0., deadline-time.monotonic()))
            elif self.timeline.is_playing():
                self.publish()
                self.io_executor.spin_once(timeout_sec=0.)
        except Exception as exc:
            self.timeline.pause();self.busy = False;self.failure = repr(exc)

    def reset_velocities(self):
        for view in self.bodies.values():
            view.set_velocities(np.zeros((1, 6), dtype=np.float32))
        return True

    def restore(self, snapshot, arm=False, restore_drives=False):
        if self.busy or self.timeline.is_playing():
            raise RuntimeError('Restore requires a held simulation')
        self.cancel()
        self.require_scene()
        for name, body in snapshot['bodies'].items():
            view = self.bodies[name]
            view.set_world_poses(np.asarray([body['position']], dtype=np.float32),
                                 np.asarray([body['orientation']], dtype=np.float32))
            view.set_velocities(np.zeros((1, 6), dtype=np.float32))
        if arm:
            self.art.set_joint_positions(np.asarray([snapshot['state']], dtype=np.float32), joint_indices=self.indices)
            self.art.set_joint_velocities(np.zeros((1, 6), dtype=np.float32), joint_indices=self.indices)
            self.apply(snapshot['drive_targets'] if restore_drives else snapshot['state'], self.stamp())
        actual = self.snapshot()
        error = max(abs(x-y) for name in snapshot['bodies']
                    for x,y in zip(snapshot['bodies'][name]['position'], actual['bodies'][name]['position']))
        orientation_error = max(min(max(abs(x-y) for x,y in zip(snapshot['bodies'][name]['orientation'], actual['bodies'][name]['orientation'])),
                                    max(abs(x+y) for x,y in zip(snapshot['bodies'][name]['orientation'], actual['bodies'][name]['orientation'])))
                                for name in snapshot['bodies'])
        if not matches(error,0.,'float32_readback'):
            raise RuntimeError(f'Initial pose restore differs by {error} m')
        if not matches(orientation_error,0.,'float32_readback') or any(not matches(body['velocity'],[0.]*6,'float32_readback') for body in actual['bodies'].values()):
            raise RuntimeError('Initial orientation/zero velocity restore failed')
        arm_error = velocity_error = 0.
        if arm:
            if len(actual['state']) != 6 or len(actual['velocity']) != 6 or len(snapshot['state']) != 6:
                raise RuntimeError('Initial arm restore requires all six measured joints')
            requested_drives=snapshot['drive_targets'] if restore_drives else snapshot['state']
            if not matches(actual['drive_targets'],requested_drives,'target_rad'):
                raise RuntimeError('Initial arm drive-target restore failed')
            arm_error = max(abs(x-y) for x,y in zip(snapshot['state'], actual['state']))
            velocity_error = max(abs(v) for v in actual['velocity'])
            # This read-back is immediate while held, before any constraint
            # impulses from a subsequent physics tick can affect velocity.
            if not np.isfinite(arm_error + velocity_error) or not matches(arm_error,0.,'target_rad') or not matches(velocity_error,0.,'float32_readback'):
                raise RuntimeError(f'Initial arm position/zero velocity restore failed: {arm_error} rad, {velocity_error} rad/s')
        return dict(epoch=self.epoch, invalid_through_ns=self.invalid_through_ns,arm_restored=bool(arm),
                    restored_drive_source='snapshot_drive_targets' if restore_drives else 'snapshot_state',
                    command_baseline=dict(epoch=self.epoch,sim_ns=actual['sim_ns'],physics_time_s=actual['physics_time_s'],drive_targets=list(actual['drive_targets'])),
                    max_position_error_m=error, max_orientation_error=orientation_error,
                    max_arm_position_error_rad=arm_error, max_arm_velocity_rad_s=velocity_error, snapshot=actual)


def install(mode='collect'):
    global _RUNTIME
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _LOADED_SHA256:
        raise RuntimeError('Runtime source changed; relaunch this instance before collection')
    if any(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest
           for name,digest in _LOADED_DEPENDENCIES.items()):
        raise RuntimeError('Runtime dependency changed; relaunch this instance before collection')
    if _RUNTIME is None:
        _RUNTIME = Runtime()
    if _RUNTIME.failure:
        raise RuntimeError('Fresh instance required: '+_RUNTIME.failure)
    _RUNTIME.cancel()
    _RUNTIME.set_mode(mode)
    import torch
    import carb
    return dict(mode=mode, joint_names=_RUNTIME.art.dof_names,
                epoch=_RUNTIME.epoch, invalid_through_ns=_RUNTIME.invalid_through_ns,
                loaded_sha256=_LOADED_SHA256,
                loaded_dependencies=_LOADED_DEPENDENCIES,
                cuda_device=torch.cuda.current_device(),cuda_device_count=torch.cuda.device_count(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),assigned_gpu_uuid=os.environ.get('SOARM_ASSIGNED_GPU_UUID'),
                shader_cache_path=carb.settings.get_settings().get('/rtx/shaderDb/shaderCachePath'),optix_cache_path=os.environ.get('OPTIX_CACHE_PATH'),
                kit_threads=carb.settings.get_settings().get('/plugins/carb.tasking.plugin/threadCount'),
                renderer_gpu=carb.settings.get_settings().get('/renderer/activeGpu'),
                physics_gpu=carb.settings.get_settings().get('/physics/cudaDevice'),
                measured_topic='/joint_states_measured', local_corners=_RUNTIME.local_corners,
                effective_shape_offsets=_RUNTIME.resolved_shape_offsets())


def runtime():
    if _RUNTIME is None:
        raise RuntimeError('Collection runtime is not installed')
    return _RUNTIME
