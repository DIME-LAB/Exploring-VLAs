"""Isaac main-thread transport: measured feedback and one-frame command handshake.

Loaded by the assigned Kit bridge, never by editing the running extension.
Collection disconnects only this scene's ROS follower executor. Commands write
its existing PhysX drive targets while paused, then exactly one render/physics
frame advances. Deployment restores the ROS follower and keeps measured feedback.
"""
import json
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import time
import uuid
from storage import atomic_json,confined
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
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim
from comparisons import matches
from tick_contract import world_step_errors
from contract import JOINTS
from protocol import CommandLedger
from finger_geometry import read_model
from miss_clearance import POLICY as MISS_POLICY,velocity_cap

_RUNTIME = None
_LOADED_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_LOADED_DEPENDENCIES = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ('contract.py', 'protocol.py','comparisons.py','tick_contract.py','finger_geometry.py','miss_clearance.py')}


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
        self.robot_base=RigidPrim('/World/SO_ARM101/base_link',name='collection_robot_base_readback',
            reset_xform_properties=False,prepare_contact_sensors=False)
        self.robot_base.initialize()
        fault_dir=confined(Path(os.environ['SOARM_INSTANCE_ROOT'])/'native-motion-faults');fault_dir.mkdir(parents=True,exist_ok=True)
        self.native_miss_fault_path=str(fault_dir/(uuid.uuid4().hex+'.json'))
        self._miss_cap=None;self._miss_saved_limits=None;self._miss_last_position=None
        self._miss_limits_dirty=False;self._miss_requested_cap=None;self._miss_motion_fault=None;self._miss_fault_durable=False
        self._miss_substeps=0;self._miss_max_velocity=0.;self._miss_max_displacement=0.
        self.world.add_physics_callback('collection_miss_velocity_check',self.check_miss_substep)
        self.finger_views={}
        for name in ('gripper_link','moving_jaw_so101_v1_link'):
            view=RigidPrim('/World/SO_ARM101/'+name,name='collection_finger_'+name,reset_xform_properties=False,prepare_contact_sensors=False)
            view.initialize();self.finger_views[name]=view
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

    def effective_finger_offsets(self):
        result={}
        for name,view in self.finger_views.items():
            contact=view._physics_view.get_contact_offsets().tolist();rest=view._physics_view.get_rest_offsets().tolist()
            if len(contact)!=1 or len(rest)!=1 or len(contact[0])!=4 or len(rest[0])!=4 or not np.isfinite(contact[0]+rest[0]).all():
                raise RuntimeError('Unverified effective finger offsets: '+name)
            result[name]=dict(contact_m=contact[0],rest_m=rest[0])
        return result

    def guarded_episode_offsets(self,name,epoch,stamp,expected_fingers,expected_block):
        if self.failure or getattr(self,'_miss_limits_dirty',False) or getattr(self,'_miss_cap',None) is not None:
            raise RuntimeError('Guarded episode requires restored, healthy native limits')
        if self.busy or self.timeline.is_playing() or epoch!=self.epoch or not matches(stamp,self.stamp(),'stamp_ns'):
            raise RuntimeError('Offset readback requires this epoch held at its anchor')
        fingers=self.effective_finger_offsets()
        view=self.bodies[name];contact=view._physics_view.get_contact_offsets().tolist();rest=view._physics_view.get_rest_offsets().tolist()
        if len(contact)!=1 or len(rest)!=1 or len(contact[0])!=1 or len(rest[0])!=1 or not np.isfinite(contact[0]+rest[0]).all():
            raise RuntimeError('Unverified guarded target offsets')
        block=dict(contact_m=contact[0],rest_m=rest[0])
        if fingers!=expected_fingers or block!=expected_block:raise RuntimeError('Native guarded-episode offsets changed after startup')
        return dict(epoch=epoch,sim_ns=self.stamp(),finger_offsets=fingers,block_offsets=block,source='fresh PhysX shape tensors')

    def publish_miss_fault(self):
        path=getattr(self,'native_miss_fault_path',None);fault=getattr(self,'_miss_motion_fault',None)
        if fault is None:raise RuntimeError('Native fault publication has no evidence')
        self._miss_fault_durable=False
        for _ in range(3):
            fault['publication_attempts']=fault.get('publication_attempts',0)+1
            fault['durability_unresolved']=False
            try:
                if not path:raise RuntimeError('Native fault publication path is unavailable')
                # atomic_json acknowledges only after file and directory fsync.
                atomic_json(Path(path),fault)
            except Exception as exc:
                fault['publication_error']=repr(exc)
                fault['publication_errors']=(fault.get('publication_errors',[])+[repr(exc)])[-8:]
                fault['durability_unresolved']=True
            else:
                self._miss_fault_durable=True
                return
        error='Native fault evidence is not durable after 3 publication attempts; remain held with unresolved limits'
        self.failure=(self.failure or fault['error'])+'; '+error
        raise RuntimeError(error)

    def restore_miss_limits(self):
        saved=getattr(self,'_miss_saved_limits',None)
        dirty=getattr(self,'_miss_limits_dirty',False)
        if saved is None and not dirty:
            if getattr(self,'_miss_cap',None) is not None:raise RuntimeError('Native cap has no saved original limits')
            self._miss_requested_cap=None
            return dict(on=False,values_rad_s=None)
        self._miss_limits_dirty=True;self._miss_cap=None
        if getattr(self,'_miss_motion_fault',None) is not None:
            # Neither cancellation nor a later reset may bypass this barrier.
            # Keep saved originals and unknown state until evidence is durable.
            self.publish_miss_fault()
        try:
            if saved is None:raise RuntimeError('Dirty native limits have no saved originals')
            self.art.set_max_joint_velocities(np.asarray([saved],dtype=np.float32),joint_indices=self.indices)
            actual=self.art.get_joint_max_velocities(joint_indices=self.indices)[0]
            if not matches(actual.tolist(),saved,'float32_readback'):raise RuntimeError('Miss velocity limit restore readback failed')
        except Exception as exc:
            if getattr(self,'_miss_motion_fault',None) is None:
                try:stamp=self.stamp()
                except Exception:stamp=None
                self._miss_motion_fault=dict(epoch=getattr(self,'epoch',None),sim_ns=stamp,error=repr(exc),source='limit rollback',
                    saved_original_limits_rad_s=saved,dirty=True,samples=list(getattr(self,'_miss_samples',[])))
            self._miss_motion_fault['rollback_error']=repr(exc);self._miss_motion_fault['terminal']=True
            self.failure='Native miss-limit rollback failed: '+repr(exc)
            self.publish_miss_fault()
            raise RuntimeError(self.failure) from exc
        self._miss_cap=None;self._miss_saved_limits=None;self._miss_limits_dirty=False;self._miss_last_position=None;self._miss_requested_cap=None
        if getattr(self,'_miss_motion_fault',None) is not None:
            self._miss_motion_fault['rollback_verified']=True;self.publish_miss_fault()
        return dict(on=False,values_rad_s=None)

    def fail_miss_motion(self,exc):
        original=repr(exc);self.failure=self.failure or original
        if getattr(self,'_miss_motion_fault',None) is None:
            try:stamp=self.stamp()
            except Exception:stamp=None
            self._miss_motion_fault=dict(epoch=getattr(self,'epoch',None),sim_ns=stamp,error=original,terminal=True,
                cap=getattr(self,'_miss_cap',None),requested_cap=getattr(self,'_miss_requested_cap',None),
                saved_original_limits_rad_s=getattr(self,'_miss_saved_limits',None),
                dirty=getattr(self,'_miss_limits_dirty',False),samples=list(getattr(self,'_miss_samples',[])))
        try:self.timeline.pause();self.busy=False
        except Exception as pause_error:
            self.failure=original+'; native pause failed: '+repr(pause_error)
            if getattr(self,'_miss_saved_limits',None) is not None:self._miss_limits_dirty=True;self._miss_cap=None
            self._miss_motion_fault['pause_error']=repr(pause_error)
            try:self.publish_miss_fault()
            except Exception:pass  # Publication stays unresolved; no native rollback is attempted.
            return RuntimeError(self.failure)
        try:
            if getattr(self,'_miss_saved_limits',None) is None and not getattr(self,'_miss_limits_dirty',False):
                self.publish_miss_fault()
            self.restore_miss_limits()
        except Exception as rollback:self.failure=original+'; '+repr(rollback)
        else:self.failure=original
        return RuntimeError(self.failure)

    def stationary_target(self,cap,check=True):
        name=cap.get('stationary_target')
        if not name or name not in self.bodies:raise RuntimeError('Native stationary-target identity is required')
        velocity=np.asarray(self.bodies[name].get_velocities()[0],dtype=float)
        if velocity.shape!=(6,) or not np.isfinite(velocity).all():raise RuntimeError('Native target velocity is unverified')
        linear=float(np.linalg.norm(velocity[:3]));angular=float(np.linalg.norm(velocity[3:]))
        passed=linear<=MISS_POLICY['stationary_linear_m_s'] and angular<=MISS_POLICY['stationary_angular_rad_s']
        result=dict(passed=passed,target=name,linear_m_s=linear,angular_rad_s=angular,velocity=velocity.tolist(),
            linear_limit_m_s=MISS_POLICY['stationary_linear_m_s'],angular_limit_rad_s=MISS_POLICY['stationary_angular_rad_s'])
        if check and not passed:raise RuntimeError('Guarded target is moving: '+repr(result))
        return result

    def set_miss_velocity_cap(self,cap):
        if self.busy or self.timeline.is_playing():raise self.fail_miss_motion(RuntimeError('Velocity-cap handover requires a held simulation'))
        if cap is None:return self.restore_miss_limits()
        if self.failure:raise RuntimeError('Native runtime is terminal: '+str(self.failure))
        try:
            self._miss_requested_cap=dict(cap)
            if cap!=velocity_cap(cap.get('phase'),cap.get('stationary_target')):raise ValueError('Unknown miss velocity-cap policy')
            self.stationary_target(cap)
            measured=self.art.get_joint_velocities()[0,self.indices]
            if not np.isfinite(measured).all() or np.any(abs(measured)>np.asarray(cap['values_rad_s'])+MISS_POLICY['joint_numeric_allowance_rad']):
                raise RuntimeError('Measured velocity exceeds requested miss cap before advance')
            saved=getattr(self,'_miss_saved_limits',None)
            if saved is not None and getattr(self,'_miss_limits_dirty',False) and getattr(self,'_miss_cap',None) is None:
                raise RuntimeError('Unresolved partial native cap installation')
            if saved is None:
                if getattr(self,'_miss_limits_dirty',False):raise RuntimeError('Unresolved dirty native limits')
                saved=self.art.get_joint_max_velocities(joint_indices=self.indices)[0]
                if saved.shape!=(6,) or not np.isfinite(saved).all() or np.any(saved<=0):raise RuntimeError('Original joint velocity limits are unverified')
                self._miss_saved_limits=saved.tolist()
            # Save and mark dirty BEFORE the first native write can partly succeed.
            # Never replace saved originals while any install/rollback is unresolved.
            self._miss_limits_dirty=True;self._miss_cap=None;self._miss_requested_cap=dict(cap)
            self.art.set_max_joint_velocities(np.asarray([cap['values_rad_s']],dtype=np.float32),joint_indices=self.indices)
            actual=self.art.get_joint_max_velocities(joint_indices=self.indices)[0]
            if not matches(actual.tolist(),cap['values_rad_s'],'float32_readback'):raise RuntimeError('Native miss velocity limit readback failed')
            self._miss_cap=dict(cap);self._miss_last_position=self.art.get_joint_positions()[0,self.indices].copy()
            self._miss_frame_start=self._miss_last_position.copy();self._miss_samples=[]
            self._miss_substeps=0;self._miss_max_velocity=0.;self._miss_max_displacement=0.
            return dict(cap)
        except Exception as exc:raise self.fail_miss_motion(exc) from exc

    def check_miss_substep(self,step,source='pre_physics_callback'):
        try:
            cap=getattr(self,'_miss_cap',None)
            if cap is None:
                if getattr(self,'_miss_limits_dirty',False):raise RuntimeError('Cannot advance with unverified dirty native limits')
                return
            limits=self.art.get_joint_max_velocities(joint_indices=self.indices)[0]
            velocity=self.art.get_joint_velocities()[0,self.indices]
            position=self.art.get_joint_positions()[0,self.indices]
            try:target=self.stationary_target(cap,check=False)
            except Exception as error:target=dict(passed=False,error=repr(error))
            bound=np.asarray(cap['values_rad_s']);eps=MISS_POLICY['joint_numeric_allowance_rad']
            delta=abs(position-self._miss_last_position);frame_delta=abs(position-self._miss_frame_start)
            local_excess=delta-bound*float(step)
            frame_excess=frame_delta-bound*MISS_POLICY['step_s']
            values=lambda array:[float(v) if math.isfinite(float(v)) else None for v in array]
            sample=dict(index=self._miss_substeps,source=source,step_s=float(step),joint_names=list(JOINTS),positions_rad=values(position),
                measured_velocity_rad_s=values(velocity),native_limits_rad_s=values(limits),
                local_excess_rad=values(local_excess),frame_excess_over_cap_rad=values(frame_excess),
                certificate_excess_rad=values(frame_excess-MISS_POLICY['reachable_numeric_allowance_rad']),
                certificate_pad_rad=MISS_POLICY['reachable_numeric_allowance_rad'],local_budget_rad=eps,stationary_target=target)
            self._miss_samples.append(sample)
            if not target['passed']:raise RuntimeError('Guarded target is not stationary: '+repr(target))
            if not matches(limits.tolist(),cap['values_rad_s'],'float32_readback') or not np.isfinite([velocity,position]).all() or np.any(abs(velocity)>bound+eps):
                raise RuntimeError('Native recovery velocity constraint failed')
            if np.any(local_excess>eps):raise RuntimeError('Native recovery substep residual exceeded its 1 urad budget')
            if np.any(frame_excess>MISS_POLICY['reachable_numeric_allowance_rad']):raise RuntimeError('Native recovery displacement exceeded the 5 urad certificate pad')
            self._miss_last_position=position.copy();self._miss_substeps+=1
            self._miss_max_velocity=max(self._miss_max_velocity,float(max(abs(velocity))))
            self._miss_max_displacement=max(self._miss_max_displacement,float(max(delta)))
        except Exception as exc:
            raise self.fail_miss_motion(exc) from exc

    def miss_cap_state(self):
        cap=getattr(self,'_miss_cap',None)
        if cap is None and getattr(self,'_miss_limits_dirty',False):
            return dict(on=None,values_rad_s=None,status='dirty_unverified',saved_original_limits_rad_s=self._miss_saved_limits,failure=self.failure)
        return dict(cap,substep_checks=self._miss_substeps,max_measured_velocity_rad_s=self._miss_max_velocity,
                    max_substep_displacement_rad=self._miss_max_displacement,substep_samples=list(self._miss_samples)) if cap else dict(on=False,values_rad_s=None)

    def snapshot(self):
        state_read_start=time.monotonic_ns()
        if getattr(self,'_miss_cap',None):self.check_miss_substep(1/120,source='final_snapshot')
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
        base_pos,base_quat=self.robot_base.get_world_poses()
        if not all(np.all(np.isfinite(v)) for v in (base_pos,base_quat)):raise RuntimeError('Non-finite native robot base pose')
        robot_base=dict(position=list(map(float,base_pos[0])),orientation=list(map(float,base_quat[0])))
        return dict(miss_velocity_cap=self.miss_cap_state(),wall_profile_native=dict(state_read_ms=(time.monotonic_ns()-state_read_start)/1e6),base_world_pose=base_pose,
                    robot_base_world_pose=robot_base,robot_base_pose_source='physx_rigid_body',epoch=self.epoch, sim_ns=self.stamp(), physics_time_s=self.world.current_time,
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

    def start_miss_hold(self,epoch,stamp,cap,state,drive_targets):
        try:
            if epoch!=self.epoch or stamp<=self.invalid_through_ns or not matches(stamp,self.stamp(),'stamp_ns'):
                raise RuntimeError('Stale miss kickoff identity')
            actual=self.art.get_joint_positions()[0,self.indices]
            drives=self.art.get_applied_actions().joint_positions[0,self.indices]
            if not matches(actual.tolist(),state,'target_rad') or not matches(drives.tolist(),drive_targets,'target_rad'):
                raise RuntimeError('Miss kickoff state/drive baseline differs')
            applied=self.set_miss_velocity_cap(cap)
            result=self.advance()
            return dict(epoch=epoch,observation_stamp_ns=stamp,command_miss_velocity_cap=applied,advance=result)
        except Exception as exc:
            if self.failure:raise
            raise self.fail_miss_motion(exc) from exc

    def advance(self):
        if self.failure:
            raise RuntimeError(self.failure)
        if self.mode != 'collect' or self.busy:
            raise RuntimeError('An advancing frame is already pending or mode is not collect')
        if getattr(self,'_miss_cap',None) or getattr(self,'_miss_limits_dirty',False):self.check_miss_substep(1/120,source='before_advance')
        self.start_time = self.world.current_time
        self.busy = True
        self.step_start_ns=time.monotonic_ns()
        self.timeline.play()
        return dict(start_ns=round(self.start_time * 1e9), target_ns=round((self.start_time + 1/30) * 1e9))

    def cancel(self):
        try:
            self.timeline.pause()
            self.busy = False
            self.set_miss_velocity_cap(None)
            self.epoch += 1
            self.invalid_through_ns = self.stamp()
            self.ledger = CommandLedger()
            return dict(held=True,failure=self.failure,epoch=self.epoch,invalid_through_ns=self.invalid_through_ns,
                        clock_baseline=dict(epoch=self.epoch,sim_ns=self.invalid_through_ns,physics_time_s=self.world.current_time),
                        miss_motion_fault=getattr(self,'_miss_motion_fault',None))
        except Exception as exc:
            if self.failure:raise
            raise self.fail_miss_motion(exc) from exc

    def command_received(self, message):
        payload = json.loads(message.data)
        try:
            if self.failure:raise RuntimeError('Native runtime is terminal: '+str(self.failure))
            if payload['epoch'] != self.epoch:
                raise ValueError('Cancelled command epoch')
            if payload['observation_stamp_ns'] <= self.invalid_through_ns:
                raise ValueError('Observation predates cancellation/reset barrier')
            if self.mode != 'collect':
                raise RuntimeError('Collection command refused in deployment mode')
            def apply(command,stamp):
                cap=payload.get('miss_velocity_cap')
                if cap is None and getattr(self,'_miss_cap',None) is None:return self.apply(command,stamp)
                applied=self.set_miss_velocity_cap(cap)
                return dict(self.apply(command,stamp),command_miss_velocity_cap=applied)
            ack = self.ledger.process(payload, apply, self.advance)
        except Exception as exc:
            ack = dict(epoch=payload.get('epoch'), observation_stamp_ns=payload.get('observation_stamp_ns'), error=repr(exc))
            if payload.get('epoch') == self.epoch:
                if self.failure is None:self.fail_miss_motion(exc)
                else:self.timeline.pause();self.busy=False
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
        if self.failure:raise RuntimeError('Fresh instance required after native failure: '+str(self.failure))
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


def physics_mode(stage):
    path='/physicsScene';prim=stage.GetPrimAtPath(path)
    if not prim.IsValid() or not prim.HasAPI(PhysxSchema.PhysxSceneAPI):raise RuntimeError('Physics dynamics are unverified')
    enabled=PhysxSchema.PhysxSceneAPI(prim).GetEnableGPUDynamicsAttr().Get()
    if enabled is not False:raise RuntimeError('CPU dynamics require an explicit false enableGPUDynamics value: '+repr(enabled))
    return dict(dynamics='cpu',enable_gpu_dynamics=False,scene_path=path)


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
    physics=physics_mode(_RUNTIME.stage)
    helper=Path(__file__).resolve().parent.parent/'collect_inproc/articulation_audit.py'
    spec=importlib.util.spec_from_file_location('collection_articulation_audit',helper)
    audit_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit_module)
    articulation=audit_module.readback(_RUNTIME.stage,_RUNTIME.art)
    articulation['helper_sha256']=hashlib.sha256(helper.read_bytes()).hexdigest()
    if mode=='collect' and articulation['fixed_base'] is not True:raise RuntimeError('Collection requires verified fixed_base=true: '+repr(articulation))
    return dict(native_miss_fault_path=_RUNTIME.native_miss_fault_path,mode=mode, physics_config=physics,joint_names=_RUNTIME.art.dof_names,articulation_audit=articulation,
                joint_index_mapping=dict(zip(JOINTS,_RUNTIME.indices)),finger_model=read_model(_RUNTIME.stage,effective_finger_offsets=_RUNTIME.effective_finger_offsets()),
                epoch=_RUNTIME.epoch, invalid_through_ns=_RUNTIME.invalid_through_ns,
                loaded_sha256=_LOADED_SHA256,
                loaded_dependencies=_LOADED_DEPENDENCIES,
                cuda_device=torch.cuda.current_device(),cuda_device_count=torch.cuda.device_count(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),assigned_gpu_uuid=os.environ.get('SOARM_ASSIGNED_GPU_UUID'),
                shader_cache_path=carb.settings.get_settings().get('/rtx/shaderDb/shaderCachePath'),optix_cache_path=os.environ.get('OPTIX_CACHE_PATH'),
                kit_threads=carb.settings.get_settings().get('/plugins/carb.tasking.plugin/threadCount'),
                renderer_gpu=carb.settings.get_settings().get('/renderer/activeGpu'),
                renderer_multi_gpu=carb.settings.get_settings().get('/renderer/multiGpu/enabled'),
                physics_gpu=carb.settings.get_settings().get('/physics/cudaDevice'),
                physics_dynamics=physics['dynamics'],
                measured_topic='/joint_states_measured', local_corners=_RUNTIME.local_corners,
                effective_shape_offsets=_RUNTIME.resolved_shape_offsets())


def runtime():
    if _RUNTIME is None:
        raise RuntimeError('Collection runtime is not installed')
    return _RUNTIME
