"""Full collector attempt with fake ROS/Kit transports; never imports Isaac or CUDA.

The robot/controller/recorder code is real. Physics is an ideal bounded position
adapter, and MoveIt validity answers are deliberately collision-free. This is
pipeline qualification, not a physical demonstration or solver experiment.
"""
import copy,json,hashlib,os,sys,threading,time,types,traceback
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from storage import atomic_json,confined,prepare_environment
from contract import JOINTS
from pause_barrier import pause_committed
from protocol import CommandLedger
from source_provenance import loaded_sources,source_digest

HERE=Path(__file__).resolve().parent


class DeferredTimeline:
    def __init__(self):self.playing=False;self.pending=False;self.commits=0
    def play(self):self.playing=True
    def pause(self):self.pending=True
    def is_playing(self):return self.playing
    def commit(self):
        # One explicit state update, without a physics step.
        self.commits+=1
        if self.pending:self.playing=False;self.pending=False


def run_full(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.context import Context
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.callback_groups import ReentrantCallbackGroup
    from std_msgs.msg import String
    from sensor_msgs.msg import JointState,Image
    from rosgraph_msgs.msg import Clock
    from moveit_msgs.msg import PlanningScene,AllowedCollisionEntry,CollisionObject
    from moveit_msgs.srv import ApplyPlanningScene,GetPlanningScene,GetStateValidity,GetMotionPlan
    from trajectory_msgs.msg import JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    import collect as implementation
    from test_review_regressions import load_class
    from so_arm101_control.compute_workspace import forward_kinematics_full
    from ros_lifecycle import retire_executor,shutdown_context
    from cup_geometry import CUP
    from pose_math import rotate

    runtime_root=prepare_environment(args.instance)
    from run_smoke import InstanceClaim
    claim=InstanceClaim(runtime_root)
    output=confined(args.output);output.mkdir(parents=True,exist_ok=False)
    (output/'TEST_ONLY_NOT_TRAINING_DATA').write_text('CPU fake transports; no native physics qualification.\n')
    fixture_path=HERE/'fixtures/regrasp-boot.json';fixture=json.loads(fixture_path.read_text())
    reference=output/'reference';reference.mkdir()
    atomic_json(reference/'run.json',dict(baseline=fixture['baseline'],baseline_cup_layout=fixture['baseline_cup_layout']))
    (reference/'attempt-0000').mkdir();replay=reference/'attempt-0000/receipt.json';atomic_json(replay,fixture)
    clock_topic='/offline_regrasp_clock_'+output.name.replace('-','_')
    clock_args=['--ros-args','-r','/clock:='+clock_topic]
    rclpy.init(args=clock_args);context=Context();context.init(args=clock_args)
    result=dict(passed=False,scope=__doc__,fixture_sha256=source_digest(fixture_path),domain=90+args.instance,
        physical_qualification=False,isaac_started=False,gpu_started=False,inject_native_fault=args.inject_native_fault)
    native=Node('fake_regrasp_transports',context=context);group=ReentrantCallbackGroup()
    clock_pub=native.create_publisher(Clock,'/clock',64)
    cameras={key:native.create_publisher(Image,topic,64) for key,topic in
             (('wrist','/wrist_camera_rgb_sim'),('top','/workspace_camera_sim'))}
    stop=threading.Event();wake=threading.Event();lock=threading.RLock();collector=None
    Runtime,scope=load_class('collection_runtime.py','Runtime')
    scope.update(String=String,JointState=JointState,atomic_json=atomic_json,pause_committed=pause_committed)
    r=object.__new__(Runtime);r.world=NS(current_time=1.);r.stamp=lambda:round(r.world.current_time*1e9)
    r.timeline=DeferredTimeline();r.mode='collect';r.busy=False;r.failure=None;r.epoch=0;r.invalid_through_ns=-1
    r.ledger=CommandLedger();r.indices=list(range(6));r.node=native;r.last_snapshot=None
    r._miss_cap=None;r._miss_saved_limits=None;r._miss_limits_dirty=False;r._miss_requested_cap=None
    r._miss_motion_fault=None;r._miss_fault_durable=False;r._miss_last_position=None
    r._miss_substeps=0;r._miss_max_velocity=0.;r._miss_max_displacement=0.;r._miss_samples=[]
    journal_dir=runtime_root/'native-motion-faults';journal_dir.mkdir(exist_ok=True)
    r.native_miss_fault_path=str(journal_dir/(output.name+'.json'))
    if Path(r.native_miss_fault_path).exists():raise RuntimeError('Fresh native test journal required')
    initial=copy.deepcopy(fixture['initial_physical'])
    q=np.asarray([initial['state']],dtype=np.float32);v=np.zeros((1,6),np.float32)
    targets=np.asarray([initial['drive_targets']],dtype=np.float32);limits=np.full((1,6),10.,np.float32)
    writes=[]
    def set_limits(values,joint_indices):
        if r.timeline.is_playing():raise AssertionError('Native limit write preceded pause commit')
        limits[:,joint_indices]=values;writes.append(dict(values=limits[0].tolist(),epoch=r.epoch,stamp=r.stamp(),held=True))
    def set_q(values,joint_indices):q[:,joint_indices]=values
    def set_v(values,joint_indices):v[:,joint_indices]=values
    r.art=NS(dof_names=list(JOINTS),get_joint_positions=lambda:q.copy(),get_joint_velocities=lambda:v.copy(),
        get_joint_max_velocities=lambda joint_indices:limits[:,joint_indices].copy(),set_max_joint_velocities=set_limits,
        set_joint_positions=set_q,set_joint_velocities=set_v,
        set_joint_position_targets=lambda values,joint_indices:targets.__setitem__((slice(None),joint_indices),values),
        set_joint_velocity_targets=lambda *a,**k:None,get_applied_actions=lambda:NS(joint_positions=targets.copy()))
    body_data=copy.deepcopy(initial['bodies']);r.bodies={}
    def shape_view(values):
        return NS(get_contact_offsets=lambda:np.asarray([values['contact_m']],np.float32),
                  get_rest_offsets=lambda:np.asarray([values['rest_m']],np.float32))
    for name in body_data:
        def set_pose(position,orientation,name=name):
            body_data[name].update(position=position[0].tolist(),orientation=orientation[0].tolist())
        def set_velocity(values,name=name):body_data[name]['velocity']=values[0].tolist()
        r.bodies[name]=NS(get_world_poses=lambda name=name:(np.asarray([body_data[name]['position']],np.float32),np.asarray([body_data[name]['orientation']],np.float32)),
            get_velocities=lambda name=name:np.asarray([body_data[name]['velocity']],np.float32),set_world_poses=set_pose,
            set_velocities=set_velocity,_physics_view=shape_view(fixture['shape_offsets'][name]))
    r.finger_views={name:NS(_physics_view=shape_view(value)) for name,value in fixture['finger_model']['effective_finger_offsets'].items()}
    r.local_corners=initial['local_corners'];r.require_scene=lambda:True
    r.snapshot=lambda:dict(epoch=r.epoch,sim_ns=r.stamp(),physics_time_s=r.world.current_time,state=q[0].tolist(),velocity=v[0].tolist(),
        drive_targets=targets[0].tolist(),bodies=copy.deepcopy(body_data),local_corners=copy.deepcopy(r.local_corners),
        base_world_pose=copy.deepcopy(initial['base_world_pose']),robot_base_world_pose=copy.deepcopy(initial['robot_base_world_pose']),
        robot_base_pose_source='physx_rigid_body',state_source='isaac_articulation',miss_velocity_cap=r.miss_cap_state(),
        wall_profile_native=dict(state_read_ms=0.))
    # Preserve the native final-substep check before every published snapshot.
    plain_snapshot=r.snapshot
    def checked_snapshot():
        if r._miss_cap is not None:r.check_miss_substep(1/120,source='final_snapshot')
        return plain_snapshot()
    r.snapshot=checked_snapshot
    r.joint_pub=native.create_publisher(JointState,'/joint_states_measured',64)
    reference_pub=native.create_publisher(JointState,'/joint_states',64)
    r.tick_pub=native.create_publisher(String,'/collection_tick',64)
    r.ack_pub=native.create_publisher(String,'/collection_command_ack',64)
    # Models the controller feedback subscriber alongside the recorder.
    feedback=native.create_subscription(JointState,'/joint_states_measured',lambda msg:None,64)
    native_publish=r.publish
    def publish():
        deadline=time.monotonic()+5
        while min([clock_pub.get_subscription_count(),r.joint_pub.get_subscription_count(),r.tick_pub.get_subscription_count(),
                   *[p.get_subscription_count() for p in cameras.values()]])<1:
            if time.monotonic()>deadline:raise TimeoutError('Fake measured-state/camera discovery did not finish')
            stop.wait(.001)
        message=Clock();message.clock.sec,message.clock.nanosec=divmod(r.stamp(),10**9);clock_pub.publish(message)
        reference=JointState();reference.header.stamp.sec,reference.header.stamp.nanosec=divmod(r.stamp(),10**9)
        reference.name=list(JOINTS);reference.position=targets[0].tolist();reference.velocity=[0.]*6;reference_pub.publish(reference)
        snapshot=native_publish()
        for pub in cameras.values():
            image=Image();image.header.stamp.sec,image.header.stamp.nanosec=divmod(r.stamp()+167,10**9)
            image.encoding='rgb8';image.width=640;image.height=480;image.step=1920;image.data=bytes(640*480*3);pub.publish(image)
        return snapshot
    r.publish=publish
    r.io_executor=NS(spin_once=lambda timeout_sec:stop.wait(min(timeout_sec,.0005)) if timeout_sec else None)
    native_advance=r.advance
    def advance():answer=native_advance();wake.set();return answer
    r.advance=advance
    def applied(message):
        with lock:r.command_received(message)
    subscription=native.create_subscription(String,'/collection_command',applied,64)
    scene=PlanningScene();scene.robot_model_name='so_arm101';scene_lock=threading.RLock();attached=[None]
    def op(value):return int.from_bytes(value,'little') if isinstance(value,bytes) else int(value)
    def tcp_transform():return forward_kinematics_full(q[0,:5].tolist())
    def attached_removed(name):
        if name!=fixture['params']['object']:return
        cup=body_data['cup_'+fixture['params']['color']];half=float(np.ptp(np.asarray(r.local_corners[name]),axis=0)[2])/2
        offset=rotate(cup['orientation'],[0.,0.,CUP['floor_top_m']+half])
        body_data[name].update(position=[a+b for a,b in zip(cup['position'],offset)],orientation=list(cup['orientation']),velocity=[0.]*6)
    def apply_scene(request,response):
        time.sleep(.001)
        with scene_lock:
            update=request.scene;world={o.id:o for o in scene.world.collision_objects};acm=scene.allowed_collision_matrix
            objects={o.object.id:o for o in scene.robot_state.attached_collision_objects}
            for obj in update.robot_state.attached_collision_objects:
                name=obj.object.id.removeprefix('lego_')
                if op(obj.object.operation)==op(CollisionObject.REMOVE):
                    objects.pop(obj.object.id,None)
                    with lock:attached[0]=None;attached_removed(name)
                else:
                    objects[obj.object.id]=copy.deepcopy(obj)
                    pose=(obj.object.primitive_poses or obj.object.mesh_poses)[0]
                    if collector is None:raise AssertionError('Attachment before collector construction')
                    matrix=collector.node._pose7_to_mat(pose.position.x,pose.position.y,pose.position.z,
                        pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w)
                    with lock:attached[0]=(name,matrix)
            for obj in update.world.collision_objects:
                if op(obj.operation)==op(CollisionObject.REMOVE):
                    world.pop(obj.id,None)
                    if obj.id in acm.entry_names:
                        i=acm.entry_names.index(obj.id);names=list(acm.entry_names);names.pop(i)
                        values=[AllowedCollisionEntry(enabled=[v for j,v in enumerate(row.enabled) if j!=i]) for k,row in enumerate(acm.entry_values) if k!=i]
                        acm.entry_names=names;acm.entry_values=values
                else:world[obj.id]=copy.deepcopy(obj)
            scene.world.collision_objects=list(world.values());scene.robot_state.attached_collision_objects=list(objects.values())
            if update.allowed_collision_matrix.entry_names:
                scene.allowed_collision_matrix=copy.deepcopy(update.allowed_collision_matrix)
                for i,row in enumerate(scene.allowed_collision_matrix.entry_values):row.enabled[i]=False
        response.success=True;return response
    def get_scene(request,response):
        with scene_lock:
            response.scene=copy.deepcopy(scene)
            with lock:
                response.scene.robot_state.joint_state.name=list(JOINTS)
                response.scene.robot_state.joint_state.position=q[0].tolist()
        return response
    validity_calls=[]
    def valid(request,response):validity_calls.append(1);response.valid=True;return response
    def plan(request,response):
        wanted=dict(zip(JOINTS,q[0].tolist()))
        for constraint in request.motion_plan_request.goal_constraints[0].joint_constraints:wanted[constraint.joint_name]=constraint.position
        trajectory=response.motion_plan_response.trajectory.joint_trajectory;trajectory.joint_names=list(JOINTS[:5])
        trajectory.points=[JointTrajectoryPoint(positions=q[0,:5].tolist(),time_from_start=Duration()),
            JointTrajectoryPoint(positions=[wanted[n] for n in JOINTS[:5]],time_from_start=Duration(sec=2))]
        response.motion_plan_response.error_code.val=1;return response
    services=[native.create_service(t,n,f,callback_group=group) for t,n,f in
              ((ApplyPlanningScene,'/apply_planning_scene',apply_scene),(GetPlanningScene,'/get_planning_scene',get_scene),
               (GetStateValidity,'/check_state_validity',valid),(GetMotionPlan,'/plan_kinematic_path',plan))]
    executor=MultiThreadedExecutor(num_threads=4,context=context);executor.add_node(native)
    ros_thread=threading.Thread(target=executor.spin);ros_thread.start()
    def physics():
        try:
            while not stop.is_set():
                if not wake.wait(.1):continue
                wake.clear()
                with lock:
                    if not r.busy:continue
                    for _ in range(4):
                        r.check_miss_substep(1/120)
                        delta=np.clip(targets-q,-limits/120,limits/120);q[:]=q+delta;v[:]=delta*120
                        r.world.current_time+=1/120
                        if attached[0] is not None and collector is not None:
                            name,offset=attached[0];pose=collector.node._mat_to_pose7(tcp_transform()@offset)
                            body_data[name].update(position=pose[:3],orientation=[pose[6],*pose[3:6]],velocity=[0.]*6)
                    r.after_update(None)
                    if r.failure:raise RuntimeError(r.failure)
        except BaseException as exc:
            result['fake_physics_error']=repr(exc)
            if collector is not None:
                collector.failure=exc
                with collector.tick_condition:collector.tick_condition.notify_all()
    physics_thread=threading.Thread(target=physics);physics_thread.start()
    proxy=types.ModuleType('collection_runtime');proxy.__file__=str(HERE/'collection_runtime.py');proxy.runtime=lambda:r
    def install(mode='collect'):
        r.cancel();r.mode=mode
        return dict(epoch=r.epoch,invalid_through_ns=r.invalid_through_ns,mode=mode,native_miss_fault_path=r.native_miss_fault_path,
            local_corners=r.local_corners,effective_shape_offsets=r.resolved_shape_offsets(),finger_model=fixture['finger_model'],
            articulation_audit=fixture['articulation_audit'],loaded_sha256=source_digest(HERE/'collection_runtime.py'),loaded_dependencies=loaded_sources(HERE),
            renderer_gpu=0,renderer_multi_gpu=False,physics_gpu=0,cuda_device=0,cuda_device_count=1,cuda_visible_devices='fake-cpu-only',
            physics_dynamics='cpu',physics_config=dict(dynamics='cpu',test_double=True))
    proxy.install=install
    def mcp(tool,params=None):
        if tool!='run_python':raise AssertionError('Replay needs no vendor reset tool: '+tool)
        code=(params or {})['code']
        if 'result=configure(30)' in code:return dict(fps=30,preset='fake unpaced')
        if 'rateLimitFrequency' in code:return True
        with lock:
            namespace={};exec(code,namespace);return namespace.get('result')
    try:
        config=NS(output=output/'run',episodes=1,max_attempts=1,resume=False,seed=340000,regrasp_only=True,normal_only=False,
            replay_receipt=replay,wait_rtf_floor=.1,episode_deadline=180,diagnostic=True,instance=args.instance)
        def gpu(self):
            self.last_gpu_footprint=dict(passed=True,test_double=True,scope='No GPUs in this test');return self.last_gpu_footprint
        with patch.dict(sys.modules,{'collection_runtime':proxy}),patch.object(implementation,'mcp',mcp),\
             patch.object(implementation,'gpu_uuid',lambda index:'fake-cpu-only'),patch.dict(os.environ,{'SOARM_RENDERER_GPU_INDEX':'0'}),\
             patch.object(implementation.Collector,'gpu_footprint',gpu):
            collector=implementation.Collector(config)
            collector.node.objects_bbox={n:dict(zip(('sx','sy','sz'),np.ptp(np.asarray(c),axis=0).tolist())) for n,c in r.local_corners.items()}
            def transform(*a,**k):
                pose=collector.node._mat_to_pose7(tcp_transform())
                return NS(transform=NS(translation=NS(x=pose[0],y=pose[1],z=pose[2]),rotation=NS(x=pose[3],y=pose[4],z=pose[5],w=pose[6])))
            collector.node._tf_buffer.lookup_transform=transform
            original_cycle=collector.node.run_cycle
            def corrective(name):
                result['corrective_handover']=dict(limits=limits[0].tolist(),cap=r.miss_cap_state(),held=not r.timeline.is_playing())
                assert r.miss_cap_state()==dict(on=False,values_rad_s=None) and np.all(limits==10.)
                if args.inject_native_fault:raise r.fail_miss_motion(RuntimeError('Injected late fake native fault'))
                return original_cycle(name)
            collector.node.run_cycle=corrective
            try:collector.run()
            except Exception as exc:
                result['run_error']=repr(exc);result['run_traceback']=traceback.format_exc()
            finally:
                try:collector.close()
                except Exception as exc:result['close_error']=repr(exc)
        receipt_path=output/'run/attempt-0000/receipt.json'
        if not receipt_path.exists():raise RuntimeError('Fake startup failed before an attempt: '+result.get('run_error','No receipt'))
        receipt=json.loads(receipt_path.read_text())
        report=receipt['recovery_report'];stages=[s['stage'] for s in report['steps']]
        expected=['regrasp_miss_open','regrasp_miss_approach','regrasp_miss_descent','regrasp_miss_close','regrasp_miss_lift','regrasp_reopen']
        assert stages==expected
        assert report['steps'][0]['completion']['already_at_target'] and report['kickoffs'][0]['ack']['held']
        assert result['corrective_handover']['cap']['on'] is False
        assert not receipt.get('finalization_errors'),receipt.get('finalization_errors')
        if args.inject_native_fault:
            assert receipt['accepted'] is False and receipt['native_miss_faults']
        else:
            assert receipt['accepted'] is True,receipt.get('error')
            assert 'run_error' not in result and 'close_error' not in result,result
            from recovery import receipt_error
            rows=[json.loads(line) for line in (receipt_path.parent/'frames.jsonl').read_text().splitlines()]
            altered=copy.deepcopy(receipt)
            altered['recovery_report']['steps'][0]['completion']['ack']['held']=False
            assert receipt_error(altered,rows) is not None,'Unacknowledged no-op stage was accepted'
            altered=copy.deepcopy(receipt)
            altered['recovery_report']['kickoffs'][0]['snapshot']['state'][5]+=.001
            assert receipt_error(altered,rows) is not None,'No-op outside jaw tolerance was accepted'
            assert report['steps'][0]['miss_planned_duration_s']==0.
            result['noop_verifier_regressions']=True
        certificate=json.loads((output/'run/collector-finalization.json').read_text())
        assert certificate['completed'] and not certificate['cleanup_errors'],certificate
        if not args.inject_native_fault:assert certificate['passed'],certificate
        result.update(passed=True,stages=stages,first_stage_noop=True,accepted=receipt['accepted'],
            finalization_completed=certificate['completed'],finalization_cleanup_errors=certificate['cleanup_errors'],
            pause_commits=r.timeline.commits,native_limit_writes=len(writes),validity_queries=len(validity_calls))
    except BaseException as exc:result['error']=repr(exc);raise
    finally:
        if collector is not None and not collector.stop.is_set():
            try:collector.close()
            except Exception as exc:result['close_error']=repr(exc)
        stop.set();wake.set();physics_thread.join(10)
        retire_executor(executor,ros_thread);native.destroy_node();shutdown_context(rclpy,context)
        if rclpy.ok():rclpy.shutdown()
        result.update(physics_thread_retired=not physics_thread.is_alive(),ros_thread_retired=not ros_thread.is_alive())
        claim.close();atomic_json(output/'validation.json',result)
    print(json.dumps(result,indent=2),flush=True)
