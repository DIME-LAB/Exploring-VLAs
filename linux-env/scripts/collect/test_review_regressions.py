from wall_profile import profile_tick,summary as wall_profile_summary,milliseconds,PROFILE_POLICY
from safe_home import make_baseline,validate_baseline
from frame_join import SensorAmbiguity,identity as frame_identity,exact_duplicate
from comparisons import comparison,matches,interval,native_frame
from tick_contract import world_step_errors
"""CPU fault/interleaving probes at the actual collector/runtime boundaries."""
import ast
from dataclasses import replace
from contextlib import contextmanager
import copy
import math
import json
import random
import subprocess
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from contract import JOINTS, corrective_action, parameters
from protocol import CancelledGeneration, CommandLedger, StepGeneration, OwnershipStalled
from dart import DEFAULT_NOISE, SmoothDart, TargetCandidate, PlannerSchedule
from command_geometry import choose_noise_scale,final_command,QUERY_POLICY,frame_capacity_seconds
from tick_contract import TickContract,TickContractError,require_snapshots,digest,verify_proof,CLOCK_POLICY
from episode_policy import ConvergenceFailure, QualityReject, POLICY, WorkerRegistry, FailureStreak, disposition
from scene_sync import physical_pose_caches, collision_pose_errors
from success import PHYSICAL_FLOOR_M, INTERIOR_FLOOR_M, initial_arm
import test_safety
from test_safety import method
from so_arm101_control.motion_wait import MotionWaitTimeout

HERE = Path(__file__).parent
ROOT = HERE.parents[2]


def load_class(filename, name):
    path = HERE/filename
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    scope = dict(profile_tick=profile_tick,wall_profile_summary=wall_profile_summary,milliseconds=milliseconds,PROFILE_POLICY=PROFILE_POLICY,make_baseline=make_baseline,validate_baseline=validate_baseline,SensorAmbiguity=SensorAmbiguity,frame_identity=frame_identity,exact_duplicate=exact_duplicate,math=math,comparison=comparison,matches=matches,interval=interval,native_frame=native_frame,world_step_errors=world_step_errors,frame_capacity_seconds=frame_capacity_seconds,QUERY_POLICY=QUERY_POLICY,CLOCK_POLICY=CLOCK_POLICY,random=random,replace=replace,copy=copy,PlannerSchedule=PlannerSchedule,TargetCandidate=TargetCandidate,choose_noise_scale=choose_noise_scale,final_command=final_command,TickContract=TickContract,TickContractError=TickContractError,require_snapshots=require_snapshots,digest=digest,verify_proof=verify_proof, DEFAULT_NOISE=DEFAULT_NOISE, SmoothDart=SmoothDart, initial_arm=initial_arm, ConvergenceFailure=ConvergenceFailure, QualityReject=QualityReject, POLICY=POLICY, WorkerRegistry=WorkerRegistry, FailureStreak=FailureStreak, disposition=disposition, MotionWaitTimeout=MotionWaitTimeout, contextmanager=contextmanager, OwnershipStalled=OwnershipStalled, threading=threading,time=time,np=np,json=json,JOINTS=JOINTS,
                 String=lambda data:NS(data=data),StepGeneration=StepGeneration,
                 CancelledGeneration=CancelledGeneration,CommandLedger=CommandLedger,
                 corrective_action=corrective_action,JOINT_LIMITS={k:(-3,3) for k in JOINTS},
                 ns=lambda s:s.sec*1000000000+s.nanosec)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),scope)
    return scope[name], scope


def frame(epoch=1):
    stamp=NS(sec=1,nanosec=0)
    joint=NS(name=JOINTS,position=[0.]*6,velocity=[0.]*6,header=NS(stamp=stamp))
    snapshot=dict(epoch=epoch,sim_ns=10**9,physics_time_s=1.,state=[0.]*6,velocity=[0.]*6,
                  drive_targets=[0.]*6,bodies={},state_source='isaac_articulation')
    def camera():return NS(header=NS(stamp=NS(sec=1,nanosec=0)),encoding='rgb8',width=640,height=480,step=1920,data=bytes(921600))
    return {k:(v,1) for k,v in dict(joint=joint,tick=snapshot,wrist=camera(),top=camera()).items()}


class ReviewRegressions(unittest.TestCase):
    def collector(self, clock_hook=lambda c,r:None):
        Collector, scope = load_class('collect.py','Collector')
        Servo, _ = load_class('collect.py','Servo')
        Runtime, _ = load_class('collection_runtime.py','Runtime')
        c=object.__new__(Collector);r=object.__new__(Runtime);v=object.__new__(Servo)
        c.gpu_footprint=lambda:dict(passed=True)
        c.generation=StepGeneration();c.generation.install(dict(epoch=1,invalid_through_ns=966666667))
        c.stalled_abort_lock=threading.Lock();c.stalled_abort_started=False
        c.stop=threading.Event();c.failure=None;c.writer=None;c.writer_lock=threading.Lock()
        c.tick_condition=threading.Condition();c.last_processed_ns=0;c.last_snapshot=None
        c.node=NS(execution=NS(collector=True),_qs_abort_evt=threading.Event(),_selected_object=None,
            _actual_positions=dict(zip(JOINTS,[0]*6)),_joint_states_callback=lambda msg:None)
        def status(value,event=None):
            c.node._last_motion_status=value
            if event is not None:event.verdict=dict(value)
        c.node._set_motion_status=status
        c.node._check_state_valid_with_contacts=lambda target:(True,[])
        c.episode_snapshots=[]
        c.update_pose_cache=lambda snapshot:None
        c.events=[];acks={}
        r.epoch=1;r.invalid_through_ns=966666667;r.failure=None;r.mode='collect';r.busy=False
        r.stamp=lambda:10**9;r.world=NS(current_time=1.)
        r.timeline=NS(pause=lambda:c.events.append('pause'),is_playing=lambda:False,
                      play=lambda:c.events.append('advance'))
        r.ledger=CommandLedger();r.indices=list(range(6))
        targets=np.zeros((1,6))
        def write(values,**kwargs):
            c.events.append('target_write');targets[:]=values
        r.art=NS(set_joint_position_targets=write,set_joint_velocity_targets=lambda *a,**k:None,
                 get_applied_actions=lambda:NS(joint_positions=targets))
        r.ack_pub=NS(publish=lambda msg:acks.update({(json.loads(msg.data)['epoch'],10**9):json.loads(msg.data)}))
        scope['timeline']=lambda play:r.advance() if play else r.cancel()
        v.node=c.node;v.generation=c.generation;v.lock=threading.RLock();v.plan=dict(trajectory=NS(points=[NS(time_from_start=NS(sec=2,nanosec=0),positions=[.1]*5)],joint_names=JOINTS[:5]),done=threading.Event(),start=10**9,epoch=1)
        v.grip_plan=None;v.reference_epoch=1;v.reference=[.1]*5+[0.];v.rng=random.Random(1);v.params=parameters(1,'red')
        v.phase='move';v.previous_executed=None
        v.planned_sim_seconds=0.;v.planned_operations=0;v.noise_index=0;v.noise_specs=[];v.previous_command=None
        def now():
            clock_hook(c,r)
            return NS(nanoseconds=10**9)
        c.node.get_clock=lambda:NS(now=now)
        v.publish_reference=lambda cmd:None
        c.servo=v
        entry=frame()
        def read(**kwargs):
            c.stop.set()  # Consume exactly one real capture-loop iteration.
            return 10**9,entry
        c.sensors=NS(read=read,read_ack=lambda epoch,stamp:acks[(epoch,stamp)],reference=None,
            condition=threading.Condition(),pending={},acks=acks,reference_at=lambda epoch,stamp:None)
        c.command_pub=NS(publish=r.command_received)
        c.generation.on_stall=c.abort_stalled_owner
        return c,r,entry

    def test_uncancelled_capture_applies_and_records_matching_generation(self):
        c,r,_=self.collector();rows=[]
        c.writer=NS(add=lambda row,images:rows.append(row))
        capture_globals=c.capture.__func__.__globals__
        capture_globals['Sensors']=NS(image=lambda msg:np.zeros((2,2,3),dtype=np.uint8))
        c.capture()
        self.assertIsNone(c.failure);self.assertEqual(c.events,['target_write','advance'])
        self.assertEqual(len(rows),1)
        self.assertEqual([rows[0][k] for k in ('observation_epoch','expert_epoch','epoch')],[1,1,1])
        np.testing.assert_allclose(rows[0]['applied_command'],rows[0]['executed_action'],atol=1e-6,rtol=0)

    def test_cancel_during_expert_clock_read_cannot_relabel_or_restart(self):
        def cancel(c,r):
            result=c.timeline(False)
            c.events.append('cancel_returned_epoch_'+str(result['epoch']))
        c,r,_=self.collector(cancel)
        c.capture()
        self.assertEqual(c.events,['pause','cancel_returned_epoch_2'])
        self.assertIsNone(c.failure);self.assertIsNone(c.last_snapshot)
        self.assertEqual(r.epoch,2);self.assertFalse(r.busy)

    def test_cancel_waits_for_inflight_dispatch_ack_and_record_retirement(self):
        c,r,_=self.collector()
        entered=threading.Event();release=threading.Event();attempted=threading.Event();retired=threading.Event()
        real_publish=c.command_pub.publish
        def blocked(msg):
            entered.set();self.assertTrue(release.wait(2));real_publish(msg)
        c.command_pub.publish=blocked
        capture=threading.Thread(target=c.capture);capture.start()
        self.assertTrue(entered.wait(2))
        def cancel():
            attempted.set();c.timeline(False);retired.set()
        reset=threading.Thread(target=cancel);reset.start();self.assertTrue(attempted.wait(2))
        self.assertFalse(retired.wait(.05))
        release.set();capture.join(2);reset.join(2)
        self.assertFalse(capture.is_alive());self.assertFalse(reset.is_alive())
        self.assertIsNone(c.failure);self.assertTrue(retired.is_set())
        self.assertEqual(c.events,['target_write','advance','pause'])
        self.assertEqual(c.last_snapshot['epoch'],1)
        self.assertEqual(c.generation.epoch,2)

    def test_same_stamp_snapshot_cannot_cross_cancel_or_restore_barrier(self):
        c,r,entry=self.collector();c.timeline(False)
        # Both old snapshots and a fabricated new-epoch snapshot at the reset
        # stamp are invalid; only a newly advanced physical frame is usable.
        for epoch in (1,2):
            entry['tick'][0]['epoch']=epoch
            c.capture();self.assertIsNone(c.last_snapshot)
            c.stop.clear()
        self.assertEqual(c.events,['pause'])
        r.command_received(NS(data=json.dumps(dict(epoch=2,observation_stamp_ns=10**9,command=[1]*6,advance=True))))
        self.assertEqual(c.events,['pause','pause'])  # rejected, held; no writes/advance

    def test_delayed_planner_event_and_child_worker_keep_original_epoch(self):
        path=ROOT/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pick_place.py'
        token=method(path,'PickPlaceNode','_motion_token')
        new=method(path,'PickPlaceNode','_new_motion_event')
        worker=method(path,'PickPlaceNode','_motion_thread')
        gate=StepGeneration();gate.install(dict(epoch=1,invalid_through_ns=0))
        node=NS(_motion_context=threading.local(),_motion_generation_provider=lambda:gate.epoch)
        node._motion_token=lambda:token(node)
        node._current_motion_event=lambda:None
        result=[];thread=worker(node,target=lambda:result.append(new(node)))
        gate.install(dict(epoch=2,invalid_through_ns=100))
        thread.start();thread.join(2)
        self.assertEqual(result[0].motion_epoch,1)
        Servo,_=load_class('collect.py','Servo');servo=object.__new__(Servo)
        servo.generation=gate
        with self.assertRaises(CancelledGeneration):servo.execute(None,result[0])

    def test_restore_reads_back_all_six_arm_positions_and_velocities(self):
        Runtime,_=load_class('collection_runtime.py','Runtime')
        fixture=test_safety.SafetyTests().snapshot();fixture.update(state=[0]*6,velocity=[0]*6,epoch=2,physics_time_s=1.,drive_targets=[0.]*6)
        for joint in range(6):
            for field,value in [('state',.2),('velocity',1.)]:
                actual=copy.deepcopy(fixture);actual[field][joint]=value
                r=object.__new__(Runtime);r.busy=False;r.timeline=NS(is_playing=lambda:False)
                r.cancel=lambda:None;r.require_scene=lambda:None;r.indices=list(range(6))
                r.epoch=2;r.invalid_through_ns=100
                r.art=NS(set_joint_positions=lambda *a,**k:None,set_joint_velocities=lambda *a,**k:None)
                r.bodies={k:NS(set_world_poses=lambda *a:None,set_velocities=lambda *a:None) for k in fixture['bodies']}
                r.apply=lambda *a:None;r.stamp=lambda:100;r.snapshot=lambda:actual
                with self.subTest(joint=joint,field=field):
                    with self.assertRaisesRegex(RuntimeError,'arm position/zero velocity'):r.restore(fixture,arm=True)
        r.snapshot=lambda:copy.deepcopy(fixture)
        self.assertEqual(r.restore(fixture,arm=True)['max_arm_position_error_rad'],0.)

    def test_mesh_and_cylinder_collision_poses_match_oriented_physical_reset(self):
        snapshot=test_safety.SafetyTests().snapshot()
        snapshot['base_world_pose']=dict(position=[.1,-.2,.05],orientation=[1,0,0,0])
        # A tilted cup exercises the body-center -> oriented base offset.
        angle=.03;snapshot['bodies']['cup_red']['orientation']=[np.cos(angle/2),np.sin(angle/2),0,0]
        objects,cups=physical_pose_caches(snapshot)
        def pose(p):
            return NS(position=NS(**dict(zip(('x','y','z'),p['position']))),
                      orientation=NS(**dict(zip(('w','x','y','z'),p['orientation']))))
        co=[]
        for key,body in snapshot['bodies'].items():
            invpos=[v-o for v,o in zip(body['position'],snapshot['base_world_pose']['position'])]
            co.append(NS(id='cup_drop_3' if key=='cup_red' else 'lego_'+key,
                header=NS(frame_id='base'),mesh_poses=[pose(dict(position=invpos,orientation=body['orientation']))],primitive_poses=[]))
        self.assertTrue(collision_pose_errors(co,snapshot)['passed'])
        co[1].primitive_poses=co[1].mesh_poses;co[1].mesh_poses=[]
        cp=cups['drop_3'];co[1].primitive_poses[0]=pose(dict(position=[cp[k] for k in ('x','y','z')],orientation=[cp[k] for k in ('qw','qx','qy','qz')]))
        self.assertTrue(collision_pose_errors(co,snapshot)['passed'])
        for index in (0,1):
            bad=copy.deepcopy(co);(bad[index].mesh_poses+bad[index].primitive_poses)[0].position.x+=99
            self.assertFalse(collision_pose_errors(bad,snapshot)['passed'])
        bad=copy.deepcopy(co);bad[0].mesh_poses[0].orientation=NS(w=0,x=1,y=0,z=0)
        self.assertFalse(collision_pose_errors(bad,snapshot)['passed'])

    def test_actual_sync_scene_rejects_correct_ids_at_wrong_poses(self):
        import sys
        Collector, scope=load_class('collect.py','Collector')
        scope['collision_pose_errors']=collision_pose_errors
        clock=iter((0.,20.));scope['time']=NS(monotonic=lambda:next(clock))
        c=object.__new__(Collector)
        c.episode_start_physical=test_safety.SafetyTests().snapshot()
        c.episode_start_physical.update(state=[0]*6,base_world_pose=dict(position=[0]*3,orientation=[1,0,0,0]))
        c.update_pose_cache=lambda snapshot:None
        p=NS(position=NS(x=99,y=99,z=99),orientation=NS(w=1,x=0,y=0,z=0))
        objects=[NS(id=key,header=NS(frame_id='base'),mesh_poses=[p],primitive_poses=[])
                 for key in ('cup_drop_3','lego_red_2x3_0')]
        response=NS(scene=NS(world=NS(collision_objects=objects),robot_state=NS(joint_state=NS(name=JOINTS,position=[0]*6))))
        c.node=NS(_cup_collision_names=['cup_drop_3'],_lego_collision_names=['lego_red_2x3_0'],
                  _drop_data={},objects_data={},_remove_lego_collision_objects=lambda:None,
                  _remove_cup_collision_objects=lambda:None,_add_cup_collision_objects=lambda **kw:None,
                  _add_lego_collision_objects=lambda **kw:None,_wait_future=lambda *a,**k:response,
                  _get_scene_client=NS(call_async=lambda req:None))
        c.node._cmd_toggle_ground_plane=lambda:None
        from contextlib import nullcontext
        c.node._motion_guard=nullcontext
        c.scene_reset=NS(clear=lambda:{},set_policy=lambda ids:None,
            verify=lambda snapshot,joints:dict(passed=False,poses=collision_pose_errors(objects,snapshot)))
        service=NS(Request=lambda:NS(components=NS(components=0)))
        with patch.dict(sys.modules,{'moveit_msgs':NS(),'moveit_msgs.srv':NS(GetPlanningScene=service)}):
            with self.assertRaisesRegex(TimeoutError,'Full planning scene reset verification'):c.sync_scene()

    def test_indexed_launcher_replaces_output_outside_instance_root(self):
        source=(ROOT/'linux-env/scripts/isaac/isaacsim_launch.sh').read_text()
        segment=source[source.index('        default_output=/tmp/soarm101-resources'):source.index('        mkdir -p "$MCP_CLIENT_OUTPUT_DIR"')]
        probe='SCRIPT_DIR='+str(ROOT/'linux-env/scripts/isaac')+'\nINSTANCE_ROOT=/mnt/a4500-backup/exploring-vlas/runtime/instance-0\nMCP_CLIENT_OUTPUT_DIR=/tmp/elsewhere\n'+segment+'\nprintf "%s" "$MCP_CLIENT_OUTPUT_DIR"\n'
        result=subprocess.run(['bash','-c',probe],text=True,capture_output=True,check=True)
        self.assertEqual(result.stdout,'/mnt/a4500-backup/exploring-vlas/runtime/instance-0/resources')


if __name__=='__main__':unittest.main()
