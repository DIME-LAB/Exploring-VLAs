"""Re-grasp quotas, measured miss evidence and real servo/driver boundaries."""
import copy,json,math,sys,threading,unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from contract import COLORS,JOINTS,parameters,corrective_action
from checkpoint import progress
from recovery import REGRASP_POLICY,regrasp_slot,recovery_parameters,recovery_geometry,detect_miss,receipt_error,recovery_summary
from query_service import StrictQueryClient
from protocol import CancelledGeneration
from episode_policy import disposition,QualityReject
from test_tick_contract import checked_window
from tick_contract import TickContract,require_snapshots
import test_motion_operations as motions
import test_review_regressions as probes

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'vla_SO-ARM101/src/so_arm101_control'))


def snapshot(stamp=10**9,epoch=1):
    return dict(epoch=epoch,sim_ns=stamp,state_source='isaac_articulation',state=[0.]*5+[.1],
                bodies={name:dict(position=[.2,0.,.0064]) for name in ('red_2x3_0','green_2x3_0','cup_blue')})


def trajectory(start,end,duration=1.):
    return NS(joint_names=JOINTS[:5],points=[NS(positions=list(q),time_from_start=NS(sec=int(t),nanosec=round(t%1*1e9))) for q,t in ((start,0.),(end,duration))])


class RecoveryTests(unittest.TestCase):
    def test_accepted_quota_balances_colours_and_retry_does_not_consume_slot(self):
        slots=[i for i in range(900) if regrasp_slot(i)]
        self.assertEqual(len(slots),162);self.assertEqual(sum(regrasp_slot(i) for i in range(50)),9)
        self.assertEqual([sum(i%3==j for i in slots) for j in range(3)],[54]*3)
        receipts=[];accepted=0
        for trial in range(20):
            p=dict(parameters(1000+trial,COLORS[accepted%3],regrasp=regrasp_slot(accepted)),trial=trial)
            ok=trial not in (5,6)
            receipt=dict(params=p,accepted=ok,timing=dict(passed=ok))
            if ok and p['recovery']=='regrasp':
                c,_=self.driver(params=p);c.regrasp_prefix(p,1,receipt);receipt['episode_id']=c.current_episode_id;receipt['test_rows']=c.test_rows
            receipts.append(receipt)
            if ok:accepted+=1
        self.assertEqual(progress(receipts,1000,row_loader=lambda r:r.get('test_rows')),(20,18))
        broken=copy.deepcopy(receipts);broken[7].pop('recovery_report')
        with self.assertRaises(ValueError):progress(broken,1000,row_loader=lambda r:r.get('test_rows'))
        self.assertEqual([r['params']['recovery'] for r in receipts[5:8]],['regrasp']*3)
        with self.assertRaises(ValueError):progress(receipts,1000,regrasp_only=True)

    def test_transient_grasp_then_drop_is_not_an_empty_first_grasp(self):
        window=[snapshot(10**9+round(i*1e9/30)) for i in range(3)]
        window[1]['bodies']['red_2x3_0']['position'][2]+=.03
        self.assertFalse(detect_miss(window,'red_2x3_0',.1)['confirmed'])

    def test_recovery_evidence_is_bound_to_raw_episode_epoch_and_hashes(self):
        c,_=self.driver();params=parameters(1,'red',regrasp=True);report={}
        c.regrasp_prefix(params,1,report)
        receipt=dict(params=params,episode_id=c.current_episode_id,**report)
        self.assertIsNone(receipt_error(receipt,c.test_rows))
        for kind in ('epoch','id','state','hash'):
            bad=copy.deepcopy(receipt);rows=copy.deepcopy(c.test_rows)
            if kind=='epoch':
                bad['recovery_report']['epoch']=999
                for k in ('measured_before','measured_after_lift'):bad['recovery_report'][k]['epoch']=999
                bad['recovery_report']['detection']['epoch']=999
            elif kind=='id':bad['episode_id']='another-episode'
            elif kind=='state':rows[3]['state'][0]+=.001
            else:bad['recovery_report']['window_frame_sha256s'][0]='0'*64
            with self.subTest(kind=kind):self.assertIsNotNone(receipt_error(bad,rows))

    def test_offset_is_seeded_and_true_geometry_is_exactly_shared(self):
        from so_arm101_control.grasp_geometry import make_grasp_plan
        pose=dict(x=.20,y=0.,z=.0064,qz=0.,qw=1.)
        bbox=dict(sx=.031,sy=.0206,sz=.0128)
        for seed in range(12):
            params=recovery_parameters(seed,True)
            self.assertEqual(params,recovery_parameters(seed,True))
            geometry=recovery_geometry(pose,bbox,params,approach_height=.02,current_wrist_roll=0.)
            self.assertEqual(geometry['true_geometry'],make_grasp_plan(pose,bbox,approach_height=.02,current_wrist_roll=0.))
            self.assertAlmostEqual(math.hypot(*geometry['offset_m'][:2]),abs(params['signed_offset_m']))
            self.assertTrue(.02<=abs(params['signed_offset_m'])<=.04)
            json.dumps(geometry)
        with self.assertRaises(ValueError):recovery_geometry(pose,bbox,dict(signed_offset_m=.01,lift_m=.03))

    def test_miss_requires_empty_lift_and_measured_same_epoch_feedback(self):
        before=snapshot();after=snapshot(1033333333)
        self.assertTrue(detect_miss([before,after],'red_2x3_0',.1)['confirmed'])
        for name in ('red_2x3_0','green_2x3_0'):
            moved=copy.deepcopy(after);moved['bodies'][name]['position'][2]+=.03
            self.assertFalse(detect_miss([before,moved],'red_2x3_0',.1)['confirmed'])
        jaw=copy.deepcopy(after);jaw['state'][5]=.14
        self.assertFalse(detect_miss([before,jaw],'red_2x3_0',.1)['confirmed'])
        for change in (dict(epoch=2),dict(sim_ns=10**9),dict(state_source='controller_reference'),dict(state=[0.]*5+[float('nan')])):
            bad=dict(after,**change)
            with self.subTest(change=change),self.assertRaises(ValueError):detect_miss([before,bad],'red_2x3_0',.1)

    def test_query_transport_faults_never_turn_into_collision_free(self):
        for ready,done,value,error in ((False,True,NS(valid=True),RuntimeError),(True,False,None,TimeoutError),(True,True,None,RuntimeError)):
            raw=NS(service_is_ready=lambda:ready,call_async=lambda request:NS(done=lambda:done,result=lambda:value))
            strict=StrictQueryClient(raw,'MoveIt')
            with self.subTest(ready=ready,done=done,value=value),self.assertRaises(error):
                strict.service_is_ready();strict.call_async(None).result()
        raw=NS(service_is_ready=lambda:True,call_async=lambda request:NS(done=lambda:True,result=lambda:NS(valid=False)))
        self.assertFalse(StrictQueryClient(raw,'MoveIt').call_async(None).result().valid)

    def test_accepted_recovery_requires_miss_evidence(self):
        receipt=dict(accepted=True,params=dict(recovery='regrasp'),end_cancel=dict(held=True),
                     retirement=dict(passed=True),timing=dict(passed=True))
        self.assertEqual(disposition(receipt)['reason'],'missing_regrasp_evidence')
        c,_=self.driver();result={};params=parameters(1,'red',regrasp=True)
        c.regrasp_prefix(params,1,result)
        receipt.update(params=params,episode_id=c.current_episode_id,**result)
        self.assertEqual(disposition(receipt,rows=c.test_rows)['kind'],'accepted')
        self.assertEqual(recovery_summary([receipt])['share'],1.)
        broken=copy.deepcopy(receipt);broken['recovery_report']['measured_after_lift']['bodies'][params['object']]['position'][2]+=.03
        self.assertEqual(disposition(broken,rows=c.test_rows)['kind'],'terminal')
        self.assertIsNotNone(receipt_error(broken,c.test_rows))
        receipt.update(accepted=False,quality_reject=dict(kind='regrasp_miss_not_confirmed',contract=require_snapshots(c.episode_snapshots)))
        self.assertEqual(disposition(receipt)['kind'],'quality_reject')

    def dual_servo(self):
        c,r,_,_,_=motions.MotionOperationTests().setup_motion();v=c.servo;v.configure(parameters(1,'red',regrasp=True))
        c.node._actual_positions=dict(zip(JOINTS,[0.]*6));v.reference=[0.]*6;v.reference_epoch=1
        v.step=lambda play:None
        event=c.node._new_motion_event();event.motion_tag='regrasp_miss_descent'
        clean=[.1]*5;miss=[-.1]*5
        v.execute(trajectory([0.]*5,clean),event,executed_trajectory=trajectory([0.]*5,miss),hold_executed=True)
        return c,v,event,clean,miss

    def test_real_servo_labels_correct_from_measurement_while_executing_miss_and_hold(self):
        c,v,event,clean,miss=self.dual_servo();q=[0.]*6;seen=[]
        for i in range(300):
            stamp=10**9+round(i/30*1e9);c.node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=stamp))
            output=v.tick(stamp,q,drive_targets=q)
            action,executed,_,_,reference,_,completed,_,candidate=output
            self.assertEqual(action,corrective_action(q,reference))
            self.assertEqual(executed[5],action[5]);seen.append(abs(action[0]-executed[0]))
            v.commit_command(candidate,1,stamp);q=list(executed)
            if event in completed:break
        self.assertIsNone(v.plan);self.assertGreater(max(seen),.01)
        np.testing.assert_allclose(q[:5],miss,atol=.008)
        self.assertEqual(v.reference[:5],clean);self.assertEqual(v.executed_hold,(1,tuple(miss)))
        output=v.tick(stamp+33333333,q,drive_targets=q)
        self.assertGreater(output[0][0],q[0]);self.assertAlmostEqual(output[1][0],q[0],delta=.008)

    def test_old_ack_cannot_install_executed_hold_after_cancel(self):
        c,v,event,_,_=self.dual_servo()
        candidate=v.tick(10**9,[0.]*6,drive_targets=[0.]*6)[-1]
        candidate=replace(candidate,held_reference=tuple([-.1]*5))
        c.timeline(False)
        with self.assertRaises(CancelledGeneration):v.commit_command(candidate,1,10**9)
        self.assertIsNone(v.executed_hold)
        v.executed_hold=(1,tuple([-.1]*5))
        with self.assertRaises(RuntimeError):v.tick(2*10**9,[0.]*6)

    def test_query_timeout_poison_retires_existing_motion_generation(self):
        c,r,_=probes.ReviewRegressions().collector()
        raw=NS(call_async=lambda request:NS(done=lambda:False,result=lambda:None))
        strict=StrictQueryClient(raw,'MoveIt',lambda exc:c.abort_stalled_owner(exc,'validity query'))
        with self.assertRaises(TimeoutError):strict.call_async(None).result()
        self.assertTrue(c.stop.is_set());self.assertIsNone(c.generation.epoch)
        self.assertIsNotNone(c.generation.stalled)
        with self.assertRaises(CancelledGeneration):c.generation.require(1)
        self.assertEqual(c.events,['pause'])

    def test_only_idle_acknowledged_executed_hold_crosses_cancel(self):
        c,v,event,clean,miss=self.dual_servo()
        candidate=v.tick(10**9,[0.]*6,drive_targets=[0.]*6)[-1]
        v.commit_command(replace(candidate,held_reference=tuple(miss)),1,10**9)
        v.plan=None
        c.timeline(False)
        self.assertEqual(v.executed_hold,(2,tuple(miss)))
        self.assertEqual(v.previous_command.held_reference,tuple(miss))
        self.assertEqual(v.reference_epoch,2)
        done=c.node._new_motion_event()
        v.execute(trajectory([0.]*5,clean),done)
        c.timeline(False)
        self.assertIsNone(v.executed_hold);self.assertIsNone(v.reference)

    def test_reset_releases_gripper_and_reconciles_before_arm_home(self):
        c,r,_,_,_=motions.MotionOperationTests().setup_motion();events=[];snap=snapshot(10**9)
        c.node._select_grasp_object=lambda name:events.append('select')
        c.node._gripper_angle_for_object=lambda name:(.2,.1)
        c.servo.gripper=lambda *args,**kwargs:events.append('open') or True
        c.node._detach_lego_sync=lambda:events.append('detach')
        c.advance_ticks=lambda count:events.append('fresh measured tick') or [snap]
        c.sync_scene=lambda **kwargs:events.append(('scene',kwargs['snapshot']))
        c.reset_release('red_2x3_0')
        self.assertEqual(events[:4],['select','open','detach','fresh measured tick'])
        self.assertEqual(events[4],('scene',snap))

    def driver(self,followed=False,cancel_at=None,params=None):
        c,r,_,_,_=motions.MotionOperationTests().setup_motion();node=c.node
        scope=c.regrasp_prefix.__func__.__globals__;scope.update(recovery_geometry=recovery_geometry,detect_miss=detect_miss)
        params=parameters(1,'red',regrasp=True) if params is None else params
        c.current_episode_id='recovery-test';c.test_rows=[];c.episode_snapshots=[];audit=TickContract()
        name=params['object'];now=[snapshot(10**9)];now[0]['bodies'][name]=dict(position=[.20,0.,.0064])
        def record_tick():
            now[0]['physics_time_s']=now[0]['sim_ns']/1e9
            s=checked_window([dict(now[0],episode_id=c.current_episode_id,phase=getattr(node,'_last_motion_tag','fixture'))])[0]
            row=s['tick_contract']['row']
            if audit.previous:row['physics_drive_targets']=list(audit.previous['applied_command'])
            row['command_velocity']=[(a-b)*30 for a,b in zip(row['executed_action'][:5],row['physics_drive_targets'][:5])]
            s['tick_contract']=audit.finish(row);now[0]=s
            c.test_rows.append(row);c.episode_snapshots.append(copy.deepcopy(s))
        record_tick()
        events=[];node.objects_lock=threading.Lock()
        node.objects_data={name:dict(x=.20,y=0.,z=.0064,qz=0.,qw=1.)}
        node._lookup_bbox=lambda name:dict(sx=.031,sy=.0206,sz=.0128)
        node._tcp_clearance_var=NS(get=lambda:1.);node._gripper_angle_for_object=lambda name:(.2,.1)
        node._remove_single_lego_from_world=lambda name:events.append(('remove',name))
        node._check_state_valid_with_contacts=lambda target:(True,[])
        def path(current,target,duration):
            events.append(('plan',dict(target)))
            if cancel_at=='planning':c.timeline(False)
            return trajectory([current[n] for n in JOINTS[:5]],[target[n] for n in JOINTS[:5]],duration)
        node._plan_linear_joint_path=path
        c.held_snapshot=lambda epoch:dict(copy.deepcopy(now[0]),epoch=epoch)
        def execute(clean,event,**kwargs):
            events.append(('arm',node._last_motion_tag,copy.deepcopy(clean.points[-1].positions),copy.deepcopy(kwargs['executed_trajectory'].points[-1].positions)))
            now[0]['sim_ns']+=33333333;now[0]['state'][:5]=kwargs['executed_trajectory'].points[-1].positions
            if followed and node._last_motion_tag=='regrasp_miss_lift':now[0]['bodies'][name]['position'][2]+=.03
            record_tick()
            event.verdict=dict(ok=True);event.set()
        c.servo.execute=execute
        def grip(target,duration,blocking,phase):
            events.append(('grip',phase));now[0]['sim_ns']+=33333333;now[0]['state'][5]=target;record_tick();return True
        c.servo.gripper=grip
        def sync(**kwargs):events.append(('sync',));c.scene_pose_check=dict(passed=True)
        c.sync_scene=sync
        return c,events

    def test_driver_misses_reopens_then_rebuilds_from_measured_poses_without_attach(self):
        c,events=self.driver();result={}
        c.regrasp_prefix(parameters(1,'red',regrasp=True),1,result)
        self.assertTrue(result['recovery_report']['miss_confirmed'])
        stages=[e[1] for e in events if e[0] in ('arm','grip')]
        self.assertEqual(stages,['regrasp_miss_open','regrasp_miss_approach','regrasp_miss_descent','regrasp_miss_close','regrasp_miss_lift','regrasp_reopen'])
        self.assertEqual(events[-1],('sync',))
        for e in events:
            if e[0]=='arm':self.assertNotEqual(e[2],e[3])
        json.dumps(result)

    def test_accidental_grasp_reopens_then_quality_rejects(self):
        c,events=self.driver(followed=True)
        with self.assertRaises(QualityReject) as raised:c.regrasp_prefix(parameters(1,'red',regrasp=True),1,{})
        self.assertEqual(raised.exception.kind,'regrasp_miss_not_confirmed')
        self.assertEqual(events[-1],('grip','regrasp_reopen'))

    def test_cancel_during_prefix_planning_cannot_dispatch_old_target(self):
        c,events=self.driver(cancel_at='planning')
        with self.assertRaises(CancelledGeneration):c.regrasp_prefix(parameters(1,'red',regrasp=True),1,{})
        self.assertFalse(any(e[0]=='arm' for e in events))


if __name__=='__main__':unittest.main()
