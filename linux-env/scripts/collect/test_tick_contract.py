"""All physical-verdict ticks, including failed/unrecorded frames, are audited."""
import copy,unittest
from tick_contract import TickContract,TickContractError,require_snapshots,STEP_NS


def checked_window(snapshots):
    """Create realistic camera/state/ACK receipts for physical-only CPU fixtures."""
    audit=TickContract();result=[]
    for i,source in enumerate(snapshots):
        s=copy.deepcopy(source);s.setdefault('epoch',1);s.setdefault('sim_ns',10**9+round(i*1e9/30))
        s.setdefault('state',[0.]*6);s.setdefault('velocity',[0.]*6);s.setdefault('state_source','isaac_articulation');s.setdefault('bodies',{})
        s.setdefault('physics_time_s',s['sim_ns']/1e9)
        for body in s['bodies'].values():body.setdefault('orientation',[1.,0.,0.,0.]);body.setdefault('velocity',[0.]*6)
        stamp=s['sim_ns'];state=s['state'];previous=audit.previous
        row=dict(camera_layout={k:dict(encoding='rgb8',width=640,height=480,step=1920,bytes=921600) for k in ('wrist','top')},execution_geometry=dict(passed=True,epoch=s['epoch'],sim_ns=stamp,target=state),tick_contract_version=1,epoch=s['epoch'],observation_epoch=s['epoch'],expert_epoch=s['epoch'],
            sim_ns=stamp,invalid_through_ns=0,joint_stamp_ns=stamp,wrist_stamp_ns=stamp,top_stamp_ns=stamp,pose_stamp_ns=stamp,
            state=state,velocity=s['velocity'],physics_state=state,physics_velocity=s['velocity'],state_source=s['state_source'],
            physics_drive_targets=previous['applied_command'] if previous else state,physics_time_s=s['physics_time_s'],physical_bodies=s['bodies'],
            wall_ns=stamp+10**9,joint_receive_ns=stamp+10**9,wrist_receive_ns=stamp+10**9,top_receive_ns=stamp+10**9,
            action=state,executed_action=state,expert_reference=state,expert_observation_stamp_ns=stamp,
            observation_stamp_ns=stamp,declared_application_stamp_ns=stamp,application_stamp_ns=stamp,command_clock_ns=stamp,
            applied_command=state,phase=source.get('phase','fixture'),episode_id=source.get('episode_id','fixture'))
        row.update(planner_execution_reference=list(state),dart_offset=[0.]*6,command_filter_delta=[0.]*6,command_velocity=[(a-b)*30 for a,b in zip(state[:5],row['physics_drive_targets'][:5])])
        s['tick_contract']=audit.finish(row);result.append(s)
    return result


class TickContractTests(unittest.TestCase):
    def test_verification_clock_gap_is_contract_failure_not_physical_failure(self):
        from success import settled_containment,SETTLED_TICKS
        from test_safety import SafetyTests
        raw=[SafetyTests().snapshot() for _ in range(SETTLED_TICKS)]
        for i,s in enumerate(raw):s['sim_ns']=10**9+round(i*1e9/30)
        rows=checked_window(raw);self.assertTrue(settled_containment(rows,'red_2x3_0','red')['passed'])
        rows[-1]['sim_ns']+=STEP_NS
        result=settled_containment(rows,'red_2x3_0','red')
        self.assertFalse(result['contract_passed'])
        self.assertFalse(result['passed'])
        from episode_policy import disposition
        from test_contract import ContractTests
        from contract import timing_audit
        receipt=dict(accepted=False,end_cancel=dict(held=True),retirement=dict(passed=True),timing=timing_audit(ContractTests().rows(),30),
            quality_reject=dict(kind='containment',contract=result['contract']))
        self.assertTrue(receipt['timing']['passed'])
        self.assertEqual(disposition(receipt)['kind'],'terminal')

    def test_contract_valid_physical_failure_remains_separate(self):
        from success import settled_containment,SETTLED_TICKS
        from test_safety import SafetyTests
        raw=[SafetyTests().snapshot(.11) for _ in range(SETTLED_TICKS)]
        for i,s in enumerate(raw):s['sim_ns']=10**9+round(i*1e9/30)
        result=settled_containment(checked_window(raw),'red_2x3_0','red')
        self.assertTrue(result['contract_passed']);self.assertFalse(result['physical_passed'])

    def test_proof_cannot_be_reused_after_pose_or_state_change(self):
        for key in ('sim_ns','epoch','state','bodies'):
            s=checked_window([{}])[0]
            if key=='state':s[key][0]+=.1
            elif key=='bodies':s[key]['unexpected']=dict(position=[0.,0.,0.])
            else:s[key]+=1
            with self.subTest(key=key),self.assertRaises(TickContractError):require_snapshots([s])


class CaptureContractTests(unittest.TestCase):
    def convergence(self,defect=None):
        from types import SimpleNamespace as NS
        import threading
        import test_review_regressions as probes
        c,r,entry=probes.ReviewRegressions().collector();stamp=8700000000
        previous=checked_window([dict(sim_ns=stamp-STEP_NS,state=[.1]*5+[0.])])[0]['tick_contract']['row']
        previous.update(wall_ns=100,joint_receive_ns=100,wrist_receive_ns=100,top_receive_ns=100)
        c.tick_audit=TickContract();c.tick_audit.previous=previous
        for key,(msg,_) in list(entry.items()):
            if key!='tick':msg.header.stamp.sec,msg.header.stamp.nanosec=divmod(stamp,10**9)
            entry[key]=(msg,200)
        snapshot=entry['tick'][0];snapshot.update(sim_ns=stamp,physics_time_s=stamp/1e9,state=[.09]*5+[0.],drive_targets=[.1]*5+[0.])
        entry['joint'][0].position=list(snapshot['state'])
        c.servo.plan=dict(epoch=1,start=10**9,done=threading.Event(),trajectory=NS(joint_names=probes.JOINTS[:5],points=[NS(positions=[.1]*5,time_from_start=NS(sec=2,nanosec=0))]))
        c.servo.reference=[.1]*5+[0.];c.servo.phase='grasp_move';c.servo.params['dart']['on_probability']=0.
        c.node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=stamp));r.stamp=lambda:stamp
        r.world.current_time=snapshot['physics_time_s']
        if defect=='first_gap':c.tick_audit.previous=None
        if defect=='camera_layout':entry['wrist'][0].encoding='rgba8'
        if defect=='clock_gap':c.tick_audit.previous['sim_ns']-=STEP_NS
        if defect=='physics_gap':snapshot['physics_time_s']-=1.
        if defect=='target_overwrite':snapshot['drive_targets'][0]+=.1
        if defect=='image_stamp':entry['wrist'][0].header.stamp.nanosec+=10000
        if defect=='nonfinite':snapshot['velocity'][0]=float('nan');entry['joint'][0].velocity[0]=float('nan')
        def read(**kwargs):c.stop.set();return stamp,entry
        c.sensors.read=read
        import json
        r.ack_pub=NS(publish=lambda msg:c.sensors.acks.update({(json.loads(msg.data)['epoch'],json.loads(msg.data)['observation_stamp_ns']):json.loads(msg.data)}))
        c.sensors.read_ack=lambda epoch,t:c.sensors.acks[(epoch,t)]
        c.capture();return c

    def test_real_convergence_tick_gets_held_ack_and_its_own_contract_before_quality(self):
        from episode_policy import disposition
        c=self.convergence();self.assertIsNone(c.failure)
        record=c.capture_failures[0];self.assertTrue(record['validated_sim_tick'])
        self.assertEqual(record['failing_tick_contract']['row']['sim_ns'],8700000000)
        receipt=dict(accepted=False,original_capture_error_preserved=True,capture_failures=c.capture_failures,
                     end_cancel=dict(held=True),retirement=dict(passed=True))
        self.assertEqual(disposition(receipt)['kind'],'quality_reject')

    def test_pipeline_faults_at_unrecorded_convergence_tick_are_terminal(self):
        for defect in ('clock_gap','physics_gap','target_overwrite','image_stamp','nonfinite','first_gap','camera_layout'):
            with self.subTest(defect=defect):
                c=self.convergence(defect)
                self.assertTrue(c.failure);self.assertTrue(c.generation.stalled)
                self.assertIn('pipeline_contract_failure',c.capture_failures[0])
                self.assertFalse(any(e=='target_write' for e in c.events))

    def test_capture_stops_before_dispatch_when_measured_lag_crosses_obstacle(self):
        from types import SimpleNamespace as NS
        from dart import PlannerSchedule
        from test_dart import path
        import threading
        import test_review_regressions as probes
        c,r,entry=probes.ReviewRegressions().collector()
        state=[.35,0.,0.,0.,0.,0.];entry['joint'][0].position=state;entry['tick'][0]['state']=state
        reference=path([(0.,[0.]*5),(1.,[.5,0.,0.,0.,0.]),(2.,[.5,.5,0.,0.,0.])])
        c.servo.plan=dict(epoch=1,start=-266666667,done=threading.Event(),trajectory=reference,
                          schedule=PlannerSchedule(reference))
        c.servo.params['dart']['on_probability']=0.
        c.node._check_state_valid_with_contacts=lambda q:(not (.30<q['shoulder_pan']<.48 and .02<q['shoulder_lift']<.14),[])
        c.capture();self.assertTrue(c.failure);self.assertEqual(c.events,['pause'])


class ClockJoinTests(unittest.TestCase):
    def test_clock_delivery_can_join_without_advancing_physics(self):
        import threading
        from types import SimpleNamespace as NS
        import test_review_regressions as probes
        c,r,_=probes.ReviewRegressions().collector();reading=threading.Event();value=[966666667]
        def clock():
            current=value[0];reading.set();return NS(nanoseconds=current)
        c.node.get_clock=lambda:NS(now=clock)
        def deliver():reading.wait(2);value[0]=10**9
        thread=threading.Thread(target=deliver);thread.start()
        c.wait_tick_clock(10**9,1);thread.join(2)
        self.assertEqual(c.events,[])

    def test_clock_leading_the_held_tick_is_terminal(self):
        from types import SimpleNamespace as NS
        import test_review_regressions as probes
        c,r,_=probes.ReviewRegressions().collector()
        c.node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=1033333333))
        with self.assertRaises(TickContractError):c.wait_tick_clock(10**9,1)
        self.assertEqual(c.events,[])


class VerificationFinalizationTests(unittest.TestCase):
    def test_invalid_verification_does_not_create_or_mask_lift_quality(self):
        import ast
        from pathlib import Path
        from types import SimpleNamespace as NS
        from tick_contract import require_snapshots
        tree=ast.parse(Path(__file__).with_name('collect.py').read_text())
        branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="receipt.get('score') and receipt['score'].get('contract_passed')")
        for valid in (False,True):
            receipt=dict(error='original verification contract failure',score=dict(contract_passed=valid),initial_physical=dict(bodies={'red':dict(position=[0.,0.,0.])}))
            scope=dict(receipt=receipt,params=dict(object='red'),writer=NS(rows=[]),snapshots=checked_window([{}]),require_snapshots=require_snapshots)
            exec(compile(ast.Module(body=[branch],type_ignores=[]),'collect.py','exec'),scope)
            if valid:self.assertEqual(receipt['quality_reject']['kind'],'insufficient_lift')
            else:
                self.assertNotIn('quality_reject',receipt)
                self.assertEqual(receipt['error'],'original verification contract failure')


class EpisodeIdentityTests(unittest.TestCase):
    def test_actual_attempt_preparation_assigns_new_id_before_reset_or_home(self):
        import ast,hashlib,threading
        from pathlib import Path
        from types import SimpleNamespace as NS
        from protocol import StepGeneration
        tree=ast.parse(Path(__file__).with_name('collect.py').read_text())
        run=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run')
        loop=next(n for n in run.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='trial')
        def assigned(node,name):return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
        path=next(n for n in loop.body if assigned(n,'path'));receipt=next(n for n in loop.body if assigned(n,'receipt'))
        attempt=next(n for n in loop.body if isinstance(n,ast.Try))
        observed=[];generation=StepGeneration();generation.install(dict(epoch=7,invalid_through_ns=0))
        c=NS(needs_home_restore=False,current_episode_id='previous episode',generation=generation,node=NS(_qs_abort_evt=threading.Event()))
        c.gpu_footprint=lambda:dict(passed=True)
        c.reset_release=lambda name:observed.append(c.current_episode_id)
        c.motion_call=lambda fn,*a:fn(*a)
        scope=dict(transport_config=dict(physics_config=dict(dynamics='cpu')),self=c,params=dict(object='red',seed=123),root=Path('/a/run'),trial=9,hashlib=hashlib,motion_epochs=set())
        reset_index=next(i for i,n in enumerate(attempt.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and any(isinstance(a,ast.Attribute) and a.attr=='reset_release' for a in n.value.args))
        exec(compile(ast.Module(body=[path,receipt,*attempt.body[:reset_index+1]],type_ignores=[]),'collect.py','exec'),scope)
        self.assertEqual(observed,[scope['receipt']['episode_id']])
        self.assertNotEqual(observed,['previous episode'])
