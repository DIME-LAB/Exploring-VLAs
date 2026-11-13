"""Home hold regressions and quality/pipeline boundaries at production interfaces."""
import copy,threading,time,unittest,json
from pathlib import Path
from types import SimpleNamespace as NS
from episode_policy import (ConvergenceFailure, QualityReject, WorkerRegistry, FailureStreak, disposition, POLICY)
from protocol import CancelledGeneration, StepGeneration
from contract import JOINTS,timing_audit
from dart import TargetLimiter
import test_review_regressions as probes
from so_arm101_control.grasp_geometry import _grasp_home_data_target
import test_motion_operations as motions
import test_contract as contracts


class HomeHoldTests(unittest.TestCase):
    def test_repeated_idle_pauses_keep_clean_target_instead_of_walking_with_measured_bias(self):
        c,r,entry=probes.ReviewRegressions().collector();v=c.servo
        v.plan=None;v.grip_plan=None;nominal=list(v.reference)
        for step in range(8):
            state=[x+.003 for x in nominal]
            c.node._actual_positions.update(zip(JOINTS,state))
            c.timeline(False)
            self.assertEqual(v.reference,nominal);self.assertEqual(v.reference_epoch,c.generation.epoch)
            with c.generation.own():
                clean,_,_,_,reference,*_=v.tick(10**9+step*33333333,state)
            self.assertEqual(reference,nominal);self.assertEqual(clean,nominal)

    def test_active_cancel_does_not_adopt_an_unfinished_plan_target(self):
        c,r,_=probes.ReviewRegressions().collector();old=list(c.servo.reference)
        c.timeline(False)
        self.assertIsNone(c.servo.reference);self.assertIsNone(c.servo.reference_epoch)
        state=[.2]*6
        with c.generation.own():c.servo.tick(2*10**9,state)
        self.assertEqual(c.servo.reference,state);self.assertNotEqual(c.servo.reference,old)

    def test_stale_reference_is_refused_before_it_can_be_executed(self):
        c,r,_=probes.ReviewRegressions().collector();c.servo.reference_epoch=0
        with self.assertRaises(CancelledGeneration):c.servo.tick(10**9,[0.]*6)
        self.assertEqual(c.events,[])

    def test_home_approach_and_hold_receive_no_execution_noise(self):
        c,r,_=probes.ReviewRegressions().collector();v=c.servo
        v.rng=NS(random=lambda:0.,gauss=lambda *a:.005)
        v.phase='grasp_home_data'
        clean,executed,*_=v.tick(10**9,[0.]*6)
        self.assertEqual(v.plan['noise'].sample(.5),[0.]*6)
        v.phase='grasp_move';v.plan.pop('noise');v.params['dart']=dict(v.params['dart'],on_probability=1.)
        clean,executed,*_=v.tick(10**9,[0.]*6)
        self.assertNotEqual(v.plan['noise'].sample(.5),[0.]*6);self.assertEqual(clean[5],executed[5])

    def test_nominal_home_hold_overrides_a_nearby_previous_target(self):
        c,r,_=probes.ReviewRegressions().collector();c.servo.plan=None
        goal=_grasp_home_data_target();c.servo.reference=[goal[n]+.006 for n in JOINTS[:5]]+[.113]
        c.hold_home()
        self.assertEqual(c.servo.reference[:5],[goal[n] for n in JOINTS[:5]])
        self.assertEqual(c.servo.reference[5],.113)
        self.assertEqual(c.servo.reference_epoch,1)

    def test_settle_home_uses_unchanged_six_tick_thresholds_and_a_bounded_sim_window(self):
        c,r,_=probes.ReviewRegressions().collector();c.servo.plan=None;goal=_grasp_home_data_target();samples=[]
        def advance(count):
            i=len(samples);delta=.012 if i<5 else .002
            row=dict(state_source='isaac_articulation',state=[goal[n]+delta for n in JOINTS[:5]]+[0.],
                     velocity=[.1]*6,sim_ns=10**9+round(i*1e9/30))
            samples.append(row);return [row]
        c.advance_ticks=advance
        result=c.settle_home('after_reset')
        self.assertTrue(result['passed']);self.assertEqual(len(samples),11)
        self.assertAlmostEqual(result['checks'][-1]['home_error_rad'],.002)
        self.assertEqual(result['checks'][-1]['position_spread_rad'],0.)
        samples.clear()
        def never_stable(count):
            samples.append(1)
            return [dict(state_source='isaac_articulation',state=[goal[n]+.012 for n in JOINTS[:5]]+[0.],
                         velocity=[0.]*6,sim_ns=10**9+round(len(samples)*1e9/30))]
        c.advance_ticks=never_stable
        with self.assertRaises(QualityReject) as failed:c.settle_home('after_reset')
        self.assertEqual(failed.exception.kind,'home_unstable');self.assertEqual(len(samples),150)


class QualityPolicyTests(unittest.TestCase):
    def good(self):return dict(accepted=True,end_cancel=dict(held=True,failure=None),retirement=dict(passed=True),timing=dict(passed=True,contract_passed=True))

    def test_saved_pilot_descent_is_a_typed_physical_deadline_at_the_same_tick(self):
        fixture=json.loads(Path(__file__).with_name('fixtures').joinpath('descent_410019.json').read_text())
        c,r,_=probes.ReviewRegressions().collector();v=c.servo
        c.generation.epoch=fixture['epoch'];v.reference=list(fixture['reference']);v.reference_epoch=fixture['epoch']
        duration=round(fixture['planned_duration_s']*1e9)
        point=NS(positions=fixture['reference'][:5],time_from_start=NS(sec=duration//10**9,nanosec=duration%10**9))
        v.plan=dict(epoch=fixture['epoch'],start=fixture['operation_start_sim_ns'],done=threading.Event(),
                    trajectory=NS(joint_names=JOINTS[:5],points=[point]))
        v.phase=fixture['phase']
        v.limiter=TargetLimiter([(-3.,3.)]*5)
        prior=v.limiter.propose(fixture['reference'],[0.]*6,fixture['reference'],fixture['epoch'],fixture['last_recorded_sim_ns'])
        v.limiter.commit(prior,fixture['epoch'],fixture['last_recorded_sim_ns'])
        with self.assertRaises(ConvergenceFailure) as failed:v.tick(fixture['failure_sim_ns'],fixture['state'])
        evidence=failed.exception.evidence
        self.assertEqual(evidence['kind'],'arm_nonconvergence')
        self.assertEqual(evidence['elapsed_sim_seconds'],7.7)
        self.assertGreater(max(abs(a-b) for a,b in zip(fixture['state'][:5],fixture['reference'][:5])),.008)

    def test_three_identical_quality_rejects_stop_and_success_resets_streak(self):
        streak=FailureStreak();bad=dict(kind='quality_reject',reason='arm_nonconvergence')
        self.assertFalse(streak.observe(bad)['stop']);self.assertFalse(streak.observe(bad)['stop'])
        self.assertTrue(streak.observe(bad)['stop'])
        ledger=[dict(disposition=bad)]*3
        resumed=FailureStreak(ledger);self.assertEqual(resumed.count,3)
        streak=FailureStreak();streak.observe(bad);streak.observe(bad)
        streak.observe(dict(kind='accepted',reason='success'));self.assertEqual(streak.count,0)
        self.assertFalse(streak.observe(bad)['stop'])
        self.assertTrue(streak.observe(dict(kind='terminal',reason='clock'))['stop'])
        terminal=FailureStreak([dict(disposition=dict(kind='terminal',reason='clock'))]);self.assertTrue(terminal.stopped)
        with self.assertRaises(ValueError):FailureStreak(ledger+[dict(disposition=dict(kind='accepted',reason='success'))])

    def test_only_stationary_and_short_duration_are_quality_audit_failures(self):
        rows=contracts.ContractTests().rows()
        for row in rows:row.update(state=[0.]*6,action=[.035]*6,expert_reference=[1.]*6,physics_state=[0.]*6)
        audit=timing_audit(rows,30)
        self.assertFalse(audit['passed']);self.assertTrue(audit['contract_passed'])
        r=dict(accepted=False,end_cancel=dict(held=True,failure=None),script_completed=True,score=dict(passed=True),retirement=dict(passed=True),timing=audit)
        self.assertEqual(disposition(r)['reason'],'stationary')
        rows[20]['application_stamp_ns']+=5*10**9;r['timing']=timing_audit(rows,30)
        self.assertEqual(disposition(r)['kind'],'terminal')

    def test_pipeline_faults_override_quality_evidence_and_unknown_errors_stop(self):
        r=self.good();r['quality_reject']=dict(kind='home_unstable')
        for error in ('clock','reset','epoch','transport','scene'):
            self.assertEqual(disposition(r,fatal=error)['kind'],'terminal')
        r['end_cancel']['failure']='native clock fault';self.assertEqual(disposition(r)['reason'],'native_cancel')
        r['end_cancel']['failure']=None
        r['retirement']['passed']=False;self.assertEqual(disposition(r)['reason'],'worker_retirement')
        self.assertEqual(disposition(dict(retirement=dict(passed=True),error='CancelledGeneration'))['kind'],'terminal')

    def test_recorded_convergence_cancel_must_be_complete_and_validated(self):
        r=dict(accepted=False,end_cancel=dict(held=True,failure=None),original_capture_error_preserved=True,retirement=dict(passed=True),
            timing=dict(passed=False,contract_passed=True),capture_failures=[dict(epoch=3,validated_sim_tick=True,
                physical_convergence=dict(kind='arm_nonconvergence',epoch=3),cancellation=dict(completed=True,result=dict(held=True,failure=None)))])
        self.assertEqual(disposition(r),dict(kind='quality_reject',reason='arm_nonconvergence'))
        r['capture_failures'][0]['validated_sim_tick']=False
        self.assertEqual(disposition(r)['kind'],'terminal')

    def test_camera_join_failure_records_active_epoch_and_is_terminal(self):
        c,r,_=probes.ReviewRegressions().collector()
        def missing(**kwargs):c.stop.set();raise TimeoutError('no images')
        c.sensors.read=missing;c.capture()
        self.assertEqual(c.capture_failures[0]['epoch'],1)
        self.assertTrue(c.stop.is_set());self.assertIsNone(c.generation.epoch)
        self.assertTrue(c.failure);self.assertEqual(r.epoch,2)

    def test_ack_timeout_and_bad_frame_do_not_become_physical_quality(self):
        for error in [TimeoutError('late ACK'),ValueError('bad frame')]:
            c,r,_=probes.ReviewRegressions().collector()
            def fail(*a,**kwargs):raise error
            c.sensors.read_ack=fail;c.capture()
            self.assertTrue(c.failure);self.assertTrue(c.generation.stalled)
            self.assertNotIn('physical_convergence',c.capture_failures[0])

    def test_stale_worker_cannot_register_a_new_descendant_after_retirement(self):
        c,r,_,_,_=motions.MotionOperationTests().setup_motion();n=c.node
        registry=WorkerRegistry(c.generation,n._motion_thread,n._motion_token)
        entered=threading.Event();release=threading.Event();effects=[]
        def old():
            entered.set();release.wait(2)
            registry.thread(target=lambda:effects.append('new')).start()
        t=registry.thread(target=old);t.start();self.assertTrue(entered.wait(1))
        c.timeline(False);release.set()
        result=registry.retire(timeout=2)
        self.assertTrue(result['passed']);self.assertFalse(effects);self.assertEqual(result['workers'],1)

    def test_worker_retirement_refuses_a_still_running_continuation(self):
        c,r,_,_,_=motions.MotionOperationTests().setup_motion();n=c.node
        registry=WorkerRegistry(c.generation,n._motion_thread,n._motion_token);entered=threading.Event();release=threading.Event()
        worker=registry.thread(target=lambda:(entered.set(),release.wait(2)));worker.start();self.assertTrue(entered.wait(1))
        c.timeline(False)
        with self.assertRaises(RuntimeError):registry.retire(timeout=.02)
        release.set();self.assertTrue(registry.retire(timeout=2)['passed'])
if __name__=='__main__':unittest.main()
