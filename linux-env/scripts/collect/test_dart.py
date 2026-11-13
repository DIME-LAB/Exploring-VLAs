"""Smooth perturbation continuity, seeded replay, bounds and ACK ownership."""
import math,random,unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
import yaml
from dart import DEFAULT_NOISE,SmoothDart,TargetLimiter,ExpertSchedule
from types import SimpleNamespace as NS


class DartTests(unittest.TestCase):
    def test_seeded_segments_have_smooth_zero_endpoints_and_no_gripper_noise(self):
        config=dict(DEFAULT_NOISE,on_probability=1.)
        for seed in range(20):
            p=SmoothDart(3.,seed,config)
            self.assertEqual(p.spec,SmoothDart(3.,seed,config).spec)
            self.assertEqual(p.sample(0.),[0.]*6);self.assertEqual(p.sample(2.4),[0.]*6)
            dt=1/300;values=[p.sample(i*dt) for i in range(901)]
            velocity=[[(b[j]-a[j])/dt for j in range(5)] for a,b in zip(values,values[1:])]
            accel=[[(b[j]-a[j])/dt for j in range(5)] for a,b in zip(velocity,velocity[1:])]
            self.assertLessEqual(max(abs(v) for row in velocity for v in row),.050001)
            self.assertLessEqual(max(abs(v) for row in accel for v in row),.30001)
            self.assertTrue(all(row[5]==0. for row in values))
            self.assertLessEqual(max(abs(v) for row in values for v in row),math.radians(1.))

    def test_short_operations_scale_or_skip_instead_of_an_abrupt_truncated_segment(self):
        for duration in (.2,.7,1.,1.5):
            p=SmoothDart(duration,7,dict(DEFAULT_NOISE,on_probability=1.))
            self.assertEqual(p.sample(duration*.8),[0.]*6)
            self.assertEqual(p.sample(duration),[0.]*6)
        self.assertEqual(SmoothDart(2.,7,DEFAULT_NOISE,enabled=False).segments,[])

    def test_target_derivatives_and_joint_braking_room_are_bounded(self):
        limiter=TargetLimiter([(-.15,.15)]*5)
        previous=[0.]*6;velocity=[0.]*5;rng=random.Random(17)
        for tick in range(1000):
            goal=[rng.uniform(-.3,.3) for _ in range(5)]+[.2]
            candidate=limiter.propose(goal,[0.]*6,previous,3,tick)
            self.assertEqual(candidate.positions[5],.2)
            next_velocity=[(candidate.positions[j]-previous[j])*30 for j in range(5)]
            self.assertLessEqual(max(abs(v) for v in next_velocity),10.+1e-6)
            self.assertLessEqual(max(abs(v-a)*30 for v,a in zip(next_velocity,velocity)),2.+1e-5)
            self.assertTrue(all(-.150001<=v<=.150001 for v in candidate.positions[:5]))
            limiter.commit(candidate,3,tick);previous=list(candidate.positions);velocity=next_velocity
        for tick in range(1000,1300):
            c=limiter.propose([0.]*6,[0.]*6,previous,3,tick);limiter.commit(c,3,tick);previous=list(c.positions)
        self.assertLess(max(abs(v) for v in previous[:5]),1e-6)
        self.assertLess(max(abs(v) for v in c.velocities),1e-6)

    def test_candidate_history_changes_only_after_matching_ack_and_can_cross_idle_epoch(self):
        limiter=TargetLimiter([(-2.,2.)]*5)
        candidate=limiter.propose([.1]*6,[0.]*6,[0.]*6,1,100)
        self.assertIsNone(limiter.state)
        with self.assertRaises(ValueError):limiter.commit(candidate,2,100)
        with self.assertRaises(FrozenInstanceError):candidate.epoch=2
        limiter.commit(candidate,1,100);limiter.handover(2)
        self.assertEqual(limiter.state.positions,candidate.positions)
        self.assertEqual(limiter.state.epoch,2)
        with self.assertRaises(ValueError):limiter.propose([.1]*6,[0.]*6,[0.]*6,1,133)

    def test_expert_retiming_preserves_path_and_obeys_its_lower_derivative_budget(self):
        points=[NS(positions=[i/20]*5,time_from_start=NS(sec=(i//20),nanosec=round((i%20)/20*1e9))) for i in range(21)]
        schedule=ExpertSchedule(NS(points=points),DEFAULT_NOISE)
        self.assertGreater(schedule.duration,1.)
        self.assertEqual(schedule.sample(0.),[0.]*5);self.assertEqual(schedule.sample(schedule.duration),[1.]*5)
        values=[schedule.sample(0.)]*2+[schedule.sample(min(schedule.duration,i/30)) for i in range(1,math.ceil(schedule.duration*30)+1)]
        values+=[values[-1]]*2
        velocity=[(b[0]-a[0])*30 for a,b in zip(values,values[1:])]
        accel=[(b-a)*30 for a,b in zip(velocity,velocity[1:])]
        self.assertLessEqual(max(abs(v) for v in velocity),1.000001)
        self.assertLessEqual(max(abs(v) for v in accel),1.400001)
        self.assertTrue(all(abs(row[0]-row[4])<1e-12 for row in values))

    def test_corner_retiming_keeps_the_original_polyline_without_shortcuts(self):
        points=[NS(positions=q,time_from_start=NS(sec=0,nanosec=t)) for t,q in
                [(0,[0.]*5),(300000000,[.5,0.,0.,0.,0.]),(600000000,[.5,.5,0.,0.,0.])]]
        schedule=ExpertSchedule(NS(points=points),DEFAULT_NOISE)
        self.assertFalse(schedule.original);self.assertEqual(len(schedule.segments),2)
        positions=[schedule.sample(i*schedule.duration/200) for i in range(201)]
        self.assertTrue(all(abs(q[1])<1e-12 or abs(q[0]-.5)<1e-12 for q in positions))
        sources=[schedule.source_time(i*schedule.duration/200) for i in range(201)]
        self.assertEqual(sources,sorted(sources))

    def test_cancel_can_retire_while_expert_retiming_is_in_progress(self):
        import threading
        import test_motion_operations as motions
        c,r,_,_,_=motions.MotionOperationTests().setup_motion();v=c.servo;v.plan=None
        event=c.node._new_motion_event();trajectory=NS(joint_names=list(('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')),
            points=[NS(positions=[0.]*5,time_from_start=NS(sec=1,nanosec=0))])
        entered=threading.Event();release=threading.Event();scope=v.execute.__func__.__globals__;original=scope['ExpertSchedule']
        def slow(*args):entered.set();release.wait(2);return original(*args)
        scope['ExpertSchedule']=slow
        try:
            worker=c.node._motion_thread(target=lambda:v.execute(trajectory,event));worker.start();self.assertTrue(entered.wait(1))
            self.assertEqual(c.timeline(False)['epoch'],2)
            release.set();worker.join(2);self.assertFalse(worker.is_alive());self.assertIsNone(v.plan)
        finally:release.set();scope['ExpertSchedule']=original

    def test_limits_do_not_exceed_the_deployment_yaml(self):
        root=Path(__file__).resolve().parents[3]
        cfg=yaml.safe_load((root/'vla_SO-ARM101/src/so_arm101_moveit_config/config/joint_limits.yaml').read_text())
        for joint in ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'):
            self.assertLessEqual(DEFAULT_NOISE['arm_velocity_rad_s'],cfg['joint_limits'][joint]['max_velocity'])
            self.assertLessEqual(DEFAULT_NOISE['arm_acceleration_rad_s2'],cfg['joint_limits'][joint]['max_acceleration'])
        with self.assertRaises(ValueError):SmoothDart(3.,1,dict(DEFAULT_NOISE,arm_acceleration_rad_s2=3.))
if __name__=='__main__':unittest.main()
