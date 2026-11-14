"""Smooth noise, original planner geometry and final swept-command checks."""
import math,unittest,threading
from types import SimpleNamespace as NS
from dataclasses import FrozenInstanceError
from dart import DEFAULT_NOISE,SmoothDart,PlannerSchedule,TargetCandidate
from command_geometry import checked_segment,choose_noise_scale,final_command,CommandCollision
import time


def path(points,duration=2.):
    return NS(joint_names=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'),points=[NS(positions=q,time_from_start=NS(sec=int(t),nanosec=round(t%1*1e9))) for t,q in points])


class DartTests(unittest.TestCase):
    def test_seeded_segments_have_smooth_zero_endpoints_and_no_gripper_noise(self):
        config=dict(DEFAULT_NOISE,on_probability=1.)
        for seed in range(20):
            p=SmoothDart(3.,seed,config);self.assertEqual(p.spec,SmoothDart(3.,seed,config).spec)
            self.assertEqual(p.sample(0.),[0.]*6);self.assertEqual(p.sample(2.4),[0.]*6)
            values=[p.sample(i/300) for i in range(901)]
            v=[[(b[j]-a[j])*300 for j in range(5)] for a,b in zip(values,values[1:])]
            a=[[(b[j]-a[j])*300 for j in range(5)] for a,b in zip(v,v[1:])]
            self.assertLessEqual(max(abs(x) for row in v for x in row),.050001)
            self.assertLessEqual(max(abs(x) for row in a for x in row),.30001)
            self.assertTrue(all(row[5]==0. for row in values))

    def test_every_short_segment_preserves_minimum_hold_and_ramp_or_skips(self):
        for duration in (.2,.7,.8,1.,1.3,1.5,2.,3.):
            for seed in range(30):
                p=SmoothDart(duration,seed,dict(DEFAULT_NOISE,on_probability=1.))
                for s in p.segments:
                    self.assertGreaterEqual(s['hold_s'],.3);self.assertGreaterEqual(s['ramp_s'],.35-1e-12)
                self.assertEqual(p.sample(duration*.8),[0.]*6)
        self.assertEqual(SmoothDart(.8,7,dict(DEFAULT_NOISE,on_probability=1.)).segments,[])

    def test_planner_time_and_corner_polyline_are_unchanged(self):
        p=path([(0.,[0.]*5),(1.,[.5,0.,0.,0.,0.]),(2.,[.5,.5,0.,0.,0.])])
        schedule=PlannerSchedule(p);self.assertEqual(schedule.duration,2.)
        for i in range(61):
            q=schedule.sample(i/30);self.assertTrue(abs(q[1])<1e-12 or abs(q[0]-.5)<1e-12)
        self.assertEqual(schedule.sample(.5),[.25,0.,0.,0.,0.])

    def test_real_servo_zero_noise_corner_and_native_ack_stay_on_planned_polyline_with_lag(self):
        import json
        import numpy as np
        import test_motion_operations as motions
        from contract import JOINTS,corrective_action
        c,r,_,_,_=motions.MotionOperationTests().setup_motion();v=c.servo
        v.params['dart']=dict(DEFAULT_NOISE,on_probability=0.)
        event=c.node._new_motion_event();event.motion_tag='grasp_move'
        v.execute(path([(0.,[0.]*5),(1.,[.5,0.,0.,0.,0.]),(2.,[.5,.5,0.,0.,0.])]),event)
        self.assertEqual(v.plan['noise'].segments,[])
        q=[0.]*6;previous=[0.]*6;acks=[];measured_departure=[]
        r.ack_pub=NS(publish=lambda msg:acks.append(json.loads(msg.data)))
        for i in range(90):
            stamp=10**9+round(i/30*1e9);r.stamp=lambda:stamp;r.busy=False
            c.node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=stamp))
            clean,command,_,_,reference,_,_,advance,candidate=v.tick(stamp,q,previous)
            self.assertEqual(clean,corrective_action(q,reference))
            self.assertTrue(abs(command[1])<1e-12 or abs(command[0]-.5)<1e-12)
            self.assertTrue(c.validate_command(candidate,q,previous)['passed'])
            r.command_received(NS(data=json.dumps(dict(epoch=1,observation_stamp_ns=stamp,command=command,advance=advance))))
            self.assertNotIn('error',acks[-1]);np.testing.assert_allclose(acks[-1]['applied_command'],command,atol=1e-7)
            v.commit_command(candidate,1,stamp)
            q=[x+.15*(u-x) for x,u in zip(q,command)];previous=command
            measured_departure.append(min(abs(q[1]),abs(q[0]-.5)))
        self.assertGreater(max(measured_departure),.001)  # Measured lag leaves the line; every applied target stays on it.
        self.assertEqual(v.previous_command.planner_reference,tuple(reference))

    def test_lagging_gripper_receives_exact_clean_command_without_noise(self):
        import test_motion_operations as motions
        c,r,_,_,_=motions.MotionOperationTests().setup_motion();v=c.servo
        c.node._actual_positions['gripper_joint']=.5
        v.reference=[0.]*5+[.5];v.reference_epoch=1
        v.gripper(.1,.01,False)
        clean,executed,*_=v.tick(10**9,[0.]*5+[.5])
        self.assertAlmostEqual(clean[5],.465)
        self.assertEqual(executed[5],clean[5])

    def test_complete_noise_operation_scales_before_dispatch_when_corridor_is_narrow(self):
        schedule=PlannerSchedule(path([(0.,[0.]*5),(3.,[.1,0.,0.,0.,0.])]))
        program=SmoothDart(3.,7,dict(DEFAULT_NOISE,on_probability=1.))
        check=lambda q:(max(abs(v) for v in q[1:5])<.003,[])
        scale,report=choose_noise_scale(schedule,program,0.,check,[(-3.,3.)]*6,lambda:None)
        self.assertLess(scale,1.);self.assertTrue(report['attempts'][-1]['passed'])
        self.assertTrue(any(not a['passed'] for a in report['attempts']))

    def test_final_measured_to_command_segment_refuses_obstacle_on_lag_shortcut(self):
        candidate=TargetCandidate(1,10**9,(.5,.15,0.,0.,0.,0.),(0.,)*5,(0.,)*6,(0.,)*6,(.5,.15,0.,0.,0.,0.))
        check=lambda q:(not (.30<q[0]<.48 and .02<q[1]<.14),[])
        with self.assertRaises(CommandCollision):final_command(candidate,[.35,0.,0.,0.,0.,0.],[.5,.13,0.,0.,0.,0.],check,[(-3.,3.)]*6)
        self.assertTrue(final_command(candidate,[.5,.13,0.,0.,0.,0.],[.5,.13,0.,0.,0.,0.],check,[(-3.,3.)]*6)['passed'])

    def test_cancel_can_retire_while_preflight_query_is_in_progress(self):
        import test_motion_operations as motions
        c,r,_,_,_=motions.MotionOperationTests().setup_motion();v=c.servo;v.plan=None;v.noise_index=0;v.noise_specs=[]
        event=c.node._new_motion_event();entered=threading.Event();release=threading.Event()
        def slow(target):entered.set();release.wait(2);return True,[]
        c.node._check_state_valid_with_contacts=slow
        worker=c.node._motion_thread(target=lambda:v.execute(path([(0.,[0.]*5),(1.,[.1]*5)]),event));worker.start()
        self.assertTrue(entered.wait(1));self.assertEqual(c.timeline(False)['epoch'],2)
        release.set();worker.join(3);self.assertFalse(worker.is_alive());self.assertIsNone(v.plan)

if __name__=='__main__':unittest.main()
