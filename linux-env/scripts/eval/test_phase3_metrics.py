"""Geometry and report regressions using the shared native containment predicate."""
import importlib.util,json,math,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[3]
COLLECT=REPO/'linux-env/scripts/collect'

@unittest.skipUnless(COLLECT.is_dir(),'Run from repository deployment environment')
class PhysicalMetricTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0,str(COLLECT))
        import numpy as np
        from phase3_metrics import PhysicalScore
        self.fk=SimpleNamespace(forward_kinematics_batch=lambda q:np.asarray([[0.,0.,.02]]))
        self.Score=PhysicalScore
    def snapshot(self,t,color='red',inside=False,lift=.0,cup_color='red'):
        q=[1.,0.,0.,0.]
        # Native cup origin is its base; a tiny whole block at .02m clears the floor.
        bodies={f'cup_{c}':dict(position=[float(i),0,0],orientation=q,velocity=[0.]*6) for i,c in enumerate(('red','green','blue'))}
        bodies['red_2x3_0']=dict(position=[float(('red','green','blue').index(cup_color)) if inside else .4,0.,.02+lift],orientation=q,velocity=[0.]*6)
        corners=[[x,y,z] for x in (-.005,.005) for y in (-.005,.005) for z in (-.004,.004)]
        return dict(sim_ns=round(t*1e9),state=[0.]*6,bodies=bodies,local_corners={'red_2x3_0':corners},base_world_pose=dict(position=[0.,0.,0.],orientation=q),robot_base_world_pose=dict(position=[0.,0.,0.],orientation=q))
    def test_initial_block_already_in_cup_does_not_count_as_lift_place(self):
        s=self.Score('red',self.snapshot(0,inside=True),self.fk)
        for n in range(31):s.update(self.snapshot(n/30,inside=True))
        self.assertFalse(s.result()['success']);self.assertFalse(s.result()['grasp_success'])
    def test_lift_then_containment_then_leave_cup(self):
        s=self.Score('red',self.snapshot(0),self.fk)
        for n in range(31):s.update(self.snapshot(n/30,lift=.04))
        self.assertTrue(s.result()['grasp_success']);self.assertFalse(s.result()['success'])
        for n in range(31,62):s.update(self.snapshot(n/30,inside=True))
        self.assertTrue(s.result()['success'])
        s.update(self.snapshot(62/30,inside=False));self.assertFalse(s.result()['success'])
    def test_requested_block_in_wrong_cup_is_not_success(self):
        s=self.Score('red',self.snapshot(0),self.fk)
        for n in range(31):s.update(self.snapshot(n/30,lift=.04))
        for n in range(31,62):s.update(self.snapshot(n/30,inside=True,cup_color='green'))
        result=s.result()
        self.assertTrue(result['target_any_placement']);self.assertTrue(result['target_wrong_cup'])
        self.assertFalse(result['correct_cup']);self.assertFalse(result['success'])

    def test_duplicate_samples_do_not_extend_settling(self):
        s=self.Score('red',self.snapshot(0),self.fk)
        for _ in range(99):s.update(self.snapshot(0,lift=.05))
        self.assertFalse(s.result()['grasp_success'])
    def test_missing_physical_ticks_fail_closed(self):
        s=self.Score('red',self.snapshot(0),self.fk);s.update(self.snapshot(0))
        with self.assertRaises(RuntimeError):s.update(self.snapshot(.2))

    def two_blocks(self,t,far_inside=False,far_lift=0.,near_inside=False,near_lift=0.):
        s=self.snapshot(t,inside=near_inside,lift=near_lift)
        far=self.snapshot(t,inside=far_inside,lift=far_lift)
        if not far_inside:far['bodies']['red_2x3_0']['position'][0]=.6
        s['bodies']['red_2x3_1']=far['bodies']['red_2x3_0']
        s['local_corners']['red_2x3_1']=s['local_corners']['red_2x3_0']
        return s
    def test_right_colour_wrong_instance_is_not_rule_success(self):
        score=self.Score('red',self.two_blocks(0),self.fk,intended_target='red_2x3_0')
        for n in range(31):score.update(self.two_blocks(n/30,far_lift=.04))
        for n in range(31,62):score.update(self.two_blocks(n/30,far_inside=True))
        result=score.result();self.assertTrue(result['requested_colour_success'])
        self.assertFalse(result['rule_target_grasp']);self.assertFalse(result['rule_target_success'])
        self.assertTrue(result['wrong_same_colour_target_grasp']);self.assertTrue(result['wrong_same_colour_target_placement'])
    def test_intended_instance_places_and_old_default_keys_stay_unchanged(self):
        score=self.Score('red',self.two_blocks(0),self.fk,intended_target='red_2x3_0')
        for n in range(31):score.update(self.two_blocks(n/30,near_lift=.04))
        for n in range(31,62):score.update(self.two_blocks(n/30,near_inside=True))
        self.assertTrue(score.result()['rule_target_success'])
        self.assertNotIn('rule_target_success',self.Score('red',self.two_blocks(0),self.fk).result())

    def test_closure_distance_stays_with_intended_object_not_nearest_any(self):
        import numpy as np
        fk=SimpleNamespace(forward_kinematics_batch=lambda q:np.asarray([[.6,0.,.02]]))
        first=self.two_blocks(0);first['state'][5]=1.
        score=self.Score('red',first,fk,intended_target='red_2x3_0');score.update(first)
        second=self.two_blocks(1/30);second['state'][5]=.9;score.update(second)
        self.assertAlmostEqual(score.result()['closing_distance_m'],0.)
        self.assertAlmostEqual(score.result()['rule_closing_distance_m'],.2)

if __name__=='__main__':unittest.main(verbosity=2)
