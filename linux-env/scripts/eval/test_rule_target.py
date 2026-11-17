import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'collect'))
from rule_target import select_target

class TargetRuleTests(unittest.TestCase):
    def snapshot(self,a=(.2,0,0),b=(.3,0,0)):
        return dict(base_world_pose=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]),
                    bodies={f'red_2x3_{i}':dict(position=list(p)) for i,p in enumerate((a,b))})
    def test_nearest_uses_xy_not_height(self):
        p=select_target(self.snapshot(a=(.2,0,2),b=(.3,0,0)),'red',True)
        self.assertEqual(p['object'],'red_2x3_0');self.assertFalse(p['ambiguous'])
    def test_ambiguous_refuses_or_uses_fixed_base_y(self):
        s=self.snapshot(a=(.2,.7,0),b=(.2,.5,0));s['base_world_pose']=dict(position=[.4,.6,0.],orientation=[0.,0.,0.,1.])
        p=select_target(s,'red');self.assertTrue(p['ambiguous']);self.assertEqual(p['object'],'red_2x3_0')
        with self.assertRaises(ValueError):select_target(s,'red',True)
    def test_tie_boundary_and_invalid_coordinates(self):
        self.assertTrue(select_target(self.snapshot(a=(0,0,0),b=(.005,0,0)),'red')['ambiguous'])
        with self.assertRaises(ValueError):select_target(self.snapshot(a=(math.nan,0,0)),'red')
        with self.assertRaises(ValueError):select_target(self.snapshot(),'purple')

if __name__=='__main__':unittest.main()
