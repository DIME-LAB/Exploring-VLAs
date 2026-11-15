import math,unittest
import numpy as np
from behaviour import diagnostics,fk
class BehaviourTests(unittest.TestCase):
 def fixture(self,close=True,rotated=False):
  arm=np.zeros(5);tip=fk.forward_kinematics_batch(arm[None,:])[0]
  base=dict(position=[.2,-.1,.05],orientation=[math.sqrt(.5),0,0,math.sqrt(.5)]) if rotated else dict(position=[0,0,0],orientation=[1,0,0,0])
  world=np.asarray([-tip[1],tip[0],tip[2]])+base['position'] if rotated else tip
  snaps=[]
  for i in range(31):
   grip=.06 if close and i>=15 else .12
   snaps.append(dict(sim_ns=round(i*1e9/30),state=list(arm)+[grip],robot_base_world_pose=base,bodies={c+'_0':dict(position=(world+np.asarray([d,0,0])).tolist()) for c,d in [('red',.02),('blue',.001),('green',.1)]}))
  return snaps,world
 def test_wrong_colour_closure_and_distances(self):
  snaps,world=self.fixture();d,_=diagnostics(snaps,'red');self.assertAlmostEqual(d['closing_time_sim_s'],.5);self.assertEqual(d['nearest_colour_at_closing'],'blue');self.assertFalse(d['target_nearer_at_closing']);self.assertAlmostEqual(d['closing_distance_m'],.02);self.assertAlmostEqual(d['min_tcp_other_m'],.001)
 def test_no_closure_and_constant_idle(self):
  snaps,_=self.fixture(False);d,_=diagnostics(snaps,'red');self.assertFalse(d['ever_closed_ge2deg']);self.assertIsNone(d['closing_time_sim_s']);self.assertAlmostEqual(d['idle_s'],1);self.assertAlmostEqual(d['tcp_dwell_pct'],100)
 def test_world_base_quaternion_transform(self):
  snaps,world=self.fixture(rotated=True);d,_=diagnostics(snaps,'red');np.testing.assert_allclose(d['closing_tcp_world'],world,atol=1e-8);self.assertAlmostEqual(d['closing_distance_m'],.02)
if __name__=='__main__':unittest.main()
