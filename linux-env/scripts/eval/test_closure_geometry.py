import copy,math,unittest
import numpy as np
from closure_geometry import confirmed_geometry

class ClosureGeometryTests(unittest.TestCase):
    def sample(self):
        m=np.eye(4);m[:3,3]=[.01,.02,.03]
        return dict(epoch=2,sim_ns=30,grasp_telemetry=dict(tcp_world_matrix=m.tolist()),bodies={'red_2x3_0':dict(position=[0,0,0],orientation=[1,0,0,0])})
    def test_signed_positions_and_relative_axis_yaw(self):
        result=confirmed_geometry(self.sample(),'red_2x3_0')
        self.assertAlmostEqual(result['xy_mm'],math.sqrt(500));self.assertAlmostEqual(result['dz_mm'],30)
        self.assertEqual(result['block_in_tcp_frame_mm'],[-10,-20,-30]);self.assertEqual(result['yaw_deg'],-90)
    def test_object_symmetry_and_degenerate_projection(self):
        s=self.sample();a=confirmed_geometry(s,'red_2x3_0');s['bodies']['red_2x3_0']['orientation']=[0,0,0,1]
        self.assertEqual(a['yaw_deg'],confirmed_geometry(s,'red_2x3_0')['yaw_deg'])
        s=self.sample();m=np.asarray(s['grasp_telemetry']['tcp_world_matrix']);m[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]];s['grasp_telemetry']['tcp_world_matrix']=m.tolist()
        self.assertIsNone(confirmed_geometry(s,'red_2x3_0')['yaw_deg'])
    def test_missing_native_pose_never_substitutes_fk(self):
        s=self.sample();s.pop('grasp_telemetry');self.assertEqual(confirmed_geometry(s,'red_2x3_0')['status'],'unavailable')

if __name__=='__main__':unittest.main()
