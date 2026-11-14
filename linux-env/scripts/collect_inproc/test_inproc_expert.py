import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
from expert import Expert
from contract import JOINTS


class ExpertGeometryTests(unittest.TestCase):
    def test_cycle_keeps_shared_yaw_specific_solutions_and_opening(self):
        expert=Expert.__new__(Expert)
        expert.params=dict(object='target',color='blue',recovery='none',approach_height_m=.02,
            approach_offset_m=[.003,-.002])
        expert.reference=[0.]*6;expert.executed_hold=[0.]*6;expert.grasp_specs=[]
        height=[.01]
        expert.backend=NS(snapshot=lambda:dict(state=[0.]*6,bodies={'target':dict(position=[.2,0,height[0]])}))
        expert.geometry=NS(selected=None,contact=False,carried=None,bind_carried=lambda snapshot:None)
        pose=dict(x=.2,y=0,z=.01,qw=1.,qz=0.);bbox=dict(sx=.03,sy=.02,sz=.01)
        expert.current_geometry=lambda:(expert.backend.snapshot(),pose,bbox,{'drop_1':dict(x=.3,y=.1,z=.05)})
        expert.emit=Mock();expert.grip=Mock();expert.select=Mock(return_value=[0.]*5)
        def arm(target,phase):
            if phase=='grasp_home':height[0]+=.04
        expert.arm=Mock(side_effect=arm)
        selected=dict(chosen_yaw=3.14,opening_rad=.73,closing_rad=.14,
            solutions=dict(approach=['yaw-specific-approach'],final=['yaw-specific-final']),
            geometry=dict(tcp=[.2,.001,.01]),attempts=[dict(yaw=3.14)])
        with patch('expert.make_reachable_grasp_plan',return_value=selected) as planner,patch(
                'expert.geometric_ik',return_value=[dict.fromkeys(JOINTS[:5],0.)]):
            result=expert.cycle()
        planner.assert_called_once_with(pose,bbox,current_wrist_roll=0.,approach_height=.02,approach_offset=[.003,-.002])
        self.assertEqual(expert.grip.call_args_list[0].args,(.73,'grasp_open'))
        self.assertEqual(expert.grip.call_args_list[1].args,(.14,'grasp_close'))
        self.assertEqual(expert.select.call_args_list[0].args,(['yaw-specific-approach'],))
        self.assertEqual(expert.select.call_args_list[1].args,(['yaw-specific-final'],))
        self.assertEqual(expert.grasp_specs[0]['geometry'],selected['geometry'])
        self.assertAlmostEqual(result['lift_m'],.04)


if __name__=='__main__':unittest.main()
