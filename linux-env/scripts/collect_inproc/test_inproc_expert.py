import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
from expert import Expert
from contract import JOINTS


class ExpertGeometryTests(unittest.TestCase):
    def test_already_open_restored_jaw_has_no_stationary_motion_interval(self):
        expert=Expert.__new__(Expert)
        target=.11324631074373954
        expert.reference=[0.]*5+[.11321702599525452]
        expert.backend=NS(snapshot=lambda:dict(sim_ns=123,state=list(expert.reference),
            drive_targets=[0.]*5+[target],velocity=[0.]*5+[1.812512709875591e-5]),stamp=Mock())
        expert.emit=Mock()
        proof=expert.grip(target,'grasp_open')
        expert.emit.assert_not_called();expert.backend.stamp.assert_not_called()
        self.assertEqual(expert.reference[5],target)
        self.assertTrue(proof['settled_at_goal']);self.assertEqual(proof['sim_ns'],123)

    def test_jaw_in_motion_or_away_from_goal_still_executes_ticks(self):
        for position,velocity,drive in ((.113217,.01,.113246),(.04649,0.,.113246),(.113217,0.,.04649)):
            with self.subTest(position=position,velocity=velocity):
                expert=Expert.__new__(Expert);expert.reference=[0.]*5+[position]
                expert.executed_hold=list(expert.reference)
                expert.backend=NS(snapshot=lambda:dict(sim_ns=0,state=[0.]*5+[position],drive_targets=[0.]*5+[drive],
                    velocity=[0.]*5+[velocity]),stamp=lambda:0)
                expert.emit=Mock(side_effect=RuntimeError('first executed tick'))
                with self.assertRaisesRegex(RuntimeError,'first executed tick'):
                    expert.grip(.113246,'grasp_open')
                expert.emit.assert_called_once()

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
