"""Retained descent obstructions exercise the shared GUI/collector grasp geometry."""
import json,math,unittest
from pathlib import Path
import numpy as np
from contract import JOINTS
from so_arm101_control.grasp_geometry import make_reachable_grasp_plan
from obstruction_model import finger_boxes,separation

FIXTURE=json.loads(Path(__file__).with_name('fixtures').joinpath('descent-obstructions.json').read_text())


class GraspObstructionTests(unittest.TestCase):
    def test_retained_diagonal_commands_overlap_the_original_target(self):
        for seed in FIXTURE['seeds']:
            if seed['seed'] not in (330001,350010):continue
            boxes=finger_boxes(FIXTURE,seed['command']);target=seed['initial_boxes'][seed['object']]
            with self.subTest(seed=seed['seed']):self.assertLess(min(separation(b,target) for b in boxes),-.0003)

    def test_every_new_descent_stays_clear_of_target_and_table(self):
        for seed in FIXTURE['seeds']:
            p=seed['params'];plan=make_reachable_grasp_plan(seed['pose'],seed['bbox'],current_wrist_roll=-1.5220963267948966,
                approach_height=p['approach_height_m'],approach_offset=p['approach_offset_m'])
            with self.subTest(seed=seed['seed']):
                self.assertIsNotNone(plan['chosen_yaw']);self.assertAlmostEqual(math.sin(plan['chosen_yaw']-plan['requested_yaw']),0.,places=12)
                self.assertGreaterEqual(plan['opening_margin_m'],.006999)
                start=plan['solutions']['approach'][0];finish=plan['solutions']['final'][0];target=seed['initial_boxes'][seed['object']]
                gap=1.;floor=1.
                for u in np.linspace(0.,1.,101):
                    q=[start[n]+u*(finish[n]-start[n]) for n in JOINTS[:5]]+[plan['opening_rad']]
                    boxes=finger_boxes(FIXTURE,q);gap=min(gap,min(separation(b,target) for b in boxes));floor=min(floor,min(b['min_z_m'] for b in boxes))
                print('shared narrow-face descent',seed['seed'],'yaw',math.degrees(plan['chosen_yaw']),'gap_m',gap,'floor_m',floor)
                self.assertGreater(gap,0.);self.assertGreaterEqual(floor,.0012)
