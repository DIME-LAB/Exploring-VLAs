"""Small near-block DART remains nonzero inside the measured jaw corridor."""
import json,math,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from contract import JOINTS
from dart import SmoothDart,ClearanceDart,DEFAULT_NOISE,PlannerSchedule
from so_arm101_control.grasp_geometry import make_reachable_grasp_plan
from so_arm101_control.compute_workspace import forward_kinematics
from finger_geometry import finger_boxes,separation

FIXTURE=json.loads(Path(__file__).with_name('fixtures').joinpath('descent-obstructions.json').read_text())


class ClearanceDartTests(unittest.TestCase):
    def program(self,seed,noise_seed=1):
        p=seed['params'];g=make_reachable_grasp_plan(seed['pose'],seed['bbox'],current_wrist_roll=-1.5220963,approach_height=p['approach_height_m'],approach_offset=p['approach_offset_m'])
        a=g['solutions']['approach'][0];b=g['solutions']['final'][0]
        trajectory=NS(points=[NS(positions=[a[n] for n in JOINTS[:5]],time_from_start=NS(sec=0,nanosec=0)),NS(positions=[b[n] for n in JOINTS[:5]],time_from_start=NS(sec=3,nanosec=0))])
        schedule=PlannerSchedule(trajectory);base=SmoothDart(3.,noise_seed,dict(DEFAULT_NOISE,on_probability=1.))
        program=ClearanceDart(base,schedule,forward_kinematics,seed['pose']['z']+seed['bbox']['sz']/2,g['geometry']['tcp'][2],.001)
        return program,schedule,g

    def test_all_retained_descent_fixtures_stay_clear_with_nonzero_near_noise(self):
        nonzero=0
        for seed in FIXTURE['seeds']:
            for noise_seed in (1,2,3):
                program,schedule,g=self.program(seed,noise_seed);target=seed['initial_boxes'][seed['object']]
                for tick in range(91):
                    t=tick/30;q=schedule.sample(t);offset=program.sample(t);boxes=finger_boxes(FIXTURE,[a+b for a,b in zip(q,offset[:5])]+[g['opening_rad']])
                    self.assertGreater(min(separation(b,target) for b in boxes),0.)
                    self.assertGreater(min(b['min_z_m'] for b in boxes),.0012)
                    self.assertEqual(offset[5],0.)
                stats=program.phase_statistics();nonzero+=stats.get('near_block',{}).get('nonzero_ticks',0)
                self.assertEqual(program.sample(3.),[0.]*6)
        self.assertGreater(nonzero,0)

    def test_wrapper_changes_only_noise_and_respects_derivative_limits(self):
        program,schedule,g=self.program(FIXTURE['seeds'][0],2);original=list(schedule.path)
        values=[program.sample(i/120) for i in range(361)];vel=np.diff(values,axis=0)*120;acc=np.diff(vel,axis=0)*120
        self.assertLessEqual(float(abs(vel).max()),.050001);self.assertLessEqual(float(abs(acc).max()),.300001)
        self.assertEqual(schedule.path,original)
        self.assertTrue(program.spec['clearance_policy']);self.assertGreater(program.near_budget,0.)
