import copy,json,math,tempfile,unittest
from pathlib import Path
from training_starts import SEEDS,NAMES,digest,entry,validate_bank,select_mid

class StartTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  bodies={n:dict(position=[0,0,0],orientation=[1,0,0,0]) for n in sorted(NAMES)}
  for seed in SEEDS:
   (self.root/f'seed-{seed}.json').write_text(json.dumps(bodies))
   (self.root/f'state-{seed}.json').write_text(json.dumps(dict(seed=seed,color='red',task='Pick a red lego and place it in red cup',mode='mid',state=[0]*6,robot_base_world_pose=dict(position=[0]*3,orientation=[1,0,0,0]))))
  (self.root/'baseline.json').write_text(json.dumps(bodies));self.seal()
 def tearDown(self):self.tmp.cleanup()
 def seal(self):
  files=[p for p in self.root.glob('*.json') if p.name!='bank-manifest.json']
  (self.root/'bank-manifest.json').write_text(json.dumps(dict(kind='seen-training-start-diagnostic',complete=True,mode='mid',seeds=list(SEEDS),files_sha256={p.name:digest(p) for p in files})))
  self.sha=digest(self.root/'bank-manifest.json')
 def test_actual_training_colour_is_not_seed_modulo_three(self):
  self.assertEqual(entry(self.root,530000,self.sha)['color'],'red');self.assertEqual(530000%3,2)
 def test_changed_physical_scene_fails_checksum(self):
  (self.root/'seed-530000.json').write_text('{}')
  with self.assertRaisesRegex(ValueError,'mutated'):entry(self.root,530000,self.sha)
 def test_heldout_seed_cannot_be_seen_training_start(self):
  with self.assertRaisesRegex(ValueError,'Unplanned'):entry(self.root,900000,self.sha)
 def test_six_finite_joints_required(self):
  p=self.root/'state-530000.json';d=json.loads(p.read_text());d['state']=[0]*5;p.write_text(json.dumps(d));self.seal()
  with self.assertRaises(AssertionError):entry(self.root,530000,self.sha)
 def test_mid_is_one_second_after_first_departure_before_closing(self):
  fs=[dict(sim_ns=round(i/30*1e9),physics_state=[math.radians(3) if i>=5 else 0,0,0,0,0,.1]) for i in range(45)]
  self.assertEqual(select_mid(fs),(5,35))
  fs[35]['physics_state'][5]=0
  with self.assertRaisesRegex(ValueError,'after closure'):select_mid(fs)
  fs[35]['physics_state'][5]=.1;fs[10]['physics_state'][5]=0
  with self.assertRaisesRegex(ValueError,'after closure'):select_mid(fs)
 def test_incorrect_source_clock_is_refused(self):
  fs=[dict(sim_ns=i*40_000_000,physics_state=[.1 if i>=5 else 0,0,0,0,0,.1]) for i in range(45)]
  with self.assertRaisesRegex(ValueError,'cadence'):select_mid(fs)

if __name__=='__main__':unittest.main()
