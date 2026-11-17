import copy,unittest
from native_bank import reference_binding,select_policy_rows,reset_protocol,recorded_initial,NAMES
from matched_reset import require_scene_source

SEEDS=[530002,530003,530004,530005,530500,530503,530505,531001,531002,531005,531009,531010,531013,531015,531016]
class RuleSeenTests(unittest.TestCase):
    def test_recorded_initial_refuses_bad_body_shapes_and_nonfinite_values(self):
        source=dict(initial_physical=dict(state=[0]*6,drive_targets=[0]*6,bodies={n:dict(position=[0]*3,orientation=[1,0,0,0]) for n in NAMES}))
        self.assertIs(recorded_initial(source),source['initial_physical'])
        name=sorted(NAMES)[0]
        for field,values in (('position',[0,0]),('position',[0,0,float('nan')]),('orientation',[1,0,0]),('orientation',[0,0,0,0])):
            bad=copy.deepcopy(source);bad['initial_physical']['bodies'][name][field]=values
            with self.assertRaises(ValueError):recorded_initial(bad)
        bad=copy.deepcopy(source);bad['initial_physical']['state'][0]=float('inf')
        with self.assertRaises(ValueError):recorded_initial(bad)
    def test_recorded_rule_reset_is_explicit_without_changing_other_suites(self):
        self.assertEqual(reset_protocol('rule-seen-training-home-15'),'collector-recorded-postsample-v1')
        self.assertEqual(reset_protocol('seen-training-home'),'collector-matched-reset-v1')
        self.assertEqual(reset_protocol('seen-training-home-15'),'collector-matched-reset-v1')
        self.assertEqual(reset_protocol('heldout-bank-v2'),'collector-sealed-postsample-v1')
        with self.assertRaises(ValueError):reset_protocol('unknown')
    def test_rule_fifteen_binds_all_per_batch_references(self):
        rows=[dict(seed=s,reference_run_sha256=str(i%3)) for i,s in enumerate(SEEDS)]
        self.assertEqual(reference_binding(rows,'rule-seen-training-home-15')['reference_sha256_by_seed'],{str(r['seed']):r['reference_run_sha256'] for r in rows})
        self.assertEqual(select_policy_rows(rows,'rule-seen-training-home-15'),rows)
        with self.assertRaises(ValueError):reference_binding(rows[:-1],'rule-seen-training-home-15')
    def test_new_rule_source_is_admitted_without_weakening_legacy(self):
        row=dict(seed=531016,recorded_target='green_2x3_0',rule_target=dict(object='green_2x3_0'))
        source=dict(accepted=True,params=dict(seed=531016,recovery='none',object='green_2x3_0'))
        require_scene_source(row,'rule-seen-training-home-15',source)
        with self.assertRaises(ValueError):require_scene_source(row,'seen-training-home-15',source)
        bad=copy.deepcopy(source);bad['params']['object']='green_2x3_1'
        with self.assertRaises(ValueError):require_scene_source(row,'rule-seen-training-home-15',bad)
        bad=copy.deepcopy(source);bad['params']['seed']+=1
        with self.assertRaises(ValueError):require_scene_source(row,'rule-seen-training-home-15',bad)

if __name__=='__main__':unittest.main()
