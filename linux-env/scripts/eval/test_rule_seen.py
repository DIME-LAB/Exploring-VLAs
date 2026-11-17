import copy,unittest
from native_bank import reference_binding,select_policy_rows
from matched_reset import require_scene_source

SEEDS=[530002,530003,530004,530005,530500,530503,530505,531001,531002,531005,531009,531010,531013,531015,531016]
class RuleSeenTests(unittest.TestCase):
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
