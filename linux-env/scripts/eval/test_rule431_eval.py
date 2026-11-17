import unittest
from rule431_eval import paths,window_plan,source_preflight_args

STOP='2030-01-01T00:00:00Z'
class RulePlanTests(unittest.TestCase):
    def test_all_scoring_chunks_require_new_contract_and_paired_default(self):
        for suite in ('rule15','held0','held1'):
            p=window_plan('score',suite,STOP,10000,weights_sha256='1'*64);a=p['argv']
            self.assertIn('--rule-target',a);self.assertIn('--retain-wrist-before-closure',a)
            self.assertEqual(a[a.index('--weights-sha256')+1],'1'*64)
            self.assertEqual(a[a.index('--paired-flow-repeat')+1],'0')
            self.assertNotIn('--rtc-off',a);self.assertIn('rule431/contracts/input-contract.json',a[a.index('--processor-reference')+1])
            if suite.startswith('held'):
                self.assertEqual(a[a.index('--bank-offset')+1],'0' if suite=='held0' else '15')
                self.assertEqual(a[a.index('--bank-count')+1],'15')
    def test_qualification_and_unscored_smoke_have_separate_roots(self):
        p=window_plan('qualify','held0',STOP);a=p['argv'];self.assertNotIn('--checkpoint',a);self.assertNotIn('--bank-count',a)
        self.assertIn('--bootstrap-control',a)
        smoke=window_plan('smoke','rule15',STOP,5000,weights_sha256='2'*64)
        self.assertIn('--unscored-policy-smoke',smoke['argv']);self.assertIn('-smoke-',smoke['root'])
        self.assertEqual(smoke['argv'][smoke['argv'].index('--episode-seeds')+1],'530002')
        pre=source_preflight_args(5000,'rule15','/example/out',smoke=True)
        self.assertIn('--check-only',pre);self.assertIn('--rule-target',pre);self.assertIn('--retain-wrist-before-closure',pre)
    def test_explicit_qualification_version_binds_bootstrap_and_score(self):
        with self.assertRaises(ValueError):window_plan('qualify','held0',STOP,attempt=2)
        p=window_plan('qualify','held0',STOP,attempt=2,qualification_attempt=2);self.assertIn('/runs/r431-q-seen5-v2',p['argv'][p['argv'].index('--bootstrap-control')+1])
        p=window_plan('score','rule15',STOP,5000,weights_sha256='3'*64,qualification_attempt=2);self.assertTrue(p['argv'][p['argv'].index('--reset-qualification')+1].endswith('r431-q-rule15-v2'))
    def test_wrong_job_step_missing_weights_and_cohort_refuse(self):
        with self.assertRaises(ValueError):paths(60000)
        with self.assertRaises(ValueError):window_plan('score','rule15',STOP,10000)
        with self.assertRaises(ValueError):window_plan('score','mixed15',STOP,10000,weights_sha256='1'*64)
        with self.assertRaises(ValueError):window_plan('smoke','held0',STOP,5000,weights_sha256='1'*64)

if __name__=='__main__':unittest.main()
