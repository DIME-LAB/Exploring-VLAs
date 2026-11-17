import unittest,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rule431_eval import paths,window_plan,source_preflight_args,prepare,SOURCE

STOP='2030-01-01T00:00:00Z'
class RulePlanTests(unittest.TestCase):
    def test_prepare_sends_budget_to_reset_preflights_but_not_model_rpc(self):
        p=paths(5000,'fixture');source=SOURCE+'/005000/pretrained_model'
        files={'model.safetensors':dict(size=1,sha256='1'*64)}
        probe=dict(ready=True,step=5000,model=source,files=files,completion=dict(artifact_marker=dict(name='rule431-manifest.json')))
        handoff=dict(step=5000,published_model=p['model'],source=source,source_before=probe,source_after=probe,verified=dict(verified=True,files=files))
        calls=[]
        def run(argv,**kwargs):calls.append(argv);return SimpleNamespace(returncode=0)
        def remote(code):return json.dumps(dict(status='PASS',schema='fixture',weights_sha256='1'*64)) if 'require_handoff' in code else ''
        with tempfile.TemporaryDirectory() as directory,patch('rule431_eval.subprocess.check_output',return_value=json.dumps(handoff)),patch('rule431_eval.subprocess.run',side_effect=run),patch('rule431_eval.remote',side_effect=remote):
            root=Path(directory);proof=root/'proof.json';proof.write_text('{}')
            prepare(5000,proof,root/'out','fixture',180)
        rpc=[call[-1] for call in calls if call[0]=='ssh' and 'cpu_rpc_smoke.py' in call[-1]]
        self.assertEqual(len(rpc),1);self.assertNotIn('--readiness-budget-s',rpc[0])
        preflights=[call[-1] for call in calls if call[0]=='ssh' and 'native_policy.py' in call[-1]]
        self.assertEqual(len(preflights),4);self.assertTrue(all('--readiness-budget-s 180' in command for command in preflights))
    def test_explicit_readiness_budget_reaches_owner_and_source_preflight(self):
        plan=window_plan('qualify','seen5',STOP,attempt=4,qualification_attempt=4,readiness_budget_s=180)
        self.assertEqual(float(plan['argv'][plan['argv'].index('--readiness-budget-s')+1]),180)
        self.assertEqual(plan['readiness_budget_s'],180)
        args=source_preflight_args(5000,'rule15','/example/out',readiness_budget_s=180)
        self.assertEqual(float(args[args.index('--readiness-budget-s')+1]),180)
        self.assertEqual(window_plan('qualify','seen5',STOP)['readiness_budget_s'],60)
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
