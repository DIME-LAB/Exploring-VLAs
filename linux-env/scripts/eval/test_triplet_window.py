"""CPU-only barrier tests: all model/card/owner adapters are explicit fakes."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('triplet_window',Path(__file__).with_name('triplet_window.py'))
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def fixtures():
    context={'supervisor':{'pid':123,'start_ticks':'456'},'host':'fake-host'}
    plan={'mode':'qualify','step':None,'root':str(w.BASE/'eval/runs/t360-q-test'),
          'gpu':0,'domain':92,'ports':[8781,8782,18092],
          'argv':['python3','-B','-u',str(w.REPO/'linux-env/scripts/eval/run_native_replay_window.py'),
                  '--bank',str(w.ROOT/'held20-v2'),'--bank-sha256','0ac0d8928a960ce0aa22670f2d75c33bbc9a2b6717ba3f0fc076bec652abab0d',
                  '--reset-collector-root',str(w.REPO),'--out',str(w.BASE/'eval/runs/t360-q-test'),'--matched-reset-qualification']}
    hold={'state':'held','card':w.GPU,'by':w.OWNER,'slot':'default','hold_id':'fake-hold',
          'holder':{'pid':123,'process_start':'456','host':'fake-host'},'beside':{},
          'foreign_compute':{'cards':{w.GPU:{'state':'clear','foreign':[],'unread':[],'platform_unheld':[]}}}}
    docs={'plan':plan,'hold':hold,'free_card':dict(passed=True,gpu_uuid=w.GPU,checked_unix=1000,nvml_clear_seconds=60,
          docker_checked=True,host_workflows_checked=True,plananything_checked=True,other_side_obligation_clear=True),
          'notifier':dict(ready=True,plan_sha256='plan',checked_unix=1000),
          'probe':dict(passed=True,step=5000,layouts=20,queries=61,weights_sha256='a'*64,bank_sha256='0ac0d8928a960ce0aa22670f2d75c33bbc9a2b6717ba3f0fc076bec652abab0d'),
          'authority':dict(gpu_step=5000,cpu_all_steps=True,approved_by='so-arm-101-vla')}
    hashes={k:k for k in docs};hashes['controller']='controller'
    start=dict(approved_by='so-arm-101-vla',preauthorised=True,**{k+'_sha256':v for k,v in hashes.items()})
    return docs,hashes,start,context


class Barrier(unittest.TestCase):
    def test_qualified_scope(self):
        d,h,s,c=fixtures();w.validate_start(s,d,h,c,1001)

    def test_tampered_receipts(self):
        for key in ('plan','hold','free_card','notifier','probe','authority','controller'):
            with self.subTest(key=key):
                d,h,s,c=fixtures();s[key+'_sha256']='tampered'
                with self.assertRaises(ValueError):w.validate_start(s,d,h,c,1001)

    def test_foreign_or_stale_refused(self):
        changes=[('holder-birth',lambda d:d['hold']['holder'].update(process_start='457')),
                 ('wrong-slot',lambda d:d['hold'].update(slot='spike')),
                 ('wrong-card',lambda d:d['hold'].update(card='GPU1')),
                 ('foreign',lambda d:d['hold']['foreign_compute']['cards'][w.GPU]['foreign'].append({'pid':987})),
                 ('stale',lambda d:d['free_card'].update(checked_unix=0)),
                 ('no-docker',lambda d:d['free_card'].update(docker_checked=False)),
                 ('no-notifier',lambda d:d['notifier'].update(ready=False)),
                 ('unfinished-probe',lambda d:d['probe'].update(queries=60)),
                 ('later-checkpoint',lambda d:d['plan'].update(mode='score',step=10000))]
        for name,change in changes:
            with self.subTest(name=name):
                d,h,s,c=fixtures();change(d)
                with self.assertRaises(ValueError):w.validate_start(s,d,h,c,1001)

    def worker_fixture(self,folder):
        d,h,s,c=fixtures();names={'plan':'plan','hold':'card-hold','free_card':'free-card','notifier':'notifier-ready','probe':'probe-gate','authority':'authority'}
        report=folder/'fake-cpu-report.json';w.write(report,dict(status='complete',predictions=61,expected_observations=20,weights_sha256='a'*64))
        d['probe'].update(result_path=str(report),result_sha256=w.digest(report))
        audit=folder/'fake-cpu-audit.json';w.write(audit,dict(status='PASS',weights_sha256='a'*64,layouts=20,queries=60,actual_loaded_normalization_verified=True,raw_inputs_and_actual_noise_paired=True,cuda_initialized=False))
        d['probe'].update(validation_path=str(audit),validation_sha256=w.digest(audit))
        d['notifier']['plan_sha256']=None
        for key,name in names.items():w.write(folder/(name+'.json'),d[key])
        d['notifier']['plan_sha256']=w.digest(folder/'plan.json');w.write(folder/'notifier-ready.json',d['notifier'])
        for key,name in names.items():s[key+'_sha256']=w.digest(folder/(name+'.json'))
        s['controller_sha256']=w.digest(w.__file__);w.write(folder/'START.json',s)
        handoff=folder/'fake-handoff.json';w.write(handoff,{'source_before':{'model':'fixture-only'}})
        modules={'triplet_eval':types.SimpleNamespace(paths=lambda step:{'model':'fixture-only','handoff':handoff,'proof':'fixture-only'},REFERENCE='fixture-only'),
                 'checkpoint_contract':types.SimpleNamespace(require_handoff=lambda *a,**k:{'files':{}},require_processor_proof=lambda *a,**k:None)}
        manager=types.SimpleNamespace(capture_supervision=lambda:{'supervisor':c['supervisor']})
        return d,modules,manager

    def test_same_worker_entrance_fake_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);d,modules,manager=self.worker_fixture(folder);calls=[]
            def fake_run(argv,**kwargs):calls.append(argv);return types.SimpleNamespace(returncode=0)
            with patch.object(w,'owned_ops',lambda x:Path(x)),patch.object(w,'managed',lambda:manager),patch.object(w.socket,'gethostname',lambda:'fake-host'),patch.object(w.time,'time',lambda:1001),patch.dict(sys.modules,modules),patch.object(w,'card_holds',lambda:{'holds':[d['hold']]}),patch.object(w.subprocess,'run',fake_run):
                w.worker(folder)
            self.assertEqual(len(calls),2);self.assertIn('--check-only',calls[0]);self.assertNotIn('--check-only',calls[1])
            self.assertEqual(w.read(folder/'terminal.json')['owner_exit'],0)
            self.assertTrue(w.read(folder/'terminal.json')['own_hold_release_pending'])

    def test_primary_error_retained_no_owner_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);d,modules,manager=self.worker_fixture(folder)
            modules['checkpoint_contract'].require_processor_proof=lambda *a,**k:(_ for _ in ()).throw(ValueError('primary model proof refusal'))
            with patch.object(w,'owned_ops',lambda x:Path(x)),patch.object(w,'managed',lambda:manager),patch.object(w.socket,'gethostname',lambda:'fake-host'),patch.object(w.time,'time',lambda:1001),patch.dict(sys.modules,modules),patch.object(w.subprocess,'run') as launch:
                with self.assertRaisesRegex(ValueError,'primary model proof refusal'):w.worker(folder)
            launch.assert_not_called();self.assertFalse(w.read(folder/'failure.json')['gpu_start_attempted'])
            self.assertIn('primary model proof refusal',w.read(folder/'failure.json')['error'])

    def test_cleanup_failure_alerts_both_recipients(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);w.write(folder/'plan.json',{'fixture_only':True})
            observed={'present':True,'supervision':{'state':'cleanup_failed','error':'owned descendants remain'},'failure':{'error':'owner failed'}}
            with patch.object(w.subprocess,'check_output',return_value=json.dumps(observed)),patch.object(w.subprocess,'run') as send,patch.object(w.time,'sleep'):
                w.monitor(Path('fixture-only'),folder)
            self.assertEqual(send.call_count,2);self.assertEqual(w.read(folder/'terminal.json'),observed)
            self.assertIn('cleanup_failed',(folder/'terminal-message.txt').read_text())

    def test_outage_alert_and_resume_no_automatic_send_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);w.write(folder/'plan.json',{'fixture_only':True})
            observed={'present':True,'supervision':{'running':False},'terminal':{'owner_exit':0}}
            errors=[OSError('ssh outage')]*3+[json.dumps(observed)]
            sent=[]
            def send(argv,**kwargs):
                sent.append(argv)
                if len(sent)==1:raise OSError('first delivery failed')
                return types.SimpleNamespace(returncode=0)
            with patch.object(w.subprocess,'check_output',side_effect=errors),patch.object(w.subprocess,'run',send),patch.object(w.time,'sleep'):
                w.monitor(Path('fixture-only'),folder)
            self.assertEqual(len(sent),4);self.assertTrue((folder/'terminal.json').exists())
            d=w.read(folder/'visibility-lost-deliveries.json');self.assertEqual(d['lead']['status'],'failed');self.assertEqual(d['eval']['status'],'delivered')
            with patch.object(w.subprocess,'run') as retry:w.alert(folder,'visibility-lost','same alert')
            retry.assert_not_called()


if __name__=='__main__':unittest.main()
