"""The retained 720144 zero-member failure through the actual Worker entrance."""
import copy,json,os,shutil,tempfile,time,unittest
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import scene_initialization as init
import startup_retries as retries
import triplet_production as p
from storage import atomic_json
from triplet_artifacts import manifest,read,sha

FIXTURE=Path(__file__).with_name('test_fixtures')/'scene-init-720144.json'


def retained(root):
    value=read(FIXTURE);job=value['job'];worker=object.__new__(p.Worker)
    (root/'runtime').mkdir(exist_ok=True)
    worker.root=root/'worker';worker.root.mkdir();worker.cut=value['collector_cut'];worker.collector_repo=Path(value['collector_repo']);worker.session=0
    worker.wrapper=root/'collect_sim.sh';worker.instance=job['instance'];worker.wave=2
    worker.stages=json.loads(value['files']['worker/stages.json'])[:3];worker.native_observation_start=0
    observations=[json.loads(x) for x in value['files']['worker/native-observations.jsonl'].splitlines()]
    worker.native={k:observations[-1][k] for k in ('pid','start_ticks','pgid')}
    for name,text in value['files'].items():
        path=root/f'datasets/counterfactual-targets/train/layout-{job["layout_seed"]}'/name[4:] if name.startswith('raw/') else worker.root/name[7:]
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    return worker,job,worker.stages[-1]


class SceneInitializationTests(unittest.TestCase):
    def test_non_object_json_is_typed_at_real_attempt_and_keeps_native_cause(self):
        for name,value in [('wall-profile-finalization.json',[]),('wall-profile-finalization.json',dict(ticks=[1],capture_failures=[])),
                ('summary.json',[]),('collector-finalization.json',None),('readiness-0.json',[])]:
            with self.subTest(name=name,value=value),tempfile.TemporaryDirectory() as d,ExitStack() as stack:
                root=Path(d);worker,job,event=retained(root);run=root/f'datasets/counterfactual-targets/train/layout-{job["layout_seed"]}'
                target=worker.root/name if name.startswith('readiness-') else run/name
                atomic_json(target,value);source=root/'retained';run.rename(source)
                def collect(*args,**kwargs):shutil.copytree(source,run);return event
                for obj,key,answer in [(p,'ROOT',root),(p,'disk',lambda:None),(p,'data_checkout',lambda *a:None),(p.Worker,'stage',collect),
                        (p,'verify_sources',lambda *a:self.fail('malformed startup reached provenance'))]:stack.enter_context(patch.object(obj,key,answer))
                with self.assertRaises(init.IncompleteInitialization) as refusal:worker.attempt(job)
                self.assertIn(init.CAUSE,str(refusal.exception));self.assertIn(str(worker.root/'collect-720144.log'),str(refusal.exception))

    def test_retained_720144_is_typed_before_missing_provenance_at_real_attempt(self):
        with tempfile.TemporaryDirectory() as d,ExitStack() as stack:
            root=Path(d);worker,job,event=retained(root);run=root/f'datasets/counterfactual-targets/train/layout-{job["layout_seed"]}'
            source=root/'retained';run.rename(source)
            def collect(*args,**kwargs):shutil.copytree(source,run);return event
            for obj,name,value in [(p,'ROOT',root),(p,'disk',lambda:None),(p,'data_checkout',lambda *a:None),(p.Worker,'stage',collect),(p,'verify_sources',lambda *a: self.fail('provenance called before startup classification'))]:stack.enter_context(patch.object(obj,name,value))
            with self.assertRaises(retries.StartupFailure) as result:worker.attempt(job)
            self.assertEqual(result.exception.kind,'scene_initialization');self.assertEqual(init.verify(result.exception.evidence),init.CAUSE)
            self.assertEqual(result.exception.evidence['stage']['pid'],827279)

    def test_unknown_postwriter_and_changed_lifecycle_never_become_startup_retries(self):
        for kind in ('member','empty_member','run','terminal','trace','rc','clock','readiness','pid','nonce','prior_collect','up','summary','source_cut','capture_fault'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d,patch.object(p,'ROOT',Path(d)):
                worker,job,event=retained(Path(d));run=worker.raw_path(job)
                if kind in ('member','empty_member'):
                    (run/'attempt-0000').mkdir()
                    if kind=='member':(run/'attempt-0000/frames.jsonl').write_text('{}\n')
                elif kind=='run':atomic_json(run/'run.json',{})
                elif kind=='terminal':atomic_json(run/'terminal-fault.json',{})
                elif kind=='trace':
                    with (worker.root/'collect-720144.log').open('a') as f:f.write('RuntimeError: unknown\n')
                elif kind=='rc':event['returncode']=None
                elif kind=='clock':atomic_json(worker.root/'clock-0.json',dict(passed=False))
                elif kind=='readiness':
                    row=read(worker.root/'readiness-0.json');row['passed']=False;atomic_json(worker.root/'readiness-0.json',row)
                elif kind=='pid':event['pid']+=1
                elif kind=='nonce':
                    f=worker.root/'native-observations.jsonl';rows=[json.loads(x) for x in f.read_text().splitlines()];rows[-1]['nonce']=rows[0]['nonce'];f.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
                elif kind=='prior_collect':worker.stages.insert(2,dict(event,name='collect-720000'))
                elif kind=='up':worker.stages[0]['returncode']=1
                elif kind=='source_cut':worker.cut='0'*40
                elif kind=='capture_fault':
                    row=read(run/'wall-profile-finalization.json');row['capture_failures']=['unknown native fault'];atomic_json(run/'wall-profile-finalization.json',row)
                else:
                    q=read(run/'summary.json');q['complete']=True;atomic_json(run/'summary.json',q)
                with self.assertRaises(init.IncompleteInitialization):init.failure(worker,job,event)

    def test_retirement_backup_and_same_seed_retry_at_real_worker_run(self):
        for fault in (None,'retirement','backup','cap'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as d,ExitStack() as stack:
                root=Path(d);stack.enter_context(patch.object(p,'ROOT',root));worker,job,event=retained(root);source=root/'retained';worker.raw_path(job).rename(source)
                spec=retries.initialize(root/'budget','c'*64);worker.plan=dict(cut=worker.cut,startup_retry_budget=spec);worker.instance=5;worker.global_root=root/'production';worker.stop_file=worker.global_root/'STOP.json';worker.supervision=None;worker.reason=None;worker.streak=0;worker.active_job=None
                worker.global_root.mkdir()
                if fault=='cap':
                    from test_startup_retries import StartupRetryTests
                    prior=StartupRetryTests().fixture(root,slot=5)
                    for _ in range(3):
                        retries.failed(spec,5,prior[1],prior[2],prior[3],None);token=retries.before_start(spec,5,prior[1]);retries.ready(spec,5,prior[1],token)
                worker.session=-1;worker.claim=None;worker.retirement_certified=True;events=[];calls=[]
                def up(w):
                    w.session+=1;w.claim=object();w.retirement_certified=False;events.append('up')
                    if w.session==0:w.stages=read(source.parent/'worker/stages.json')[:3]
                def down(w):
                    if w.claim is None:return
                    events.append('down');w.claim=None
                    if fault=='retirement':raise RuntimeError('Uncertified cleanup')
                    w.retirement_certified=True;atomic_json(w.root/f'owned-down-{w.session}.json',dict(passed=True,errors=[],cleanup=dict(remaining=[]),ports_listening={'8787':False,'8788':False}))
                def stage(w,name,*args,**kwargs):
                    calls.append(copy.deepcopy(job));events.append('collect')
                    if w.session==0:shutil.copytree(source,w.raw_path(job));return event
                    w.raw_path(job).mkdir();atomic_json(w.raw_path(job)/'run.json',{})
                    return dict(event,name=name,returncode=0)
                def backup(src,files,dest,log):
                    events.append('backup')
                    if fault=='backup':raise OSError('Backup failed')
                    shutil.copytree(src,dest);return dict(passed=True,files_sha256=files,destination=str(dest))
                for obj,name,value in [(p,'ROOT',root),(p,'DUAL',root/'dual'),(p,'disk',lambda:None),(p,'data_checkout',lambda *a:None),(p.Worker,'check',lambda *_:None),(p.Worker,'up',up),(p.Worker,'down',down),(p.Worker,'stage',stage),(p,'backup',backup),(p,'backup_terminal',lambda *a:dict(passed=True)),(p.Worker,'qualify',lambda *a:dict(status='selected',reason='accepted')),(p.Worker,'await_admission',lambda *a:None),(p.Worker,'rpc',lambda *a:dict(arm_restored=True,command_baseline=True))]:stack.enter_context(patch.object(obj,name,value))
                # A successful second collector's receipt is only used by the
                # between-group restore; no episode is selected by this test.
                stage_impl=stage
                def transport(w,name,*args,**kwargs):
                    result=stage_impl(w,name,*args,**kwargs)
                    if w.session>0:
                        (w.raw_path(job)/'attempt-0000').mkdir();atomic_json(w.raw_path(job)/'attempt-0000/receipt.json',dict(initial_physical={}))
                    return result
                stack.enter_context(patch.object(p.Worker,'stage',transport))
                if fault:
                    with self.assertRaises(RuntimeError):worker.run([job])
                    self.assertEqual(events.count('up'),1);self.assertEqual(retries.summary(spec)['counts']['total'],3 if fault=='cap' else 0)
                else:
                    worker.run([job]);self.assertEqual(events,['up','collect','down','backup','up','collect','down'])
                    self.assertEqual([j['layout_seed'] for j in calls],[720144,720144]);self.assertEqual(retries.summary(spec)['counts']['total'],1)
                    self.assertFalse(retries.summary(spec)['pending']);self.assertTrue((worker.root/'scene-init-720144-session-0/summary.json').exists())
                    self.assertFalse(worker.stop_file.exists())

    def test_missing_provenance_and_source_files_name_the_missing_input(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaisesRegex(ValueError,'run.json'):p.verify_sources(root,'a'*40,root)
            atomic_json(root/'run.json',dict(git_commit='a'*40,source_sha256={'gone.py':'f'*64}))
            with self.assertRaisesRegex(ValueError,'gone.py'):p.verify_sources(root,'a'*40,root)


if __name__=='__main__':unittest.main()
