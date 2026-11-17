"""Real budget and Worker retry entrances; only native startup/collector transports are fake."""
import copy,json,os,shutil,tempfile,time,unittest,threading
from pathlib import Path
from unittest.mock import patch,MagicMock
from types import SimpleNamespace as NS
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from storage import atomic_json
import startup_retries as s
import triplet_production as p
from triplet_production_plan import initial_wave

FIXTURE=Path(__file__).with_name('test_fixtures')/'controller-startup-unconfigured.txt'

class StartupRetryTests(unittest.TestCase):
    def fixture(self,root,slot=3):
        root=Path(root);spec=s.initialize(root/'budget','c'*64)
        job=copy.deepcopy(next(j for j in initial_wave() if j['instance']==slot));log=root/('up-'+str(slot)+'.log');log.write_bytes(FIXTURE.read_bytes())
        down=root/('down-'+str(slot)+'.json');atomic_json(down,dict(passed=True,errors=[],cleanup=dict(remaining=[]),ports_listening={str(8777+2*slot):False,str(8778+2*slot):False}))
        error=s.failed_exit(dict(name='up-0',pid=os.getpid(),started_wall_ns=time.time_ns(),returncode=1),log)
        return spec,job,error,down

    def test_actual_controller_timeout_is_typed_and_unknown_fault_is_not(self):
        with tempfile.TemporaryDirectory() as d:
            spec,job,error,down=self.fixture(d)
            self.assertEqual(error.kind,'controller_activation');self.assertIn("'unconfigured'",error.cause)
            log=Path(d)/'unknown.log';log.write_text(FIXTURE.read_text()+"\nModuleNotFoundError: missing transport\n")
            self.assertIsNone(s.failed_exit(dict(name='up-0',returncode=1),log))
            self.assertIsNone(s.failed_exit(dict(name='clock-0',returncode=1),error.log))
            self.assertIsNone(s.failed_exit(dict(name='up-0',returncode=0),error.log))

    def test_actual_up_entrance_distinguishes_startup_clock_identity_and_unknown_faults(self):
        for kind in ('controller','bootstrap','readiness','clock','identity','unknown'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d,ExitStack() as stack:
                root=Path(d);table=root/'table';table.write_text('fixture table')
                capsule=dict(pin=dict(passed=True,assigned_gpu_uuid=p.ASSIGNED_UUID,source=dict(path=str(table),sha256=p.sha(table))))
                recipe=dict(cut='a'*40,production_root=str(root/'production'),runtime_root=str(root/'runtime'),bootstrap_capsule_path=str(table),bootstrap_capsule_sha256=p.sha(table))
                def stage(worker,name,*args,**kwargs):
                    text=FIXTURE.read_text() if kind=='controller' else 'success'
                    if kind=='unknown':text+='\\nModuleNotFoundError: missing dependency\\n'
                    (worker.root/(name+'.log')).write_text(text)
                    event=dict(name=name,pid=os.getpid(),started_wall_ns=1,returncode=1 if name.startswith('up-') and kind in ('controller','unknown') else 0)
                    worker.stages.append(event);return event
                def rpc(worker,*args):
                    if kind=='bootstrap':raise TimeoutError('startup native transport timeout')
                    if kind=='identity':raise RuntimeError('Native RPC is not bound to an owned live responder')
                    return True
                def clock(*args):
                    if kind=='clock':raise RuntimeError('Clock contract violation')
                import stage_readiness
                sock=MagicMock();sock.__enter__.return_value=sock;sock.connect_ex.return_value=1
                for obj,name,value in ((p,'ROOT',root),(p,'confined',lambda x:Path(x)),(p,'capture_supervision',lambda:None),(p,'disk',lambda:None),(p,'inventory',lambda:[]),(p,'load_bootstrap_capsule',lambda *a,**k:capsule),(p,'OwnedProcesses',lambda *a:NS()),(p.Worker,'check',lambda *_:None),(p.Worker,'stage',stage),(p.Worker,'rpc',rpc),(p.clock_status,'require',clock)):
                    stack.enter_context(patch.object(obj,name,value))
                stack.enter_context(patch.object(p.socket,'socket',return_value=sock))
                stack.enter_context(patch.object(stage_readiness,'wait',side_effect=TimeoutError('Stage readiness deadline')))
                worker=p.Worker(recipe,3,0)
                try:
                    with self.assertRaises(RuntimeError) as result:worker.up()
                    if kind in ('controller','bootstrap','readiness'):
                        self.assertIsInstance(result.exception,s.StartupFailure)
                        self.assertTrue(result.exception.stage['name'].startswith('up-'))
                        self.assertEqual(result.exception.kind,{'controller':'controller_activation','bootstrap':'bootstrap_rpc','readiness':'scene_readiness'}[kind])
                    else:self.assertNotIsInstance(result.exception,s.StartupFailure)
                finally:
                    if worker.claim is not None:worker.claim.close();worker.claim=None

    def test_persistent_slot_limit_and_pending_retry_survive_continuations(self):
        with tempfile.TemporaryDirectory() as d:
            spec,job,error,down=self.fixture(d);s.failed(spec,3,job,error,down,None)
            for _ in range(3):
                self.assertIsNotNone(s.before_start(spec,3,job));s.failed(spec,3,job,error,down,None)
            resumed=s.initialize(Path(d)/'budget','c'*64)
            with self.assertRaisesRegex(RuntimeError,'cap reached'):s.before_start(resumed,3,job)
            self.assertEqual(s.summary(resumed)['counts'],dict(total=3,slots={'3':3,'4':0,'5':0,'6':0}))
            other=copy.deepcopy(job);other['layout_seed']+=4
            with self.assertRaisesRegex(RuntimeError,'different seed'):s.before_start(resumed,3,other)

    def test_run_limit_is_shared_across_slots_and_new_admissions(self):
        with tempfile.TemporaryDirectory() as d:
            for slot,retries in ((3,3),(4,3),(5,3),(6,1)):
                spec,job,error,down=self.fixture(d,slot)
                for _ in range(retries):
                    s.failed(spec,slot,job,error,down,None);token=s.before_start(spec,slot,job);s.ready(spec,slot,job,token)
            s.failed(spec,6,job,error,down,None)
            resumed=s.initialize(Path(d)/'budget','c'*64)
            with self.assertRaisesRegex(RuntimeError,'cap reached'):s.before_start(resumed,6,job)
            self.assertEqual(s.summary(resumed)['counts']['total'],10)

    def test_stale_ready_cannot_clear_a_newer_failure_or_free_a_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            spec,job,error,down=self.fixture(d);s.failed(spec,3,job,error,down,None)
            old=s.before_start(spec,3,job);s.failed(spec,3,job,error,down,None)
            with self.assertRaisesRegex(ValueError,'stale'):s.ready(spec,3,job,old)
            new=s.before_start(spec,3,job);s.ready(spec,3,job,new)
            self.assertFalse(s.summary(spec)['pending']);self.assertEqual(s.summary(spec)['counts']['total'],2)

    def test_partial_or_rewound_ledger_and_missing_retirement_fail_closed(self):
        for kind in ('state','orphan','prefix','journal-reset','journal-partial','root-swap','ports','port-types','cleanup'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                spec,job,error,down=self.fixture(d)
                if kind in ('ports','port-types','cleanup'):
                    row=json.loads(down.read_text())
                    if kind=='ports':row['ports_listening']={}
                    elif kind=='port-types':row['ports_listening']={str(8777+2*job['instance']):0,str(8778+2*job['instance']):0}
                    else:row['cleanup']['remaining']=[{'unresolved':True}]
                    atomic_json(down,row)
                    with self.assertRaises(ValueError):s.failed(spec,3,job,error,down,None)
                    continue
                s.failed(spec,3,job,error,down,None)
                if kind!='journal-reset':spec=s.initialize(Path(d)/'budget','c'*64)
                if kind=='state':(Path(d)/'budget/state.json').unlink()
                elif kind=='orphan':(Path(d)/'budget/event-9999').mkdir()
                elif kind=='journal-partial':
                    with (Path(d)/'budget/journal.jsonl').open('ab') as stream:stream.write(b'{')
                elif kind=='root-swap':
                    (Path(d)/'budget').rename(Path(d)/'previous');shutil.copytree(Path(d)/'previous',Path(d)/'budget')
                else:
                    shutil.rmtree(Path(d)/'budget/event-0001');atomic_json(Path(d)/'budget/state.json',dict(events=[]))
                    if kind=='prefix':(Path(d)/'budget/journal.jsonl').write_bytes(b'')
                with self.assertRaises((ValueError,FileNotFoundError)):s.summary(spec)

    def test_failed_durable_index_write_refuses_another_start(self):
        with tempfile.TemporaryDirectory() as d:
            spec,job,error,down=self.fixture(d);write=s.atomic_json
            def interrupted(path,value):
                if Path(path).name=='state.json':raise OSError('index write failed')
                return write(path,value)
            with patch.object(s,'atomic_json',interrupted),self.assertRaises(OSError):s.failed(spec,3,job,error,down,None)
            with self.assertRaisesRegex(ValueError,'journal differ'):s.before_start(spec,3,job)

    def test_concurrent_slots_serialize_budget_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            fixtures=[self.fixture(d,slot) for slot in range(3,7)]
            def run(args):
                spec,job,error,down=args;slot=job['instance'];s.failed(spec,slot,job,error,down,None)
                token=s.before_start(spec,slot,job);s.ready(spec,slot,job,token)
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,fixtures))
            summary=s.summary(fixtures[0][0]);self.assertEqual(summary['counts']['total'],4);self.assertFalse(summary['pending'])

    def worker_run(self,d,failures=1,bad=None,slot=3):
        root=Path(d);spec,job,error,down=self.fixture(root,slot);recipe=dict(cut='a'*40,production_root=str(root/'production'),runtime_root=str(root/'runtime'),startup_retry_budget=spec)
        events=[];calls=[];source=error.log
        def up(worker):
            worker.session+=1;worker.claim=object();worker.retirement_certified=False;events.append('up')
            log=worker.root/f'up-{worker.session}.log';log.write_bytes(source.read_bytes())
            worker.stages.append(dict(name=f'up-{worker.session}',pid=os.getpid(),started_wall_ns=time.time_ns(),returncode=1 if worker.session<failures else 0))
            if bad=='unknown':raise RuntimeError('Unknown native ownership failure')
            if worker.session<failures:
                if bad=='writer':
                    run=root/f'datasets/counterfactual-targets/train/layout-{job["layout_seed"]}'
                    run.mkdir(parents=True);(run/'frames.jsonl').write_text('{}')
                raise s.failed_exit(worker.stages[-1],log)
        def retire(worker):
            if worker.claim is None:return
            events.append('down');worker.claim=None
            if bad=='cleanup':raise RuntimeError('Owned cleanup failed')
            worker.retirement_certified=True
            atomic_json(worker.root/f'owned-down-{worker.session}.json',json.loads(down.read_text()))
        def attempt(worker,current):
            calls.append(copy.deepcopy(current));events.append('collect')
            if bad=='after_writer':raise s.StartupFailure('controller_activation','late',dict(name='up-0',returncode=1),source)
        with patch.object(p,'ROOT',root),patch.object(p,'confined',lambda x:Path(x)),patch.object(p,'capture_supervision',lambda:None),patch.object(p.Worker,'check',lambda *_:None),patch.object(p.Worker,'up',up),patch.object(p.Worker,'down',retire),patch.object(p.Worker,'attempt',attempt),patch.object(p,'backup_terminal',return_value={'passed':True}):
            worker=p.Worker(recipe,slot,0)
            if bad or failures>3:
                with self.assertRaises(RuntimeError):worker.run([job])
                self.assertTrue(worker.stop_file.exists())
            else:worker.run([job])
        return dict(events=events,calls=calls,summary=s.summary(spec),root=worker.root)

    def test_real_worker_run_retires_before_retry_and_preserves_seed(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.worker_run(d)
            self.assertEqual(result['events'],['up','down','up','collect','down'])
            self.assertEqual([j['layout_seed'] for j in result['calls']],[720000])
            self.assertEqual(result['summary']['counts']['total'],1)
            evidence=json.loads((result['root']/'startup-failure-0.json').read_text())
            self.assertIn('unconfigured',evidence['event']['error'])

    def test_real_worker_run_stops_on_writer_unknown_cleanup_and_exhaustion(self):
        for bad in ('writer','unknown','cleanup','after_writer'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as d:
                result=self.worker_run(d,failures=0 if bad=='after_writer' else 1,bad=bad)
                self.assertEqual(result['summary']['counts']['total'],0)
                self.assertEqual(result['events'].count('up'),1)
        with tempfile.TemporaryDirectory() as d:
            result=self.worker_run(d,failures=4)
            self.assertEqual(result['events'].count('up'),4)
            self.assertEqual(result['summary']['counts']['total'],3)
            self.assertFalse(result['calls'])

    def test_pending_later_job_resumes_first_at_the_real_worker_entrance(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);spec,first,error,down=self.fixture(d)
            later=copy.deepcopy(first);later['layout_seed']+=4;later['requested_layout_seed']+=4;later['replacement_chain']=[later['layout_seed']]
            s.failed(spec,3,later,error,down,None);calls=[]
            recipe=dict(cut='a'*40,production_root=str(root/'production'),runtime_root=str(root/'runtime'),startup_retry_budget=spec)
            def up(worker):worker.session+=1;worker.claim=object();worker.retirement_certified=False
            def retire(worker):worker.claim=None;worker.retirement_certified=True
            with patch.object(p,'ROOT',root),patch.object(p,'confined',lambda x:Path(x)),patch.object(p,'capture_supervision',lambda:None),patch.object(p.Worker,'check',lambda *_:None),patch.object(p.Worker,'up',up),patch.object(p.Worker,'down',retire),patch.object(p.Worker,'attempt',lambda _,job:calls.append(job['layout_seed'])):
                p.Worker(recipe,3,0).run([first,later])
            self.assertEqual(calls,[later['layout_seed'],first['layout_seed']]);self.assertFalse(s.summary(spec)['pending'])
            self.assertEqual(s.summary(spec)['counts']['total'],1)

    def test_missing_moved_or_recreated_ledger_cannot_reset_the_budget(self):
        for kind in ('moved','deleted','recreated','both_deleted'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                spec,job,error,down=self.fixture(d);s.failed(spec,3,job,error,down,None);s.before_start(spec,3,job)
                previous=s.initialize(Path(d)/'budget','c'*64);root=Path(spec['root'])
                if kind=='moved':root.rename(Path(d)/'old-budget')
                else:shutil.rmtree(root)
                if kind=='both_deleted':Path(previous['identity_receipt_path']).unlink()
                if kind=='recreated':
                    # Even a fresh, structurally valid empty ledger has a different inode.
                    root.mkdir();(root/'lock').touch();(root/'journal.jsonl').touch()
                    atomic_json(root/'state.json',dict(events=[]));atomic_json(root/'policy.json',dict(s.POLICY,origin_checkpoint_sha256='c'*64))
                with self.assertRaises((ValueError,FileNotFoundError)):s.initialize(root,'c'*64,[previous])
                if kind!='both_deleted':
                    with self.assertRaises((ValueError,FileNotFoundError)):s.initialize(root,'c'*64)

    def test_terminal_backup_carries_the_ledger_and_external_identity_pin(self):
        for failed in (False,True):
            with self.subTest(backup_failure=failed),tempfile.TemporaryDirectory() as d:
                root=Path(d);spec,job,error,down=self.fixture(d);s.failed(spec,3,job,error,down,None);s.before_start(spec,3,job)
                production=root/'production';production.mkdir();recipe=dict(production_root=str(production),startup_retry_budget=spec);calls=[]
                (root/'runtime').mkdir()
                def backup(source,files,destination,log):
                    calls.append((Path(source),files,Path(destination)))
                    if failed:raise OSError('destination full')
                    return dict(passed=True,files_sha256=files,destination=str(destination))
                with patch.object(p,'ROOT',root),patch.object(p,'DUAL',root/'dual'),patch.object(p,'backup',backup):
                    report=p.backup_remaining_terminal(recipe,[],'original startup stop')
                if failed:
                    self.assertIn('destination full',report['errors'][0]['error']);self.assertIsNone(report['startup_retry_budget'])
                else:
                    self.assertEqual(len(calls),2);self.assertEqual(calls[0][0],Path(spec['root']))
                    self.assertEqual(calls[1][1],{Path(spec['identity_receipt_path']).name:spec['identity_receipt_sha256']})
                    self.assertEqual(report['startup_retry_budget']['snapshot']['counts']['total'],1)
                    self.assertEqual(report['startup_retry_budget']['snapshot']['root_identity'],spec['root_identity'])

if __name__=='__main__':unittest.main()
