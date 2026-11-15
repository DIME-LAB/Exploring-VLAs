"""Linux CPU subprocess regressions with mocked NVML, no GPU contexts."""
import ast,copy,json,os,select,signal,subprocess,sys,tempfile,time,types,unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'collect'))
from native_gpu_audit import RegisteredGPUAudit,attribute,birth
from run_smoke import identity
from managed_process import start,stop

CARDS=[dict(uuid='gpu0',index=0),dict(uuid='gpu1',index=1)]
def observation(rows):
    return dict(rows=rows,coverage={c['uuid']:dict(compute='verified',graphics='verified') for c in CARDS},errors=[])
def context(pid,gpu='gpu0',memory=3700*1024**2,api='compute'):
    return dict(pid=pid,gpu_uuid=gpu,memory_bytes=memory,api=api)

class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name);self.paths=[];self.workers=[]
        self.audit=RegisteredGPUAudit(self.root,self.root/'registry.json')
        clean=dict(os.environ);clean.pop('SOARM_INSTANCE_ROOT',None)
        with mock.patch.dict(os.environ,clean,clear=True):
            for name in ('isaacsim-soarm101.pid','control.pid'):
                path=self.root/name;start(path,self.root/(name+'.log'),[sys.executable,'-c','import time; time.sleep(120)']);self.paths.append(path)
    def tearDown(self):
        for proc in self.workers:
            if proc.poll() is None:proc.terminate()
            proc.wait(timeout=5)
        self.audit.close()
        for path in reversed(self.paths):stop(path)
        self.temporary.cleanup()
    def worker(self):
        proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'],start_new_session=True)
        self.workers.append(proc);return proc
    def test_worker_born_during_sample_refuses_then_admits_only_after_registration(self):
        late=[]
        def sampler(cards):
            late.append(self.worker());return observation([context(os.getpid()),context(late[0].pid,memory=1024)])
        with self.assertRaisesRegex(RuntimeError,'Unregistered GPU context'):
            self.audit.sample('gpu0',CARDS,sampler)
        record=json.loads((self.root/'registry.json').read_text())
        self.assertNotIn(late[0].pid,[r['pid'] for r in record['processes']])
        result,meta=self.audit.sample('gpu0',CARDS,lambda _:observation([context(os.getpid()),context(late[0].pid,memory=1024)]))
        self.assertAlmostEqual(result['per_gpu_memory_mib']['gpu0'],3700+1/1024)
        self.assertIn(late[0].pid,[r['pid'] for r in json.loads((self.root/'registry.json').read_text())['processes']])
        self.assertGreater(meta['registry_recorded_ns'],record['recorded_before_nvml_ns'])
    def regroup_worker(self):
        script='import os,sys;print("READY",flush=True);sys.stdin.readline();os.setpgid(0,0);print("REGROUPED",flush=True);sys.stdin.readline()'
        proc=subprocess.Popen([sys.executable,'-c',script],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        self.workers.append(proc);self.assertEqual(proc.stdout.readline().strip(),'READY');return proc
    def regroup(self,proc):
        proc.stdin.write('GROUP'+chr(10));proc.stdin.flush();self.assertEqual(proc.stdout.readline().strip(),'REGROUPED')
    def test_cpu_pgid_change_reregisters_before_sample_but_gpu_change_during_sample_refuses(self):
        proc=self.regroup_worker();self.audit.registry();self.regroup(proc)
        result,_=self.audit.sample('gpu0',CARDS,lambda _:observation([context(os.getpid()),context(proc.pid,memory=1024)]))
        self.assertGreater(result['per_gpu_memory_mib']['gpu0'],3700)
        late=self.regroup_worker();self.audit.registry()
        def sampler(cards):
            self.regroup(late);return observation([context(os.getpid()),context(late.pid,memory=1024)])
        with self.assertRaisesRegex(RuntimeError,'changed PGID'):self.audit.sample('gpu0',CARDS,sampler)
        self.audit.sample('gpu0',CARDS,lambda _:observation([context(os.getpid()),context(late.pid,memory=1024)]))
    def test_cpu_only_worker_churn_does_not_invalidate_stable_gpu_holder(self):
        def sampler(cards):
            proc=self.worker();proc.terminate();proc.wait(timeout=5)
            return observation([context(os.getpid())])
        for _ in range(8):
            result,_=self.audit.sample('gpu0',CARDS,sampler);self.assertEqual(result['per_gpu_memory_mib']['gpu0'],3700)
    def test_unknown_assigned_context_and_ours_on_other_card_are_refused(self):
        with self.assertRaisesRegex(RuntimeError,'Unregistered'):
            self.audit.sample('gpu0',CARDS,lambda _:observation([context(os.getppid())]))
        proc=self.worker()
        with self.assertRaisesRegex(RuntimeError,'owned device attribution'):
            self.audit.sample('gpu0',CARDS,lambda _:observation([context(os.getpid()),context(proc.pid,'gpu1')]))
    def test_unregistered_owned_context_on_other_card_is_not_hidden(self):
        def sampler(cards):
            proc=self.worker();return observation([context(os.getpid()),context(proc.pid,'gpu1')])
        with self.assertRaisesRegex(RuntimeError,'Unregistered'):
            self.audit.sample('gpu0',CARDS,sampler)
    def test_other_users_other_card_context_is_ignored(self):
        result,meta=self.audit.sample('gpu0',CARDS,lambda _:observation([context(os.getpid()),context(os.getppid(),'gpu1')]))
        self.assertEqual(meta['ignored_other_card_rows'],1);self.assertEqual(result['per_gpu_memory_mib']['gpu0'],3700)
    def test_reused_or_exited_gpu_holder_and_missing_memory_are_refused(self):
        registered={os.getpid():identity(os.getpid())};changed=copy.deepcopy(registered);changed[os.getpid()]['start_ticks']='wrong'
        with self.assertRaisesRegex(RuntimeError,'exited, reused PID or changed PGID'):
            attribute('gpu0',changed,CARDS,observation([context(os.getpid())]),identity)
        with self.assertRaisesRegex(RuntimeError,'exited, reused PID or changed PGID'):
            attribute('gpu0',registered,CARDS,observation([context(os.getpid())]),lambda pid:None)
        with self.assertRaisesRegex(RuntimeError,'coverage, memory'):
            self.audit.sample('gpu0',CARDS,lambda _:observation([context(os.getpid(),memory=None)]))
    def test_prelaunch_external_graphics_is_not_owned_and_new_contexts_still_refuse(self):
        external=context(os.getppid(),memory=4096,api='graphics')
        baseline=self.audit.freeze_external('gpu0',CARDS,lambda _:observation([external]))
        self.assertFalse(baseline['owned'])
        result,metadata=self.audit.sample('gpu0',CARDS,lambda _:observation([external,context(os.getpid())]))
        self.assertEqual(result['per_gpu_memory_mib']['gpu0'],3700)
        self.assertEqual(len(metadata['external_graphics_rows']),1)
        self.assertNotIn(os.getppid(),[r['pid'] for r in result['processes']])
        with self.assertRaisesRegex(RuntimeError,'Unregistered'):
            self.audit.sample('gpu0',CARDS,lambda _:observation([dict(external,api='compute'),context(os.getpid())]))
        with self.assertRaisesRegex(RuntimeError,'already frozen'):
            self.audit.freeze_external('gpu0',CARDS,lambda _:observation([]))
    def test_external_baseline_never_allows_pid_reuse_or_new_owned_gpu1(self):
        external=context(os.getppid(),memory=4096,api='graphics')
        self.audit.freeze_external('gpu0',CARDS,lambda _:observation([external]))
        certificate=self.audit.external_graphics[(os.getppid(),'gpu0','graphics')]
        certificate['start_ticks']='wrong'
        with self.assertRaisesRegex(RuntimeError,'External graphics identity changed'):
            self.audit.sample('gpu0',CARDS,lambda _:observation([external,context(os.getpid())]))
        with self.assertRaisesRegex(RuntimeError,'owned device attribution'):
            self.audit.sample('gpu0',CARDS,lambda _:observation([context(os.getpid(),'gpu1')]))
    def test_external_baseline_refuses_compute_unreadable_memory_and_incomplete_coverage(self):
        for row in [context(os.getppid()),context(os.getppid(),memory=None,api='graphics'),context(os.getpid(),api='graphics')]:
            with self.assertRaisesRegex(RuntimeError,'before launch|unavailable'):
                self.audit.freeze_external('gpu0',CARDS,lambda _,row=row:observation([row]))
        with self.assertRaisesRegex(RuntimeError,'coverage unverified'):
            self.audit.freeze_external('gpu0',CARDS,lambda _:dict(observation([]),errors=['unavailable']))

    def test_pruning_closes_only_exited_pidfds(self):
        proc=self.worker();self.audit.registry();oldfd=self.audit.children.handles[proc.pid]
        proc.terminate();proc.wait(timeout=5);self.audit.prune()
        self.assertNotIn(proc.pid,self.audit.children.known)
        with self.assertRaises(OSError):os.fstat(oldfd)
    def test_scan_einval_is_bounded_and_never_turns_a_failed_scan_into_verified(self):
        original=self.audit.children.scan
        with mock.patch.object(self.audit.children,'scan',side_effect=[OSError(22,'exited task'),None]):
            self.audit.registry()
        with mock.patch.object(self.audit.children,'scan',side_effect=OSError(22,'persistent')) as scan:
            with self.assertRaises(OSError):self.audit.registry()
            self.assertEqual(scan.call_count,3)
    def test_tampered_managed_supervisor_certificate_is_refused(self):
        path=Path(str(self.paths[0])+'.identity.json');record=json.loads(path.read_text());original=copy.deepcopy(record);record['supervisor']['start_ticks']='wrong';path.write_text(json.dumps(record))
        try:
            with self.assertRaises(RuntimeError):self.audit.registry()
        finally:
            path.write_text(json.dumps(original))

class IsolatedEntrypointTests(unittest.TestCase):
    def test_constructor_resolves_collector_without_test_path_injection(self):
        with tempfile.TemporaryDirectory() as d:
            folder=str(Path(__file__).resolve().parent)
            code='import sys;from pathlib import Path;sys.path.insert(0,'+repr(folder)+');from native_gpu_audit import RegisteredGPUAudit;obj=RegisteredGPUAudit(Path('+repr(d)+'),Path('+repr(d)+')/"registry.json");obj.close();print("PASS isolated owner dependency path")'
            result=subprocess.run([sys.executable,'-I','-c',code],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,check=True)
            self.assertIn('PASS isolated',result.stdout)

class OwnerThresholdTests(unittest.TestCase):
    def guard(self,root,owned=3700,global_used=500,total=1000):
        source=Path(__file__).with_name('run_native_replay_window.py').read_text()
        tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        fn=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='memory_guard')
        audit=types.SimpleNamespace(sample=mock.Mock(return_value=(dict(per_gpu_memory_mib={'gpu0':owned}),{})))
        clock=types.SimpleNamespace(monotonic=mock.Mock(side_effect=range(100,1000)),time=lambda:1.)
        env=dict(json=json,guarded=True,last_gpu_sample=[0.],bad_samples=[0],ledger={},root=root,REPO=Path(__file__).resolve().parents[3],time=clock,subprocess=types.SimpleNamespace(check_output=lambda *a,**kw:f'{global_used}, {total}'),registered_audit=audit)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual-owner-memory-guard>','exec'),env)
        return env,audit
    def test_three_strikes_and_reset_only_on_clean_sample(self):
        with tempfile.TemporaryDirectory() as d:
            env,audit=self.guard(Path(d));audit.sample.side_effect=[RuntimeError('unknown')]*3
            import gpu_device,nvml_processes
            with mock.patch.object(gpu_device,'inventory',return_value=CARDS),mock.patch.object(nvml_processes,'sample'):
                env['memory_guard']('default.log');env['memory_guard']('default.log')
                with self.assertRaisesRegex(RuntimeError,'unverified for3samples'):env['memory_guard']('default.log')
        with tempfile.TemporaryDirectory() as d:
            env,audit=self.guard(Path(d));audit.sample.side_effect=[RuntimeError('unknown'),(dict(per_gpu_memory_mib={'gpu0':3700}),{})]
            with mock.patch.object(gpu_device,'inventory',return_value=CARDS),mock.patch.object(nvml_processes,'sample'):
                env['memory_guard']('default.log');self.assertEqual(env['bad_samples'][0],1)
                env['memory_guard']('default.log');self.assertEqual(env['bad_samples'][0],0)
    def test_original_memory_thresholds_and_launch_skip_are_unchanged(self):
        import gpu_device,nvml_processes
        with mock.patch.object(gpu_device,'inventory',return_value=CARDS),mock.patch.object(nvml_processes,'sample'):
            for owned,used,message in [(7168.01,500,'exceeded7GiB'),(3700,790,'79% ceiling')]:
                with tempfile.TemporaryDirectory() as d:
                    env,audit=self.guard(Path(d),owned,used)
                    with self.assertRaisesRegex(RuntimeError,message):env['memory_guard']('default.log')
            with tempfile.TemporaryDirectory() as d:
                env,audit=self.guard(Path(d),7168,789.9);env['memory_guard']('default.log')
            with tempfile.TemporaryDirectory() as d:
                env,audit=self.guard(Path(d));audit.sample.side_effect=RuntimeError('must not sample during launch')
                env['memory_guard']('launch.log');audit.sample.assert_not_called()
if __name__=='__main__':unittest.main()
