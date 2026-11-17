"""Continuation uses real disposition/ledger entrances; native and SSH transports are fake."""
import copy,json,os,shutil,tempfile,time,unittest
from pathlib import Path
from contextlib import ExitStack,contextmanager,nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch
import triplet_production as p
import triplet_continuation as c
import triplet_artifacts as a
from triplet_production_plan import initial_wave
from storage import atomic_json
from target_triplets import validate_run
import test_triplet_production as runner_tests


@contextmanager
def original_fixture():
    with tempfile.TemporaryDirectory() as d,ExitStack() as stack:
        root=Path(d);origin=root/'datasets/counterfactual-targets/production-v1';dual=root/'dual'
        stack.enter_context(patch.object(p,'DUAL',dual))
        origin.parent.mkdir(parents=True)
        runner_tests.ProductionRunnerTests().fake_attempt(root,production_root=origin)
        recipe=dict(cut='a'*40,jobs=initial_wave(),production_root=str(origin),runtime_root=str(root/'runtime'),bootstrap_capsule_path=str(root/'capsule.json'),bootstrap_capsule_sha256='a'*64)
        stack.enter_context(patch.object(p,'plan',lambda *args,**kwargs:copy.deepcopy(recipe)))
        atomic_json(origin/'plan.json',recipe);atomic_json(origin/'wave-00.json',dict(jobs=initial_wave(),outcomes=[]))
        outcome=a.read(origin/'wave-00/slot-3/outcome-720000.json')
        atomic_json(origin/'wave-00/slot-3/admitted-720000.json',outcome)
        atomic_json(origin/'progress.json',dict(wave=0,prior_selected=0,outcomes=[outcome]))
        owner=root/'setup/coordinator.pid.identity.json';owner.parent.mkdir()
        atomic_json(owner,dict(state='stopped',nonce='nonce',supervision_error=None,cleanup_error=None,cleanup=dict(remaining=[]),workload=dict(pid=777777,start_ticks='5')))
        stop=dict(error="RuntimeError('Startup failed; see up-3.log')",slot=5,wall_ns=100,supervision=dict(receipt_path=str(owner),nonce='nonce'))
        atomic_json(origin/'STOP.json',stop)
        atomic_json(origin/'production-finalization.json',dict(passed=False,retirement_errors=[],cleanup=dict(remaining=[]),terminal_backups=dict(errors=[]),backup_children_cleanup=dict(remaining=[])))
        atomic_json(owner.parent/'post-stop-owned-down.json',dict(passed=True,remaining=[],scoped_survivors=[],unresolved_environments=[],original_cleanup_failures=[],ports={'8783':False},production_passed=False,sources=[dict(path=str(owner),sha256=a.sha(owner))]))
        run=root/'datasets/counterfactual-targets/train/layout-720000'
        proof=a.read(outcome['proof_path']);destination=Path(proof['backup']['destination']);shutil.copytree(run,destination)
        for obj,name,value in [(p,'ROOT',root),(p,'REPO',root),(p,'confined',lambda x:Path(x)),(p,'frozen',lambda *a,**k:None),(p,'data_checkout',lambda *a,**k:None),(p,'identity',lambda _:None),(c,'confined',lambda x:Path(x)),(c,'COLLECTOR_CUT','a'*40),(a,'destination',lambda *a:None),(a,'validate_run',lambda run:validate_run(run,base=root))]:stack.enter_context(patch.object(obj,name,value))
        stack.enter_context(patch('triplet_media.inspect_run',lambda run:a.read(run/'production-media.json')))
        def remote(command,**kwargs):
            code=command[-1];assert code.startswith('python3 -c ')
            import shlex,subprocess,sys
            return subprocess.check_output([sys.executable,'-c',shlex.split(code)[2]],text=True)
        # Avoid patching subprocess.check_output globally; backup-only transport.
        real_backup=c.backup_recheck
        def backup(proof):
            if a.manifest(Path(proof['backup']['destination']))!=proof['files_sha256']:raise ValueError('Backup bytes changed')
            return dict(passed=True,files=len(proof['files_sha256']))
        stack.enter_context(patch.object(c,'backup_recheck',backup))
        yield NS(root=root,origin=origin,recipe=recipe,outcome=outcome,run=run,owner=owner,checkpoint=a.sha(origin/'progress.json'),backup=destination)


class ContinuationTests(unittest.TestCase):
    def test_actual_original_admission_replays_source_backup_and_dead_workload(self):
        with original_fixture() as f:
            result=c.origin_state(f.origin,f.root,f.checkpoint)
            self.assertEqual(result['outcomes'],[f.outcome]);self.assertEqual(result['owner_nonce'],'nonce')
            self.assertIn(str(f.origin/'STOP.json'),result['bindings'])
            self.assertFalse((f.origin/'REGISTRY-SHA256.json').exists())
    def test_original_admission_refuses_source_backup_job_member_and_birth_changes(self):
        for kind in ('raw','backup','checkpoint','job','admission','live-workload','orphan','duplicate','cut','old-quarantine'):
            with self.subTest(kind=kind),original_fixture() as f:
                if kind=='raw':(f.run/'attempt-0000/frames.jsonl').write_text('changed')
                elif kind=='backup':(f.backup/'attempt-0000/top.mp4').write_bytes(b'changed')
                elif kind=='checkpoint':atomic_json(f.origin/'progress.json',dict(wave=0,prior_selected=0,outcomes=[]))
                elif kind=='job':
                    wave=a.read(f.origin/'wave-00.json');wave['jobs'][0]['requested_layout_seed']+=1;atomic_json(f.origin/'wave-00.json',wave)
                elif kind=='admission':atomic_json(f.origin/'wave-00/slot-3/admitted-720000.json',{})
                elif kind=='orphan':atomic_json(f.origin/'proofs/train/orphan.json',{})
                elif kind=='duplicate':atomic_json(f.origin/'wave-00/slot-3/outcome-duplicate.json',f.outcome)
                elif kind=='cut':
                    recipe=a.read(f.origin/'plan.json');recipe['cut']='b'*40;atomic_json(f.origin/'plan.json',recipe)
                origin=f.root/'quarantine/90e1d99' if kind=='old-quarantine' else f.origin
                extra=patch.object(p,'identity',return_value=dict(start_ticks='5',state='R')) if kind=='live-workload' else nullcontext()
                with extra,self.assertRaises((ValueError,RuntimeError,FileNotFoundError)):c.origin_state(origin,f.root,f.checkpoint)
    def test_caller_seal_and_original_failure_history_are_mandatory(self):
        with original_fixture() as f:
            state=c.origin_state(f.origin,f.root,f.checkpoint)
            admission=f.root/'manual-admission/admission.json';admission.parent.mkdir();value=dict(kind='same-run-continuation',collector_cut='a'*40,runner_cut='b'*40,collector_repo=str(f.root),origin_root=str(f.origin),origin_bindings=state['bindings'],inherited_outcomes=[f.outcome],next_production_root=str(f.origin/'continuations/one'),next_runtime_root=str(f.root/'runtime/continuations/one'))
            value['preparation_files_sha256']={}
            atomic_json(admission,value)
            recipe=c.load_plan(admission,a.sha(admission),'b'*40)
            self.assertEqual(recipe['cut'],'a'*40);self.assertEqual(recipe['runner_cut'],'b'*40);self.assertEqual(recipe['startup_configure_timeout_s'],300)
            for digest in (None,'0'*64):
                with self.assertRaisesRegex(ValueError,'Caller-published'):c.load_plan(admission,digest,'b'*40)
            digest=a.sha(admission);atomic_json(f.origin/'STOP.json',dict(error='changed'))
            with self.assertRaisesRegex(ValueError,'history changed'):c.load_plan(admission,digest,'b'*40)
    def test_actual_backup_recheck_refuses_added_or_changed_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'file').write_text('source');proof=dict(files_sha256=a.manifest(root),backup=dict(destination=str(root)))
            real=p.subprocess.check_output
            def remote(argv,**kwargs):
                import shlex,sys
                return real([sys.executable,'-c',shlex.split(argv[-1])[2]],text=True)
            with patch.object(p,'DUAL',root),patch.object(p.subprocess,'check_output',remote):
                self.assertTrue(c.backup_recheck(proof)['passed'])
                (root/'late').write_text('new')
                with self.assertRaises(Exception):c.backup_recheck(proof)
    def test_configure_budget_changes_only_the_explicit_collection_rpc(self):
        script=Path(p.__file__).parents[1]/'isaac/isaacsim_launch.sh'
        text=script.read_text();branch=text.split('    collection-rate)')[1].split('    rate)')[0]
        self.assertIn('SOARM_COLLECTION_CONFIGURE_TIMEOUT:-30',branch)
        self.assertIn('--timeout "$configure_timeout"',branch)
        self.assertNotIn('SOARM_COLLECTION_CONFIGURE_TIMEOUT',text.replace(branch,''))
        wrapper=Path(p.__file__).parents[1]/'collect_sim.sh'
        source=wrapper.read_text();self.assertIn('DATA_SCRIPT_DIR="$REPO_ROOT/linux-env/scripts"',source)
        self.assertIn('exec python3 -u "$DATA_SCRIPT_DIR/collect/collect.py"',source)
        self.assertIn('bash "$launcher" collection-rate 30 --instance "$index"',source)

    def test_real_launcher_uses_300_seconds_only_when_explicitly_requested(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);tools=root/'bin';tools.mkdir();frozen=root/'frozen';code=frozen/'linux-env/scripts/collect/configure_clock.py'
            code.parent.mkdir(parents=True);code.write_text('FROZEN_SOURCE=True')
            recorder=root/'argv.json'
            shim=tools/'python3';shim.write_text('#!'+sys.executable+'\nimport os,sys,json\nif sys.argv[1:3]==["-m","kit_mcp.client"]:\n open(os.environ["ARGV_RECORD"],"w").write(json.dumps(sys.argv[1:]))\nelse:os.execv('+repr(sys.executable)+',['+repr(sys.executable)+',*sys.argv[1:]])\n');shim.chmod(0o755)
            script=Path(p.__file__).parents[1]/'isaac/isaacsim_launch.sh'
            env=dict(os.environ,PATH=str(tools)+os.pathsep+os.environ['PATH'],SOARM_COLLECTOR_REPO=str(frozen),ARGV_RECORD=str(recorder))
            for key in ('SOARM_INSTANCE_INDEX','SOARM_INSTANCE_ROOT'):env.pop(key,None)
            for budget in ('30','300'):
                env['SOARM_COLLECTION_CONFIGURE_TIMEOUT']=budget
                result=subprocess.run(['bash',str(script),'collection-rate','30'],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                args=json.loads(recorder.read_text());self.assertEqual(args[args.index('--timeout')+1],budget)
                self.assertTrue(json.loads(args[3])['code'].startswith('FROZEN_SOURCE=True'))
            env['SOARM_COLLECTION_CONFIGURE_TIMEOUT']='301';recorder.unlink()
            result=subprocess.run(['bash',str(script),'collection-rate','30'],env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse(recorder.exists())

    def test_actual_coordinator_completes_partial_wave_without_recounting_or_erasing_stop(self):
        from triplet_production_plan import initial_wave
        with tempfile.TemporaryDirectory() as d,ExitStack() as stack:
            root=Path(d);old=root/'old';old.mkdir();atomic_json(old/'STOP.json',dict(error='original startup timeout'))
            old_sha=a.sha(old/'STOP.json');jobs=initial_wave();job_by_seed={j['layout_seed']:j for j in jobs}
            def outcome(j):return dict(split=j['split'],layout_seed=j['layout_seed'],requested_layout_seed=j['requested_layout_seed'],status='selected',reason='accepted',proof_path=str(old/(str(j['layout_seed'])+'.json')),proof_sha256='a'*64)
            inherited=[outcome(j) for j in jobs[1:]]
            recipe=dict(cut='a'*40,runner_cut='b'*40,collector_repo=str(root/'frozen'),production_root=str(root/'new'),runtime_root=str(root/'runtime/worker-invocation'),start_stagger_s=60,
                continuation_admission_path=str(root/'admission.json'),inherited_outcomes=inherited)
            release=root/'release.json';atomic_json(release,dict(passed=True));spawned=[];finals=[]
            ledger=NS(known={},scan=lambda:None,add=lambda *_:None,stop=lambda:dict(remaining=[]))
            class Watch:
                def __init__(self,where,callback,report):self.report=report;self.closed=False
                def start(self):atomic_json(self.report,dict(healthy=True))
                def require(self):pass
                def close(self):self.closed=True
            def spawn(argv,**kwargs):
                slot=int(argv[argv.index('--worker')+1]);spawned.append(slot)
                directory=Path(recipe['production_root'])/f'wave-00/slot-{slot}';directory.mkdir(parents=True)
                for job in jobs:
                    if job['instance']==slot and job['layout_seed']==720000:atomic_json(directory/f'outcome-{job["layout_seed"]}.json',outcome(job))
                atomic_json(directory/'worker-finalization.json',dict(passed=True,cut='a'*40,instance=slot,wave=0,sessions=0))
                return NS(pid=999999,poll=lambda:0)
            def backups(value):atomic_json(Path(value['runtime_root'])/'registry-backup.json',dict(passed=True))
            ticks=iter(range(0,10000000,61))
            patches=[(p,'ROOT',root),(p,'require_host',lambda:None),(p,'qualified_recipe',lambda _:None),(p,'disk',lambda:{}),(p,'confined',lambda x:Path(x)),
                (p,'gpu_uuid',lambda _:p.ASSIGNED_UUID),(p,'OwnedProcesses',lambda *a:ledger),(p,'DiskWatch',Watch),
                (p,'verify_outcome',lambda row,recipe:dict(job=job_by_seed[row['layout_seed']])),
                (p,'registries',lambda *a:dict(train=dict(sealed=True),heldout=dict(sealed=True))),
                (p,'backup_registries',backups),(p,'capture_supervision',lambda:None),(c,'reverify',lambda _:dict(passed=True))]
            for obj,name,value in patches:stack.enter_context(patch.object(obj,name,value))
            stack.enter_context(patch.object(p.subprocess,'check_output',return_value=''))
            stack.enter_context(patch.object(p.subprocess,'Popen',spawn))
            stack.enter_context(patch.object(p.time,'monotonic',lambda:next(ticks)))
            p.coordinate(recipe,release)
            wave=a.read(Path(recipe['production_root'])/'wave-ledger.json')[0]
            self.assertEqual(len(wave['outcomes']),140);self.assertEqual(spawned,[3])
            self.assertEqual(a.sha(old/'STOP.json'),old_sha)
            self.assertTrue(a.read(Path(recipe['production_root'])/'PUBLICATION-COMPLETE.json')['passed'])
            with self.assertRaises(FileExistsError):p.coordinate(recipe,release)
            self.assertEqual(spawned,[3])

    def test_full_fake_preparation_finishes_two_exports_and_binds_terminal_replacement(self,failure=None):
        from collection_status import begin,terminal,finish
        with original_fixture() as f,ExitStack() as stack:
            transports={}
            for seed in c.INTERRUPTED_EXPORTS:
                transports[seed]=runner_tests.ProductionRunnerTests().fake_attempt(f.root,production_root=f.root/f'fake-{seed}',layout_seed=seed,return_transports=True)
                proof=a.read(f.root/f'fake-{seed}/proofs/train/{seed}.json');job=proof['job']
                directory=f.origin/f'wave-00/slot-{job["instance"]}';directory.mkdir(parents=True,exist_ok=True)
                atomic_json(directory/'stages.json',[proof['stage']])
            seed=c.PEER_INTERRUPTION;job=next(j for j in initial_wave() if j['layout_seed']==seed)
            run=f.root/f'datasets/counterfactual-targets/train/layout-{seed}';run.mkdir()
            started=time.time_ns();certificate=begin(run)
            atomic_json(run/'run.json',dict(git_commit='a'*40,source_sha256={'fixture_source.py':a.sha(f.root/'fixture_source.py')}))
            attempt=run/'attempt-0000';attempt.mkdir();(attempt/'frames.jsonl').write_text('{}\n')
            atomic_json(run/'summary.json',dict(complete=False,producer_invocation=certificate['invocation']))
            terminal(run,dict(error="TimeoutError('MoveIt state validity query did not complete')",source='MoveIt validity query'))
            finish(run,certificate,[])
            directory=f.origin/f'wave-00/slot-{job["instance"]}';directory.mkdir(parents=True,exist_ok=True)
            atomic_json(directory/'stages.json',[dict(name='collect-'+str(seed),pid=os.getpid(),started_wall_ns=started,returncode=None)])
            if failure=='media_verdict':
                (f.root/f'datasets/counterfactual-targets/train/layout-{c.INTERRUPTED_EXPORTS[0]}/attempt-0001/top.mp4').write_bytes(b'contrast frame transport')
            def stage(worker,name,argv,**kwargs):
                if failure=='media_tool' and name=='media-720087' or failure=='export_tool' and name=='report-720087':
                    result=dict(name=name,pid=os.getpid(),started_wall_ns=time.time_ns(),returncode=1,wall_s=.1);worker.stages.append(result);return result
                seed=int(name.rsplit('-',1)[1]);result=transports[seed]['stage'](worker,name,argv,**kwargs);worker.stages.append(result);return result
            def backup(source,files,destination,log):
                if failure=='backup_io':raise OSError('receiver unavailable')
                self.assertEqual(a.manifest(source),files)
                shutil.copytree(source,destination,dirs_exist_ok=True)
                return dict(passed=True,destination=str(destination),files_sha256=files)
            table=f.root/'table.log';table.write_text('verified fixture table')
            for obj,name,value in [(p,'require_host',lambda:None),(p,'disk',lambda:{}),(p,'profile',lambda:dict(export_python=__import__('sys').executable)),
                (p,'load_bootstrap_capsule',lambda *a:dict(pin=dict(source=dict(path=str(table))))),
                (p,'OwnedProcesses',lambda *a:NS(stop=lambda:dict(remaining=[]))),
                (p,'backup',backup),(p.Worker,'stage',stage)]:stack.enter_context(patch.object(obj,name,value))
            if failure in ('media_tool','export_tool','backup_io'):
                with self.assertRaises((RuntimeError,OSError)):c.prepare(f.origin,f.root,'b'*40,f.root/'prepare-one',f.checkpoint)
                self.assertFalse((f.root/'prepare-one/admission.json').exists())
                self.assertFalse((f.root/'prepare-one/acknowledged-interruption-720087.json').exists())
                self.assertFalse((f.root/'prepare-one/qualified/outcome-720087.json').exists())
                return
            result=c.prepare(f.origin,f.root,'b'*40,f.root/'prepare-one',f.checkpoint)
            self.assertTrue(result['passed']);self.assertEqual(result['inherited'],4);self.assertEqual(result['selected'],2 if failure=='media_verdict' else 3)
            admission=a.read(result['admission_path']);self.assertEqual(admission['completed_exports'],list(c.INTERRUPTED_EXPORTS))
            self.assertTrue((f.origin/'STOP.json').exists());self.assertFalse(a.read(f.origin/'production-finalization.json')['passed'])
            recipe=c.load_plan(result['admission_path'],result['admission_sha256'],'b'*40)
            peer=next(r for r in recipe['inherited_outcomes'] if r['layout_seed']==seed)
            self.assertEqual(peer['reason'],'operator_acknowledged_peer_stop')
            self.assertTrue(p.verify_outcome(peer,recipe)['original_terminal'])
            if failure=='media_verdict':
                refused=next(r for r in recipe['inherited_outcomes'] if r['layout_seed']==720087)
                checked=p.verify_outcome(refused,recipe)
                self.assertEqual(refused['reason'],'image_psnr');self.assertEqual(checked['media_stage']['returncode'],2)
                self.assertEqual(c.require_completed_checks(checked),'media')
                forged=copy.deepcopy(checked);forged['media_stage']['returncode']=1
                with self.assertRaises(ValueError):c.require_completed_checks(forged)
                forged=copy.deepcopy(checked);forged['media_receipt']['written_wall_ns']=0
                with self.assertRaises(ValueError):c.require_completed_checks(forged)
                return
            obsolete=f.root/'obsolete-interruption.json';badproof=dict(kind='interrupted_export_rejected',qualification_failure='tool error',export_stages={})
            atomic_json(obsolete,badproof);oldrow=dict(peer,proof_path=str(obsolete),proof_sha256=a.sha(obsolete))
            with self.assertRaisesRegex(ValueError,'not completed gate verdicts'):p.verify_outcome(oldrow,recipe)
            with (f.root/'evidence-backup.log').open('w') as log:c.backup_evidence(recipe,log)
            self.assertTrue((p.DUAL/f.origin.relative_to(f.root)/'proofs/train/720000.json').exists())
            self.assertTrue((p.DUAL/Path(peer['proof_path']).relative_to(f.root)).exists())
            for kind in ('duplicate','missing-original','edited-requested-slot','edited-proof-sha'):
                changed=copy.deepcopy(admission)
                if kind=='duplicate':changed['inherited_outcomes'].append(changed['inherited_outcomes'][0])
                elif kind=='missing-original':changed['inherited_outcomes']=changed['inherited_outcomes'][1:]
                elif kind=='edited-requested-slot':changed['inherited_outcomes'][-1]['requested_layout_seed']+=1
                else:changed['inherited_outcomes'][-1]['proof_sha256']='0'*64
                candidate_root=f.root/('candidate-'+kind);shutil.copytree(Path(result['admission_path']).parent,candidate_root)
                candidate=candidate_root/'admission.json';atomic_json(candidate,changed)
                with self.subTest(kind=kind),self.assertRaises((ValueError,RuntimeError)):
                    c.reverify(c.load_plan(candidate,a.sha(candidate),'b'*40))
            marker=run/'terminal-fault.json';fault=a.read(marker);fault['events'][0]['wall_ns']=99;atomic_json(marker,fault)
            with self.assertRaises(ValueError):p.verify_outcome(peer,recipe)

    def test_data_cut_refuses_a_different_head_even_when_scoped_sources_match(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.DEVNULL).strip()
            git('init','--initial-branch=dev');(root/'README.md').write_text('first')
            git('add','README.md');git('-c','user.name=CPU test','-c','user.email=cpu@example.invalid','commit','-qm','first')
            cut=git('rev-parse','HEAD');p.data_checkout(root,cut)
            (root/'README.md').write_text('second');git('add','README.md');git('-c','user.name=CPU test','-c','user.email=cpu@example.invalid','commit','-qm','second')
            p.frozen(cut,repo=root)
            with self.assertRaisesRegex(ValueError,'HEAD differs'):p.data_checkout(root,cut)

    def test_real_media_entry_refuses_existing_receipt_without_overwriting(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);receipt=root/'production-media.json';receipt.write_text('{"original":true}');before=receipt.read_bytes()
            script=Path(p.__file__).with_name('triplet_media.py')
            result=subprocess.run([sys.executable,'-B',str(script),str(root/'raw'),'--output',str(receipt)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('Fresh production-media receipt required',result.stderr)
            self.assertEqual(receipt.read_bytes(),before)

    def test_preparation_tool_and_backup_errors_never_publish_permanent_refusals(self):
        for failure in ('media_tool','export_tool','backup_io'):
            with self.subTest(failure=failure):self.test_full_fake_preparation_finishes_two_exports_and_binds_terminal_replacement(failure)
    def test_completed_failed_media_verdict_has_fresh_stage_evidence(self):
        self.test_full_fake_preparation_finishes_two_exports_and_binds_terminal_replacement('media_verdict')


if __name__=='__main__':unittest.main()
