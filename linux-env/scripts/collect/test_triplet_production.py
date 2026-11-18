"""Production runner entrances with real native/group/finalization code, fake transports."""
import copy,hashlib,json,os,subprocess,sys,tempfile,time,unittest,threading
from pathlib import Path
from unittest.mock import patch
import numpy as np
from types import SimpleNamespace as NS
import triplet_production as production
import triplet_artifacts as artifacts
from triplet_media import measure,POLICY as MEDIA_POLICY
from triplet_production_plan import initial_wave
from target_triplets import spec,member,seal_run,validate_run
from collection_status import begin,finish,terminal
from tick_contract import require_snapshots
from storage import atomic_json
import test_target_triplets


class ProductionRunnerTests(unittest.TestCase):
    def test_manifest_refuses_symlink_leaf_dangling_entry_and_ancestor(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'root';root.mkdir();(root/'a').write_text('source');original=artifacts.manifest(root)
            for target in (root/'a',root/'missing'):
                link=root/'link';link.symlink_to(target)
                with self.assertRaisesRegex(ValueError,'Non-regular'):artifacts.manifest(root)
                link.unlink()
            alias=Path(d)/'alias';alias.symlink_to(root,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'ancestor'):artifacts.manifest(alias)
            self.assertEqual(original,artifacts.manifest(root))
    def test_registry_projection_is_separate_and_keeps_authoritative_sha(self):
        source=dict(sealed=True,role='authoritative-rtx-native',production_root=str(production.ROOT/'datasets/counterfactual-targets/production-v1'),
            finalization_path=str(production.ROOT/'final.json'),wave_ledger_path=str(production.ROOT/'waves.json'),
            layouts=[dict(requested_layout_seed=720000,dataset_root=str(production.ROOT/'dataset'),raw_root=str(production.ROOT/'raw'),validation_path=str(production.ROOT/'validation.json'))],
            failed_groups=[dict(proof_path=str(production.ROOT/'proof.json'),raw_root=str(production.ROOT/'bad'))])
        before=copy.deepcopy(source);projected=production.backup_projection(source,'a'*64)
        self.assertEqual(source,before);self.assertEqual(projected['role'],'derived-dual-backup-projection')
        self.assertEqual(projected['authoritative_registry_sha256'],'a'*64)
        self.assertTrue(projected['layouts'][0]['raw_root'].startswith(str(production.DUAL)))
        self.assertEqual(projected['authoritative_registry_path'],source['production_root']+'/train-registry.json')
        source['layouts'][0]['raw_root']='/mnt/storage1/foreign'
        with self.assertRaisesRegex(ValueError,'Non-RTX'):production.backup_projection(source,'a'*64)
    def test_source_manifest_refuses_missing_changed_foreign_or_fixture_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';run.mkdir();(root/'code.py').write_text('pass')
            value=dict(git_commit='a'*40,source_sha256={'code.py':artifacts.sha(root/'code.py')})
            for kind in ('valid','missing','changed','foreign','fixture','cut'):
                v=copy.deepcopy(value)
                if kind=='missing':v['source_sha256']={}
                elif kind=='changed':v['source_sha256']['code.py']='b'*64
                elif kind=='foreign':v['source_sha256']={'../outside':'a'*64}
                elif kind=='fixture':v['cpu_fixture_only']=True
                elif kind=='cut':v['git_commit']='b'*40
                (run/'run.json').write_text(json.dumps(v))
                with patch.object(production,'REPO',root):
                    if kind=='valid':production.verify_sources(run,'a'*40)
                    else:
                        with self.subTest(kind=kind),self.assertRaises(ValueError):production.verify_sources(run,'a'*40)
    def test_production_refuses_other_hosts_instead_of_relaxing_by_hostname(self):
        with patch.object(production,'profile',return_value=dict(name='dual-a4500')):
            with self.assertRaisesRegex(ValueError,'only on rtx6000'):production.require_host()
    def test_stop_publication_preserves_the_first_cause_and_each_later_fault(self):
        with tempfile.TemporaryDirectory() as d:
            production.publish_stop(Path(d),dict(error='original camera fault'))
            production.publish_stop(Path(d),dict(error='secondary cleanup fault'))
            self.assertEqual(artifacts.read(Path(d)/'STOP.json')['error'],'original camera fault')
            self.assertEqual(len(list(Path(d).glob('fault-*.json'))),2)
    def test_coordinator_admission_must_match_the_published_group(self):
        with tempfile.TemporaryDirectory() as d:
            worker=object.__new__(production.Worker);worker.root=Path(d);worker.check=lambda:None
            atomic_json(worker.root/'admitted-720000.json',dict(proof_sha256='changed'))
            with self.assertRaisesRegex(ValueError,'admission differs'):worker.await_admission(720000,dict(proof_sha256='original'))
    def test_final_outcome_published_immediately_before_worker_exit_is_drained(self):
        rows=[];retired=[]
        def poll():rows.append('last outcome');return 0
        with tempfile.TemporaryDirectory() as d:
            result=production.wait_workers([(NS(poll=poll),None,3)],lambda:list(rows),lambda:None,retired.append,Path(d)/'STOP')
        self.assertEqual(result,['last outcome']);self.assertEqual(retired,[3])
    def test_worker_exit_or_failed_retirement_stops_without_waiting_for_peers(self):
        with tempfile.TemporaryDirectory() as d:
            for kind in ('exit','retirement'):
                def retired(slot):raise RuntimeError('Native cleanup unresolved')
                with self.subTest(kind=kind),self.assertRaises(RuntimeError):
                    production.wait_workers([(NS(poll=lambda:-9 if kind=='exit' else 0),None,3),(NS(poll=lambda:None),None,4)],
                        lambda:[],lambda:None,retired,Path(d)/'STOP',pause=lambda _:self.fail('Should stop immediately'))
    def test_disk_drop_after_admission_publishes_stop_and_requests_owned_retirement(self):
        for filesystem in ('root','bulk'):
            with self.subTest(filesystem=filesystem),tempfile.TemporaryDirectory() as d:
                called=threading.Event();samples=[0]
                def probe():
                    samples[0]+=1
                    if samples[0]>1:raise RuntimeError(filesystem+' below floor after admission')
                    return dict(root_free_bytes=30*2**30,bulk_free_bytes=60*2**30)
                watch=production.DiskWatch(Path(d),called.set,probe=probe,interval=.01)
                watch.start();self.assertTrue(called.wait(2));watch.close()
                self.assertIn(filesystem,artifacts.read(Path(d)/'STOP.json')['error'])
                with self.assertRaisesRegex(RuntimeError,'disk monitor'):watch.require()
                with patch.object(production,'disk',return_value={}):
                    with self.assertRaises(RuntimeError):production.publication_gate(Path(d),watch)
                    with self.assertRaisesRegex(RuntimeError,'STOP'):
                        production.registries(dict(production_root=d),dict(complete=True))
                self.assertFalse((Path(d)/'REGISTRY-SHA256.json').exists())
    def test_real_stage_polling_checks_disk_after_its_initial_admission(self):
        with tempfile.TemporaryDirectory() as d:
            worker=object.__new__(production.Worker);worker.root=Path(d);worker.global_root=Path(d);worker.stop_file=Path(d)/'STOP.json'
            worker.cut='a'*40;worker.instance=3;worker.env={};worker.claim=NS(fd=0);worker.stages=[]
            worker.ledger=NS(add=lambda _:None,scan=lambda:None)
            process=NS(pid=123,poll=lambda:None,returncode=None)
            with patch.object(production,'frozen'),patch.object(production.subprocess,'Popen',return_value=process),\
                    patch.object(production,'disk',side_effect=[{},RuntimeError('bulk below floor after admission')]) as gate:
                with self.assertRaisesRegex(RuntimeError,'below floor'):worker.stage('fake-recording',['fake'])
            self.assertEqual(gate.call_count,2);self.assertTrue(worker.stop_file.exists())
    def test_disk_numeric_floors_are_checked_for_both_filesystems(self):
        for root,bulk,passes in ((20,50,True),(19.99,70,False),(30,49.99,False)):
            with patch('shutil.disk_usage',side_effect=[NS(free=int(root*2**30)),NS(free=int(bulk*2**30))]):
                if passes:production.disk()
                else:
                    with self.assertRaisesRegex(RuntimeError,'disk gate'):production.disk()
    def test_unresolved_down_cannot_become_an_immutable_retirement_on_a_second_call(self):
        worker=object.__new__(production.Worker);worker.claim=None;worker.retirement_certified=False
        with self.assertRaisesRegex(RuntimeError,'remains unresolved'):worker.down()
    def test_coordinator_backs_up_a_partial_group_when_the_worker_could_not(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);job=initial_wave()[0];recipe=dict(cut='a'*40,production_root=str(root/'production'))
            Path(recipe['production_root']).mkdir();run=root/'datasets/counterfactual-targets/train/layout-720000';run.mkdir(parents=True)
            (run/'partial-evidence.json').write_text('{"partial":1}')
            def backup(source,files,destination,log):return dict(passed=True,files_sha256=files,destination=str(destination))
            with patch.object(production,'ROOT',root),patch.object(production,'backup',backup):
                first=production.backup_remaining_terminal(recipe,[job],'original collector fault')
                self.assertEqual(len(first['records']),1);self.assertEqual(first['errors'],[])
                self.assertEqual(production.backup_remaining_terminal(recipe,[job],'same fault')['records'],[])
                (run/'partial-evidence.json').write_text('{"partial":2}')
                self.assertEqual(len(production.backup_remaining_terminal(recipe,[job],'late byte change')['records']),1)
            self.assertFalse((Path(recipe['production_root'])/'REGISTRY-SHA256.json').exists())
    def test_terminal_fallback_skips_inherited_dispositions_and_preserves_own_partial(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);jobs=initial_wave()[:2]+[next(j for j in initial_wave() if j['layout_seed']==720094)]
            recipe=dict(production_root=str(root/'production'),inherited_outcomes=[
                dict(split=jobs[0]['split'],layout_seed=jobs[0]['layout_seed'],status='selected'),
                dict(split=jobs[1]['split'],layout_seed=jobs[1]['layout_seed'],status='refused')])
            self.assertEqual({(type(row['split']),type(row['layout_seed'])) for row in recipe['inherited_outcomes']+jobs},{(str,int)})
            Path(recipe['production_root']).mkdir();calls=[]
            for job in jobs:
                run=root/f'datasets/counterfactual-targets/{job["split"]}/layout-{job["layout_seed"]}'
                run.mkdir(parents=True);(run/'evidence.json').write_text('{}')
            def backup(recipe,job,error):
                calls.append(job['layout_seed']);return dict(passed=True)
            with patch.object(production,'ROOT',root),patch.object(production,'backup_terminal',backup):
                result=production.backup_remaining_terminal(recipe,jobs,'startup failed')
            self.assertEqual(calls,[jobs[2]['layout_seed']])
            self.assertEqual(result,dict(records=[dict(passed=True)],errors=[]))
            self.assertFalse((Path(recipe['production_root'])/'terminal-diagnostics').exists())
    def test_standalone_help_bootstraps_own_sources_with_empty_pythonpath(self):
        env=dict(os.environ,PYTHONPATH='',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run([sys.executable,'-B',str(Path(production.__file__)),'--help'],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--plan',result.stdout)
    def test_only_source_verified_visible_margin_or_contract_valid_quality_is_replaceable(self):
        for kind in ('visible','visible_extra_fault','visible_writer','quality','quality_bad_tick'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                job=initial_wave()[0];h=test_target_triplets.TripletTests().fixture(d,720000)
                run=h.root;start=time.time_ns();cert=begin(run);params=member(h.config,0)
                episode=hashlib.sha256((str(run)+':0:720000').encode()).hexdigest();r=dict(params=params,episode_id=episode)
                if kind.startswith('visible'):
                    h.data['bodies']['red_2x3_1']['position']=list(h.data['bodies']['red_2x3_0']['position'])
                    try:h.collector._prepare_scene_attempt(params,{},r,set(),run,run/'attempt-0000')
                    except ValueError as exc:r['error']=repr(exc)
                    else:self.fail('Ambiguous layout was accepted')
                else:
                    h.collector._prepare_scene_attempt(params,{},r,set(),run,run/'attempt-0000')
                    contract=require_snapshots([r['episode_start_physical']])
                    if kind=='quality_bad_tick':contract['proofs'][0]['row']['wrist_stamp_ns']+=10**9
                    r['quality_reject']=dict(kind='home_unstable',contract=contract,evidence=dict(reason='measured physical instability'))
                r.update(accepted=False,end_cancel=dict(held=True),retirement=dict(passed=True),camera_boundary=dict(passed=True),
                    disposition=dict(kind='terminal' if kind.startswith('visible') else 'quality_reject',reason='unclassified_failure' if kind.startswith('visible') else 'home_unstable'))
                directory=run/'attempt-0000';directory.mkdir(exist_ok=True)
                atomic_json(directory/'receipt.json',r)
                atomic_json(run/'run.json',dict(counterfactual_triplet=h.config,normal_only=True,instance=3))
                atomic_json(run/'summary.json',dict(producer_invocation=cert['invocation'],complete=False,triplet_rejected=True))
                if kind.startswith('visible'):
                    terminal(run,dict(error=r['error'],source='attempt disposition'),directory)
                    if kind=='visible_extra_fault':terminal(run,dict(error='Camera ambiguity',source='sensor'))
                    if kind=='visible_writer':(directory/'frames.jsonl').write_text('{}\n')
                finish(run,cert,[])
                with patch.object(artifacts,'destination',lambda *a:None):
                    if kind in ('visible','quality'):
                        status,reason,_=artifacts.classify(run,job,dict(pid=os.getpid(),started_wall_ns=start))
                        self.assertEqual(status,'refused');self.assertEqual(reason,'visible_target_margin' if kind=='visible' else 'quality:home_unstable')
                    else:
                        with self.assertRaises((ValueError,RuntimeError)):artifacts.classify(run,job,dict(pid=os.getpid(),started_wall_ns=start))

    def fake_attempt(self,root,bad=None,production_root=None,layout_seed=720000,return_transports=False):
        """Real Worker.run -> attempt -> classify -> seal; only I/O transports fake."""
        root=Path(root);job=next(j for j in initial_wave() if j['layout_seed']==layout_seed);recipe=dict(cut='a'*40,production_root=str(production_root or root/'production'),runtime_root=str(root/'runtime'))
        recipe_root=Path(recipe['production_root']);recipe_root.mkdir()
        source=root/'fixture_source.py';source.write_text('pass')
        rows_written=[];events=[];native_groups=[]
        def decode(run):
            group=validate_run(run,base=root)
            states=[]
            for i in range(3):
                states.append(json.loads((run/f'attempt-{i:04d}/frames.jsonl').read_text().splitlines()[0])['state'])
            actions=np.zeros((3,50,6));actions[:,0,0]=[0,.01,.02]
            images={c:[np.full((480,640,3),100 if (run/f'attempt-{i:04d}/{c}.mp4').read_bytes()==b'fake video transport' else 200,np.uint8) for i in range(3)] for c in ('top','wrist')}
            result=measure(group['first_recorded_physical_sha256'],states,actions,images)
            files={n:h for n,h in artifacts.manifest(run).items() if n.startswith('attempt-') and n.rsplit('/',1)[1] in ('receipt.json','frames.jsonl','top.mp4','wrist.mp4','first-recorded-native.json')}
            result.update(layout_seed=job['layout_seed'],split=job['split'],triplet_id=group['triplet_id'],
                validated_group_sha256=artifacts.sha(run/'counterfactual-triplet.json'),raw_files_sha256=files)
            return result
        def stage(worker,name,argv,**kwargs):
            events.append(name);start=time.time_ns();run=root/f'datasets/counterfactual-targets/train/layout-{job["layout_seed"]}'
            if name.startswith('collect-'):
                self.assertIn('--normal-only',argv);self.assertNotIn('--resume',argv)
                if bad=='collect_timeout':
                    run.mkdir(parents=True);(run/'partial-evidence.json').write_text('{"original":"collector timeout"}')
                    raise TimeoutError('Stage deadline: collect-720000')
                run.mkdir(parents=True);cert=begin(run);h=test_target_triplets.Harness(run,spec(job['layout_seed'],'train'))
                prepare=h.collector._prepare_scene_attempt
                def identified(params,baseline,receipt,*args):
                    receipt['episode_id']=hashlib.sha256((str(run)+':'+str(params['trial'])+':'+str(job['layout_seed'])).encode()).hexdigest()
                    return prepare(params,baseline,receipt,*args)
                h.collector._prepare_scene_attempt=identified
                h.complete(recorded_ticks=50)
                config=artifacts.read(run/'run.json');config.update(normal_only=True,instance=job['instance'],git_commit='a'*40,source_sha256={'fixture_source.py':artifacts.sha(source)})
                atomic_json(run/'run.json',config)
                for i in range(3):
                    for camera in ('top','wrist'):(run/f'attempt-{i:04d}/{camera}.mp4').write_bytes(b'fake video transport')
                seal=seal_run(run,base=root)
                atomic_json(run/'summary.json',dict(producer_invocation=cert['invocation'],complete=True,episodes=3,counterfactual_triplet_seal=seal))
                finish(run,cert,[]);native_groups.append(validate_run(run,base=root))
                if bad=='source':(run/'unexpected').symlink_to(source)
            elif name.startswith('media-'):
                # Execute the real CLI body. Only camera/expert decoding and the
                # fixture's filesystem root are transports supplied by this test.
                import triplet_media,storage,io
                from contextlib import redirect_stdout
                def media_input(run):
                    media=decode(run)
                    if bad=='media':media['physical_hashes'][1]='b'*64
                    return media
                code=0
                with patch.object(sys,'argv',['triplet_media.py',str(run),'--output',str(run/'production-media.json')]), \
                        patch.object(storage,'confined',lambda value:Path(value)),patch.object(triplet_media,'inspect_run',media_input),redirect_stdout(io.StringIO()):
                    try:triplet_media.main()
                    except SystemExit as exc:code=exc.code
                    except Exception:code=1
                if code:return dict(name=name,pid=os.getpid(),started_wall_ns=start,returncode=code,wall_s=.1)
            elif name.startswith(('report-','report-validation-')):
                report=run/('report-validation' if name.startswith('report-validation-') else 'report');report.mkdir()
                atomic_json(report/'validation.json',dict(passed=bad!='export',loaded=True,timing_passed=True,fps=30,episodes=3,frames=150,
                    completed_wall_ns=time.time_ns(),export_content=dict(video=dict(decoded_frame_pairs=300)),counterfactual_triplet=native_groups[-1]))
                atomic_json(report/'export-environment.json',dict(passed=True,dependencies_unchanged=True))
            else:raise AssertionError('Unexpected transport stage: '+name)
            return dict(name=name,pid=os.getpid(),started_wall_ns=start,returncode=0,wall_s=.1)
        def up(worker):events.append('up');worker.claim=object()
        def down(worker):
            if worker.claim is not None:events.append('down');worker.claim=None
        def backup(source,files,destination,log):
            diagnostic='terminal-diagnostics' in source.parts
            events.append('diagnostic-backup' if diagnostic else 'sha-backup')
            if bad=='backup_persistent' or bad=='backup' and not diagnostic:raise RuntimeError('SHA backup failed')
            self.assertEqual(files,artifacts.manifest(source))
            return dict(passed=True,destination=str(destination),files_sha256=files)
        def rpc(worker,code):events.append('held-restore');return dict(arm_restored=bad!='restore',command_baseline=dict(epoch=10))
        with patch.object(production,'ROOT',root),patch.object(production,'REPO',root),patch.object(production,'frozen'),\
                patch.object(production,'disk',return_value=dict(passed=True)),patch.object(production,'profile',return_value=dict(export_python=sys.executable)),\
                patch.object(production.Worker,'stage',stage),patch.object(production.Worker,'up',up),patch.object(production.Worker,'down',down),\
                patch.object(production.Worker,'await_admission',lambda *a:events.append('coordinator-admission')),\
                patch.object(production.Worker,'rpc',rpc),patch.object(production,'backup',backup),\
                patch.object(artifacts,'destination',lambda run,config:None),patch.object(artifacts,'validate_run',lambda run:validate_run(run,base=root)),\
                patch('triplet_media.inspect_run',decode):
            worker=production.Worker(recipe,job['instance'],0)
            if bad:
                with self.assertRaises(RuntimeError):worker.run([job,initial_wave()[4]])
                self.assertTrue((recipe_root/'STOP.json').exists());self.assertFalse(artifacts.read(worker.root/'worker-finalization.json')['passed'])
                final=artifacts.read(worker.root/'worker-finalization.json')
                if bad=='backup_persistent':self.assertIsNotNone(final['terminal_backup_error'])
                else:
                    self.assertIsNotNone(final['terminal_backup']);self.assertIsNone(final['terminal_backup_error'])
                    diagnostic=artifacts.read(Path(artifacts.read(final['terminal_backup']['path'])['snapshot'])/'terminal-disposition.json')
                    self.assertTrue(diagnostic['training_forbidden']);self.assertEqual(diagnostic['job'],job)
                    if bad=='collect_timeout':self.assertIn('partial-evidence.json',diagnostic['catalog']['group']['regular_files_sha256'])
                    if bad=='source':self.assertTrue(any(e['kind']=='symlink' for e in diagnostic['catalog']['group']['entries']))
            else:
                worker.run([job]);outcome=artifacts.read(worker.root/f'outcome-{job["layout_seed"]}.json')
                proof=production.verify_outcome(outcome,recipe)
                entry=artifacts.selected_entry(job,Path(proof['raw_root']),proof['files_sha256'])
                self.assertEqual(len(entry['raw_files_sha256']),15);self.assertEqual(outcome['status'],'selected')
                self.assertTrue(artifacts.read(worker.root/'worker-finalization.json')['passed'])
        self.assertEqual(events[-1],'diagnostic-backup' if bad else 'down')
        if return_transports:return dict(stage=stage,backup=backup,decode=decode,job=job)
        return events
    def test_full_fake_group_driver_reaches_seal_backup_restore_and_owned_retirement(self):
        with tempfile.TemporaryDirectory() as d:
            events=self.fake_attempt(d)
            self.assertEqual(events,['up','collect-720000','media-720000','report-720000','report-validation-720000','sha-backup','coordinator-admission','held-restore','down'])
    def test_fake_driver_refuses_source_media_export_backup_and_restore_failures(self):
        for kind in ('collect_timeout','source','media','export','backup','backup_persistent','restore'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:self.fake_attempt(d,kind)


if __name__=='__main__':unittest.main()
