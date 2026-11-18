"""Completed-wave admission entrances; old media/ROS/Git are transports."""
import copy,hashlib,json,os,shutil,sys,tempfile,time,unittest
from pathlib import Path
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
from types import SimpleNamespace as NS
import completed_continuation as c
import triplet_continuation as legacy
import triplet_production as p
import triplet_artifacts as a
import triplet_continuation as admission_api
import startup_retries as retries
from storage import atomic_json
from collection_status import begin,terminal,finish
from triplet_production_plan import initial_wave,reduce_waves,identity_counts,guards
from target_triplets import spec,destination,validate_run
import test_bootstrap_refusal as bootstrap_tests
from test_startup_retries import FIXTURE
import test_triplet_production as runner_tests


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as d,ExitStack() as stack:
        root=Path(d);dual=root/'dual';source=root/'fixture_source.py';source.write_text('pass')
        for obj,name,value in [(p,'ROOT',root),(p,'DUAL',dual),(p,'REPO',root),(p,'frozen',lambda *a,**k:None),
                (p,'data_checkout',lambda *a,**k:None),(p,'identity',lambda *_:None),(p,'require_host',lambda:None),
                (p,'disk',lambda:{}),(p,'capture_supervision',lambda:None),(legacy,'COLLECTOR_CUT','a'*40),
                (a,'destination',lambda run,config:destination(run,config,base=root)),
                (a,'validate_run',lambda run:validate_run(run,base=root))]:stack.enter_context(patch.object(obj,name,value))
        parent=c.parent_root();parent.mkdir(parents=True);jobs=initial_wave();outcomes=[];statuses={};inherited=[]
        refused=set(range(0,120,5))|{120,125,130}
        for i,job in enumerate(jobs):
            run=root/f'datasets/counterfactual-targets/{job["split"]}/layout-{job["layout_seed"]}';run.mkdir(parents=True)
            reason='accepted' if i not in refused else 'quality:planner_reject' if i in (110,115) else 'operator_acknowledged_peer_stop' if i==105 else 'visible_target_margin'
            status='selected' if reason=='accepted' else 'refused';statuses[job['layout_seed']]=(status,reason)
            atomic_json(run/'run.json',dict(git_commit='a'*40,source_sha256={'fixture_source.py':a.sha(source)}))
            files=a.manifest(run);dest=dual/'datasets/counterfactual-targets'/job['split']/run.name;shutil.copytree(run,dest)
            proof=dict(cut='a'*40,job=job,stage=dict(name='collect-'+str(job['layout_seed'])),raw_root=str(run),files_sha256=files,media=None,export_stages={},backup=dict(passed=True,destination=str(dest),files_sha256=files))
            path=parent/'proofs'/job['split']/(str(job['layout_seed'])+'.json');path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,proof)
            row=dict(split=job['split'],layout_seed=job['layout_seed'],requested_layout_seed=job['requested_layout_seed'],status=status,reason=reason,proof_path=str(path),proof_sha256=a.sha(path));outcomes.append(row)
            folder=parent/f'wave-00/slot-{job["instance"]}';folder.mkdir(parents=True,exist_ok=True)
            for prefix in ('outcome-','admitted-'):atomic_json(folder/(prefix+str(job['layout_seed'])+'.json'),row)
        wave=dict(jobs=jobs,outcomes=outcomes);state=reduce_waves([wave]);assert len(state['selected'])==113
        budget=retries.initialize(root/'runtime/counterfactual-startup-retries'/('c'*64),'c'*64)
        up=root/'up.log';up.write_bytes(FIXTURE.read_bytes());down=root/'down.json';atomic_json(down,dict(passed=True,errors=[],cleanup=dict(remaining=[]),ports_listening={'8787':False,'8788':False}))
        job=jobs[94];error=retries.failed_exit(dict(name='up-0',pid=os.getpid(),started_wall_ns=time.time_ns(),returncode=1),up)
        retries.failed(budget,5,job,error,down,None);reservation=retries.before_start(budget,5,job);retries.ready(budget,5,job,reservation)
        old=root/'setup/continuation-admission-5eaf5ad-v1';old.mkdir(parents=True)
        atomic_json(old/'admission.json',dict(kind='same-run-continuation',origin_checkpoint_sha256='c'*64,origin_root=str(parent.parent.parent),origin_bindings={},startup_retry_budget=budget))
        recipe=dict(cut='a'*40,runner_cut=c.PRIOR_RUNNER,collector_repo=str(root),source_sha256='fixture',production_root=str(parent),runtime_root=str(root/'runtime/counterfactual-production-v1/continuations/continuation-admission-5eaf5ad-v1'),
            continuation_admission_path=str(old/'admission.json'),continuation_admission_sha256=a.sha(old/'admission.json'),inherited_outcomes=outcomes[:94],startup_retry_budget=budget,bootstrap_capsule_path='fixture',bootstrap_capsule_sha256='f'*64,start_stagger_s=60)
        atomic_json(parent/'plan.json',recipe);atomic_json(parent/'wave-ledger.json',[wave]);atomic_json(parent/'wave-00.json',wave);atomic_json(parent/'wave-01.json',dict(jobs=state['next_wave'],outcomes=[]));atomic_json(parent/'progress.json',dict(wave=1,prior_selected=113,outcomes=[]))
        owner=c.owner_path();owner.parent.mkdir(parents=True,exist_ok=True);atomic_json(owner,dict(state='stopped',nonce='parent',supervisor=dict(pid=99999997,start_ticks='1'),workload=dict(pid=99999998,start_ticks='2'),cleanup=dict(remaining=[])))
        atomic_json(owner.parent/'post-stop-owned-down.json',dict(passed=True,remaining_owned=[],recorded_births=2,ports_listening={str(n):False for n in range(8783,8791)},managed_status=dict(supervision_ok=True),managed_stop=dict(supervision_ok=True),sources=[]))
        real_classify=a.classify
        def classify(run,job,*args):
            if job['layout_seed'] in statuses:return (*statuses[job['layout_seed']],None)
            return real_classify(run,job,*args)
        stack.enter_context(patch.object(p,'classify',classify));stack.enter_context(patch.object(c,'historical_parent',lambda r:copy.deepcopy(recipe)))
        def recheck(proof):
            if a.manifest(Path(proof['backup']['destination']))!=proof['files_sha256']:raise ValueError('Backup changed')
            return dict(passed=True)
        stack.enter_context(patch.object(legacy,'backup_recheck',recheck))
        real_checks=legacy.require_completed_checks
        stack.enter_context(patch.object(legacy,'require_completed_checks',lambda proof:None if proof['job']['layout_seed'] in (720087,720092) else real_checks(proof)))
        yield NS(root=root,parent=parent,recipe=recipe,wave=wave,state=state,old_admission=old/'admission.json',stack=stack)


class CompletedContinuationTests(unittest.TestCase):
    def finish_parent(self,f):
        atomic_json(f.parent/'STOP.json',dict(slot=6,error='fixture bootstrap identity failure',wall_ns=time.time_ns(),supervision=dict(nonce='parent',receipt_path=str(c.owner_path()))))
        atomic_json(f.parent/'production-finalization.json',dict(passed=False,retirement_errors=[],cleanup=dict(remaining=[]),backup_children_cleanup=dict(remaining=[]),terminal_backups=dict(errors=[]),startup_retries=retries.summary(f.recipe['startup_retry_budget'])))
        return c.checkpoint(f.parent)[1]

    def test_completed_parent_reverifies_every_proof_and_rejects_tampering(self):
        with fixture() as f:
            digest=self.finish_parent(f);state=c.parent_state(f.parent,digest)
            self.assertEqual(len(state['state']['selected']),113)
            path=Path(f.wave['outcomes'][100]['proof_path']);value=a.read(path);value['files_sha256']['edited']='0'*64;atomic_json(path,value)
            with self.assertRaisesRegex(ValueError,'checkpoint'):c.parent_state(f.parent,digest)
            with self.assertRaisesRegex(ValueError,'proof changed'):c.parent_state(f.parent,c.checkpoint(f.parent)[1])

    def test_current_admission_entrance_refuses_the_original_94_76_base(self):
        with fixture() as f:
            self.finish_parent(f)
            with self.assertRaisesRegex(ValueError,'94/76'):legacy.load_plan(f.old_admission,a.sha(f.old_admission),'b'*40)

    def prepare_case(self,f,bad=None):
        transports={}
        # Native/media/export transports use the real existing qualifier and
        # its real media CLI; no Kit, ROS service or GPU is launched.
        extended=initial_wave()+f.state['next_wave']
        with patch.object(runner_tests,'initial_wave',lambda:extended):
            for seed in c.EXPORTS:
                transports[seed]=runner_tests.ProductionRunnerTests().fake_attempt(f.root,production_root=f.root/('fake-'+str(seed)),layout_seed=seed,return_transports=True)
                proof=a.read(f.root/f'fake-{seed}/proofs/train/{seed}.json');job=proof['job'];folder=f.parent/f'wave-01/slot-{job["instance"]}';folder.mkdir(parents=True,exist_ok=True);atomic_json(folder/'stages.json',[proof['stage']])
        if bad=='media_refusal':(f.root/'datasets/counterfactual-targets/train/layout-720120/attempt-0001/top.mp4').write_bytes(b'different fixture video')
        peer_job=next(j for j in f.state['next_wave'] if j['layout_seed']==c.PEER);peer=f.root/f'datasets/counterfactual-targets/train/layout-{c.PEER}';peer.mkdir(parents=True);started=time.time_ns();certificate=begin(peer)
        atomic_json(peer/'run.json',dict(git_commit='a'*40,source_sha256={'fixture_source.py':a.sha(f.root/'fixture_source.py')},normal_only=True,instance=peer_job['instance'],counterfactual_triplet=spec(c.PEER,'train')));atomic_json(peer/'summary.json',dict(producer_invocation=certificate['invocation'],complete=False))
        peer_stage=dict(name='collect-'+str(c.PEER),pid=os.getpid(),started_wall_ns=started,returncode=None)
        folder=f.parent/f'wave-01/slot-{peer_job["instance"]}';folder.mkdir(parents=True,exist_ok=True);atomic_json(folder/'stages.json',[peer_stage])
        run,_,stage,_=bootstrap_tests.BootstrapRefusalTests().fixture(f.root)
        identity_job=next(j for j in f.state['next_wave'] if j['layout_seed']==c.IDENTITY);folder=f.parent/f'wave-01/slot-{identity_job["instance"]}';folder.mkdir(parents=True,exist_ok=True);atomic_json(folder/'stages.json',[stage])
        stop_ns=time.time_ns();terminal(peer,dict(error='MoveIt state validity service is unavailable',source='MoveIt validity query'));finish(peer,certificate,[])
        atomic_json(f.parent/'STOP.json',dict(slot=6,error='fixture bootstrap identity failure',wall_ns=stop_ns,supervision=dict(nonce='parent',receipt_path=str(c.owner_path()))))
        atomic_json(f.parent/'production-finalization.json',dict(passed=False,retirement_errors=[],cleanup=dict(remaining=[]),backup_children_cleanup=dict(remaining=[]),terminal_backups=dict(errors=[]),startup_retries=retries.summary(f.recipe['startup_retry_budget'])))
        if bad=='peer_before_stop':
            marker=a.read(peer/'terminal-fault.json');marker['events'][0]['wall_ns']=stop_ns-1;atomic_json(peer/'terminal-fault.json',marker)
        table=f.root/'table.log';table.write_text('fixture Kit table')
        def stage(worker,name,*args,**kwargs):
            if bad=='export_tool' and name=='report-720120':return dict(name=name,pid=os.getpid(),started_wall_ns=time.time_ns(),returncode=1)
            seed=int(name.rsplit('-',1)[1]);value=transports[seed]['stage'](worker,name,*args,**kwargs);worker.stages.append(value);return value
        def backup(source,files,destination,log):
            self.assertEqual(a.manifest(source),files);shutil.copytree(source,destination,dirs_exist_ok=True)
            return dict(passed=True,files_sha256=files,destination=str(destination))
        output=f.root/'setup/continuation-admission-combined-v1'
        with patch('triplet_media.inspect_run',lambda run:transports[int(run.name.split('-')[1])]['decode'](run)),patch.object(p.Worker,'stage',stage),patch.object(p,'backup',backup),patch.object(p,'profile',lambda:dict(export_python=sys.executable)),patch.object(p,'load_bootstrap_capsule',lambda *a:dict(pin=dict(source=dict(path=str(table))))):
            return c.prepare(f.parent,f.root,'b'*40,output,c.checkpoint(f.parent)[1]),output

    def test_real_preparation_carries_completed_wave_and_all_four_dispositions(self):
        with fixture() as f:
            result,output=self.prepare_case(f);self.assertEqual((result['inherited'],result['selected']),(144,115))
            admission=a.read(result['admission_path']);self.assertEqual(len(admission['inherited_waves'][0]['outcomes']),140)
            self.assertEqual(len(admission['pending_wave_outcomes']),4)
            counts=identity_counts(admission['inherited_outcomes']);self.assertEqual(counts['exact_identity_refusals'],1);self.assertEqual(counts['attempted_groups'],144)
            recipe=legacy.load_plan(result['admission_path'],result['admission_sha256'],'b'*40)
            self.assertEqual(len(recipe['inherited_outcomes']),4);self.assertEqual(len(recipe['inherited_waves']),1)
            self.assertEqual(retries.summary(recipe['startup_retry_budget'])['counts']['total'],1)
            self.assertEqual(a.read(f.parent/'wave-ledger.json'),[f.wave]);self.assertFalse((Path(recipe['production_root'])/'REGISTRY-SHA256.json').exists())
            forged=copy.deepcopy(admission);forged['startup_retry_budget']=retries.initialize(f.root/'runtime/replacement-budget','d'*64)
            path=Path(result['admission_path']);atomic_json(path,forged)
            with self.assertRaisesRegex(ValueError,'replaced its parent startup retry ledger'):legacy.load_plan(path,a.sha(path),'b'*40)
            atomic_json(path,admission)
            forged=copy.deepcopy(admission);forged['startup_retry_budget']['prefix']=[];atomic_json(path,forged)
            with self.assertRaisesRegex(ValueError,'parent retry prefix'):legacy.load_plan(path,a.sha(path),'b'*40)
            atomic_json(path,admission)



    def test_completed_failed_media_is_an_evidenced_refusal_not_a_tool_failure(self):
        with fixture() as f:
            result,_=self.prepare_case(f,'media_refusal');self.assertEqual((result['inherited'],result['selected']),(144,114))
            admission=a.read(result['admission_path']);row=next(r for r in admission['pending_wave_outcomes'] if r['layout_seed']==720120)
            self.assertEqual((row['status'],row['reason']),('refused','image_psnr'))
            proof=a.read(row['proof_path']);self.assertEqual(legacy.require_completed_checks(proof),'media')
            self.assertEqual(proof['media_stage']['returncode'],2);self.assertEqual(proof['export_stages'],{})

    def test_actual_coordinator_continues_wave_one_without_rescheduling_completed_groups(self):
        with fixture() as f,ExitStack() as stack:
            old_hashes=a.manifest(f.parent);jobs=f.state['next_wave']
            def outcome(j):return dict(split=j['split'],layout_seed=j['layout_seed'],requested_layout_seed=j['requested_layout_seed'],status='selected',reason='accepted',proof_path=str(f.root/(str(j['layout_seed'])+'.json')),proof_sha256='a'*64)
            inherited=[outcome(j) for j in jobs[:4]]
            recipe=dict(f.recipe,runner_cut='b'*40,production_root=str(f.parent.parent/'new'),runtime_root=str(f.root/'runtime/new'),prior_production_root=str(f.parent),inherited_waves=[f.wave],inherited_outcomes=inherited)
            release=f.root/'release.json';atomic_json(release,dict(passed=True));spawned=[]
            by_seed={j['layout_seed']:j for j in jobs};ledger=NS(known={},scan=lambda:None,add=lambda *_:None,stop=lambda:dict(remaining=[]))
            class Watch:
                def __init__(self,where,callback,report):self.report=report;self.closed=False
                def start(self):atomic_json(self.report,dict(healthy=True))
                def require(self):pass
                def close(self):self.closed=True
            def spawn(argv,**kwargs):
                slot=int(argv[argv.index('--worker')+1]);wave=int(argv[argv.index('--wave')+1]);self.assertEqual(wave,1)
                directory=Path(recipe['production_root'])/f'wave-{wave:02d}/slot-{slot}';directory.mkdir(parents=True)
                for job in jobs[4:]:
                    if job['instance']==slot:
                        spawned.append(job['layout_seed']);atomic_json(directory/f'outcome-{job["layout_seed"]}.json',outcome(job))
                atomic_json(directory/'worker-finalization.json',dict(passed=True,cut='a'*40,instance=slot,wave=wave,sessions=0))
                return NS(pid=999999,poll=lambda:0)
            def backup(value):atomic_json(Path(value['runtime_root'])/'registry-backup.json',dict(passed=True))
            ticks=iter(range(0,10000000,61))
            for obj,name,value in [(p,'qualified_recipe',lambda _:None),(p,'confined',lambda x:Path(x)),(p,'gpu_uuid',lambda _:p.ASSIGNED_UUID),
                    (p,'OwnedProcesses',lambda *a:ledger),(p,'DiskWatch',Watch),(p,'verify_outcome',lambda row,recipe:dict(job=by_seed[row['layout_seed']])),
                    (p,'registries',lambda *a:dict(train=dict(sealed=True),heldout=dict(sealed=True))),(p,'backup_registries',backup),
                    (admission_api,'reverify',lambda _:dict(passed=True))]:stack.enter_context(patch.object(obj,name,value))
            stack.enter_context(patch.object(p.ctypes,'CDLL',return_value=NS(prctl=lambda *a:0)))
            stack.enter_context(patch.object(p.subprocess,'check_output',return_value=''));stack.enter_context(patch.object(p.subprocess,'Popen',spawn))
            stack.enter_context(patch.object(p.time,'monotonic',lambda:next(ticks)))
            p.coordinate(recipe,release)
            waves=a.read(Path(recipe['production_root'])/'wave-ledger.json');self.assertEqual(waves[0],f.wave);self.assertEqual(len(reduce_waves(waves)['selected']),140)
            self.assertEqual(set(spawned),{j['layout_seed'] for j in jobs[4:]});self.assertEqual(len(spawned),23)
            self.assertFalse(set(spawned)&{j['layout_seed'] for j in f.wave['jobs']})
            self.assertEqual(a.manifest(f.parent),old_hashes);self.assertTrue(a.read(Path(recipe['production_root'])/'PUBLICATION-COMPLETE.json')['passed'])

    def test_peer_stop_does_not_count_as_identity_or_reset_or_extend_quality_streaks(self):
        job=initial_wave()[0]
        rows=[dict(job,reason='quality:planner_reject'),dict(job,reason='operator_acknowledged_peer_stop'),dict(job,reason='quality:planner_reject')]
        guards(rows,[]);self.assertEqual(identity_counts(rows)['exact_identity_refusals'],0)
        with self.assertRaisesRegex(ValueError,'three consecutive'):guards(rows+[dict(job,reason='quality:planner_reject')],[])

    def test_export_io_and_pre_stop_peer_fault_do_not_publish_admission(self):
        for kind in ('export_tool','peer_before_stop'):
            with self.subTest(kind=kind),fixture() as f:
                with self.assertRaises((ValueError,RuntimeError,OSError)):self.prepare_case(f,kind)
                self.assertFalse((f.root/'setup/continuation-admission-combined-v1/admission.json').exists())


if __name__=='__main__':unittest.main()
