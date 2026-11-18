"""167/132 completed-wave continuation through the existing preparation entrance."""
import copy,json,os,shutil,sys,tempfile,time,unittest
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
from types import SimpleNamespace as NS
import completed_continuation as c
import triplet_continuation as legacy
import triplet_production as p
import triplet_artifacts as a
import startup_retries as retries
from storage import atomic_json
from collection_status import begin,terminal,finish
from target_triplets import spec
from triplet_production_plan import reduce_waves,identity_counts
from test_completed_continuation import fixture
import test_triplet_production as runner_tests
from test_scene_initialization import retained,FIXTURE
import scene_initialization as init

WAVES=Path(__file__).with_name('test_fixtures')/'scene-init-parent-waves.json'
HISTORICAL_PARENT=c.historical_parent


@contextmanager
def latest_fixture():
    with fixture() as f:
        root=f.root;parent=f.parent.parent/'continuation-admission-0a80e4e-v1';parent.mkdir()
        ledger=a.read(WAVES);statuses={};outcomes=[]
        for i,wave in enumerate(ledger):
            by_seed={r['layout_seed']:r for r in wave['outcomes']}
            for job in wave['jobs']:
                row=by_seed[job['layout_seed']]
                statuses[job['layout_seed']]=(row['status'],row['reason'])
                run=root/f'datasets/counterfactual-targets/{job["split"]}/layout-{job["layout_seed"]}';run.mkdir(parents=True,exist_ok=True)
                atomic_json(run/'run.json',dict(git_commit='a'*40,source_sha256={'fixture_source.py':a.sha(root/'fixture_source.py')}))
                files=a.manifest(run);dest=root/'dual/datasets/counterfactual-targets'/job['split']/run.name;shutil.copytree(run,dest,dirs_exist_ok=True)
                proof=dict(cut='a'*40,job=job,stage=dict(name='collect-'+str(job['layout_seed'])),raw_root=str(run),files_sha256=files,media=None,export_stages={},backup=dict(passed=True,destination=str(dest),files_sha256=files))
                path=parent/'proofs'/job['split']/(str(job['layout_seed'])+'.json');path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,proof)
                row.update(proof_path=str(path),proof_sha256=a.sha(path));outcomes.append(row)
                folder=parent/f'wave-{i:02d}/slot-{job["instance"]}';folder.mkdir(parents=True,exist_ok=True)
                for prefix in ('outcome-','admitted-'):atomic_json(folder/(prefix+str(job['layout_seed'])+'.json'),row)
            atomic_json(parent/f'wave-{i:02d}.json',wave)
        state=reduce_waves(ledger);assert len(state['selected'])==132
        atomic_json(parent/'wave-ledger.json',ledger);atomic_json(parent/'wave-02.json',dict(jobs=state['next_wave'],outcomes=[]));atomic_json(parent/'progress.json',dict(wave=2,prior_selected=132,outcomes=[]))
        old=root/'setup/current-parent-admission';old.mkdir();atomic_json(old/'admission.json',dict(kind=c.KIND,origin_checkpoint_sha256='c'*64,origin_root=str(f.parent.parent.parent),origin_bindings={},startup_retry_budget=f.recipe['startup_retry_budget']))
        recipe=dict(f.recipe,runner_cut='0a80e4e867a58f1fec48ea3a9b85d179cc70343c',production_root=str(parent),runtime_root=str(root/'runtime/current'),inherited_waves=ledger[:1],inherited_outcomes=ledger[1]['outcomes'][:4],continuation_admission_path=str(old/'admission.json'),continuation_admission_sha256=a.sha(old/'admission.json'))
        atomic_json(parent/'plan.json',recipe)
        # The production coordinator stores inherited waves in its plan/ledger;
        # it does not copy their wave-NN.json headers into the child root.
        (parent/'wave-00.json').unlink()
        owner=c.parent_owner(parent);owner.parent.mkdir(parents=True,exist_ok=True);atomic_json(owner,dict(state='stopped',nonce='current-parent',supervisor=dict(pid=99999997,start_ticks='1'),workload=dict(pid=99999998,start_ticks='2'),cleanup=dict(remaining=[])))
        atomic_json(owner.parent/'post-stop-owned-down.json',dict(passed=True,remaining_owned=[],recorded_births=2,ports_listening={str(n):False for n in range(8783,8791)},managed_status=dict(supervision_ok=True),managed_stop=dict(supervision_ok=True),sources=[]))
        stop=dict(slot=5,error="FileNotFoundError(2, 'No such file or directory')",wall_ns=1794978990139379403,supervision=dict(nonce='current-parent',receipt_path=str(owner)))
        atomic_json(parent/'STOP.json',stop);atomic_json(parent/'production-finalization.json',dict(passed=False,retirement_errors=[],cleanup=dict(remaining=[]),backup_children_cleanup=dict(remaining=[]),terminal_backups=dict(errors=[]),startup_retries=retries.summary(recipe['startup_retry_budget'])))
        real=a.classify
        f.stack.enter_context(patch.object(p,'classify',lambda run,job,*args:(*statuses[job['layout_seed']],None) if job['layout_seed'] in statuses else real(run,job,*args)))
        f.stack.enter_context(patch.object(c,'historical_parent',lambda _:copy.deepcopy(recipe)))
        yield NS(**{**f.__dict__,'parent':parent,'recipe':recipe,'waves':ledger,'state':state,'stop':stop})


class SceneContinuationTests(unittest.TestCase):
    def test_plan_and_ledger_replay_ignore_process_tmpdir(self):
        with latest_fixture() as f,patch.dict(os.environ,{'TMPDIR':'/tmp'}),patch.object(tempfile,'tempdir','/tmp'):
            self.assertEqual(tempfile.gettempdir(),'/tmp')
            result,_=self.prepare_case(f)
            recipe=legacy.load_plan(result['admission_path'],result['admission_sha256'],'b'*40)
            ledger=retries.summary(recipe['startup_retry_budget'])
            self.assertEqual((result['inherited'],result['selected']),(170,134))
            self.assertEqual(ledger['counts']['total'],1);self.assertEqual(ledger['pending'],{'5':['train',720144]})
            admission=a.read(result['admission_path']);job=admission['pending_initialization_retry']['job']
            retries.before_start(recipe['startup_retry_budget'],5,job)
            self.assertEqual(retries.summary(recipe['startup_retry_budget'])['counts']['total'],2)
            self.assertFalse(list((f.root/'runtime').glob('scene-init-replay-*')))

    def test_historical_outer_cannot_be_reinterpreted_as_its_own_new_parent(self):
        with latest_fixture() as f:
            recipe=a.read(f.parent/'plan.json');recipe['collector_repo']=str(f.root/'repo/counterfactual-production');atomic_json(f.parent/'plan.json',recipe)
            path=Path(recipe['continuation_admission_path']);admission=a.read(path);admission['prior_production_root']=str(f.parent);atomic_json(path,admission)
            with patch.object(c.subprocess,'check_output',return_value=b'frozen source'),self.assertRaisesRegex(ValueError,'approved 5eaf parent'):
                HISTORICAL_PARENT(f.parent)

    def test_167_132_parent_reverifies_proof_and_backup_and_refuses_140_113_rewind(self):
        for kind in ('healthy','proof','backup','rewind'):
            with self.subTest(kind=kind),latest_fixture() as f:
                digest=c.checkpoint(f.parent)[1]
                if kind=='proof':
                    path=Path(f.waves[1]['outcomes'][1]['proof_path']);value=a.read(path);value['files_sha256']['edited']='0'*64;atomic_json(path,value)
                    digest=c.checkpoint(f.parent)[1]
                elif kind=='backup':
                    proof=a.read(f.waves[1]['outcomes'][1]['proof_path']);(Path(proof['backup']['destination'])/'run.json').write_text('{}')
                elif kind=='rewind':atomic_json(f.parent/'wave-ledger.json',f.waves[:1]);digest=c.checkpoint(f.parent)[1]
                if kind=='healthy':self.assertEqual(len(c.parent_state(f.parent,digest)['state']['selected']),132)
                else:
                    with self.assertRaises((ValueError,KeyError)):c.parent_state(f.parent,digest)

    def prepare_case(self,f,bad=None):
        transports={};extended=f.waves[0]['jobs']+f.waves[1]['jobs']+f.state['next_wave'];d=c.profile(f.parent)
        with patch.object(runner_tests,'initial_wave',lambda:extended):
            for seed in d['exports']:
                transports[seed]=runner_tests.ProductionRunnerTests().fake_attempt(f.root,production_root=f.root/('fake-'+str(seed)),layout_seed=seed,return_transports=True)
                proof=a.read(f.root/f'fake-{seed}/proofs/train/{seed}.json');job=proof['job'];folder=f.parent/f'wave-02/slot-{job["instance"]}';folder.mkdir(parents=True,exist_ok=True);atomic_json(folder/'stages.json',[proof['stage']])
                if seed==720147:
                    path=f.parent/'proofs/train/720147.json';atomic_json(path,proof)
                    row=dict(split='train',requested_layout_seed=job['requested_layout_seed'],layout_seed=seed,status='selected',reason='accepted',proof_path=str(path),proof_sha256=a.sha(path))
                    atomic_json(folder/'outcome-720147.json',row)
                    shutil.copytree(proof['raw_root'],proof['backup']['destination'],dirs_exist_ok=True)
        peer_job=next(j for j in f.state['next_wave'] if j['layout_seed']==720145);peer=f.root/'datasets/counterfactual-targets/train/layout-720145';peer.mkdir(parents=True)
        cert=begin(peer);cert['started_wall_ns']=f.stop['wall_ns']-10**9
        atomic_json(peer/'run.json',dict(git_commit='a'*40,source_sha256={'fixture_source.py':a.sha(f.root/'fixture_source.py')},normal_only=True,instance=6,counterfactual_triplet=spec(720145,'train')))
        stage=dict(name='collect-720145',pid=os.getpid(),started_wall_ns=cert['started_wall_ns']-1,returncode=None)
        folder=f.parent/'wave-02/slot-6';folder.mkdir(parents=True,exist_ok=True);atomic_json(folder/'stages.json',[stage])
        terminal(peer,dict(error='MoveIt state validity query did not complete',source='MoveIt validity query'))
        marker=a.read(peer/'terminal-fault.json');marker['events'][0]['wall_ns']=1794978991459962612+(1 if bad=='peer_time' else 0);atomic_json(peer/'terminal-fault.json',marker);finish(peer,cert,[])
        # Translate only the frozen checkout path in the retained traceback;
        # all timeout/lifecycle/native observation bytes remain the live data.
        temp=f.root/'retained';temp.mkdir();w,job,event=retained(temp)
        run=temp/f'datasets/counterfactual-targets/train/layout-720144';target=f.root/'datasets/counterfactual-targets/train/layout-720144';run.rename(target)
        folder=f.parent/'wave-02/slot-5';folder.mkdir(parents=True,exist_ok=True)
        for file in w.root.rglob('*'):
            if file.is_file():dest=folder/file.relative_to(w.root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(file.read_bytes())
        atomic_json(folder/'stages.json',w.stages+[dict(name='down-0',returncode=0)])
        (folder/'collect-720144.log').write_text(init.traceback_text(f.root))
        atomic_json(folder/'owned-down-0.json',dict(passed=True,errors=[],cleanup=dict(remaining=[]),ports_listening={'8787':False,'8788':False}))
        if bad=='writer':(target/'attempt-0000').mkdir()
        table=f.root/'table.log';table.write_text('fixture table')
        def stage(worker,name,*args,**kwargs):
            if bad=='export_tool' and name=='report-720149':return dict(name=name,pid=os.getpid(),started_wall_ns=time.time_ns(),returncode=1)
            seed=int(name.rsplit('-',1)[1]);value=transports[seed]['stage'](worker,name,*args,**kwargs);worker.stages.append(value);return value
        def backup(source,files,destination,log):
            self.assertEqual(a.manifest(source),files);shutil.copytree(source,destination,dirs_exist_ok=True)
            return dict(passed=True,files_sha256=files,destination=str(destination))
        output=f.root/'setup/continuation-admission-new-v1'
        with patch('triplet_media.inspect_run',lambda run:transports[int(run.name.split('-')[1])]['decode'](run)),patch.object(p.Worker,'stage',stage),patch.object(p,'backup',backup),patch.object(p,'profile',lambda:dict(export_python=sys.executable)),patch.object(p,'load_bootstrap_capsule',lambda *a:dict(pin=dict(source=dict(path=str(table))))):
            return c.prepare(f.parent,f.root,'b'*40,output,c.checkpoint(f.parent)[1]),output

    def test_real_prepare_carries_167_132_qualifies_peers_and_records_one_pending_start(self):
        with latest_fixture() as f:
            result,output=self.prepare_case(f);self.assertEqual((result['inherited'],result['selected']),(170,134))
            admission=a.read(result['admission_path']);self.assertEqual(len(admission['inherited_waves']),2)
            self.assertEqual({r['layout_seed'] for r in admission['pending_wave_outcomes']},{720145,720147,720149})
            counts=identity_counts(admission['inherited_outcomes']);self.assertEqual((counts['exact_identity_refusals'],counts['attempted_groups']),(1,170))
            ledger=retries.summary(admission['startup_retry_budget']);self.assertEqual(ledger['counts']['total'],1);self.assertEqual(ledger['pending'],{'5':['train',720144]})
            self.assertNotIn(720144,{r['layout_seed'] for r in admission['inherited_outcomes']});self.assertTrue((output/'scene-init-720144/summary.json').exists())
            recipe=legacy.load_plan(result['admission_path'],result['admission_sha256'],'b'*40)
            job=admission['pending_initialization_retry']['job'];retries.before_start(recipe['startup_retry_budget'],5,job)
            self.assertEqual(retries.summary(recipe['startup_retry_budget'])['counts']['total'],2)

    def test_real_prepare_never_converts_tool_writer_or_peer_time_errors_to_dispositions(self):
        for bad in ('export_tool','writer','peer_time'):
            with self.subTest(bad=bad),latest_fixture() as f:
                with self.assertRaises((ValueError,RuntimeError,OSError)):self.prepare_case(f,bad)
                self.assertFalse((f.root/'setup/continuation-admission-new-v1/admission.json').exists())


if __name__=='__main__':unittest.main()
