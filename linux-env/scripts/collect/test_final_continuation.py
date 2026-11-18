"""Final 172/136 carry and source-bound slot retirement through real entrances."""
import copy,hashlib,json,os,shutil,unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import completed_continuation as c
import startup_retries as retries
import triplet_artifacts as a
import triplet_continuation as legacy
import triplet_production as p
from storage import atomic_json
from triplet_production_plan import reduce_waves,identity_counts
from test_scene_init_continuation import latest_fixture
import test_scene_init_continuation as scene_tests

@contextmanager
def retired_fixture():
    with latest_fixture() as f:
        prepared,_,transports=scene_tests.SceneContinuationTests().prepare_case(f,return_transports=True)
        f.stack.enter_context(patch('triplet_media.inspect_run',lambda run:transports[int(run.name.split('-')[1])]['decode'](run)))
        old_history=c.historical_parent
        recipe=legacy.load_plan(prepared['admission_path'],prepared['admission_sha256'],'b'*40)
        parent=f.parent.parent/'continuation-admission-68334cc-v1';parent.mkdir()
        recipe.update(runner_cut=c.RETIRED_RUNNER,production_root=str(parent),
            runtime_root=str(f.root/'runtime/counterfactual-production-v1/continuations/continuation-admission-68334cc-v1'),
            slots=[dict(instance=i,domain=90+i) for i in (3,4,5,6)])
        atomic_json(parent/'plan.json',recipe)
        pending=copy.deepcopy(recipe['inherited_outcomes'])
        statuses={720148:('selected','accepted'),730023:('selected','accepted')}
        real_classify=p.classify
        f.stack.enter_context(patch.object(p,'classify',lambda run,job,*args:
            (*statuses[job['layout_seed']],None) if job['layout_seed'] in statuses else real_classify(run,job,*args)))
        for seed,split in ((720148,'train'),(730023,'heldout')):
            job=next(j for j in f.state['next_wave'] if j['layout_seed']==seed)
            run=f.root/f'datasets/counterfactual-targets/{split}/layout-{seed}';run.mkdir()
            atomic_json(run/'run.json',dict(git_commit='a'*40,source_sha256={'fixture_source.py':a.sha(f.root/'fixture_source.py')}))
            files=a.manifest(run);dest=f.root/'dual/datasets/counterfactual-targets'/split/run.name;shutil.copytree(run,dest)
            stage=dict(name='collect-'+str(seed),pid=os.getpid(),started_wall_ns=1,returncode=0)
            proof=dict(cut='a'*40,job=job,stage=stage,raw_root=str(run),files_sha256=files,
                media=None,export_stages={},backup=dict(passed=True,destination=str(dest),files_sha256=files))
            path=parent/'proofs'/split/(str(seed)+'.json');path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,proof)
            row={k:job[k] for k in ('split','layout_seed','requested_layout_seed')}
            row.update(status='selected',reason='accepted',proof_path=str(path),proof_sha256=a.sha(path));pending.append(row)
            folder=parent/f'wave-02/slot-{job["instance"]}';folder.mkdir(parents=True,exist_ok=True)
            stages=a.read(folder/'stages.json') if (folder/'stages.json').exists() else []
            atomic_json(folder/'stages.json',stages+[stage])
            for prefix in ('outcome-','admitted-'):atomic_json(folder/(prefix+str(seed)+'.json'),row)
        atomic_json(parent/'wave-02.json',dict(jobs=f.state['next_wave'],outcomes=recipe['inherited_outcomes']))
        rows=[r for w in f.waves for r in w['outcomes']]+pending
        atomic_json(parent/'progress.json',dict(wave=2,prior_selected=132,outcomes=pending,**identity_counts(rows)))
        budget=recipe['startup_retry_budget'];admission=a.read(prepared['admission_path'])
        record=admission['pending_initialization_retry'];event=record['event']['event']
        error=retries.StartupFailure(event['kind'],event['error'],event['stage'],Path(record['event']['event_path']).parent/'up.log')
        error.evidence=record['evidence']
        down=Path(record['event']['event_path']).parent/'owned-down.json'
        for _ in range(2):
            reservation=retries.before_start(budget,5,event['job'])
            retries.ready(budget,5,event['job'],reservation)
            retries.failed(budget,5,event['job'],error,down,None)
        saved=retries.summary(budget)
        assert saved['counts']==dict(total=3,slots={'3':0,'4':0,'5':3,'6':0})
        owner=c.parent_owner(parent);owner.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(owner,dict(state='stopped',nonce='retired683',cleanup=dict(remaining=[])))
        atomic_json(owner.parent/'post-stop-owned-down.json',dict(passed=True,remaining_owned=[],recorded_births=2,
            ports_listening={str(n):False for n in range(8783,8791)},managed_status=dict(supervision_ok=True),
            managed_stop=dict(supervision_ok=True),sources=[]))
        atomic_json(parent/'STOP.json',dict(slot=5,error='RuntimeError("Startup retry cap reached: retained fixture")',
            wall_ns=1794988666726773265,supervision=dict(nonce='retired683',receipt_path=str(owner))))
        atomic_json(parent/'production-finalization.json',dict(passed=False,retirement_errors=[],cleanup=dict(remaining=[]),
            backup_children_cleanup=dict(remaining=[]),terminal_backups=dict(errors=[]),startup_retries=saved))
        f.stack.enter_context(patch.object(c,'historical_parent',lambda root:
            copy.deepcopy(recipe) if Path(root)==parent else old_history(root)))
        yield NS(**{**f.__dict__,'parent':parent,'recipe':recipe,'pending':pending,'saved':saved})

def prepare(f):
    table=f.root/'table.log';table.write_text('fixture table')
    with patch.object(p,'load_bootstrap_capsule',lambda *a:dict(pin=dict(source=dict(path=str(table))))):
        result=c.prepare(f.parent,f.root,'d'*40,f.root/'setup/final-admission',c.checkpoint(f.parent)[1])
    return result,legacy.load_plan(result['admission_path'],result['admission_sha256'],'d'*40)

def rebind_admission(path,value):
    atomic_json(path,value)
    return a.sha(path)

class FinalContinuationTests(unittest.TestCase):
    def test_real_prepare_carries_172_136_and_routes_pending_without_reset(self):
        with retired_fixture() as f:
            before=c.checkpoint(f.parent)
            result,recipe=prepare(f)
            self.assertEqual((result['inherited'],result['selected']),(172,136))
            self.assertEqual(c.checkpoint(f.parent),before)
            admission=a.read(result['admission_path'])
            self.assertEqual((identity_counts(admission['inherited_outcomes'])['exact_identity_refusals'],len(admission['inherited_outcomes'])),(1,172))
            now=retries.summary(recipe['startup_retry_budget'])
            self.assertEqual(now['counts'],f.saved['counts']);self.assertEqual(now['events'],f.saved['events'])
            self.assertEqual(now['pending'],{'5':['train',720144]})
            jobs=reduce_waves(recipe['inherited_waves'],slot5_to4=True)['next_wave']
            pending=[j for j in jobs if j['layout_seed'] not in {r['layout_seed'] for r in admission['pending_wave_outcomes']}]
            self.assertEqual({j['layout_seed'] for j in pending},{720144,720146,720150})
            self.assertTrue(all(j['instance']==4 for j in pending))
            self.assertEqual({s['instance'] for s in recipe['slots']},{3,4,6})
            # A fresh physical slot starts without resetting the exhausted slot.
            job=next(j for j in pending if j['layout_seed']==720144)
            self.assertIsNone(retries.before_start(recipe['startup_retry_budget'],4,job))
            self.assertEqual(retries.summary(recipe['startup_retry_budget'])['counts'],f.saved['counts'])
            self.assertEqual(c.reverify(recipe)['outcomes'],172)
            failure=retries.failed_exit(dict(name='up-0',pid=os.getpid(),started_wall_ns=1,returncode=1),f.root/'up.log')
            down=f.root/'down4.json';atomic_json(down,dict(passed=True,errors=[],cleanup=dict(remaining=[]),ports_listening={'8785':False,'8786':False}))
            retries.failed(recipe['startup_retry_budget'],4,job,failure,down,None)
            reservation=retries.before_start(recipe['startup_retry_budget'],4,job)
            retries.ready(recipe['startup_retry_budget'],4,job,reservation)
            used=retries.summary(recipe['startup_retry_budget'])
            self.assertEqual(used['counts'],dict(total=4,slots={'3':0,'4':1,'5':3,'6':0}))
            self.assertEqual(c.reverify(recipe)['outcomes'],172)
            # Rewriting the live counter files cannot erase the bound prefix.
            root=Path(recipe['startup_retry_budget']['root'])
            atomic_json(root/'state.json',dict(events=[]));(root/'journal.jsonl').write_text('')
            with self.assertRaises(ValueError):retries.before_start(recipe['startup_retry_budget'],4,job)

    def test_real_admission_refuses_routing_and_cap_reset(self):
        for mode in ('slot5','prefix','ledger','counts'):
            with self.subTest(mode=mode),retired_fixture() as f:
                result,recipe=prepare(f);path=Path(result['admission_path']);v=a.read(path)
                if mode=='slot5':v['slot5_to4']=False
                elif mode=='prefix':v['startup_retry_budget']['prefix']=[]
                elif mode=='ledger':v['startup_retry_budget']['root']=str(f.root/'new-ledger')
                else:
                    final=f.parent/'production-finalization.json';saved=a.read(final)
                    saved['startup_retries']['counts']['total']=0;atomic_json(final,saved)
                    v['parent_checkpoint'],v['parent_checkpoint_sha256']=c.checkpoint(f.parent)
                digest=rebind_admission(path,v)
                with self.assertRaises((ValueError,RuntimeError,OSError,KeyError)):
                    parsed=legacy.load_plan(path,digest,'d'*40);c.reverify(parsed)

    def test_real_parent_refuses_tampered_683_proof_and_backup(self):
        for mode in ('proof','backup','membership'):
            with self.subTest(mode=mode),retired_fixture() as f:
                row=next(r for r in f.pending if r['layout_seed']==720148);proof=a.read(row['proof_path'])
                if mode=='proof':
                    proof['files_sha256']['changed']='0'*64;atomic_json(row['proof_path'],proof)
                elif mode=='backup':(Path(proof['backup']['destination'])/'run.json').write_text('{}')
                else:
                    progress=a.read(f.parent/'progress.json');progress['outcomes'].pop();atomic_json(f.parent/'progress.json',progress)
                with self.assertRaises((ValueError,RuntimeError,OSError,KeyError)):
                    c.parent_state(f.parent,c.checkpoint(f.parent)[1])

    def test_worker_entrance_refuses_slot5_before_runtime_creation(self):
        with self.assertRaisesRegex(ValueError,'Slot 5 is retired'):
            p.Worker(dict(slot5_to4=True),5,2)

    def test_reducer_refuses_old_slot_and_preserves_next_reserve_allocation(self):
        with retired_fixture() as f:
            logical=reduce_waves(f.waves)['next_wave']
            jobs=reduce_waves(f.waves,slot5_to4=True)['next_wave']
            rows=copy.deepcopy(f.pending)
            for job in jobs:
                if job['layout_seed'] in {r['layout_seed'] for r in rows}:continue
                rows.append(dict(split=job['split'],requested_layout_seed=job['requested_layout_seed'],
                    layout_seed=job['layout_seed'],status='selected',reason='accepted',proof_path='fixture',proof_sha256='1'*64))
            routed=dict(jobs=jobs,outcomes=rows)
            state=reduce_waves(f.waves+[routed],slot5_to4=True)
            self.assertEqual([j['layout_seed'] for j in state['next_wave']],[720151])
            self.assertNotIn(5,{j['instance'] for j in state['next_wave']})
            with self.assertRaisesRegex(ValueError,'deterministic seed/slot'):
                reduce_waves(f.waves+[dict(jobs=logical,outcomes=rows)],slot5_to4=True)

if __name__=='__main__':unittest.main()
