"""Native preparation -> outer classifier -> SHA-backed layout refusal."""
import copy,hashlib,json,os,shutil,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import bootstrap_refusal as b
import triplet_artifacts as a
import triplet_production as p
import triplet_boundary as boundary
from triplet_production_plan import identity_counts,guards
from target_triplets import spec,destination
from collection_status import begin,terminal,finish
from storage import atomic_json
from contract import timing_audit
from test_target_triplets import Harness

LIVE=Path(__file__).with_name('test_fixtures')/'bootstrap-720123.json'


class BootstrapRefusalTests(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);live=json.loads(LIVE.read_text());job=live['job'];run=root/'datasets/counterfactual-targets/train/layout-720123';run.mkdir(parents=True)
        started=time.time_ns();cert=begin(run);h=Harness(run,spec(720123,'train'));saved={};prepare=h.collector._prepare_scene_attempt
        h.data['bodies']['cup_green']=copy.deepcopy(live['reference']['bodies']['cup_green'])
        def identified(params,baseline,receipt,*args):
            index=params['trial'];receipt['episode_id']=hashlib.sha256((str(run)+':'+str(index)+':720123').encode()).hexdigest();saved[index]=receipt
            return prepare(params,baseline,receipt,*args)
        h.collector._prepare_scene_attempt=identified
        def moving(data):
            if h.member_index==2:data['bodies']['cup_green']=copy.deepcopy(live['observed']['bodies']['cup_green'])
        h.after_bootstrap=moving
        try:h.complete(recorded_ticks=60)
        except ValueError as exc:
            self.assertIn('Triplet physical agreement failed',str(exc));error=repr(exc)
        else:self.fail('The actual Collector preparation admitted the unequal bootstrap')
        self.assertIsNone(h.collector.writer)
        for i in (0,1):
            f=run/f'attempt-{i:04d}/receipt.json';r=a.read(f);rows=[json.loads(line) for line in a.read(f.parent/'frames.jsonl',True).splitlines()]
            timing=timing_audit(rows,30);self.assertTrue(timing['passed'],timing)
            r.update(timing=timing,end_cancel=dict(held=True),retirement=dict(passed=True),capture_failures=[],camera_boundary=dict(passed=True),script_completed=True,score=dict(passed=True,contract_passed=True),disposition=dict(kind='accepted',reason='success'))
            atomic_json(f,r)
        failed=saved[2];failed.update(accepted=False,exception_type='ValueError',error=error,capture_failures=[],camera_boundary=dict(passed=True),end_cancel=dict(held=True),retirement=dict(passed=True),disposition=dict(kind='terminal',reason='unclassified_failure'))
        directory=run/'attempt-0002';directory.mkdir();atomic_json(directory/'receipt.json',failed)
        source=root/'fixture_source.py'
        if not source.exists():source.write_text('fixture transport')
        atomic_json(run/'run.json',dict(counterfactual_triplet=h.config,normal_only=True,instance=job['instance'],git_commit='a'*40,source_sha256={'fixture_source.py':a.sha(source)}))
        atomic_json(run/'summary.json',dict(producer_invocation=cert['invocation'],complete=False,episodes=2,triplet_rejected=True))
        terminal(run,dict(error=error,source='attempt disposition'),directory);finish(run,cert,[])
        return run,job,dict(name='collect-720123',pid=os.getpid(),started_wall_ns=started,returncode=1),h

    def test_retained_live_720123_replays_native_restore_clock_ack_and_failure_bits(self):
        live=json.loads(LIVE.read_text());b.restored(live['priming'],live['anchor']);b.restored(live['restore'],live['anchor'])
        r=live['restore'];boundary.bootstrap(r['snapshot'],{k:v for k,v in r.items() if k!='snapshot'},live['observed'])
        proof=b.failure_comparison(live['observed'],[live['reference']],live['error'])
        self.assertFalse(proof['passed']);self.assertTrue(proof['per_body']['cup_green']['position_passed'])
        self.assertTrue(proof['per_body']['cup_green']['orientation_passed']);self.assertEqual(len(proof['strict_differences']),6)
        changed=copy.deepcopy(live['observed']);changed['bodies']['cup_green']['velocity'][0]=0.
        with self.assertRaises(ValueError):b.failure_comparison(changed,[live['reference']],live['error'])

    def test_real_worker_qualify_rejects_whole_prefix_and_reverifies_backup(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run,job,stage,_=self.fixture(root);production=root/'production';recipe=dict(cut='a'*40,collector_repo=str(root),production_root=str(production),runtime_root=str(root/'runtime'))
            def backup(source,files,destination,log):
                self.assertEqual(a.manifest(source),files);shutil.copytree(source,destination)
                return dict(passed=True,files_sha256=files,destination=str(destination))
            with patch.object(p,'ROOT',root),patch.object(p,'DUAL',root/'dual'),patch.object(p,'capture_supervision',lambda:None),patch.object(p,'backup',backup),patch.object(a,'destination',lambda run,config:destination(run,config,base=root)):
                worker=p.Worker(recipe,job['instance'],1);row=worker.qualify(job,stage)
                self.assertEqual((row['status'],row['reason']),('refused','exact_identity'))
                proof=p.verify_outcome(row,recipe);self.assertEqual(proof['group']['member'],2)
                self.assertEqual(proof['group']['prefix_members'],2);self.assertTrue(proof['backup']['passed'])
                self.assertFalse((production/'REGISTRY-SHA256.json').exists())
                forged=copy.deepcopy(proof);forged['group']['boundary_sha256']='0'*64
                path=Path(row['proof_path']);atomic_json(path,forged);edited=dict(row,proof_sha256=a.sha(path))
                with self.assertRaisesRegex(ValueError,'provenance'):p.verify_outcome(edited,recipe)
            self.assertFalse((run/'attempt-0002/frames.jsonl').exists())
            self.assertEqual(len((run/'attempt-0000/frames.jsonl').read_text().splitlines()),60)

    def test_writer_pipeline_source_epoch_restore_and_cleanup_faults_stay_terminal(self):
        kinds=('writer','video','first_recorded','capture','extra_fault','missing_boundary','changed_error','bad_clock','bad_epoch','bad_ack','bad_restore','prefix_tick','prefix_native','prefix_camera','retirement','finalization')
        for kind in kinds:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                run,job,stage,_=self.fixture(Path(d));failed=run/'attempt-0002/receipt.json';r=a.read(failed);path=run/'triplet-boundaries/attempt-0002.json';doc=a.read(path)
                if kind=='writer':(failed.parent/'frames.jsonl').touch()
                elif kind=='video':(failed.parent/'top.mp4').touch()
                elif kind=='first_recorded':doc['phases']['first_recorded']={}
                elif kind=='capture':r['capture_failures']=[dict(error='sensor fault')]
                elif kind=='extra_fault':terminal(run,dict(error='Camera ambiguity',source='sensor'))
                elif kind=='missing_boundary':path.unlink()
                elif kind=='changed_error':r['error']=r['error'].replace('cup_green','cup_red')
                elif kind in ('bad_clock','bad_epoch'):
                    k='sim_ns' if kind=='bad_clock' else 'epoch';doc['phases']['bootstrap']['snapshot'][k]-=1_000_000_000 if k=='sim_ns' else 1
                elif kind=='bad_ack':doc['phases']['final_restore']['restore']['command_baseline']['drive_targets'][0]+=.1
                elif kind=='bad_restore':doc['phases']['final_restore']['restore']['arm_restored']=False
                elif kind=='prefix_tick':
                    f=run/'attempt-0001/frames.jsonl';rows=[json.loads(line) for line in f.read_text().splitlines()];rows[-1]['top_stamp_ns']+=1_000_000_000;f.write_text(''.join(json.dumps(x)+'\n' for x in rows))
                elif kind=='prefix_native':
                    f=run/'attempt-0001/first-recorded-native.json';value=a.read(f);value['episode_id']='foreign';atomic_json(f,value)
                elif kind=='prefix_camera':
                    f=run/'attempt-0001/receipt.json';value=a.read(f);value['camera_boundary']['passed']=False;atomic_json(f,value)
                elif kind=='retirement':r['retirement']['passed']=False
                elif kind=='finalization':r['finalization_errors']=['executor did not retire']
                if kind!='missing_boundary':atomic_json(path,doc)
                if kind in ('first_recorded','bad_clock','bad_epoch','bad_ack','bad_restore'):r['triplet_boundary_evidence']=copy.deepcopy(doc['phases'])
                atomic_json(failed,r)
                # Preserve a valid lifecycle seal after source corruption, so
                # the inner contract rather than a stale summary is exercised.
                with patch.object(a,'destination',lambda run,config:destination(run,config,base=Path(d))),self.assertRaises((ValueError,KeyError,RuntimeError,OSError)):
                    a.classify(run,job,stage)

    def test_progress_identity_counts_use_all_attempts_and_the_same_ten_percent_guard(self):
        outcomes=[dict(reason='accepted') for _ in range(9)]+[dict(reason='exact_identity')]
        counts=identity_counts(outcomes);self.assertEqual(counts['exact_identity_refusals'],1);self.assertEqual(counts['attempted_groups'],10)
        self.assertEqual(counts['exact_identity_fraction'],.1);guards(outcomes,[])
        with self.assertRaisesRegex(ValueError,'exceed 10%'):guards(outcomes+[dict(reason='exact_identity')],[])


if __name__=='__main__':unittest.main()
