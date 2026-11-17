"""CPU checks for per-batch seen-home provenance and qualification binding."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from native_environment import QUALIFICATION_FILES
from native_bank import SEEN15_SEEDS,NAMES,load_bank,reference_binding,select_policy_rows
from matched_reset_client import require_qualification

class Seen15Tests(unittest.TestCase):
    def bank(self,root):
        files={};references={};audit=root/'audit.json';audit.write_text('{}')
        for i,seed in enumerate(SEEN15_SEEDS):
            batch='b00' if i<6 else 'b01' if i<12 else 'b02';normal=root/batch;normal.mkdir(exist_ok=True)
            ref=normal/'run.json'
            if not ref.exists():ref.write_text(json.dumps(dict(batch=batch)))
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            references[batch]=dict(path=str(ref),sha256=sha(ref))
            attempt=normal/str(seed);attempt.mkdir();receipt=attempt/'receipt.json';frames=attempt/'frames.jsonl';frames.write_text('{"sim_ns":33333333}\n')
            color=('red','green','blue')[i%3];task=f'Pick a {color} lego and place it in {color} cup'
            bodies={n:dict(position=[0,0,0],orientation=[1,0,0,0]) for n in NAMES}
            prepared=dict(state=[.1]*6,drive_targets=[.2]*6,robot_base_world_pose={},bodies=bodies)
            receipt.write_text(json.dumps(dict(accepted=True,params=dict(seed=seed,color=color,task=task,recovery='none'),episode_start_physical=prepared)))
            state=dict(seed=seed,color=color,task=task,state=prepared['state'],recorded_drive_targets=prepared['drive_targets'],robot_base_world_pose={},source_receipt=str(receipt),source_receipt_sha256=sha(receipt),source_frames=str(frames),source_frames_sha256=sha(frames),source_batch=batch,reference_run=str(ref),reference_run_sha256=sha(ref),global_episode_index=i)
            for prefix,value in (('state',state),('seed',bodies)):
                p=root/f'{prefix}-{seed}.json';p.write_text(json.dumps(value));files[p.name]=sha(p)
        manifest=dict(kind='next120-seen-training-home',mode='home',complete=True,format_version=1,seeds=list(SEEN15_SEEDS),global_episode_indices=list(range(15)),colour_counts=dict(red=5,green=5,blue=5),source_audit_path=str(audit),source_audit_sha256=sha(audit),files_sha256=files,reference_runs=references)
        p=root/'bank-manifest.json';p.write_text(json.dumps(manifest));return sha(p)
    def test_per_batch_sources_and_full_cohort(self):
        with tempfile.TemporaryDirectory() as d:
            rows,suite=load_bank(Path(d),self.bank(Path(d)))
            self.assertEqual(suite,'seen-training-home-15');self.assertEqual(len(rows),15)
            self.assertEqual(select_policy_rows(rows,suite),rows)
            binding=reference_binding(rows,suite)['reference_sha256_by_seed']
            self.assertEqual(list(binding),list(map(str,SEEN15_SEEDS)));self.assertEqual(len(set(binding.values())),3)
            for offset,count in ((0,5),(1,15),(0,14)):
                with self.assertRaises(ValueError):select_policy_rows(rows,suite,offset,count)
            with self.assertRaises(ValueError):reference_binding(rows[:-1],suite)
            with self.assertRaises(ValueError):reference_binding(rows,'seen-training-home')
    def test_resealed_wrong_reference_or_recorded_home_refused(self):
        for field,value in (('reference_run_sha256','0'*64),('global_episode_index',14),('state',[.3]*6)):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                root=Path(d);self.bank(root);p=root/'state-530000.json';v=json.loads(p.read_text());v[field]=value;p.write_text(json.dumps(v))
                manifest=root/'bank-manifest.json';m=json.loads(manifest.read_text());m['files_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest();manifest.write_text(json.dumps(m))
                with self.assertRaises(ValueError):load_bank(root,hashlib.sha256(manifest.read_bytes()).hexdigest())
    def test_qualification_requires_complete_exact_per_seed_reference_map(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'source';src.mkdir();out=root/'qual';(out/'qualification').mkdir(parents=True)
            names=QUALIFICATION_FILES;shas={}
            for name in names:(src/name).write_text(name);shas[name]=hashlib.sha256((src/name).read_bytes()).hexdigest()
            host=dict(root=d);collector=root/'collector';collector.mkdir()
            ledger=dict(status='matched-reset-qualified',cleanup_exit_code=0,host_profile=host,source_sha256=shas)
            (out/'window-ledger.json').write_text(json.dumps(ledger));mapping={str(seed):str(i//6)*64 for i,seed in enumerate(SEEN15_SEEDS)}
            summary=dict(status='PASS',suite='seen-training-home-15',seeds=list(SEEN15_SEEDS),reference_run_sha256=None,reference_run_sha256_by_seed=mapping,bootstrap_control=None,episodes=15,commands=150,bank_sha256='bank15',reset_collector_root=str(collector))
            (out/'qualification/summary.json').write_text(json.dumps(summary))
            def check(**kwargs):return require_qualification(out,host,collector,'bank15',src,suite='seen-training-home-15',**kwargs)
            check(reference_sha256_by_seed=mapping)
            with self.assertRaises(AssertionError):check(reference_sha256='0'*64)
            bad=dict(mapping);bad['530500']='0'*64
            with self.assertRaises(AssertionError):check(reference_sha256_by_seed=bad)
            bad=dict(mapping);bad.pop('531002')
            with self.assertRaises(AssertionError):check(reference_sha256_by_seed=bad)
if __name__=='__main__':unittest.main()
