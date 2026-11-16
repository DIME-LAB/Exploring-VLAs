"""CPU boundary checks for sealed source provenance and fixed policy chunks."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from native_bank import load_bank,select_policy_rows,NAMES,COLLECTOR_SHA,HELPER_SHA

class BankTests(unittest.TestCase):
    def bank(self,root):
        files={};meta={};parameters={}
        def write(name,value,inventory):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));inventory[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        reference=root/'reference.json';reference.write_text(json.dumps(dict(baseline={'red':[0,0,0]},baseline_cup_layout={'mode':'line'})))
        provenance=dict(reference_run=str(reference),reference_run_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),baseline={'red':[0,0,0]},baseline_cup_layout={'mode':'line'},scene_bank_source_sha256=HELPER_SHA,context=dict(git_commit='bf7336d93d5ba054c5edc66cea685725eb30eb40',mode='eval',source_sha256={'linux-env/scripts/collect/collect.py':COLLECTOR_SHA,'linux-env/scripts/collect/scene_bank.py':HELPER_SHA}))
        write('metadata/provenance.json',provenance,meta);write('metadata/finalization.json',dict(passed=True),meta)
        for seed in range(900000,900030):
            color=('red','green','blue')[seed%3];params=dict(seed=seed,color=color,split='eval',recovery='none',task=f'Pick a {color} lego and place it in {color} cup');parameters[str(seed)]=params
            bodies={n:dict(position=[0,0,0],orientation=[1,0,0,0]) for n in NAMES}
            physical=dict(state=[0]*6,drive_targets=[0]*6,robot_base_world_pose={},bodies=bodies,epoch=1,sim_ns=1_000_000_000)
            prepared=copy.deepcopy(physical);prepared.update(sim_ns=1_200_000_000,drive_targets=[.001]*6)
            write(f'seed-{seed}.json',bodies,files)
            write(f'metadata/seed-{seed}.json',dict(accepted=True,scene_bank_only=True,params=params,initial_physical=physical,episode_start_physical=prepared),meta)
        manifest=dict(format_version=2,purpose='held-out-collector-scenes',complete=True,producer='collect.Collector.run_scene_bank',seeds=list(range(900000,900030)),files_sha256=files,sampler_metadata_sha256=meta,parameters=parameters)
        p=root/'bank-manifest.json';p.write_text(json.dumps(manifest));return hashlib.sha256(p.read_bytes()).hexdigest()
    def reseal_metadata(self,root,name,update):
        p=root/name;value=json.loads(p.read_text());update(value);p.write_text(json.dumps(value))
        manifest=root/'bank-manifest.json';data=json.loads(manifest.read_text());data['sampler_metadata_sha256'][name]=hashlib.sha256(p.read_bytes()).hexdigest();manifest.write_text(json.dumps(data));return hashlib.sha256(manifest.read_bytes()).hexdigest()
    def test_all_thirty_pair_by_physical_layout_and_prepared_targets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sha=self.bank(root);rows,suite=load_bank(root,sha)
            self.assertEqual(suite,'heldout-bank-v2');self.assertEqual(len(rows),30);self.assertEqual([r['color'] for r in rows[:3]],['red','green','blue'])
            self.assertEqual(rows[0]['recorded_drive_targets'],[.001]*6)
            self.assertIsNone(rows[0]['first_frame_reference']);self.assertEqual(rows[0]['source_receipt'],str((root/'metadata/seed-900000.json').resolve()))
    def test_receipt_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sha=self.bank(root);p=root/'metadata/seed-900000.json';p.write_text(p.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'file SHA'):load_bank(root,sha)
    def test_manifest_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sha=self.bank(root);p=root/'bank-manifest.json';p.write_text(p.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'manifest SHA'):load_bank(root,sha)
    def test_wrong_reference_sha_and_baseline_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.bank(root)
            sha=self.reseal_metadata(root,'metadata/provenance.json',lambda p:p.update(reference_run_sha256='0'*64))
            with self.assertRaisesRegex(ValueError,'reference run SHA'):load_bank(root,sha)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.bank(root)
            sha=self.reseal_metadata(root,'metadata/provenance.json',lambda p:p.update(baseline={'red':[.16,0,0]}))
            with self.assertRaisesRegex(ValueError,'baseline differs'):load_bank(root,sha)
    def test_bad_verification_offset_or_sampler_code_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.bank(root)
            sha=self.reseal_metadata(root,'metadata/seed-900000.json',lambda p:p['episode_start_physical'].update(sim_ns=1_000_000_000))
            with self.assertRaisesRegex(ValueError,'verification-frame offset'):load_bank(root,sha)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.bank(root)
            sha=self.reseal_metadata(root,'metadata/provenance.json',lambda p:p['context']['source_sha256'].update({'linux-env/scripts/collect/collect.py':'0'*64}))
            with self.assertRaisesRegex(ValueError,'code provenance'):load_bank(root,sha)
    def test_fixed_balanced_chunks_and_complete_seen_suite(self):
        rows=list(range(30))
        self.assertEqual(select_policy_rows(rows,'heldout-bank-v2',0,15),list(range(15)))
        self.assertEqual(select_policy_rows(rows,'heldout-bank-v2',15,15),list(range(15,30)))
        for offset,count in ((1,15),(0,30),(15,None),(15,14)):
            with self.assertRaises(ValueError):select_policy_rows(rows,'heldout-bank-v2',offset,count)
        self.assertEqual(select_policy_rows(list(range(5)),'seen-training-home'),list(range(5)))
        with self.assertRaises(ValueError):select_policy_rows(list(range(5)),'seen-training-home',0,4)
        with self.assertRaises(ValueError):select_policy_rows(list(range(29)),'heldout-bank-v2',15,15)
    def test_ordered_resume_subset_never_changes_base_chunk_or_defaults(self):
        rows=[dict(seed=900000+i) for i in range(30)]
        selected=select_policy_rows(rows,'heldout-bank-v2',0,15,list(range(900002,900015)))
        self.assertEqual([r['seed'] for r in selected],list(range(900002,900015)))
        self.assertEqual(select_policy_rows(rows,'heldout-bank-v2',0,15),rows[:15])
        for requested in ([],[900002,900002],[900003,900002],[900015],[899999]):
            with self.assertRaises(ValueError):select_policy_rows(rows,'heldout-bank-v2',0,15,requested)
        seen=[dict(seed=530000+i) for i in range(5)]
        self.assertEqual(select_policy_rows(seen,'seen-training-home',episode_seeds=[530002,530004]),[seen[2],seen[4]])
    def test_seen_bank_keeps_recorded_state_and_binds_actual_training_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);files={};normal=root/'training/normal';normal.mkdir(parents=True);(normal/'run.json').write_text('{}')
            wanted=[]
            for seed in range(530000,530005):
                color=('red','green','blue')[(seed-530000)%3];attempt=normal/f'attempt-{seed}';attempt.mkdir()
                receipt=attempt/'receipt.json';receipt.write_text(json.dumps(dict(accepted=True,params=dict(seed=seed,color=color,task=f'Pick a {color} lego and place it in {color} cup',recovery='none'))))
                frames=attempt/'frames.jsonl';frames.write_text('{"sim_ns": 33333333}\n')
                state=dict(seed=seed,color=color,task=f'Pick a {color} lego and place it in {color} cup',state=[.1]*6,recorded_drive_targets=[.2]*6,source_receipt=str(receipt),source_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),source_frames=str(frames),source_frames_sha256=hashlib.sha256(frames.read_bytes()).hexdigest())
                bodies={n:dict(position=[0,0,0],orientation=[1,0,0,0]) for n in NAMES};wanted.append(dict(state,bodies=bodies))
                for prefix,value in [('state',state),('seed',bodies)]:
                    name=f'{prefix}-{seed}.json';p=root/name;p.write_text(json.dumps(value));files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
            manifest=root/'bank-manifest.json';manifest.write_text(json.dumps(dict(kind='seen-training-start-diagnostic',mode='home',seeds=list(range(530000,530005)),files_sha256=files)));sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
            rows,suite=load_bank(root,sha);self.assertEqual(suite,'seen-training-home')
            for actual,original in zip(rows,wanted):
                for key in original:self.assertEqual(actual[key],original[key])
                self.assertEqual(actual['reference_run'],str(normal/'run.json'));self.assertEqual(actual['first_frame_reference'],actual['source_frames'])
            frames.write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'Training source file SHA'):load_bank(root,sha)
if __name__=='__main__':unittest.main()
