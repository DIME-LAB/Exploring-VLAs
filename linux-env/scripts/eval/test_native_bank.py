import hashlib,json,tempfile,unittest
from pathlib import Path
from native_bank import load_bank,NAMES

class BankTests(unittest.TestCase):
    def bank(self,root):
        files={};meta={};parameters={}
        def write(name,value,inventory):
            p=root/name;p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(value));inventory[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        for seed in range(900000,900030):
            color=('red','green','blue')[seed%3];params=dict(seed=seed,color=color,split='eval',recovery='none',task=f'Pick a {color} lego and place it in {color} cup');parameters[str(seed)]=params
            bodies={n:dict(position=[0,0,0],orientation=[1,0,0,0]) for n in NAMES}
            write(f'seed-{seed}.json',bodies,files)
            write(f'metadata/seed-{seed}.json',dict(accepted=True,scene_bank_only=True,params=params,initial_physical=dict(state=[0]*6,drive_targets=[0]*6,robot_base_world_pose={},bodies=bodies)),meta)
        manifest=dict(format_version=2,purpose='held-out-collector-scenes',complete=True,producer='collect.Collector.run_scene_bank',seeds=list(range(900000,900030)),files_sha256=files,sampler_metadata_sha256=meta,parameters=parameters)
        p=root/'bank-manifest.json';p.write_text(json.dumps(manifest));return hashlib.sha256(p.read_bytes()).hexdigest()
    def test_all_thirty_pair_by_physical_layout(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sha=self.bank(root);rows,suite=load_bank(root,sha)
            self.assertEqual(suite,'heldout-bank-v2');self.assertEqual(len(rows),30);self.assertEqual([r['color'] for r in rows[:3]],['red','green','blue'])
    def test_receipt_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sha=self.bank(root);p=root/'metadata/seed-900000.json';p.write_text(p.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'file SHA'):load_bank(root,sha)
    def test_manifest_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sha=self.bank(root);p=root/'bank-manifest.json';p.write_text(p.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'manifest SHA'):load_bank(root,sha)
if __name__=='__main__':unittest.main()
