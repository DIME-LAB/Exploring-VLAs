"""Exercise triplet preparation through the Collector and shared native restore."""
import argparse,copy,json,os,subprocess,sys,tempfile,threading,unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from contract import parameters,EVAL_SEEDS
from scene_identity import BODY_NAMES,identity
from target_triplets import TripletSession,POLICY,TRAIN_SEEDS,HELDOUT_SEEDS,spec,member,plan,destination,configure_args,validate_run
import test_scene_identity
from test_review_regressions import load_class
from storage import CANONICAL_ROOT
from tick_contract import STEP_NS


class Harness:
    def __init__(self,root,config):
        self.root=root;self.config=config;self.runtime,self.data=test_scene_identity.SceneIdentityTests().runtime()
        for color in ('red','green','blue'):self.data['bodies'][color+'_2x3_1']['position'][0]+=.125
        self.runtime.stamp=lambda:self.data['sim_ns'];self.normal_calls=0;self.visible=True;self.bad=None;self.advances=[]
        Collector,_=load_class('collect.py','Collector');c=object.__new__(Collector);self.collector=c
        @contextmanager
        def own(*a):yield
        c.generation=NS(epoch=self.runtime.epoch,own=own,require=lambda *a:None)
        def cancel():
            self.runtime.epoch+=1;self.runtime.invalid_through_ns=self.data['sim_ns'];c.generation.epoch=self.runtime.epoch
        self.runtime.cancel=cancel
        c.writer=None;c.triplet_session=TripletSession(root,config)
        def normal(*args):self.normal_calls+=1;args[2]['cup_randomization']=dict(status='success',seed=config['layout_seed'])
        c._prepare_normal_scene_attempt=normal
        c.held_snapshot=lambda epoch:self.runtime.snapshot()
        c.timeline=lambda play:cancel() if not play else None
        def restore(code):
            scope={}
            with patch.dict('sys.modules',{'collection_runtime':NS(runtime=lambda:self.runtime)}):exec(code,scope)
            c.generation.epoch=scope['result']['epoch'];return scope['result']
        c.restore_physical=restore
        def advance(count):
            self.advances.append(count);out=[]
            for _ in range(count):
                self.data['sim_ns']+=STEP_NS;self.data['physics_time_s']+=1/30
                value=self.runtime.snapshot()
                if count==1 and self.bad:
                    if self.bad=='body':value['bodies']['cup_green']['position'][0]=float(np.nextafter(np.float32(.125),np.float32(1.)))
                    if self.bad=='epoch':value['epoch']-=1
                    if self.bad=='clock':value['sim_ns']+=5000000000
                out.append(value)
            return out
        c.advance_ticks=advance
        c.scene_reset=NS(clear=lambda:dict(verified=True));c.gpu_footprint=lambda:dict(passed=True);c.camera_graph=lambda:dict(passed=True)
        def sync(snapshot):c.scene_pose_check=dict(passed=True);c.scene_clear_check=dict(verified=True)
        c.sync_scene=sync
        c.node=NS(_qs_abort_evt=threading.Event(),objects_data={},_drop_data={},_grasp_approach_height_var=NS(set=lambda v:None))
        def cache(s):
            c.node.objects_data={n:copy.deepcopy(b) for n,b in s['bodies'].items() if not n.startswith('cup_')}
            c.node._drop_data={n[4:]:copy.deepcopy(b) for n,b in s['bodies'].items() if n.startswith('cup_')}
        c.update_pose_cache=cache;c.servo=NS(configure=lambda p:None);c.sensors=NS(duplicate_messages={})
        c.mutate_kit=lambda *a:{member(config,i,c.triplet_session.anchor)['object']:self.visible for i in range(3)}
    def prepare(self,index):
        c=self.collector;p=member(self.config,index);receipt=dict(params=p,episode_id='episode-'+str(index))
        c._prepare_scene_attempt(p,{},receipt,set(),self.root,self.root/f'attempt-{index:04d}')
        return receipt
    def complete(self):
        receipts=[]
        for index in range(3):
            r=self.prepare(index);r.update(accepted=True,timing=dict(passed=True));receipts.append(r)
            directory=self.root/f'attempt-{index:04d}';directory.mkdir()
            (directory/'receipt.json').write_text(json.dumps(r))
            s=r['episode_start_physical'];m=r['counterfactual']
            row=dict(episode_id=r['episode_id'],layout_id=m['layout_id'],triplet_id=m['triplet_id'],
                epoch=s['epoch'],sim_ns=s['sim_ns'],pose_stamp_ns=s['sim_ns'],
                physics_state=s['state'],physics_velocity=s['velocity'],physics_drive_targets=s['drive_targets'],
                physical_bodies=s['bodies'],base_world_pose=s['base_world_pose'],robot_base_world_pose=s['robot_base_world_pose'])
            (directory/'frames.jsonl').write_text(json.dumps(row)+'\n')
            for b in self.data['bodies'].values():b['position'][0]+=.5
            self.data['state'][0]+=.25
        (self.root/'run.json').write_text(json.dumps(dict(counterfactual_triplet=self.config)))
        return receipts


class TripletTests(unittest.TestCase):
    def fixture(self,d,seed=720002,split='train'):
        config=spec(seed,split);root=Path(d)/'datasets/counterfactual-targets'/split/config['layout_id'];root.mkdir(parents=True)
        return Harness(root,config)
    def test_actual_collector_preparation_restores_rgb_identically_and_samples_once(self):
        with tempfile.TemporaryDirectory() as d:
            h=self.fixture(d);receipts=h.complete();proof=validate_run(h.root,base=Path(d))
            self.assertTrue(proof['complete']);self.assertEqual(h.normal_calls,1)
            self.assertEqual(h.advances,[20,1]*3)
            self.assertEqual(len({identity(r['initial_physical']) for r in receipts}),1)
            self.assertEqual(len({identity(r['episode_start_physical']) for r in receipts}),1)
            self.assertEqual({r['params']['color'] for r in receipts},{'red','green','blue'})
            common=lambda p:{k:v for k,v in p.items() if k not in ('color','object','task','trial','target_selection')}
            self.assertEqual([common(r['params']) for r in receipts],[common(receipts[0]['params'])]*3)
            self.assertEqual(len({r['params']['object'].rsplit('_',1)[1] for r in receipts}),1)
            self.assertTrue(all(r['params']['target_selection']['margin_m']>.005 for r in receipts))
    def test_ambiguous_candidate_pair_refuses_the_whole_layout_before_writer(self):
        with tempfile.TemporaryDirectory() as d:
            h=self.fixture(d);h.data['bodies']['blue_2x3_1']['position']=list(h.data['bodies']['blue_2x3_0']['position'])
            with self.assertRaisesRegex(ValueError,'<=5 mm'):h.prepare(0)
            self.assertIsNone(h.collector.writer)
    def test_real_preparation_updates_the_params_used_by_cycle_execution(self):
        seed=next(s for s in TRAIN_SEEDS if member(spec(s,'train'),0)['object'].endswith('_1'))
        with tempfile.TemporaryDirectory() as d:
            h=self.fixture(d,seed);params=member(h.config,0);receipt=dict(params=params,episode_id='chosen')
            self.assertTrue(params['object'].endswith('_1'))
            h.collector._prepare_scene_attempt(params,{},receipt,set(),h.root,h.root/'attempt-0000')
            self.assertIs(params,receipt['params']);self.assertTrue(params['object'].endswith('_0'))
            self.assertEqual(params['object'],params['target_selection']['object'])
    def test_body_clock_or_epoch_difference_fails_before_a_writer(self):
        for bad in ('body','clock','epoch'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as d:
                h=self.fixture(d);h.prepare(0);h.bad=bad
                with self.assertRaises((ValueError,RuntimeError)):h.prepare(1)
                self.assertIsNone(h.collector.writer);self.assertEqual(h.normal_calls,1)
    def test_heldout_triplet_restores_identically_and_cannot_export_under_train(self):
        with tempfile.TemporaryDirectory() as d:
            h=self.fixture(d,730000,'heldout');receipts=h.complete()
            proof=validate_run(h.root,base=Path(d))
            self.assertEqual(proof['split'],'heldout')
            self.assertEqual(len({identity(r['initial_physical']) for r in receipts}),1)
            config=json.loads((h.root/'run.json').read_text())
            config['counterfactual_triplet']['split']='train'
            (h.root/'run.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):validate_run(h.root,base=Path(d))
    def test_hidden_target_refuses_instead_of_changing_which_block(self):
        with tempfile.TemporaryDirectory() as d:
            h=self.fixture(d);h.visible=False
            with self.assertRaisesRegex(RuntimeError,'target must be visible'):h.prepare(0)
            self.assertEqual(h.normal_calls,1);self.assertIsNone(h.collector.writer)
    def test_partial_or_edited_members_cannot_export(self):
        for edit in ('partial','task','noise','layout','visibility','physical','first_row','later_id','duplicate_episode','selection'):
            with self.subTest(edit=edit),tempfile.TemporaryDirectory() as d:
                h=self.fixture(d);h.complete();path=h.root/'attempt-0001/receipt.json';r=json.loads(path.read_text())
                if edit=='partial':path.unlink()
                elif edit=='first_row':
                    path=h.root/'attempt-0001/frames.jsonl';row=json.loads(path.read_text());row['physics_state'][0]=float(np.nextafter(np.float32(row['physics_state'][0]),np.float32(1.)));path.write_text(json.dumps(row)+'\n')
                elif edit=='later_id':
                    path=h.root/'attempt-0001/frames.jsonl';line=path.read_text();row=json.loads(line);row['layout_id']='other';path.write_text(line+json.dumps(row)+'\n')
                else:
                    if edit=='task':r['params']['task']='Pick a blue lego and place it in blue cup'
                    if edit=='noise':r['params']['noise_seed']+=1
                    if edit=='layout':r['counterfactual']['layout_id']='other'
                    if edit=='visibility':r['counterfactual']['visibility']={}
                    if edit=='physical':r['initial_physical']['bodies']['cup_red']['position'][0]+=.001
                    if edit=='duplicate_episode':r['episode_id']='episode-0'
                    if edit=='selection':r['params']['target_selection']['candidates'][0]['distance_xy_m']+=.01
                    path.write_text(json.dumps(r))
                with self.assertRaises((ValueError,KeyError)):validate_run(h.root,base=Path(d))
    def test_plan_has_420_balanced_episodes_and_separate_disjoint_layouts(self):
        p=plan('/mnt/storage3/aaugus11/exploring-vlas',[3,4,5,6]);self.assertEqual(p['episodes'],420)
        self.assertEqual((p['training_episodes'],p['heldout_episodes']),(360,60))
        self.assertFalse(set(TRAIN_SEEDS+HELDOUT_SEEDS)&set(range(500000,700000)))
        self.assertFalse(set(TRAIN_SEEDS+HELDOUT_SEEDS)&set(EVAL_SEEDS))
        for instance in (3,4,5,6):self.assertEqual(sum(j['instance']==instance for j in p['jobs']),35)
        orders={tuple(spec(s,'train')['color_order']) for s in TRAIN_SEEDS};self.assertEqual(len(orders),6)
        self.assertTrue(all('/'+j['config']['split']+'/' in j['output'] for j in p['jobs']))
        with self.assertRaises(ValueError):parameters(730000,'red',split='train')
        with self.assertRaises(ValueError):parameters(720000,'red',regrasp=True)
        with self.assertRaises(ValueError):spec(720000,'heldout')
    def test_mode_refuses_resume_recovery_and_mixed_split_paths(self):
        parser=argparse.ArgumentParser()
        for field in ('resume','replay_receipt','scene_bank','regrasp_only'):
            args=NS(counterfactual_layout_seed=720000,counterfactual_split='train',output=CANONICAL_ROOT/'datasets/counterfactual-targets/train/layout-720000',resume=False,replay_receipt=None,scene_bank=None,regrasp_only=False)
            setattr(args,field,True)
            with self.subTest(field=field),self.assertRaises(SystemExit):configure_args(args,parser)
        with self.assertRaises(ValueError):destination(Path('/tmp/heldout'),spec(720000,'train'))
    def test_clean_interpreter_cpu_plan_emits_one_train_and_one_heldout_triplet(self):
        with tempfile.TemporaryDirectory(dir=CANONICAL_ROOT/'tmp') as d:
            out=Path(d)/'plan.json';script=Path(__file__).with_name('target_triplets.py')
            result=subprocess.run([sys.executable,str(script),'--plan','--output',str(out),'--train-layouts','1','--heldout-layouts','1','--instances','3'],
                env=dict(os.environ,PYTHONPATH='',CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            value=json.loads(out.read_text());self.assertEqual(value['episodes'],6)
            self.assertEqual([j['config']['split'] for j in value['jobs']],['train','heldout'])
            self.assertFalse(value['production_authorized']);self.assertIn('scene_identity.py',value['source_sha256'])


if __name__=='__main__':unittest.main()
