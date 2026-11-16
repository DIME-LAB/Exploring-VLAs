"""Exercise triplet preparation through the Collector and shared native restore."""
import argparse,copy,json,os,subprocess,sys,tempfile,threading,unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from contract import parameters,EVAL_SEEDS,corrective_action
from scene_identity import BODY_NAMES,identity
from target_triplets import TripletSession,POLICY,TRAIN_SEEDS,HELDOUT_SEEDS,spec,member,plan,destination,configure_args,validate_run,seal_run,require_mode
import test_scene_identity
from test_review_regressions import load_class
from storage import CANONICAL_ROOT
from tick_contract import STEP_NS,TickContract


class Harness:
    def __init__(self,root,config):
        self.root=root;self.config=config;self.runtime,self.data=test_scene_identity.SceneIdentityTests().runtime()
        for color in ('red','green','blue'):self.data['bodies'][color+'_2x3_1']['position'][0]+=.125
        self.runtime.stamp=lambda:self.data['sim_ns'];self.normal_calls=0;self.visible=True;self.bad=None;self.advances=[]
        native=self.runtime.snapshot
        self.runtime.snapshot=lambda:dict(native(),state_source='isaac_articulation',robot_base_pose_source='physx_rigid_body')
        self.audit=TickContract();self.after_bootstrap=None
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
            c.generation.epoch=scope['result']['epoch'];self.audit.install_boundary(scope['result'],restore=True);return scope['result']
        c.restore_physical=restore
        def advance(count):
            self.advances.append(count);out=[]
            for _ in range(count):
                self.data['sim_ns']+=STEP_NS;self.data['physics_time_s']+=1/30
                if count==1 and self.after_bootstrap:self.after_bootstrap(self.data)
                value=self.checked_snapshot()
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
    def checked_snapshot(self):
        s=self.runtime.snapshot();stamp=s['sim_ns'];epoch=s['epoch'];target=list(s['drive_targets'])
        session=self.collector.triplet_session
        row=dict(tick_contract_version=1,epoch=epoch,observation_epoch=epoch,expert_epoch=epoch,sim_ns=stamp,pose_stamp_ns=stamp,
            physics_time_s=s['physics_time_s'],invalid_through_ns=self.runtime.invalid_through_ns,state_source='isaac_articulation',
            state=s['state'],physics_state=s['state'],velocity=s['velocity'],physics_velocity=s['velocity'],physics_drive_targets=target,
            physical_bodies=s['bodies'],base_world_pose=s['base_world_pose'],robot_base_world_pose=s['robot_base_world_pose'],robot_base_pose_source='physx_rigid_body',
            episode_id=self.collector.current_episode_id,layout_id=session.config['layout_id'],triplet_id=session.triplet_id,
            camera_layout={k:dict(encoding='rgb8',width=640,height=480,step=1920,bytes=921600) for k in ('wrist','top')},
            execution_geometry=dict(passed=True,epoch=epoch,sim_ns=stamp,target=target),executing=True,phase='fixture',
            action=corrective_action(s['state'],target),executed_action=target,expert_reference=target,
            expert_observation_stamp_ns=stamp,observation_stamp_ns=stamp,declared_application_stamp_ns=stamp,application_stamp_ns=stamp,
            command_clock_ns=stamp,applied_command=target,planner_execution_reference=target,dart_offset=[0.]*6,
            command_filter_delta=[0.]*6,command_velocity=[0.]*5)
        for k in ('joint_stamp_ns','wrist_stamp_ns','top_stamp_ns'):row[k]=stamp
        for k in ('wall_ns','joint_receive_ns','wrist_receive_ns','top_receive_ns'):row[k]=stamp+10**9
        self.audit.observe(row);s['tick_contract']=self.audit.finish(row)
        return s
    def prepare(self,index):
        c=self.collector;p=member(self.config,index);receipt=dict(params=p,episode_id='episode-'+str(index))
        c._prepare_scene_attempt(p,{},receipt,set(),self.root,self.root/f'attempt-{index:04d}')
        return receipt
    def complete(self,recorded_ticks=1,first_offset_ticks=0):
        receipts=[]
        for index in range(3):
            r=self.prepare(index);r.update(accepted=True,timing=dict(passed=True));receipts.append(r)
            directory=self.root/f'attempt-{index:04d}';rows=[]
            for tick in range(recorded_ticks):
                if tick or first_offset_ticks:
                    self.data['sim_ns']+=STEP_NS;self.data['physics_time_s']+=1/30
                    if tick:self.data['state'][0]=float(np.float32(.125+.001*tick));self.data['velocity'][0]=.03
                    s=self.checked_snapshot()
                else:s=r['episode_start_physical']
                row=copy.deepcopy(s['tick_contract']['row']);self.collector.triplet_session.record_start(s,row,directory);rows.append(row)
            (directory/'receipt.json').write_text(json.dumps(r))
            (directory/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            for b in self.data['bodies'].values():b['position'][0]+=.5
            self.data['state'][0]+=.25
        (self.root/'run.json').write_text(json.dumps(dict(counterfactual_triplet=self.config)))
        seal=seal_run(self.root,base=self.root.parents[3]);(self.root/'summary.json').write_text(json.dumps(dict(counterfactual_triplet_seal=seal)))
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
    def test_common_first_row_substitution_refuses_at_zero_and_next_tick(self):
        for offset in (0,1):
            with self.subTest(offset=offset),tempfile.TemporaryDirectory() as d:
                h=self.fixture(d);h.complete(first_offset_ticks=offset)
                for path in h.root.glob('attempt-*/frames.jsonl'):
                    row=json.loads(path.read_text());row['physical_bodies']['red_2x3_0']['position'][0]+=.25;path.write_text(json.dumps(row)+'\n')
                with self.assertRaisesRegex(ValueError,'native tick proof'):validate_run(h.root,base=Path(d),sealed=False)
    def test_bootstrap_rule_rejects_ambiguous_or_swapped_observed_candidates(self):
        for kind in ('ambiguous','swapped'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                h=self.fixture(d)
                for color in ('red','green','blue'):
                    for i,x in enumerate((.100,.106)):h.data['bodies'][f'{color}_2x3_{i}']['position']=[x,0.,0.]
                def move(data):
                    for color in ('red','green','blue'):
                        for i,x in enumerate((.100,.104) if kind=='ambiguous' else (.106,.100)):
                            data['bodies'][f'{color}_2x3_{i}']['position']=[x,0.,0.]
                h.after_bootstrap=move
                with self.assertRaisesRegex(ValueError,'<=5 mm|differs from anchor'):h.prepare(0)
                self.assertIsNone(h.collector.writer)
    def test_next_recorded_tick_rechecks_rule_before_native_certificate_or_row(self):
        with tempfile.TemporaryDirectory() as d:
            h=self.fixture(d);r=h.prepare(0);path=h.root/'attempt-0000'
            for i,x in enumerate((.100,.104)):h.data['bodies'][f'red_2x3_{i}']['position']=[x,0.,0.]
            h.data['sim_ns']+=STEP_NS;h.data['physics_time_s']+=1/30;s=h.checked_snapshot()
            with self.assertRaisesRegex(ValueError,'<=5 mm'):h.collector.triplet_session.record_start(s,s['tick_contract']['row'],path)
            self.assertFalse((path/'first-recorded-native.json').exists())
    def test_reused_native_member_chain_refuses_before_group_seal_checks(self):
        with tempfile.TemporaryDirectory() as d:
            h=self.fixture(d);receipts=h.complete()
            for i in (1,2):
                r=receipts[i]
                for key in ('initial_physical','episode_start_physical','initial_restore_check'):r[key]=copy.deepcopy(receipts[0][key])
                (h.root/f'attempt-{i:04d}/receipt.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'reused or overlapped'):validate_run(h.root,base=Path(d),sealed=False)
    def test_member_priming_requires_both_independent_clocks_to_advance(self):
        from triplet_boundary import later
        previous=dict(epoch=1,sim_ns=10**9,physics_time_s=1.)
        current=dict(epoch=2,sim_ns=10**9+20*STEP_NS,physics_time_s=1.+20/30)
        later(current,previous)
        for key in ('sim_ns','physics_time_s'):
            changed=dict(current);changed[key]=previous[key]+1
            if key=='physics_time_s':changed[key]=previous[key]+.001
            with self.subTest(clock=key),self.assertRaisesRegex(ValueError,'overlapped'):later(changed,previous)
    def test_missing_or_null_mode_and_edited_configuration_cannot_unbind_group(self):
        for change in ('missing','null','extra'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                h=self.fixture(d);h.complete();path=h.root/'run.json';doc=json.loads(path.read_text())
                if change=='missing':doc.pop('counterfactual_triplet')
                elif change=='null':doc['counterfactual_triplet']=None
                else:doc['unbound_change']=True
                path.write_text(json.dumps(doc))
                with self.assertRaisesRegex(ValueError,'configuration|inputs'):validate_run(h.root,base=Path(d))
    def test_each_triplet_mode_marker_alone_refuses_normal_export_interpretation(self):
        for marker in ('path','anchor','group','dataset','receipt','first_native','later_frame','reserved_seed','summary','null'):
            with self.subTest(marker=marker),tempfile.TemporaryDirectory() as d:
                root=Path(d)/('counterfactual-targets' if marker=='path' else 'normal');root.mkdir();doc={}
                names=dict(anchor='triplet-anchor.json',group='counterfactual-triplet.json',dataset='dataset/meta/counterfactual_triplets.json',
                    receipt='attempt-0000/receipt.json',first_native='attempt-0000/first-recorded-native.json',later_frame='attempt-0000/frames.jsonl',
                    reserved_seed='attempt-0000/receipt.json',summary='summary.json')
                if marker=='null':doc['counterfactual_triplet']=None
                elif marker in names:
                    p=root/names[marker];p.parent.mkdir(parents=True,exist_ok=True)
                    value={'counterfactual':{}} if marker=='receipt' else {'params':{'seed':720000}} if marker=='reserved_seed' else {'counterfactual_triplet_seal':{}} if marker=='summary' else {}
                    p.write_text('{}\n'+json.dumps(dict(layout_id='layout-720000'))+'\n' if marker=='later_frame' else json.dumps(value))
                with self.assertRaisesRegex(ValueError,'configuration'):require_mode(root,doc)
    def test_actual_capture_certifies_start_before_writer_and_refuses_failed_certificate(self):
        from test_review_regressions import ReviewRegressions
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as d:
                c,_,entry=ReviewRegressions().collector();events=[]
                entry['tick'][0]['base_world_pose']=dict(position=[0.]*3,orientation=[1.,0.,0.,0.])
                def record(snapshot,row,path):
                    events.append('native')
                    self.assertEqual(snapshot['tick_contract']['row'],{k:v for k,v in row.items() if k!='wall_profile'})
                    if fail:raise ValueError('Native first-record certificate refused')
                c.triplet_session=NS(anchor=True,config=dict(layout_id='layout'),triplet_id='triplet',record_start=record)
                c.current_attempt_path=Path(d);c.writer=NS(add=lambda *a:events.append('write'))
                c.capture.__func__.__globals__['Sensors']=NS(image=lambda msg:np.zeros((2,2,3),dtype=np.uint8))
                c.capture();self.assertEqual(events,['native'] if fail else ['native','write'])
                if fail:self.assertIn('certificate refused',c.failure)
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
