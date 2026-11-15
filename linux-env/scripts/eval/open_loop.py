"""CPU-only eager fidelity on seen/held-out frames via saved native processors."""
from pathlib import Path
import argparse,hashlib,json,os,time
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import default_collate
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from behaviour import fk,rotate
from direction_metrics import motion_directions
from checkpoint_identity import checkpoint_label
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--live-npz',type=Path);p.add_argument('--parity-only',action='store_true');p.add_argument('--home',action='store_true');p.add_argument('--colour-swap',action='store_true');p.add_argument('--flow-seeds',type=int,nargs='+',default=[42]);p.add_argument('--out',type=Path);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args();root=a.root;out=a.out or root/'behaviour';case_file=root/'behaviour/open-loop-cases.json';cases=json.loads(case_file.read_text());out.mkdir(parents=True,exist_ok=True)
assert len(a.flow_seeds)==len(set(a.flow_seeds)) and a.flow_seeds
assert a.colour_swap or a.flow_seeds==[42],'Multiple flow seeds require --colour-swap'
assert not (a.colour_swap and (out/'colour-swap.json').exists()),'Refusing to overwrite an existing colour-swap run'
if a.out is None:
    configs=[json.loads(c.read_text()) for c in root.glob('attempt-*/run_config.json')]
    assert configs and all(Path(c['pretrained'])==a.checkpoint for c in configs),'A different checkpoint requires --out'
output_name='colour-swap.json' if a.colour_swap else ('open-loop-live.json' if a.live_npz else ('open-loop-home.json' if a.home else 'open-loop.json'))
assert not (out/output_name).exists(),'Refusing to overwrite an existing diagnostic'
assert not a.colour_swap or a.home,'Colour-direction probes require --home'
if a.home:
    for c in cases:
        c['anchor_frame']=0;c['frame']=json.loads(next(Path(c['source_frames']).open()))
if a.colour_swap:
    cases=[dict(c,flow_noise_seed=s) for c in cases for s in a.flow_seeds]
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
torch.set_num_threads(4)
model=a.checkpoint
config=PreTrainedConfig.from_pretrained(model);config.device='cpu';config.compile_model=False;config.rtc_config=None
weights_sha256=hashlib.sha256((model/'model.safetensors').read_bytes()).hexdigest()
print(json.dumps(dict(stage='loading-model',device=config.device,checkpoint_path=str(model),checkpoint_label=checkpoint_label(model),weights_sha256=weights_sha256)),flush=True)
policy=SmolVLAPolicy.from_pretrained(model,config=config).to('cpu').eval()
pre,post=make_pre_post_processors(policy.config,pretrained_path=model,preprocessor_overrides={'device_processor':{'device':'cpu'},'rename_observations_processor':{'rename_map':{}}},postprocessor_overrides={'device_processor':{'device':'cpu'}})
job=Path('/mnt/a4500-backup/exploring-vlas/train-data/phase3.2/jobs/smolvla-phase3-normal-60k-s42-20261114')
manifest=json.loads((job/'batch-manifest.json').read_text());source_roots={b['root'] for b in manifest['batches']};assert all(c['dataset'] in source_roots for c in cases if c['kind']=='seen-training');assert all(c['dataset'] not in source_roots for c in cases if c['kind']=='held-out-pilot')
train_seeds=set()
for b in manifest['batches']:
 for line in (Path(b['root']).parent/'episodes.jsonl').open():
  r=json.loads(line)
  if r['accepted']:train_seeds.add(r['params']['seed'])
assert all(c['seed'] not in train_seeds for c in cases if c['kind']=='held-out-pilot')

def predict(batch,sampling_seed=42):
    policy.reset();torch.manual_seed(sampling_seed)
    started=time.monotonic()
    with torch.inference_mode():
        prepared=pre(batch);norm=policy.predict_action_chunk(prepared);pred=post(norm.flatten(0,1)).reshape(1,50,6)[0].float().cpu().numpy()
    assert pred.shape==(50,6) and np.isfinite(pred).all()
    return pred,time.monotonic()-started,prepared

results=[];datasets={}
def write_colour_results():
    data=dict(mode='CPU eager RTCoff; same pixels/state/noise within prompt triplet; colour words only',checkpoint_path=str(model),checkpoint_step=int(model.parent.name),checkpoint_label=checkpoint_label(model),weights_sha256=weights_sha256,flow_noise_seeds=a.flow_seeds,case_manifest_sha256=hashlib.sha256(case_file.read_bytes()).hexdigest(),results=results)
    (out/'colour-swap.json').write_text(json.dumps(data,indent=2)+'\n')
for case in cases[:1] if a.parity_only else cases:
    path=Path(case['dataset'])
    if str(path) not in datasets:
        ds=LeRobotDataset('local/phase3-diagnostic',root=path,video_backend='torchcodec',delta_timestamps={'action':[i/30 for i in range(50)]},return_uint8=False)
        tab=pq.read_table(sorted(path.glob('data/**/*.parquet')),columns=['index','episode_index','frame_index']).to_pandas()
        datasets[str(path)]=(ds,tab)
    ds,tab=datasets[str(path)];anchor=tab[(tab.episode_index==case['episode_index'])&(tab.frame_index==case['anchor_frame'])];assert len(anchor)==1;index=int(anchor.iloc[0]['index']);sample=ds[index];batch=default_collate([sample]);truth=sample['action'].numpy();state=sample['observation.state'].numpy();assert not sample['action_is_pad'].any();assert batch['task']==[case['task']]
    assert np.max(np.abs(np.degrees(case['frame']['state'])-state))<1e-4
    if a.colour_swap:
        prompts={};arrays={};colors=('red','green','blue');base=case['frame']['robot_base_world_pose'];start_local=fk.forward_kinematics_batch(np.radians(state[None,:5]))[0];start_world=np.asarray(rotate(base['orientation'],start_local.tolist()))+base['position'];saved_state=batch['observation.state'].clone()
        def fingerprints():return {k:hashlib.sha256(batch[k].contiguous().numpy().tobytes()).hexdigest() for k in (*policy.config.image_features,'observation.state')}
        input_hashes=fingerprints()
        for color in colors:
            changed=dict(batch);changed['task']=[f'Pick a {color} lego and place it in {color} cup'];pred,seconds,prepared=predict(changed,case['flow_noise_seed'])
            assert torch.equal(batch['observation.state'],saved_state)
            assert fingerprints()==input_hashes,'Prediction mutated a raw input tensor'
            local=fk.forward_kinematics_batch(np.radians(pred[:,:5]));tcp=np.asarray([np.asarray(rotate(base['orientation'],pt.tolist()))+base['position'] for pt in local]);endpoint=tcp[-1];motion=endpoint-start_world;distances={};directions={}
            for targetcolor in colors:
                names=[n for n in case['frame']['physical_bodies'] if n.startswith(targetcolor+'_')];positions=np.asarray([case['frame']['physical_bodies'][n]['position'] for n in names]);vectors=positions-start_world;norms=np.linalg.norm(vectors,axis=1)*np.linalg.norm(motion)
                distances[targetcolor]=float(np.linalg.norm(positions-endpoint,axis=1).min()*1000);directions[targetcolor]=float(np.max((vectors@motion)/np.maximum(norms,1e-12)))
            tokens=prepared['observation.language.tokens'][0].cpu().tolist();mask=prepared['observation.language.attention_mask'][0].cpu().tolist()
            blocks={n:b['position'] for n,b in case['frame']['physical_bodies'].items() if n.split('_')[0] in colors}
            assert all(sum(n.startswith(c+'_') for n in blocks)==2 for c in colors)
            prompts[color]=dict(task=changed['task'][0],tokens=tokens,token_mask=mask,normalized_state=prepared['observation.state'][0].cpu().tolist(),endpoint_world_m=endpoint.tolist(),displacement_world_m=motion.tolist(),endpoint_distance_nearest_block_mm=distances,direction_cosine_best_block=directions,geometry=motion_directions(start_world,tcp,blocks,color),wall_s=seconds)
            arrays['pred_'+color]=pred;arrays['tcp_'+color]=tcp
        pairs=[]
        for left,right in [('red','green'),('red','blue'),('green','blue')]:
            diff=np.abs(arrays['pred_'+left]-arrays['pred_'+right]);td=np.linalg.norm(arrays['tcp_'+left]-arrays['tcp_'+right],axis=1)*1000
            tl=prompts[left]['tokens'];tt=prompts[right]['tokens'];assert tl!=tt
            pairs.append(dict(prompts=[left,right],arm_l1_deg=float(diff[:,:5].mean()),gripper_l1_deg=float(diff[:,5].mean()),tcp_mean_shift_mm=float(td.mean()),tcp_endpoint_shift_mm=float(td[-1]),tcp_max_shift_mm=float(td.max()),changed_token_positions=[i for i,(x,y) in enumerate(zip(tl,tt)) if x!=y]))
        control=None
        if not results:
            identical=dict(batch);identical['task']=[prompts['red']['task']];repeat,_,_=predict(identical,case['flow_noise_seed']);control=float(np.abs(repeat-arrays['pred_red']).max());assert control<1e-6
        source_control=None
        if case['flow_noise_seed']==a.flow_seeds[0]:
            native,_,_=predict(batch,case['flow_noise_seed']);source_control=float(np.abs(native-arrays['pred_'+case['color']]).max());assert source_control<1e-6
        assert fingerprints()==input_hashes
        result={k:case[k] for k in ('kind','seed','color','dataset','episode_index','anchor_frame')};result.update(prompts=prompts,pairs=pairs,identical_prompt_repeat_max_abs_deg=control,home_tcp_world_m=start_world.tolist(),state_degrees=state.tolist(),pixels_and_state_fixed=True,sampling_seed=case['flow_noise_seed'],input_sha256=input_hashes,block_positions_world_m=blocks,expert_target_body=case['frame'].get('target_body'))
        suffix=f"-noise-{case['flow_noise_seed']}"
        npz=out/f"colour-swap-{case['kind']}-{case['seed']}{suffix}.npz";assert not npz.exists(),'Refusing to overwrite prediction arrays'
        np.savez_compressed(npz,**arrays,state=state,truth=truth)
        result['source_prompt_control_max_abs_deg']=source_control
        result['robot_base_world_pose']=base;result['native_camera_order']=list(policy.config.image_features)
        assert all(prompts[c]['normalized_state']==prompts['red']['normalized_state'] for c in colors)
        results.append(result);write_colour_results();print(json.dumps(dict(kind=case['kind'],seed=case['seed'],flow_noise_seed=case['flow_noise_seed'],pairs=pairs)),flush=True);continue
    pred,seconds,prepared=predict(batch);err=np.abs(pred-truth);hold=np.abs(state[None,:]-truth)
    tcp_pred=fk.forward_kinematics_batch(np.radians(pred[:,:5]));tcp_true=fk.forward_kinematics_batch(np.radians(truth[:,:5]));tcp_err=np.linalg.norm(tcp_pred-tcp_true,axis=1)*1000
    name=('home-' if a.home else '')+f"{case['kind']}-{case['seed']}";npz=out/f'open-loop-{name}.npz';assert not npz.exists(),'Refusing to overwrite prediction arrays';np.savez_compressed(npz,pred=pred,truth=truth,state=state,tcp_pred=tcp_pred,tcp_true=tcp_true)
    result={k:case[k] for k in ('kind','seed','color','episode_index','anchor_frame','closure_frame','task','dataset')};result.update(index=index,device='cpu',rtc=False,seed=case['seed'],sampling_seed=42,wall_s=seconds,first_action_arm_l1_deg=float(err[0,:5].mean()),first10_arm_l1_deg=float(err[:10,:5].mean()),full50_arm_l1_deg=float(err[:,:5].mean()),full50_gripper_l1_deg=float(err[:,5].mean()),hold_first10_arm_l1_deg=float(hold[:10,:5].mean()),hold_full50_arm_l1_deg=float(hold[:,:5].mean()),tcp_error_mean_mm=float(tcp_err.mean()),tcp_error_p95_mm=float(np.quantile(tcp_err,.95)),tcp_error_max_mm=float(tcp_err.max()),native_camera_order=list(policy.config.image_features),raw_image_ranges={k:[float(batch[k].min()),float(batch[k].max())] for k in policy.config.image_features},normalized_state=prepared['observation.state'][0].cpu().tolist())
    if a.live_npz:
        assert case is cases[0] or case['seed']==cases[0]['seed']
        live=np.load(a.live_npz);livebatch=dict(batch)
        for key,short in [('observation.images.top','top'),('observation.images.wrist','wrist')]:
            pixels=live[short];assert pixels.shape==(480,640,3) and pixels.dtype==np.uint8
            livebatch[key]=torch.from_numpy(pixels.copy()).permute(2,0,1)[None].float()/255
        livebatch['observation.state']=batch['observation.state'].clone()
        lp,latency,_=predict(livebatch);delta=np.abs(lp-pred);ltcp=fk.forward_kinematics_batch(np.radians(lp[:,:5]));diff=np.linalg.norm(ltcp-tcp_pred,axis=1)*1000
        np.savez_compressed(out/'live-vs-dataset-predictions.npz',dataset=pred,live=lp,truth=truth,state=state,live_tcp=ltcp,dataset_tcp=tcp_pred,truth_tcp=tcp_true)
        result['live_comparison']=dict(cpu_wall_s=latency,identical_state=True,paired_sampling_seed=42,action_delta_arm_mean_deg=float(delta[:,:5].mean()),action_delta_gripper_mean_deg=float(delta[:,5].mean()),tcp_shift_mean_mm=float(diff.mean()),tcp_shift_p95_mm=float(np.quantile(diff,.95)),tcp_shift_max_mm=float(diff.max()),live_arm_l1_vs_expert_deg=float(np.abs(lp-truth)[:,:5].mean()))
        for key,short in [('observation.images.top','top'),('observation.images.wrist','wrist')]:
            recorded=batch[key][0].permute(1,2,0).numpy()*255;actual=live[short].astype(float);mse=float(np.mean((recorded-actual)**2));result['live_comparison'][short]=dict(pixel_mae_dn=float(np.mean(np.abs(recorded-actual))),psnr_db=10*np.log10(255**2/mse) if mse else None)
            np.savez_compressed(out/f'image-parity-{short}.npz',dataset=recorded.clip(0,255).round().astype(np.uint8),live=live[short])
    results.append(result);print(json.dumps(result),flush=True)
    (out/('open-loop-live.json' if a.live_npz else ('open-loop-home.json' if a.home else 'open-loop.json'))).write_text(json.dumps(dict(checkpoint_path=str(model),checkpoint_step=int(model.parent.name),weights_sha256=hashlib.sha256((model/'model.safetensors').read_bytes()).hexdigest(),train_episodes=900,training_seed_count=len(train_seeds),held_out_batch_absent=True,mode='eager CPU, RTC off, paired sampling seed42',model_dtypes=sorted(set(str(p.dtype) for p in policy.parameters())),results=results),indent=2)+'\n')
print('COMPLETE',flush=True)
