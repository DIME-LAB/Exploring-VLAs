"""CPU teacher-forced early-trajectory errors through saved native processors."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os,resource,time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import default_collate
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--weights-sha256',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--frames',type=int,default=180);a=p.parse_args();assert not a.out.exists();a.out.mkdir(parents=True)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available() and not torch.cuda.is_initialized();assert os.getpriority(os.PRIO_PROCESS,0)==10
 torch.set_num_threads(4);torch.set_num_interop_threads(1);pa.set_cpu_count(4);pa.set_io_thread_count(1)
 assert hashlib.sha256((a.checkpoint/'model.safetensors').read_bytes()).hexdigest()==a.weights_sha256
 metadata=json.loads((a.dataset/'meta/collection_episodes.json').read_text());cases=[row for row in metadata if row['params']['seed'] in range(530000,530005)];cases.sort(key=lambda r:r['params']['seed']);assert len(cases)==5 and [r['episode_index'] for r in cases]==list(range(5))
 config=PreTrainedConfig.from_pretrained(a.checkpoint);config.device='cpu';config.compile_model=False;config.rtc_config=None;policy=SmolVLAPolicy.from_pretrained(a.checkpoint,config=config).to('cpu').eval()
 pre,post=make_pre_post_processors(policy.config,pretrained_path=a.checkpoint,preprocessor_overrides={'device_processor':{'device':'cpu'},'rename_observations_processor':{'rename_map':{}}},postprocessor_overrides={'device_processor':{'device':'cpu'}})
 root=Path(__file__).resolve().parents[3];path=root/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/compute_workspace.py';spec=importlib.util.spec_from_file_location('teacher_fk',path);fk=importlib.util.module_from_spec(spec);spec.loader.exec_module(fk)
 ds=LeRobotDataset('local/smolvla-phase3-normal-60k-s42-20261114',root=a.dataset,episodes=list(range(5)),video_backend='torchcodec',delta_timestamps={'action':[i/30 for i in range(50)]},return_uint8=False)
 table=pq.read_table(sorted(a.dataset.glob('data/**/*.parquet')),columns=['index','episode_index','frame_index']).to_pydict();indices={(ep,frame):int(index) for ep,frame,index in zip(table['episode_index'],table['frame_index'],table['index']) if ep<5 and frame<a.frames};rows=[]
 receipt=dict(status='running',started_utc=time.strftime('%FT%TZ',time.gmtime()),weights_sha256=a.weights_sha256,checkpoint=str(a.checkpoint),dataset=str(a.dataset),sampling='RTC off, policy reset and same flow seed42 at every teacher-forced frame',frames_per_episode=a.frames,episodes=[r['params']['seed'] for r in cases],cpu=dict(pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),torch_threads=torch.get_num_threads(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_available=torch.cuda.is_available(),cuda_initialized=torch.cuda.is_initialized()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fk_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),dataset_metadata_sha256=hashlib.sha256((a.dataset/'meta/collection_episodes.json').read_bytes()).hexdigest())
 (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');started=time.monotonic()
 for case in cases:
  episode=case['episode_index'];seed=case['params']['seed'];predictions=[];truths=[];states=[]
  for frame in range(a.frames):
   sample=ds[indices[(episode,frame)]];assert int(sample['episode_index'])==episode and int(sample['frame_index'])==frame and not sample['action_is_pad'].any();batch=default_collate([sample]);assert batch['task']==[case['params']['task']]
   policy.reset();torch.manual_seed(42);t=time.monotonic()
   with torch.inference_mode():
    prepared=pre(batch);normalized=policy.predict_action_chunk(prepared);pred=post(normalized.flatten(0,1)).reshape(1,50,6)[0].float().cpu().numpy()
   truth=sample['action'].numpy();state=sample['observation.state'].numpy();assert pred.shape==truth.shape==(50,6) and np.isfinite(pred).all()
   err=np.abs(pred-truth);tcp=fk.forward_kinematics_batch(np.radians(pred[:,:5]));expert=fk.forward_kinematics_batch(np.radians(truth[:,:5]));tcp_error=np.linalg.norm(tcp-expert,axis=1)*1000;hold=np.linalg.norm(fk.forward_kinematics_batch(np.radians(np.broadcast_to(state[:5],(50,5))))-expert,axis=1)*1000
   row=dict(seed=seed,color=case['params']['color'],frame=frame,time_sim_s=frame/30,prediction_wall_s=time.monotonic()-t,first_action_arm_l1_deg=float(err[0,:5].mean()),first10_arm_l1_deg=float(err[:10,:5].mean()),full50_arm_l1_deg=float(err[:,:5].mean()),first10_gripper_l1_deg=float(err[:10,5].mean()),full50_gripper_l1_deg=float(err[:,5].mean()),first_action_tcp_mm=float(tcp_error[0]),first10_tcp_mean_mm=float(tcp_error[:10].mean()),first10_tcp_p95_mm=float(np.quantile(tcp_error[:10],.95)),full50_tcp_mean_mm=float(tcp_error.mean()),full50_tcp_p95_mm=float(np.quantile(tcp_error,.95)),full50_tcp_max_mm=float(tcp_error.max()),hold_first10_tcp_mean_mm=float(hold[:10].mean()),hold_full50_tcp_mean_mm=float(hold.mean()))
   rows.append(row);predictions.append(pred.copy());truths.append(truth.copy());states.append(state.copy())
   with (a.out/'frames.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
   if frame%30==29:print(json.dumps(dict(event='teacher-forced',seed=seed,frames=frame+1,first10_tcp_mean_mm=float(np.mean([r['first10_tcp_mean_mm'] for r in rows[-30:]])),wall_s=time.monotonic()-started)),flush=True)
  np.savez_compressed(a.out/f'seed-{seed}.npz',prediction=np.asarray(predictions),expert=np.asarray(truths),state=np.asarray(states))
  with (a.out/'frames.csv').open('w') as stream:writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
 receipt.update(status='complete',finished_utc=time.strftime('%FT%TZ',time.gmtime()),predictions=len(rows),wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cuda_initialized_final=torch.cuda.is_initialized());assert not receipt['cuda_initialized_final'];(a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
