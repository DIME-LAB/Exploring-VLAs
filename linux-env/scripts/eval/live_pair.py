"""CPU matched-state camera-pair parity for the native policy input path."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import default_collate
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from PIL import Image

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--native-root',type=Path,required=True);p.add_argument('--teacher-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists();a.out.mkdir(parents=True)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available() and not torch.cuda.is_initialized();torch.set_num_threads(4);torch.set_num_interop_threads(1);pa.set_cpu_count(4);pa.set_io_thread_count(1)
 config=PreTrainedConfig.from_pretrained(a.checkpoint);config.device='cpu';config.compile_model=False;config.rtc_config=None;policy=SmolVLAPolicy.from_pretrained(a.checkpoint,config=config).to('cpu').eval()
 pre,post=make_pre_post_processors(policy.config,pretrained_path=a.checkpoint,preprocessor_overrides={'device_processor':{'device':'cpu'},'rename_observations_processor':{'rename_map':{}}},postprocessor_overrides={'device_processor':{'device':'cpu'}})
 root=Path(__file__).resolve().parents[3];spec=importlib.util.spec_from_file_location('pair_fk',root/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/compute_workspace.py');fk=importlib.util.module_from_spec(spec);spec.loader.exec_module(fk)
 ds=LeRobotDataset('local/smolvla-phase3-normal-60k-s42-20261114',root=a.dataset,episodes=list(range(5)),video_backend='torchcodec',delta_timestamps={'action':[i/30 for i in range(50)]},return_uint8=False)
 table=pq.read_table(sorted(a.dataset.glob('data/**/*.parquet')),columns=['index','episode_index','frame_index']).to_pydict();indices={ep:index for ep,frame,index in zip(table['episode_index'],table['frame_index'],table['index']) if ep<5 and frame==0};rows=[]
 def predict(batch):
  before={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in batch.items() if isinstance(v,torch.Tensor)}
  policy.reset();torch.manual_seed(42)
  with torch.inference_mode():result=post(policy.predict_action_chunk(pre(batch)).flatten(0,1)).reshape(1,50,6)[0].float().cpu().numpy()
  after={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in batch.items() if isinstance(v,torch.Tensor)}
  assert before==after,'Prediction changed a raw input tensor'
  return result
 for ep in range(5):
  seed=530000+ep;native=a.native_root/f'ep{ep:02d}';result=json.loads((native/'result.json').read_text());live=np.load(native/'first-cameras.npz');batch=default_collate([ds[int(indices[ep])]]);original=predict(batch);teacher=np.load(a.teacher_root/f'seed-{seed}.npz')['prediction'][0];repeat=float(abs(original-teacher).max());assert repeat<1e-5
  unchanged=predict(batch);assert np.max(abs(unchanged-original))<1e-6
  changed=dict(batch);pixels={}
  for camera in ('top','wrist'):
   key='observation.images.'+camera;recorded=(batch[key][0].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8);actual=live[camera];assert actual.shape==recorded.shape==(480,640,3) and actual.dtype==np.uint8
   mse=float(np.mean((recorded.astype(float)-actual.astype(float))**2));pixels[camera]=dict(mae_dn=float(np.mean(abs(recorded.astype(float)-actual.astype(float)))),psnr_db=float(10*np.log10(255**2/mse)) if mse else None,recorded_median=np.median(recorded.reshape(-1,3),axis=0).tolist(),live_median=np.median(actual.reshape(-1,3),axis=0).tolist())
   Image.fromarray(recorded).save(a.out/f'{seed}-{camera}-dataset.png');Image.fromarray(actual).save(a.out/f'{seed}-{camera}-live.png');changed[key]=torch.from_numpy(actual.copy()).permute(2,0,1)[None].float()/255
  live_pred=predict(changed)
  top_only=dict(batch);top_only['observation.images.top']=changed['observation.images.top'];top_pred=predict(top_only)
  wrist_only=dict(batch);wrist_only['observation.images.wrist']=changed['observation.images.wrist'];wrist_pred=predict(wrist_only)
  actual_state=np.asarray(result['client']['requests'][0]['capture']['state_degrees'],np.float32);actual_batch=dict(changed);actual_batch['observation.state']=torch.from_numpy(actual_state)[None];actual_pred=predict(actual_batch)
  tcp=fk.forward_kinematics_batch(np.radians(original[:,:5]));live_tcp=fk.forward_kinematics_batch(np.radians(live_pred[:,:5]));actual_tcp=fk.forward_kinematics_batch(np.radians(actual_pred[:,:5]));diff=np.linalg.norm(tcp-live_tcp,axis=1)*1000;diff_actual=np.linalg.norm(tcp-actual_tcp,axis=1)*1000
  top_shift=np.linalg.norm(fk.forward_kinematics_batch(np.radians(top_pred[:,:5]))-tcp,axis=1)*1000;wrist_shift=np.linalg.norm(fk.forward_kinematics_batch(np.radians(wrist_pred[:,:5]))-tcp,axis=1)*1000
  row=dict(unchanged_input_repeat_max_deg=float(abs(unchanged-original).max()),raw_tensors_not_mutated=True,top_only_tcp_mean_mm=float(top_shift.mean()),wrist_only_tcp_mean_mm=float(wrist_shift.mean()),seed=seed,color=result['color'],fixed_flow_seed=42,teacher_repeat_max_abs_deg=repeat,live_state_vs_dataset_max_deg=float(abs(actual_state-batch['observation.state'][0].numpy()).max()),images=pixels,image_only_action_arm_l1_deg=float(abs(live_pred-original)[:,:5].mean()),image_only_tcp_mean_mm=float(diff.mean()),image_only_tcp_max_mm=float(diff.max()),full_live_input_tcp_mean_mm=float(diff_actual.mean()),full_live_input_tcp_max_mm=float(diff_actual.max()))
  rows.append(row);np.savez_compressed(a.out/f'seed-{seed}.npz',dataset=original,top_only=top_pred,wrist_only=wrist_pred,live_images=live_pred,live_state_and_images=actual_pred,expert=ds[int(indices[ep])]['action'].numpy());(a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
 assert not torch.cuda.is_initialized()
if __name__=='__main__':main()
