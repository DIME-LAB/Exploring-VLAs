"""CPU decoder parity and same-frame GOP sensitivity controls."""
from pathlib import Path
import argparse,hashlib,json,os
import av,numpy as np,torch
from torchcodec.decoders import VideoDecoder
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from PIL import Image
from codec_pair import export_encode,cv_frames
import importlib.util
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists();a.out.mkdir()
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_available() and not torch.cuda.is_initialized();torch.set_num_threads(4);torch.set_num_interop_threads(1)
config=PreTrainedConfig.from_pretrained(a.checkpoint);config.device='cpu';config.compile_model=False;config.rtc_config=None
policy=SmolVLAPolicy.from_pretrained(a.checkpoint,config=config).to('cpu').eval()
pre,post=make_pre_post_processors(policy.config,pretrained_path=a.checkpoint,preprocessor_overrides={'device_processor':{'device':'cpu'},'rename_observations_processor':{'rename_map':{}}},postprocessor_overrides={'device_processor':{'device':'cpu'}})
root=Path(__file__).resolve().parents[3];spec=importlib.util.spec_from_file_location('gop_fk',root/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/compute_workspace.py');fk=importlib.util.module_from_spec(spec);spec.loader.exec_module(fk)
bank=Path('/mnt/storage3/aaugus11/exploring-vlas/eval/training-home-v1');rows=[]
for seed in range(530000,530005):
 d=a.root/f'seed-{seed}';state=json.loads((bank/f'state-{seed}.json').read_text());batches={v:{'observation.state':torch.tensor([state['state']],dtype=torch.float32)*180/np.pi,'task':[state['task']]} for v in ('frame0I','frame1P','singleI')};parity={}
 for camera in ('top','wrist'):
  for suffix in ('source-export','live-export','live-both'):
   path=d/f'{camera}-{suffix}.mp4';decoder=VideoDecoder(str(path),device='cpu',num_ffmpeg_threads=1);tc=decoder.get_frames_at(indices=[0,1]).data.permute(0,2,3,1).numpy()
   with av.open(str(path)) as container:frames=list(container.decode(video=0))[:2];pv=np.stack([f.to_ndarray(format='rgb24') for f in frames])
   assert np.array_equal(tc,pv),f'Decoder mismatch {path}';parity[f'{camera}-{suffix}']=True
  image=np.asarray(Image.open(d/f'{camera}-raw_live.png'))
  # A single-frame export vs frame0/1 in a g2 sequence after the same collector encoding.
  stage1=d/f'{camera}-live-collector.mp4'
  collector_first=next(cv_frames(stage1));single,info=export_encode(iter([collector_first]),a.out/f'{seed}-{camera}-single.mp4')
  with av.open(str(d/f'{camera}-live-both.mp4')) as container:frames=list(container.decode(video=0))[:2]
  for v,img in [('frame0I',frames[0].to_ndarray(format='rgb24')),('frame1P',frames[1].to_ndarray(format='rgb24')),('singleI',single)]:
   batches[v]['observation.images.'+camera]=torch.from_numpy(img.copy()).permute(2,0,1)[None].float()/255
 predictions={}
 for v,b in batches.items():
  policy.reset();torch.manual_seed(42)
  with torch.inference_mode():predictions[v]=post(policy.predict_action_chunk(pre(b)).flatten(0,1)).reshape(1,50,6)[0].numpy()
 tcp={k:fk.forward_kinematics_batch(np.radians(v[:,:5])) for k,v in predictions.items()};effects={}
 for v in ('frame1P','singleI'):
  shift=np.linalg.norm(tcp[v]-tcp['frame0I'],axis=1)*1000;effects[v]=dict(mean_mm=float(shift.mean()),max_mm=float(shift.max()),first10_mean_mm=float(shift[:10].mean()),arm_l1_deg=float(abs(predictions[v]-predictions['frame0I'])[:,:5].mean()))
 row=dict(seed=seed,decoder_parity=parity,effects_vs_g2_frame0=effects);rows.append(row);(a.out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
assert not torch.cuda.is_initialized()
