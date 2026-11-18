"""CPU saved-pipeline fixture check; no policy weights and no simulation."""
from pathlib import Path
import argparse,json,os,hashlib
import numpy as np
import torch
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import batch_to_transition,transition_to_batch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',type=Path);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available() and not torch.cuda.is_initialized()
torch.set_num_threads(1 if a.reference else 4);torch.set_num_interop_threads(1)
processor=PolicyProcessorPipeline.from_pretrained(pretrained_model_name_or_path=str(a.checkpoint),config_filename='policy_preprocessor.json',overrides={'device_processor':{'device':'cpu'},'rename_observations_processor':{'rename_map':{}},'normalizer_processor':{'device':'cpu'}},to_transition=batch_to_transition,to_output=transition_to_batch)
z=np.load(a.checkpoint/'inputs.npz');payload=json.loads((a.checkpoint/'payload.json').read_text())
top=torch.from_numpy(z['top'].copy()).permute(2,0,1).float()/255;wrist=torch.from_numpy(z['wrist'].copy()).permute(2,0,1).float()/255
batch={'observation.images.top':top,'observation.images.wrist':wrist,'observation.state':torch.from_numpy(z['state'].copy()),'task':payload['task']}
result=processor(batch);assert torch.equal(result['observation.images.top'][0],top[:,128:473,99:559]) and torch.equal(result['observation.images.wrist'][0],wrist)
try:processor(dict(batch,**{'observation.images.top':top[:,128:473,99:559]}))
except ValueError as e:assert 'Do not crop twice' in str(e)
else:raise AssertionError('Double crop was accepted')
proof=dict(status='PASS',fixture_only=True,actual_training_checkpoint=False,scope='Saved processor fixture only; actual checkpoint/RPC/live readiness not established',checkpoint=str(a.checkpoint),top_shape=list(result['observation.images.top'].shape),wrist_shape=list(result['observation.images.wrist'].shape),top_pixel_exact=True,wrist_bit_exact=True,double_crop_refused=True,cuda_initialized=torch.cuda.is_initialized(),pythonpath=os.environ.get('PYTHONPATH'))
if a.reference:
 from lerobot.async_inference.helpers import raw_observation_to_observation
 from lerobot.configs import PolicyFeature,FeatureType
 from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
 features={'observation.state':dict(dtype='float32',shape=(6,),names=payload['joints'])}
 images={}
 raw={key:float(z['state'][i]) for i,key in enumerate(payload['joints'])}
 for key in ('top','wrist'):
  name='observation.images.'+key;features[name]=dict(dtype='image',shape=(480,640,3),names=['height','width','channels']);images[name]=PolicyFeature(type=FeatureType.VISUAL,shape=(3,480,640));raw[key]=z[key]
 raw['task']=payload['task']
 prepared=processor(raw_observation_to_observation(raw,features,images))
 def digest(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
 hashes={key:digest(prepared[key]) for key in ('observation.images.top','observation.images.wrist','observation.language.tokens','observation.language.attention_mask')}
 hashes['observation.state']=digest(prepared['observation.state'].reshape(-1,6))
 hashes['top_after_resize']=digest(resize_with_pad(prepared['observation.images.top'],512,512,pad_value=0)*2-1)
 expected=json.loads(a.reference.read_text())['hashes']
 if hashes!=expected:raise ValueError('Eval saved-processor hashes differ from training reference: '+json.dumps({key:dict(actual=value,expected=expected.get(key)) for key,value in hashes.items() if expected.get(key)!=value}))
 proof['pixel_state_token_resize_hashes']=hashes
 proof['reference_cpu_threads']=torch.get_num_threads()
 assert not torch.cuda.is_initialized()
a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
