"""CPU saved-pipeline fixture check; no policy weights and no simulation."""
from pathlib import Path
import argparse,json,os
import numpy as np
import torch
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import batch_to_transition,transition_to_batch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available() and not torch.cuda.is_initialized()
torch.set_num_threads(4);torch.set_num_interop_threads(1)
processor=PolicyProcessorPipeline.from_pretrained(pretrained_model_name_or_path=str(a.checkpoint),config_filename='policy_preprocessor.json',overrides={'device_processor':{'device':'cpu'},'rename_observations_processor':{'rename_map':{}}},to_transition=batch_to_transition,to_output=transition_to_batch)
z=np.load(a.checkpoint/'inputs.npz');payload=json.loads((a.checkpoint/'payload.json').read_text())
top=torch.from_numpy(z['top'].copy()).permute(2,0,1).float()/255;wrist=torch.from_numpy(z['wrist'].copy()).permute(2,0,1).float()/255
batch={'observation.images.top':top,'observation.images.wrist':wrist,'observation.state':torch.from_numpy(z['state'].copy()),'task':payload['task']}
result=processor(batch);assert torch.equal(result['observation.images.top'][0],top[:,128:473,99:559]) and torch.equal(result['observation.images.wrist'][0],wrist)
try:processor(dict(batch,**{'observation.images.top':top[:,128:473,99:559]}))
except ValueError as e:assert 'Do not crop twice' in str(e)
else:raise AssertionError('Double crop was accepted')
proof=dict(status='PASS',scope='Saved crop fixture pipeline only; actual B training checkpoint still requires observer parity and CPU RPC',checkpoint=str(a.checkpoint),top_shape=list(result['observation.images.top'].shape),wrist_shape=list(result['observation.images.wrist'].shape),top_pixel_exact=True,wrist_bit_exact=True,double_crop_refused=True,cuda_initialized=torch.cuda.is_initialized(),pythonpath=os.environ.get('PYTHONPATH'))
a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
