"""Exercise the full audit on explicit synthetic no-motion probe records."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from behaviour import fk
from direction_metrics import motion_directions
from report_directions import ANCHORS,COLORS,SEEDS

class ProbeReportTests(unittest.TestCase):
 def test_all_low_motion_and_missing_noise_refusal(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);key='unfreeze30k+2500';folder=root/key;folder.mkdir();model='/mnt/a4500-backup/exploring-vlas/eval/checkpoints/unfreeze30k/002500/pretrained_model';digest='a'*64
   state=np.zeros(6,dtype=np.float32);home=fk.forward_kinematics_batch(np.zeros((1,5)))[0];pred=np.zeros((50,6),dtype=np.float32);tcp=np.tile(home,(50,1));base=dict(position=[0,0,0],orientation=[1,0,0,0]);positions={c+'_'+str(i):(home+np.array([(.1 if i==0 else -.1),j*.07+.04,-.03])).tolist() for j,c in enumerate(COLORS) for i in range(2)};rows=[]
   for kind,seed in sorted(ANCHORS):
    for noise in SEEDS:
     prompts={};arrays={}
     for i,c in enumerate(COLORS):
      tokens=[0]*10;tokens[2]=tokens[9]=i+1
      prompts[c]=dict(task=f'Pick a {c} lego and place it in {c} cup',tokens=tokens,token_mask=[True]*10,normalized_state=[0]*6,geometry=motion_directions(home,tcp,positions,c),displacement_world_m=[0]*3)
      arrays['pred_'+c]=pred;arrays['tcp_'+c]=tcp
     pairs=[dict(prompts=[l,r],changed_token_positions=[2,9],arm_l1_deg=0,tcp_endpoint_shift_mm=0) for l,r in [('red','green'),('red','blue'),('green','blue')]]
     rows.append(dict(kind=kind,seed=seed,sampling_seed=noise,anchor_frame=0,pixels_and_state_fixed=True,input_sha256={'synthetic':'b'*64},state_degrees=state.tolist(),home_tcp_world_m=home.tolist(),block_positions_world_m=positions,identical_prompt_repeat_max_abs_deg=0 if len(rows)==0 else None,source_prompt_control_max_abs_deg=0 if noise==42 else None,native_camera_order=['observation.images.top','observation.images.wrist'],robot_base_world_pose=base,expert_target_body='red_0',prompts=prompts,pairs=pairs))
     np.savez_compressed(folder/f'colour-swap-{kind}-{seed}-noise-{noise}.npz',**arrays,state=state,truth=pred)
   doc=dict(checkpoint_step=2500,checkpoint_path=model,checkpoint_label=key,weights_sha256=digest,flow_noise_seeds=SEEDS,case_manifest_sha256='c'*64,results=rows);file=folder/'colour-swap.json';file.write_text(json.dumps(doc));manifest=root/'expected.json';manifest.write_text(json.dumps({key:dict(step=2500,path=model,sha256=digest)}))
   cmd=[sys.executable,str(Path(__file__).with_name('report_directions.py')),str(root),'--models',key,'--expected-models',str(manifest),'--reference-20k',str(root/'unused-reference')]
   result=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
   summary=json.loads((root/'summary.json').read_text())['checkpoints'][key];self.assertEqual(summary['n_chunks'],90);self.assertEqual(summary['low_horizontal_n'],90);self.assertEqual(summary['classified_xy_n'],0);self.assertIsNone(summary['mean_direction_resultant']);self.assertEqual(summary['checkpoint_label'],key)
   doc['results']=rows[:-1];file.write_text(json.dumps(doc));result=subprocess.run(cmd,capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertIn('Expected exactly 30 unique anchor/noise records',result.stderr)
if __name__=='__main__':unittest.main()
