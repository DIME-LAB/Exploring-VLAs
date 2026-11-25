#!/usr/bin/env python3
"""Frozen Phase3 evaluation manifest, deployment checks and owned episode batches."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

COLORS = ('red', 'green', 'blue')
SEEDS = tuple(range(900000,900030))
BANK_SHA256 = '3b3569389d3301cc26fc0b9ed892c032888b2b4531f97156296f20f079648387'
CHECKPOINTS = (10000,20000,40000,60000)
PARAPHRASES = ('put the {c} brick in the {c} cup', 'move the {c} block to the {c} bin', '{c} lego into the {c} cup please')
REPO = Path(__file__).resolve().parents[3]
CHECKPOINT_ROOT = Path('/mnt/extreme4tb/aaugus11/exploring-vlas/eval/checkpoints/smolvla-phase3-normal-60k-s42-20261114')


def task(color):
    return f'Pick a {color} lego and place it in {color} cup'


def manifest():
    main = [dict(suite='main', seed=seed, color=COLORS[seed%3], phrase='exact', task=task(COLORS[seed%3]), rtc=True) for seed in SEEDS]
    swap = [dict(suite='colour-swap', seed=seed, color=color, phrase='exact', task=task(color), rtc=True) for seed in SEEDS[:6] for color in COLORS]
    para = [dict(suite='paraphrase', seed=seed, color=color, phrase=f'heldout-{i+1}', task=text.format(c=color), rtc=True) for seed in SEEDS[:6] for color in COLORS for i,text in enumerate(PARAPHRASES)]
    return dict(version=1, scene_bank_sha256=BANK_SHA256, inference_mode='eager', action_clock='simulation', fps=30, actions_per_chunk=50, reobserve_after_executed_actions=10,
                rtc=dict(execution_horizon=10,max_guidance_weight=10,schedule='EXP',delay='previous measured prediction sim time + current capture age, ceil(*30), capped at prefix'),
                episode_budget_sim_s=45, wall_watchdog_s=900, state='/joint_states_measured', cameras=['top','wrist'],
                domain=92, localhost_only=True, gpu=0, checkpoint_steps=CHECKPOINTS,
                selection='highest main success count; tie to later checkpoint; incomplete/invalid main set cannot win',
                per_checkpoint=main,
                best_checkpoint_only=[dict(row,rtc=False,suite='main-rtc-off') for row in main]+swap+para,
                language_probe_selection='first six held-out layouts; all three colours from identical saved poses',
                sampling_noise='scene pairing only; policy flow RNG not paired; report this limit')


def validate_checkpoint(path):
    path=Path(path).expanduser().resolve()
    config=json.loads((path/'config.json').read_text())
    if config.get('type')!='smolvla':raise ValueError('Phase3 expects a SmolVLA checkpoint')
    images=[key for key in config['input_features'] if key.startswith('observation.images.')]
    if images != ['observation.images.top','observation.images.wrist']:
        raise ValueError(f'Checkpoint image order differs: {images}')
    for key in images:
        if config['input_features'][key]['shape'] != [3,480,640]:
            raise ValueError('Checkpoint camera geometry differs')
    if config['input_features']['observation.state']['shape'] != [6] or config['output_features']['action']['shape'] != [6]:
        raise ValueError('Checkpoint joint shape differs')
    if config.get('chunk_size') != 50 or config.get('empty_cameras',0) != 0:
        raise ValueError('Checkpoint chunk/camera contract differs')
    for name in ('model.safetensors','policy_preprocessor.json','policy_postprocessor.json'):
        if not (path/name).is_file():
            raise FileNotFoundError(path/name)
    return dict(path=str(path), image_order=images, config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),
                model_sha256=hashlib.sha256((path/'model.safetensors').read_bytes()).hexdigest(),compile_model=config.get('compile_model'))


def validate_bank(bank):
    bank=Path(bank)
    manifest=json.loads((bank/'bank-manifest.json').read_text())
    if manifest['seeds']!=list(SEEDS) or manifest['colors']!=list(COLORS):
        raise ValueError('Frozen bank seed/colour contract differs')
    expected={'baseline.json'}|{f'seed-{seed}.json' for seed in SEEDS}
    if set(manifest['files_sha256'])!=expected:
        raise ValueError('Incomplete frozen scene bank')
    for name,digest in manifest['files_sha256'].items():
        if hashlib.sha256((bank/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Scene bank was mutated: '+name)
    actual=hashlib.sha256((bank/'bank-manifest.json').read_bytes()).hexdigest()
    if actual!=BANK_SHA256:raise ValueError('Bank differs from pinned evaluation protocol')
    return actual


def require_deployment(ros, mcp):
    from rcl_interfaces.srv import GetParameters
    checks={}
    for name in ('controller_manager','arm_controller','gripper_controller'):
        service=ros.node.create_client(GetParameters,f'/{name}/get_parameters')
        try:
            if not service.wait_for_service(timeout_sec=10):
                raise RuntimeError('Missing parameter service: '+name)
            req=GetParameters.Request();req.names=['use_sim_time']
            response=ros.wait(service.call_async(req),10)
            if not response.values or not response.values[0].bool_value:
                raise RuntimeError(name+' must be launched with use_sim_time=true')
            checks[name]=True
        finally:
            ros.node.destroy_client(service)
    native=mcp('run_python',{'code': "import os\nimport omni.usd\nfrom isaacsim.core.api import World\nimport collection_runtime as c\nr=c.runtime()\nresult={'domain':os.environ.get('ROS_DOMAIN_ID'),'physics_dt':r.world.get_physics_dt(),'mode':r.mode,'playing':r.timeline.is_playing(),'rendering_dt':r.world.get_rendering_dt(),'measured_topic':'/joint_states_measured'}"})
    if native['domain']!='92' or native['mode']!='deploy' or not native['playing'] or abs(native['rendering_dt']-1/30)>1e-8 or abs(native['physics_dt']-1/120)>1e-8:
        raise RuntimeError('Phase3 scene clock contract differs: '+repr(native))
    return dict(controllers_sim_time=checks,native=native)


def new_score(color,initial):
    path=REPO/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/compute_workspace.py'
    spec=importlib.util.spec_from_file_location('eval_fk',path);fk=importlib.util.module_from_spec(spec);spec.loader.exec_module(fk)
    from phase3_metrics import PhysicalScore
    return PhysicalScore(color,initial,fk)


def command(row, checkpoint, out, bank, max_s=45):
    cameras={key:dict(type='ros2',topic=topic,encoding='rgb8',width=640,height=480,fps=30) for key,topic in
             (('top','/workspace_camera_sim'),('wrist','/wrist_camera_rgb_sim'))}
    cmd=[str(Path.home()/'.pixi/bin/pixi'),'run','--manifest-path',str(REPO/'linux-env/pixi.toml'),'python3','-u',
         str(Path(__file__).with_name('run_episodes.py')),'--sim-time','--fps','30','--action-duration-s',str(1/30),
         '--joint-states-topic','/joint_states_measured','--cameras-json',json.dumps(cameras),'--pretrained',str(checkpoint),
         '--color',row['color'],'--task',row['task'],'--episodes','1','--seeds',str(row['seed']),
         '--sim-rate-hz','30','--max-s',str(max_s),'--actions-per-chunk','50','--chunk-size-threshold','.8',
         '--rtc.execution-horizon','10','--rtc.max-guidance-weight','10','--server-port','18092',
         '--server-startup-timeout','900','--scene-bank',str(bank),'--out',str(out),
         '--condition',f"{row['suite']}-{row['color']}-{row['phrase']}-rtc-{'on' if row['rtc'] else 'off'}"]
    if not row['rtc']:cmd.append('--no-rtc')
    return cmd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('plan','check','run','freeze-bank'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scene-bank',type=Path,default=Path('/mnt/extreme4tb/aaugus11/exploring-vlas/eval/scene-bank-v1'))
    p.add_argument('--suite',choices=('main','main-rtc-off','colour-swap','paraphrase'),default='main')
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--count',type=int,default=1,help='Consecutive manifest rows in this announced window')
    p.add_argument('--max-s',type=float,default=45)
    args=p.parse_args()
    plan=manifest()
    if args.operation=='freeze-bank':
        from run_episodes import Ros,mcp
        os.environ['CYCLONEDDS_URI']=f'file://{REPO}/linux-env/cyclonedds-eval.xml'
        os.environ['RMW_IMPLEMENTATION']='rmw_cyclonedds_cpp'
        for name in ('ROS_LOCALHOST_ONLY','ROS_AUTOMATIC_DISCOVERY_RANGE','ROS_STATIC_PEERS'):os.environ.pop(name,None)
        if os.environ.get('ROS_DOMAIN_ID')!='92':p.error('Explicit domain92 required')
        ros=Ros(True,'/joint_states_measured')
        try:
            import time
            deadline=time.monotonic()+15
            while ros.physical is None:
                if time.monotonic()>deadline:raise TimeoutError('Missing physical snapshot feed')
                time.sleep(.05)
            ros.home()
            code=f'import sys; sys.path.insert(0,{str(Path(__file__).parent)!r}); import scene_reset; result=scene_reset.freeze({str(args.scene_bank)!r})'
            proof=mcp('run_python',{'code':code})
            args.out.mkdir(parents=True,exist_ok=True);(args.out/'bank-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
        finally:ros.close()
        return
    if args.operation=='plan':
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
        print(json.dumps({'main_per_checkpoint':len(plan['per_checkpoint']),'best_only':len(plan['best_checkpoint_only'])}))
        return
    if args.checkpoint is None:p.error('--checkpoint required')
    proof=validate_checkpoint(args.checkpoint)
    if args.operation=='check':
        args.out.mkdir(parents=True,exist_ok=True);(args.out/'checkpoint-contract.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof));return
    rows=[row for row in plan['per_checkpoint']+plan['best_checkpoint_only'] if row['suite']==args.suite]
    if not 0<=args.index<len(rows):p.error('--index out of range')
    if os.environ.get('ROS_DOMAIN_ID')!='92':
        p.error('Live run requires explicit domain92; eval XML confines DDS to loopback')
    env=dict(os.environ,SOARM_EVAL_INFERENCE_MODE='eager',PYTHONDONTWRITEBYTECODE='1',CYCLONEDDS_URI=f'file://{REPO}/linux-env/cyclonedds-eval.xml',RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',SOARM_POLICY_GPU='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    selected=rows[args.index:args.index+args.count]
    if args.count<1 or len(selected)!=args.count:p.error('Window extends beyond suite')
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'checkpoint-contract.json').write_text(json.dumps(proof,indent=2)+'\n')
    receipt=args.out/'episode-manifest.json';receipt.write_text(json.dumps(selected,indent=2)+'\n')
    cmd=command(selected[0],args.checkpoint,args.out,args.scene_bank,args.max_s)
    cmd[cmd.index('--episodes')+1]=str(len(selected))
    cmd[cmd.index('--seeds')+1]=','.join(str(row['seed']) for row in selected)
    cmd+=['--episode-manifest',str(receipt)]
    cache=Path('/mnt/extreme4tb/aaugus11/exploring-vlas/eval/cache')
    for name,part in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),('TORCH_HOME','torch'),('TMPDIR','tmp'),('HF_HOME','huggingface'),('ROS_LOG_DIR','ros-logs')]:
        directory=cache/part;directory.mkdir(parents=True,exist_ok=True);env[name]=str(directory)
    for name in ('ROS_LOCALHOST_ONLY','ROS_AUTOMATIC_DISCOVERY_RANGE','ROS_STATIC_PEERS'):env.pop(name,None)
    subprocess.run(cmd,env=env,check=True)


if __name__=='__main__':main()
