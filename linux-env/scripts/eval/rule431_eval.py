"""Rule431 CPU preparation and explicit announced-window plans; no auto GPU queue."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT='/mnt/extreme4tb/aaugus11/exploring-vlas/eval'
REPO='/mnt/extreme4tb/aaugus11/exploring-vlas/eval-native-rule431'
SOURCE='/mnt/storage3/aaugus11/smolvla-train/phase3.2/rule431-base25k/training/checkpoints'
STEPS=(5000,10000,15000,20000,25000)
REFERENCE=ROOT+'/rule431/contracts/input-contract.json'
BANKS={
 'rule15':('rule-seen15-v1','b4bc43f02af3d727bb4f55febc76da0cf7a61813027e7ad042de2b15e2d2383d',None),
 'seen5':('training-home-v1','dbae9bdc10cc5755d6aeba9a6038de3b778614572c6950afa5d258f61502b2d6',None),
 'mixed15':('training-home-15-v1','94ff8a452f9441eb2d5c742e39ad6b65321c0a54986b83cc579fb4b42a07b582',None),
 'held0':('scene-bank-v2','cedd3f3db23170abfa9f53069c7f4f0c9cab11e0686ed0746a1024e3b4ec1fe3',0),
 'held1':('scene-bank-v2','cedd3f3db23170abfa9f53069c7f4f0c9cab11e0686ed0746a1024e3b4ec1fe3',15)}
POLICY_PYTHON=ROOT+'/prep33/dual-policy-python.sh'

def paths(step,revision='v1'):
    import re
    if step not in STEPS or not re.fullmatch(r'[a-z0-9-]+',revision):raise ValueError('Invalid rule431 step/revision')
    tag=f'{step:06d}'
    return dict(model=ROOT+f'/checkpoints/rule431-base25k/{tag}/pretrained_model',
                handoff=ROOT+f'/checkpoints/rule431-base25k/{tag}/handoff.json',
                prep=ROOT+f'/runs/r431-cpu-{tag}-{revision}',tag=tag)

def source_preflight_args(step,suite,out,revision='v1',smoke=False,readiness_budget_s=60):
    p=paths(step,revision);bank,sha,offset=BANKS[suite]
    args=['python3','-B',REPO+'/linux-env/scripts/eval/native_policy.py','--check-only',
          '--collector-matched-reset','--reset-collector-root',ROOT+'/../eval-bank-bf7336d',
          '--checkpoint',p['model'],'--bank',ROOT+'/'+bank,'--bank-sha256',sha,
          '--out',out,'--grasp-telemetry','--rule-target','--retain-wrist-before-closure',
          '--processor-reference',REFERENCE,'--paired-flow-repeat','0','--readiness-budget-s',str(readiness_budget_s)]
    if offset is not None:args+=['--bank-offset',str(offset),'--bank-count','15']
    if smoke:args+=['--episode-seeds','530002']
    return args

def window_plan(mode,suite,stop,step=None,revision='v1',attempt=1,weights_sha256=None,qualification_attempt=1,readiness_budget_s=60):
    if suite not in BANKS or attempt<1 or qualification_attempt<1:raise ValueError('Unknown suite/attempt')
    datetime.datetime.strptime(stop,'%Y-%m-%dT%H:%M:%SZ')
    if readiness_budget_s not in (60,180):raise ValueError('Readiness budget must be60 or explicitly approved180')
    bank,sha,offset=BANKS[suite]
    owner=REPO+'/linux-env/scripts/eval/run_native_replay_window.py'
    args=['python3','-B','-u',owner,'--bank',ROOT+'/'+bank,'--bank-sha256',sha,
          '--reset-collector-root',ROOT+'/../eval-bank-bf7336d','--stop-by-utc',stop,'--readiness-budget-s',str(readiness_budget_s)]
    if mode=='qualify':
        if attempt!=qualification_attempt:raise ValueError('Qualification attempt must match its bootstrap qualification version')
        if suite not in ('seen5','rule15','held0'):raise ValueError('Qualification uses full seen5/rule15/held30 bank')
        root=ROOT+f'/runs/r431-q-{suite}-v{attempt}'
        args+=['--matched-reset-qualification','--out',root]
        if suite=='held0':args+=['--bootstrap-control',ROOT+f'/runs/r431-q-seen5-v{qualification_attempt}']
    elif mode in ('score','smoke'):
        if suite not in ('rule15','held0','held1') or step not in STEPS:raise ValueError('Scoring needs one rule431 checkpoint and a fixed suite')
        if mode=='smoke' and suite!='rule15':raise ValueError('Smoke uses the first rule-seen layout only')
        import re
        if not isinstance(weights_sha256,str) or re.fullmatch(r'[0-9a-f]{64}',weights_sha256) is None:raise ValueError('A verified staged weights SHA is required')
        p=paths(step,revision);suffix='smoke' if mode=='smoke' else suite
        root=ROOT+f"/runs/r431-{p['tag']}-{suffix}-r0-v{attempt}"
        q=ROOT+(f'/runs/r431-q-held0-v{qualification_attempt}' if offset is not None else f'/runs/r431-q-rule15-v{qualification_attempt}')
        args+=['--out',root,'--matched-reset-diagnostic','--matched-reset-arms','default',
               '--checkpoint',p['model'],'--weights-sha256',weights_sha256,'--checkpoint-handoff',p['handoff'],
               '--processor-proof',ROOT+f"/rule431/proofs/{p['tag']}/processor-proof.json",
               '--processor-reference',REFERENCE,'--reset-qualification',q,
               '--cpu-rpc-proof',p['prep']+'/rpc/result.json',
               '--source-preflight-proof',p['prep']+f'/{suffix}/preflight.json',
               '--policy-python',POLICY_PYTHON,'--paired-flow-repeat','0',
               '--grasp-telemetry','--rule-target','--retain-wrist-before-closure']
        if offset is not None:args+=['--bank-offset',str(offset),'--bank-count','15']
        if mode=='smoke':args+=['--episode-seeds','530002','--unscored-policy-smoke']
    else:raise ValueError('Unknown window mode')
    if len((root+'/default/rpc.sock').encode())>=104:raise ValueError('Native IPC path too long')
    return dict(kind='rule431-window-plan-v1',mode=mode,suite=suite,step=step,revision=revision,
                attempt=attempt,root=root,stop_by_utc=stop,readiness_budget_s=readiness_budget_s,gpu=0,domain=92,ports=[8781,8782,18092],
                argv=args,execution='NOT STARTED; requires lead GO, ABI and fresh60sGPU0-clear check',
                historical_requested_colour_first_close_noise_sd_mm=29.878698518817213,
                noise_limit='Descriptive prior on5mixedlayouts, not a calibrated bound for the new rule-target metric/cohort')

def remote(code):
    return subprocess.check_output(['ssh','dual-a4500','env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 -B -'],input=code,text=True)

def prepare(step,processor_proof,local_state,revision,readiness_budget_s=60):
    p=paths(step,revision);local_state.mkdir(parents=True,exist_ok=True)
    receipt=json.loads(subprocess.check_output(['ssh','dual-a4500','cat '+p['handoff']],text=True))
    sys.path.insert(0,str(Path(__file__).parent))
    from checkpoint_identity import validate_handoff_identity
    validate_handoff_identity(receipt,p['model'],step,SOURCE)
    if receipt['source_before']['completion']['artifact_marker']['name']!='rule431-manifest.json':raise ValueError('Wrong model family completion marker')
    target=ROOT+f"/rule431/proofs/{p['tag']}/processor-proof.json"
    remote("from pathlib import Path\np=Path("+repr(target)+");p.parent.mkdir(parents=True,exist_ok=True)\n")
    subprocess.run(['scp',str(processor_proof),'dual-a4500:'+target],check=True)
    verify="""import sys,json
from pathlib import Path
sys.path.insert(0,REPO+'/linux-env/scripts/eval')
from checkpoint_contract import require_handoff,require_processor_proof
p=PATHS;h=require_handoff(p['model'],p['handoff'],WEIGHTS,published=True)
proof=require_processor_proof(Path(json.loads(Path(p['handoff']).read_text())['source_before']['model']),PROOF,h['files'],published_handoff=p['handoff'],reference_path=REFERENCE)
print(json.dumps(dict(status='PASS',schema=proof['schema'],weights_sha256=WEIGHTS)))
""".replace('REPO',repr(REPO)).replace('PATHS',repr(p)).replace('WEIGHTS',repr(receipt['source_before']['files']['model.safetensors']['sha256'])).replace('PROOF',repr(target)).replace('REFERENCE',repr(REFERENCE))
    contract=json.loads(remote(verify));(local_state/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    weights=contract['weights_sha256']
    env=['env','-u','SOARM_EVAL_TORCH_MEMORY_GIB','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','TMPDIR='+ROOT+'/tmp','HF_HOME='+ROOT+'/cache/huggingface','XDG_CACHE_HOME='+ROOT+'/cache','nice','-n','10']
    cmd=env+[POLICY_PYTHON,REPO+'/linux-env/scripts/eval/cpu_rpc_smoke.py','--checkpoint',p['model'],'--out',p['prep']+'/rpc','--paired-flow-repeat','0']
    with (local_state/'rpc.log').open('w') as log:subprocess.run(['ssh','dual-a4500',shlex.join(cmd)],stdout=log,stderr=subprocess.STDOUT,check=True)
    for suite in ('rule15','held0','held1','smoke'):
        argv=source_preflight_args(step,'rule15' if suite=='smoke' else suite,p['prep']+'/'+suite,revision,smoke=suite=='smoke',readiness_budget_s=readiness_budget_s)+['--weights-sha256',weights]
        shell='source /opt/ros/humble/setup.bash && source '+shlex.quote(REPO+'/vla_SO-ARM101/install/setup.bash')+' && unset ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS && '+shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','ROS_DOMAIN_ID=92','SOARM_KIT_MCP_PORT=8781','SOARM_COLLECTION_ROOT=/mnt/extreme4tb/aaugus11/exploring-vlas','RMW_IMPLEMENTATION=rmw_cyclonedds_cpp','CYCLONEDDS_URI=file://'+REPO+'/linux-env/cyclonedds.xml','ROS_LOG_DIR='+ROOT+'/ros-logs','TMPDIR='+ROOT+'/tmp',*argv])
        with (local_state/(suite+'.log')).open('w') as log:subprocess.run(['ssh','dual-a4500','bash -c '+shlex.quote(shell)],stdout=log,stderr=subprocess.STDOUT,check=True)
    summary=dict(status='CPU_READY_NOT_LIVE_QUALIFIED',step=step,weights_sha256=weights,paths=p,
                 readiness_budget_s=readiness_budget_s,processor_reference=REFERENCE,preset=(p['model'],'','{}'),preset_name=f'rule431-{step//1000}k')
    (local_state/'ready.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--step',type=int,choices=STEPS,required=True);a.add_argument('--processor-proof',type=Path,required=True);a.add_argument('--local-state',type=Path,required=True);a.add_argument('--revision',default='v1');a.add_argument('--readiness-budget-s',type=float,choices=(60.,180.),default=60.)
    a=sub.add_parser('plan');a.add_argument('--mode',choices=('qualify','score','smoke'),required=True);a.add_argument('--suite',choices=tuple(BANKS),required=True);a.add_argument('--step',type=int,choices=STEPS);a.add_argument('--stop-by-utc',required=True);a.add_argument('--revision',default='v1');a.add_argument('--readiness-budget-s',type=float,choices=(60.,180.),default=60.);a.add_argument('--attempt',type=int,default=1);a.add_argument('--qualification-attempt',type=int,default=1)
    args=p.parse_args()
    if args.command=='prepare':prepare(args.step,args.processor_proof,args.local_state,args.revision,args.readiness_budget_s)
    else:
        weights=None
        if args.mode!='qualify':
            p=paths(args.step,args.revision);receipt=json.loads(subprocess.check_output(['ssh','dual-a4500','cat '+p['handoff']],text=True))
            from checkpoint_identity import validate_handoff_identity
            validate_handoff_identity(receipt,p['model'],args.step,SOURCE)
            weights=receipt['source_before']['files']['model.safetensors']['sha256']
        print(json.dumps(window_plan(args.mode,args.suite,args.stop_by_utc,args.step,args.revision,args.attempt,weights,args.qualification_attempt,args.readiness_budget_s),indent=2))

if __name__=='__main__':main()
