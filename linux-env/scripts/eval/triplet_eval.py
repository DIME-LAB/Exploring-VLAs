"""Triplet checkpoint CPU gates and native owner plans; never launches a GPU."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from triplet_bank import sha, load, SUITE, write_json
from native_bank import select_policy_rows,parse_episode_seeds

BASE = Path('/mnt/extreme4tb/aaugus11/exploring-vlas')
ROOT = BASE/'eval/triplet360'
REPO = BASE/'eval-native-triplet360'
BANK = ROOT/'held20-v2'
BANK_SHA = '0ac0d8928a960ce0aa22670f2d75c33bbc9a2b6717ba3f0fc076bec652abab0d'
SOURCE = '/mnt/storage3/aaugus11/smolvla-train/phase3.2/triplet360-base3ep/training/checkpoints'
STEPS = (5000,10000,15000,20000,20825)
REFERENCE = ROOT/'contracts/input-contract.json'
PYTHON = BASE/'eval/prep33/dual-policy-python.sh'


def paths(step,revision="v1"):
    if not re.fullmatch('[a-z0-9-]+',revision):raise ValueError('Invalid CPU revision')
    if step not in STEPS:
        raise ValueError('Unknown triplet checkpoint step')
    checkpoint=BASE/f'eval/checkpoints/triplet360-base3ep/{step:06d}'
    return dict(model=checkpoint/'pretrained_model',handoff=checkpoint/'handoff.json',
                proof=ROOT/f'proofs/{step:06d}/processor-proof.json',prep=BASE/f'eval/runs/t360-cpu-{step:06d}-{revision}')


def native_args(step,offset,out,weights,episode_seeds=None):
    if offset not in (0,15,30,45):raise ValueError('Use fixed fifteen-case chunks')
    p=paths(step)
    result=['python3','-B',str(REPO/'linux-env/scripts/eval/native_policy.py'),
        '--collector-matched-reset','--reset-collector-root',str(REPO),
        '--checkpoint',str(p['model']),'--weights-sha256',weights,
        '--bank',str(BANK),'--bank-sha256',BANK_SHA,'--bank-offset',str(offset),'--bank-count','15',
        '--out',str(out),'--grasp-telemetry','--rule-target','--retain-wrist-before-closure',
        '--processor-reference',str(REFERENCE),'--paired-flow-repeat','0','--readiness-budget-s','180']
    if episode_seeds is not None:result+=['--episode-seeds',','.join(map(str,episode_seeds))]
    return result


def plan(mode,step,offset,stop,weights=None,attempt=1,qualification_attempt=1,cpu_revision="v1",episode_seeds=None):
    import datetime
    datetime.datetime.strptime(stop,'%Y-%m-%dT%H:%M:%SZ')
    if attempt<1 or qualification_attempt<1:raise ValueError('Attempt must be positive')
    owner=REPO/'linux-env/scripts/eval/run_native_replay_window.py'
    args=['python3','-B','-u',str(owner),'--bank',str(BANK),'--bank-sha256',BANK_SHA,
          '--reset-collector-root',str(REPO),'--stop-by-utc',stop,'--readiness-budget-s','180']
    if mode=='qualify':
        root=BASE/f'eval/runs/t360-q-v{attempt}'
        args+=['--out',str(root),'--matched-reset-qualification']
    elif mode=='score':
        p=paths(step,cpu_revision)
        if offset not in (0,15,30,45) or not re.fullmatch('[0-9a-f]{64}',weights or ''):
            raise ValueError('Score plan requires fixed chunk and actual weights SHA')
        root=BASE/f'eval/runs/t{step:06d}-h{offset//15}-r0-v{attempt}'
        args+=['--out',str(root),'--matched-reset-diagnostic','--matched-reset-arms','default',
          '--checkpoint',str(p['model']),'--weights-sha256',weights,'--checkpoint-handoff',str(p['handoff']),
          '--processor-proof',str(p['proof']),'--processor-reference',str(REFERENCE),
          '--reset-qualification',str(BASE/f'eval/runs/t360-q-v{qualification_attempt}'),
          '--cpu-rpc-proof',str(p['prep']/'rpc/result.json'),
          '--source-preflight-proof',str(p['prep']/f'h{offset//15}/preflight.json'),
          '--policy-python',str(PYTHON),'--paired-flow-repeat','0','--grasp-telemetry','--rule-target',
          '--retain-wrist-before-closure','--bank-offset',str(offset),'--bank-count','15']
    else:raise ValueError('Unknown plan mode')
    if episode_seeds is not None:
        if mode!='score' or not episode_seeds or len(set(episode_seeds))!=len(episode_seeds):raise ValueError('Recovery requires a unique ordered score subset')
        args+=['--episode-seeds',','.join(map(str,episode_seeds))]
    if len(str(root/'default/rpc.sock').encode())>=104:raise ValueError('IPC path too long')
    return dict(kind='triplet360-window-plan-v1',mode=mode,step=step,offset=offset,root=str(root),
        stop_by_utc=stop,argv=args,gpu=0,domain=92,ports=[8781,8782,18092],
        state='NOT STARTED; requires actual checkpoint, independent review, fresh live reset qualification and lead GO',
        free_card='Strict ABI; extended hostlock/Docker/host-process/NVML check; other-side-first; 60 seconds GPU0 clear',
        reset_protocol='collector-recorded-triplet-v1; no sampler; producer prime20/restore/bootstrap1, then recorder-equivalent first tick')


def prepare(step,revision="v1",offset=None,episode_seeds=None):
    """Use the existing contract, production CPU RPC, and native check-only entrances."""
    from checkpoint_contract import require_handoff,require_processor_proof
    from checkpoint_identity import validate_handoff_identity
    p=paths(step,revision);receipt=json.loads(p['handoff'].read_text())
    validate_handoff_identity(receipt,str(p['model']),step,SOURCE)
    weights=receipt['source_before']['files']['model.safetensors']['sha256']
    h=require_handoff(p['model'],p['handoff'],weights,published=True)
    if 'triplet360-manifest.json' not in h['files'] or 'rule431-manifest.json' in h['files']:
        raise ValueError('Wrong checkpoint family')
    proof=require_processor_proof(receipt['source_before']['model'],p['proof'],h['files'],published_handoff=p['handoff'],reference_path=REFERENCE)
    all_rows,suite=load(BANK,BANK_SHA)
    if episode_seeds is not None and offset is None:raise ValueError('Recovery CPU preflight requires its fixed chunk offset')
    offsets=(0,15,30,45) if offset is None else (offset,)
    selected={str(value):[r['seed'] for r in select_policy_rows(all_rows,suite,value,15,episode_seeds)] for value in offsets}
    p['prep'].mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(BASE/'eval/tmp'))
    cache=BASE/'eval/cache'
    env.update(HF_HOME=str(cache/'huggingface'),HF_HUB_CACHE=str(cache/'huggingface/hub'),XDG_CACHE_HOME=str(cache),TORCH_HOME=str(cache/'torch'),HF_DATASETS_CACHE=str(cache/'datasets-triplet360'))
    env.pop('SOARM_EVAL_TORCH_MEMORY_GIB',None)
    subprocess.run([str(PYTHON),str(REPO/'linux-env/scripts/eval/cpu_rpc_smoke.py'),
                    '--checkpoint',str(p['model']),'--out',str(p['prep']/'rpc'),'--paired-flow-repeat','0'],env=env,check=True)
    for value in offsets:
        subprocess.run(native_args(step,value,p['prep']/f'h{value//15}',weights,episode_seeds)+['--check-only'],env=env,check=True)
    result=dict(status='CPU_CHECKPOINT_READY_LIVE_QUALIFICATION_REQUIRED',step=step,weights_sha256=weights,
                processor_proof_sha256=proof['sha256'],bank_sha256=BANK_SHA,gpu_used=False,selected_case_keys_by_offset=selected,cpu_revision=revision)
    write_json(p['prep']/'ready.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--step',type=int,choices=STEPS,required=True);prep.add_argument('--cpu-revision',default='v1');prep.add_argument('--offset',type=int,choices=(0,15,30,45));prep.add_argument('--episode-seeds',type=parse_episode_seeds)
    win=sub.add_parser('plan');win.add_argument('--mode',choices=('qualify','score'),required=True)
    win.add_argument('--step',type=int,choices=STEPS);win.add_argument('--offset',type=int,choices=(0,15,30,45),default=0)
    win.add_argument('--stop-by-utc',required=True);win.add_argument('--attempt',type=int,default=1);win.add_argument('--qualification-attempt',type=int,default=1);win.add_argument('--cpu-revision',default='v1');win.add_argument('--episode-seeds',type=parse_episode_seeds)
    a=p.parse_args()
    if a.command=='prepare':result=prepare(a.step,a.cpu_revision,a.offset,a.episode_seeds)
    else:
        weights=None
        if a.mode=='score':
            cp=paths(a.step,a.cpu_revision);ready=json.loads((cp['prep']/'ready.json').read_text())
            if ready['status']!='CPU_CHECKPOINT_READY_LIVE_QUALIFICATION_REQUIRED' or ready['bank_sha256']!=BANK_SHA:
                raise ValueError('Checkpoint CPU gates incomplete')
            all_rows,suite=load(BANK,BANK_SHA)
            selected=[r['seed'] for r in select_policy_rows(all_rows,suite,a.offset,15,a.episode_seeds)]
            if ready.get('selected_case_keys_by_offset',{}).get(str(a.offset))!=selected:raise ValueError('CPU preflight selected cases differ from requested recovery chunk')
            weights=ready['weights_sha256']
            from matched_reset_client import require_qualification
            from native_bank import reference_binding
            from native_environment import qualified_host
            rows=all_rows
            require_qualification(Path('/mnt/extreme4tb/aaugus11/exploring-vlas/eval/runs')/f't360-q-v{a.qualification_attempt}',qualified_host(REPO),REPO,BANK_SHA,REPO/'linux-env/scripts/eval',suite=suite,**reference_binding(rows,suite))
        result=plan(a.mode,a.step,a.offset,a.stop_by_utc,weights,a.attempt,a.qualification_attempt,a.cpu_revision,a.episode_seeds)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
