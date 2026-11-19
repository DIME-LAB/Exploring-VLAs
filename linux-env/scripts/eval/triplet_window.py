"""CPU-managed launch barrier around the reviewed native evaluation owner."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shlex
import socket
import shutil
import subprocess
import sys
import time

BASE=Path('/mnt/a4500-backup/exploring-vlas')
REPO=BASE/'eval-native-triplet360'
ROOT=BASE/'eval/triplet360'
GPU='GPU-108a1a60-bca3-abe5-812c-dfd1f2b2a1ba'
OWNER='w_eval (so-arm-101-vla) 01a1d4af'
CUT='503e39a692d47056ea325b7a20baa2f1d6900c91'


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)


def owned_ops(path):
    path=Path(path).resolve()
    require(path.is_relative_to(ROOT/'operations'),'Operations must stay under owned eval backup storage')
    return path


def managed():
    folder=REPO/'linux-env/scripts/collect'
    require(subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==CUT,'Collector cut changed')
    for name in ('managed_process.py','run_smoke.py'):
        require((folder/name).read_bytes()==subprocess.check_output(['git','-C',str(REPO),'show',CUT+':linux-env/scripts/collect/'+name]),'Managed source changed: '+name)
    sys.path.insert(0,str(folder))
    import managed_process
    return managed_process


def validate_start(start,documents,hashes,context,now):
    """Pure barrier validator; GPU launch uses this same entrance."""
    require(start.get('approved_by')=='so-arm-101-vla' and start.get('preauthorised') is True,'Missing lead authority')
    for key in ('plan','hold','free_card','notifier','probe','authority'):
        require(start.get(key+'_sha256')==hashes[key],'Start artifact changed: '+key)
    require(start.get('controller_sha256')==hashes['controller'],'Controller source changed')
    plan,hold,check,note,probe,authority=(documents[k] for k in ('plan','hold','free_card','notifier','probe','authority'))
    require(authority.get('gpu_step')==5000 and authority.get('cpu_all_steps') is True and authority.get('approved_by')=='so-arm-101-vla','Wrong preauthorised scope')
    require(plan.get('mode') in ('qualify','score') and (plan['mode']=='qualify' or plan.get('step')==5000),'Later checkpoint requires separate lead authority')
    require(plan.get('gpu')==0 and plan.get('domain')==92 and plan.get('ports')==[8781,8782,18092],'Wrong GPU/domain/ports')
    require(plan['argv'][:4]==['python3','-B','-u',str(REPO/'linux-env/scripts/eval/run_native_replay_window.py')],'Unexpected owner entrypoint')
    require(Path(plan['root']).is_relative_to(BASE/'eval/runs') and '--check-only' not in plan['argv'],'Invalid owner plan')
    argv=plan['argv']
    def flag(name):
        require(argv.count(name)==1,'Missing/duplicate owner flag '+name)
        index=argv.index(name);require(index+1<len(argv) and not argv[index+1].startswith('--'),'Missing owner flag value')
        return argv[index+1]
    require(flag('--out')==plan['root'] and flag('--bank')==str(ROOT/'held20-v2') and flag('--bank-sha256')=='0ac0d8928a960ce0aa22670f2d75c33bbc9a2b6717ba3f0fc076bec652abab0d','Owner plan bank/root mismatch')
    require(flag('--reset-collector-root')==str(REPO),'Wrong collector root')
    if plan['mode']=='qualify':
        require(argv.count('--matched-reset-qualification')==1 and '--checkpoint' not in argv and '--matched-reset-diagnostic' not in argv,'Qualification metadata differs from owner mode')
    else:
        require(argv.count('--matched-reset-diagnostic')==1 and '--matched-reset-qualification' not in argv and flag('--matched-reset-arms')=='default','Wrong scoring arm')
        require(flag('--checkpoint')==str(BASE/'eval/checkpoints/triplet360-base3ep/005000/pretrained_model'),'Only 5k is preauthorised')
        require(flag('--paired-flow-repeat')=='0' and int(flag('--bank-offset'))==plan['offset'] and flag('--bank-count')=='15','Wrong paired chunk')
        require(not any(k in argv for k in ('--cadence-diagnostic','--codec-diagnostic','--replacement-diagnostic','--memorisation-diagnostic')),'Unapproved diagnostic')
    require(hold.get('state')=='held' and hold.get('card')==GPU and hold.get('by')==OWNER and hold.get('slot')=='default','Wrong hold')
    holder=hold['holder'];supervisor=context['supervisor']
    require(holder['pid']==supervisor['pid'] and str(holder['process_start'])==str(supervisor['start_ticks']) and holder['host']==context['host'],'Hold belongs to another supervisor/birth/host')
    foreign=hold['foreign_compute']['cards'][GPU]
    require(foreign.get('state')=='clear' and all(isinstance(foreign.get(k),list) and not foreign[k] for k in ('foreign','unread','platform_unheld')),'Hold reports unresolved foreign work')
    require(not any(hold['beside'].values()),'No evaluation co-tenancy on GPU0')
    require(check.get('passed') is True and check.get('gpu_uuid')==GPU and 0<=now-check['checked_unix']<=120,'Free-card check absent/stale')
    require(all(check.get(k) is True for k in ('docker_checked','host_workflows_checked','plananything_checked','other_side_obligation_clear')) and check.get('nvml_clear_seconds',0)>=60,'Extended free-card views incomplete')
    require(note.get('ready') is True and note.get('plan_sha256')==hashes['plan'] and 0<=now-note['checked_unix']<=120,'Detached notifier not ready')
    require(probe.get('passed') is True and probe.get('step')==5000 and probe.get('layouts')==20 and probe.get('queries')==61,'5k CPU probe not complete')
    require(len(probe.get('weights_sha256',''))==64 and probe.get('bank_sha256')=='0ac0d8928a960ce0aa22670f2d75c33bbc9a2b6717ba3f0fc076bec652abab0d','Wrong CPU probe binding')
    if plan['mode']=='score':require(plan['argv'][plan['argv'].index('--weights-sha256')+1]==probe['weights_sha256'],'Probe/score checkpoint differs')


def arm(ops):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not sys.flags.optimize,'CPU-only non-optimized arm required')
    ops=owned_ops(ops);require(not (ops/'controller.pid.identity.json').exists(),'Preserve existing controller attempt')
    require(shutil.disk_usage(ROOT).free>=100*1024**3,'Backup free-space floor')
    plan=read(ops/'plan.json')
    require(subprocess.run(plan['argv']+['--check-only'],stdout=(ops/'owner-preflight.json').open('w'),check=False).returncode==0,'Owner check-only refused')
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['PYTHONOPTIMIZE']='0'
    os.environ.pop('SOARM_SUPERVISION_CONTEXT',None);os.environ.pop('SOARM_INSTANCE_CLAIM_FD',None)
    module=managed()
    pid=module.start(ops/'controller.pid',ops/'controller.log',[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--ops',str(ops)])
    return dict(armed=True,supervisor_pid=pid,gpu_started=False)


def worker(ops):
    ops=owned_ops(ops);module=managed();context=module.capture_supervision();require(context is not None,'Managed supervision required')
    context=dict(context,host=socket.gethostname())
    write(ops/'waiting.json',dict(status='WAITING_NO_GPU',context=context,checked_unix=time.time()))
    deadline=time.monotonic()+1800
    try:
        while not (ops/'START.json').exists():
            require(not (ops/'STOP_REQUEST.json').exists(),'Owned stop requested before start')
            require(time.monotonic()<deadline,'Start barrier expired; no GPU started');time.sleep(.2)
        contents={key:(ops/(name+'.json')).read_bytes() for key,name in [('plan','plan'),('hold','card-hold'),('free_card','free-card'),('notifier','notifier-ready'),('probe','probe-gate'),('authority','authority')]}
        documents={key:json.loads(value) for key,value in contents.items()}
        hashes={key:hashlib.sha256(value).hexdigest() for key,value in contents.items()}
        hashes['controller']=digest(__file__)
        validate_start(read(ops/'START.json'),documents,hashes,context,time.time())
        plan=documents['plan']
        # The reviewed diagnostic owns query validation; bind its completed
        # result to the actual published model, including for reset qualification.
        probe=documents['probe'];report=read(probe['result_path'])
        require(digest(probe['result_path'])==probe['result_sha256'] and report.get('status')=='complete' and report.get('predictions')==61 and report.get('expected_observations')==20 and report.get('weights_sha256')==probe['weights_sha256'],'CPU probe result differs')
        audit=read(probe['validation_path'])
        require(digest(probe['validation_path'])==probe['validation_sha256'] and audit.get('status')=='PASS' and audit.get('weights_sha256')==probe['weights_sha256'] and audit.get('layouts')==20 and audit.get('queries')==60 and audit.get('actual_loaded_normalization_verified') is True and audit.get('raw_inputs_and_actual_noise_paired') is True and audit.get('cuda_initialized') is False,'CPU probe audit differs')
        from triplet_eval import paths,REFERENCE
        from checkpoint_contract import require_handoff,require_processor_proof
        p=paths(5000);handoff=require_handoff(p['model'],p['handoff'],probe['weights_sha256'],published=True)
        require_processor_proof(read(p['handoff'])['source_before']['model'],p['proof'],handoff['files'],published_handoff=p['handoff'],reference_path=REFERENCE)
        # Reuse all source/model/reset/processor guards immediately before launch.
        with (ops/'owner-final-preflight.json').open('w') as out:subprocess.run(plan['argv']+['--check-only'],stdout=out,check=True)
        live=card_holds()
        matches=[h for h in live['holds'] if h.get('hold_id')==documents['hold'].get('hold_id')]
        require(len(matches)==1 and matches[0].get('state')=='held' and matches[0].get('by')==OWNER and matches[0].get('holder')==documents['hold']['holder'],'Live own hold changed')
        require(not any(h.get('state')=='held' and h.get('card')==GPU and h.get('hold_id')!=matches[0]['hold_id'] for h in live['holds']),'Another GPU0 hold appeared')
        write(ops/'live-holds.json',live)
        write(ops/'owner-start.json',dict(checked_unix=time.time(),context=context,start=read(ops/'START.json')))
        with (ops/'owner.log').open('w') as out:rc=subprocess.run(plan['argv'],stdout=out,stderr=subprocess.STDOUT).returncode
        ledger_path=Path(plan['root'])/'window-ledger.json';ledger=read(ledger_path) if ledger_path.exists() else None
        write(ops/'terminal.json',dict(owner_exit=rc,ledger=ledger,ledger_sha256=digest(ledger_path) if ledger else None,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),own_hold_release_pending=True))
    except BaseException as error:
        write(ops/'failure.json',dict(error=repr(error),checked_unix=time.time(),gpu_start_attempted=(ops/'owner-start.json').exists()))
        raise


def card_holds():
    workspace=Path.home()/'prompt2assembly'
    env=dict(os.environ,PYTHONPATH=str(workspace/'sim-instrumentation'),PA_WORKSPACE=str(workspace),PYTHONDONTWRITEBYTECODE='1')
    command=[str(workspace/'assembly-ingest/.venv/bin/python'),'-B','-m','sim_instrumentation.hostlock','card-holds','dual-a4500']
    value=json.loads(subprocess.check_output(command,cwd=workspace/'sim-instrumentation',env=env,text=True,timeout=30))
    require(value.get('host')=='dual-a4500' and isinstance(value.get('holds'),list),'Unresolved card holds')
    return value


def snapshot(ops):
    """Read supervisor retirement and the native owner's terminal evidence."""
    ops=owned_ops(ops);module=managed();present=(ops/'controller.pid.identity.json').exists()
    result=dict(present=present,terminal=read(ops/'terminal.json') if (ops/'terminal.json').exists() else None,
                failure=read(ops/'failure.json') if (ops/'failure.json').exists() else None)
    if present:result.update(supervision=module.status(ops/'controller.pid'),identity=read(ops/'controller.pid.identity.json'))
    return result


def alert(local,kind,message):
    """Attempt both recipients independently, retaining failed deliveries without retry."""
    message_path=local/(kind+'-message.txt');message_path.write_text(message+'\n')
    receipt=local/(kind+'-deliveries.json');deliveries=read(receipt) if receipt.exists() else {}
    ask='/Users/aldrininbaraj/.claude/plugins/cache/devdroid/session-tools/26a97560d68e/skills/ask/scripts/ask.py'
    for name,target in (('lead',['--session','so-arm-101-vla']),('eval',['--codex','--session','01a1d4af-6e1d-7c83-8a8c-bc2dc3b54c44','--exact','--mid-turn','--no-wait'])):
        if name in deliveries:continue
        deliveries[name]=dict(status='attempted',checked_unix=time.time());write(receipt,deliveries)
        try:subprocess.run([sys.executable,ask,*target,'--prompt-file',str(message_path)],check=True)
        except Exception as error:deliveries[name].update(status='failed',error=repr(error))
        else:deliveries[name]['status']='delivered'
        write(receipt,deliveries)


def monitor(ops,local):
    """Detached Mac notifier; it neither starts GPUs nor releases any hold."""
    local=Path(local).resolve();require(not (local/'notifier-ready.json').exists(),'Preserve notifier attempts');local.mkdir(parents=True,exist_ok=True)
    remote_script=str(REPO/'linux-env/scripts/eval/triplet_window.py')
    errors=0
    while True:
        command=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','python3','-B',remote_script,'snapshot','--ops',str(ops)]
        try:
            observed=json.loads(subprocess.check_output(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','dual-a4500',shlex.join(command)],text=True,timeout=45))
            require(observed.get('present') is True,'Supervisor identity receipt disappeared; retirement unverified')
            errors=0
        except (subprocess.SubprocessError,ValueError,OSError) as error:
            errors+=1
            write(local/'monitor-error.json',dict(error=repr(error),consecutive_errors=errors,checked_unix=time.time(),hold_release_authorised=False))
            if errors>=3:alert(local,'visibility-lost','Eval notifier lost supervisor visibility at '+str(ops)+'; monitoring continues. Inspect '+str(local/'monitor-error.json')+'. GPU retirement is UNVERIFIED; do not release the hold without owned-down proof.')
            time.sleep(min(60,10*errors));continue
        ready=dict(ready=True,pid=os.getpid(),host=socket.gethostname(),checked_unix=time.time(),plan_sha256=digest(local/'plan.json'),gpu_allowed=False)
        write(local/'notifier-ready.json',ready);write(local/'latest.json',observed)
        if observed['supervision'].get('running') is not True:
            terminal=observed.get('terminal') or {}
            write(local/'terminal.json',observed)
            message='Eval supervisor TERMINAL/UNVERIFIED at '+str(ops)+'; owner_exit='+str(terminal.get('owner_exit'))+'; failure='+str(observed.get('failure'))+'; supervision='+json.dumps(observed['supervision'])+'. Retirement evidence '+str(local/'latest.json')+'. Verify owned-down/GPU retirement before releasing ONLY our recorded PlanAnything hold. No success claim before audit.'
            alert(local,'terminal',message);return
        time.sleep(15)


def main():
    require(not sys.flags.optimize,'Optimized execution is refused')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('arm','worker','snapshot','monitor'));p.add_argument('--ops',type=Path,required=True);p.add_argument('--local',type=Path);a=p.parse_args()
    if a.operation=='arm':print(json.dumps(arm(a.ops)))
    elif a.operation=='worker':worker(a.ops)
    elif a.operation=='snapshot':print(json.dumps(snapshot(a.ops)))
    else:require(a.local is not None,'Mac local receipts required');monitor(a.ops,a.local)


if __name__=='__main__':main()
