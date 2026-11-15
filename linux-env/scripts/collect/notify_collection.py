"""Deliver durable collection milestones from a detached Mac process."""
import argparse,fcntl,json,os,shlex,subprocess,sys,time,uuid
from pathlib import Path


def birth(pid):
    if sys.platform.startswith('linux'):
        return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
    return subprocess.check_output(['ps','-p',str(pid),'-o','lstart='],text=True).strip()


def publish(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def fetch(args):
    code="from pathlib import Path; import json; o=Path("+repr(args.remote_root)+"); v=json.loads((o/'monitor.json').read_text()); v['final']=json.loads((o/'supervisor-final.json').read_text()) if (o/'supervisor-final.json').exists() else None; v['backups']={str(i):len(list(o.glob('sealed-backup-'+str(i)+'-cohort-*.json'))) for i in (3,4)}; print(json.dumps(v))"
    result=subprocess.run(['ssh','-oBatchMode=yes','-oConnectTimeout=15',args.host,
        shlex.join(['python3','-c',code])],capture_output=True,text=True,timeout=30)
    result.check_returncode();value=json.loads(result.stdout)
    if set(value['slots'])!={'3','4'}:raise RuntimeError('Incomplete collection monitor')
    return value


def deliver(args,state,key,message):
    prior=state['events'].get(key)
    if prior:
        if prior['kind']!='accepted':raise RuntimeError('Notification delivery unresolved; inspect '+key)
        return
    attempt=uuid.uuid4().hex;prompt=args.out/(attempt+'.txt');outcome=args.out/(attempt+'-outcome.json')
    prompt.write_text(message+'\n')
    state['events'][key]=dict(kind='attempting',attempt=attempt,prompt=str(prompt),outcome=str(outcome))
    publish(args.out/'state.json',state)  # A crash never becomes an acknowledgement.
    result=subprocess.run([sys.executable,args.ask_script,'--session',args.recipient,
        '--prompt-file',str(prompt),'--outcome',str(outcome)],capture_output=True,text=True)
    accepted=json.loads(outcome.read_text()).get('kind')=='accepted' if outcome.is_file() else False
    state['events'][key].update(kind='accepted' if accepted and result.returncode==0 else 'unresolved',
        returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,wall_ns=time.time_ns())
    publish(args.out/'state.json',state)
    if state['events'][key]['kind']!='accepted':raise RuntimeError('Notification was not acknowledged: '+key)


def worker(args):
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        context=dict(host=args.host,remote_root=args.remote_root,recipient=args.recipient,
            ask_script=str(Path(args.ask_script).resolve()))
        state=json.loads((args.out/'state.json').read_text()) if (args.out/'state.json').exists() else dict(context=context,events={})
        if state.get('context')!=context:raise RuntimeError('Notifier outbox belongs to another collection or recipient')
        begun=time.monotonic();ready=False
        while True:
            try:
                value=fetch(args);publish(args.out/'latest.json',value)
                publish(args.out/'heartbeat.json',dict(pid=os.getpid(),start_ticks=birth(os.getpid()),pgid=os.getpgid(0),token=args.token,
                    wall_ns=time.time_ns(),remote_wall_ns=value['wall_ns']))
                ready=True
                for slot,entry in value['slots'].items():
                    for milestone in range(10,entry['selected_cohorts']+1,10):
                        deliver(args,state,'slot'+slot+'-cohorts'+str(milestone),
                            'Collection progress slot'+slot+' '+str(milestone)+' cohorts: '+json.dumps(value))
                if value['faults'] or value['stopping']:
                    deliver(args,state,'stop','STOP collection (no relaunch): '+json.dumps(value))
                if value['final'] is not None:
                    deliver(args,state,'final','Collection final cleanup/result (no relaunch): '+json.dumps(value))
                    publish(args.out/'completed.json',dict(passed=True,token=args.token,wall_ns=time.time_ns(),events=state['events']))
                    return
            except Exception as error:
                publish(args.out/'error.json',dict(error=repr(error),wall_ns=time.time_ns()))
                if any(e['kind']!='accepted' for e in state['events'].values()):raise
                if not ready and time.monotonic()-begun>=30:raise
                # Transport reads can recover; a send is never retried automatically.
            time.sleep(args.interval)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--spawn',action='store_true');mode.add_argument('--worker',action='store_true')
    p.add_argument('--host',default='rtx6000');p.add_argument('--remote-root',required=True)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--ask-script',required=True);p.add_argument('--recipient',required=True)
    p.add_argument('--interval',type=float,default=15);p.add_argument('--token',default='')
    args=p.parse_args()
    if args.interval<=0:p.error('Positive interval required')
    if args.worker:return worker(args)
    args.out.mkdir(parents=True,exist_ok=True);token=uuid.uuid4().hex
    argv=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker','--host',args.host,
        '--remote-root',args.remote_root,'--out',str(args.out),'--ask-script',args.ask_script,
        '--recipient',args.recipient,'--interval',str(args.interval),'--token',token]
    with (args.out/'worker.log').open('a') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True)
    publish(args.out/'spawn.json',dict(pid=child.pid,token=token,argv=argv,wall_ns=time.time_ns()))
    deadline=time.monotonic()+40
    while time.monotonic()<deadline:
        receipt=args.out/'heartbeat.json'
        if receipt.exists():
            ready=json.loads(receipt.read_text())
            if ready['token']==token and ready['pid']==child.pid and ready['pgid']==child.pid:
                print(json.dumps(dict(passed=True,detached=True,**ready)));return
        if child.poll() is not None:raise RuntimeError('Notifier exited before readiness: '+str(child.returncode))
        time.sleep(.05)
    raise RuntimeError('Notifier did not publish its readiness heartbeat')


if __name__=='__main__':main()
