#!/usr/bin/env python3
"""Launcher-owned process trees, with identity receipts and pidfd-only shutdown."""
import argparse,ctypes,fcntl,hashlib,json,os,signal,socket,struct,subprocess,sys,time,uuid
from contextlib import contextmanager
from pathlib import Path
from run_smoke import OwnedProcesses,identity,pidfd_ready


@contextmanager
def claim(path):
    fd=os.open(str(path)+'.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);yield
    finally:os.close(fd)


def record_path(path):return Path(str(path)+'.identity.json')


def read_leaf(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:return stream.read()


def save(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex)
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if temporary.exists():temporary.unlink()


def recorded_remaining(row):
    recorded={}
    for old in [*([row['workload']] if 'workload' in row else []),*row.get('children',[]),*row.get('cleanup',{}).get('tracked',[]),*row.get('cleanup',{}).get('remaining',[])]:
        if not isinstance(old,dict) or not isinstance(old.get('pid'),int) or not isinstance(old.get('start_ticks'),str):
            raise RuntimeError('Unresolved cleanup: unverified descendant evidence; see '+str(row.get('pidfile')))
        recorded[(old['pid'],old['start_ticks'])]=old
    remaining=[]
    for old in recorded.values():
        current=identity(old['pid'])
        if current and current['start_ticks']==old['start_ticks']:remaining.append(current)
    return remaining


class SupervisionFailure(RuntimeError):
    """A process may have exited without completing successful supervision."""
    def __init__(self,message,row):
        verification_error=None
        try:remaining=recorded_remaining(row)
        except RuntimeError as exc:remaining=[];verification_error=str(exc)
        supervisor=row.get('supervisor') or {};current=identity(supervisor.get('pid',-1))
        alive=bool(current and current['start_ticks']==supervisor.get('start_ticks') and current['state']!='Z')
        self.report=dict(stopped=False,supervision_ok=False,
            process_exited=None if verification_error else not alive and not remaining,
            error=message,supervision_error=row.get('supervision_error'),cleanup_error=row.get('cleanup_error'),
            evidence_error=verification_error,state=row.get('state'),nonce=row.get('nonce'),
            supervisor=supervisor,workload=row.get('workload'),remaining=remaining,cleanup=row.get('cleanup'),
            receipt_path=str(record_path(Path(row['pidfile']))) if row.get('pidfile') else None,
            signal_journal=row.get('signal_journal'))
        super().__init__(message+': '+json.dumps(self.report,sort_keys=True))


def require_retired(row):
    try:remaining=recorded_remaining(row)
    except RuntimeError as exc:raise SupervisionFailure(str(exc),row) from exc
    if remaining:raise SupervisionFailure('Unresolved cleanup: recorded descendants still alive',row)
    if row.get('supervision_error') or row.get('cleanup_error') or row.get('state') in ('failed','cleanup_failed'):
        raise SupervisionFailure('Managed supervision failed; process exit is not successful supervision',row)
    if (row.get('state')!='stopped' or not isinstance(row.get('workload'),dict) or
        not isinstance(row.get('cleanup'),dict) or row['cleanup'].get('remaining')!=[] or
        not isinstance(row['cleanup'].get('tracked'),list)):
        raise SupervisionFailure('Unresolved cleanup: incomplete supervisor retirement evidence',row)


def verified(path):
    receipt=record_path(path)
    row=json.loads(read_leaf(receipt)) if receipt.exists() else None
    if not path.exists():
        if row:
            old=row.get('supervisor') or {};current=identity(old.get('pid',-1))
            if current and current['start_ticks']==old.get('start_ticks') and current['state']!='Z':
                raise SupervisionFailure('Unresolved cleanup: missing supervisor marker',row)
            require_retired(row)
        return None
    pid=int(read_leaf(path));current=identity(pid)
    if row is None:
        if current:raise RuntimeError('Refusing live PID without a launcher identity receipt: '+str(path))
        return None
    old=row['supervisor']
    if pid!=old['pid']:raise RuntimeError('PID file differs from its ownership receipt: '+str(path))
    if current is not None and any(current[key]!=old[key] for key in ('pid','start_ticks','pgid')):
        raise RuntimeError('Refusing foreign/reused launcher PID: '+str(path))
    if row.get('supervision_error') or row.get('cleanup_error') or row.get('state') in ('failed','cleanup_failed'):
        raise SupervisionFailure('Unresolved cleanup or failed supervision',row)
    if current is None or current['state']=='Z':
        require_retired(row);return None
    if row.get('pidfile')!=str(path.resolve()):raise RuntimeError('Ownership receipt belongs to another launcher path')
    if row.get('state') not in ('running','stopping'):
        raise SupervisionFailure('Unresolved cleanup: invalid supervisor state',row)
    return row


def status(path):
    try:row=verified(path)
    except SupervisionFailure as exc:return exc.report
    return dict(running=row is not None,stopped=row is None,process_exited=row is None,
        supervision_ok=True if row is None else None,supervision_error=None,cleanup_error=None,
        supervisor=row['supervisor'] if row else None)


def start(path,log,command):
    path.parent.mkdir(parents=True,exist_ok=True)
    with claim(path):
        if verified(path):raise RuntimeError('This launcher already owns a running process: '+str(path))
        nonce=uuid.uuid4().hex
        env=dict(os.environ);env.pop('SOARM_INSTANCE_CLAIM_FD',None)
        # Operation claims stay with the caller; a manual instance must remain
        # stoppable after startup returns. The smoke runner holds its own claim.
        if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise OSError('Child subreaper unavailable')
        startup=OwnedProcesses(path.parent,time.time_ns())
        with log.open('w') as output:
            process=subprocess.Popen([sys.executable,'-B',__file__,'supervise','--pidfile',str(path),
                '--nonce',nonce,'--',*command],stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True,env=env)
        startup.add(process.pid)
        try:
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                startup.scan()
                if record_path(path).exists():
                    row=json.loads(read_leaf(record_path(path)))
                    if row.get('nonce')==nonce and (row.get('supervision_error') or row.get('cleanup_error') or row.get('state') in ('failed','cleanup_failed')):
                        raise SupervisionFailure('Managed process failed at startup',row)
                    if row.get('nonce')==nonce and row.get('state') in ('running','stopped'):
                        if row['state']=='running':assert verified(path)['nonce']==nonce
                        return process.pid
                if process.poll() is not None:raise RuntimeError('Managed process failed at startup; see '+str(log))
                time.sleep(.02)
            raise TimeoutError('Launcher supervisor did not produce its receipt; see '+str(log))
        except BaseException:
            cleanup=startup.stop()
            if cleanup['remaining']:raise RuntimeError('Startup tree did not retire: '+repr(cleanup))
            raise
        finally:
            for fd in startup.handles.values():os.close(fd)


def supervise(path,nonce,command):
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise OSError('Child subreaper unavailable')
    ledger=OwnedProcesses(path.parent,time.time_ns());stopping=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda signum,frame:stopping.append(signum))
    address='\0soarm-'+str(os.getuid())+'-'+nonce
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET);listener.bind(address);listener.listen(4);listener.settimeout(.1)
    worker=None;failure=None
    row=dict(instance_root=os.environ.get('SOARM_INSTANCE_ROOT'),supervisor=identity(os.getpid()),nonce=nonce,address=address,pidfile=str(path),state='starting',command=command)
    signals=Path(str(path)+'.signals.jsonl');row['signal_journal']=str(signals)
    def record_signal(event):
        fd=os.open(signals,os.O_WRONLY|os.O_APPEND|os.O_CREAT|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'a') as stream:
            stream.write(json.dumps(dict(event,nonce=nonce,pidfile=str(path)))+'\n');stream.flush();os.fsync(stream.fileno())
    ledger.before_signal=record_signal
    try:
        context=dict(receipt_path=str(record_path(path)),nonce=nonce,supervisor=row['supervisor'])
        env=dict(os.environ,SOARM_SUPERVISION_CONTEXT=json.dumps(context))
        worker=subprocess.Popen(command,start_new_session=True,close_fds=True,env=env)
        ledger.add(worker.pid)
        row.update(workload=identity(worker.pid),state='running')
        save(path,os.getpid());save(record_path(path),row);last_count=-1
        while not stopping:
            ledger.scan()
            if len(ledger.tracked())!=last_count:
                row['children']=ledger.tracked();save(record_path(path),row);last_count=len(ledger.tracked())
            if worker.poll() is not None:break
            try:
                connection,_=listener.accept()
                with connection:
                    connection.settimeout(1)
                    peer=struct.unpack('3i',connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
                    request=json.loads(connection.recv(1024))
                    if peer[1]!=os.getuid() or request!={'stop':nonce}:raise RuntimeError('Unrecognized process control request')
                    stopping.append('requested');connection.sendall(b'{"accepted":true}\n')
            except socket.timeout:pass
    except BaseException as exc:
        failure=exc;row['supervision_error']=repr(exc)
        raise
    finally:
        row['workload_returncode']=worker.poll() if worker is not None else None
        row['state']='stopping'
        try:save(record_path(path),row)
        except OSError as exc:print('Cannot save stopping receipt: '+repr(exc),flush=True)
        row['stop_signals']=stopping
        try:row['cleanup']=ledger.stop()
        except BaseException as exc:
            row['cleanup_error']=repr(exc)
            row['cleanup']=dict(tracked=ledger.tracked(),remaining=[dict(unverified=repr(exc))])
        row['state']='cleanup_failed' if row['cleanup']['remaining'] else ('failed' if failure else 'stopped')
        save(record_path(path),row);listener.close()
        if row['cleanup']['remaining'] and failure is None:raise RuntimeError('Managed children did not retire; see '+str(record_path(path)))



def verified_exit(path,expected):
    final=json.loads(read_leaf(record_path(Path(expected['pidfile']))))
    if final.get('nonce')!=expected['nonce'] or any(final['supervisor'][key]!=expected['supervisor'][key]
                                                  for key in ('pid','start_ticks','pgid')):
        raise RuntimeError('Final supervisor ownership differs')
    require_retired(final)
    return dict(stopped=True,process_exited=True,supervision_ok=True,cleanup=final['cleanup'])



def stop_owned(path,expected):
    row=verified(path)
    if row is None:return verified_exit(path,expected)
    if row['nonce']!=expected['nonce']:raise RuntimeError('Launcher ownership changed before stop')
    origin=Path(row['pidfile'])
    with claim(origin):
        row=verified(path)
        if row is None:return verified_exit(path,expected)
        if row['nonce']!=expected['nonce']:raise RuntimeError('Launcher ownership changed while waiting for stop')
        pid=row['supervisor']['pid'];fd=os.pidfd_open(pid)
        try:
            again=verified(path)
            if again is None:return verified_exit(path,expected)
            if again['nonce']!=row['nonce']:raise RuntimeError('Launcher ownership changed before control request')
            if pidfd_ready(fd):return verified_exit(path,expected)
            with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as client:
                client.settimeout(5);client.connect(row['address'])
                peer=struct.unpack('3i',client.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
                if peer[:2]!=(pid,os.getuid()) or not verified(path):raise RuntimeError('Process control peer identity differs')
                client.sendall(json.dumps(dict(stop=row['nonce'])).encode())
                if json.loads(client.recv(1024))!={'accepted':True}:raise RuntimeError('Process stop was not accepted')
            if not pidfd_ready(fd,35):raise TimeoutError('Owned supervisor did not retire; no numeric signal fallback')
            return verified_exit(path,expected)
        finally:os.close(fd)



def stop(path):
    row=verified(path)
    if row is None:return dict(stopped=True,already_down=True,process_exited=True,supervision_ok=True)
    runtime=row.get('instance_root')
    if runtime is None:return stop_owned(path,row)
    runtime=Path(runtime).resolve();inherited=os.environ.get('SOARM_INSTANCE_CLAIM_FD')
    if inherited is not None:
        fd=int(inherited);actual=os.fstat(fd);expected=(runtime/'instance.lock').stat()
        if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):raise RuntimeError('Inherited instance claim differs')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return stop_owned(path,row)
    from run_smoke import InstanceClaim
    owner=InstanceClaim(runtime)
    try:return stop_owned(path,row)
    finally:owner.close()


def same_birth(a,b):
    return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('pid','start_ticks','pgid'))


def capture_supervision(timeout=10):
    """Bind the caller (or its direct parent) to its launching supervisor once."""
    encoded=os.environ.get('SOARM_SUPERVISION_CONTEXT')
    if encoded is None:return None  # An explicitly foreground command has no supervisor.
    context=json.loads(encoded);path=Path(context['receipt_path']);end=time.monotonic()+timeout
    while True:
        parent=identity(os.getppid());observer=identity(os.getpid())
        try:row=json.loads(read_leaf(path))
        except FileNotFoundError:row=None
        if row is not None and row.get('nonce')==context['nonce']:
            if (not same_birth(row.get('supervisor'),context['supervisor']) or
                str(record_path(Path(row['pidfile'])))!=str(path)):
                raise RuntimeError('Launching supervisor receipt differs from the immutable context')
            if row.get('state')!='running' or row.get('supervision_error') or row.get('cleanup_error'):
                raise SupervisionFailure('Launching supervisor is not healthy',row)
            workload=row.get('workload')
            if same_birth(observer,workload) and same_birth(parent,row['supervisor']):break
            if same_birth(parent,workload):break  # The production coordinator's direct worker.
            raise RuntimeError('Caller is not the recorded workload or its direct worker')
        if time.monotonic()>=end:raise TimeoutError('Launching supervisor did not publish this workload birth')
        time.sleep(.01)
    return dict(context,workload=workload,observer=observer)


def supervision_evidence(context):
    path=Path(context['receipt_path']);data=read_leaf(path);row=json.loads(data)
    if (row.get('nonce')!=context['nonce'] or not same_birth(row.get('supervisor'),context['supervisor']) or
        not same_birth(row.get('workload'),context['workload']) or str(record_path(Path(row['pidfile'])))!=str(path)):
        raise RuntimeError('Original supervisor receipt changed ownership')
    # Journals can span several invocations of one managed path. Only this
    # nonce's events are evidence for this workload; every such event is bound.
    expected=Path(row['pidfile']+'.signals.jsonl');journal=row.get('signal_journal')
    if journal is not None and Path(journal)!=expected:raise RuntimeError('Supervisor signal journal path changed')
    events=[];journal_sha=None;journal_error=None
    try:
        if journal is not None:
            try:text=read_leaf(expected)
            except FileNotFoundError:text=None
            if text is not None:
                journal_sha=hashlib.sha256(text.encode()).hexdigest()
                for line in text.splitlines():
                    event=json.loads(line)
                    if event.get('nonce')!=context['nonce']:continue
                    if not same_birth(event.get('sender'),context['supervisor']) or event.get('pidfile')!=row['pidfile']:
                        raise RuntimeError('Supervisor signal event changed ownership')
                    events.append(event)
    except (OSError,ValueError,RuntimeError) as exc:journal_error=repr(exc)
    return dict(receipt_path=str(path),receipt_sha256=hashlib.sha256(data.encode()).hexdigest(),
        nonce=context['nonce'],supervisor=row['supervisor'],workload=row['workload'],state=row['state'],
        supervision_error=row.get('supervision_error'),cleanup_error=row.get('cleanup_error'),
        signal_journal=journal,signal_journal_sha256=journal_sha,signal_intents=events,signal_evidence_error=journal_error)


def failure_report(context,exception):
    execution_error=repr(exception);evidence=exception.report if isinstance(exception,SupervisionFailure) else None
    if context is not None:
        try:evidence=supervision_evidence(context)
        except BaseException as exc:
            return dict(error='Supervisor evidence unavailable: '+repr(exc),execution_error=execution_error,
                supervision=dict(context=context,evidence_error=repr(exc)))
    original=(evidence or {}).get('supervision_error') or (evidence or {}).get('cleanup_error')
    return dict(error='Supervisor failure: '+original if original else execution_error,
        execution_error=execution_error,supervision=evidence)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['start','supervise','status','stop','kit-log'])
    p.add_argument('--pidfile',type=Path,required=True);p.add_argument('--log',type=Path);p.add_argument('--nonce')
    arguments=sys.argv[1:];split=arguments.index('--') if '--' in arguments else len(arguments)
    a=p.parse_args(arguments[:split]);command=arguments[split+1:]
    if a.operation=='start':print(start(a.pidfile.resolve(),a.log.resolve(),command))
    elif a.operation=='supervise':supervise(a.pidfile,a.nonce,command)
    elif a.operation=='stop':print(json.dumps(stop(a.pidfile)))
    elif a.operation=='kit-log':
        row=verified(a.pidfile)
        if row is None:sys.exit(1)
        for old in row.get('children',[]):
            current=identity(old['pid'])
            if current is None or current['start_ticks']!=old['start_ticks']:continue
            try:
                for fd in Path('/proc',str(old['pid']),'fd').iterdir():
                    try:target=os.readlink(fd)
                    except OSError:continue
                    if Path(target).name.startswith('kit_') and target.endswith('.log'):
                        print(target);return
            except OSError:pass
        sys.exit(1)
    else:
        result=status(a.pidfile)
        if result['supervision_ok'] is False:print(json.dumps(result));sys.exit(2)
        if not result['running']:sys.exit(1)
        print(result['supervisor']['pid'])


if __name__=='__main__':
    try:main()
    except SupervisionFailure as exc:
        print(json.dumps(exc.report));sys.exit(2)
