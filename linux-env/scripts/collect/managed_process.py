#!/usr/bin/env python3
"""Launcher-owned process trees, with identity receipts and pidfd-only shutdown."""
import argparse,ctypes,fcntl,json,os,select,signal,socket,struct,subprocess,sys,time,uuid
from contextlib import contextmanager
from pathlib import Path
from run_smoke import OwnedProcesses,identity


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
    with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2)
    os.replace(temporary,path)


def recorded_remaining(row):
    recorded={}
    for old in [*row.get('children',[]),*row.get('cleanup',{}).get('tracked',[]),*row.get('cleanup',{}).get('remaining',[])]:
        recorded[(old['pid'],old['start_ticks'])]=old
    remaining=[]
    for old in recorded.values():
        current=identity(old['pid'])
        if current and current['start_ticks']==old['start_ticks']:remaining.append(current)
    return remaining


def verified(path):
    receipt=record_path(path)
    row=json.loads(read_leaf(receipt)) if receipt.exists() else None
    if not path.exists():
        if row and recorded_remaining(row):raise RuntimeError('Unresolved cleanup: missing supervisor marker; see '+str(receipt))
        return None
    pid=int(read_leaf(path));current=identity(pid)
    if row is None:
        if current:raise RuntimeError('Refusing live PID without a launcher identity receipt: '+str(path))
        return None
    old=row['supervisor']
    if pid!=old['pid']:raise RuntimeError('PID file differs from its ownership receipt: '+str(path))
    if row.get('state')=='cleanup_failed' or current is None or current['state']=='Z':
        remaining=recorded_remaining(row)
        if remaining:raise RuntimeError('Unresolved cleanup: recorded descendants still alive '+repr(remaining)+'; see '+str(receipt))
    if current is None:return None
    if any(current[key]!=old[key] for key in ('pid','start_ticks','pgid')):
        raise RuntimeError('Refusing foreign/reused launcher PID: '+str(path))
    if row.get('pidfile')!=str(path.resolve()):raise RuntimeError('Ownership receipt belongs to another launcher path')
    if current['state']=='Z':return None
    if row.get('state') not in ('running','stopping'):return None
    return row


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
    worker=None
    row=dict(instance_root=os.environ.get('SOARM_INSTANCE_ROOT'),supervisor=identity(os.getpid()),nonce=nonce,address=address,pidfile=str(path),state='starting',command=command)
    try:
        worker=subprocess.Popen(command,start_new_session=True,close_fds=True)
        ledger.add(worker.pid)
        row.update(workload=identity(worker.pid),state='running')
        save(path,os.getpid());save(record_path(path),row);last_count=-1
        while not stopping:
            ledger.scan()
            if len(ledger.known)!=last_count:
                row['children']=list(ledger.known.values());save(record_path(path),row);last_count=len(ledger.known)
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
    finally:
        row['workload_returncode']=worker.poll() if worker is not None else None
        row['state']='stopping'
        try:save(record_path(path),row)
        except OSError as exc:print('Cannot save stopping receipt: '+repr(exc),flush=True)
        row['cleanup']=ledger.stop();row['state']='stopped' if not row['cleanup']['remaining'] else 'cleanup_failed'
        save(record_path(path),row);listener.close()
        if row['cleanup']['remaining']:raise RuntimeError('Managed children did not retire; see '+str(record_path(path)))


def stop_owned(path,expected):
    row=verified(path)
    if row is None:return dict(stopped=True,already_down=True)
    if row['nonce']!=expected['nonce']:raise RuntimeError('Launcher ownership changed before stop')
    origin=Path(row['pidfile'])
    with claim(origin):
        row=verified(path)
        if row is None:return dict(stopped=True,already_down=True)
        if row['nonce']!=expected['nonce']:raise RuntimeError('Launcher ownership changed while waiting for stop')
        pid=row['supervisor']['pid'];fd=os.pidfd_open(pid)
        try:
            again=verified(path)
            if again is None:return dict(stopped=True,already_down=True)
            if again['nonce']!=row['nonce']:raise RuntimeError('Launcher ownership changed before control request')
            if select.select([fd],[],[],0)[0]:return dict(stopped=True,already_down=True)
            with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as client:
                client.settimeout(5);client.connect(row['address'])
                peer=struct.unpack('3i',client.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
                if peer[:2]!=(pid,os.getuid()) or not verified(path):raise RuntimeError('Process control peer identity differs')
                client.sendall(json.dumps(dict(stop=row['nonce'])).encode())
                if json.loads(client.recv(1024))!={'accepted':True}:raise RuntimeError('Process stop was not accepted')
            if not select.select([fd],[],[],35)[0]:raise TimeoutError('Owned supervisor did not retire; no numeric signal fallback')
            final=json.loads(read_leaf(record_path(origin)))
            if final.get('nonce')!=row['nonce'] or final['state']!='stopped' or final['cleanup']['remaining']:
                raise RuntimeError('Owned tree cleanup was not proven')
            return dict(stopped=True,cleanup=final['cleanup'])
        finally:os.close(fd)



def stop(path):
    row=verified(path)
    if row is None:return dict(stopped=True,already_down=True)
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
        row=verified(a.pidfile)
        if row is None:sys.exit(1)
        print(row['supervisor']['pid'])


if __name__=='__main__':main()
