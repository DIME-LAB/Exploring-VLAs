"""Attribute native-eval GPU contexts to identities registered before sampling."""
from pathlib import Path
import ctypes,errno,hashlib,json,os,select,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'collect'))

def birth(row):
    return row['pid'],row['start_ticks'],row['pgid']

def still_same(old,identity):
    now=identity(old['pid'])
    return now is not None and now['state']!='Z' and birth(now)==birth(old)

def owned_ancestor(pid,registered,identity):
    seen=set()
    while pid>1 and pid not in seen:
        seen.add(pid);row=identity(pid)
        if row is None:return False
        if pid in registered:
            if birth(row)!=birth(registered[pid]):raise RuntimeError('Registered ancestor birth changed')
            return True
        pid=row['ppid']
    return False

BENIGN_GRAPHICS_LIMIT=16*1024**2

def context_info(row,identity):
    process=identity(row['pid'])
    value=dict(row,memory_mib=None if row['memory_bytes'] is None else row['memory_bytes']/1024**2)
    value['identity']=process
    value['birth_time_unix_approx']=None
    if process is not None:
        try:
            boot=next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
            value['birth_time_unix_approx']=boot+int(process['start_ticks'])/os.sysconf('SC_CLK_TCK')
        except (OSError,ValueError,KeyError):pass
    value['process_name']=None if process is None else Path(process.get('command','').split(' ',1)[0]).name
    return value

def refuse_context(reason,row,identity):
    return RuntimeError(reason+': '+json.dumps(context_info(row,identity),sort_keys=True))

def attribute(expected,registered,cards,observed,identity,external_graphics=None):
    """Owned identities take priority; only small foreign graphics may be benign."""
    from gpu_device import audit_sample
    admitted={};ignored=0;benign=[]
    compute={r['pid'] for r in observed['rows'] if r['gpu_uuid']==expected and r['api']=='compute'}
    maximum={}
    for row in observed['rows']:
        if row['gpu_uuid']==expected:
            memory=row['memory_bytes']
            maximum[row['pid']]=None if memory is None or maximum.get(row['pid'],0) is None else max(maximum.get(row['pid'],0),memory)
    for row in observed['rows']:
        pid=row['pid'];old=registered.get(pid)
        if old is not None:
            if not still_same(old,identity):raise refuse_context('Registered GPU holder exited, reused PID or changed PGID',row,identity)
            admitted[pid]=old
        elif owned_ancestor(pid,registered,identity):
            raise refuse_context('Unregistered GPU context; owned descendant requires a fresh registered sample',row,identity)
        elif row['gpu_uuid']==expected:
            memory=maximum[pid]
            if row['api']!='graphics' or pid in compute or memory is None or not 0<=memory<=BENIGN_GRAPHICS_LIMIT:
                raise refuse_context('Foreign GPU0 context is not benign graphics (graphics-only <=16MiB required)',row,identity)
            current=context_info(row,identity)
            if current['identity'] is None or current['identity']['state']=='Z':
                raise refuse_context('Foreign graphics process identity unavailable',row,identity)
            current.update(classification='benign-foreign-graphics',gpu0_process_max_mib=memory/1024**2,owned=False)
            benign.append(current)
        else:
            ignored+=1
    result=audit_sample(expected,admitted,cards,observed)
    if result['outcome']!='verified' or result['foreign_gpu_contexts']:
        raise RuntimeError('GPU coverage, memory or owned device attribution failed: '+repr(result))
    return result,dict(admitted_births=[list(birth(v)) for v in admitted.values()],ignored_other_card_rows=ignored,benign_foreign_graphics_rows=benign)

class RegisteredGPUAudit:
    """Reuse launcher receipts and the collector's pidfd ownership ledger."""
    def __init__(self,runtime,receipt):
        from run_smoke import OwnedProcesses
        # Keep descendants attributable even if a short-lived parent exits.
        if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:
            raise OSError(ctypes.get_errno(),'Child subreaper unavailable')
        self.runtime=Path(runtime);self.receipt=Path(receipt)
        self.children=OwnedProcesses(self.runtime,time.time_ns())
        self.external_graphics={};self.external_baseline=None

    def freeze_external(self,expected,cards,sampler):
        """Record existing external graphics before any runtime launch, never own them."""
        from gpu_device import audit_sample
        from run_smoke import identity
        from managed_process import save
        if self.external_baseline is not None:raise RuntimeError('External graphics baseline already frozen')
        before=time.time_ns();observed=sampler(cards)
        coverage=audit_sample(expected,{},cards,observed)
        if coverage['outcome']!='verified':raise RuntimeError('External graphics baseline coverage unverified')
        owner={self.children.root['pid']:self.children.root}
        rows=[];certificates={}
        for row in observed['rows']:
            if row['gpu_uuid']!=expected:continue
            if row['api']!='graphics' or owned_ancestor(row['pid'],owner,identity):
                raise RuntimeError('Assigned GPU has non-external-graphics context before launch')
            certificate=identity(row['pid'])
            if certificate is None or certificate['state']=='Z' or row['memory_bytes'] is None or not 0<=row['memory_bytes']<=BENIGN_GRAPHICS_LIMIT:
                raise RuntimeError('External graphics baseline identity or memory unavailable')
            key=(row['pid'],row['gpu_uuid'],row['api'])
            certificates[key]=certificate
            rows.append(dict(row,identity=certificate))
        value=dict(recorded_before_launch_ns=before,completed_ns=time.time_ns(),assigned_gpu_uuid=expected,owned=False,rows=rows,coverage=observed['coverage'])
        path=self.receipt.with_name('external-graphics-baseline.json');save(path,value)
        self.external_graphics=certificates
        self.external_baseline=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**value)
        return self.external_baseline

    def prune(self):
        for pid,fd in list(self.children.handles.items()):
            if select.select([fd],[],[],0)[0]:
                os.close(fd);del self.children.handles[pid];self.children.known.pop(pid,None)

    def registry(self):
        from managed_process import verified,record_path,save
        from run_smoke import identity
        self.prune()
        for attempt in range(3):
            try:self.children.scan();break
            except OSError as error:
                if error.errno not in (errno.EINVAL,errno.ESRCH) or attempt==2:raise
                self.prune()
        origins=[];rows=[(r,None) for r in [self.children.root,*self.children.known.values()]]
        for name in ('isaacsim-soarm101.pid','control.pid'):
            path=self.runtime/name;record=verified(path)
            if record is None:raise RuntimeError('Registered GPU audit requires a live managed '+name)
            rows.extend((r,record['supervisor']) for r in [record['supervisor'],record['workload'],*record.get('children',[])])
            origins.append(dict(path=str(record_path(path)),nonce=record['nonce']))
        registered={}
        for old,supervisor in rows:
            now=identity(old['pid'])
            if now is None or now['state']=='Z' or now['start_ticks']!=old['start_ticks']:continue
            if supervisor is not None and not owned_ancestor(now['pid'],{supervisor['pid']:supervisor},identity):continue
            # Registration is before NVML: a pinned CPU child may have
            # regrouped since the previous tick. Pin its current PGID now;
            # attribute() still refuses any change across the GPU sample.
            old=dict(old,pgid=now['pgid'])
            if old['pid'] in registered and birth(registered[old['pid']])!=birth(old):
                raise RuntimeError('Conflicting ownership certificates')
            registered[old['pid']]=old
        value=dict(recorded_before_nvml_ns=time.time_ns(),processes=list(registered.values()),managed_origins=origins)
        save(self.receipt,value)
        digest=hashlib.sha256(self.receipt.read_bytes()).hexdigest()
        return registered,dict(registry_size=len(registered),registry_sha256=digest,registry_recorded_ns=value['recorded_before_nvml_ns'])

    def sample(self,expected,cards,sampler):
        from run_smoke import identity
        registered,metadata=self.registry()
        observed=sampler(cards)
        # The owner samples at 1Hz, outside the dispatch thread. Preserve every
        # assigned-card context even when classification refuses the sample.
        receipt=self.receipt.with_name('gpu-context-observations.jsonl')
        value=dict(sample_wall_ns=time.time_ns(),assigned_gpu_uuid=expected,rows=[context_info(r,identity) for r in observed['rows'] if r['gpu_uuid']==expected])
        try:
            result,checked=attribute(expected,registered,cards,observed,identity,self.external_graphics)
        except RuntimeError as error:
            value.update(status='refused',error=str(error))
            with receipt.open('a') as stream:stream.write(json.dumps(value)+'\n')
            raise RuntimeError(str(error)+'; observed_gpu0='+json.dumps(value['rows'],sort_keys=True)) from error
        value.update(status='verified',benign_foreign_graphics_rows=checked['benign_foreign_graphics_rows'])
        with receipt.open('a') as stream:stream.write(json.dumps(value)+'\n')
        owned_pids={r['pid'] for r in result['processes']}
        return result,dict(metadata,gpu_context_receipt=str(receipt),owned_gpu_contexts=[r for r in value['rows'] if r['pid'] in owned_pids],**checked)

    def close(self):
        for fd in self.children.handles.values():os.close(fd)
        self.children.handles.clear()
