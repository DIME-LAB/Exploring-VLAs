"""Attribute native-eval GPU contexts to identities registered before sampling."""
from pathlib import Path
import ctypes,errno,hashlib,json,os,select,time

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

def attribute(expected,registered,cards,observed,identity):
    """CPU tree churn is irrelevant; every admitted GPU holder needs a certificate."""
    from gpu_device import audit_sample
    admitted={};ignored=0
    for row in observed['rows']:
        pid=row['pid'];old=registered.get(pid)
        if old is not None:
            if not still_same(old,identity):raise RuntimeError('Registered GPU holder exited, reused PID or changed PGID')
            admitted[pid]=old
        elif row['gpu_uuid']==expected or owned_ancestor(pid,registered,identity):
            raise RuntimeError('Unregistered GPU context; a fresh registered sample is required')
        else:
            # Other users' contexts on other cards are never claimed as ours.
            ignored+=1
    result=audit_sample(expected,admitted,cards,observed)
    if result['outcome']!='verified' or result['foreign_gpu_contexts']:
        raise RuntimeError('GPU coverage, memory or owned device attribution failed: '+repr(result))
    return result,dict(admitted_births=[list(birth(v)) for v in admitted.values()],ignored_other_card_rows=ignored)

class RegisteredGPUAudit:
    """Reuse launcher receipts and the collector's pidfd ownership ledger."""
    def __init__(self,runtime,receipt):
        from run_smoke import OwnedProcesses
        # Keep descendants attributable even if a short-lived parent exits.
        if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:
            raise OSError(ctypes.get_errno(),'Child subreaper unavailable')
        self.runtime=Path(runtime);self.receipt=Path(receipt)
        self.children=OwnedProcesses(self.runtime,time.time_ns())

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
        result,checked=attribute(expected,registered,cards,observed,identity)
        return result,dict(metadata,**checked)

    def close(self):
        for fd in self.children.handles.values():os.close(fd)
        self.children.handles.clear()
