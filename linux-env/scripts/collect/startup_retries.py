"""Typed pre-writer startup retries, with one persistent budget per original run."""
import copy,fcntl,json,os,stat,time,uuid
from contextlib import contextmanager
from pathlib import Path
from storage import atomic_json
from triplet_artifacts import read,sha,checked_tree

POLICY=dict(version=1,slot_retries=3,run_retries=10)
KINDS={'controller_activation','kit_start','configure_rpc','up_deadline','bootstrap_rpc','scene_readiness'}

class StartupFailure(RuntimeError):
    def __init__(self,kind,cause,stage,log):
        if kind not in KINDS:raise ValueError('Unknown startup failure kind')
        self.kind=kind;self.cause=str(cause);self.stage=copy.deepcopy(stage);self.log=Path(log)
        super().__init__(kind+': '+self.cause)


def failed_exit(event,log):
    if not event.get('name','').startswith('up-') or event.get('returncode') in (0,None):return None
    text=read(log,True).decode('utf-8')
    if any(s in text for s in ('ModuleNotFoundError','ImportError','FileNotFoundError','Unresolved scene','OwnershipStalled','CancelledGeneration','Native RPC is not bound','ABI fingerprint')):
        return None
    if 'TimeoutError: Instance controllers did not become active:' in text:kind='controller_activation'
    elif "no response for 'run_python' within" in text:kind='configure_rpc'
    elif any(s in text for s in ('ERROR: Isaac Sim exited.','ERROR: exited immediately;','TIMEOUT: ','[Errno 104] Connection reset by peer',"no response for 'soarm101.quick_start' within")) and ('Isaac Sim' in text or 'ISAACSIM_WAIT_CAP' in text):kind='kit_start'
    else:return None
    details=[line.strip() for line in text.splitlines() if any(s in line for s in ('TimeoutError:','no response for','ERROR:','TIMEOUT:'))]
    return StartupFailure(kind,details[-1] if details else 'up stage exited '+str(event['returncode']),event,log)


@contextmanager
def locked(root):
    root=Path(root)
    for parent in (root,*root.parents):
        if not stat.S_ISDIR(parent.lstat().st_mode):raise ValueError('Non-directory startup retry ancestor')
    identity=(root.stat().st_dev,root.stat().st_ino)
    fd=os.open(root/'lock',os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):raise ValueError('Non-regular retry lock')
        deadline=time.monotonic()+15
        while True:
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>=deadline:raise TimeoutError('Startup retry budget ownership stalled')
                time.sleep(.05)
        current=(root/'lock').lstat();opened=os.fstat(fd)
        if identity!=(root.stat().st_dev,root.stat().st_ino) or (current.st_dev,current.st_ino)!=(opened.st_dev,opened.st_ino):raise ValueError('Startup retry directory identity changed')
        yield
    finally:os.close(fd)


def initialize(root,checkpoint,previous=()):
    root=Path(root)
    pin=root.with_name(root.name+'.identity.json')
    if not isinstance(checkpoint,str) or len(checkpoint)!=64 or any(c not in '0123456789abcdef' for c in checkpoint):raise ValueError('Original checkpoint SHA required')
    for parent in root.parents:
        if os.path.lexists(parent) and not stat.S_ISDIR(parent.lstat().st_mode):raise ValueError('Non-directory startup retry ancestor')
    for spec in previous:
        if spec['root']!=str(root) or spec['origin_checkpoint_sha256']!=checkpoint:raise ValueError('Prior startup budget belongs to another run')
    if not root.exists():
        if previous or os.path.lexists(pin):raise ValueError('Previously bound startup retry ledger is missing')
        root.mkdir(parents=True,exist_ok=False)
        for name in ('lock','journal.jsonl'):
            fd=os.open(root/name,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600);os.close(fd)
        atomic_json(root/'policy.json',dict(POLICY,origin_checkpoint_sha256=checkpoint))
        atomic_json(root/'state.json',dict(events=[]))
    spec=dict(root=str(root),root_identity=[root.stat().st_dev,root.stat().st_ino],policy_sha256=sha(root/'policy.json'),origin_checkpoint_sha256=checkpoint,prefix=[])
    with locked(root):
        for old in previous:validated(old)
        identity={key:spec[key] for key in ('root','root_identity','policy_sha256','origin_checkpoint_sha256')}
        if os.path.lexists(pin):
            inside=read(root/'identity.json')
            if read(pin)!=inside or {k:inside.get(k) for k in identity}!=identity:raise ValueError('Startup retry identity receipt differs')
        else:
            if os.path.lexists(root/'identity.json'):raise ValueError('Startup retry external identity receipt is missing')
            identity['nonce']=uuid.uuid4().hex
            atomic_json(root/'identity.json',identity);atomic_json(pin,identity)
        spec.update(identity_receipt_path=str(pin),identity_receipt_sha256=sha(pin))
        state,_,_=validated(spec)
        spec['prefix']=[item['sha256'] for item in state['events']]
    return spec


def validated(spec):
    root=Path(spec['root']);checked_tree(root)
    if 'identity_receipt_path' in spec:
        pin=root.with_name(root.name+'.identity.json')
        identity={key:spec[key] for key in ('root','root_identity','policy_sha256','origin_checkpoint_sha256')}
        saved=read(pin)
        if str(pin)!=spec['identity_receipt_path'] or sha(pin)!=spec['identity_receipt_sha256'] or read(root/'identity.json')!=saved or {k:saved.get(k) for k in identity}!=identity or not isinstance(saved.get('nonce'),str) or len(saved['nonce'])!=32:raise ValueError('Startup retry identity receipt changed')
    if [root.stat().st_dev,root.stat().st_ino]!=spec['root_identity']:raise ValueError('Startup retry directory identity changed')
    if sha(root/'policy.json')!=spec['policy_sha256'] or read(root/'policy.json')!=dict(POLICY,origin_checkpoint_sha256=spec['origin_checkpoint_sha256']):raise ValueError('Startup retry policy changed')
    state=read(root/'state.json')
    if set(state)!={'events'} or not isinstance(state['events'],list):raise ValueError('Incomplete startup retry ledger')
    journal=read(root/'journal.jsonl',True)
    if journal and not journal.endswith(b'\n'):raise ValueError('Partial startup retry journal')
    if [json.loads(line) for line in journal.splitlines()]!=state['events']:raise ValueError('Startup retry ledger and journal differ')
    expected={'lock','journal.jsonl','policy.json','state.json'};counts={str(i):0 for i in range(3,7)};pending={};last_retry={};total=0
    if os.path.lexists(root/'identity.json'):expected.add('identity.json')
    for number,item in enumerate(state['events'],1):
        name=f'event-{number:04d}';expected.add(name)
        if type(item.get('id')) is not int or item['id']!=number or sha(root/name/'event.json')!=item.get('sha256'):raise ValueError('Startup retry history changed')
        event=read(root/name/'event.json');slot=event.get('slot');job=event.get('job',{})
        if type(event.get('id')) is not int or event['id']!=number:raise ValueError('Startup retry event index differs')
        if type(slot) is not int or slot not in range(3,7) or type(job.get('instance')) is not int or job['instance']!=slot or type(job.get('layout_seed')) is not int or type(job.get('split')) is not str:raise ValueError('Startup retry job identity differs')
        key=(job['split'],job['layout_seed']);kind=event.get('event')
        if kind=='failed':
            if event.get('kind') not in KINDS:raise ValueError('Unknown startup retry reason')
            if {p.name for p in (root/name).iterdir()}!={'event.json','up.log','owned-down.json'}:raise ValueError('Incomplete startup retry evidence')
            if sha(root/name/'up.log')!=event['up_log_sha256'] or sha(root/name/'owned-down.json')!=event['owned_down_sha256']:raise ValueError('Startup retry evidence changed')
            stage=event.get('stage',{})
            if not stage.get('name','').startswith('up-') or type(stage.get('pid')) is not int or stage['pid']<=0 or type(stage.get('started_wall_ns')) is not int or stage['started_wall_ns']<=0:raise ValueError('Startup retry lacks an up-stage receipt')
            if event['kind'] in ('controller_activation','kit_start','configure_rpc'):
                failure=failed_exit(stage,root/name/'up.log')
                if failure is None or failure.kind!=event['kind'] or failure.cause!=event['error']:raise ValueError('Startup failure classification differs from its log')
            elif event['kind']=='up_deadline':
                if stage.get('returncode') is not None:raise ValueError('Startup deadline stage is inconsistent')
            elif stage.get('returncode')!=0:raise ValueError('Bootstrap/readiness failure lacks a completed launch')
            down=read(root/name/'owned-down.json')
            ports=down.get('ports_listening',{})
            if down.get('passed') is not True or down.get('errors')!=[] or down.get('cleanup',{}).get('remaining')!=[] or set(ports)!={str(8777+2*slot),str(8778+2*slot)} or any(value is not False for value in ports.values()):raise ValueError('Startup retirement is not certified')
            pending[slot]=key
            last_retry.pop(slot,None)
        elif kind=='retry':
            if pending.get(slot)!=key:raise ValueError('Retry lacks a matching startup failure')
            counts[str(slot)]+=1;total+=1
            last_retry[slot]=number
            if counts[str(slot)]>POLICY['slot_retries'] or total>POLICY['run_retries']:raise ValueError('Startup retry budget exceeded')
        elif kind=='ready':
            if pending.get(slot)!=key or event.get('retry_id')!=last_retry.get(slot) or slot not in last_retry:raise ValueError('Ready event changed pending startup operation')
            pending.pop(slot);last_retry.pop(slot)
        else:raise ValueError('Unknown startup retry event')
        if kind!='failed' and {p.name for p in (root/name).iterdir()}!={'event.json'}:raise ValueError('Orphan startup retry evidence')
    if {p.name for p in root.iterdir()}!=expected:raise ValueError('Incomplete or orphan startup retry event')
    if [x['sha256'] for x in state['events'][:len(spec['prefix'])]]!=spec['prefix']:raise ValueError('Startup retry history was rewound')
    return state,dict(total=total,slots=counts),pending


def _append(spec,state,event,evidence=None):
    event=copy.deepcopy(event)
    root=Path(spec['root']);number=len(state['events'])+1;path=root/f'event-{number:04d}';path.mkdir()
    if evidence:
        for name,source in evidence.items():
            with (path/name).open('xb') as stream:stream.write(read(source,True));stream.flush();os.fsync(stream.fileno())
    atomic_json(path/'event.json',dict(event,id=number))
    item=dict(id=number,sha256=sha(path/'event.json'))
    fd=os.open(root/'journal.jsonl',os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
    with os.fdopen(fd,'a') as stream:stream.write(json.dumps(item)+'\n');stream.flush();os.fsync(stream.fileno())
    state['events'].append(item);atomic_json(root/'state.json',state)
    validated(spec)
    return dict(item,event_path=str(path/'event.json'),event=event)


def failed(spec,slot,job,error,down,context):
    root=Path(spec['root'])
    with locked(root):
        state,_,_=validated(spec)
        event=dict(event='failed',slot=slot,job=job,kind=error.kind,error=error.cause,stage=error.stage,context=context,
            up_log_sha256=sha(error.log),owned_down_sha256=sha(down),wall_ns=time.time_ns())
        return _append(spec,state,event,{'up.log':error.log,'owned-down.json':down})


def before_start(spec,slot,job):
    with locked(spec['root']):
        state,counts,pending=validated(spec)
        if slot not in pending:return None
        if pending[slot]!=(job['split'],job['layout_seed']):raise RuntimeError('Startup retry cannot advance to a different seed')
        if counts['slots'][str(slot)]>=POLICY['slot_retries'] or counts['total']>=POLICY['run_retries']:
            previous=[read(Path(spec['root'])/f"event-{item['id']:04d}"/'event.json') for item in reversed(state['events'])]
            failure=next(row for row in previous if row['slot']==slot and row['event']=='failed')
            raise RuntimeError('Startup retry cap reached: '+str(counts)+'; original '+failure['kind']+': '+failure['error'])
        return _append(spec,state,dict(event='retry',slot=slot,job=job,wall_ns=time.time_ns()))


def ready(spec,slot,job,reservation):
    with locked(spec['root']):
        state,_,pending=validated(spec)
        if slot not in pending:
            if reservation is not None:raise ValueError('Ready startup reservation is stale')
            return None
        if pending[slot]!=(job['split'],job['layout_seed']):raise RuntimeError('Ready startup changed seed')
        if reservation is None or reservation.get('event',{}).get('job')!=job:raise ValueError('Ready startup lacks its retry reservation')
        latest=next((item for item in reversed(state['events']) if read(Path(spec['root'])/f"event-{item['id']:04d}"/'event.json')['slot']==slot),None)
        if latest is None or latest['id']!=reservation.get('id') or latest['sha256']!=reservation.get('sha256'):raise ValueError('Ready startup reservation is stale')
        return _append(spec,state,dict(event='ready',slot=slot,job=job,retry_id=reservation['id'],wall_ns=time.time_ns()))


def summary(spec):
    with locked(spec['root']):
        state,counts,pending=validated(spec)
        return dict(policy=dict(POLICY),root=spec['root'],root_identity=spec['root_identity'],policy_sha256=spec['policy_sha256'],origin_checkpoint_sha256=spec['origin_checkpoint_sha256'],identity_receipt_path=spec.get('identity_receipt_path'),identity_receipt_sha256=spec.get('identity_receipt_sha256'),events=state['events'],counts=counts,pending={str(k):list(v) for k,v in pending.items()},state_sha256=sha(Path(spec['root'])/'state.json'))


def resume_jobs(spec,slot,jobs):
    """Resume the owned pending start first; preserve every other assigned job."""
    jobs=list(jobs);pending=summary(spec)['pending'].get(str(slot))
    if pending is None:return jobs
    matches=[job for job in jobs if [job['split'],job['layout_seed']]==pending and job['instance']==slot]
    if len(matches)!=1:raise ValueError('Pending startup job is absent or duplicated in the assigned continuation')
    job=matches[0]
    return [job]+[other for other in jobs if other is not job]
