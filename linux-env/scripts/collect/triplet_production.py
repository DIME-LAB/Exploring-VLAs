#!/usr/bin/env python3
"""Four owned RTX instances, source-verified RGB groups and deterministic reserves.

Print the immutable plan with --plan. --go additionally requires the explicit
training-release certificate and starts the four workers with a 60 s stagger.
"""
import argparse,ctypes,hashlib,json,os,shlex,signal,socket,stat,subprocess,sys,time,threading,tempfile
from contextlib import nullcontext
from pathlib import Path
sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'vla_SO-ARM101/src/so_arm101_control'))
from storage import CANONICAL_ROOT,atomic_json,confined
from host_config import profile
from run_smoke import InstanceClaim,OwnedProcesses,frozen,identity,pidfd_ready
from managed_process import capture_supervision,failure_report,supervision_evidence
from gpu_device import gpu_uuid
from triplet_production_plan import initial_wave,reduce_waves,guards,POLICY
from triplet_artifacts import read,sha,manifest,classify,selected_entry,checked_tree
from bootstrap_capsule import ASSIGNED_UUID,load as load_bootstrap_capsule
from backup_space import probe_code as backup_probe_code,serialized_copy,sealed_payload,receiver_code,validate_admission
from gpu_device import inventory
import clock_status
import startup_retries

ROOT=Path('/mnt/storage3/aaugus11/exploring-vlas')
DUAL=Path('/mnt/a4500-backup/exploring-vlas/datasets/counterfactual-production-v1')
SELF=Path(__file__).resolve()
SSH=['ssh','-oBatchMode=yes','-oConnectTimeout=15','-oStrictHostKeyChecking=yes','-oHostKeyAlias=dual-a4500',
    '-oControlMaster=no','-oControlPath=none','-oControlPersist=no',
    '-oUserKnownHostsFile='+str(ROOT/'setup/dual-known_hosts'),'aaugus11@10.179.23.212']


def require_host():
    if profile()['name']!='rtx6000' or CANONICAL_ROOT!=ROOT:raise ValueError('Production is qualified only on rtx6000')


def plan(cut,repo=None):
    require_host()
    if repo is None:frozen(cut)
    else:data_checkout(Path(repo),cut)
    capsule=load_bootstrap_capsule(ROOT/'setup/counterfactual-kit-bootstrap-v1/capsule.json')
    return dict(version=1,cut=cut,source_sha256=sha(SELF if repo is None else Path(repo)/SELF.relative_to(REPO)),policy=POLICY,jobs=initial_wave(),
        bootstrap_capsule_path=capsule['capsule_path'],bootstrap_capsule_sha256=capsule['capsule_sha256'],
        production_root=str(ROOT/'datasets/counterfactual-targets/production-v1'),
        runtime_root=str(ROOT/'runtime/counterfactual-production-v1'),assigned_gpu_uuid=ASSIGNED_UUID,
        bulk_floor_gib=50,root_floor_gib=20,start_stagger_s=60,nice=10,
        authoritative_registries={s:str(ROOT/f'datasets/counterfactual-targets/production-v1/{s}-registry.json') for s in ('train','heldout')},
        dual_backup_root=str(DUAL),raw_retention='keep original CRF18 and all receipts/frames, no rolling deletes',
        slots=[dict(instance=i,domain=90+i,bridge_port=8777+2*i,http_port=8778+2*i,physics_threads=4,kit_threads=6) for i in (3,4,5,6)])


def disk():
    from shutil import disk_usage
    measured=dict(root_free_bytes=disk_usage('/').free,bulk_free_bytes=disk_usage(ROOT).free)
    if measured['root_free_bytes']<20*1024**3 or measured['bulk_free_bytes']<50*1024**3:raise RuntimeError('Production disk gate: '+repr(measured))
    return measured


class DiskWatch:
    """Keep checking while the main thread is inside a blocking RPC or backup."""
    def __init__(self,root,on_fault,probe=None,interval=.5,report=None):
        self.root=Path(root);self.on_fault=on_fault;self.probe=probe or disk;self.interval=interval
        self.report=Path(report) if report is not None else self.root/'disk-monitor.json'
        self.stop=threading.Event();self.thread=None;self.fault=None;self.samples=0;self.last=None;self.minima={};self.closed=False
        self.last_write=0.
    def record(self):
        atomic_json(self.report,dict(healthy=self.fault is None,closed=self.closed,interval_s=self.interval,
            samples=self.samples,last=self.last,minima_bytes=self.minima,fault=self.fault,wall_ns=time.time_ns()))
    def sample(self):
        if self.fault:return False
        try:
            self.last=self.probe();self.samples+=1
            for key,value in self.last.items():self.minima[key]=min(self.minima.get(key,value),value)
            if time.monotonic()-self.last_write>=5:self.record();self.last_write=time.monotonic()
            return True
        except BaseException as exc:
            self.fault=dict(error=repr(exc),source='active disk monitor',wall_ns=time.time_ns(),last_healthy=self.last)
            try:publish_stop(self.root,self.fault);self.record()
            except BaseException as write_error:self.fault['publication_error']=repr(write_error)
            # This callback signals only the coordinator's proven worker pidfds.
            # It also runs if the filesystem cannot publish a STOP file.
            try:self.on_fault()
            except BaseException as callback_error:self.fault['retirement_request_error']=repr(callback_error)
            return False
    def start(self):
        self.sample();self.require()
        def loop():
            while not self.stop.wait(self.interval):
                if not self.sample():return
        self.thread=threading.Thread(target=loop,name='production-disk-monitor',daemon=True);self.thread.start()
    def require(self):
        if self.fault:raise RuntimeError('Production disk monitor stopped admission: '+repr(self.fault))
    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
            if self.thread.is_alive():raise RuntimeError('Disk monitor did not retire')
        self.closed=True;self.record()


def publication_gate(root,watch):
    if watch is not None:watch.require()
    disk()
    if (Path(root)/'STOP.json').exists():raise RuntimeError('STOP forbids production publication')


def publish_stop(root,event):
    """Publish a durable first fault without replacing another worker's cause."""
    import uuid
    path=Path(root)/('fault-'+uuid.uuid4().hex+'.json');atomic_json(path,event)
    try:os.link(path,Path(root)/'STOP.json',follow_symlinks=False)
    except FileExistsError:pass
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def wait_workers(children,observe,scan,retired,stop,pause=time.sleep):
    """Drain a final publication after exit and refuse an uncertified child exit."""
    done=set()
    while True:
        scan();observe()
        if stop.exists():raise RuntimeError('Production worker requested stop')
        for process,_,slot in children:
            code=process.poll()
            if code is None:continue
            if code:raise RuntimeError('Production worker exited before certification: '+str(slot))
            if slot not in done:retired(slot);done.add(slot)
        if len(done)==len(children):return observe()
        pause(10)


def retirement_supervision(context,previous=None):
    """Refresh durable owner evidence after local retirement, preserving earlier causes.

    The owner's terminal receipt follows its workload's exit. This read proves
    only its latest durable state; final outer certification still must pass.
    """
    if context is None:return previous,None
    execution_error=(previous or {}).get('execution_error')
    prior_evidence=(previous or {}).get('supervision') or {}
    prior_original=prior_evidence.get('supervision_error') or prior_evidence.get('cleanup_error')
    try:evidence=supervision_evidence(context)
    except BaseException as exc:
        unavailable=dict(context=context,evidence_error=repr(exc))
        return dict(error=(previous or {})['error'] if prior_original else 'Supervisor evidence unavailable after retirement: '+repr(exc),
            execution_error=execution_error,supervision=unavailable,supervision_refresh_passed=False,
            earlier_failure=previous),unavailable
    original=evidence.get('supervision_error') or evidence.get('cleanup_error')
    healthy=not original and evidence['state']=='running' and not evidence.get('signal_evidence_error')
    if previous is None and healthy:return None,evidence
    report=dict(previous or {},execution_error=execution_error,supervision=evidence,
        supervision_refresh_passed=not bool(evidence.get('signal_evidence_error')))
    if original:report['error']='Supervisor failure: '+original
    elif prior_original:
        # A partial or inconsistent later source must not erase a known cause.
        report.update(error=previous['error'],earlier_failure=previous,supervision_refresh_passed=False,
            supervision_refresh_error='Previously recorded supervisor error is missing from the latest receipt')
    elif previous is None:report['error']='Supervisor retirement evidence is not healthy: '+repr(evidence)
    return report,evidence


def verify_sources(run,cut,repo=None):
    repo=REPO if repo is None else Path(repo)
    config=read(run/'run.json')
    if config.get('git_commit')!=cut or config.get('cpu_fixture_only'):raise ValueError('Collector checkout/provenance differs')
    if not config.get('source_sha256'):raise ValueError('Loaded source manifest is missing')
    for name,value in config['source_sha256'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or sha(repo/path)!=value:raise ValueError('Loaded collector source differs: '+name)


def backup(source,files,destination,log):
    """Copy immutable files only and verify every byte remotely before selection."""
    if not Path(destination).is_relative_to(DUAL):raise ValueError('Backup outside the assigned projection root')
    if manifest(source)!=files:raise ValueError('Source changed before SHA backup')
    specs={name:dict(size=(source/name).stat(follow_symlinks=False).st_size,sha256=value) for name,value in files.items()}
    incoming=sum(row['size'] for row in specs.values())
    with serialized_copy(ROOT/'runtime/counterfactual-backup-space.lock'):
        code=backup_probe_code(destination,incoming,len(files))
        reply=subprocess.check_output([*SSH,'python3 -c '+shlex.quote(code)],text=True,stderr=log)
        admission=json.loads(reply);validate_admission(admission,destination,specs)
        print(json.dumps(dict(event='dual_backup_admission',**admission)),file=log,flush=True)
        with sealed_payload(source,specs,admission,ROOT/'runtime') as wire:
            if manifest(source)!=files:raise ValueError('Source changed before bounded backup transfer')
            reply=subprocess.run([*SSH,'python3 -c '+shlex.quote(receiver_code(destination))],stdin=wire,text=True,
                check=True,stdout=subprocess.PIPE,stderr=log)
        receipt=json.loads(reply.stdout)
        if (receipt.get('passed') is not True or receipt.get('destination')!=str(destination) or
            receipt.get('files_sha256')!=files or receipt.get('transferred_bytes')!=incoming or
            receipt.get('capacity_admission')!=admission):raise ValueError('Backup receiver did not certify the admitted bytes')
        if manifest(source)!=files:raise ValueError('Source changed during SHA backup')
        return dict(receipt,source=str(source),completed_wall_ns=time.time_ns())



def diagnostic_copy(source,target,namespace):
    """Copy regular bytes; catalog links/devices without following or reading them."""
    source=Path(source);entries=[];regular={};pending=[source]
    for parent in source.parents:
        try:mode=parent.lstat().st_mode
        except FileNotFoundError:return dict(source=str(source),entries=[dict(path='.',kind='absent')],regular_files_sha256={})
        if not stat.S_ISDIR(mode):
            return dict(source=str(source),blocked_ancestor=str(parent),entries=[],regular_files_sha256={})
    while pending:
        path=pending.pop()
        try:info=path.lstat()
        except FileNotFoundError:
            entries.append(dict(path=str(path),kind='absent'));continue
        relative='.' if path==source else str(path.relative_to(source))
        row=dict(path=relative,mode=info.st_mode,size=info.st_size,mtime_ns=info.st_mtime_ns)
        if stat.S_ISLNK(info.st_mode):row.update(kind='symlink',link_target=os.readlink(path))
        elif stat.S_ISDIR(info.st_mode):row['kind']='directory';pending.extend(sorted(path.iterdir(),reverse=True))
        elif stat.S_ISREG(info.st_mode):
            name=namespace+'-%06d'%len(regular);output=target/name
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            with os.fdopen(fd,'rb') as stream,output.open('xb') as saved:
                before=os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):raise ValueError('Diagnostic source changed type')
                for chunk in iter(lambda:stream.read(8*1024**2),b''):saved.write(chunk)
                saved.flush();os.fsync(saved.fileno());after=os.fstat(stream.fileno())
            value=sha(output);regular[relative]=value
            row.update(kind='regular',snapshot_file=name,sha256=value,changed_during_copy=(before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns))
        else:row['kind']='nonregular-no-content-read'
        entries.append(row)
    return dict(source=str(source),entries=entries,regular_files_sha256=regular)


def backup_terminal(recipe,job,error,worker_root=None):
    """Called only after owned retirement; diagnostic copies never enter selection."""
    import uuid
    root=Path(recipe['production_root']);run=ROOT/f'datasets/counterfactual-targets/{job["split"]}/layout-{job["layout_seed"]}'
    token=job['split']+'-'+str(job['layout_seed'])+'-'+uuid.uuid4().hex
    snapshot=root/'terminal-diagnostics'/token;snapshot.mkdir(parents=True,exist_ok=False)
    catalog=dict(group=diagnostic_copy(run,snapshot,'group'))
    if worker_root is not None:
        # Stage logs are closed after down; caches and previous group snapshots
        # are intentionally outside this diagnostic set.
        for i,path in enumerate(sorted(Path(worker_root).glob('*.log'))+sorted(Path(worker_root).glob('stages.json'))):
            catalog['worker-'+str(i)]=diagnostic_copy(path,snapshot,'worker-'+str(i))
    atomic_json(snapshot/'terminal-disposition.json',dict(version=1,diagnostic_only=True,training_forbidden=True,
        cut=recipe['cut'],job=job,original_error=error,raw_root=str(run),catalog=catalog,wall_ns=time.time_ns()))
    files=manifest(snapshot);destination=DUAL/snapshot.relative_to(ROOT)
    logpath=root/(token+'-backup.log')
    with logpath.open('x') as log:receipt=backup(snapshot,files,destination,log)
    value=dict(passed=True,diagnostic_only=True,training_forbidden=True,raw_root=str(run),job=job,
        original_error=error,catalog_sha256=sha(snapshot/'terminal-disposition.json'),snapshot=str(snapshot),backup=receipt,
        source_regular_files_sha256=catalog['group']['regular_files_sha256'])
    output=root/'terminal-backups'/(token+'.json');output.parent.mkdir(exist_ok=True);atomic_json(output,value)
    return dict(path=str(output),sha256=sha(output))


def backup_startup_budget(recipe,log):
    spec=recipe.get('startup_retry_budget')
    if not spec:return None
    report=startup_retries.summary(spec)
    source=Path(spec['root']);files=manifest(source)
    receipt=backup(source,files,DUAL/source.relative_to(ROOT),log)
    pin=Path(spec['identity_receipt_path'])
    with tempfile.TemporaryDirectory(prefix='retry-identity-backup-',dir=ROOT/'runtime') as temp:
        payload=Path(temp);(payload/pin.name).write_bytes(read(pin,True))
        identity=backup(payload,{pin.name:spec['identity_receipt_sha256']},DUAL/pin.parent.relative_to(ROOT),log)
    return dict(snapshot=report,files_sha256=files,ledger=receipt,identity_receipt=identity)


def backup_remaining_terminal(recipe,jobs,error):
    """Coordinator fallback after every descendant has been proven gone."""
    root=Path(recipe['production_root']);records=[];errors=[]
    inherited={(row['split'],row['layout_seed']) for row in recipe.get('inherited_outcomes',[])}
    for job in jobs:
        if (job['split'],job['layout_seed']) in inherited:continue
        run=ROOT/f'datasets/counterfactual-targets/{job["split"]}/layout-{job["layout_seed"]}'
        if not os.path.lexists(run):continue
        try:
            files=manifest(run);proof=root/'proofs'/job['split']/(str(job['layout_seed'])+'.json')
            if proof.exists():
                known=read(proof)
                if known.get('backup',{}).get('passed') is True and known.get('files_sha256')==files:continue
            if any((row:=read(p)).get('raw_root')==str(run) and row.get('passed') is True and row.get('source_regular_files_sha256')==files
                for p in (root/'terminal-backups').glob('*.json')):continue
        except (OSError,ValueError):pass  # Preserve invalid entries as a diagnostic catalog.
        try:records.append(backup_terminal(recipe,job,error))
        except BaseException as exc:errors.append(dict(layout_seed=job['layout_seed'],error=repr(exc)))
    retry_backup=None
    if recipe.get('startup_retry_budget'):
        try:
            path=Path(recipe['production_root'])/'startup-retry-stop-backup.log'
            with path.open('x') as log:retry_backup=backup_startup_budget(recipe,log)
        except BaseException as exc:errors.append(dict(source='startup retry ledger',error=repr(exc)))
    result=dict(records=records,errors=errors)
    if recipe.get('startup_retry_budget'):result['startup_retry_budget']=retry_backup
    return result


def data_checkout(repo,cut):
    frozen(cut,repo=Path(repo))
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if actual!=cut:raise ValueError('Collector checkout HEAD differs from its data cut: '+actual)


def qualified_recipe(recipe):
    """The outer runner and the data-producing checkout have independent cuts."""
    if recipe.get('collector_repo'):
        data_checkout(Path(recipe['collector_repo']),recipe['cut'])
        frozen(recipe['runner_cut'])
    else:frozen(recipe['cut'])


class Worker:
    def __init__(self,recipe,instance,wave):
        self.plan=recipe;self.instance=instance;self.wave=wave;self.cut=recipe['cut'];self.port=8777+2*instance
        self.root=confined(Path(recipe['production_root'])/f'wave-{wave:02d}/slot-{instance}')
        self.root.mkdir(parents=True,exist_ok=False);self.runtime=None;self.ledger=None;self.claim=None;self.stages=[];self.native=None
        self.global_root=Path(recipe['production_root']);self.wrapper=REPO/'linux-env/scripts/collect_sim.sh'
        self.session=-1;self.stop_file=self.global_root/'STOP.json';self.reason=None;self.streak=0
        self.active_job=None
        self.collector_repo=Path(recipe.get('collector_repo',REPO))
        self.supervision=capture_supervision()
        self.retirement_certified=True
        self.startup_retries=[]
    def check(self):
        qualified_recipe(getattr(self,'plan',dict(cut=self.cut)))
        try:disk()
        except BaseException as exc:
            publish_stop(self.global_root,dict(error=repr(exc),source='worker disk gate',slot=self.instance,wall_ns=time.time_ns()))
            raise
        if self.stop_file.exists():raise RuntimeError('Production peer stop requested: '+str(self.stop_file))
    def await_admission(self,seed,outcome):
        path=self.root/('admitted-'+str(seed)+'.json');end=time.monotonic()+180
        while not path.exists():
            self.check()
            if time.monotonic()>end:raise TimeoutError('Coordinator did not verify group disposition')
            self.ledger.scan();time.sleep(.2)
        if read(path)!=outcome:raise ValueError('Coordinator admission differs from the published disposition')
    def stage(self,name,argv,timeout=3600,export=False,cleanup=False):
        if not cleanup:self.check()
        start=time.time_ns();end=time.monotonic()+timeout;env=dict(self.env)
        if export:
            if getattr(self,'plan',{}).get('collector_repo'):env['CUDA_VISIBLE_DEVICES']=''
            for key in ('PYTHONPATH','LD_LIBRARY_PATH','AMENT_PREFIX_PATH','CMAKE_PREFIX_PATH','COLCON_PREFIX_PATH','ROS_DISTRO','ROS_VERSION','ROS_PYTHON_VERSION'):env.pop(key,None)
        with (self.root/(name+'.log')).open('x') as log:
            process=subprocess.Popen(argv,cwd=getattr(self,'collector_repo',REPO),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(self.claim.fd,))
            self.ledger.add(process.pid)
            try:
                while process.poll() is None:
                    self.ledger.scan()
                    if not cleanup:self.check()
                    if time.monotonic()>end:raise TimeoutError('Stage deadline: '+name)
                    time.sleep(.1)
            finally:
                self.ledger.scan();row=dict(name=name,pid=process.pid,started_wall_ns=start,returncode=process.returncode,wall_s=(time.time_ns()-start)/1e9)
                self.stages.append(row);atomic_json(self.root/'stages.json',self.stages);print(json.dumps(row),flush=True)
        return row
    def rpc(self,code):
        self.check()
        sys.path.insert(0,str(REPO/'omni-kit-mcp'))
        from kit_mcp.client import call
        import uuid
        nonce=uuid.uuid4().hex;before=None
        if self.native:
            before=identity(self.native['pid'])
            if before is None or before['start_ticks']!=self.native['start_ticks'] or before['pgid']!=self.native['pgid']:raise RuntimeError('Native responder identity changed')
        query=code+"\nimport os\n_f=open('/proc/self/stat').read().rsplit(')',1)[1].split()\nresult={'answer':result,'pid':os.getpid(),'start_ticks':_f[19],'pgid':int(_f[2]),'nonce':"+repr(nonce)+'}'
        self.ledger.scan();result=call('run_python',dict(code=query),port=self.port,timeout=30)
        if isinstance(result,dict) and 'result' in result:result=result['result']
        self.ledger.scan();pid=result['pid'];owned=self.ledger.known.get(pid);now=identity(pid);fd=self.ledger.handles.get(pid)
        if (result['nonce']!=nonce or owned is None or now is None or fd is None or pidfd_ready(fd) or
            result['start_ticks']!=owned['start_ticks'] or result['start_ticks']!=now['start_ticks'] or result['pgid']!=now['pgid'] or
            before is not None and any(result[k]!=before[k] for k in ('pid','start_ticks','pgid'))):raise RuntimeError('Native RPC is not bound to an owned live responder')
        self.native={k:result[k] for k in ('pid','start_ticks','pgid')}
        with (self.root/'native-observations.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n');stream.flush();os.fsync(stream.fileno())
        return result['answer']
    def up(self):
        self.session+=1;self.native=None
        self.runtime=confined(Path(self.plan['runtime_root'])/f'wave-{self.wave:02d}/slot-{self.instance}/session-{self.session:03d}')
        self.runtime.mkdir(parents=True,exist_ok=False);self.claim=InstanceClaim(self.runtime)
        self.retirement_certified=False
        if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise OSError('Subreaper unavailable')
        self.ledger=OwnedProcesses(self.runtime,time.time_ns())
        for port in (self.port,self.port+1):
            with socket.socket() as sock:
                sock.settimeout(.3)
                if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Private port occupied; never adopt it: '+str(port))
        capsule=load_bootstrap_capsule(self.plan['bootstrap_capsule_path'],self.plan['bootstrap_capsule_sha256'],cards=inventory())
        pin=capsule['pin'];table=Path(pin['source']['path'])
        if pin.get('passed') is not True or pin.get('assigned_gpu_uuid')!=ASSIGNED_UUID or sha(table)!=pin['source']['sha256']:
            raise RuntimeError('Qualified RTX Kit table changed')
        (self.runtime/'isaac.log').write_bytes(read(table,True))
        self.env=dict(os.environ,ISAACSIM_GPU='0',SOARM_SIM_NICE='0',SOARM_PHYSICS_THREADS='4',SOARM_KIT_THREADS='6',
            SOARM_INSTANCE_ROOT=str(self.runtime),SOARM_COLLECTION_ROOT=str(ROOT),HF_LEROBOT_HOME=str(ROOT/'datasets'),
            PYTHONDONTWRITEBYTECODE='1',PYTHONFAULTHANDLER='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            SOARM_EVAL_LOCALHOST_ONLY='1',SOARM_INSTANCE_CLAIM_FD=str(self.claim.fd),ISAACSIM_LOG=str(self.runtime/'isaac.log'),
            CUDA_VISIBLE_DEVICES=ASSIGNED_UUID,SOARM_ASSIGNED_GPU_UUID=ASSIGNED_UUID)
        if self.plan.get('collector_repo'):
            self.env.update(SOARM_COLLECTOR_REPO=str(self.collector_repo),SOARM_COLLECTION_CONFIGURE_TIMEOUT='300')
            atomic_json(self.root/f'startup-budget-{self.session}.json',dict(configure_timeout_s=300,
                configure_source_sha256=sha(self.collector_repo/'linux-env/scripts/collect/configure_clock.py'),
                collector_cut=self.cut,runner_cut=self.plan['runner_cut'],session=self.session))
        for name,relative in dict(TMPDIR='tmp',TEMP='tmp',TMP='tmp',XDG_CACHE_HOME='cache',HF_HOME='cache/huggingface',ROS_LOG_DIR='ros-logs').items():
            path=self.runtime/relative;path.mkdir(parents=True,exist_ok=True);self.env[name]=str(path)
        for key in ('ISAACSIM_EXTRA_ARGS','ROS_LOCALHOST_ONLY','DISPLAY','XAUTHORITY'):self.env.pop(key,None)
        name='up-'+str(self.session);log=self.root/(name+'.log')
        try:
            disk();event=self.stage(name,['bash',str(self.wrapper),'up',str(self.instance)],timeout=1500)
        except TimeoutError as exc:
            self.check()
            if str(exc)=='Stage deadline: '+name and self.stages[-1]['name']==name:
                raise startup_retries.StartupFailure('up_deadline',repr(exc),self.stages[-1],log) from exc
            raise
        if event['returncode']:
            self.check()
            known=startup_retries.failed_exit(event,log)
            if known is not None:raise known
            raise RuntimeError('Startup failed; see '+event['name']+'.log')
        # The first challenge establishes ownership; the second binds before and
        # after the RPC. Clock qualification installs this checkout's runtime.
        up_event=event
        try:
            self.rpc('result=True')
            self.rpc('result=True')
        except (TimeoutError,ConnectionError) as exc:
            self.check()
            raise startup_retries.StartupFailure('bootstrap_rpc',repr(exc),event,log) from exc
        cert=self.root/f'clock-{self.session}.json'
        event=self.stage('clock-'+str(self.session),['bash',str(self.wrapper),'clock-sanity',str(self.instance),'--output',str(cert)],timeout=1200)
        clock_status.require(cert,event['pid'],event['started_wall_ns'],event['returncode'])
        import stage_readiness
        try:
            stage_readiness.wait(lambda:dict(stage_readiness=self.rpc(stage_readiness.probe_code())),
                lambda p:atomic_json(self.root/f'readiness-{self.session}.json',p))
        except (TimeoutError,ConnectionError) as exc:
            self.check()
            raise startup_retries.StartupFailure('scene_readiness',repr(exc),up_event,log) from exc

    def ensure_ready(self,job):
        if self.claim is not None:return
        budget=self.plan.get('startup_retry_budget')
        if budget is None:self.up();return
        while True:
            self.check()
            run=ROOT/f'datasets/counterfactual-targets/{job["split"]}/layout-{job["layout_seed"]}'
            if os.path.lexists(run):raise RuntimeError('Startup retry refuses existing episode artefacts: '+str(run))
            reservation=startup_retries.before_start(budget,self.instance,job)
            if reservation is not None:
                atomic_json(self.root/f'startup-retry-before-{self.session+1}.json',reservation)
                self.startup_retries.append(reservation)
            try:self.up()
            except startup_retries.StartupFailure as exc:
                atomic_json(self.root/f'startup-up-error-{self.session}.json',dict(kind=exc.kind,error=exc.cause,stage=exc.stage,up_log=str(exc.log),phase='before_retirement',wall_ns=time.time_ns()))
                try:self.down()
                except BaseException as cleanup:
                    raise RuntimeError('Startup cleanup failed; original '+repr(exc)+'; cleanup '+repr(cleanup)) from cleanup
                if not self.retirement_certified:raise RuntimeError('Startup retry lacks certified retirement')
                if os.path.lexists(run):raise RuntimeError('Startup retry refuses episode artefacts: '+str(run))
                record=startup_retries.failed(budget,self.instance,job,exc,self.root/f'owned-down-{self.session}.json',self.supervision)
                atomic_json(self.root/f'startup-failure-{self.session}.json',record)
                self.startup_retries.append(record)
                continue
            completed=startup_retries.ready(budget,self.instance,job,reservation)
            if completed is not None:self.startup_retries.append(completed)
            return
    def down(self):
        if self.claim is None:
            if not self.retirement_certified:raise RuntimeError('Previous owned cleanup remains unresolved')
            return
        failures=[];cleanup=None;ports={}
        try:
            if self.ledger is not None and hasattr(self,'env'):
                try:
                    row=self.stage('down-'+str(self.session),['bash',str(self.wrapper),'down',str(self.instance)],timeout=90,cleanup=True)
                    if row['returncode']:failures.append('down command failed')
                except BaseException as exc:failures.append(repr(exc))
            if self.ledger is not None:
                try:cleanup=self.ledger.stop()
                except BaseException as exc:failures.append(repr(exc))
            for port in (self.port,self.port+1):
                with socket.socket() as sock:
                    sock.settimeout(.3);ports[str(port)]=sock.connect_ex(('127.0.0.1',port))==0
            passed=cleanup is not None and not cleanup['remaining'] and not any(ports.values()) and not failures
            self.retirement_certified=passed
            atomic_json(self.root/f'owned-down-{self.session}.json',dict(passed=passed,cleanup=cleanup,ports_listening=ports,errors=failures))
            if not passed:raise RuntimeError('Owned cleanup was not certified')
        finally:self.claim.close();self.claim=None
    def attempt(self,job):
        disk();seed=job['layout_seed'];run=ROOT/f'datasets/counterfactual-targets/{job["split"]}/layout-{seed}'
        if run.exists():raise FileExistsError('Fresh layout required; never adopt output: '+str(run))
        event=self.stage('collect-'+str(seed),['bash',str(self.wrapper),'run',str(self.instance),'--counterfactual-layout-seed',str(seed),
            '--counterfactual-split',job['split'],'--normal-only','--output',str(run)],timeout=1800)
        outcome=self.qualify(job,event)
        status,reason=outcome['status'],outcome['reason']
        self.await_admission(seed,outcome)
        if status=='refused':
            self.streak=self.streak+1 if self.reason==reason else 1;self.reason=reason
            self.down()
            if self.streak>=3:raise RuntimeError('Three consecutive same-reason group rejections: '+reason)
        else:
            self.reason=None;self.streak=0
            restored=self.rpc('import collection_runtime as c\nresult=c.runtime().restore('+repr(read(run/'attempt-0000/receipt.json')['initial_physical'])+',arm=True)')
            if not restored.get('arm_restored') or not restored.get('command_baseline'):raise RuntimeError('Between-group baseline restore failed')
            atomic_json(self.root/('between-group-'+str(seed)+'.json'),restored)
        return outcome
    def qualify(self,job,event):
        """Finish the same media/export/backup admission for fresh or interrupted collection."""
        seed=job['layout_seed'];run=ROOT/f'datasets/counterfactual-targets/{job["split"]}/layout-{seed}'
        verify_sources(run,self.cut,getattr(self,'collector_repo',REPO));media=None;exports={};media_stage=None;media_receipt=None
        if read(run/'summary.json').get('complete') is True:
            python=profile()['export_python']
            media_stage=self.stage('media-'+str(seed),[python,'-B',str(SELF.with_name('triplet_media.py')),str(run),'--output',str(run/'production-media.json')],export=True)
            if media_stage['returncode'] not in (0,2):raise RuntimeError('Media checker did not finish')
            media=read(run/'production-media.json')
            media_receipt=dict(path='production-media.json',sha256=sha(run/'production-media.json'),written_wall_ns=(run/'production-media.json').stat().st_mtime_ns)
            if (media_stage['returncode']==0)!=(media.get('passed') is True):raise RuntimeError('Media checker exit/verdict differs')
            if media['passed']:
                for existing in (False,True):
                    name='report-validation' if existing else 'report'
                    args=['bash',str(self.wrapper),'export',str(self.instance),str(run),'--repo-id','local/counterfactual_'+str(seed),'--report-dir',str(run/name)]
                    if existing:args.append('--validate-existing')
                    exports[name]=self.stage(name+'-'+str(seed),args,export=True)
                    if exports[name]['returncode']:raise RuntimeError('Export checker failed')
        status,reason,group=classify(run,job,event,media,exports)
        files=manifest(run);proof=dict(version=1,cut=self.cut,job=job,stage=event,status=status,reason=reason,
            raw_root=str(run),files_sha256=files,media=media,media_stage=media_stage,media_receipt=media_receipt,export_stages=exports,group=group)
        destination=DUAL/'datasets/counterfactual-targets'/job['split']/run.name
        with (self.root/('backup-'+str(seed)+'.log')).open('x') as log:proof['backup']=backup(run,files,destination,log)
        path=self.global_root/'proofs'/job['split']/(str(seed)+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError('Disposition proof cannot be replaced')
        atomic_json(path,proof)
        outcome={k:job[k] for k in ('split','requested_layout_seed','layout_seed')}
        outcome.update(status=status,reason=reason,proof_path=str(path),proof_sha256=sha(path))
        atomic_json(self.root/('outcome-'+str(seed)+'.json'),outcome)
        return outcome
    def run(self,jobs):
        self.startup_retries=[]
        error=None;failure=None;backup_result=None;backup_error=None;retired=False;retirement_errors=[]
        try:
            if self.plan.get('startup_retry_budget'):jobs=startup_retries.resume_jobs(self.plan['startup_retry_budget'],self.instance,jobs)
            for job in jobs:
                self.active_job=job
                self.check()
                self.ensure_ready(job)
                self.attempt(job)
                self.active_job=None
        except BaseException as exc:
            failure=failure_report(self.supervision,exc);error=failure['error']
            publish_stop(self.global_root,dict(failure,slot=self.instance,wall_ns=time.time_ns()))
        finally:
            try:self.down();retired=True
            except BaseException as exc:
                down_failure=failure_report(self.supervision,exc);retirement_errors.append(down_failure);error=error or down_failure['error'];failure=failure or down_failure
                publish_stop(self.global_root,dict(down_failure,slot=self.instance,source='owned-down',wall_ns=time.time_ns()))
            if error is not None and retired and self.active_job is not None:
                try:backup_result=backup_terminal(self.plan,self.active_job,error,self.root)
                except BaseException as exc:
                    backup_error=repr(exc);detail=failure_report(self.supervision,exc);retirement_errors.append(detail)
                    publish_stop(self.global_root,dict(detail,source='terminal backup',slot=self.instance,wall_ns=time.time_ns()))
            failure,latest=retirement_supervision(self.supervision,failure)
            if failure is not None:
                error=failure['error'];publish_stop(self.global_root,dict(failure,source='worker retirement supervision',slot=self.instance,wall_ns=time.time_ns()))
            result=dict(passed=error is None,error=error,retirement_errors=retirement_errors,supervision_after_retirement=latest,instance=self.instance,wave=self.wave,cut=self.cut,sessions=self.session+1,
                startup_retries=self.startup_retries,
                terminal_backup=backup_result,terminal_backup_error=backup_error,
                supervision_failure=failure if error is not None else None,completed_wall_ns=time.time_ns())
            atomic_json(self.root/'worker-finalization.json',result)
        if error:raise RuntimeError(error)


def verify_outcome(outcome,recipe):
    path=Path(outcome['proof_path']);checked_tree(path.parent)
    if confined(path)!=path:raise ValueError('Disposition proof path is not canonical')
    if sha(path)!=outcome['proof_sha256']:raise ValueError('Disposition proof changed')
    proof=read(path)
    if str(proof.get('kind','')).startswith('interrupted_export_'):raise ValueError('Interrupted export errors are not completed gate verdicts')
    if proof.get('kind')=='acknowledged_peer_interruption':
        from triplet_continuation import verify_interruption
        return verify_interruption(outcome,proof,recipe)
    job=proof['job'];run=Path(proof['raw_root'])
    if proof['cut']!=recipe['cut'] or manifest(run)!=proof['files_sha256']:raise ValueError('Sealed group bytes/source cut changed')
    verify_sources(run,recipe['cut'],recipe.get('collector_repo'))
    actual=classify(run,job,proof['stage'],proof['media'],proof['export_stages'])
    if (actual[0],actual[1])!=(outcome['status'],outcome['reason']):raise ValueError('Outcome differs from verified raw disposition')
    if any(outcome[k]!=job[k] for k in ('split','layout_seed','requested_layout_seed')):raise ValueError('Disposition changed membership')
    if recipe.get('continuation_admission_path') and job['layout_seed'] in (720087,720092):
        from triplet_continuation import require_completed_checks
        require_completed_checks(proof)
    backup_proof=proof['backup']
    expected=DUAL/'datasets/counterfactual-targets'/job['split']/run.name
    if backup_proof.get('passed') is not True or backup_proof['files_sha256']!=proof['files_sha256'] or backup_proof['destination']!=str(expected):
        raise ValueError('Source-bound SHA backup is missing')
    return proof


def registries(recipe,state):
    if not state['complete']:raise ValueError('Incomplete production cannot publish a registry')
    root=Path(recipe['production_root']);result={}
    publication_gate(root,None)
    rebuilt=reduce_waves(read(root/'wave-ledger.json'))
    if state!=rebuilt:raise ValueError('Registry state differs from the deterministic source-bound wave ledger')
    for split in ('train','heldout'):
        layouts=[];failed=[]
        for item in state['selected']:
            if item['split']!=split:continue
            outcome={k:item[k] for k in ('split','layout_seed','requested_layout_seed','proof_path','proof_sha256')}
            outcome.update(status='selected',reason='accepted');proof=verify_outcome(outcome,recipe)
            layouts.append(selected_entry(item,Path(proof['raw_root']),proof['files_sha256']))
        for item in state['failed_groups']:
            if item['split']!=split:continue
            outcome={k:item[k] for k in ('split','layout_seed','requested_layout_seed','proof_path','proof_sha256')}
            outcome.update(status='refused',reason=item['disposition']);proof=verify_outcome(outcome,recipe)
            failed.append({**{k:item[k] for k in ('requested_layout_seed','layout_seed','split','proof_path','proof_sha256','disposition')},
                'raw_root':proof['raw_root'],'raw_files_sha256':proof['files_sha256']})
        final=read(root/'production-finalization.json')
        if final.get('passed') is not True or (root/'STOP.json').exists():raise ValueError('Production retirement failed')
        result[split]=dict(version=1,sealed=True,role='authoritative-rtx-native',cut=recipe['cut'],plan_sha256=sha(root/'plan.json'),
            production_root=str(root),finalization_path=str(root/'production-finalization.json'),finalization_sha256=sha(root/'production-finalization.json'),
            wave_ledger_path=str(root/'wave-ledger.json'),wave_ledger_sha256=sha(root/'wave-ledger.json'),
            layouts=sorted(layouts,key=lambda x:x['requested_layout_seed']),failed_groups=failed,
            replacement_allocations=[a for a in state['replacement_allocations'] if (a['requested_layout_seed']<730000)==(split=='train')])
        if recipe.get('startup_retry_budget'):
            current=startup_retries.summary(recipe['startup_retry_budget'])
            if final.get('startup_retries')!=current:raise ValueError('Final startup retry evidence differs from its ledger')
            result[split]['startup_retry_budget']=dict(admission_binding=recipe['startup_retry_budget'],final=current)
    return result


def backup_projection(registry,authoritative_sha):
    """A separately named derivative preserves the identity of RTX originals."""
    import copy
    import re
    if not re.fullmatch('[0-9a-f]{64}',authoritative_sha):raise ValueError('Authoritative registry SHA is required')
    if registry.get('role')!='authoritative-rtx-native' or not registry.get('sealed'):
        raise ValueError('Only an authoritative sealed registry may be projected')
    out=copy.deepcopy(registry)
    def translate(value):
        path=Path(value)
        if not path.is_relative_to(ROOT):raise ValueError('Non-RTX registry path: '+str(path))
        return str(DUAL/path.relative_to(ROOT))
    out.update(role='derived-dual-backup-projection',authoritative_registry_sha256=authoritative_sha,
        authoritative_registry_path=str(Path(registry['production_root'])/('train-registry.json' if registry['layouts'][0]['requested_layout_seed']<730000 else 'heldout-registry.json')))
    for key in ('production_root','finalization_path','wave_ledger_path'):out[key]=translate(out[key])
    for row in out['layouts']:
        for key in ('dataset_root','raw_root','validation_path'):row[key]=translate(row[key])
    for row in out['failed_groups']:
        for key in ('raw_root','proof_path'):row[key]=translate(row[key])
    return out


def backup_registries(recipe):
    root=Path(recipe['production_root']);destination=DUAL/root.relative_to(ROOT)
    publication_gate(root,None)
    logpath=Path(recipe['runtime_root'])/'registry-backup.log'
    with logpath.open('x') as log:
        evidence=None
        if recipe.get('continuation_admission_path'):
            from triplet_continuation import backup_evidence
            evidence=backup_evidence(recipe,log)
        receipt=backup(root,manifest(root),destination,log)
        if evidence is not None:receipt['continuation_evidence']=evidence
    # Project only the explicit path fields in a separate file. The original
    # registry bytes in the backup keep their RTX paths and published SHA.
    projections={s:backup_projection(read(root/(s+'-registry.json')),sha(root/(s+'-registry.json'))) for s in ('train','heldout')}
    with tempfile.TemporaryDirectory(prefix='registry-projection-',dir=ROOT/'runtime') as temp:
        projected=Path(temp);receipt['projections']={}
        for split,value in projections.items():
            if split not in ('train','heldout'):raise ValueError('Unknown backup projection split')
            path=projected/(split+'-registry.backup-projection.json');atomic_json(path,value)
            receipt['projections'][split]=dict(path=str(destination/path.name),sha256=sha(path),
                authoritative_registry_sha256=value['authoritative_registry_sha256'])
        with logpath.open('a') as log:receipt['projection_backup']=backup(projected,manifest(projected),destination,log)
    atomic_json(Path(recipe['runtime_root'])/'registry-backup.json',receipt)
    return receipt


def coordinate(recipe,release):
    from triplet_continuation import claim_run
    with claim_run() if recipe.get('continuation_admission_path') else nullcontext():
        supervision=capture_supervision()
        require_host();qualified_recipe(recipe);disk()
        release_bytes=read(release,True);release_proof=json.loads(release_bytes)
        if release_proof.get('passed') is not True:raise ValueError('Explicit training release certificate required')
        if gpu_uuid(0)!=ASSIGNED_UUID:raise ValueError('Assigned physical GPU differs')
        start=time.monotonic()
        while time.monotonic()-start<60:
            disk()
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU compute is not empty before START')
            time.sleep(1)
        root=confined(recipe['production_root']);root.mkdir(parents=True,exist_ok=False)
        atomic_json(root/'plan.json',recipe);atomic_json(root/'training-release.json',release_proof)
        atomic_json(root/'coordinator.json',identity(os.getpid()))
        runtime=Path(recipe['runtime_root']);runtime.mkdir(parents=True,exist_ok=False)
        if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise OSError('Coordinator subreaper unavailable')
        ledger=OwnedProcesses(runtime,time.time_ns());waves=[];error=None;children=[];verified={};cleanup=None;watch=None;jobs=[];terminal_backups=None;failure=None;retirement_errors=[];retry_report=None
        try:
            def retire_workers():
                for process,_,_ in list(children):
                    if process.poll() is None and process.pid in ledger.known:ledger.send(process.pid,signal.SIGTERM)
            watch=DiskWatch(root,retire_workers,report=runtime/'disk-monitor.json');watch.start()
            if recipe.get('continuation_admission_path'):
                from triplet_continuation import reverify
                reverify(recipe)
            while True:
                state=reduce_waves(waves)
                if state['complete']:break
                jobs=state['next_wave'];number=len(waves);path=root/f'wave-{number:02d}.json'
                inherited=recipe.get('inherited_outcomes',[]) if number==0 else []
                inherited_keys={(r['split'],r['layout_seed']) for r in inherited}
                atomic_json(path,dict(jobs=jobs,outcomes=inherited));begun=time.monotonic();children=[]
                def observe():
                    publication_gate(root,watch)
                    outcomes=inherited+[read(p) for p in sorted((root/f'wave-{number:02d}').glob('slot-*/outcome-*.json'))]
                    seen=set()
                    for outcome in outcomes:
                        member=(outcome['split'],outcome['layout_seed'])
                        if member in seen:raise ValueError('Duplicate live group disposition')
                        seen.add(member)
                        expected=next((j for j in jobs if (j['split'],j['layout_seed'])==member),None)
                        if expected is None or expected['requested_layout_seed']!=outcome['requested_layout_seed']:raise ValueError('Orphan live group disposition')
                        key=(outcome['proof_path'],outcome['proof_sha256'])
                        if key not in verified:
                            proof=verify_outcome(outcome,recipe)
                            if proof['job']!=expected:raise ValueError('Proof changed its assigned deterministic job')
                            verified[key]=True
                    guards([r for w in waves for r in w['outcomes']]+outcomes,state['replacement_allocations'])
                    for outcome in outcomes:
                        if (outcome['split'],outcome['layout_seed']) in inherited_keys:continue
                        planned=next((j for j in jobs if j['layout_seed']==outcome['layout_seed'] and j['split']==outcome['split']),None)
                        if planned is None:raise ValueError('Unplanned worker outcome')
                        target=root/f'wave-{number:02d}/slot-{planned["instance"]}'/('admitted-'+str(outcome['layout_seed'])+'.json')
                        if not target.exists():atomic_json(target,outcome)
                    atomic_json(root/'progress.json',dict(wave=number,outcomes=outcomes,prior_selected=len(state['selected']),wall_ns=time.time_ns()))
                    return outcomes
                for ordinal,slot in enumerate((3,4,5,6)):
                    if not any(j['instance']==slot and (j['split'],j['layout_seed']) not in inherited_keys for j in jobs):continue
                    while time.monotonic()-begun<ordinal*recipe['start_stagger_s']:
                        ledger.scan();observe()
                        if (root/'STOP.json').exists():raise RuntimeError('Worker stopped during stagger')
                        time.sleep(.2)
                    log=(root/f'wave-{number:02d}-slot-{slot}.log').open('x')
                    publication_gate(root,watch)
                    process=subprocess.Popen([sys.executable,'-B',str(SELF),'--worker',str(slot),'--wave',str(number),'--recipe',str(root/'plan.json')],
                        cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    ledger.add(process.pid);children.append((process,log,slot))
                def retired(slot):
                    final=read(root/f'wave-{number:02d}/slot-{slot}/worker-finalization.json')
                    if (final.get('passed') is not True or final.get('cut')!=recipe['cut'] or final.get('instance')!=slot or
                        final.get('wave')!=number):raise RuntimeError('Worker did not finalize: '+str(slot))
                    for session in range(final['sessions']):
                        down=read(root/f'wave-{number:02d}/slot-{slot}/owned-down-{session}.json')
                        if down.get('passed') is not True or down.get('errors') or down.get('cleanup',{}).get('remaining')!=[] or any(down['ports_listening'].values()):
                            raise RuntimeError('Owned instance cleanup is unresolved: '+str(slot))
                outcomes=wait_workers(children,observe,ledger.scan,retired,root/'STOP.json')
                for _,log,_ in children:log.close()
                wave=dict(jobs=jobs,outcomes=outcomes);reduce_waves(waves+[wave]);waves.append(wave)
                atomic_json(path,wave);atomic_json(root/'wave-ledger.json',waves)
            cleanup=ledger.stop()
            if cleanup['remaining']:raise RuntimeError('Coordinator descendants remain alive')
            finals={str(p.relative_to(root)):sha(p) for p in sorted(root.glob('wave-*/slot-*/worker-finalization.json'))}
            if any(read(root/p).get('passed') is not True for p in finals):raise RuntimeError('Worker finalization failed')
            retry_report=startup_retries.summary(recipe['startup_retry_budget']) if recipe.get('startup_retry_budget') else None
            atomic_json(root/'production-finalization.json',dict(passed=True,error=None,cleanup=cleanup,worker_finalizations_sha256=finals,startup_retries=retry_report,wall_ns=time.time_ns()))
            for split,registry in registries(recipe,state).items():
                publication_gate(root,watch);atomic_json(root/(split+'-registry.json'),registry)
            # Publish the independent caller seals only after finalization and both
            # complete registries exist. No checkpoint can substitute for this file.
            publication_gate(root,watch);atomic_json(root/'REGISTRY-SHA256.json',{s:sha(root/(s+'-registry.json')) for s in ('train','heldout')})
            backup_registries(recipe)
            publication_gate(root,watch);watch.close();publication_gate(root,watch)
            publication_ledger=OwnedProcesses(runtime,time.time_ns());publication_ledger.scan();publication_cleanup=publication_ledger.stop()
            if publication_cleanup['remaining']:raise RuntimeError('Publication/backup children remain alive')
            atomic_json(runtime/'publication-cleanup.json',dict(passed=True,cleanup=publication_cleanup))
            publication_gate(root,watch)
            final_failure,latest=retirement_supervision(supervision)
            if final_failure is not None:raise RuntimeError(final_failure['error'])
            atomic_json(root/'PUBLICATION-COMPLETE.json',dict(passed=True,supervision_after_retirement=latest,registries=read(root/'REGISTRY-SHA256.json'),
                finalization_sha256=sha(root/'production-finalization.json'),backup_receipt_sha256=sha(Path(recipe['runtime_root'])/'registry-backup.json'),
                disk_monitor_path=str(runtime/'disk-monitor.json'),disk_monitor_sha256=sha(runtime/'disk-monitor.json'),
                publication_cleanup_path=str(runtime/'publication-cleanup.json'),publication_cleanup_sha256=sha(runtime/'publication-cleanup.json'),wall_ns=time.time_ns()))
        except BaseException as exc:
            failure=failure_report(supervision,exc);error=failure['error']
            publish_stop(root,dict(failure,wall_ns=time.time_ns()))
            # Let each worker certify its own shutdown; parent pidfds remain the
            # final bound cleanup if a worker cannot retire within its watchdog.
            end=time.monotonic()+150
            while any(p.poll() is None for p,_,_ in children) and time.monotonic()<end:ledger.scan();time.sleep(.2)
        finally:
            if watch is not None and not watch.closed:
                try:watch.close()
                except BaseException as exc:
                    detail=failure_report(supervision,exc);retirement_errors.append(detail);error=error or detail['error'];failure=failure or detail
                    publish_stop(root,dict(detail,source='disk monitor retirement',wall_ns=time.time_ns()))
            if cleanup is None:
                try:cleanup=ledger.stop()
                except BaseException as exc:
                    detail=failure_report(supervision,exc);retirement_errors.append(detail);error=error or detail['error'];failure=failure or detail
                    cleanup=dict(remaining=[dict(unverified=repr(exc))])
                    publish_stop(root,dict(detail,source='coordinator cleanup',wall_ns=time.time_ns()))
            if error is not None:
                if not cleanup['remaining']:
                    try:terminal_backups=backup_remaining_terminal(recipe,jobs,error)
                    except BaseException as exc:
                        detail=failure_report(supervision,exc);retirement_errors.append(detail)
                        terminal_backups=dict(errors=[detail]);publish_stop(root,dict(detail,source='terminal backup retirement',wall_ns=time.time_ns()))
                else:terminal_backups=dict(errors=[dict(error='Owned writers remain; immutable diagnostic copy is not yet possible')])
                try:
                    tail=OwnedProcesses(runtime,time.time_ns());tail.scan();tail_cleanup=tail.stop()
                except BaseException as exc:
                    retirement_errors.append(failure_report(supervision,exc));tail_cleanup=dict(remaining=[dict(unverified=repr(exc))])
                failure,latest=retirement_supervision(supervision,failure)
                if failure is not None:
                    error=failure['error'];publish_stop(root,dict(failure,source='final retirement supervision',wall_ns=time.time_ns()))
                if recipe.get('startup_retry_budget'):
                    try:retry_report=startup_retries.summary(recipe['startup_retry_budget'])
                    except BaseException as exc:retirement_errors.append(failure_report(supervision,exc))
                atomic_json(root/'production-finalization.json',dict(passed=False,error=error,supervision_failure=failure,retirement_errors=retirement_errors,supervision_after_retirement=latest,cleanup=cleanup,terminal_backups=terminal_backups,startup_retries=retry_report,
                    backup_children_cleanup=tail_cleanup,wall_ns=time.time_ns()))
        if error:raise RuntimeError(error)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cut');parser.add_argument('--recipe',type=Path)
    parser.add_argument('--wave',type=int);parser.add_argument('--worker',type=int,choices=(3,4,5,6));parser.add_argument('--release-file',type=Path)
    parser.add_argument('--continuation',type=Path);parser.add_argument('--continuation-sha256')
    parser.add_argument('--prepare-continuation',type=Path);parser.add_argument('--collector-repo',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--plan',action='store_true');parser.add_argument('--go',action='store_true');args=parser.parse_args()
    require_host()
    if args.worker is not None:
        recipe=read(args.recipe)
        if recipe.get('continuation_admission_path'):
            from triplet_continuation import load_plan
            expected_recipe=load_plan(recipe['continuation_admission_path'],recipe['continuation_admission_sha256'],recipe['runner_cut'])
        else:expected_recipe=plan(recipe['cut'])
        if recipe!=expected_recipe:raise ValueError('Worker recipe changed')
        expected=read(Path(recipe['production_root'])/'coordinator.json');parent=identity(os.getppid())
        if parent is None or any(parent[k]!=expected[k] for k in ('pid','start_ticks')):raise RuntimeError('Worker must be spawned by its recorded live coordinator')
        if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError('Parent retirement signal unavailable')
        if os.getppid()!=expected['pid']:raise RuntimeError('Coordinator retired during worker startup')
        wave=read(Path(recipe['production_root'])/f'wave-{args.wave:02d}.json')
        worker=Worker(recipe,args.worker,args.wave)
        signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Owned worker stop')))
        inherited=recipe.get('inherited_outcomes',[]) if args.wave==0 else []
        inherited_keys={(r['split'],r['layout_seed']) for r in inherited}
        prior=[r for j in wave['jobs'] for r in inherited if (r['split'],r['layout_seed'])==(j['split'],j['layout_seed']) and j['instance']==args.worker]
        for row in prior:
            if row['status']=='selected':worker.reason=None;worker.streak=0
            else:worker.streak=worker.streak+1 if worker.reason==row['reason'] else 1;worker.reason=row['reason']
        worker.run([j for j in wave['jobs'] if j['instance']==args.worker and (j['split'],j['layout_seed']) not in inherited_keys]);return
    if args.cut is None:parser.error('--cut is required')
    if args.prepare_continuation:
        if args.collector_repo is None or args.output is None:parser.error('Preparation requires --collector-repo and --output')
        from triplet_continuation import prepare
        print(json.dumps(prepare(args.prepare_continuation,args.collector_repo,args.cut,args.output,args.checkpoint_sha256),indent=2));return
    if args.continuation:
        from triplet_continuation import load_plan
        recipe=load_plan(args.continuation,args.continuation_sha256,args.cut)
    else:recipe=plan(args.cut)
    if args.plan and not args.go:print(json.dumps(recipe,indent=2));return
    if not args.go or args.release_file is None:parser.error('--go requires --release-file')
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise RuntimeError('Start this runner with nice -n 10')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Owned production stop')))
    coordinate(recipe,confined(args.release_file))


if __name__=='__main__':main()
