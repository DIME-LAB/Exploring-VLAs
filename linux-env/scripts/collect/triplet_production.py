#!/usr/bin/env python3
"""Four owned RTX instances, source-verified RGB groups and deterministic reserves.

Print the immutable plan with --plan. --go additionally requires the explicit
training-release certificate and starts the four workers with a 60 s stagger.
"""
import argparse,ctypes,hashlib,json,os,select,shlex,signal,socket,stat,subprocess,sys,time,threading
from pathlib import Path
sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'vla_SO-ARM101/src/so_arm101_control'))
from storage import CANONICAL_ROOT,atomic_json,confined
from host_config import profile
from run_smoke import InstanceClaim,OwnedProcesses,frozen,identity
from gpu_device import gpu_uuid
from triplet_production_plan import initial_wave,reduce_waves,guards,POLICY
from triplet_artifacts import read,sha,manifest,classify,selected_entry,checked_tree
import clock_status

ROOT=Path('/mnt/storage3/aaugus11/exploring-vlas')
DUAL=Path('/mnt/a4500-backup/exploring-vlas/datasets/counterfactual-production-v1')
ASSIGNED_UUID='GPU-c2772797-ad71-f07e-0101-db0f188f3504'
SELF=Path(__file__).resolve()
SSH=['ssh','-oBatchMode=yes','-oConnectTimeout=15','-oStrictHostKeyChecking=yes','-oHostKeyAlias=dual-a4500',
    '-oControlMaster=no','-oControlPath=none','-oControlPersist=no',
    '-oUserKnownHostsFile='+str(ROOT/'setup/dual-known_hosts'),'aaugus11@10.179.23.212']


def require_host():
    if profile()['name']!='rtx6000' or CANONICAL_ROOT!=ROOT:raise ValueError('Production is qualified only on rtx6000')


def plan(cut):
    require_host();frozen(cut)
    return dict(version=1,cut=cut,source_sha256=sha(SELF),policy=POLICY,jobs=initial_wave(),
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


def verify_sources(run,cut):
    config=read(run/'run.json')
    if config.get('git_commit')!=cut or config.get('cpu_fixture_only'):raise ValueError('Collector checkout/provenance differs')
    if not config.get('source_sha256'):raise ValueError('Loaded source manifest is missing')
    for name,value in config['source_sha256'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or sha(REPO/path)!=value:raise ValueError('Loaded collector source differs: '+name)


def backup(source,files,destination,log):
    """Copy immutable files only and verify every byte remotely before selection."""
    if not Path(destination).is_relative_to(DUAL):raise ValueError('Backup outside the assigned projection root')
    for name,value in files.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or '\n' in name or not name or sha(source/name)!=value:
            raise ValueError('Backup source changed: '+name)
    subprocess.run([*SSH,'mkdir -p '+shlex.quote(str(destination))],check=True,stdout=log,stderr=subprocess.STDOUT)
    subprocess.run(['rsync','-a','--partial','-e',shlex.join(SSH[:-1]),str(source)+'/',SSH[-1]+':'+str(destination)+'/'],check=True,stdout=log,stderr=subprocess.STDOUT)
    text=''.join(value+'  '+name+'\n' for name,value in sorted(files.items()))
    subprocess.run([*SSH,'cd '+shlex.quote(str(destination))+' && sha256sum -c -'],input=text,text=True,check=True,stdout=log,stderr=subprocess.STDOUT)
    if manifest(source)!=files:raise ValueError('Source changed during SHA backup')
    return dict(passed=True,source=str(source),destination=str(destination),files_sha256=files,completed_wall_ns=time.time_ns())


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


def backup_remaining_terminal(recipe,jobs,error):
    """Coordinator fallback after every descendant has been proven gone."""
    root=Path(recipe['production_root']);records=[];errors=[]
    for job in jobs:
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
    return dict(records=records,errors=errors)


class Worker:
    def __init__(self,recipe,instance,wave):
        self.plan=recipe;self.instance=instance;self.wave=wave;self.cut=recipe['cut'];self.port=8777+2*instance
        self.root=confined(Path(recipe['production_root'])/f'wave-{wave:02d}/slot-{instance}')
        self.root.mkdir(parents=True,exist_ok=False);self.runtime=None;self.ledger=None;self.claim=None;self.stages=[];self.native=None
        self.global_root=Path(recipe['production_root']);self.wrapper=REPO/'linux-env/scripts/collect_sim.sh'
        self.session=-1;self.stop_file=self.global_root/'STOP.json';self.reason=None;self.streak=0
        self.active_job=None
        self.retirement_certified=True
    def check(self):
        frozen(self.cut)
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
            for key in ('PYTHONPATH','LD_LIBRARY_PATH','AMENT_PREFIX_PATH','CMAKE_PREFIX_PATH','COLCON_PREFIX_PATH','ROS_DISTRO','ROS_VERSION','ROS_PYTHON_VERSION'):env.pop(key,None)
        with (self.root/(name+'.log')).open('x') as log:
            process=subprocess.Popen(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(self.claim.fd,))
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
        if (result['nonce']!=nonce or owned is None or now is None or fd is None or select.select([fd],[],[],0)[0] or
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
        pin=read(ROOT/'setup/bootstrap-gpu-table.json');table=Path(pin['source']['path'])
        if pin.get('passed') is not True or pin.get('assigned_gpu_uuid')!=ASSIGNED_UUID or sha(table)!=pin['source']['sha256']:
            raise RuntimeError('Qualified RTX Kit table changed')
        (self.runtime/'isaac.log').write_bytes(read(table,True))
        self.env=dict(os.environ,ISAACSIM_GPU='0',SOARM_SIM_NICE='0',SOARM_PHYSICS_THREADS='4',SOARM_KIT_THREADS='6',
            SOARM_INSTANCE_ROOT=str(self.runtime),SOARM_COLLECTION_ROOT=str(ROOT),HF_LEROBOT_HOME=str(ROOT/'datasets'),
            PYTHONDONTWRITEBYTECODE='1',PYTHONFAULTHANDLER='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            SOARM_EVAL_LOCALHOST_ONLY='1',SOARM_INSTANCE_CLAIM_FD=str(self.claim.fd),ISAACSIM_LOG=str(self.runtime/'isaac.log'),
            CUDA_VISIBLE_DEVICES=ASSIGNED_UUID,SOARM_ASSIGNED_GPU_UUID=ASSIGNED_UUID)
        for name,relative in dict(TMPDIR='tmp',TEMP='tmp',TMP='tmp',XDG_CACHE_HOME='cache',HF_HOME='cache/huggingface',ROS_LOG_DIR='ros-logs').items():
            path=self.runtime/relative;path.mkdir(parents=True,exist_ok=True);self.env[name]=str(path)
        for key in ('ISAACSIM_EXTRA_ARGS','ROS_LOCALHOST_ONLY','DISPLAY','XAUTHORITY'):self.env.pop(key,None)
        disk();event=self.stage('up-'+str(self.session),['bash',str(self.wrapper),'up',str(self.instance)],timeout=1500)
        if event['returncode']:raise RuntimeError('Startup failed; see '+event['name']+'.log')
        # The first challenge establishes ownership; the second binds before and
        # after the RPC. Clock qualification installs this checkout's runtime.
        self.rpc('result=True')
        self.rpc('result=True')
        cert=self.root/f'clock-{self.session}.json'
        event=self.stage('clock-'+str(self.session),['bash',str(self.wrapper),'clock-sanity',str(self.instance),'--output',str(cert)],timeout=1200)
        clock_status.require(cert,event['pid'],event['started_wall_ns'],event['returncode'])
        import stage_readiness
        stage_readiness.wait(lambda:dict(stage_readiness=self.rpc(stage_readiness.probe_code())),
            lambda p:atomic_json(self.root/f'readiness-{self.session}.json',p))
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
        verify_sources(run,self.cut);media=None;exports={}
        if read(run/'summary.json').get('complete') is True:
            python=profile()['export_python']
            media_stage=self.stage('media-'+str(seed),[python,'-B',str(SELF.with_name('triplet_media.py')),str(run),'--output',str(run/'production-media.json')],export=True)
            if media_stage['returncode'] not in (0,2):raise RuntimeError('Media checker did not finish')
            media=read(run/'production-media.json')
            if media['passed']:
                for existing in (False,True):
                    name='report-validation' if existing else 'report'
                    args=['bash',str(self.wrapper),'export',str(self.instance),str(run),'--repo-id','local/counterfactual_'+str(seed),'--report-dir',str(run/name)]
                    if existing:args.append('--validate-existing')
                    exports[name]=self.stage(name+'-'+str(seed),args,export=True)
                    if exports[name]['returncode']:raise RuntimeError('Export checker failed')
        status,reason,group=classify(run,job,event,media,exports)
        files=manifest(run);proof=dict(version=1,cut=self.cut,job=job,stage=event,status=status,reason=reason,
            raw_root=str(run),files_sha256=files,media=media,export_stages=exports,group=group)
        destination=DUAL/'datasets/counterfactual-targets'/job['split']/run.name
        with (self.root/('backup-'+str(seed)+'.log')).open('x') as log:proof['backup']=backup(run,files,destination,log)
        path=self.global_root/'proofs'/job['split']/(str(seed)+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError('Disposition proof cannot be replaced')
        atomic_json(path,proof)
        outcome={k:job[k] for k in ('split','requested_layout_seed','layout_seed')}
        outcome.update(status=status,reason=reason,proof_path=str(path),proof_sha256=sha(path))
        atomic_json(self.root/('outcome-'+str(seed)+'.json'),outcome)
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
    def run(self,jobs):
        error=None;backup_result=None;backup_error=None;retired=False
        try:
            for job in jobs:
                self.active_job=job
                self.check()
                if self.claim is None:self.up()
                self.attempt(job)
                self.active_job=None
        except BaseException as exc:
            error=repr(exc)
            publish_stop(self.global_root,dict(error=error,slot=self.instance,wall_ns=time.time_ns()))
        finally:
            try:self.down();retired=True
            except BaseException as exc:
                error=error or repr(exc)
                publish_stop(self.global_root,dict(error=repr(exc),slot=self.instance,source='owned-down',wall_ns=time.time_ns()))
            if error is not None and retired and self.active_job is not None:
                try:backup_result=backup_terminal(self.plan,self.active_job,error,self.root)
                except BaseException as exc:
                    backup_error=repr(exc);publish_stop(self.global_root,dict(error=backup_error,source='terminal backup',slot=self.instance,wall_ns=time.time_ns()))
            result=dict(passed=error is None,error=error,instance=self.instance,wave=self.wave,cut=self.cut,sessions=self.session+1,
                terminal_backup=backup_result,terminal_backup_error=backup_error,completed_wall_ns=time.time_ns())
            atomic_json(self.root/'worker-finalization.json',result)
        if error:raise RuntimeError(error)


def verify_outcome(outcome,recipe):
    path=Path(outcome['proof_path']);checked_tree(path.parent)
    if confined(path)!=path:raise ValueError('Disposition proof path is not canonical')
    if sha(path)!=outcome['proof_sha256']:raise ValueError('Disposition proof changed')
    proof=read(path);job=proof['job'];run=Path(proof['raw_root'])
    if proof['cut']!=recipe['cut'] or manifest(run)!=proof['files_sha256']:raise ValueError('Sealed group bytes/source cut changed')
    verify_sources(run,recipe['cut'])
    actual=classify(run,job,proof['stage'],proof['media'],proof['export_stages'])
    if (actual[0],actual[1])!=(outcome['status'],outcome['reason']):raise ValueError('Outcome differs from verified raw disposition')
    if any(outcome[k]!=job[k] for k in ('split','layout_seed','requested_layout_seed')):raise ValueError('Disposition changed membership')
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
    with logpath.open('x') as log:receipt=backup(root,manifest(root),destination,log)
    # Project only the explicit path fields in a separate file. The original
    # registry bytes in the backup keep their RTX paths and published SHA.
    projections={s:backup_projection(read(root/(s+'-registry.json')),sha(root/(s+'-registry.json'))) for s in ('train','heldout')}
    code="""import os,json,hashlib,sys
from pathlib import Path
root=Path(sys.argv[1]);values=json.load(sys.stdin);result={}
assert root.is_relative_to(Path('/mnt/a4500-backup/exploring-vlas/datasets/counterfactual-production-v1'))
for split,value in values.items():
 assert split in ('train','heldout')
 path=root/(split+'-registry.backup-projection.json')
 data=json.dumps(value,indent=2).encode()
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 result[split]={'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'authoritative_registry_sha256':value['authoritative_registry_sha256']}
fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(result))
"""
    reply=subprocess.check_output([*SSH,'python3 -c '+shlex.quote(code)+' '+shlex.quote(str(destination))],input=json.dumps(projections),text=True)
    receipt['projections']=json.loads(reply);atomic_json(Path(recipe['runtime_root'])/'registry-backup.json',receipt)
    return receipt


def coordinate(recipe,release):
    require_host();frozen(recipe['cut']);disk()
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
    ledger=OwnedProcesses(runtime,time.time_ns());waves=[];error=None;children=[];verified={};cleanup=None;watch=None;jobs=[];terminal_backups=None
    try:
        def retire_workers():
            for process,_,_ in list(children):
                if process.poll() is None and process.pid in ledger.known:ledger.send(process.pid,signal.SIGTERM)
        watch=DiskWatch(root,retire_workers,report=runtime/'disk-monitor.json');watch.start()
        while True:
            state=reduce_waves(waves)
            if state['complete']:break
            jobs=state['next_wave'];number=len(waves);path=root/f'wave-{number:02d}.json'
            atomic_json(path,dict(jobs=jobs,outcomes=[]));begun=time.monotonic();children=[]
            def observe():
                publication_gate(root,watch)
                outcomes=[read(p) for p in sorted((root/f'wave-{number:02d}').glob('slot-*/outcome-*.json'))]
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
                    planned=next((j for j in jobs if j['layout_seed']==outcome['layout_seed'] and j['split']==outcome['split']),None)
                    if planned is None:raise ValueError('Unplanned worker outcome')
                    target=root/f'wave-{number:02d}/slot-{planned["instance"]}'/('admitted-'+str(outcome['layout_seed'])+'.json')
                    if not target.exists():atomic_json(target,outcome)
                atomic_json(root/'progress.json',dict(wave=number,outcomes=outcomes,prior_selected=len(state['selected']),wall_ns=time.time_ns()))
                return outcomes
            for ordinal,slot in enumerate((3,4,5,6)):
                if not any(j['instance']==slot for j in jobs):continue
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
        atomic_json(root/'production-finalization.json',dict(passed=True,error=None,cleanup=cleanup,worker_finalizations_sha256=finals,wall_ns=time.time_ns()))
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
        atomic_json(root/'PUBLICATION-COMPLETE.json',dict(passed=True,registries=read(root/'REGISTRY-SHA256.json'),
            finalization_sha256=sha(root/'production-finalization.json'),backup_receipt_sha256=sha(Path(recipe['runtime_root'])/'registry-backup.json'),
            disk_monitor_path=str(runtime/'disk-monitor.json'),disk_monitor_sha256=sha(runtime/'disk-monitor.json'),
            publication_cleanup_path=str(runtime/'publication-cleanup.json'),publication_cleanup_sha256=sha(runtime/'publication-cleanup.json'),wall_ns=time.time_ns()))
    except BaseException as exc:
        error=repr(exc)
        publish_stop(root,dict(error=error,wall_ns=time.time_ns()))
        # Let each worker certify its own shutdown; parent pidfds remain the
        # final bound cleanup if a worker cannot retire within its watchdog.
        end=time.monotonic()+150
        while any(p.poll() is None for p,_,_ in children) and time.monotonic()<end:ledger.scan();time.sleep(.2)
    finally:
        if watch is not None and not watch.closed:
            try:watch.close()
            except BaseException as exc:error=error or repr(exc);publish_stop(root,dict(error=repr(exc),source='disk monitor retirement',wall_ns=time.time_ns()))
        if cleanup is None:
            try:cleanup=ledger.stop()
            except BaseException as exc:
                error=error or repr(exc);cleanup=dict(remaining=[dict(unverified=repr(exc))])
                publish_stop(root,dict(error=repr(exc),source='coordinator cleanup',wall_ns=time.time_ns()))
        if error is not None:
            if not cleanup['remaining']:terminal_backups=backup_remaining_terminal(recipe,jobs,error)
            else:terminal_backups=dict(errors=[dict(error='Owned writers remain; immutable diagnostic copy is not yet possible')])
            try:
                tail=OwnedProcesses(runtime,time.time_ns());tail.scan();tail_cleanup=tail.stop()
            except BaseException as exc:tail_cleanup=dict(remaining=[dict(unverified=repr(exc))])
            atomic_json(root/'production-finalization.json',dict(passed=False,error=error,cleanup=cleanup,terminal_backups=terminal_backups,
                backup_children_cleanup=tail_cleanup,wall_ns=time.time_ns()))
    if error:raise RuntimeError(error)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cut');parser.add_argument('--recipe',type=Path)
    parser.add_argument('--wave',type=int);parser.add_argument('--worker',type=int,choices=(3,4,5,6));parser.add_argument('--release-file',type=Path)
    parser.add_argument('--plan',action='store_true');parser.add_argument('--go',action='store_true');args=parser.parse_args()
    require_host()
    if args.worker is not None:
        recipe=read(args.recipe)
        if recipe!=plan(recipe['cut']):raise ValueError('Worker recipe changed')
        expected=read(Path(recipe['production_root'])/'coordinator.json');parent=identity(os.getppid())
        if parent is None or any(parent[k]!=expected[k] for k in ('pid','start_ticks')):raise RuntimeError('Worker must be spawned by its recorded live coordinator')
        if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError('Parent retirement signal unavailable')
        if os.getppid()!=expected['pid']:raise RuntimeError('Coordinator retired during worker startup')
        wave=read(Path(recipe['production_root'])/f'wave-{args.wave:02d}.json')
        worker=Worker(recipe,args.worker,args.wave)
        signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Owned worker stop')))
        worker.run([j for j in wave['jobs'] if j['instance']==args.worker]);return
    if args.cut is None:parser.error('--cut is required')
    recipe=plan(args.cut)
    if args.plan and not args.go:print(json.dumps(recipe,indent=2));return
    if not args.go or args.release_file is None:parser.error('--go requires --release-file')
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise RuntimeError('Start this runner with nice -n 10')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Owned production stop')))
    coordinate(recipe,confined(args.release_file))


if __name__=='__main__':main()
