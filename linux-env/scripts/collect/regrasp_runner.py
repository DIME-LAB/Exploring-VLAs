#!/usr/bin/env python3
"""Own one instance; select only certified RGB recovery cohorts for production."""
import argparse,ast,copy,ctypes,hashlib,json,math,os,re,signal,socket,subprocess,time,uuid
from pathlib import Path
from contract import COLORS,parameters,timing_audit
from recovery import receipt_error
from recovery_start import POLICY,ProtectedSettle,protected_state,world_anchor,sample_snapshot
from tick_contract import digest,require_snapshots
from storage import CANONICAL_ROOT,atomic_json,confined
import collection_status,clock_status
from run_smoke import REPO,InstanceClaim,OwnedProcesses,frozen,identity

WRITER_FILES=('frames.jsonl','write-profiles.jsonl','wrist.mp4','top.mp4',
              'wrist-encode.log','top-encode.log','final-wrist.png','final-top.png')
POLICY_VERSION=1


def leaf(path):
    """Evidence leaves must be regular files, never links to another invocation."""
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:return json.load(stream)


def semantic_error(value):
    """Unwrap exception/string renderings without executing receipt text."""
    for _ in range(8):
        try:node=ast.parse(value,mode='eval').body
        except (SyntaxError,TypeError):return value
        if isinstance(node,ast.Constant) and isinstance(node.value,str):value=node.value;continue
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and
                node.func.id in ('RuntimeError','OwnershipStalled') and len(node.args)==1 and not node.keywords and
                isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str)):
            value=node.args[0].value;continue
        return value
    return value


def retired_terminal(run,stage):
    """Certify process retirement only; never turn a terminal collector into PASS."""
    c=leaf(run/'collector-finalization.json');summary=leaf(run/'summary.json')
    if (c.get('version')!=1 or c.get('completed') is not True or c.get('passed') is not False or
            c.get('cleanup_errors')!=[] or not c.get('terminal_state') or
            not re.fullmatch('[0-9a-f]{32}',str(c.get('invocation'))) or
            any(type(c.get(k)) is not int or c[k]<=0 for k in ('producer_pid','started_wall_ns','completed_wall_ns')) or
            c.get('producer_pid')!=stage['pid'] or c.get('started_wall_ns',0)<stage['started_wall_ns'] or
            c.get('completed_wall_ns',0)<c.get('started_wall_ns',1) or
            summary.get('producer_invocation')!=c['invocation'] or
            hashlib.sha256((run/'summary.json').read_bytes()).hexdigest()!=c.get('summary_sha256')):
        raise RuntimeError('Terminal producer retirement is unverified')
    return c


def velocity_only(anchor,snapshot):
    motion=protected_state(anchor,snapshot)
    if not all(c['passed'] for c in motion['base_checks']):raise RuntimeError('Protected base changed')
    if set(motion['bodies'])!=set(anchor['bodies']) or len(motion['bodies'])!=9:
        raise RuntimeError('Protected body set is incomplete')
    for body in motion['bodies'].values():
        if not all(math.isfinite(body[k]) for k in ('max_corner_motion_m','linear_m_s','angular_rad_s')) or body['max_corner_motion_m']>POLICY['protected_corner_motion_max_m']:
            raise RuntimeError('Protected displacement is a terminal fault')
    return motion


def screen_evidence(receipt,fault):
    """Replay native non-velocity contracts; a screen is the only allowed failure."""
    report=receipt['recovery_report'];baseline=report['baseline']
    if report.get('policy')!=POLICY or report.get('passed') is not False:
        raise RuntimeError('Unrecognized synthetic setup policy/stage')
    if report.get('world_anchor_sha256')!=digest(world_anchor(baseline)):
        raise RuntimeError('Protected anchor changed')
    require_snapshots([baseline])
    if not report.get('fresh_offsets') or not report.get('model'):
        raise RuntimeError('Fresh guarded shape offsets/model are missing')
    if report.get('fault_epoch')!=fault.get('epoch'):raise RuntimeError('Native fault epoch differs')
    if any(fault.get(k) is not None for k in ('cap','requested_cap','saved_original_limits_rad_s')) or fault.get('dirty') is not False:
        raise RuntimeError('Native limit state is unresolved')
    if (fault.get('terminal') is not True or fault.get('durability_unresolved') is not False or
            fault.get('publication_attempts',0)<1 or fault.get('pause_error') or fault.get('rollback_error')):
        raise RuntimeError('Native fault durability/held retirement is unverified')
    error=semantic_error(fault['error']);samples=[]
    if error=='Synthetic start requires every protected body stationary':
        if any(k in report for k in ('selection','restore','protected_start','record_after_sim_ns')):
            raise RuntimeError('Precondition error belongs to a later setup stage')
        samples=[dict(snapshot=baseline,motion=velocity_only(baseline,baseline))]
    elif error.startswith(('Protected scene disturbed during settle: ','Protected settle coverage or motion failed: ')):
        evidence=fault['protected_settle'];restore=report['restore']['snapshot']
        if report['restore'].get('arm_restored') is not True or restore['epoch']!=fault['epoch']:
            raise RuntimeError('Synthetic restore is unverified')
        clean=lambda s:dict(copy.deepcopy(s),bodies={n:dict(copy.deepcopy(v),velocity=[0.]*6) for n,v in s['bodies'].items()})
        monitor=ProtectedSettle(baseline,clean(restore))
        def add(sample,step=None):
            snap=sample['snapshot'];motion=velocity_only(baseline,snap)
            if digest(motion)!=digest(sample['motion']):raise RuntimeError('Native motion proof differs from its snapshot')
            samples.append(dict(snapshot=snap,motion=motion))
            if step is not None:
                monitor.before_substep(step,clean(snap));checked=monitor.pending[-1]
            else:
                checked=monitor._sample(clean(snap))
            if digest(checked['checks'])!=digest(sample['checks']):
                raise RuntimeError('Native substep clock/drive proof differs')
        frames=evidence['frames'];boundary=sample_snapshot(restore)
        if len(frames)>6 or len(evidence['pending'])>4:raise RuntimeError('Excess native protected samples')
        for i,frame in enumerate(frames):
            if digest(frame['start'])!=digest(boundary):raise RuntimeError('Protected frame start differs')
            if frame.get('errors') or frame['frame_index']!=i or frame['epoch']!=restore['epoch'] or len(frame['substeps'])!=4:
                raise RuntimeError('Protected frame coverage differs')
            for j,sample in enumerate(frame['substeps']):
                if sample.get('index')!=j:raise RuntimeError('Protected callback order differs')
                add(sample,sample['step_s'])
            add(frame['final']);monitor.finish_frame(clean(frame['final']['snapshot']));boundary=sample_snapshot(frame['final']['snapshot'])
        for j,sample in enumerate(evidence['pending']):
            if sample['index']!=j:raise RuntimeError('Pending callback order differs')
            add(sample,sample['step_s'])
        failure=evidence['failure']
        if failure['stage']=='pre_physics_callback':
            if not evidence['pending'] or digest(failure['sample'])!=digest(evidence['pending'][-1]):
                raise RuntimeError('Fault does not bind its failing substep')
        elif failure['stage']=='final_post_solve':
            if not frames or digest(failure['frame'])!=digest(frames[-1]):
                raise RuntimeError('Fault does not bind its failing frame')
        else:raise RuntimeError('Unknown native protected failure')
        previous=report.get('settle_snapshots',[])
        if previous:
            require_snapshots(previous)
            if any(digest(s['protected_settle'])!=digest(frames[i]) for i,s in enumerate(previous)):
                raise RuntimeError('Returned settle frames differ from the durable fault')
    else:raise RuntimeError('Native fault is not a protected velocity screen')
    failed=[i for i,s in enumerate(samples) if not s['motion']['passed']]
    if not failed:raise RuntimeError('No measured protected velocity-screen rejection')
    if failed!=[len(samples)-1]:
        raise RuntimeError('Screen evidence continued after its first physical failure')
    if fault.get('sim_ns')!=samples[-1]['snapshot']['sim_ns']:
        raise RuntimeError('Native fault stamp differs from its failing sample')
    peaks={}
    for name in baseline['bodies']:
        bodies=[s['motion']['bodies'][name] for s in samples]
        peaks[name]=dict(samples=len(bodies),max_linear_m_s=max(v['linear_m_s'] for v in bodies),
            max_angular_rad_s=max(v['angular_rad_s'] for v in bodies),max_corner_motion_m=max(v['max_corner_motion_m'] for v in bodies))
    exceeded=[n for n,p in peaks.items() if p['max_linear_m_s']>POLICY['stationary_linear_m_s'] or p['max_angular_rad_s']>POLICY['stationary_angular_rad_s']]
    if not exceeded:raise RuntimeError('No measured protected velocity-screen rejection')
    return dict(reason='protected_velocity_screen',bodies=exceeded,peaks=peaks,native_error=error)


def classify(run,trial,seed,stage,runtime):
    """Only the current pre-writer screen can be quarantined; all doubt stops."""
    path=run/('attempt-%04d'%trial);receipt=leaf(path/'receipt.json')
    expected=dict(parameters(seed,COLORS[trial],regrasp=True),trial=trial)
    if receipt.get('params')!=expected:raise RuntimeError('Attempt seed/colour/parameters differ')
    episode_id=hashlib.sha256((str(run)+':'+str(trial)+':'+str(seed)).encode()).hexdigest()
    if receipt.get('episode_id')!=episode_id or receipt.get('recovery_report',{}).get('episode_id')!=episode_id:
        raise RuntimeError('Attempt/recovery identity differs')
    if (path/'receipt.json').stat().st_mtime_ns<stage['started_wall_ns']:
        raise RuntimeError('Attempt receipt predates its producer')
    if receipt.get('accepted') is True:
        certificate=collection_status.require_finalized(run,stage['pid'],stage['started_wall_ns'])
        rows=[json.loads(line) for line in (path/'frames.jsonl').read_text().splitlines()]
        if not timing_audit(rows,30)['passed'] or receipt_error(receipt,rows):
            raise RuntimeError('Accepted raw timing/recovery proof differs')
        if not receipt.get('timing',{}).get('passed'):raise RuntimeError('Accepted attempt has no passing timing audit')
        summary=leaf(run/'summary.json')
        if summary.get('episodes')!=trial+1 or summary.get('attempts')!=trial+1:
            raise RuntimeError('Collector accepted ledger differs')
        return dict(kind='accepted',receipt_sha256=digest(receipt),retirement=certificate,
                    frames=receipt['timing']['frames'],peaks=receipt['recovery_report']['protected_finish']['velocity_peaks'])
    if any((path/name).exists() for name in WRITER_FILES) or any(k in receipt for k in ('timing','score','script_completed','settled_physical')):
        raise RuntimeError('Fault after writer opening is terminal')
    if receipt.get('capture_failures') or receipt.get('camera_boundary',{}).get('passed') is not True:
        raise RuntimeError('Camera/capture contract failure is terminal')
    if receipt.get('camera_publishers') is None:raise RuntimeError('Camera provenance is missing')
    retired=retired_terminal(run,stage)
    retirement=receipt.get('retirement',{})
    if not retirement.get('passed') and semantic_error(retirement.get('error'))!=semantic_error(receipt.get('error')):
        raise RuntimeError('An unrelated motion-retirement failure is terminal')
    markers=[leaf(run/'terminal-fault.json'),leaf(path/'terminal-fault.json')]
    native_path=Path(receipt['native_miss_fault_file'])
    if not native_path.resolve().is_relative_to((runtime/'native-motion-faults').resolve()):
        raise RuntimeError('Native fault belongs to another instance')
    native=leaf(native_path)
    if native_path.stat().st_mtime_ns<stage['started_wall_ns']:raise RuntimeError('Native fault predates its producer')
    if not any(digest(f)==digest(native) for f in receipt.get('native_miss_faults',[])):
        raise RuntimeError('Receipt does not bind the durable native fault')
    if any(semantic_error(f.get('error'))!=semantic_error(native['error']) or f.get('epoch')!=native['epoch']
           for f in receipt.get('native_miss_faults',[])):
        raise RuntimeError('Another native fault is also present')
    proof=screen_evidence(receipt,native);message=proof['native_error']
    allowed_sources={'synthetic start verification','attempt disposition','before cleanup','after cleanup'}
    if markers[0]!=markers[1] or receipt.get('terminal_failure')!=markers[0]:
        raise RuntimeError('Terminal markers differ')
    for marker in markers:
        if not marker.get('events'):raise RuntimeError('Terminal event ledger is empty')
        for event in marker['events']:
            if (not retired['started_wall_ns']<=event.get('wall_ns',-1)<=retired['completed_wall_ns'] or event.get('source') not in allowed_sources or event.get('sensor_evidence') is not None or
                    semantic_error(event.get('error'))!=message):
                raise RuntimeError('A second terminal cause cannot be quarantined')
    if semantic_error(receipt.get('error'))!=message:
        raise RuntimeError('Attempt error differs from its native screen evidence')
    return dict(kind='screen_reject',receipt_sha256=digest(receipt),retirement=retired,native_fault=native,native_fault_path=str(native_path),**proof)


def require_down(proof):
    if proof.get('passed') is not True or proof.get('remaining')!=[] or any(proof.get('ports_listening',{}).values()) or len(proof.get('ports_listening',{}))!=2:
        raise RuntimeError('Owned-down is unverified')


def require_export(value,episodes=3):
    if (value.get('passed') is not True or value.get('loaded') is not True or value.get('fps')!=30 or
            value.get('episodes')!=episodes or not value.get('timing_passed') or
            value.get('frame_phases',{}).get('counts')!={'normal':0,'miss_prefix':0,'corrective':value.get('frames')}):
        raise RuntimeError('Cohort export/load/phase certificate differs')
    content=value.get('export_content',{})
    if (not content.get('metadata',{}).get('passed') or not content.get('video',{}).get('passed') or
            content.get('video',{}).get('scope')!='ALL frames' or
            content.get('video',{}).get('decoded_frame_pairs')!=2*value['frames']):
        raise RuntimeError('Cohort all-frame content proof is missing')


class Runner:
    """The same orchestration is exercised by fake and actual process transports."""
    def __init__(self,config,adapter,publish):
        self.config=config;self.adapter=adapter;self.publish=publish
    def run(self,state=None):
        state=copy.deepcopy(state) if state is not None else dict(version=POLICY_VERSION,config=self.config,events=[],cohorts=[],
            next_seed=self.config['seed_start'],attempts=0,screen_rejects=0,selected_episodes=0,
            unselected_prefix_episodes=0,unselected_prefix_wall_s=0.,pending=None,stopped=False)
        if state['config']!=self.config or state['pending'] or state['stopped']:raise RuntimeError('Runner resume is inconsistent, incomplete or stopped')
        if state.get('events') and state is not None:
            if state.get('final_owned_down') is None:raise RuntimeError('Runner checkpoint has no owned retirement')
            require_down(state['final_owned_down'])
            if (len(state['events'])!=state['attempts'] or
                    [e['seed'] for e in state['events']]!=list(range(self.config['seed_start'],state['next_seed'])) or
                    sum(e['outcome']['kind']=='screen_reject' for e in state['events'])!=state['screen_rejects']):
                raise RuntimeError('Runner checkpoint counters/seeds differ')
            selected=0
            for cohort in state['cohorts']:
                events=[e for e in state['events'] if e['cohort']==cohort['index']]
                if [e['trial'] for e in events]!=list(range(len(events))) or len(events)>3:
                    raise RuntimeError('Runner checkpoint cohort order differs')
                if cohort.get('selected'):
                    if len(events)!=3 or any(e['outcome']['kind']!='accepted' for e in events):
                        raise RuntimeError('Selected checkpoint is not an RGB cohort')
                    require_export(cohort['validation']);selected+=3
                elif not cohort.get('quarantined') or not events or events[-1]['outcome']['kind']!='screen_reject':
                    raise RuntimeError('Incomplete runner cohort cannot resume')
            if selected!=state['selected_episodes']:raise RuntimeError('Runner checkpoint selection differs')
            self.adapter.verify_prior(state)
        state.pop('final_owned_down',None)
        def save():self.publish(copy.deepcopy(state))
        save()
        try:
            while state['selected_episodes']<3*self.config['cohorts']:
                number=len(state['cohorts']);cohort=dict(index=number,base_seed=state['next_seed'],accepted=0,attempt_wall_s=0.,selected=False)
                state['cohorts'].append(cohort);save();cohort_started=time.monotonic()
                if state['next_seed']>self.config['seed_end']:raise RuntimeError('Reserved seed range exhausted')
                self.adapter.start(cohort)
                restart=False
                for trial in range(3):
                    seed=state['next_seed']
                    if seed>self.config['seed_end']:raise RuntimeError('Reserved seed range exhausted')
                    state['pending']=dict(cohort=number,trial=trial,seed=seed);save()
                    begun=time.monotonic();state['attempts']+=1;state['next_seed']+=1
                    try:outcome=self.adapter.attempt(cohort,trial,seed)
                    except BaseException as error:
                        state['events'].append(dict(state['pending'],outcome=dict(kind='terminal',error=repr(error)),wall_s=time.monotonic()-begun))
                        state['pending']=None
                        raise
                    wall=time.monotonic()-begun;cohort['attempt_wall_s']+=wall
                    if outcome['kind']=='accepted':cohort['accepted']+=1
                    elif outcome['kind']=='screen_reject':
                        # The durable fault survives untouched; only this outer selection changes.
                        down=self.adapter.stop();require_down(down)
                        state['screen_rejects']+=1;restart=True;cohort['quarantined']=True;cohort['owned_down']=down
                        cohort['total_wall_s']=time.monotonic()-cohort_started
                        state['unselected_prefix_episodes']+=cohort['accepted']
                        state['unselected_prefix_wall_s']+=cohort.get('prefix_wall_s',0.)
                    else:raise RuntimeError('Unknown attempt outcome')
                    event=dict(state['pending'],outcome=outcome,wall_s=wall)
                    state['events'].append(event);state['pending']=None
                    if outcome['kind']=='accepted':cohort['prefix_wall_s']=cohort.get('prefix_wall_s',0.)+wall
                    save()
                    if state['attempts']>=20 and state['screen_rejects']*10>state['attempts']:
                        raise RuntimeError('Resting-screen rejection rate exceeds10percent after20attempts')
                    if restart:break
                    if trial<2:self.adapter.between(cohort,trial)
                if restart:continue
                down=self.adapter.stop();require_down(down);cohort['owned_down']=down
                validation=self.adapter.export(cohort);require_export(validation)
                cohort.update(selected=True,validation=validation,total_wall_s=time.monotonic()-cohort_started);state['selected_episodes']+=3;save()
            state.update(complete=True,color_counts={c:state['selected_episodes']//3 for c in COLORS})
        except BaseException as exc:
            state.update(stopped=True,error=repr(exc),complete=False)
            raise
        finally:
            try:
                down=self.adapter.stop();require_down(down);state['final_owned_down']=down
            except BaseException as cleanup:
                state.update(stopped=True,complete=False,cleanup_error=repr(cleanup))
            selected={c['index'] for c in state['cohorts'] if c.get('selected')}
            prefixes=[e for e in state['events'] if e['outcome']['kind']=='accepted' and e['cohort'] not in selected]
            state['unselected_prefix_episodes']=len(prefixes)
            state['unselected_prefix_wall_s']=sum(e['wall_s'] for e in prefixes)
            state['physical_accepted_episodes']=sum(e['outcome']['kind']=='accepted' for e in state['events'])
            state['quarantined_extra_wall_s']=sum(c.get('total_wall_s',0.) for c in state['cohorts'] if c.get('quarantined'))
            state['selected_cohorts']=[dict(index=c['index'],dataset='cohort-%04d/run/dataset'%c['index'],
                seed_start=c['base_seed'],seed_end=c['base_seed']+2,colors=dict(red=1,green=1,blue=1))
                for c in state['cohorts'] if c.get('selected')]
            save()
            if state.get('cleanup_error'):raise RuntimeError(state['cleanup_error'])
        return state


class Processes:
    def __init__(self,config,output,claim):
        self.config=config;self.output=output;self.claim=claim
        self.runtime=confined(CANONICAL_ROOT/('runtime/instance-'+str(config['instance'])))
        self.ledger=None;self.live=False
        self.stage_index=len(list(output.glob('stage-*.log')))
        self.current_cohort=None
        from gpu_device import gpu_uuid
        card=gpu_uuid(config['gpu'])
        self.env=dict(os.environ,ISAACSIM_GPU=str(config['gpu']),CUDA_VISIBLE_DEVICES=card,SOARM_ASSIGNED_GPU_UUID=card,
            SOARM_SIM_NICE='10',SOARM_PHYSICS_THREADS='4',SOARM_INSTANCE_ROOT=str(self.runtime),
            SOARM_COLLECTION_ROOT=str(CANONICAL_ROOT),HF_LEROBOT_HOME=str(CANONICAL_ROOT/'datasets'),
            SOARM_INSTANCE_CLAIM_FD=str(claim.fd),ISAACSIM_LOG=str(self.runtime/'isaac.log'),
            PYTHONDONTWRITEBYTECODE='1',PYTHONFAULTHANDLER='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        for key in ('ISAACSIM_EXTRA_ARGS','ROS_LOCALHOST_ONLY'):self.env.pop(key,None)
        self.wrapper=str(REPO/'linux-env/scripts/collect_sim.sh')
    def stage(self,name,argv):
        frozen(self.config['cut']);self.stage_index+=1;log=self.output/('stage-%05d-%s.log'%(self.stage_index,name))
        if self.ledger is None:self.ledger=OwnedProcesses(self.runtime,time.time_ns())
        with log.open('x') as stream:
            started=time.time_ns()
            p=subprocess.Popen(argv,cwd=REPO,env=self.env,stdout=stream,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=(self.claim.fd,))
            self.ledger.add(p.pid)
            deadline=time.monotonic()+dict(up=1200,clock=300,attempt=7200,down=90,export=1800)[name]
            try:
                while p.poll() is None:
                    self.ledger.scan()
                    if time.monotonic()>deadline:raise TimeoutError('Outer '+name+' watchdog: '+str(log))
                    time.sleep(.1)
            finally:
                self.ledger.scan()
                value=dict(name=name,pid=p.pid,started_wall_ns=started,returncode=p.returncode,log=str(log))
                atomic_json(log.with_suffix('.json'),value)
        return value
    def root(self,cohort):return self.output/('cohort-%04d'%cohort['index'])
    def capacity(self):
        from host_config import profile
        import shutil
        root_free=shutil.disk_usage('/').free;bulk_free=shutil.disk_usage(CANONICAL_ROOT).free
        required=50*2**30
        if profile()['name']=='rtx6000':
            job=CANONICAL_ROOT.parent/'smolvla-train/phase3.2/jobs/smolvla-phase3-normal-60k-s42-20261114'
            transfer=leaf(job/'transfer-proof.json')
            if transfer.get('passed') is not True or transfer.get('destination_job')!=str(job):
                raise RuntimeError('Training pull is not certified')
            required+=20*2**30+max(13*2**30,int(transfer.get('checkpoint_space_required_bytes',0)))
        if root_free<20*2**30 or bulk_free<required:raise RuntimeError('Production storage reservation is not available')
    def start(self,cohort):
        if self.live:raise RuntimeError('A previous instance is still owned')
        self.capacity()
        self.current_cohort=cohort
        root=self.root(cohort);root.mkdir(exist_ok=False)
        self.live=True
        stage=self.stage('up',['bash',self.wrapper,'up',str(self.config['instance'])])
        if stage['returncode']:raise RuntimeError('Owned startup failed: '+stage['log'])
        clock=root/'clock.json';stage=self.stage('clock',['bash',self.wrapper,'clock-sanity',str(self.config['instance']),'--output',str(clock)])
        clock_status.require(clock,stage['pid'],stage['started_wall_ns'],stage['returncode'])
    def attempt(self,cohort,trial,seed):
        self.capacity()
        run=self.root(cohort)/'run'
        if trial:collection_status.require_finalized(run)
        argv=['bash',self.wrapper,'run',str(self.config['instance']),'--episodes','3','--seed',str(cohort['base_seed']),
              '--regrasp-only','--output',str(run),'--max-attempts',str(trial+1)]
        if trial:argv.append('--resume')
        stage=self.stage('attempt',argv)
        if not (run/('attempt-%04d/receipt.json'%trial)).exists():
            from run_smoke import StartupFailure
            raise StartupFailure('attempt',stage['returncode'],Path(stage['log']))
        return classify(run,trial,seed,stage,self.runtime)
    def between(self,cohort,trial):
        import sys
        sys.path.insert(0,str(REPO/'omni-kit-mcp'))
        from kit_mcp.client import call
        receipt=leaf(self.root(cohort)/'run'/('attempt-%04d/receipt.json'%trial))
        code='import collection_runtime as c\nr=c.runtime()\nresult=r.restore('+repr(receipt['initial_physical'])+',arm=True)'
        result=call('run_python',dict(code=code),host='127.0.0.1',port=8777+2*self.config['instance'],timeout=30)
        if isinstance(result,dict) and 'result' in result:result=result['result']
        if not isinstance(result,dict) or not result.get('arm_restored') or not result.get('command_baseline'):
            raise RuntimeError('Between-process physical baseline did not verify')
        atomic_json(self.root(cohort)/('between-%d.json'%trial),result)
    def probe_ports(self):
        ports={}
        for port in (8777+2*self.config['instance'],8778+2*self.config['instance']):
            with socket.socket() as sock:
                sock.settimeout(.5);ports[str(port)]=sock.connect_ex(('127.0.0.1',port))==0
        return ports
    def stop(self):
        if getattr(self,'failed_cleanup',None):raise RuntimeError('Previous owned-down failed: '+repr(self.failed_cleanup))
        if self.ledger is None:
            ports=self.probe_ports()
            proof=dict(passed=not any(ports.values()),remaining=[],ports_listening=ports,scope='no owned process started')
            require_down(proof);return proof
        errors=[]
        foreign=[]
        for name in ('isaacsim-soarm101.pid','control.pid'):
            path=self.runtime/name
            if path.exists():
                current=identity(int(path.read_text()))
                if current and (current['pid'] not in self.ledger.known or
                        self.ledger.known[current['pid']]['start_ticks']!=current['start_ticks']):
                    foreign.append(dict(path=str(path),identity=current))
        if foreign:errors.append('Refuse foreign private launcher markers: '+repr(foreign))
        elif self.live:
            try:
                s=self.stage('down',['bash',self.wrapper,'down',str(self.config['instance'])])
                if s['returncode']:errors.append('Owned launcher down refused')
            except BaseException as exc:errors.append(repr(exc))
        cleanup=self.ledger.stop();self.ledger=None;self.live=False;ports=self.probe_ports()
        proof=dict(cleanup,errors=errors,ports_listening=ports,passed=not errors and not cleanup['remaining'] and not any(ports.values()))
        atomic_json(self.output/('owned-down-%05d.json'%self.stage_index),proof)
        if not proof['passed']:self.failed_cleanup=copy.deepcopy(proof)
        require_down(proof)
        if self.current_cohort is not None:
            import shutil
            root=self.root(self.current_cohort)
            for name in ('isaac.log','control.log'):
                source=self.runtime/name;target=root/name
                if source.exists() and not target.exists():shutil.copyfile(source,target)
        return proof
    def verify_prior(self,state):
        for cohort in state['cohorts']:
            if not cohort.get('selected'):continue
            root=self.root(cohort);seal=leaf(root/'artifact-seal.json')
            if digest(seal)!=cohort['validation']['runner_artifact_seal_sha256']:
                raise RuntimeError('Cohort artifact seal changed')
            if any(p.is_symlink() for p in root.rglob('*')):raise RuntimeError('Sealed cohort contains symbolic links')
            current={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in root.rglob('*') if p.is_file() and p.name!='artifact-seal.json'}
            if current!=seal['files']:raise RuntimeError('Sealed cohort files changed')
            collection_status.require_finalized(root/'run')
    def export(self,cohort):
        run=self.root(cohort)/'run';report=self.root(cohort)/'report'
        stage=self.stage('export',['bash',self.wrapper,'export',str(self.config['instance']),str(run),
            '--repo-id','local/soarm101_synthetic_recovery','--report-dir',str(report)])
        value=leaf(report/'validation.json');require_export(value)
        if value.get('started_wall_ns',0)<stage['started_wall_ns']:raise RuntimeError('Export certificate predates its stage')
        root=self.root(cohort)
        files={}
        for path in root.rglob('*'):
            if path.is_symlink():raise RuntimeError('Cohort artifacts must not contain symbolic links')
            if path.is_file():files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        seal=dict(version=1,files=files)
        atomic_json(root/'artifact-seal.json',seal)
        return dict(value,runner_artifact_seal_sha256=digest(seal))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instance',type=int,required=True,choices=range(10));p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--cohorts',type=int,default=50);p.add_argument('--seed-start',type=int,required=True);p.add_argument('--seed-end',type=int,required=True)
    p.add_argument('--cut',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    modes=p.add_mutually_exclusive_group(required=True);modes.add_argument('--plan',action='store_true');modes.add_argument('--go',action='store_true')
    a=p.parse_args()
    if not 600000<=a.seed_start<=a.seed_end<700000 or a.cohorts<1:p.error('Use the reserved recovery seeds and positive cohort count')
    output=confined(a.output);config=dict(version=POLICY_VERSION,instance=a.instance,gpu=a.gpu,cohorts=a.cohorts,
        seed_start=a.seed_start,seed_end=a.seed_end,cut=a.cut,host=os.uname().nodename,output=str(output))
    frozen(a.cut)
    if a.plan:print(json.dumps(config,indent=2));return
    runtime=confined(CANONICAL_ROOT/('runtime/instance-'+str(a.instance)));runtime.mkdir(parents=True,exist_ok=True)
    previous=leaf(output/'runner.json') if a.resume else None
    if not a.resume:output.mkdir(exist_ok=False,parents=True)
    claim=InstanceClaim(runtime)
    try:
        for name in ('isaacsim-soarm101.pid','control.pid'):
            q=runtime/name
            if q.exists() and identity(int(q.read_text())):raise RuntimeError('Refuse an existing private process; no adoption')
        for port in (8777+2*a.instance,8778+2*a.instance):
            with socket.socket() as sock:
                sock.settimeout(.3)
                if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Assigned port already in use')
        if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise OSError('Cannot own descendant cleanup')
        def interrupted(*_):raise KeyboardInterrupt('Production runner stop requested')
        signal.signal(signal.SIGTERM,interrupted)
        invocations=output/'invocations';invocations.mkdir(exist_ok=True)
        atomic_json(invocations/(uuid.uuid4().hex+'.json'),dict(identity(os.getpid()),cut=a.cut))
        adapter=Processes(config,output,claim)
        Runner(config,adapter,lambda state:atomic_json(output/'runner.json',state)).run(previous)
    finally:claim.close()


if __name__=='__main__':main()
