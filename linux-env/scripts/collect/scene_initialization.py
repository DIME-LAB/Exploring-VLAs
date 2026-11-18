"""Recognize a clean, zero-member first-tick timeout without admitting any data.

This is an outer startup retry only. The native collector and its held-frame
budget remain unchanged; every other incomplete collection is terminal.
"""
import copy,hashlib,json,math,os,re,tempfile
from pathlib import Path
from triplet_artifacts import read,sha,manifest
import collection_status,clock_status,stage_readiness

RAW={'collector-finalization.json','summary.json','wall-profile-finalization.json'}
CAUSE='TimeoutError: Held-frame capacity exceeded 325.8 wall seconds; query/ACK/clock/recording did not retire'


class IncompleteInitialization(RuntimeError):
    pass


def traceback_text(repo):
    source=Path(repo)/'linux-env/scripts/collect/collect.py'
    frames=[(2098,'<module>','main()'),(2087,'main','else:collector.run()'),
        (1749,'run','clock_config, transport_config, baseline, source_hashes, initial_gpu_footprint, provenance = self._initialize_scene_session(root, previous, attempted)'),
        (1504,'_initialize_scene_session','self.advance_ticks(1)'),
        (1167,'advance_ticks',"raise TimeoutError('Held-frame capacity exceeded '+str(frame_capacity_seconds())+' wall seconds; query/ACK/clock/recording did not retire')")]
    return 'Traceback (most recent call last):\n'+''.join(f'  File "{source}", line {line}, in {name}\n    {code}\n' for line,name,code in frames)+CAUSE+'\n'


def verify(evidence):
    """Replay the source-bound startup decision, including durable ledger copies."""
    import triplet_continuation as c
    if evidence.get('version')!=1 or evidence.get('collector_cut')!=c.COLLECTOR_CUT:raise ValueError('Scene startup source cut differs')
    stage=evidence['stage'];stages=evidence['stages'];session=evidence['session'];job=evidence['job'];files=evidence['files']
    expected={*(('raw/'+n) for n in RAW),'worker/collect.log','worker/clock.json',
        'worker/clock-status/collector-finalization.json','worker/clock-status/summary.json','worker/readiness.json','worker/native-observations.jsonl'}
    if set(files)!=expected or any(type(v) is not str for v in files.values()):raise ValueError('Scene startup has unexpected/missing evidence')
    if evidence['files_sha256']!={n:hashlib.sha256(v.encode()).hexdigest() for n,v in files.items()}:raise ValueError('Scene startup evidence bytes differ')
    if type(session) is not int or session<0 or type(job['layout_seed']) is not int or type(job['instance']) is not int:raise ValueError('Scene startup identity malformed')
    if (stage.get('name') not in ('collect-'+str(job['layout_seed']),'collect-'+str(job['layout_seed'])+'-startup-'+str(session)) or stage.get('returncode')!=1 or type(stage.get('pid')) is not int or
        type(stage.get('started_wall_ns')) is not int or type(stage.get('wall_s')) not in (int,float) or not stage['wall_s']>=325.8):raise ValueError('Scene initialization stage is not a completed native timeout')
    if not stages or stages[-1]!=stage:raise ValueError('Scene initialization stage is not current')
    ups=[i for i,row in enumerate(stages) if row['name'].startswith('up-')]
    if not ups:raise ValueError('Scene startup lacks its fresh instance')
    recent=stages[ups[-1]:]
    if [r['name'] for r in recent]!=[f'up-{session}',f'clock-{session}',stage['name']] or any(r.get('returncode')!=0 for r in recent[:2]):raise ValueError('Scene initialization was not the first collection of a qualified fresh instance')
    if not recent[0]['started_wall_ns']<recent[1]['started_wall_ns']<stage['started_wall_ns']:raise ValueError('Scene startup stage clocks differ')
    if files['worker/collect.log']!=traceback_text(evidence['collector_repo']):raise ValueError('Scene startup traceback is unknown')
    with tempfile.TemporaryDirectory(prefix='scene-init-replay-') as d:
        root=Path(d)
        for name,text in files.items():
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
        cert=collection_status.require_finalized(root/'raw',stage['pid'],stage['started_wall_ns'])
        summary=read(root/'raw/summary.json')
        if summary!={'complete':False,'producer_invocation':cert['invocation'],'end_marker':'COLLECTION_IN_PROGRESS'}:raise ValueError('Scene initialization has member or terminal summary data')
        profile=read(root/'raw/wall-profile-finalization.json');ticks=profile.get('ticks')
        if profile.get('capture_failures')!=[] or not isinstance(ticks,list) or len(ticks)!=1:raise ValueError('Scene initialization capture had an unknown fault or completed tick')
        tick=ticks[0]
        if (tick.get('recorded') is not False or tick.get('sim_ns') is not None or tick.get('stage')!='sensor_join' or tick.get('outcome')!='cancelled' or
            tick.get('error')!="CancelledGeneration('Sensor join closed')" or type(tick.get('epoch')) is not int or tick['epoch']<=0):raise ValueError('Scene initialization capture is not the retained unjoined cancellation')
        for key in ('capture_processing_ms','tick_total_ms'):
            value=tick.get(key)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Scene initialization capture timing malformed')
        if cert['completed_wall_ns']>stage['started_wall_ns']+int(stage['wall_s']*1e9)+1_000_000:raise ValueError('Scene startup finalization is outside its stage')
        clock=recent[1];clock_status.require(root/'worker/clock.json',clock['pid'],clock['started_wall_ns'],clock['returncode'])
        ready=read(root/'worker/readiness.json')
        if ready.get('passed') is not True or ready.get('stable_samples')!=3 or len(ready.get('samples',[]))<3:raise ValueError('Scene startup readiness is not certified')
        samples=ready['samples'][-3:];roots=set()
        for sample in samples:
            observation=sample['observation']['stage_readiness'];roots.add(observation['stage_root'])
            if sample['errors'] or stage_readiness.errors(observation) or not clock['started_wall_ns']<observation['wall_ns']<stage['started_wall_ns']:raise ValueError('Scene startup readiness differs')
        if len(roots)!=1:raise ValueError('Scene startup stage changed')
        native=[json.loads(line) for line in files['worker/native-observations.jsonl'].splitlines()]
        if len(native)<5 or len({n['nonce'] for n in native})!=len(native):raise ValueError('Scene startup native challenges missing/reused')
        identity=evidence['native_identity']
        if set(identity)!={'pid','start_ticks','pgid'} or type(identity['pid']) is not int or identity['pid']<=0:raise ValueError('Scene startup native identity missing')
        if any({k:n[k] for k in identity}!=identity or not re.fullmatch('[0-9a-f]{32}',n['nonce']) for n in native):raise ValueError('Scene startup native responder changed')
        if [n['answer'] for n in native[:2]]!=[True,True] or [n['answer'] for n in native[-3:]]!=[s['observation']['stage_readiness'] for s in samples]:raise ValueError('Scene startup readiness/native capture differs')
    return CAUSE


def capture(worker,job,event):
    run=worker.raw_path(job)
    # This exact file set also rejects all member directories, frame ledgers,
    # writer artefacts, run/provenance files and terminal markers.
    if set(manifest(run))!=RAW or {p.name for p in run.iterdir()}!=RAW:raise IncompleteInitialization('Pre-provenance collection is not a zero-member initialization: '+str(run))
    files={'raw/'+n:read(run/n,True).decode() for n in RAW}
    names={'collect.log':event['name']+'.log','clock.json':f'clock-{worker.session}.json',
        'clock-status/collector-finalization.json':f'clock-{worker.session}-status/collector-finalization.json',
        'clock-status/summary.json':f'clock-{worker.session}-status/summary.json',
        'readiness.json':f'readiness-{worker.session}.json','native-observations.jsonl':'native-observations.jsonl'}
    for key,name in names.items():
        raw=read(worker.root/name,True)
        if key=='native-observations.jsonl':raw=raw[getattr(worker,'native_observation_start',0):]
        files['worker/'+key]=raw.decode()
    evidence=dict(version=1,collector_cut=worker.cut,collector_repo=str(worker.collector_repo),job=copy.deepcopy(job),stage=event,
        stages=copy.deepcopy(worker.stages),session=worker.session,native_identity=worker.native,files=files,
        files_sha256={n:hashlib.sha256(v.encode()).hexdigest() for n,v in files.items()})
    try:verify(evidence)
    except (KeyError,ValueError,RuntimeError,OSError,TypeError) as exc:raise IncompleteInitialization('Unrecognized pre-provenance collection; '+str(run)+'; '+repr(exc)) from exc
    return evidence


def failure(worker,job,event):
    import startup_retries
    try:evidence=capture(worker,job,event)
    except (OSError,ValueError,KeyError,TypeError,IncompleteInitialization) as exc:
        log=worker.root/(event.get('name','unknown')+'.log')
        try:original=read(log,True).decode()
        except (OSError,ValueError,UnicodeError) as missing:original='Cannot read original collector log: '+repr(missing)
        raise IncompleteInitialization('Incomplete pre-provenance collection: '+str(worker.raw_path(job))+'; '+repr(exc)+'; original collector log '+str(log)+':\n'+original) from exc
    result=startup_retries.StartupFailure('scene_initialization',CAUSE,event,worker.root/(event['name']+'.log'))
    result.evidence=evidence
    return result


def archive(run,target,error,log_path):
    """Called only after certified retirement, before reserving another start."""
    import triplet_production as p
    verify(error.evidence);run=Path(run);target=Path(target)
    files={n.removeprefix('raw/'):v for n,v in error.evidence['files_sha256'].items() if n.startswith('raw/')}
    if manifest(run)!=files:raise RuntimeError('Scene initialization raw changed before retirement backup')
    if os.path.lexists(target):raise FileExistsError('Scene initialization archive already exists')
    destination=p.DUAL/target.relative_to(p.ROOT)
    with Path(log_path).open('x') as log:receipt=p.backup(run,files,destination,log)
    if receipt.get('passed') is not True or receipt.get('files_sha256')!=files or receipt.get('destination')!=str(destination) or manifest(run)!=files:raise RuntimeError('Scene initialization diagnostic backup was not certified')
    run.rename(target)
    for parent in {run.parent,target.parent}:
        fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.fsync(fd)
        finally:os.close(fd)
    error.evidence['archive']=dict(path=str(target),backup=receipt)
