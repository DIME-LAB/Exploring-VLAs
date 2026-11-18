"""Source-bound continuation of explicitly approved retired completed waves.

Preparation is CPU-only and requires an independently supplied metadata
checkpoint. It never adopts an arbitrary run or changes the parent history.
"""
import copy,hashlib,json,os,re,signal,subprocess,time
from pathlib import Path
from storage import atomic_json
from triplet_artifacts import read,sha,manifest,retirement
from triplet_production_plan import initial_wave,reduce_waves,guards
import triplet_continuation as legacy
import startup_retries

KIND='completed-wave-continuation'
PRIOR_RUNNER='5eaf5add4c86e42481a3273ce9b84cfa35457da3'
EXPORTS=(720120,720121)
PEER=720122
IDENTITY=720123
RETIRED_RUNNER='68334ccb1fc76c0e36b538ac0fd3a9a6a96f0446'


def profile(root):
    """Only the explicitly reviewed retired parents are admissible."""
    import triplet_production as p
    if Path(root)==parent_root():
        return dict(runner=PRIOR_RUNNER,wave=1,dispositions=140,selected=113,exports=EXPORTS,peer=PEER,identity=IDENTITY,startup=None,slot=6,peer_error='MoveIt state validity service is unavailable')
    if Path(root)==p.ROOT/'datasets/counterfactual-targets/production-v1/continuations/continuation-admission-0a80e4e-v1':
        return dict(runner='0a80e4e867a58f1fec48ea3a9b85d179cc70343c',wave=2,dispositions=167,selected=132,exports=(720147,720149),peer=720145,identity=None,startup=720144,slot=5,peer_error='MoveIt state validity query did not complete')
    if Path(root)==p.ROOT/'datasets/counterfactual-targets/production-v1/continuations/continuation-admission-68334cc-v1':
        return dict(runner=RETIRED_RUNNER,wave=2,dispositions=172,selected=136,base_dispositions=167,base_selected=132,
            exports=(),peer=None,identity=None,startup=None,slot=5,partial=True)
    raise ValueError('Unapproved completed continuation parent')


def parent_owner(root):
    import triplet_production as p
    d=profile(root)
    name='68334cc' if d['runner']==RETIRED_RUNNER else '0a80e4e'
    return owner_path() if d['runner']==PRIOR_RUNNER else p.ROOT/('setup/counterfactual-continuation-'+name+'/coordinator.pid.identity.json')


def parent_root():
    import triplet_production as p
    return p.ROOT/'datasets/counterfactual-targets/production-v1/continuations/continuation-admission-5eaf5ad-v1'


def owner_path():
    import triplet_production as p
    return p.ROOT/'setup/counterfactual-continuation-5eaf5ad/coordinator.pid.identity.json'


def checkpoint(root):
    import triplet_production as p
    root=legacy.regular(root/'plan.json').parent
    profile(root)
    files={str(f):sha(legacy.regular(f)) for f in sorted(root.rglob('*.json')) if 'terminal-diagnostics' not in f.relative_to(root).parts}
    owner=parent_owner(root);down=owner.parent/'post-stop-owned-down.json'
    for f in (owner,down):files[str(f)]=sha(legacy.regular(f))
    for source in read(down)['sources']:
        f=legacy.regular(source['path'])
        if not f.is_relative_to(p.ROOT) or 'quarantine' in f.relative_to(p.ROOT).parts or sha(f)!=source['sha256']:
            raise ValueError('Completed parent retirement source changed')
        files[str(f)]=source['sha256']
    value=dict(version=1,parent_root=str(root),files_sha256=files)
    return value,hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def historical_parent(root):
    import triplet_production as p
    old=read(root/'plan.json');data=p.ROOT/'repo/counterfactual-production'
    prior=profile(root)['runner']
    if old.get('runner_cut')!=prior or old.get('collector_repo')!=str(data) or old.get('cut')!=legacy.COLLECTOR_CUT:
        raise ValueError('Completed parent source identity differs')
    p.data_checkout(data,legacy.COLLECTOR_CUT)
    source=subprocess.check_output(['git','-C',str(p.REPO),'show',prior+':linux-env/scripts/collect/triplet_production.py'])
    if prior==PRIOR_RUNNER:expected=legacy.legacy_recipe(old['continuation_admission_path'],old['continuation_admission_sha256'],prior,hashlib.sha256(source).hexdigest())
    else:
        admission=read(old['continuation_admission_path'])
        expected_parent=parent_root() if prior!=RETIRED_RUNNER else p.ROOT/'datasets/counterfactual-targets/production-v1/continuations/continuation-admission-0a80e4e-v1'
        if admission.get('prior_production_root')!=str(expected_parent):raise ValueError('Historical authority must retain its approved 5eaf parent/0a80 ancestry')
        expected=load_plan(old['continuation_admission_path'],old['continuation_admission_sha256'],prior,_historical_source=hashlib.sha256(source).hexdigest())
        ancestor,digest=checkpoint(Path(admission['prior_production_root']))
        if ancestor!=admission['parent_checkpoint'] or digest!=admission['parent_checkpoint_sha256']:raise ValueError('Historical parent checkpoint changed')
    if old!=expected or Path(old['production_root'])!=root:raise ValueError('Completed parent plan/admission differs')
    return old


def parent_state(root,digest,allow_retry_extension=False):
    import triplet_production as p
    root=Path(root);cp,actual=checkpoint(root)
    if not re.fullmatch('[0-9a-f]{64}',str(digest)) or actual!=digest:raise ValueError('Caller-bound completed checkpoint differs')
    d=profile(root);owner_file=parent_owner(root)
    recipe=historical_parent(root);stop=read(root/'STOP.json');final=read(root/'production-finalization.json');owner=read(owner_file);down=read(owner_file.parent/'post-stop-owned-down.json')
    if (stop.get('slot')!=d['slot'] or final.get('passed') is not False or final.get('retirement_errors') or
        final.get('cleanup',{}).get('remaining')!=[] or final.get('backup_children_cleanup',{}).get('remaining')!=[] or
        final.get('terminal_backups',{}).get('errors') or owner.get('state')!='stopped' or owner.get('supervision_error') or
        owner.get('cleanup_error') or owner.get('cleanup',{}).get('remaining')!=[] or down.get('passed') is not True or
        down.get('remaining_owned')!=[] or down.get('recorded_births',0)<=0 or
        set(down.get('ports_listening',{}))!={str(i) for i in range(8783,8791)} or any(v is not False for v in down['ports_listening'].values()) or
        down.get('managed_status',{}).get('supervision_ok') is not True or down.get('managed_stop',{}).get('supervision_ok') is not True or
        stop.get('supervision',{}).get('nonce')!=owner.get('nonce') or stop.get('supervision',{}).get('receipt_path')!=str(owner_file)):
        raise ValueError('Completed parent retirement is unresolved')
    identities={}
    def collect(value):
        if isinstance(value,dict):
            if type(value.get('pid')) is int and value.get('start_ticks') is not None:identities[(value['pid'],str(value['start_ticks']))]=value
            for one in value.values():collect(one)
        elif isinstance(value,list):
            for one in value:collect(one)
    collect(owner)
    for source in down['sources']:collect(read(source['path']))
    for pid,birth in identities:
        current=p.identity(pid)
        if current and str(current['start_ticks'])==birth and current['state']!='Z':raise ValueError('Completed parent workload/descendant survives')
    ledger=recipe['inherited_waves'] if d.get('partial') else read(root/'wave-ledger.json')
    if d.get('partial') and (root/'wave-ledger.json').exists():raise ValueError('Stopped partial parent has an unexpected completed ledger')
    state=reduce_waves(ledger);progress=read(root/'progress.json')
    if (len(ledger)!=d['wave'] or sum(len(w['outcomes']) for w in ledger)!=d.get('base_dispositions',d['dispositions']) or len(state['selected'])!=d.get('base_selected',d['selected']) or
        progress.get('wave')!=d['wave'] or progress.get('prior_selected')!=d.get('base_selected',d['selected']) or not d.get('partial') and progress.get('outcomes')!=[]):
        raise ValueError('Completed continuation requires the source-bound completed waves, never the original 94/76 base')
    for i,wave in enumerate(ledger):
        own=root/f'wave-{i:02d}.json';inherited_waves=recipe.get('inherited_waves',[])
        if i<len(inherited_waves):
            if inherited_waves[i]!=wave or own.exists() and read(own)!=wave:raise ValueError('Inherited completed wave/checkpoint differs')
        elif read(own)!=wave:raise ValueError('Completed wave/checkpoint differs')
        by_key={(r['split'],r['layout_seed']):r for r in wave['outcomes']}
        inherited={(r['split'],r['layout_seed']):r for r in recipe['inherited_outcomes']+[r for w in recipe.get('inherited_waves',[]) for r in w['outcomes']]}
        for job in wave['jobs']:
            row=by_key[(job['split'],job['layout_seed'])]
            if (job['split'],job['layout_seed']) in inherited:
                if row!=inherited[(job['split'],job['layout_seed'])]:raise ValueError('Parent changed an inherited disposition')
            else:
                folder=root/f'wave-{i:02d}/slot-{job["instance"]}'
                if any(read(folder/(prefix+str(job['layout_seed'])+'.json'))!=row for prefix in ('outcome-','admitted-')):raise ValueError('Parent admission/outcome membership differs')
            proof=p.verify_outcome(row,recipe)
            if proof['job']!=job:raise ValueError('Parent proof belongs to another job')
            legacy.backup_recheck(proof)
    partial=read(root/f'wave-{d["wave"]:02d}.json')
    if partial['jobs']!=state['next_wave'] or partial['outcomes']!=(recipe['inherited_outcomes'] if d.get('partial') else []):
        raise ValueError('Parent partial replacement wave changed')
    pending=[]
    if d.get('partial'):
        pending=progress['outcomes'];by_key={(j['split'],j['layout_seed']):j for j in state['next_wave']}
        if (len(pending)!=5 or len({(r['split'],r['layout_seed']) for r in pending})!=5 or
            {r['layout_seed'] for r in pending}!={720145,720147,720148,720149,730023} or
            any(r not in pending for r in recipe['inherited_outcomes'])):raise ValueError('Stopped partial parent membership differs')
        inherited={(r['split'],r['layout_seed']):r for r in recipe['inherited_outcomes']}
        for row in pending:
            key=(row['split'],row['layout_seed']);job=by_key.get(key)
            if job is None:raise ValueError('Orphan partial parent disposition')
            if key in inherited:
                if row!=inherited[key]:raise ValueError('Partial parent changed an inherited disposition')
            else:
                folder=root/f'wave-{d["wave"]:02d}/slot-{job["instance"]}'
                if any(read(folder/(prefix+str(job['layout_seed'])+'.json'))!=row for prefix in ('outcome-','admitted-')):raise ValueError('Partial outcome/admission differs')
            proof=p.verify_outcome(row,recipe);legacy.backup_recheck(proof)
            if proof['job']!=job:raise ValueError('Partial proof belongs to another job')
            if key not in inherited:
                stage=next((s for s in read(folder/'stages.json') if s['name']=='collect-'+str(job['layout_seed'])),None)
                if proof['stage']!=stage:raise ValueError('Partial proof changed its producing stage')
        all_rows=[r for w in ledger for r in w['outcomes']]+pending
        if len(all_rows)!=d['dispositions'] or sum(r['status']=='selected' for r in all_rows)!=d['selected']:raise ValueError('Stopped parent 172/136 membership differs')
        guards(all_rows,state['replacement_allocations'])
    # A worker may finish backup while the coordinator is retiring. This is
    # not an admitted disposition. Keep its proof immutable, verify it before
    # re-export, and qualify the collector afresh in the child namespace.
    unadmitted=list((root/f'wave-{d["wave"]:02d}').glob('slot-*/outcome-*.json'))
    if d.get('partial'):
        expected_worker={root/f'wave-02/slot-{by_key[(r["split"],r["layout_seed"])]["instance"]}/outcome-{r["layout_seed"]}.json' for r in pending if (r['split'],r['layout_seed']) not in inherited}
        if set(unadmitted)!=expected_worker:raise ValueError('Orphan/missing partial worker disposition')
        unadmitted=[]
    if d['startup'] is None and unadmitted:raise ValueError('Unexpected unadmitted parent outcome')
    for path in unadmitted:
        row=read(path);job=next((j for j in state['next_wave'] if j['layout_seed']==720147),None)
        expected=root/f'wave-02/slot-3/outcome-720147.json'
        if (job is None or path!=expected or (path.parent/'admitted-720147.json').exists() or row.get('status')!='selected' or row.get('reason')!='accepted' or
            row.get('proof_path')!=str(root/'proofs/train/720147.json') or sha(row['proof_path'])!=row.get('proof_sha256')):raise ValueError('Unadmitted parent outcome identity differs')
        proof=read(row['proof_path']);stage=next(e for e in read(path.parent/'stages.json') if e['name']=='collect-720147')
        if proof.get('job')!=job or proof.get('stage')!=stage or proof.get('cut')!=legacy.COLLECTOR_CUT or any(row.get(k)!=job[k] for k in ('split','layout_seed','requested_layout_seed')):raise ValueError('Unadmitted parent source/stage differs')
        if not allow_retry_extension:
            p.verify_outcome(row,recipe);legacy.require_completed_checks(proof);legacy.backup_recheck(proof)
    expected_proofs={Path(r['proof_path']) for w in ledger for r in w['outcomes'] if Path(r['proof_path']).is_relative_to(root/'proofs')}
    expected_proofs.update(Path(read(path)['proof_path']) for path in unadmitted)
    expected_proofs.update(Path(r['proof_path']) for r in pending if Path(r['proof_path']).is_relative_to(root/'proofs'))
    if set((root/'proofs').glob('*/*.json'))!=expected_proofs:raise ValueError('Orphan/missing completed parent proof')
    current=startup_retries.summary(recipe['startup_retry_budget']);saved=final.get('startup_retries')
    if not isinstance(saved,dict):raise ValueError('Parent retry finalization is missing')
    if allow_retry_extension:
        if (current['policy']!=saved['policy'] or current['root']!=saved['root'] or
            current['events'][:len(saved['events'])]!=saved['events']):raise ValueError('Parent retry history was removed or replaced')
    elif any(current.get(k)!=v for k,v in saved.items()):raise ValueError('Retry ledger differs from the retired parent finalization')
    if d.get('partial'):
        if (saved['counts']!=dict(total=3,slots={'3':0,'4':0,'5':3,'6':0}) or saved['pending']!={'5':['train',720144]} or
            len(saved['events'])!=10 or not str(stop.get('error','')).startswith('RuntimeError("Startup retry cap reached:')):raise ValueError('Stopped parent retry cap/history differs')
    return dict(recipe=recipe,waves=ledger,state=state,checkpoint=cp,stop=stop,pending_outcomes=pending)


def peer_source(run,job,stage,stop_ns,collector_repo,parent=None):
    import triplet_production as p
    from target_triplets import spec
    d=profile(parent or parent_root());peer=d['peer']
    if job['layout_seed']!=peer or job['split']!='train':raise ValueError('Not the approved peer seed')
    config=read(run/'run.json')
    if config.get('normal_only') is not True or config.get('instance')!=job['instance'] or config.get('counterfactual_triplet')!=spec(peer,'train'):
        raise ValueError('Peer collector seed/mode/slot differs')
    cert=retirement(run,stage,terminal=True);p.verify_sources(run,legacy.COLLECTOR_CUT,collector_repo)
    events=read(run/'terminal-fault.json')['events'];allowed={'MoveIt validity query','attempt disposition','before cleanup','after cleanup'}
    if (not cert['started_wall_ns']<stop_ns<cert['completed_wall_ns'] or not events or any(e.get('wall_ns',0)<=stop_ns or e.get('wall_ns',0)>cert['completed_wall_ns'] or e.get('source') not in allowed or
        d['peer_error'] not in str(e.get('error')) for e in events)):
        raise ValueError('Not the explicitly approved post-STOP peer interruption')
    if peer==720145 and (stop_ns!=1794978990139379403 or min(e['wall_ns'] for e in events)!=1794978991459962612):raise ValueError('720145 approved peer times differ')
    return min(e['wall_ns'] for e in events)


def peer_outcome(job,stage,state,output):
    import triplet_production as p
    peer=job['layout_seed'];run=p.ROOT/f'datasets/counterfactual-targets/train/layout-{peer}';stop_ns=state['stop']['wall_ns']
    first_fault_ns=peer_source(run,job,stage,stop_ns,state['recipe']['collector_repo'],state['checkpoint']['parent_root']);files=manifest(run)
    proof=dict(kind='completed_wave_peer_interruption',original_terminal=True,training_forbidden=True,cut=legacy.COLLECTOR_CUT,job=job,stage=stage,raw_root=str(run),files_sha256=files,
        root_stop_ns=stop_ns,first_fault_ns=first_fault_ns,parent_checkpoint=state['checkpoint'])
    with (output/'peer-backup.log').open('x') as log:proof['backup']=p.backup(run,files,p.DUAL/'datasets/counterfactual-targets/train'/run.name,log)
    path=output/f'acknowledged-peer-{peer}.json';atomic_json(path,proof)
    return dict(split=job['split'],layout_seed=peer,requested_layout_seed=job['requested_layout_seed'],status='refused',reason='operator_acknowledged_peer_stop',proof_path=str(path),proof_sha256=sha(path))


def verify_peer(row,proof,recipe):
    import triplet_production as p
    # An inherited peer disposition keeps its own admission authority. Its
    # ancestor plan is source-checked by parent_state, never relabelled here.
    wanted=Path(proof['parent_checkpoint']['parent_root']);seen=set()
    while True:
        admission=read(recipe['continuation_admission_path'])
        if Path(admission['prior_production_root'])==wanted:break
        ancestor=Path(admission['prior_production_root']);profile(ancestor)
        if ancestor in seen:raise ValueError('Cyclic peer admission ancestry')
        seen.add(ancestor);recipe=historical_parent(ancestor)
    d=profile(wanted);peer=d['peer']
    if sha(recipe['continuation_admission_path'])!=recipe['continuation_admission_sha256'] or admission.get('kind')!=KIND:
        raise ValueError('Peer disposition lacks its caller-bound completed admission')
    expected_job=next(j for j in reduce_waves(admission['inherited_waves'])['next_wave'] if j['layout_seed']==peer)
    expected_stage=next(e for e in read(wanted/f'wave-{d["wave"]:02d}/slot-{expected_job["instance"]}/stages.json') if e['name']=='collect-'+str(peer))
    if proof['job']!=expected_job or proof['stage']!=expected_stage or any(row[k]!=expected_job[k] for k in ('split','requested_layout_seed','layout_seed')):
        raise ValueError('Peer disposition job/stage differs from its parent')
    if (row not in admission['inherited_outcomes'] or row['status']!='refused' or row['reason']!='operator_acknowledged_peer_stop' or
        proof.get('original_terminal') is not True or proof.get('training_forbidden') is not True or
        proof['job']['layout_seed']!=peer or proof['parent_checkpoint']!=admission['parent_checkpoint'] or proof['root_stop_ns']!=admission['parent_stop_ns']):
        raise ValueError('Peer disposition identity differs')
    run=Path(proof['raw_root']);expected=p.ROOT/f'datasets/counterfactual-targets/train/layout-{peer}'
    if run!=expected or manifest(run)!=proof['files_sha256']:raise ValueError('Peer source bytes/path differ')
    first=peer_source(run,expected_job,expected_stage,proof['root_stop_ns'],recipe['collector_repo'],wanted)
    if proof.get('cut')!=legacy.COLLECTOR_CUT or first!=proof['first_fault_ns']:raise ValueError('Peer source time/cut differs')
    backup=proof['backup']
    if backup.get('passed') is not True or backup['files_sha256']!=proof['files_sha256'] or backup['destination']!=str(p.DUAL/'datasets/counterfactual-targets/train'/run.name):raise ValueError('Peer diagnostic backup is unverified')
    return proof


def initialization_source(state,raw=None):
    """Rebuild the original zero-member failure from the retired worker."""
    import triplet_production as p
    import scene_initialization as init
    from types import SimpleNamespace
    root=Path(state['checkpoint']['parent_root']);d=profile(root)
    job=next(j for j in state['state']['next_wave'] if j['layout_seed']==d['startup'])
    folder=root/f'wave-{d["wave"]:02d}/slot-{job["instance"]}'
    stages=read(folder/'stages.json');index=next(i for i,e in enumerate(stages) if e['name']=='collect-'+str(job['layout_seed']))
    stage=stages[index];session=int(next(e['name'] for e in reversed(stages[:index]) if e['name'].startswith('up-')).split('-')[1])
    observations=[json.loads(x) for x in read(folder/'native-observations.jsonl',True).splitlines()]
    native={k:observations[-1][k] for k in ('pid','start_ticks','pgid')}
    run=p.ROOT/f'datasets/counterfactual-targets/{job["split"]}/layout-{job["layout_seed"]}' if raw is None else Path(raw)
    worker=SimpleNamespace(root=folder,collector_repo=Path(state['recipe']['collector_repo']),cut=legacy.COLLECTOR_CUT,stages=stages[:index+1],session=session,native=native,raw_path=lambda _:run)
    error=init.failure(worker,job,stage)
    down=folder/f'owned-down-{session}.json'
    cert=read(down)
    if cert.get('passed') is not True or cert.get('errors')!=[] or cert.get('cleanup',{}).get('remaining')!=[] or any(cert.get('ports_listening',{}).values()):raise ValueError('Initialization parent instance retirement is not certified')
    if d['startup']==720144 and (state['stop']['wall_ns']!=1794978990139379403 or state['stop']['error']!="FileNotFoundError(2, 'No such file or directory')"):raise ValueError('Initialization parent root STOP differs')
    return job,error,down,run


def prepare_initialization_retry(state,output,budget):
    import scene_initialization as init
    job,error,down,run=initialization_source(state)
    init.archive(run,output/f'scene-init-{job["layout_seed"]}',error,output/'scene-init-backup.log')
    event=startup_retries.failed(budget,job['instance'],job,error,down,state['stop']['supervision'])
    return dict(event=event,job=job,evidence=error.evidence,owned_down_sha256=sha(down))


def verify_initialization_retry(admission):
    import scene_initialization as init
    record=admission.get('pending_initialization_retry')
    if not isinstance(record,dict):raise ValueError('Pending initialization retry evidence is missing')
    event=record['event'];path=Path(event['event_path']);budget=admission['startup_retry_budget']
    previous=read(Path(admission['prior_production_root'])/'production-finalization.json')['startup_retries']['events']
    if (path!=Path(budget['root'])/f'event-{len(previous)+1:04d}/event.json' or sha(path)!=event['sha256'] or read(path)!=dict(event['event'],id=event['id']) or
        event['event'].get('event')!='failed' or event['event'].get('kind')!='scene_initialization' or event['event']['job']!=record['job']):raise ValueError('Pending initialization ledger event differs')
    evidence=read(path.parent/'scene-init-evidence.json')
    if evidence!=record['evidence']:raise ValueError('Pending initialization evidence differs from the ledger')
    init.verify(evidence)
    root=Path(admission['prior_production_root']);ledger=admission['inherited_waves'];state=dict(checkpoint=admission['parent_checkpoint'],recipe=read(root/'plan.json'),state=reduce_waves(ledger),stop=read(root/'STOP.json'))
    job,error,down,_=initialization_source(state,evidence['archive']['path'])
    if job!=record['job'] or error.evidence!={k:v for k,v in evidence.items() if k!='archive'} or sha(down)!=record['owned_down_sha256']:raise ValueError('Pending initialization no longer matches the retired source')


def prepare(root,data_repo,cut,output,digest):
    import triplet_production as p
    root=Path(root);output=p.confined(output)
    with legacy.claim_run(root):
        p.require_host();p.frozen(cut);p.disk();state=parent_state(root,digest);d=profile(root)
        if Path(data_repo)!=Path(state['recipe']['collector_repo']):raise ValueError('Completed continuation data checkout differs')
        output.mkdir(parents=True,exist_ok=False);bound=output/'parent-evidence';bound.mkdir()
        for name,value in state['checkpoint']['files_sha256'].items():
            path=Path(name);dest=bound/path.relative_to(p.ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(read(path,True))
            if sha(dest)!=value:raise ValueError('Completed checkpoint changed during snapshot')
        original=read(state['recipe']['continuation_admission_path'])
        budget=startup_retries.initialize(Path(state['recipe']['startup_retry_budget']['root']),original['origin_checkpoint_sha256'],legacy.prior_retry_specs(original['origin_checkpoint_sha256']))
        data_recipe=dict(state['recipe'],runner_cut=cut,source_sha256=sha(p.SELF),production_root=str(output/'qualified'),runtime_root=str(output/'runtime'))
        for key in ('continuation_admission_path','continuation_admission_sha256'):data_recipe.pop(key,None)
        worker=object.__new__(p.Worker);worker.plan=data_recipe;worker.cut=legacy.COLLECTOR_CUT;worker.root=output/'qualified';worker.root.mkdir();worker.global_root=worker.root;worker.stop_file=worker.root/'STOP.json'
        worker.runtime=output/'runtime';worker.runtime.mkdir();worker.claim=p.InstanceClaim(worker.runtime);worker.collector_repo=Path(data_repo);worker.wrapper=p.REPO/'linux-env/scripts/collect_sim.sh';worker.stages=[]
        worker.ledger=p.OwnedProcesses(worker.runtime,time.time_ns());worker.env=dict(os.environ,SOARM_COLLECTOR_REPO=str(data_repo),SOARM_INSTANCE_ROOT=str(worker.runtime),SOARM_COLLECTION_ROOT=str(p.ROOT),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        capsule=p.load_bootstrap_capsule(state['recipe']['bootstrap_capsule_path'],state['recipe']['bootstrap_capsule_sha256']);(worker.runtime/'isaac.log').write_bytes(read(capsule['pin']['source']['path'],True))
        def retire():
            worker.ledger.scan()
            for pid in list(worker.ledger.handles):worker.ledger.send(pid,signal.SIGTERM)
        watch=p.DiskWatch(worker.root,retire,report=output/'preparation-disk-monitor.json');watch.start();pending=copy.deepcopy(state['pending_outcomes'])
        try:
            for seed in [s for s in (*d['exports'],d['identity'],d['peer']) if s is not None]:
                watch.require();job=next(j for j in state['state']['next_wave'] if j['layout_seed']==seed);worker.instance=job['instance']
                stage=next(e for e in read(root/f'wave-{d["wave"]:02d}/slot-{job["instance"]}/stages.json') if e['name']=='collect-'+str(seed))
                run=p.ROOT/f'datasets/counterfactual-targets/train/layout-{seed}'
                if seed in d['exports']:
                    retirement(run,stage);p.verify_sources(run,legacy.COLLECTOR_CUT,data_repo)
                    if read(run/'summary.json').get('complete') is not True:raise ValueError('Interrupted healthy collector is incomplete')
                    archive=run/'continuation-export-history'/output.name;archive.mkdir(parents=True,exist_ok=False)
                    for name in ('dataset','report','report-validation','production-media.json'):
                        source=run/name
                        if os.path.lexists(source):
                            if source.is_dir():p.checked_tree(source)
                            else:legacy.regular(source)
                            source.rename(archive/name)
                    row=worker.qualify(job,stage);legacy.require_completed_checks(read(row['proof_path']));pending.append(row)
                elif seed==d['identity']:
                    row=worker.qualify(job,stage)
                    if (row['status'],row['reason'])!=('refused','exact_identity'):raise ValueError('720123 did not prove the approved identity refusal')
                    pending.append(row)
                else:pending.append(peer_outcome(job,stage,state,output))
            startup=None
            if d['startup'] is not None:
                startup=prepare_initialization_retry(state,output,budget)
                budget=startup_retries.initialize(Path(budget['root']),original['origin_checkpoint_sha256'],[budget])
        finally:
            try:watch.close()
            finally:
                try:cleanup=worker.ledger.stop()
                finally:worker.claim.close()
            atomic_json(output/'preparation-cleanup.json',dict(passed=not cleanup['remaining'] and watch.fault is None,cleanup=cleanup))
            if cleanup['remaining'] or watch.fault is not None:raise ValueError('Completed preparation cleanup/disk is unresolved')
        inherited=[r for w in state['waves'] for r in w['outcomes']]+pending
        guards(inherited,state['state']['replacement_allocations'])
        value=dict(version=1,kind=KIND,collector_cut=legacy.COLLECTOR_CUT,collector_repo=str(data_repo),runner_cut=cut,
            prior_production_root=str(root),parent_checkpoint=state['checkpoint'],parent_checkpoint_sha256=digest,parent_stop_ns=state['stop']['wall_ns'],
            parent_evidence_path=str(bound),parent_evidence_files_sha256=manifest(bound),inherited_waves=state['waves'],pending_wave_outcomes=pending,inherited_outcomes=inherited,
            origin_root=original['origin_root'],origin_bindings=original['origin_bindings'],origin_checkpoint_sha256=original['origin_checkpoint_sha256'],startup_retry_budget=budget,
            next_production_root=str(root.parent/output.name),next_runtime_root=str(Path(state['recipe']['runtime_root']).parent/output.name),
            preparation_cleanup_path=str(output/'preparation-cleanup.json'),preparation_cleanup_sha256=sha(output/'preparation-cleanup.json'),
            preparation_disk_path=str(output/'preparation-disk-monitor.json'),preparation_disk_sha256=sha(output/'preparation-disk-monitor.json'))
        if startup is not None:value['pending_initialization_retry']=startup
        if d.get('partial'):value['slot5_to4']=True
        value['preparation_files_sha256']=manifest(output);admission=output/'admission.json';atomic_json(admission,value)
        recipe=load_plan(admission,sha(admission),cut);reverify(recipe)
        return dict(passed=True,admission_path=str(admission),admission_sha256=sha(admission),inherited=len(inherited),selected=sum(r['status']=='selected' for r in inherited))


def load_plan(path,digest,cut,_historical_source=None):
    import triplet_production as p
    path=legacy.regular(path);a=read(path)
    if sha(path)!=digest or a.get('kind')!=KIND or a.get('runner_cut')!=cut or a.get('collector_cut')!=legacy.COLLECTOR_CUT:
        raise ValueError('Caller-bound completed continuation source differs')
    if manifest(path.parent)!={**a['preparation_files_sha256'],path.name:digest}:raise ValueError('Completed admission evidence changed')
    parent=Path(a['prior_production_root']);d=profile(parent)
    if len(a['inherited_waves'])!=d['wave'] or sum(len(w['outcomes']) for w in a['inherited_waves'])!=d.get('base_dispositions',d['dispositions']) or len(reduce_waves(a['inherited_waves'])['selected'])!=d.get('base_selected',d['selected']):
        raise ValueError('Completed parent waves are missing; original 94/76 is forbidden')
    original=read(parent/'plan.json');recipe=copy.deepcopy(original)
    budget=a['startup_retry_budget'];prior=original['startup_retry_budget']
    if any(budget.get(k)!=prior[k] for k in ('root','root_identity','policy_sha256','origin_checkpoint_sha256')):
        raise ValueError('Completed continuation replaced its parent startup retry ledger')
    expected_prefix=[item['sha256'] for item in read(parent/'production-finalization.json')['startup_retries']['events']]
    if d['startup'] is not None:
        pending=a.get('pending_initialization_retry');verify_initialization_retry(a)
        expected_prefix.append(pending['event']['sha256'])
    if budget.get('prefix')!=expected_prefix or not budget.get('identity_receipt_path') or not budget.get('identity_receipt_sha256'):
        raise ValueError('Completed continuation lost its parent retry prefix/identity receipt')
    if d.get('partial'):
        if a.get('slot5_to4') is not True or len(a['inherited_outcomes'])!=172 or sum(r['status']=='selected' for r in a['inherited_outcomes'])!=136:
            raise ValueError('Final continuation must preserve 172/136 and exclude slot 5')
        recipe['slot5_to4']=True
        recipe['slots']=[s for s in recipe['slots'] if s['instance']!=5]
    elif 'slot5_to4' in a:raise ValueError('Slot reassignment is not authorized for this parent')
    recipe.update(runner_cut=cut,source_sha256=_historical_source or sha(p.SELF),continuation_kind=KIND,prior_production_root=a['prior_production_root'],
        production_root=a['next_production_root'],runtime_root=a['next_runtime_root'],continuation_admission_path=str(path),continuation_admission_sha256=digest,
        inherited_waves=a['inherited_waves'],inherited_outcomes=a['pending_wave_outcomes'],startup_retry_budget=a['startup_retry_budget'])
    root=Path(recipe['production_root'])
    if root!=parent.parent/path.parent.name or Path(recipe['runtime_root']).parent!=Path(original['runtime_root']).parent:
        raise ValueError('Completed continuation namespace differs')
    recipe['authoritative_registries']={s:str(root/(s+'-registry.json')) for s in ('train','heldout')}
    startup_retries.summary(recipe['startup_retry_budget'])
    if _historical_source is None:p.qualified_recipe(recipe)
    return recipe


def reverify(recipe):
    import triplet_production as p
    a=read(recipe['continuation_admission_path'])
    if load_plan(recipe['continuation_admission_path'],recipe['continuation_admission_sha256'],recipe['runner_cut'])!=recipe:raise ValueError('Completed recipe differs')
    parent=Path(a['prior_production_root']);d=profile(parent)
    state=parent_state(parent,a['parent_checkpoint_sha256'],allow_retry_extension=True)
    if state['checkpoint']!=a['parent_checkpoint'] or state['waves']!=a['inherited_waves'] or state['stop']['wall_ns']!=a['parent_stop_ns'] or manifest(Path(a['parent_evidence_path']))!=a['parent_evidence_files_sha256']:
        raise ValueError('Completed parent snapshot differs')
    if [r for w in a['inherited_waves'] for r in w['outcomes']]+a['pending_wave_outcomes']!=a['inherited_outcomes']:
        raise ValueError('Completed inherited membership differs')
    seeds=tuple(s for s in (*d['exports'],d['peer'],d['identity']) if s is not None)
    if d.get('partial'):
        if a['pending_wave_outcomes']!=state['pending_outcomes']:raise ValueError('Final continuation changed its parent partial outcomes')
    elif {(r['layout_seed'],r['split']) for r in a['pending_wave_outcomes']}!={(s,'train') for s in seeds} or len(a['pending_wave_outcomes'])!=len(seeds):
        raise ValueError('Missing/orphan interrupted wave disposition')
    expected={(j['split'],j['layout_seed']):j for w in state['waves'] for j in w['jobs']}
    expected.update({(j['split'],j['layout_seed']):j for j in state['state']['next_wave']})
    for row in a['inherited_outcomes']:
        proof=p.verify_outcome(row,recipe);legacy.backup_recheck(proof)
        if proof['job']!=expected[(row['split'],row['layout_seed'])]:raise ValueError('Inherited proof job differs from deterministic waves')
        if row['layout_seed'] in seeds:
            stage=next(e for e in read(parent/f'wave-{d["wave"]:02d}/slot-{proof["job"]["instance"]}/stages.json') if e['name']=='collect-'+str(row['layout_seed']))
            if proof['stage']!=stage:raise ValueError('Inherited interrupted stage differs from its source')
        if row['layout_seed'] in d['exports']:legacy.require_completed_checks(proof)
    cleanup=read(a['preparation_cleanup_path']);disk=read(a['preparation_disk_path'])
    if (sha(a['preparation_cleanup_path'])!=a['preparation_cleanup_sha256'] or cleanup.get('passed') is not True or cleanup['cleanup']['remaining']!=[] or
        sha(a['preparation_disk_path'])!=a['preparation_disk_sha256'] or disk.get('healthy') is not True or disk.get('closed') is not True or disk.get('fault') is not None):
        raise ValueError('Completed preparation lifecycle is unresolved')
    return dict(passed=True,outcomes=len(a['inherited_outcomes']))


def backup_evidence(recipe,log):
    import triplet_production as p
    a=read(recipe['continuation_admission_path']);reverify(recipe)
    snapshots=p.backup(Path(a['parent_evidence_path']),a['parent_evidence_files_sha256'],p.DUAL,log)
    parent=read(Path(a['prior_production_root'])/'plan.json');old=read(parent['continuation_admission_path']);histories=[]
    while old.get('kind')==KIND:
        prior=Path(parent['continuation_admission_path']).parent
        histories.append(p.backup(prior,manifest(prior),p.DUAL/prior.relative_to(p.ROOT),log))
        parent=read(Path(old['prior_production_root'])/'plan.json');old=read(parent['continuation_admission_path'])
    original=p.backup(Path(old['origin_snapshot_path']),old['origin_snapshot_files_sha256'],p.DUAL/Path(old['origin_root']).relative_to(p.ROOT),log)
    bound=p.backup(Path(old['bound_evidence_path']),old['bound_evidence_files_sha256'],p.DUAL,log)
    prior=Path(parent['continuation_admission_path']).parent
    history=p.backup(prior,manifest(prior),p.DUAL/prior.relative_to(p.ROOT),log)
    current=Path(recipe['continuation_admission_path']).parent
    preparation=p.backup(current,manifest(current),p.DUAL/current.relative_to(p.ROOT),log)
    return dict(parent=snapshots,original=original,bound=bound,prior_admission=history,completed_history=histories,preparation=preparation,startup_retry_budget=p.backup_startup_budget(recipe,log))
