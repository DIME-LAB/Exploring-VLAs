"""Source-bound continuation of the retired 5eaf completed wave.

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


def parent_root():
    import triplet_production as p
    return p.ROOT/'datasets/counterfactual-targets/production-v1/continuations/continuation-admission-5eaf5ad-v1'


def owner_path():
    import triplet_production as p
    return p.ROOT/'setup/counterfactual-continuation-5eaf5ad/coordinator.pid.identity.json'


def checkpoint(root):
    import triplet_production as p
    root=legacy.regular(root/'plan.json').parent
    if root!=parent_root():raise ValueError('Only the canonical 5eaf run may supply completed waves')
    files={str(f):sha(legacy.regular(f)) for f in sorted(root.rglob('*.json')) if 'terminal-diagnostics' not in f.relative_to(root).parts}
    owner=owner_path();down=owner.parent/'post-stop-owned-down.json'
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
    if old.get('runner_cut')!=PRIOR_RUNNER or old.get('collector_repo')!=str(data) or old.get('cut')!=legacy.COLLECTOR_CUT:
        raise ValueError('Completed parent source identity differs')
    p.data_checkout(data,legacy.COLLECTOR_CUT)
    source=subprocess.check_output(['git','-C',str(p.REPO),'show',PRIOR_RUNNER+':linux-env/scripts/collect/triplet_production.py'])
    expected=legacy.legacy_recipe(old['continuation_admission_path'],old['continuation_admission_sha256'],PRIOR_RUNNER,hashlib.sha256(source).hexdigest())
    if old!=expected or Path(old['production_root'])!=root:raise ValueError('Completed parent plan/admission differs')
    return old


def parent_state(root,digest,allow_retry_extension=False):
    import triplet_production as p
    root=Path(root);cp,actual=checkpoint(root)
    if not re.fullmatch('[0-9a-f]{64}',str(digest)) or actual!=digest:raise ValueError('Caller-bound completed checkpoint differs')
    recipe=historical_parent(root);stop=read(root/'STOP.json');final=read(root/'production-finalization.json');owner=read(owner_path());down=read(owner_path().parent/'post-stop-owned-down.json')
    if (stop.get('slot')!=6 or final.get('passed') is not False or final.get('retirement_errors') or
        final.get('cleanup',{}).get('remaining')!=[] or final.get('backup_children_cleanup',{}).get('remaining')!=[] or
        final.get('terminal_backups',{}).get('errors') or owner.get('state')!='stopped' or owner.get('supervision_error') or
        owner.get('cleanup_error') or owner.get('cleanup',{}).get('remaining')!=[] or down.get('passed') is not True or
        down.get('remaining_owned')!=[] or down.get('recorded_births',0)<=0 or
        set(down.get('ports_listening',{}))!={str(i) for i in range(8783,8791)} or any(v is not False for v in down['ports_listening'].values()) or
        down.get('managed_status',{}).get('supervision_ok') is not True or down.get('managed_stop',{}).get('supervision_ok') is not True or
        stop.get('supervision',{}).get('nonce')!=owner.get('nonce') or stop.get('supervision',{}).get('receipt_path')!=str(owner_path())):
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
    ledger=read(root/'wave-ledger.json');state=reduce_waves(ledger);progress=read(root/'progress.json')
    if (len(ledger)!=1 or len(ledger[0]['outcomes'])!=140 or len(state['selected'])!=113 or
        progress.get('wave')!=1 or progress.get('prior_selected')!=113 or progress.get('outcomes')!=[]):
        raise ValueError('Completed continuation requires the 140/113 wave, never the original 94/76 base')
    for i,wave in enumerate(ledger):
        if read(root/f'wave-{i:02d}.json')!=wave:raise ValueError('Completed wave/checkpoint differs')
        by_key={(r['split'],r['layout_seed']):r for r in wave['outcomes']}
        inherited={(r['split'],r['layout_seed']):r for r in recipe['inherited_outcomes']}
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
    partial=read(root/'wave-01.json')
    if partial['jobs']!=state['next_wave'] or partial['outcomes']!=[] or any((root/'wave-01').glob('slot-*/outcome-*.json')):
        raise ValueError('Parent partial replacement wave changed')
    current=startup_retries.summary(recipe['startup_retry_budget']);saved=final.get('startup_retries')
    if not isinstance(saved,dict):raise ValueError('Parent retry finalization is missing')
    if allow_retry_extension:
        if (current['policy']!=saved['policy'] or current['root']!=saved['root'] or
            current['events'][:len(saved['events'])]!=saved['events']):raise ValueError('Parent retry history was removed or replaced')
    elif any(current.get(k)!=v for k,v in saved.items()):raise ValueError('Retry ledger differs from the retired parent finalization')
    return dict(recipe=recipe,waves=ledger,state=state,checkpoint=cp,stop=stop)


def peer_source(run,job,stage,stop_ns,collector_repo):
    import triplet_production as p
    from target_triplets import spec
    if job['layout_seed']!=PEER or job['split']!='train':raise ValueError('Not the approved peer seed')
    config=read(run/'run.json')
    if config.get('normal_only') is not True or config.get('instance')!=job['instance'] or config.get('counterfactual_triplet')!=spec(PEER,'train'):
        raise ValueError('Peer collector seed/mode/slot differs')
    cert=retirement(run,stage,terminal=True);p.verify_sources(run,legacy.COLLECTOR_CUT,collector_repo)
    events=read(run/'terminal-fault.json')['events'];allowed={'MoveIt validity query','attempt disposition','before cleanup','after cleanup'}
    if (not cert['started_wall_ns']<stop_ns<cert['completed_wall_ns'] or not events or any(e.get('wall_ns',0)<=stop_ns or e.get('wall_ns',0)>cert['completed_wall_ns'] or e.get('source') not in allowed or
        'MoveIt state validity service is unavailable' not in str(e.get('error')) for e in events)):
        raise ValueError('Not the explicitly approved post-STOP 720122 peer interruption')
    return min(e['wall_ns'] for e in events)


def peer_outcome(job,stage,state,output):
    import triplet_production as p
    run=p.ROOT/f'datasets/counterfactual-targets/train/layout-{PEER}';stop_ns=state['stop']['wall_ns']
    first_fault_ns=peer_source(run,job,stage,stop_ns,state['recipe']['collector_repo']);files=manifest(run)
    proof=dict(kind='completed_wave_peer_interruption',original_terminal=True,training_forbidden=True,cut=legacy.COLLECTOR_CUT,job=job,stage=stage,raw_root=str(run),files_sha256=files,
        root_stop_ns=stop_ns,first_fault_ns=first_fault_ns,parent_checkpoint=state['checkpoint'])
    with (output/'peer-backup.log').open('x') as log:proof['backup']=p.backup(run,files,p.DUAL/'datasets/counterfactual-targets/train'/run.name,log)
    path=output/'acknowledged-peer-720122.json';atomic_json(path,proof)
    return dict(split=job['split'],layout_seed=PEER,requested_layout_seed=job['requested_layout_seed'],status='refused',reason='operator_acknowledged_peer_stop',proof_path=str(path),proof_sha256=sha(path))


def verify_peer(row,proof,recipe):
    import triplet_production as p
    admission=read(recipe['continuation_admission_path'])
    if sha(recipe['continuation_admission_path'])!=recipe['continuation_admission_sha256'] or admission.get('kind')!=KIND:
        raise ValueError('Peer disposition lacks its caller-bound completed admission')
    expected_job=next(j for j in reduce_waves(admission['inherited_waves'])['next_wave'] if j['layout_seed']==PEER)
    expected_stage=next(e for e in read(parent_root()/f'wave-01/slot-{expected_job["instance"]}/stages.json') if e['name']=='collect-'+str(PEER))
    if proof['job']!=expected_job or proof['stage']!=expected_stage or any(row[k]!=expected_job[k] for k in ('split','requested_layout_seed','layout_seed')):
        raise ValueError('Peer disposition job/stage differs from its parent')
    if (row not in admission['inherited_outcomes'] or row['status']!='refused' or row['reason']!='operator_acknowledged_peer_stop' or
        proof.get('original_terminal') is not True or proof.get('training_forbidden') is not True or
        proof['job']['layout_seed']!=PEER or proof['parent_checkpoint']!=admission['parent_checkpoint'] or proof['root_stop_ns']!=admission['parent_stop_ns']):
        raise ValueError('Peer disposition identity differs')
    run=Path(proof['raw_root']);expected=p.ROOT/f'datasets/counterfactual-targets/train/layout-{PEER}'
    if run!=expected or manifest(run)!=proof['files_sha256']:raise ValueError('Peer source bytes/path differ')
    first=peer_source(run,expected_job,expected_stage,proof['root_stop_ns'],recipe['collector_repo'])
    if proof.get('cut')!=legacy.COLLECTOR_CUT or first!=proof['first_fault_ns']:raise ValueError('Peer source time/cut differs')
    backup=proof['backup']
    if backup.get('passed') is not True or backup['files_sha256']!=proof['files_sha256'] or backup['destination']!=str(p.DUAL/'datasets/counterfactual-targets/train'/run.name):raise ValueError('Peer diagnostic backup is unverified')
    return proof


def prepare(root,data_repo,cut,output,digest):
    import triplet_production as p
    root=Path(root);output=p.confined(output)
    with legacy.claim_run(root):
        p.require_host();p.frozen(cut);p.disk();state=parent_state(root,digest)
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
        watch=p.DiskWatch(worker.root,retire,report=output/'preparation-disk-monitor.json');watch.start();pending=[]
        try:
            for seed in (*EXPORTS,IDENTITY,PEER):
                watch.require();job=next(j for j in state['state']['next_wave'] if j['layout_seed']==seed);worker.instance=job['instance']
                stage=next(e for e in read(root/f'wave-01/slot-{job["instance"]}/stages.json') if e['name']=='collect-'+str(seed))
                run=p.ROOT/f'datasets/counterfactual-targets/train/layout-{seed}'
                if seed in EXPORTS:
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
                elif seed==IDENTITY:
                    row=worker.qualify(job,stage)
                    if (row['status'],row['reason'])!=('refused','exact_identity'):raise ValueError('720123 did not prove the approved identity refusal')
                    pending.append(row)
                else:pending.append(peer_outcome(job,stage,state,output))
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
        value['preparation_files_sha256']=manifest(output);admission=output/'admission.json';atomic_json(admission,value)
        recipe=load_plan(admission,sha(admission),cut);reverify(recipe)
        return dict(passed=True,admission_path=str(admission),admission_sha256=sha(admission),inherited=len(inherited),selected=sum(r['status']=='selected' for r in inherited))


def load_plan(path,digest,cut):
    import triplet_production as p
    path=legacy.regular(path);a=read(path)
    if sha(path)!=digest or a.get('kind')!=KIND or a.get('runner_cut')!=cut or a.get('collector_cut')!=legacy.COLLECTOR_CUT:
        raise ValueError('Caller-bound completed continuation source differs')
    if manifest(path.parent)!={**a['preparation_files_sha256'],path.name:digest}:raise ValueError('Completed admission evidence changed')
    if Path(a['prior_production_root'])!=parent_root() or len(a['inherited_waves'])!=1 or len(a['inherited_waves'][0]['outcomes'])!=140:
        raise ValueError('Completed 140/113 parent is missing; original 94/76 is forbidden')
    original=read(parent_root()/'plan.json');recipe=copy.deepcopy(original)
    budget=a['startup_retry_budget'];prior=original['startup_retry_budget']
    if any(budget.get(k)!=prior[k] for k in ('root','root_identity','policy_sha256','origin_checkpoint_sha256')):
        raise ValueError('Completed continuation replaced its parent startup retry ledger')
    expected_prefix=[item['sha256'] for item in read(parent_root()/'production-finalization.json')['startup_retries']['events']]
    if budget.get('prefix')!=expected_prefix or not budget.get('identity_receipt_path') or not budget.get('identity_receipt_sha256'):
        raise ValueError('Completed continuation lost its parent retry prefix/identity receipt')
    recipe.update(runner_cut=cut,source_sha256=sha(p.SELF),continuation_kind=KIND,prior_production_root=a['prior_production_root'],
        production_root=a['next_production_root'],runtime_root=a['next_runtime_root'],continuation_admission_path=str(path),continuation_admission_sha256=digest,
        inherited_waves=a['inherited_waves'],inherited_outcomes=a['pending_wave_outcomes'],startup_retry_budget=a['startup_retry_budget'])
    root=Path(recipe['production_root'])
    if root!=parent_root().parent/path.parent.name or Path(recipe['runtime_root']).parent!=Path(original['runtime_root']).parent:
        raise ValueError('Completed continuation namespace differs')
    recipe['authoritative_registries']={s:str(root/(s+'-registry.json')) for s in ('train','heldout')}
    startup_retries.summary(recipe['startup_retry_budget']);p.qualified_recipe(recipe)
    return recipe


def reverify(recipe):
    import triplet_production as p
    a=read(recipe['continuation_admission_path'])
    if load_plan(recipe['continuation_admission_path'],recipe['continuation_admission_sha256'],recipe['runner_cut'])!=recipe:raise ValueError('Completed recipe differs')
    state=parent_state(parent_root(),a['parent_checkpoint_sha256'],allow_retry_extension=True)
    if state['checkpoint']!=a['parent_checkpoint'] or state['waves']!=a['inherited_waves'] or state['stop']['wall_ns']!=a['parent_stop_ns'] or manifest(Path(a['parent_evidence_path']))!=a['parent_evidence_files_sha256']:
        raise ValueError('Completed parent snapshot differs')
    if [r for w in a['inherited_waves'] for r in w['outcomes']]+a['pending_wave_outcomes']!=a['inherited_outcomes']:
        raise ValueError('Completed inherited membership differs')
    if {(r['layout_seed'],r['split']) for r in a['pending_wave_outcomes']}!={(s,'train') for s in (*EXPORTS,PEER,IDENTITY)}:
        raise ValueError('Missing/orphan interrupted wave disposition')
    expected={(j['split'],j['layout_seed']):j for w in state['waves'] for j in w['jobs']}
    expected.update({(j['split'],j['layout_seed']):j for j in state['state']['next_wave']})
    for row in a['inherited_outcomes']:
        proof=p.verify_outcome(row,recipe);legacy.backup_recheck(proof)
        if proof['job']!=expected[(row['split'],row['layout_seed'])]:raise ValueError('Inherited proof job differs from deterministic waves')
        if row['layout_seed'] in (*EXPORTS,PEER,IDENTITY):
            stage=next(e for e in read(parent_root()/f'wave-01/slot-{proof["job"]["instance"]}/stages.json') if e['name']=='collect-'+str(row['layout_seed']))
            if proof['stage']!=stage:raise ValueError('Inherited interrupted stage differs from its source')
        if row['layout_seed'] in EXPORTS:legacy.require_completed_checks(proof)
    cleanup=read(a['preparation_cleanup_path']);disk=read(a['preparation_disk_path'])
    if (sha(a['preparation_cleanup_path'])!=a['preparation_cleanup_sha256'] or cleanup.get('passed') is not True or cleanup['cleanup']['remaining']!=[] or
        sha(a['preparation_disk_path'])!=a['preparation_disk_sha256'] or disk.get('healthy') is not True or disk.get('closed') is not True or disk.get('fault') is not None):
        raise ValueError('Completed preparation lifecycle is unresolved')
    return dict(passed=True,outcomes=len(a['inherited_outcomes']))


def backup_evidence(recipe,log):
    import triplet_production as p
    a=read(recipe['continuation_admission_path']);reverify(recipe)
    snapshots=p.backup(Path(a['parent_evidence_path']),a['parent_evidence_files_sha256'],p.DUAL,log)
    parent=read(parent_root()/'plan.json');old=read(parent['continuation_admission_path'])
    original=p.backup(Path(old['origin_snapshot_path']),old['origin_snapshot_files_sha256'],p.DUAL/Path(old['origin_root']).relative_to(p.ROOT),log)
    bound=p.backup(Path(old['bound_evidence_path']),old['bound_evidence_files_sha256'],p.DUAL,log)
    prior=Path(parent['continuation_admission_path']).parent
    history=p.backup(prior,manifest(prior),p.DUAL/prior.relative_to(p.ROOT),log)
    current=Path(recipe['continuation_admission_path']).parent
    preparation=p.backup(current,manifest(current),p.DUAL/current.relative_to(p.ROOT),log)
    return dict(parent=snapshots,original=original,bound=bound,prior_admission=history,preparation=preparation,startup_retry_budget=p.backup_startup_budget(recipe,log))
