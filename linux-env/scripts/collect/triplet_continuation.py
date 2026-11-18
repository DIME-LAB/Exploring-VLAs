"""Checked same-run continuation; prior failures remain immutable evidence.

Preparation is CPU-only. Collection requires a separately reviewed outer runner,
a caller-published admission SHA and the normal --go/release/ownership barriers.
"""
import copy,hashlib,inspect,json,os,shlex,stat,time,signal
from contextlib import contextmanager
from pathlib import Path
from storage import atomic_json,confined
from triplet_artifacts import read,sha,manifest,checked_tree,retirement
from triplet_production_plan import initial_wave,guards
from backup_space import checked_script_payload,check_script_receiver_code

COLLECTOR_CUT='503e39a692d47056ea325b7a20baa2f1d6900c91'
INTERRUPTED_EXPORTS=(720087,720092)
PEER_INTERRUPTION=720097


@contextmanager
def claim_run(prior_production_root=None):
    """One preparation/coordinator owns the original run; a second invocation refuses."""
    import triplet_production as p
    root=p.confined(p.ROOT/'runtime/counterfactual-continuation-claim');root.mkdir(parents=True,exist_ok=True)
    claim=p.InstanceClaim(root)
    try:
        previous=p.ROOT/'datasets/counterfactual-targets/production-v1/continuations'
        allowed=set() if prior_production_root is None else {Path(prior_production_root)}
        if allowed:
            from completed_continuation import profile,parent_root
            profile(next(iter(allowed)))
            if next(iter(allowed))!=parent_root():allowed.add(parent_root())
        if previous.exists() and set(previous.iterdir())-allowed:raise FileExistsError('A continuation already exists; a new recovery decision is required')
        yield
    finally:claim.close()


def regular(path):
    path=Path(path)
    for parent in path.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):raise ValueError('Non-directory evidence ancestor')
    read(path,True)
    return path


def backup_recheck(proof):
    """Run the existing regular-tree hash checker on the destination, read-only."""
    import triplet_production as p
    import triplet_artifacts as a
    destination=Path(proof['backup']['destination'])
    if not destination.is_relative_to(p.DUAL):raise ValueError('Backup is outside this run')
    code='import os,stat,json,hashlib\nfrom pathlib import Path\n'
    code+='\n'.join(inspect.getsource(f) for f in (a.read,a.sha,a.checked_tree,a.manifest))
    code+='\nactual=manifest(Path('+repr(str(destination))+'))\n'
    code+='assert actual=='+repr(proof['files_sha256'])+', "Backup bytes changed"\n'
    code+='print(json.dumps({"passed":True,"files":len(actual)}))\n'
    answer=json.loads(p.subprocess.check_output([*p.SSH,'python3 -c '+shlex.quote(check_script_receiver_code())],input=checked_script_payload(code)))
    if answer!={'passed':True,'files':len(proof['files_sha256'])}:raise ValueError('Backup recheck was not certified')
    return answer


def origin_state(origin,collector_repo,checkpoint_sha):
    """Rebuild a partial wave, including every source-bound disposition and birth."""
    import triplet_production as p
    origin=confined(origin);collector_repo=confined(collector_repo)
    if origin!=p.ROOT/'datasets/counterfactual-targets/production-v1':raise ValueError('Only the same canonical production run may continue')
    p.data_checkout(collector_repo,COLLECTOR_CUT)
    if sha(origin/'progress.json')!=checkpoint_sha:raise ValueError('Original checkpoint differs from the caller binding')
    recipe=read(origin/'plan.json')
    if recipe['cut']!=COLLECTOR_CUT or recipe['jobs']!=initial_wave() or recipe!=p.plan(COLLECTOR_CUT,repo=collector_repo):raise ValueError('Original source cut/jobs/plan differ')
    stop=read(origin/'STOP.json');final=read(origin/'production-finalization.json')
    if (stop.get('error')!="RuntimeError('Startup failed; see up-3.log')" or stop.get('slot')!=5 or
        final.get('passed') is not False or final.get('retirement_errors') or final.get('cleanup',{}).get('remaining')!=[] or
        final.get('terminal_backups',{}).get('errors') or final.get('backup_children_cleanup',{}).get('remaining')!=[]):
        raise ValueError('Original stopped invocation is not a retired startup failure')
    context=stop['supervision'];owner=Path(context['receipt_path']);managed=read(regular(owner))
    if (managed.get('state')!='stopped' or managed.get('supervision_error') or managed.get('cleanup_error') or
        managed.get('cleanup',{}).get('remaining')!=[] or managed.get('nonce')!=context['nonce']):
        raise ValueError('Original owner did not certify retirement')
    cert=owner.parent/'post-stop-owned-down.json';down=read(regular(cert))
    if (down.get('passed') is not True or any(down.get(k)!=[] for k in ('remaining','scoped_survivors','unresolved_environments','original_cleanup_failures')) or
        any(down.get('ports',{}).values()) or down.get('production_passed') is not False):
        raise ValueError('Independent owned-down certificate is unresolved')
    identities={};bindings={str(origin/n):sha(origin/n) for n in ('plan.json','progress.json','STOP.json','production-finalization.json','wave-00.json')}
    bindings.update({str(owner):sha(owner),str(cert):sha(cert)})
    def collect(value):
        if isinstance(value,dict):
            if isinstance(value.get('pid'),int) and value.get('start_ticks') is not None:
                identities[(value['pid'],str(value['start_ticks']))]=value
            for one in value.values():collect(one)
        elif isinstance(value,list):
            for one in value:collect(one)
    for item in down['sources']:
        path=regular(item['path']);value=item['sha256']
        if sha(path)!=value:raise ValueError('Retirement source changed')
        bindings[str(path)]=value;collect(read(path))
    for pid,birth in identities:
        current=p.identity(pid)
        if current and str(current['start_ticks'])==birth and current['state']!='Z':raise ValueError('Original owned process remains live')
    wave=read(origin/'wave-00.json')
    if wave['jobs']!=initial_wave() or wave['outcomes']!=[]:raise ValueError('Original partial wave changed')
    if (origin/'wave-ledger.json').exists():raise ValueError('This entrance requires the stopped partial first wave')
    outcomes=[read(f) for f in sorted((origin/'wave-00').glob('slot-*/outcome-*.json'))]
    progress=read(origin/'progress.json')
    if progress['wave']!=0 or progress['prior_selected']!=0 or sorted(progress['outcomes'],key=lambda r:r['layout_seed'])!=sorted(outcomes,key=lambda r:r['layout_seed']):
        raise ValueError('Original outcome/checkpoint membership differs')
    paths={r['proof_path'] for r in outcomes}
    if paths!={str(f) for f in (origin/'proofs').glob('*/*.json')}:raise ValueError('Duplicate/orphan original proof')
    seen=set();verified=[];data_recipe=dict(recipe,collector_repo=str(collector_repo))
    for row in outcomes:
        key=(row['split'],row['layout_seed'])
        if key in seen:raise ValueError('Duplicate original outcome')
        seen.add(key);proof=p.verify_outcome(row,data_recipe)
        job=next((j for j in initial_wave() if (j['split'],j['layout_seed'])==key),None)
        if job is None or proof['job']!=job:raise ValueError('Original job assignment changed')
        worker=origin/f'wave-00/slot-{job["instance"]}'
        for name in ('outcome-','admitted-'):
            path=worker/(name+str(job['layout_seed'])+'.json')
            if read(regular(path))!=row:raise ValueError('Original admission identity changed')
            bindings[str(path)]=sha(path)
        backup_recheck(proof);verified.append(row)
    guards([r for j in initial_wave() for r in verified if (r['split'],r['layout_seed'])==(j['split'],j['layout_seed'])],[])
    return dict(recipe=recipe,outcomes=verified,bindings=bindings,owner_nonce=context['nonce'],stop_wall_ns=stop['wall_ns'])


def verify_interruption(outcome,proof,recipe):
    """One explicitly acknowledged post-STOP failure may be replaced, never accepted."""
    import triplet_production as p
    admission=read(regular(recipe['continuation_admission_path']))
    if sha(recipe['continuation_admission_path'])!=recipe['continuation_admission_sha256']:raise ValueError('Continuation admission changed')
    job=proof['job'];seed=job['layout_seed'];kind=proof.get('kind')
    if any(outcome[k]!=job[k] for k in ('split','layout_seed','requested_layout_seed')):raise ValueError('Interruption membership changed')
    expected=next((j for j in initial_wave() if j['layout_seed']==seed),None)
    if job!=expected:raise ValueError('Interruption job changed')
    peer=kind=='acknowledged_peer_interruption' and seed==PEER_INTERRUPTION
    if not peer:raise ValueError('Only the explicit post-STOP peer interruption is replaceable here')
    reason='operator_acknowledged_peer_stop'
    if (outcome not in admission['inherited_outcomes'] or outcome['status']!='refused' or outcome['reason']!=reason or
        proof['original_terminal'] is not True or proof['cut']!=COLLECTOR_CUT or proof['origin_bindings']!=admission['origin_bindings']):
        raise ValueError('Interruption is not bound to the explicit continuation admission')
    for name,value in proof['origin_bindings'].items():
        if sha(regular(name))!=value:raise ValueError('Original failure history changed')
    original_stop=Path(admission['origin_root'])/'STOP.json'
    if str(original_stop) not in proof['origin_bindings'] or proof['stop_wall_ns']!=read(original_stop)['wall_ns']:raise ValueError('Original stop time differs')
    run=Path(proof['raw_root'])
    if run!=p.ROOT/f'datasets/counterfactual-targets/train/layout-{seed}':raise ValueError('Interrupted raw root differs')
    if manifest(run)!=proof['files_sha256']:raise ValueError('Interrupted raw bytes changed')
    p.verify_sources(run,COLLECTOR_CUT,recipe['collector_repo']);retirement(run,proof['stage'],terminal=True)
    events=read(run/'terminal-fault.json').get('events',[])
    if not events or any(e.get('wall_ns',0)<=proof['stop_wall_ns'] or 'MoveIt state validity query did not complete' not in e.get('error','') for e in events):
        raise ValueError('Unknown or pre-STOP terminal fault')
    backup=proof['backup']
    expected=p.DUAL/'datasets/counterfactual-targets/train'/run.name
    if backup.get('passed') is not True or backup['files_sha256']!=proof['files_sha256'] or backup['destination']!=str(expected):raise ValueError('Interrupted bytes are not backed up')
    return proof


def require_completed_checks(proof):
    """An interrupted complete collector needs a fresh completed media check.

    The frozen exporter publishes a finished certificate only on success.
    Its exceptions/in-progress marker cannot authorize a permanent refusal.
    """
    stage=proof.get('media_stage');receipt=proof.get('media_receipt');media=proof.get('media')
    seed=proof['job']['layout_seed'];run=Path(proof['raw_root'])
    if not isinstance(stage,dict) or stage.get('name')!='media-'+str(seed) or not isinstance(receipt,dict) or not isinstance(media,dict):raise ValueError('Completed interrupted media evidence missing')
    expected=0 if media.get('passed') is True else 2 if media.get('passed') is False else None
    if expected is None or stage.get('returncode')!=expected or stage.get('wall_s',-1)<0 or stage.get('pid',0)<=0:raise ValueError('Media exit/verdict evidence differs')
    if (receipt.get('path')!='production-media.json' or receipt.get('sha256')!=proof['files_sha256'].get('production-media.json') or
        receipt.get('written_wall_ns',0)<stage.get('started_wall_ns',1) or
        receipt['written_wall_ns']!=(run/'production-media.json').stat().st_mtime_ns):raise ValueError('Fresh media receipt evidence missing')
    if expected==2:
        if proof['status']!='refused' or proof['reason'] not in ('exact_identity','image_psnr','departure_labels') or proof.get('export_stages')!={}:raise ValueError('Failed media verdict differs from disposition')
    elif (proof['status']!='selected' or set(proof.get('export_stages',{}))!={'report','report-validation'} or
          any(row.get('returncode')!=0 for row in proof['export_stages'].values())):raise ValueError('Completed export evidence missing')
    return 'media' if expected==2 else 'media+export+validate-existing'


def acknowledged_outcome(job,stage,state,output):
    import triplet_production as p
    seed=job['layout_seed'];run=p.ROOT/f'datasets/counterfactual-targets/train/layout-{seed}'
    files=manifest(run)
    with (output/f'interruption-{seed}-backup.log').open('x') as log:
        backup=p.backup(run,files,p.DUAL/'datasets/counterfactual-targets/train'/run.name,log)
    proof=dict(kind='acknowledged_peer_interruption',original_terminal=True,cut=COLLECTOR_CUT,job=job,stage=stage,raw_root=str(run),files_sha256=files,
        backup=backup,group=None,origin_bindings=state['bindings'],stop_wall_ns=state['stop_wall_ns'])
    path=output/f'acknowledged-interruption-{seed}.json';atomic_json(path,proof)
    return dict(split=job['split'],requested_layout_seed=job['requested_layout_seed'],layout_seed=seed,status='refused',
        reason='operator_acknowledged_peer_stop',proof_path=str(path),proof_sha256=sha(path))


def snapshot_origin(origin,target):
    """Freeze the stopped headers/proofs before a child continuation namespace exists."""
    files=manifest(origin);target.mkdir()
    for name,digest in files.items():
        import triplet_production as p
        p.disk()
        destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(read(Path(origin)/name,True));stream.flush();os.fsync(stream.fileno())
        if sha(destination)!=digest:raise ValueError('Original evidence changed while snapshotting')
    if manifest(origin)!=files or manifest(target)!=files:raise ValueError('Original evidence snapshot differs')
    return files


def backup_legacy_evidence(recipe,log):
    """Keep projected proof paths readable, including proofs from the stopped invocation."""
    import triplet_production as p
    if load_legacy_plan(recipe['continuation_admission_path'],recipe['continuation_admission_sha256'],recipe['runner_cut'])!=recipe:raise ValueError('Continuation admission changed before backup')
    admission=read(recipe['continuation_admission_path'])
    source=Path(admission['origin_snapshot_path']);files=admission['origin_snapshot_files_sha256']
    original=p.backup(source,files,p.DUAL/Path(admission['origin_root']).relative_to(p.ROOT),log)
    bound=p.backup(Path(admission['bound_evidence_path']),admission['bound_evidence_files_sha256'],p.DUAL,log)
    source=Path(recipe['continuation_admission_path']).parent
    preparation=p.backup(source,manifest(source),p.DUAL/source.relative_to(p.ROOT),log)
    retry_budget=None
    if recipe.get('startup_retry_budget'):
        retry_budget=p.backup_startup_budget(recipe,log)
    return dict(original_evidence=original,bound_evidence=bound,preparation=preparation,startup_retry_budget=retry_budget)


def prior_retry_specs(checkpoint):
    """A new preparation must reconcile every surviving admission's ledger pin."""
    import triplet_production as p
    specs=[]
    for path in sorted((p.ROOT/'setup').glob('continuation-admission-*/admission.json')):
        admission=read(regular(path))
        if admission.get('origin_checkpoint_sha256')!=checkpoint:continue
        if 'startup_retry_budget' in admission:
            spec=admission['startup_retry_budget']
            if not isinstance(spec,dict):raise ValueError('Prior admission has an invalid startup retry binding')
            specs.append(spec)
    return specs


def prepare(origin,collector_repo,runner_cut,output,checkpoint_sha):
    """CPU-only qualification, completion of healthy exports, and a caller-bound admission."""
    with claim_run():
        import triplet_production as p
        p.require_host();p.frozen(runner_cut);p.disk()
        output=confined(output);output.mkdir(parents=True,exist_ok=False)
        state=origin_state(origin,collector_repo,checkpoint_sha)
        import startup_retries
        retry_budget=startup_retries.initialize(p.ROOT/'runtime/counterfactual-startup-retries'/checkpoint_sha,checkpoint_sha,prior_retry_specs(checkpoint_sha))
        snapshot=output/'origin-evidence';snapshot_files=snapshot_origin(origin,snapshot)
        bound=output/'bound-evidence';bound.mkdir();bound_files={}
        for name,digest in state['bindings'].items():
            p.disk();relative=str(Path(name).relative_to(p.ROOT));destination=bound/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(read(name,True));stream.flush();os.fsync(stream.fileno())
            if sha(destination)!=digest:raise ValueError('Bound original evidence changed')
            bound_files[relative]=digest
        token=output.name;original=Path(origin);data_recipe=dict(state['recipe'],collector_repo=str(collector_repo),runner_cut=runner_cut,
            production_root=str(output/'qualified'),runtime_root=str(output/'runtime'))
        root=Path(data_recipe['production_root']);root.mkdir();runtime=Path(data_recipe['runtime_root']);runtime.mkdir()
        inherited=list(state['outcomes']);completed=[]
        # Reuse the real Worker's media/export/backup path, without starting Isaac.
        worker=object.__new__(p.Worker);worker.plan=data_recipe;worker.cut=COLLECTOR_CUT;worker.instance=3;worker.wave=0
        worker.root=root;worker.global_root=root;worker.stop_file=root/'STOP.json';worker.wrapper=p.REPO/'linux-env/scripts/collect_sim.sh'
        worker.collector_repo=Path(collector_repo);worker.stages=[];worker.runtime=runtime;worker.claim=p.InstanceClaim(runtime)
        worker.ledger=p.OwnedProcesses(runtime,time.time_ns());worker.env=dict(os.environ,SOARM_COLLECTOR_REPO=str(collector_repo),
            SOARM_INSTANCE_ROOT=str(runtime),SOARM_COLLECTION_ROOT=str(p.ROOT),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        # Export's existing device-table validation is read-only; no Kit is launched.
        capsule=p.load_bootstrap_capsule(state['recipe']['bootstrap_capsule_path'],state['recipe']['bootstrap_capsule_sha256'])
        (runtime/'isaac.log').write_bytes(read(capsule['pin']['source']['path'],True))
        def retire_cpu():
            worker.ledger.scan()
            for pid in list(worker.ledger.handles):worker.ledger.send(pid,signal.SIGTERM)
        watch=p.DiskWatch(root,retire_cpu,report=output/'preparation-disk-monitor.json');watch.start()
        try:
            for seed in INTERRUPTED_EXPORTS:
                p.disk();job=next(j for j in initial_wave() if j['layout_seed']==seed)
                stage=next(x for x in read(original/f'wave-00/slot-{job["instance"]}/stages.json') if x['name']=='collect-'+str(seed))
                run=p.ROOT/f'datasets/counterfactual-targets/train/layout-{seed}'
                retirement(run,stage);p.verify_sources(run,COLLECTOR_CUT,collector_repo)
                if read(run/'summary.json').get('complete') is not True:raise ValueError('Interrupted export lacks a complete collector')
                saved=run/'continuation-export-history'/token;saved.mkdir(parents=True,exist_ok=False)
                for name in ('dataset','report','report-validation','production-media.json'):
                    source=run/name
                    if source.exists() or source.is_symlink():
                        if source.is_dir():checked_tree(source)
                        else:regular(source)
                        source.rename(saved/name)
                worker.instance=job['instance']
                # Tool, dependency, I/O and incomplete-export failures abort. Only
                # the real classifier's completed media verdict can refuse a group.
                outcome=worker.qualify(job,stage)
                inherited.append(outcome);completed.append(seed)
            job=next(j for j in initial_wave() if j['layout_seed']==PEER_INTERRUPTION)
            stage=next(x for x in read(original/f'wave-00/slot-{job["instance"]}/stages.json') if x['name']=='collect-'+str(PEER_INTERRUPTION))
            run=p.ROOT/f'datasets/counterfactual-targets/train/layout-{PEER_INTERRUPTION}'
            retirement(run,stage,terminal=True);p.verify_sources(run,COLLECTOR_CUT,collector_repo)
            inherited.append(acknowledged_outcome(job,stage,state,output))
        finally:
            try:
                watch.close()
            finally:
                try:cleanup=worker.ledger.stop()
                finally:worker.claim.close()
            atomic_json(output/'preparation-cleanup.json',dict(passed=not cleanup['remaining'] and watch.fault is None,cleanup=cleanup))
            if watch.fault is not None:raise RuntimeError('CPU preparation disk monitor failed')
            if cleanup['remaining']:raise RuntimeError('CPU preparation cleanup unresolved')
        value=dict(version=1,kind='same-run-continuation',collector_cut=COLLECTOR_CUT,collector_repo=str(collector_repo),runner_cut=runner_cut,startup_retry_budget=retry_budget,
            origin_root=str(origin),origin_checkpoint_sha256=checkpoint_sha,origin_bindings=state['bindings'],origin_owner_nonce=state['owner_nonce'],
            origin_snapshot_path=str(snapshot),origin_snapshot_files_sha256=snapshot_files,bound_evidence_path=str(bound),bound_evidence_files_sha256=bound_files,
            preparation_disk_path=str(output/'preparation-disk-monitor.json'),preparation_disk_sha256=sha(output/'preparation-disk-monitor.json'),
            inherited_outcomes=inherited,completed_exports=completed,acknowledged_terminal_seed=PEER_INTERRUPTION,
            next_production_root=str(original/'continuations'/token),next_runtime_root=str(Path(state['recipe']['runtime_root'])/'continuations'/token),
            preparation_cleanup_path=str(output/'preparation-cleanup.json'),preparation_cleanup_sha256=sha(output/'preparation-cleanup.json'))
        value['preparation_files_sha256']=manifest(output)
        admission=output/'admission.json';atomic_json(admission,value)
        recipe=load_legacy_plan(admission,sha(admission),runner_cut)
        reverify_legacy(recipe)
        return dict(passed=True,admission_path=str(admission),admission_sha256=sha(admission),inherited=len(inherited),selected=sum(r['status']=='selected' for r in inherited))


def legacy_recipe(path,digest,runner_cut,source_sha256):
    import triplet_production as p
    path=regular(path)
    raw=read(path,True)
    if not digest or hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Caller-published continuation SHA required')
    a=json.loads(raw)
    if 'preparation_files_sha256' not in a or manifest(path.parent)!={**a['preparation_files_sha256'],path.name:digest}:raise ValueError('Prepared evidence tree changed')
    if a.get('kind')!='same-run-continuation' or a.get('collector_cut')!=COLLECTOR_CUT or a.get('runner_cut')!=runner_cut:raise ValueError('Continuation source/runner identity differs')
    import startup_retries
    if not isinstance(a.get('startup_retry_budget'),dict):raise ValueError('Continuation startup retry budget is missing')
    retry_budget=a['startup_retry_budget']
    if Path(retry_budget['root'])!=p.ROOT/'runtime/counterfactual-startup-retries'/a['origin_checkpoint_sha256'] or retry_budget['origin_checkpoint_sha256']!=a['origin_checkpoint_sha256']:raise ValueError('Startup retry budget belongs to another original run')
    startup_retries.summary(retry_budget)
    for name,value in a['origin_bindings'].items():
        if sha(regular(name))!=value:raise ValueError('Original history changed')
    origin=Path(a['origin_root']);original=read(origin/'plan.json')
    recipe=copy.deepcopy(original)
    recipe.update(collector_repo=a['collector_repo'],runner_cut=runner_cut,source_sha256=source_sha256,
        production_root=a['next_production_root'],runtime_root=a['next_runtime_root'],startup_configure_timeout_s=300,
        continuation_admission_path=str(path),continuation_admission_sha256=digest,inherited_outcomes=a['inherited_outcomes'],startup_retry_budget=retry_budget)
    root=Path(recipe['production_root'])
    if root.parent!=origin/'continuations' or Path(recipe['runtime_root']).parent!=Path(original['runtime_root'])/'continuations':raise ValueError('Continuation namespace differs')
    recipe['authoritative_registries']={s:str(root/(s+'-registry.json')) for s in ('train','heldout')}
    return recipe


def load_legacy_plan(path,digest,runner_cut):
    import triplet_production as p
    recipe=legacy_recipe(path,digest,runner_cut,sha(p.SELF));p.qualified_recipe(recipe)
    return recipe


def reverify_legacy(recipe):
    import triplet_production as p
    if load_legacy_plan(recipe['continuation_admission_path'],recipe['continuation_admission_sha256'],recipe['runner_cut'])!=recipe:raise ValueError('Continuation recipe changed')
    a=read(recipe['continuation_admission_path'])
    if manifest(Path(a['origin_snapshot_path']))!=a['origin_snapshot_files_sha256']:raise ValueError('Original evidence snapshot changed')
    if manifest(Path(a['bound_evidence_path']))!=a['bound_evidence_files_sha256']:raise ValueError('Bound evidence snapshot changed')
    monitor=read(a['preparation_disk_path'])
    if sha(a['preparation_disk_path'])!=a['preparation_disk_sha256'] or monitor.get('healthy') is not True or monitor.get('closed') is not True or monitor.get('fault') is not None:raise ValueError('Preparation disk monitor unverified')
    state=origin_state(a['origin_root'],a['collector_repo'],a['origin_checkpoint_sha256'])
    if {str(Path(n).relative_to(p.ROOT)):v for n,v in state['bindings'].items()}!=a['bound_evidence_files_sha256']:raise ValueError('Bound evidence membership changed')
    if state['bindings']!=a['origin_bindings'] or state['owner_nonce']!=a['origin_owner_nonce']:raise ValueError('Original invocation binding changed')
    cleanup=read(a['preparation_cleanup_path'])
    if sha(a['preparation_cleanup_path'])!=a['preparation_cleanup_sha256'] or cleanup.get('passed') is not True or cleanup['cleanup']['remaining']!=[]:raise ValueError('Preparation cleanup unverified')
    known={(x['split'],x['layout_seed']):x for x in state['outcomes']}
    seen=set();actual=[]
    for row in a['inherited_outcomes']:
        key=(row['split'],row['layout_seed'])
        if key in seen:raise ValueError('Duplicate inherited outcome')
        seen.add(key)
        if key in known and row!=known[key]:raise ValueError('Original disposition changed')
        if key not in known and row['layout_seed'] not in (*INTERRUPTED_EXPORTS,PEER_INTERRUPTION):raise ValueError('Orphan inherited outcome')
        proof=p.verify_outcome(row,recipe);backup_recheck(proof);actual.append(row)
    expected=set(known)|{('train',s) for s in (*INTERRUPTED_EXPORTS,PEER_INTERRUPTION)}
    if seen!=expected:raise ValueError('Missing original or interrupted group disposition')
    guards([r for j in initial_wave() for r in actual if (r['split'],r['layout_seed'])==(j['split'],j['layout_seed'])],[])
    return dict(passed=True,outcomes=len(actual))


def load_plan(path,digest,runner_cut):
    if read(regular(path)).get('kind')!='completed-wave-continuation':
        raise ValueError('This run requires source-bound completed waves; the original 94/76 admission is forbidden')
    from completed_continuation import load_plan as load
    return load(path,digest,runner_cut)


def reverify(recipe):
    from completed_continuation import reverify as check
    return check(recipe)


def backup_evidence(recipe,log):
    from completed_continuation import backup_evidence as save
    return save(recipe,log)
