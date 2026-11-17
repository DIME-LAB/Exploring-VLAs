"""Source-bound production dispositions and RTX-native registry entries."""
import ast,hashlib,json,os,re,stat
from pathlib import Path
from storage import CANONICAL_ROOT
import collection_status
from contract import COLORS,visible_target,timing_audit
from episode_policy import disposition
from scene_identity import physical_state,verify_pose_evidence
from target_triplets import spec,member,validate_run,destination


def checked_tree(root):
    root=Path(root)
    for path in (root,*root.parents):
        if not stat.S_ISDIR(path.lstat().st_mode):raise ValueError('Non-directory artifact ancestor: '+str(path))
    pending=[root]
    while pending:
        for path in pending.pop().iterdir():
            mode=path.lstat().st_mode
            if stat.S_ISDIR(mode):pending.append(path)
            elif not stat.S_ISREG(mode):raise ValueError('Non-regular artifact entry: '+str(path))


def read(path,raw=False):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):raise ValueError('Non-regular evidence leaf: '+str(path))
        value=stream.read()
    return value if raw else json.loads(value)


def sha(path):return hashlib.sha256(read(path,True)).hexdigest()


def visible_error(value):
    """Parse the producer's exception representation, independent of quote style."""
    for _ in range(3):
        try:node=ast.parse(value,mode='eval').body
        except (SyntaxError,TypeError,ValueError):return None
        if isinstance(node,ast.Constant) and isinstance(node.value,str):value=node.value;continue
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='ValueError' and
            len(node.args)==1 and not node.keywords and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str) and
            node.args[0].value.startswith('Visible target distance margin is <=5 mm: ')):
            return node.args[0].value
        return None
    return None


def manifest(root):
    checked_tree(root)
    return {str(p.relative_to(root)):sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def retirement(run,stage,terminal=False):
    """Terminal retirement certifies cleanup only, never collector success."""
    if not terminal:return collection_status.require_finalized(run,stage['pid'],stage['started_wall_ns'])
    cert=read(run/'collector-finalization.json');summary=read(run/'summary.json')
    if (cert.get('version')!=1 or cert.get('completed') is not True or cert.get('passed') is not False or
        cert.get('cleanup_errors')!=[] or not cert.get('terminal_state') or
        not re.fullmatch('[0-9a-f]{32}',str(cert.get('invocation'))) or
        cert.get('producer_pid')!=stage['pid'] or cert.get('started_wall_ns',0)<stage['started_wall_ns'] or
        cert.get('completed_wall_ns',0)<cert.get('started_wall_ns',1) or
        summary.get('producer_invocation')!=cert['invocation'] or sha(run/'summary.json')!=cert.get('summary_sha256')):
        raise ValueError('Terminal producer cleanup is unverified')
    return cert


def export_check(run,group,name,stage):
    report=run/name;v=read(report/'validation.json')
    if (v.get('passed') is not True or v.get('loaded') is not True or v.get('timing_passed') is not True or
        v.get('fps')!=30 or v.get('episodes')!=3 or v.get('frames',0)<=0 or
        v.get('completed_wall_ns',0)<stage['started_wall_ns'] or
        v.get('export_content',{}).get('video',{}).get('decoded_frame_pairs')!=2*v['frames']):
        raise ValueError('Fresh all-frame export/LeRobot certificate is missing: '+str(report))
    verify_pose_evidence(v['counterfactual_triplet'],group,'Export group differs from raw group')
    env=read(report/'export-environment.json')
    if env.get('passed') is not True or env.get('dependencies_unchanged') is not True:
        raise ValueError('Export dependency audit failed')
    return v


def classify(run,job,stage,media=None,export_stages=None):
    """Recompute the disposition; unknown/terminal pipeline failures raise."""
    run=Path(run);checked_tree(run);destination(run,spec(job['layout_seed'],job['split']))
    config=read(run/'run.json')
    if config.get('counterfactual_triplet')!=spec(job['layout_seed'],job['split']) or config.get('normal_only') is not True:
        raise ValueError('Run seed/split/mode differs from its planned job')
    if config.get('instance')!=job['instance']:raise ValueError('Group belongs to another assigned slot')
    paths=sorted(run.glob('attempt-*/receipt.json'))
    for index,path in enumerate(paths):
        if path.parent.name!=f'attempt-{index:04d}':raise ValueError('Non-bijective attempt ledger')
        r=read(path);expected=hashlib.sha256((str(run)+':'+str(index)+':'+str(job['layout_seed'])).encode()).hexdigest()
        if r.get('episode_id')!=expected:raise ValueError('Attempt identity differs from its path/seed/trial')
        params=member(config['counterfactual_triplet'],index)
        common=lambda p:{k:v for k,v in p.items() if k not in ('object','target_selection')}
        if common(r['params'])!=common(params) or r['params']['object'] not in (params['color']+'_2x3_0',params['color']+'_2x3_1'):
            raise ValueError('Attempt parameters differ from the deterministic triplet')
    summary=read(run/'summary.json')
    if summary.get('complete') is True:
        retirement(run,stage);group=validate_run(run)
        if media is None:raise ValueError('Production media must be recomputed from original recorder videos')
        from triplet_media import inspect_run
        measured=inspect_run(run)
        for key in ('passed','reasons','policy','physical_hashes','first_state_bits_equal','validated_group_sha256','raw_files_sha256','layout_seed','split','triplet_id'):
            if media.get(key)!=measured[key]:raise ValueError('Stored production-media source/verdict differs: '+key)
        media=measured
        if media.get('validated_group_sha256')!=sha(run/'counterfactual-triplet.json'):
            raise ValueError('Media is bound to a different native group')
        for name,value in media['raw_files_sha256'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts or sha(run/name)!=value:
                raise ValueError('Media raw source changed: '+name)
        if media.get('passed') is not True:
            reasons=media.get('reasons')
            if not reasons or not set(reasons)<={'exact_identity','image_psnr','departure_labels'}:raise ValueError('Unknown media refusal')
            return 'refused','exact_identity' if 'exact_identity' in reasons else reasons[0],group
        if set(export_stages or {})!={'report','report-validation'}:raise ValueError('Both export entrances must pass')
        for name,event in export_stages.items():export_check(run,group,name,event)
        return 'selected','accepted',group
    receipts=[read(p) for p in paths]
    if not receipts or len(receipts)>3 or any(r.get('accepted') is not True for r in receipts[:-1]):
        raise ValueError('Incomplete/orphan triplet failure ledger')
    failed=receipts[-1];index=len(receipts)-1
    if failed.get('accepted') is not False:raise ValueError('Incomplete group without a failed member')
    if failed.get('camera_boundary',{}).get('passed') is not True:
        raise ValueError('Capture/join failure is terminal')
    if failed.get('finalization_errors') or failed.get('terminal_publication_error'):
        raise ValueError('Attempt finalization failed')
    if failed.get('disposition',{}).get('kind')=='quality_reject':
        retirement(run,stage)
        for record in failed.get('capture_failures',[]):
            one=dict(failed,quality_reject=None,capture_failures=[record],original_capture_error_preserved=True)
            checked=disposition(one)
            if checked['kind']!='quality_reject' or checked['reason'] not in ('arm_nonconvergence','gripper_nonconvergence'):
                raise ValueError('A non-physical capture failure is terminal')
        for i,previous in enumerate(receipts[:-1]):
            rows=[json.loads(line) for line in read(run/f'attempt-{i:04d}/frames.jsonl',True).splitlines()]
            if not timing_audit(rows,30)['passed'] or previous.get('disposition',{}).get('kind')!='accepted':
                raise ValueError('A preceding member has an invalid timing/disposition contract')
        rows_path=run/f'attempt-{index:04d}/frames.jsonl'
        rows=[json.loads(line) for line in read(rows_path,True).splitlines()] if rows_path.exists() else None
        verdict=disposition(failed,rows=rows)
        if verdict!=failed['disposition'] or verdict['kind']!='quality_reject':raise ValueError('Physical rejection contract differs')
        if rows and not timing_audit(rows,30).get('contract_passed',False):raise ValueError('Failing episode tick contract failed')
        return 'refused','quality:'+verdict['reason'],None
    # This pre-writer admission refusal has native numerical evidence. Other
    # terminal collector faults are never eligible for a replacement.
    if index!=0 or visible_error(failed.get('error')) is None:
        raise ValueError('Terminal or unknown triplet fault: '+str(failed.get('error')))
    retirement(run,stage,terminal=True)
    if failed.get('capture_failures'):raise ValueError('Capture failure accompanied the layout refusal')
    if any((run/'attempt-0000'/name).exists() for name in ('frames.jsonl','top.mp4','wrist.mp4')):
        raise ValueError('Visible-margin refusal occurred after the writer opened')
    if failed.get('end_cancel',{}).get('held') is not True or failed['end_cancel'].get('failure') or failed.get('retirement',{}).get('passed') is not True:
        raise ValueError('Visible-margin retirement failed')
    native=read(run/'triplet-boundaries/attempt-0000.json')['phases']['canonical_restore']['restore']
    if native.get('arm_restored') is not True or not native.get('command_baseline'):raise ValueError('Canonical restore was not verified')
    physical_state(native['snapshot'])
    choices={color:visible_target(native['snapshot'],color) for color in COLORS}
    if not any(v['ambiguous'] for v in choices.values()):raise ValueError('No native visible-margin refusal')
    # Any unrelated terminal event still stops the run, even when one colour is ambiguous.
    marker=read(run/'terminal-fault.json')
    if not marker.get('events') or any(visible_error(e.get('error'))!=visible_error(failed['error']) for e in marker['events']):
        raise ValueError('Unrelated native terminal event')
    return 'refused','visible_target_margin',dict(choices=choices)


def selected_entry(job,run,files):
    """RTX originals are authoritative; paths are never rewritten in this entry."""
    report=Path(run)/'report-validation/validation.json'
    raw={n:h for n,h in files.items() if re.fullmatch(r'attempt-000[0-2]/(receipt.json|frames.jsonl|top.mp4|wrist.mp4|first-recorded-native.json)',n)}
    if len(raw)!=15:raise ValueError('Three complete raw RGB members are required')
    return {**{k:job[k] for k in ('requested_layout_seed','layout_seed','replaces_layout_seed','replacement_chain')},
        'dataset_root':str(Path(run)/'dataset'),'raw_root':str(run),'validation_path':str(report),
        'validation_sha256':files['report-validation/validation.json'],'raw_files_sha256':raw}
