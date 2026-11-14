"""Monotone terminal markers and affirmative per-process collector finalization."""
import hashlib,json,os,time,uuid,re
from pathlib import Path
from storage import atomic_json,confined


def reject_terminal(root):
    root=confined(root)
    markers=[root/'terminal-fault.json',*root.glob('attempt-*/terminal-fault.json')]
    present=[str(p) for p in markers if p.exists()]
    if present:raise RuntimeError('Terminal collection marker: '+repr(present))
    if (root/'summary.json').exists():
        summary=json.loads((root/'summary.json').read_text())
        if summary.get('terminal_failure') or summary.get('stop') or summary.get('end_marker')=='COLLECTION_ABORTED':
            raise RuntimeError('Terminal collector summary')
    for path in root.glob('attempt-*/receipt.json'):
        row=json.loads(path.read_text())
        if row.get('terminal_failure') or row.get('disposition',{}).get('kind')=='terminal':
            raise RuntimeError('Terminal attempt receipt: '+str(path))


def terminal(root,event,attempt=None):
    """The root marker is published FIRST; none of these markers are ever cleared."""
    root=confined(root);root.mkdir(parents=True,exist_ok=True)
    marker=root/'terminal-fault.json'
    record=json.loads(marker.read_text()) if marker.exists() else dict(version=1,events=[])
    record['events'].append(dict(event,wall_ns=time.time_ns()))
    atomic_json(marker,record)
    if attempt is not None:
        attempt=confined(attempt)
        if attempt.parent!=root:raise RuntimeError('Terminal attempt is outside this run')
        attempt.mkdir(parents=True,exist_ok=True);atomic_json(attempt/'terminal-fault.json',record)
        receipt=attempt/'receipt.json'
        if receipt.exists():
            row=json.loads(receipt.read_text());row.update(accepted=False,terminal_failure=record,
                disposition=dict(kind='terminal',reason='late_terminal_fault'))
            atomic_json(receipt,row)
    path=root/'summary.json';summary=json.loads(path.read_text()) if path.exists() else {}
    summary.update(complete=False,terminal_failure=record,end_marker='COLLECTION_ABORTED')
    atomic_json(path,summary)
    return record


def begin(root):
    reject_terminal(root)
    predecessor=None
    if ((Path(root)/'collector-finalization.json').exists() or (Path(root)/'run.json').exists() or
            any(Path(root).glob('attempt-*'))):
        predecessor=require_finalized(root)
        archive=Path(root)/'finalizations';archive.mkdir(exist_ok=True)
        archived=dict(certificate=predecessor,summary_raw_json=(Path(root)/'summary.json').read_text())
        path=archive/(predecessor['invocation']+'.json')
        if path.exists():
            if json.loads(path.read_text())!=archived:raise RuntimeError('Predecessor archive differs from its certified state')
        else:atomic_json(path,archived)
    value=dict(version=1,invocation=uuid.uuid4().hex,producer_pid=os.getpid(),started_wall_ns=time.time_ns(),
               completed=False,passed=False)
    if predecessor:value['predecessor_invocation']=predecessor['invocation']
    summary_path=Path(root)/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update(complete=False,producer_invocation=value['invocation'],end_marker='COLLECTION_IN_PROGRESS')
    atomic_json(summary_path,summary)
    atomic_json(Path(root)/'collector-finalization.json',value)
    return value


def finish(root,certificate,errors):
    value=dict(certificate or {},completed=True,completed_wall_ns=time.time_ns(),cleanup_errors=errors,passed=False)
    try:reject_terminal(root)
    except Exception as exc:value['terminal_state']=repr(exc)
    if not certificate:value['terminal_state']='Collector invocation did not begin'
    summary_path=Path(root)/'summary.json'
    if summary_path.exists():
        raw=summary_path.read_bytes();summary=json.loads(raw)
        value['summary_sha256']=hashlib.sha256(raw).hexdigest()
        if summary.get('producer_invocation')!=value.get('invocation'):value['terminal_state']='Summary belongs to another invocation'
    else:value['terminal_state']='Summary is missing'
    value['passed']=bool(certificate) and not errors and 'terminal_state' not in value
    atomic_json(Path(root)/'collector-finalization.json',value)
    return value


def require_finalized(root,producer_pid=None,not_before_ns=None):
    reject_terminal(root)
    certificate=json.loads((Path(root)/'collector-finalization.json').read_text())
    if certificate.get('completed') is not True or certificate.get('passed') is not True or certificate.get('cleanup_errors'):
        raise RuntimeError('Collector finalization is incomplete or failed')
    required=('version','invocation','producer_pid','started_wall_ns','completed_wall_ns','cleanup_errors','summary_sha256')
    if (any(key not in certificate for key in required) or certificate['version']!=1 or
            not isinstance(certificate['invocation'],str) or not re.fullmatch('[0-9a-f]{32}',certificate['invocation']) or
            not isinstance(certificate['producer_pid'],int) or certificate['producer_pid']<=0 or
            not isinstance(certificate['started_wall_ns'],int) or not isinstance(certificate['completed_wall_ns'],int) or
            certificate['started_wall_ns']<=0 or certificate['completed_wall_ns']<certificate['started_wall_ns'] or
            certificate['cleanup_errors']!=[]):raise RuntimeError('Malformed collector finalization certificate')
    summary_path=Path(root)/'summary.json'
    summary=json.loads(summary_path.read_text())
    if summary.get('producer_invocation')!=certificate['invocation']:raise RuntimeError('Finalization invocation differs from summary')
    if hashlib.sha256(summary_path.read_bytes()).hexdigest()!=certificate.get('summary_sha256'):
        raise RuntimeError('Collector summary changed after finalization')
    if producer_pid is not None and certificate.get('producer_pid')!=producer_pid:
        raise RuntimeError('Collector finalization belongs to another producer')
    if not_before_ns is not None and certificate.get('started_wall_ns',0)<not_before_ns:
        raise RuntimeError('Collector finalization predates the producing stage')
    return certificate
