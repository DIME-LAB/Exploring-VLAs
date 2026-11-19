#!/usr/bin/env python3
"""CPU-only, read-only-on-source checkpoint handoff; never starts inference."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

SOURCE_ROOT = '/mnt/storage3/aaugus11'
TARGET_ROOT = '/mnt/a4500-backup/exploring-vlas/eval'

REQUIRED = ('config.json', 'model.safetensors', 'train_config.json',
            'policy_preprocessor.json', 'policy_postprocessor.json')


def inventory(directory, with_hashes=False):
    """Reject symlinks and record a stable regular-file tree without loading tensors."""
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError('Model directory must not be a symlink')
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Unexpected model symlink: '+str(path))
        if not path.is_file():
            continue
        before = path.stat()
        row = dict(size=before.st_size, mtime_ns=before.st_mtime_ns)
        if with_hashes:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4*1024*1024), b''):
                    digest.update(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('Source changed while hashing: '+str(path))
            row['sha256'] = digest.hexdigest()
        result[str(path.relative_to(directory))] = row
    return result


def source_probe(request):
    root = Path(request['source_checkpoints'])
    if not root.is_relative_to(SOURCE_ROOT):
        raise ValueError('Source must stay under our RTX storage root')
    step = int(request['step'])
    folder = root/f'{step:06d}'
    model = folder/'pretrained_model'
    if not model.is_dir():
        return dict(ready=False, reason='checkpoint-not-created', step=step)
    if folder.is_symlink() or root.resolve() != root.absolute():
        raise ValueError('Unexpected source checkpoint/root symlink')
    last = root/'last'
    if not last.is_symlink():
        return dict(ready=False, reason='trainer-last-marker-absent', step=step)
    try:
        target = os.readlink(last)
    except FileNotFoundError:
        return dict(ready=False,reason='trainer-last-marker-changing',step=step)
    if re.fullmatch(r'\d{6}', target) is None or int(target) < step:
        return dict(ready=False, reason='trainer-save-not-acknowledged', last=target, step=step)
    if not (root/target).is_dir():
        return dict(ready=False, reason='trainer-last-target-missing', step=step)
    training_step = folder/'training_state/training_step.json'
    if training_step.exists():
        try:
            saved_step=json.loads(training_step.read_text())['step']
        except FileNotFoundError:
            return dict(ready=False,reason='optimizer-retention-marker-changing',step=step)
        if saved_step != step:
            raise ValueError('Trainer step marker differs')
    elif int(target) == step:
        return dict(ready=False, reason='latest-training-step-marker-absent', step=step)
    files = inventory(model, request.get('hashes', False))
    if any(name not in files or not files[name]['size'] for name in REQUIRED):
        return dict(ready=False, reason='incomplete-model-file-set', files=files, step=step)
    marker = request.get('completion_marker')
    marker_proof = None
    if marker:
        if Path(marker).name != marker or marker in ('.','..'):
            raise ValueError('Completion marker must be a plain file name')
        if marker not in files or not files[marker]['size']:
            return dict(ready=False, reason='custom-completion-marker-absent', step=step)
        raw = (model/marker).read_bytes()
        value = json.loads(raw)
        if not isinstance(value,dict):raise ValueError('Completion marker must contain a JSON object')
        marker_proof = dict(name=marker,sha256=hashlib.sha256(raw).hexdigest())
    # Processor state files are referenced by their JSON, not a guessed suffix.
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        def check_refs(value):
            if isinstance(value, dict):
                for child in value.values(): check_refs(child)
            elif isinstance(value, list):
                for child in value: check_refs(child)
            elif isinstance(value, str) and value.endswith('.safetensors'):
                if value not in files or not files[value]['size']:
                    raise ValueError('Missing processor state: '+value)
        check_refs(json.loads((model/name).read_text()))
    documents={}
    if request.get('processor_documents'):
        if not marker:raise ValueError('Processor document transfer gate needs a family marker')
        documents={name:json.loads((model/name).read_text()) for name in ('policy_preprocessor.json','policy_postprocessor.json',marker)}
    return dict(ready=True, step=step, model=str(model), files=files,processor_documents=documents,
                completion=dict(marker=str(last), target=target,
                    meaning='native trainer updates last only after save_checkpoint returns',
                    training_step_present=training_step.exists(),artifact_marker=marker_proof))


def prepare_target(request):
    import shutil
    root = Path(request['destination']).resolve()
    if not root.is_relative_to(TARGET_ROOT):
        raise ValueError('Destination must stay on eval backup storage')
    root.mkdir(parents=True, exist_ok=True)
    final = root/f"{request['step']:06d}"
    if final.exists():
        receipt = final/'handoff.json'
        if not receipt.is_file():
            raise FileExistsError('Existing target has no handoff receipt; will not overwrite')
        return dict(existing=True, directory=str(final),receipt=json.loads(receipt.read_text()))
    if shutil.disk_usage(root).free < request['bytes']+100*1024**3:
        raise RuntimeError('Insufficient backup space for verified checkpoint staging')
    pending = root/f".{request['step']:06d}.incoming-{request['transfer_id']}"
    pending.mkdir()
    (pending/'pretrained_model').mkdir()
    return dict(existing=False, directory=str(pending))


def verify_target(request):
    pending = Path(request['directory'])
    if not pending.resolve().is_relative_to(TARGET_ROOT):
        raise ValueError('Target outside backup eval root')
    actual = inventory(pending/'pretrained_model', True)
    expected = request['files']
    if set(actual) != set(expected):
        raise ValueError('Transferred file set differs from source')
    for name, row in actual.items():
        if (row['size'], row['sha256']) != (expected[name]['size'], expected[name]['sha256']):
            raise ValueError('Transferred content differs: '+name)
    sys.path.insert(0, str(Path.home()/'Downloads/Exploring-VLAs/linux-env/scripts/eval'))
    from phase3 import validate_checkpoint
    return dict(verified=True, files=actual, contract=validate_checkpoint(pending/'pretrained_model'))


def publish_target(request):
    pending = Path(request['directory'])
    final = Path(request['destination'])/f"{request['step']:06d}"
    if final.exists():
        raise FileExistsError('Target appeared during transfer; refusing overwrite')
    if pending.resolve().parent != final.parent.resolve() or not final.parent.resolve().is_relative_to(TARGET_ROOT):
        raise ValueError('Publish must stay in the prepared backup directory')
    receipt = pending/'handoff.json'
    receipt.write_text(json.dumps(request['receipt'], indent=2)+'\n')
    with receipt.open('rb') as stream: os.fsync(stream.fileno())
    os.rename(pending, final)
    return dict(published=str(final), model=str(final/'pretrained_model'))


def publish_processor_proof(request):
    """Publish admitted proof bytes atomically before exposing their handoff."""
    import shutil
    root=Path(request['root'])
    if not root.is_absolute() or root.resolve()!=root.absolute() or not root.is_relative_to(TARGET_ROOT):
        raise ValueError('Processor proof root must be confined without symlinks')
    raw=request['text'].encode('utf-8')
    if not 0<int(request['step'])<1000000 or re.fullmatch('[0-9a-f]{32}',request['transfer_id']) is None:
        raise ValueError('Invalid proof step/transfer identity')
    if hashlib.sha256(raw).hexdigest()!=request['sha256']:
        raise ValueError('Admitted processor proof bytes changed')
    json.loads(raw)
    root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free < len(raw)+100*1024**3:
        raise RuntimeError('Processor proof backup free-space floor')
    folder=root/f"{int(request['step']):06d}"
    folder.mkdir(exist_ok=True)
    if folder.resolve()!=folder.absolute():raise ValueError('Processor proof folder symlink')
    final=folder/'processor-proof.json'
    if final.is_symlink():raise ValueError('Processor proof file symlink')
    if final.exists():
        if final.read_bytes()!=raw:raise FileExistsError('Immutable processor proof differs')
    else:
        temporary=folder/('.processor-proof-'+request['transfer_id']+'.tmp')
        descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:
            with os.fdopen(descriptor,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
            os.link(temporary,final)
            descriptor=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        finally:temporary.unlink(missing_ok=True)
    return dict(path=str(final),sha256=request['sha256'],published=True)


REMOTE_FUNCTIONS = (inventory, source_probe, prepare_target, verify_target, publish_target, publish_processor_proof)


def remote(host, function, request):
    header = ('import hashlib,json,os,re,sys\nfrom pathlib import Path\n'
              f'REQUIRED={REQUIRED!r}\nSOURCE_ROOT={SOURCE_ROOT!r}\nTARGET_ROOT={TARGET_ROOT!r}\n')
    program = header+'\n'.join(inspect.getsource(f) for f in REMOTE_FUNCTIONS)
    program += f'\nprint(json.dumps({function}({request!r})))\n'
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,
        'env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 nice -n 10 python3 -'],
        input=program, text=True, capture_output=True, timeout=1800)
    if result.returncode:
        raise RuntimeError(f'{host}/{function}: {result.stderr[-3000:]}')
    return json.loads(result.stdout)


def same_files(left, right):
    return left == right


def transfer(args, snapshot):
    if args.completion_marker=='triplet360-manifest.json' and (not getattr(args,'processor_proof_directory',None) or getattr(args,'processor_family',None)!='triplet360' or not getattr(args,'processor_reference',None)):
        raise ValueError('Triplet360 requires the pre-transfer processor gate')
    step = snapshot['step']
    expected = remote(args.source_host, 'source_probe', dict(
        source_checkpoints=args.source_checkpoints, step=step, hashes=True, completion_marker=args.completion_marker,processor_documents=bool(getattr(args,'processor_proof_directory',None))))
    if not expected['ready'] or any(
        {k:v for k,v in expected['files'][name].items() if k!='sha256'} != row
        for name,row in snapshot['files'].items()) or set(expected['files'])!=set(snapshot['files']):
        raise RuntimeError('Checkpoint changed after its stability gate')
    processor_admission=None
    if getattr(args,'processor_proof_directory',None):
        from checkpoint_contract import require_transfer_processor_proof
        processor_admission=require_transfer_processor_proof(expected,Path(args.processor_proof_directory)/f'saved-crop-{step:06d}-proof.json',args.processor_reference,args.processor_family)
    proof_root=getattr(args,'published_processor_proof_root',None)
    proof_publication=None
    if proof_root:
        if processor_admission is None:raise ValueError('Proof publication requires actual pre-transfer admission')
        proof_publication=remote(args.target_host,'publish_processor_proof',dict(root=str(proof_root),step=step,
            text=Path(processor_admission['path']).read_bytes().decode('utf-8'),sha256=processor_admission['sha256'],transfer_id=uuid.uuid4().hex))
    prepared = remote(args.target_host, 'prepare_target', dict(destination=args.destination,
        step=step, bytes=sum(r['size'] for r in expected['files'].values()), transfer_id=uuid.uuid4().hex))
    if prepared['existing']:
        prior=prepared['receipt']
        if args.completion_marker=='triplet360-manifest.json' and (prior.get('processor_admission') or {}).get('sha256')!=(processor_admission or {}).get('sha256'):
            raise ValueError('Existing immutable triplet target lacks matching pre-transfer admission')
        if prior['step']!=step or prior['source']!=expected['model'] or prior['source_before']['files']!=expected['files']:
            raise ValueError('Existing immutable checkpoint receipt differs')
    if not prepared['existing']:
        command = ['rsync','-a','--protect-args','--',
            f"{args.source_host}:{expected['model']}/", prepared['directory']+'/pretrained_model/']
        # rsync runs on the target; the source is read-only over SSH.
        import shlex
        subprocess.run(['ssh','-o','BatchMode=yes',args.target_host,
            ' '.join(shlex.quote(s) for s in command)],check=True,timeout=1800,
            stdout=subprocess.DEVNULL)
    verified = remote(args.target_host, 'verify_target', dict(
        directory=prepared['directory'], files=expected['files']))
    after = remote(args.source_host, 'source_probe', dict(
        source_checkpoints=args.source_checkpoints, step=step, hashes=True, completion_marker=args.completion_marker))
    if not after['ready'] or after['files']!=expected['files']:
        raise RuntimeError('Source changed during transfer; target remains unpublished')
    receipt = dict(step=step, published_model=str(Path(args.destination)/f'{step:06d}'/'pretrained_model'), source_host=args.source_host, source=expected['model'],
        source_before=expected, source_after=after, verified=verified,processor_admission=processor_admission,
        stable_seconds=args.stable_seconds, published_utc=time.strftime('%FT%TZ',time.gmtime()))
    if proof_publication:receipt['processor_proof_publication']=proof_publication
    if prepared['existing']:
        return dict(model=prepared['directory']+'/pretrained_model', receipt=receipt, already_staged=True)
    published = remote(args.target_host,'publish_target',dict(directory=prepared['directory'],
        destination=args.destination,step=step,receipt=receipt))
    return dict(published,receipt=receipt,already_staged=False)


def write_status(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temporary,path)


def notify_entry(args,state,key,state_path,message):
    """One attempt per checkpoint; a messaging outage must not stop staging."""
    entry=state['staged'][key]
    if entry.get('notified') or entry.get('notification_status'):return
    entry['notification_status']='attempted';state['notification_in_progress']=key;write_status(state_path,state)
    try:
        subprocess.run(json.loads(args.notify_command_json)+['--prompt-file',str(message)],check=True,timeout=120)
    except Exception as error:
        entry.update(notification_status='unknown' if isinstance(error,subprocess.TimeoutExpired) else 'failed',notification_error=repr(error))
        print(json.dumps(dict(event='NOTIFICATION_FAILED_NO_RETRY',step=key,error=repr(error))),flush=True)
    else:entry.update(notified=True,notification_status='delivered')
    finally:
        state.pop('notification_in_progress',None);write_status(state_path,state)


def preserve_interrupted_notification(state):
    key=state.get('notification_in_progress')
    if key is not None:
        entry=state['staged'][key]
        if entry.get('notification_status') in (None,'attempted'):entry['notification_status']='unknown'
        entry.setdefault('notification_error',state.get('error','Prior notification interrupted; delivery unknown; no retry'))
        state.pop('notification_in_progress',None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-host',default='rtx6000');p.add_argument('--target-host',default='dual-a4500')
    p.add_argument('--source-checkpoints',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--state-dir',type=Path,required=True)
    p.add_argument('--steps',default='10000,20000,40000,60000')
    p.add_argument('--poll-seconds',type=float,default=60);p.add_argument('--stable-seconds',type=float,default=60)
    p.add_argument('--notify-command-json',help='argv that accepts a final --prompt-file argument')
    p.add_argument('--completion-marker',help='Required JSON marker in pretrained_model, in addition to native last')
    p.add_argument('--notification-context',default='Waiting for your GPU0 GO before scored main30RTC-on.')
    p.add_argument('--processor-proof-directory',type=Path,help='Optional actual-save proofs; copying waits for saved-crop-NNNNNN-proof.json')
    p.add_argument('--published-processor-proof-root',type=Path,help='Opt-in atomic target proof publication before checkpoint handoff')
    p.add_argument('--processor-reference',type=Path)
    p.add_argument('--processor-family',choices=('rule431','triplet360'))
    p.add_argument('--once',action='store_true')
    args=p.parse_args();steps=[int(s) for s in args.steps.split(',')]
    if min(args.poll_seconds,args.stable_seconds)<=0 or any(s<=0 for s in steps):p.error('Positive polling/stability/steps required')
    args.state_dir.mkdir(parents=True,exist_ok=True)
    if args.completion_marker=='triplet360-manifest.json' and (not args.processor_proof_directory or args.processor_family!='triplet360' or not args.processor_reference):raise ValueError('Triplet360 requires the pre-transfer processor gate')
    lock=(args.state_dir/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=args.state_dir/'status.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(staged={},observed={})
    preserve_interrupted_notification(state)
    configuration=dict(source_host=args.source_host,target_host=args.target_host,source_checkpoints=args.source_checkpoints,destination=args.destination,steps=steps,stable_seconds=args.stable_seconds)
    if args.completion_marker:configuration['completion_marker']=args.completion_marker
    if args.processor_proof_directory:
        if not args.processor_reference or not args.processor_family:raise ValueError('Processor transfer gate requires family and sealed reference')
        configuration.update(processor_proof_directory=str(args.processor_proof_directory.resolve()),processor_reference=str(args.processor_reference.resolve()),processor_family=args.processor_family)
    if args.published_processor_proof_root:configuration['published_processor_proof_root']=str(args.published_processor_proof_root)
    if state.get('configuration',configuration)!=configuration:raise ValueError('Watcher state belongs to another transfer configuration')
    state.update(configuration=configuration,pid=os.getpid(),source=args.source_checkpoints,destination=args.destination,status='watching')
    print(json.dumps(dict(event='WATCHING',pid=os.getpid(),steps=steps,source=args.source_checkpoints,destination=args.destination)),flush=True)
    observed={}
    try:
        while True:
            for step in steps:
                key=f'{step:06d}'
                if key not in state['staged']:
                    snap=remote(args.source_host,'source_probe',dict(source_checkpoints=args.source_checkpoints,step=step,completion_marker=args.completion_marker))
                    previous=observed.get(key);now=time.monotonic()
                    if snap['ready']:
                        if previous is None or not same_files(previous[0],snap['files']):
                            observed[key]=(snap['files'],now)
                        elif now-previous[1]>=args.stable_seconds:
                            if args.processor_proof_directory and not (args.processor_proof_directory/f'saved-crop-{step:06d}-proof.json').is_file():
                                state['observed'][key]=dict(ready=False,reason='actual-processor-proof-pending',checked_utc=time.strftime('%FT%TZ',time.gmtime()))
                                write_status(state_path,state)
                                continue
                            result=transfer(args,snap)
                            state['staged'][key]=dict(model=result['model'],receipt_file=f'{key}.json',notified=False)
                            write_status(args.state_dir/f'{key}.json',result['receipt'])
                            print(json.dumps(dict(event='STAGED',step=key,model=result['model'],weights_sha256=result['receipt']['source_before']['files']['model.safetensors']['sha256'])),flush=True)
                    else:observed.pop(key,None)
                    state['observed'][key]=dict(ready=snap['ready'],reason=snap.get('reason'),checked_utc=time.strftime('%FT%TZ',time.gmtime()))
                    write_status(state_path,state)
                entry=state['staged'].get(key)
                if entry and not entry['notified'] and args.notify_command_json:
                    receipt=json.loads((args.state_dir/entry['receipt_file']).read_text())
                    digest=receipt['source_before']['files']['model.safetensors']['sha256']
                    message=args.state_dir/f'notify-{key}.txt'
                    message.write_text(f'Checkpoint {key} STAGED and SHA256-VERIFIED CPU-only at dual-a4500:{entry["model"]}. Weights SHA256 {digest}. Native trainer last marker + >={args.stable_seconds:g}s stable sizes/mtimes + source before/after SHA and target hashes checked. Complete processors/config included; optimizer stays on RTX. Receipt: {args.destination}/{key}/handoff.json. No GPU job started. {args.notification_context}\n')
                    notify_entry(args,state,key,state_path,message)
            if all(f'{step:06d}' in state['staged'] for step in steps):
                state['status']='complete';write_status(state_path,state);return
            if args.once:return
            time.sleep(args.poll_seconds)
    except Exception as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}')
        write_status(state_path,state)
        if args.notify_command_json and not state.get('notification_in_progress'):
            failure=args.state_dir/'notify-failure.txt'
            failure.write_text(f'Checkpoint handoff watcher FAILED (no GPU started, training job untouched): {state["error"]}. Inspect {state_path}; unpublished incoming directories are preserved.\n')
            subprocess.run(json.loads(args.notify_command_json)+['--prompt-file',str(failure)],check=False,timeout=120)
        raise


if __name__=='__main__':main()
