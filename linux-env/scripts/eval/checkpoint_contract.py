"""Read-only full-checkpoint fingerprints for native evaluation."""
import hashlib,json
from pathlib import Path

def checkpoint_files(directory):
    directory=Path(directory)
    if directory.is_symlink():raise ValueError('Checkpoint directory must not be a symlink')
    result={}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():raise ValueError('Checkpoint contains a symlink')
        if not path.is_file():continue
        before=path.stat();digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Checkpoint changed while hashing')
        result[str(path.relative_to(directory))]=dict(size=before.st_size,sha256=digest.hexdigest())
    return result

def require_handoff(directory,receipt_path,weights_sha256,*,published=False):
    receipt_path=Path(receipt_path);receipt=json.loads(receipt_path.read_text())
    if receipt['source_before']['files']!=receipt['source_after']['files'] or receipt['verified']['verified'] is not True:raise ValueError('Handoff lacks stable verified source files')
    if published:
        if not (receipt['source_before']['ready'] and receipt['source_after']['ready'] and receipt['stable_seconds']>=60):raise ValueError('Published handoff lacks source completion/stability')
        expected_path=receipt['published_model']
    else:expected_path=receipt['source_before']['model']
    if Path(directory).resolve()!=Path(expected_path).resolve():raise ValueError('Evaluation source path differs from handoff')
    actual=checkpoint_files(directory)
    expected={name:{key:row[key] for key in ('size','sha256')} for name,row in receipt['source_before']['files'].items()}
    if published and {name:{key:row[key] for key in ('size','sha256')} for name,row in receipt['verified']['files'].items()}!=expected:raise ValueError('Published verification differs from stable source')
    if actual!=expected:raise ValueError('Checkpoint files differ from verified handoff')
    if actual['model.safetensors']['sha256']!=weights_sha256:raise ValueError('Checkpoint weights differ')
    return dict(receipt=str(receipt_path),receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),files=actual)

def require_processor_proof(directory,proof_path,files):
    directory=Path(directory);proof_path=Path(proof_path);proof=json.loads(proof_path.read_text())
    if not (proof['passed'] is True and proof['cpu_only'] is True and proof['training_written_processor_reloaded'] is True and proof['fixture_only'] is False):raise ValueError('Actual saved training processor proof required')
    if Path(proof['checkpoint']).resolve()!=directory.resolve() or Path(proof['completed_last_marker']).resolve()!=directory.parent.resolve():raise ValueError('Processor proof belongs to another checkpoint')
    if proof['module_sha256']!=files['workspace_crop_step.py']['sha256']:raise ValueError('Processor module differs from proof')
    if proof['raw_policy_input_keys']!=['observation.state','observation.images.top','observation.images.wrist']:raise ValueError('Processor input keys differ')
    for name,size in proof['stable_file_sizes'].items():
        if files[name]['size']!=size:raise ValueError('Processor proof file size differs')
    required={'observation.images.top','observation.images.wrist','observation.language.tokens','observation.state','top_after_native_resize'}
    hashes=proof['pixel_state_token_and_resize_hashes']
    if set(hashes)!=required or any(len(value)!=64 for value in hashes.values()):raise ValueError('Incomplete processor parity hashes')
    return dict(path=str(proof_path),sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),proof=proof)
