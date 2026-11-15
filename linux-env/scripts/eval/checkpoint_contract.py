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

def require_handoff(directory,receipt_path,weights_sha256):
    receipt_path=Path(receipt_path);receipt=json.loads(receipt_path.read_text())
    if receipt['source_before']['files']!=receipt['source_after']['files'] or receipt['verified']['verified'] is not True:raise ValueError('Handoff lacks stable verified source files')
    if Path(directory).resolve()!=Path(receipt['source_before']['model']).resolve():raise ValueError('Evaluation source path differs from handoff')
    actual=checkpoint_files(directory)
    expected={name:{key:row[key] for key in ('size','sha256')} for name,row in receipt['source_before']['files'].items()}
    if actual!=expected:raise ValueError('Checkpoint files differ from verified handoff')
    if actual['model.safetensors']['sha256']!=weights_sha256:raise ValueError('Checkpoint weights differ')
    return dict(receipt=str(receipt_path),receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),files=actual)
