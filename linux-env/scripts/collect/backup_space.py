"""Destination admission for copies into our own dual backup projection."""
import fcntl,inspect,os,stat
from contextlib import contextmanager
from pathlib import Path

BASE=Path('/mnt/a4500-backup/exploring-vlas')
PROJECTION=BASE/'datasets/counterfactual-production-v1'


def require_capacity(free_bytes,incoming_bytes,file_count):
    if any(type(v) is not int or v<0 for v in (free_bytes,incoming_bytes,file_count)):
        raise ValueError('Invalid backup capacity measurement')
    floor=100*1024**3
    # Full source size covers partial/overwritten destinations conservatively;
    # allow extra space for directory entries, blocks and rsync temporary files.
    overhead=64*1024**2+file_count*16*1024
    required=floor+incoming_bytes+overhead
    if free_bytes<required:
        raise RuntimeError('Dual /mnt/a4500-backup would cross 100 GiB floor: free_GiB='+format(free_bytes/1024**3,'.3f')+', free_bytes='+str(free_bytes)+
            ', incoming='+str(incoming_bytes)+', overhead='+str(overhead)+', required='+str(required))
    return dict(passed=True,free_bytes=free_bytes,incoming_bytes=incoming_bytes,file_count=file_count,
        floor_bytes=floor,overhead_bytes=overhead,required_free_bytes=required,
        projected_free_bytes=free_bytes-incoming_bytes-overhead)


def probe_code(destination,incoming_bytes,file_count):
    """Send the same tested integer admission function to the destination host."""
    return 'import json,os,stat\nfrom pathlib import Path\n'+inspect.getsource(require_capacity)+f'''
base=Path({str(BASE)!r});projection=Path({str(PROJECTION)!r});destination=Path({str(destination)!r})
if '..' in destination.parts or not destination.is_relative_to(projection):
 raise ValueError('Backup destination is outside our projection')
# Inspect root-to-leaf, refusing a link before examining anything below it.
# The estate archive and any symlink to it are never traversed.
for path in reversed((destination,*destination.parents)):
 try:mode=path.lstat().st_mode
 except FileNotFoundError:continue
 if not stat.S_ISDIR(mode):raise ValueError('Backup ancestor is not a real directory: '+str(path))
space=os.statvfs(base)
result=require_capacity(space.f_bavail*space.f_frsize,{incoming_bytes!r},{file_count!r})
result.update(destination=str(destination),filesystem_root=str(base),checked_wall_ns=__import__('time').time_ns())
print(json.dumps(result))
'''


@contextmanager
def serialized_copy(path):
    """Serialize our four copies through admission, transfer and verification."""
    path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK,0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):raise RuntimeError('Backup lock is not regular')
        fcntl.flock(fd,fcntl.LOCK_EX)
        yield
    finally:os.close(fd)
