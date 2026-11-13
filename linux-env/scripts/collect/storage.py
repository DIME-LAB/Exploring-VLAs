"""Storage gates and allocated-file growth for this instance's writable root files."""
import json
import os
from pathlib import Path
import shutil
import stat
import time
import uuid

CANONICAL_ROOT = Path('/mnt/a4500-backup/exploring-vlas')


def confined(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(CANONICAL_ROOT.resolve()):
        raise ValueError(f'Output must stay beneath {CANONICAL_ROOT}: {resolved}')
    return resolved


def prepare_environment(instance):
    """Apply confinement before imports that create ROS/trace/temp/cache files."""
    import sys
    for name in ('SOARM_COLLECTION_ROOT', 'HF_LEROBOT_HOME', 'SOARM_INSTANCE_ROOT'):
        if os.environ.get(name):
            confined(os.environ[name])
    runtime = confined(os.environ.get('SOARM_INSTANCE_ROOT', CANONICAL_ROOT / f'runtime/instance-{instance}'))
    paths = dict(SOARM_INSTANCE_ROOT=runtime, SOARM_GRASP_TRACE_DIR=runtime/'grasp-traces',
        TMPDIR=runtime/'tmp', TEMP=runtime/'tmp', TMP=runtime/'tmp',
        ROS_LOG_DIR=runtime/'ros-logs', XDG_CACHE_HOME=runtime/'cache',
        HF_HOME=runtime/'cache/huggingface', OPTIX_CACHE_PATH=runtime/'cache/optix',
        HF_LEROBOT_HOME=confined(os.environ.get('HF_LEROBOT_HOME', CANONICAL_ROOT/'datasets')))
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    return runtime


def wait_for_storage(bulk, stop, emit=print, usage=shutil.disk_usage):
    started = time.monotonic()
    waiting = False
    while True:
        root_free, bulk_free = usage('/').free, usage(bulk).free
        if root_free < 20 * 1024**3:
            raise RuntimeError('Root filesystem below last-resort 20 GiB floor')
        if bulk_free >= (52 if waiting else 50) * 1024**3:
            return time.monotonic() - started if waiting else 0.
        waiting = True
        emit(json.dumps(dict(event='disk_wait', root_free_gib=root_free / 1024**3,
                             bulk_free_gib=bulk_free / 1024**3, resume_bulk_gib=52)))
        if stop.wait(60):
            raise RuntimeError('Collection interrupted during disk wait')


class RootFileGrowth:
    """Space growth, not write traffic. Shared cache growth is an upper bound.

    Inspect writable regular-file descriptors of our process trees. Preserve
    paths after close, so an episode's encoder exit cannot hide tracked growth.
    In-place cache rewrites consume no additional disk space and are not counted.
    """
    def __init__(self, runtime):
        self.root_device = os.stat('/').st_dev
        self.pids = {os.getpid()}
        for name in ('control.pid', 'isaacsim-soarm101.pid'):
            path = runtime / name
            if path.exists():
                self.pids.add(int(path.read_text()))
        self.paths = set()
        self.before = self.snapshot()

    def snapshot(self):
        pending, visited = list(self.pids), set()
        while pending:
            pid = pending.pop()
            if pid in visited:
                continue
            visited.add(pid)
            try:
                pending.extend(int(p) for p in Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
                for fd in Path(f'/proc/{pid}/fd').iterdir():
                    try:
                        info = Path(f'/proc/{pid}/fdinfo/{fd.name}').read_text()
                        flags = int(next(line.split()[1] for line in info.splitlines() if line.startswith('flags:')), 8)
                        value = fd.stat()
                        if flags & 3 and stat.S_ISREG(value.st_mode) and value.st_dev == self.root_device:
                            self.paths.add(os.readlink(fd))
                    except (OSError, StopIteration):
                        continue
            except OSError:
                continue
        result = {}
        for path in self.paths:
            try:
                value = os.stat(path)
                result[path] = dict(size=value.st_size, allocated=value.st_blocks * 512)
            except OSError:
                continue
        return result

    def finish(self):
        after = self.snapshot()
        files = []
        for path, value in after.items():
            initial = self.before.get(path, dict(size=0, allocated=0))
            files.append(dict(path=path, growth_bytes=max(0, value['size'] - initial['size']),
                allocated_growth_bytes=max(0, value['allocated'] - initial['allocated']),
                shared_cache=('OptixCache' in path or '/.cache/ov/' in path)))
        return dict(method='writable-root-file growth; shared cache upper bound; not rewrite traffic',
                    allocated_growth_bytes=sum(f['allocated_growth_bytes'] for f in files),
                    apparent_growth_bytes=sum(f['growth_bytes'] for f in files), files=files)


def control_identity(runtime, pidfile='control.pid'):
    """Identify this instance's launch process across PID reuse and reboots."""
    try:
        pid = int((runtime/pidfile).read_text())
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid, start_ticks=fields[19],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (OSError, ValueError, IndexError):
        return None


def scene_stack_ready(previous, current):
    """Only a demonstrably different control process may follow an unresolved scene request."""
    return previous is not None and current is not None and previous != current


class SceneSafetyMarker:
    def __init__(self, runtime):
        self.runtime = confined(runtime)
        self.path = self.runtime/'scene-unresolved.json'

    def _read(self):
        try:
            fd=os.open(self.path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC|os.O_NONBLOCK)
        except FileNotFoundError:
            return None
        with os.fdopen(fd,'r') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError('Marker is not a regular file')
            text=stream.read(1024*1024+1)
            if len(text)>1024*1024:raise ValueError('Marker exceeds 1 MiB')
            return text

    def _write(self, text):
        temporary=self.path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
        try:
            with os.fdopen(fd,'w') as stream:
                stream.write(text);stream.flush();os.fsync(stream.fileno())
            # Rename replaces a destination symlink itself, never its target.
            os.replace(temporary,self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def require_fresh_stack(self):
        original='unavailable (marker unreadable or malformed)'
        try:
            text=self._read()
            if text is None:return
            failure=json.loads(text)
            original=failure['error']
            identities=failure['identities'];required=failure['required_restart']
            permitted={'control':'control.pid','isaac':'isaacsim-soarm101.pid'}
            if ('control' not in required or set(identities)!=set(required) or
                any(permitted.get(name)!=pidfile for name,pidfile in required.items())):
                raise ValueError('Missing required restart identity fields')
            for name,identity in identities.items():
                if identity is not None and (not isinstance(identity,dict) or
                    set(identity)!={'pid','start_ticks','boot_id'} or
                    not isinstance(identity['pid'],int) or identity['pid']<=0 or
                    not str(identity['start_ticks']).isdigit() or not isinstance(identity['boot_id'],str) or
                    not identity['boot_id']):
                    raise ValueError('Malformed recorded identity: '+name)
            unknown=[name for name in required if identities[name] is None]
            if unknown:
                raise ValueError('identity unknown at record time: '+', '.join(unknown)+
                    '; a restart cannot clear this marker; manual scene reconciliation required')
            current={name:control_identity(self.runtime,pidfile) for name,pidfile in required.items()}
            blocked=[name for name in required if not scene_stack_ready(identities[name],current[name])]
            if blocked:
                missing=[name for name in blocked if current[name] is None]
                raise ValueError('restart required for: '+', '.join(blocked)+
                    ('; current identity missing: '+', '.join(missing) if missing else ''))
        except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
            raise RuntimeError(f'Unresolved scene marker {self.path}; original error: {original}; {exc}. '
                'See README "Collecting sim data", manual scene reconciliation.') from exc

    def record(self, error, source='MoveIt'):
        required={'control':'control.pid'}
        if source=='Kit':required['isaac']='isaacsim-soarm101.pid'
        payload = dict(error=str(error), source=source, required_restart=required,
                       identities={name:control_identity(self.runtime,pidfile) for name,pidfile in required.items()},
                       recorded_wall_ns=time.time_ns())
        self._write(json.dumps(payload,indent=2))


if __name__ == '__main__':
    import sys
    for value in sys.argv[1:]:
        confined(value)
