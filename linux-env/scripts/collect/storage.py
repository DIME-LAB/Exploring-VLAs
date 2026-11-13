"""Storage gates and allocated-file growth for this instance's writable root files."""
import json
import os
from pathlib import Path
import shutil
import stat
import time


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
