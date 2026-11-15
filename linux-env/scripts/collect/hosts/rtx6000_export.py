"""Audit read-only shared dependencies around the pinned collector export."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path('/mnt/storage3/aaugus11/exploring-vlas')
PIN = ROOT/'setup/export-dependencies.json'
FORK = ROOT/'repo/Exploring-VLAs/lerobot'
CUT = 'e1e201098b2056827908e76f58bbf5733beb4d82'


def snapshot():
    packages = []
    for dist in importlib.metadata.distributions():
        files = {}
        for name in ('METADATA', 'RECORD', 'direct_url.json'):
            content = dist.read_text(name)
            if content is not None:
                files[name] = hashlib.sha256(content.encode()).hexdigest()
        packages.append(dict(name=dist.metadata['Name'], version=dist.version,
                             metadata=str(dist._path), files=files))
    return sorted(packages, key=lambda row: (row['name'], row['metadata']))


def require_fork():
    cut = subprocess.check_output(['git', '-C', str(FORK), 'rev-parse', 'HEAD'], text=True).strip()
    if cut != CUT:
        raise RuntimeError('Export writer is not the pinned collector fork: ' + cut)
    subprocess.run(['git', '-C', str(FORK), 'diff', '--exit-code', 'HEAD'], check=True,
                   stdout=subprocess.DEVNULL)
    return cut


def main():
    require_fork()
    before = snapshot()
    if sys.argv[1:] == ['--pin']:
        with PIN.open('x') as stream:
            json.dump(before, stream, indent=2)
        print('Pinned export dependency metadata:', len(before))
        return
    if before != json.loads(PIN.read_text()):
        raise RuntimeError('Shared export dependencies changed since qualification')
    target = Path(sys.argv[1]).resolve()
    expected = Path(__file__).resolve().parents[1]/'export.py'
    if target != expected:
        raise RuntimeError('Unexpected collector export entrypoint: ' + str(target))
    sys.argv = sys.argv[1:]
    report = Path(sys.argv[sys.argv.index('--report-dir')+1])
    if not report.resolve().is_relative_to(ROOT):
        raise RuntimeError('Export report escapes the host root')
    sys.path.insert(0, str(target.parent))
    error = None
    try:
        runpy.run_path(str(target), run_name='__main__')
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        after = snapshot()
        passed = error is None and before == after and require_fork() == CUT
        report.mkdir(parents=True, exist_ok=True)
        receipt = dict(passed=passed, error=error, dependencies_unchanged=before == after,
                       packages_before=before, packages_after=after,
                       pinned_fork=CUT, fork_path=str(FORK), python=sys.executable,
                       shared_dependencies='/mnt/storage3/aaugus11/smolvla-train/venv/lib/python3.12/site-packages')
        (report/'export-environment.json').write_text(json.dumps(receipt, indent=2)+'\n')
        if before != after:
            raise RuntimeError('Shared export dependencies changed during export; batch refused')


if __name__ == '__main__':
    main()
