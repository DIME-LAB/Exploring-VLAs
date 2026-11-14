"""Wall watchdog around a single owned native worker, with pidfd cleanup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import ctypes
import hashlib
import shared
from storage import confined,atomic_json
from run_smoke import OwnedProcesses
from collection_status import require_finalized,finish


def heartbeat(stage=None,deadline_s=120.):
    path=confined(Path(os.environ['SOARM_INSTANCE_ROOT'])/'inproc-heartbeat.json')
    if stage is not None:
        atomic_json(path,dict(producer_pid=os.getpid(),stage=stage,deadline_s=deadline_s))
    else:os.utime(path,None)


def main():
    command=sys.argv[1:]
    if command and command[0]=='--':command=command[1:]
    if not command:raise ValueError('Worker command required')
    runtime=confined(Path(os.environ['SOARM_INSTANCE_ROOT']))
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise OSError('Worker subreaper unavailable')
    ledger=OwnedProcesses(runtime,time.time_ns())
    started_wall=time.time_ns()
    started=time.monotonic();last_seen=started;stamp=None;budget=900.;failure=None;worker=None
    try:
        worker=subprocess.Popen(command,start_new_session=True);ledger.add(worker.pid)
        while worker.poll() is None:
            ledger.scan()
            path=runtime/'inproc-heartbeat.json'
            if path.exists():
                state=json.loads(path.read_text())
                if state['producer_pid']==worker.pid:
                    newest=path.stat().st_mtime_ns
                    if newest!=stamp:stamp=newest;last_seen=time.monotonic()
                    budget=state['deadline_s']
            if time.monotonic()-last_seen>budget:
                failure=dict(reason='wall_watchdog',deadline_s=budget,worker_pid=worker.pid)
                break
            time.sleep(.2)
    except BaseException as exc:failure=dict(reason='monitor_failure',error=repr(exc))
    finally:
        # Cleanup can reap an adopted worker before Popen sees its status; ECHILD
        # then looks like exit0. Preserve only the status observed before cleanup.
        observed_returncode=worker.poll() if worker else None
        cleanup=ledger.stop()
        if failure is None and worker is not None and observed_returncode==0 and not cleanup['remaining']:
            try:
                root=confined(Path(command[command.index('--output')+1]))
                ready_path=root/'worker-exit-ready.json'
                if ready_path.exists():
                    ready=json.loads(ready_path.read_text());certificate=json.loads((root/'collector-finalization.json').read_text())
                    if (not ready['passed'] or ready['producer_pid']!=worker.pid or certificate['producer_pid']!=worker.pid or
                        certificate['started_wall_ns']<started_wall or ready['invocation']!=certificate['invocation'] or
                        ready['summary_sha256']!=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest()):
                        raise RuntimeError('Worker readiness does not match its producer or summary')
                    finish(root,certificate,[])
                require_finalized(root,producer_pid=worker.pid,not_before_ns=started_wall)
                summary=json.loads((root/'summary.json').read_text())
                if summary.get('end_marker') not in ('COLLECTION_FINISHED','PHYSICS_CAPTURE_FINISHED','BOOT_PROBE_FINISHED'):
                    raise RuntimeError('Worker did not publish its end marker')
            except BaseException as exc:failure=dict(reason='finalization_failure',error=repr(exc))
        result=dict(passed=failure is None and worker is not None and observed_returncode==0 and not cleanup['remaining'],
            failure=failure,worker_returncode=observed_returncode,cleanup=cleanup,
            wall_s=time.monotonic()-started)
        if not result['passed'] and result['failure'] is None:
            result['failure']=dict(reason='worker_exit' if worker is not None and observed_returncode!=0 else 'cleanup_failure',
                returncode=observed_returncode)
        atomic_json(runtime/'inproc-watchdog.json',result)
    if not result['passed']:raise RuntimeError('Worker or watchdog failed: '+repr(result['failure']))


if __name__=='__main__':main()
