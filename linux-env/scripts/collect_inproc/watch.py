"""Wall watchdog around a single owned native worker, with pidfd cleanup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shared
from storage import confined,atomic_json
from run_smoke import OwnedProcesses
from collection_status import require_finalized


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
        if failure is None and worker.poll()==0:
            try:
                root=confined(Path(command[command.index('--output')+1]))
                require_finalized(root,producer_pid=worker.pid,not_before_ns=started_wall)
                summary=json.loads((root/'summary.json').read_text())
                if summary.get('end_marker') not in ('COLLECTION_FINISHED','PHYSICS_CAPTURE_FINISHED'):
                    raise RuntimeError('Worker did not publish its end marker')
            except BaseException as exc:failure=dict(reason='finalization_failure',error=repr(exc))
    except BaseException as exc:failure=dict(reason='monitor_failure',error=repr(exc))
    finally:
        cleanup=ledger.stop()
        result=dict(passed=failure is None and worker is not None and worker.poll()==0 and not cleanup['remaining'],
            failure=failure,worker_returncode=worker.poll() if worker else None,cleanup=cleanup,
            wall_s=time.monotonic()-started)
        atomic_json(runtime/'inproc-watchdog.json',result)
    if not result['passed']:raise RuntimeError('Worker or watchdog failed: '+repr(result['failure']))


if __name__=='__main__':main()
