#!/usr/bin/env python3
"""Exercise real launch/stop helpers with only short Python process trees."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from managed_process import start,stop,verified,record_path
from run_smoke import InstanceClaim,identity
from storage import confined,atomic_json


def wait_for(condition,seconds=10):
    end=time.monotonic()+seconds
    while not condition():
        if time.monotonic()>end:raise TimeoutError('Process regression condition did not complete')
        time.sleep(.03)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=confined(a.output);root.mkdir(parents=True,exist_ok=False);os.environ['SOARM_INSTANCE_ROOT']=str(root);os.environ.pop('SOARM_INSTANCE_CLAIM_FD',None);report=dict(passed=False)
    scripts=Path(__file__).resolve().parents[1]
    active=[]
    try:
        for name,orphan in [('tree',False),('orphan',True)]:
            pidfile=root/(name+'.pid');ready=root/(name+'-child.json')
            code='import subprocess,sys,time,json;from pathlib import Path;p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],start_new_session=True);Path('+repr(str(ready))+').write_text(json.dumps({"pid":p.pid}));time.sleep('+('0.2' if orphan else '60')+')'
            pid=start(pidfile,root/(name+'.log'),[sys.executable,'-c',code]);active.append(pidfile)
            wait_for(ready.exists);child=json.loads(ready.read_text())['pid']
            if orphan:
                wait_for(lambda:json.loads(record_path(pidfile).read_text())['state']=='stopped')
                cleanup=json.loads(record_path(pidfile).read_text())['cleanup']
            else:
                owner=InstanceClaim(root)
                try:
                    try:stop(pidfile)
                    except RuntimeError as exc:assert 'already claimed' in str(exc)
                    else:raise AssertionError('Direct stop ignored active owner')
                finally:owner.close()
                cleanup=stop(pidfile)['cleanup']
            assert not cleanup['remaining'] and identity(child) is None
            assert child in [r['pid'] for r in cleanup['tracked']]
            report[name]=dict(supervisor_pid=pid,child_pid=child,cleanup=cleanup,passed=True)
        # A plain PID file grants no authority, even when it names a live same-UID process.
        foreign=root/'isaacsim-soarm101.pid';foreign.write_text(str(os.getpid()))
        try:stop(foreign)
        except RuntimeError as exc:report['foreign_refusal']=str(exc)
        else:raise AssertionError('Foreign PID was accepted')
        env=dict(os.environ,SOARM_INSTANCE_ROOT=str(root),LOG_DIR=str(root/'logs'))
        env.pop('SOARM_INSTANCE_CLAIM_FD',None)
        owner=InstanceClaim(root);claim_probes=[]
        try:
            for script,argv in [('isaac/isaacsim_launch.sh',['close','--instance','0']),
                                ('isaac/isaacsim_launch.sh',['restart','--instance','0']),
                                ('collect_sim.sh',['down','0']),('deploy_sim.sh',['down'])]:
                command=['bash',str(scripts/script),*argv]
                probe=subprocess.run(command,env=env,capture_output=True,text=True,timeout=15)
                assert probe.returncode and 'already claimed' in probe.stderr,(script,probe.stdout,probe.stderr)
                claim_probes.append(dict(command=command,returncode=probe.returncode,stderr=probe.stderr))
        finally:owner.close()
        # Exercise the same production entrypoints with legacy/foreign PID-only files.
        for path in (root/'control.pid',root/'logs/soarm_deploy/policy.pid',root/'logs/soarm_linux_stack.pid'):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(str(os.getpid()))
        stale_probes=[]
        for script,argv in [('isaac/isaacsim_launch.sh',['close','--instance','0']),
                            ('isaac/isaacsim_launch.sh',['restart','--instance','0']),
                            ('collect_sim.sh',['down','0']),('deploy_sim.sh',['down']),('stack_stop.sh',[])]:
            probe=subprocess.run(['bash',str(scripts/script),*argv],env=env,capture_output=True,text=True,timeout=15)
            assert probe.returncode and 'without a launcher identity receipt' in probe.stderr,(script,probe.stdout,probe.stderr)
            stale_probes.append(dict(script=script,returncode=probe.returncode,stderr=probe.stderr))
        report.update(passed=True,active_other_owner_refused=claim_probes,stale_pid_entrypoint_refusals=stale_probes)
    finally:
        for pidfile in active:
            if verified(pidfile):stop(pidfile)
        atomic_json(root/'validation.json',report)
    print(json.dumps(dict(passed=report['passed'],owned_tree=True,orphan=True,claim_refusals=len(claim_probes))))


if __name__=='__main__':main()
