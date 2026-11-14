#!/usr/bin/env python3
"""Exercise instance claims and pidfd cleanup on processes this probe spawns."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from storage import confined,atomic_json
from run_smoke import InstanceClaim,OwnedProcesses,identity


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    root=confined(a.output);root.mkdir(parents=True,exist_ok=False)
    claim=InstanceClaim(root);ledger=OwnedProcesses(root,time.time_ns());children=[];result=dict(passed=False)
    try:
        code='from pathlib import Path;from run_smoke import InstanceClaim;InstanceClaim(Path('+repr(str(root))+'))'
        blocked=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
        assert blocked.returncode!=0 and 'already claimed' in blocked.stderr
        env=dict(os.environ,SOARM_INSTANCE_CLAIM_FD=str(claim.fd))
        helper=Path(__file__).with_name('instance_claim.sh')
        inherited=subprocess.run(['bash','-c','source "$1"; claim_collection_instance "$2"; echo claim-inherited','claim',str(helper),str(root)],
                                 env=env,pass_fds=(claim.fd,),capture_output=True,text=True)
        assert inherited.returncode==0 and 'claim-inherited' in inherited.stdout
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
        children.append(child);ledger.add(child.pid)
        assert ledger.known[child.pid]['start_ticks']==identity(child.pid)['start_ticks']
        cleanup=ledger.stop();child.wait(timeout=2)
        assert not cleanup['remaining'] and identity(child.pid) is None
        result.update(passed=True,overlapping_claim_refused=True,inherited_claim_validated=True,
                      pidfd_cleanup=cleanup,child_identity_absent=True)
    finally:
        if ledger.handles:ledger.stop()
        for child in children:
            if child.poll() is None:raise RuntimeError('Owned process probe did not retire')
        claim.close();atomic_json(root/'validation.json',result)
    print(json.dumps(dict(passed=result['passed'],overlapping_claim_refused=True,pidfd_child_retired=True)))


if __name__=='__main__':main()
