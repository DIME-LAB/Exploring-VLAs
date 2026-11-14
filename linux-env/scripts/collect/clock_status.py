"""Clock checks count only with a receipt bound to retired ROS owners."""
import hashlib,json
from pathlib import Path
import collection_status
from storage import atomic_json,confined


def status_root(output):
    output=confined(output)
    return output.parent/(output.stem+'-status')


def begin(output):
    root=status_root(output);root.mkdir(parents=True,exist_ok=False)
    return collection_status.begin(root)


def finish(output,certificate,result,errors):
    root=status_root(output)
    result=dict(result,passed=bool(result.get('checks_passed')) and not errors,cleanup_errors=errors)
    atomic_json(output,result)
    atomic_json(root/'summary.json',dict(complete=result['passed'],producer_invocation=certificate['invocation'],
        end_marker='CLOCK_CHECK_FINISHED' if result['passed'] else 'CLOCK_CHECK_FAILED',
        result_sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest()))
    return collection_status.finish(root,certificate,errors or ([] if result['passed'] else [dict(stage='clock checks',error='Checks did not pass')]))


def require(output,producer_pid,not_before_ns,returncode):
    # Even a late signal after a certificate is a failed clock probe. Exit zero
    # alone never certifies successful checks or native-resource retirement.
    if returncode!=0:raise RuntimeError('Clock probe exited abnormally: '+str(returncode))
    root=status_root(output)
    certificate=collection_status.require_finalized(root,producer_pid,not_before_ns)
    summary=json.loads((root/'summary.json').read_text())
    if not summary.get('complete') or summary['result_sha256']!=hashlib.sha256(Path(output).read_bytes()).hexdigest():
        raise RuntimeError('Clock result is incomplete or changed after finalization')
    result=json.loads(Path(output).read_text())
    if result.get('passed') is not True or len(result.get('regimes',[]))!=2:raise RuntimeError('Clock checks did not pass')
    return certificate
