"""Verified, replayable home baseline for unrecorded simulator resets."""
import copy
from comparisons import matches
from contract import JOINTS
from success import initial_arm
from tick_contract import digest


def validate_baseline(baseline,home_target):
    if not baseline or baseline.get('version')!=1:raise RuntimeError('No verified home baseline is available for retry')
    samples=baseline['samples'];snapshot=baseline['snapshot']
    target=[home_target[name] for name in JOINTS[:5]]
    if not matches(target,baseline['home_target']):raise RuntimeError('Home calibration differs from the verified baseline')
    check=initial_arm(samples,target)
    if not check['passed']:raise RuntimeError('Home baseline verification failed: '+repr(check))
    last=copy.deepcopy(samples[-1]);last.pop('tick_contract',None)
    if not matches(last,snapshot) or digest(snapshot)!=baseline['snapshot_sha256']:
        raise RuntimeError('Home baseline snapshot differs from its verified samples')
    row=samples[-1]['tick_contract']['row']
    if not matches(snapshot.get('drive_targets'),row.get('physics_drive_targets')):
        raise RuntimeError('Home baseline drive targets differ from its native proof')
    if len(snapshot['drive_targets'])!=6:raise RuntimeError('Home reset requires all six verified drive targets')
    return check


def make_baseline(home,episode_id):
    samples=copy.deepcopy(home['snapshots']);snapshot=copy.deepcopy(samples[-1]);snapshot.pop('tick_contract',None)
    baseline=dict(version=1,snapshot=snapshot,samples=samples,home_target=[home['target'][name] for name in JOINTS[:5]],
                  snapshot_sha256=digest(snapshot),source_episode_id=episode_id)
    validate_baseline(baseline,home['target'])
    return baseline
