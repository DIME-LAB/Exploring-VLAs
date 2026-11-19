"""Restore a sealed triplet through the producer's own preparation sequence."""
import copy
from pathlib import Path

from triplet_bank import COLLECTOR_CUT, RESET, SUITE, document


def prepare_recorded_scene(collector,row,baseline,receipt,motion_epochs,output,path):
    """Initialize scene bookkeeping through Collector.scene's real replay branch."""
    if collector.args.replay_receipt is not None:
        raise ValueError('Recorded triplet initialization needs a fresh collector')
    source=document(row['source_receipt'],row['source_receipt_sha256'])
    collector.args.replay_receipt=Path(row['source_receipt'])
    try:
        cups=collector.scene(copy.deepcopy(source['params']),baseline)
    finally:
        collector.args.replay_receipt=None
    initialization=dict(protocol='collector-scene-recorded-replay-v1',sampler_invocations=0,
        unrecorded_scene_barrier_ticks=2,source_receipt=row['source_receipt'],source_receipt_sha256=row['source_receipt_sha256'],
        planning_scene_clear_check=copy.deepcopy(collector.scene_clear_check),
        planning_scene_pose_check=copy.deepcopy(collector.scene_pose_check),cup_provenance=copy.deepcopy(cups))
    receipt['params']=install_recorded_session(collector,row,output)
    collector._prepare_scene_attempt(receipt['params'],baseline,receipt,motion_epochs,output,path)
    receipt['eval_scene_initialization']=initialization


def install_recorded_session(collector, row, output):
    from target_triplets import TripletSession, member
    anchor = document(row['anchor_path'], row['anchor_sha256'])
    source = document(row['source_receipt'], row['source_receipt_sha256'])
    if source['params']['seed'] != row['layout_seed'] or source['params']['color'] != row['color']:
        raise ValueError('Triplet source member differs')
    if source['counterfactual']['triplet_id'] != anchor['triplet_id'] or row['triplet_id'] != anchor['triplet_id']:
        raise ValueError('Triplet source anchor differs')
    if source['params'] != member(anchor['config'], row['source_trial'], anchor['anchor']):
        raise ValueError('Triplet target/parameters differ')
    session = TripletSession(Path(output), anchor['config'])
    session.anchor = copy.deepcopy(anchor['anchor'])
    session.bootstrap = copy.deepcopy(anchor['bootstrap'])
    session.triplet_id = anchor['triplet_id']
    session.cup_sampler = copy.deepcopy(anchor['cup_sampler'])
    # This invocation restores the recorded anchor; it never samples a scene.
    session.sampled = 0
    collector.triplet_session = session
    return member(anchor['config'], row['source_trial'])


def require_bootstrap(actual, expected):
    from scene_identity import require_pose_agreement
    return require_pose_agreement(actual, expected)
