"""Restore a sealed triplet through the producer's own preparation sequence."""
import copy
from pathlib import Path

from triplet_bank import COLLECTOR_CUT, RESET, SUITE, document


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
