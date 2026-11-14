"""One owner computes and proves a command from a frozen physical observation."""
import copy
import hashlib
import time
import numpy as np
import shared
from contract import corrective_action
from dart import TargetCandidate
from tick_contract import TickContract

CAMERAS = ('wrist', 'top')


def command(snapshot, reference, execution_reference, offset):
    clean = corrective_action(snapshot['state'], reference)
    planner = list(execution_reference[:5]) + [clean[5]]
    if len(offset) != 6 or offset[5] != 0:
        raise ValueError('DART must contain six values and an unperturbed gripper')
    executed = [a+b for a, b in zip(planner, offset)]
    candidate = TargetCandidate(snapshot['epoch'], snapshot['sim_ns'], tuple(executed),
        tuple((a-b)*30 for a, b in zip(executed[:5], snapshot['drive_targets'][:5])),
        tuple(a-b for a, b in zip(executed, clean)), (0.,)*6, tuple(executed),
        planner_reference=tuple(planner))
    return clean, candidate


def image_identity(image, stamp):
    if image.dtype != np.uint8 or image.shape != (480, 640, 3):
        raise ValueError('Camera must provide 640x480 RGB uint8')
    return dict(sim_ns=stamp, sha256=hashlib.sha256(image.tobytes()).hexdigest())


class Capture:
    def __init__(self, backend, collision):
        self.backend, self.collision = backend, collision
        self.contract = TickContract()
        self.floor = None

    def boundary(self, result):
        self.contract.install_boundary(result, restore=True)
        self.floor = result['invalid_through_ns']

    def tick(self, reference, executed_reference, offset, phase, episode_id,
             images=None, snapshot=None):
        start = time.perf_counter_ns()
        s = copy.deepcopy(snapshot if snapshot is not None else self.backend.snapshot())
        read_end = time.perf_counter_ns()
        stamp, epoch = s['sim_ns'], s['epoch']
        received = time.monotonic_ns()
        row = dict(tick_contract_version=1, epoch=epoch, observation_epoch=epoch,
            expert_epoch=epoch, invalid_through_ns=self.floor, sim_ns=stamp,
            pose_stamp_ns=stamp, joint_stamp_ns=stamp, wrist_stamp_ns=stamp,
            top_stamp_ns=stamp, wall_ns=received, joint_receive_ns=received,
            wrist_receive_ns=received, top_receive_ns=received,
            state=list(s['state']), velocity=list(s['velocity']),
            physics_state=list(s['state']), physics_velocity=list(s['velocity']),
            physics_drive_targets=list(s['drive_targets']), physics_time_s=s['physics_time_s'],
            physical_bodies=copy.deepcopy(s['bodies']), state_source=s['state_source'],
            episode_id=episode_id, phase=phase, executing=True,
            camera_layout={key:dict(encoding='rgb8', width=640, height=480,
                step=1920, bytes=640*480*3) for key in CAMERAS})
        if images is not None:
            if set(images) != set(CAMERAS):
                raise ValueError('Both cameras must belong to the captured frame')
            row['camera_identity'] = {k:image_identity(images[k], stamp) for k in CAMERAS}
        else:
            # Physics-pass rows are never exportable until replay is materialized.
            row['render_pending'] = True
        self.contract.observe(row)
        clean, candidate = command(s, reference, executed_reference, offset)
        compute_end = time.perf_counter_ns()
        geometry = self.collision(candidate, s)
        check_end = time.perf_counter_ns()
        ack = self.backend.apply(candidate.positions, stamp, epoch)
        row.update(action=clean, executed_action=list(candidate.positions),
            expert_reference=list(reference), expert_observation_stamp_ns=stamp,
            observation_stamp_ns=stamp, declared_application_stamp_ns=stamp,
            command_clock_ns=stamp, planner_execution_reference=list(candidate.planner_reference),
            dart_offset=list(offset), command_filter_delta=[0.]*6,
            planned_noise_offset=list(candidate.noise_offset),
            command_velocity=list(candidate.velocities), execution_geometry=geometry, **ack)
        proof = self.contract.finish(row)
        checked = copy.deepcopy(s)
        checked['tick_contract'] = proof
        timing = dict(state_read_ms=(read_end-start)/1e6,
            expert_ms=(compute_end-read_end)/1e6, collision_ms=(check_end-compute_end)/1e6,
            apply_contract_ms=(time.perf_counter_ns()-check_end)/1e6)
        return row, checked, timing
