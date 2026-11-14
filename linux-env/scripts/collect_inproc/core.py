"""One owner computes and proves a command from a frozen physical observation."""
import copy
import hashlib
import time
import numpy as np
import shared
from contract import corrective_action
from dart import TargetCandidate
from tick_contract import TickContract,TickContractError,STEP_NS,vector,world_step_errors
from comparisons import matches

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
    return dict(sim_ns=stamp,stamp_ns=stamp,sha256=hashlib.sha256(image.tobytes()).hexdigest())


class Capture:
    def __init__(self, backend, collision):
        self.backend, self.collision = backend, collision
        self.contract = TickContract()
        self.floor = None
        self.pending=None;self.sequence=0

    def boundary(self, result):
        if self.pending is not None:raise RuntimeError('Cannot reset with an unfinished image join')
        self.contract.install_boundary(result, restore=True)
        self.floor = result['invalid_through_ns']

    def prepare(self,reference,executed_reference,offset,phase,episode_id,snapshot=None):
        """Issue the clean/executed command at t; keep its real ACK for the join."""
        if self.pending is not None:raise RuntimeError('Previous image join is unfinished')
        start = time.perf_counter_ns()
        s = copy.deepcopy(snapshot if snapshot is not None else self.backend.snapshot())
        read_end = time.perf_counter_ns()
        stamp, epoch = s['sim_ns'], s['epoch']
        clock=self.backend.clock()
        previous=self.contract.previous or self.contract.boundary
        errors=[]
        if self.floor is None or previous is None:errors.append('Command has no verified boundary')
        elif (epoch!=previous['epoch'] or stamp<=self.floor or not matches(
                stamp-previous.get('pose_stamp_ns',previous['sim_ns']),STEP_NS,'native_ns')):
            errors.append('Command skips or crosses its retained boundary')
        else:
            errors.extend(world_step_errors(s['physics_time_s'],previous['physics_time_s']))
            targets=previous.get('applied_command',previous.get('drive_targets'))
            if not matches(s['drive_targets'],targets,'target_rad'):errors.append('Previously ACKed targets did not persist')
        if epoch!=clock['epoch'] or stamp!=clock['sim_ns']:errors.append('Command observation is stale')
        if s['state_source']!='isaac_articulation':errors.append('Command state is not measured')
        for key in ('state','velocity','drive_targets'):
            if not vector(s.get(key),6):errors.append('Invalid command observation '+key)
        if errors:raise TickContractError(errors)
        received = time.monotonic_ns()
        row = dict(tick_contract_version=1, epoch=epoch, observation_epoch=epoch,
            expert_epoch=epoch, invalid_through_ns=self.floor, sim_ns=stamp,
            pose_stamp_ns=stamp, joint_stamp_ns=stamp,wall_ns=received,joint_receive_ns=received,
            state=list(s['state']), velocity=list(s['velocity']),
            physics_state=list(s['state']), physics_velocity=list(s['velocity']),
            physics_drive_targets=list(s['drive_targets']), physics_time_s=s['physics_time_s'],
            physical_bodies=copy.deepcopy(s['bodies']), state_source=s['state_source'],
            episode_id=episode_id, phase=phase, executing=True,
            camera_layout={key:dict(encoding='rgb8', width=640, height=480,
                step=1920, bytes=640*480*3) for key in CAMERAS})
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
        if not matches(row['executed_action'],row['applied_command'],'target_rad'):
            raise TickContractError(['ACK target mismatch'])
        self.sequence+=1
        timing=dict(state_read_ms=(read_end-start)/1e6,expert_ms=(compute_end-read_end)/1e6,
            collision_ms=(check_end-compute_end)/1e6,apply_ms=(time.perf_counter_ns()-check_end)/1e6)
        self.pending=dict(token=self.sequence,snapshot=s,row=row,timing=timing)
        return self.sequence

    def complete(self,token,images,camera_stamps):
        """Materialize only original camera frames for the retained observation."""
        if images is None:raise ValueError('A live image join requires both camera arrays')
        return self._complete(token,images,camera_stamps)

    def playing_tick(self,reference,executed_reference,offset,phase,episode_id,snapshot):
        """One control advance, whether its original image arrives now or next update."""
        token=self.prepare(reference,executed_reference,offset,phase,episode_id,snapshot)
        if self.backend.frame_ready(snapshot):
            images,profile=self.backend.render(expected=snapshot)
            row,checked,timing=self.complete(token,images,profile['camera_stamps'])
            advance_ms=self.backend.step()
        else:
            advance_ms=self.backend.step()
            images,profile=self.backend.render(expected=snapshot)
            row,checked,timing=self.complete(token,images,profile['camera_stamps'])
        return row,checked,timing,images,profile,advance_ms

    def _complete(self,token,images,camera_stamps,physics_only=False):
        start=time.perf_counter_ns()
        if self.pending is None or token!=self.pending['token']:raise RuntimeError('Stale or duplicated image join')
        prepared=self.pending;s=prepared['snapshot'];row=copy.deepcopy(prepared['row'])
        if set(camera_stamps)!=set(CAMERAS):raise ValueError('Both original camera stamps are required')
        clock=self.backend.clock();lag=clock['sim_ns']-s['sim_ns']
        if clock['epoch']!=s['epoch'] or not (matches(lag,0,'native_ns') or matches(lag,STEP_NS,'native_ns')):
            raise RuntimeError('Image join crossed an epoch or exceeded one physical update')
        if matches(lag,STEP_NS,'native_ns'):
            errors=world_step_errors(clock['physics_time_s'],s['physics_time_s'])
            if errors:raise TickContractError(errors)
        elif not matches(clock['physics_time_s'],s['physics_time_s'],'world_s'):
            raise TickContractError(['Held image join changed the World clock'])
        received=time.monotonic_ns()
        row.update(wrist_stamp_ns=camera_stamps['wrist'],top_stamp_ns=camera_stamps['top'],
            wrist_receive_ns=received,top_receive_ns=received)
        if not physics_only:
            if set(images)!=set(CAMERAS):raise ValueError('Both cameras must belong to the captured frame')
            row['camera_identity']={k:dict(image_identity(images[k],camera_stamps[k]),
                native_tick=dict(epoch=s['epoch'],sim_ns=s['sim_ns'])) for k in CAMERAS}
        else:row['render_pending']=True
        self.contract.observe(row)
        proof = self.contract.finish(row)
        checked = copy.deepcopy(s)
        checked['tick_contract'] = proof
        timing=dict(prepared['timing'],join_contract_ms=(time.perf_counter_ns()-start)/1e6,
            image_arrival_sim_ns=clock['sim_ns'],image_delay_updates=0 if matches(lag,0,'native_ns') else 1)
        self.pending=None
        return row, checked, timing

    def tick(self, reference, executed_reference, offset, phase, episode_id,
             images=None,snapshot=None,camera_stamps=None):
        # Held and physics-only callers retain the existing synchronous entrance.
        if images is not None:
            if set(images)!=set(CAMERAS):raise ValueError('Both cameras must belong to the captured frame')
            for pixels in images.values():image_identity(pixels,0)
        if camera_stamps is not None:
            stamp=(snapshot if snapshot is not None else self.backend.snapshot())['sim_ns']
            if set(camera_stamps)!=set(CAMERAS) or any(not matches(value,stamp,'stamp_ns') for value in camera_stamps.values()):
                raise TickContractError(['Camera stamps do not match the held observation'])
        token=self.prepare(reference,executed_reference,offset,phase,episode_id,snapshot)
        stamp=self.pending['snapshot']['sim_ns']
        stamps=camera_stamps if camera_stamps is not None else dict(wrist=stamp,top=stamp)
        return self._complete(token,images,stamps,physics_only=images is None)
