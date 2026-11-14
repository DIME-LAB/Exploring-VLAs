#!/usr/bin/env python3
"""ROS sim-tick collector. Run with system Humble Python; export with pixi separately."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import copy
import hashlib
import collections
import json
import math
import os
from pathlib import Path
import queue
import random
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import collection_status
from ros_lifecycle import retire_executor,shutdown_context
from wall_profile import profile_tick,summary as wall_profile_summary,milliseconds,PROFILE_POLICY,joined_profiles
sys.dont_write_bytecode = True

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'vla_SO-ARM101/src/so_arm101_control'))
sys.path.insert(0, str(REPO / 'omni-kit-mcp'))
from kit_mcp.client import call as kit_call
from so_arm101_control.grasp_geometry import JOINT_LIMITS
from so_arm101_control.motion_wait import MotionWaiter, MotionWaitTimeout
from gpu_device import owned_gpu_audit,gpu_uuid,GpuIsolationError
from safe_home import make_baseline,validate_baseline
from frame_join import SensorAmbiguity,identity as frame_identity,exact_duplicate,publisher_identity,publisher_key,CAMERA_SELECTION_POLICY
from image_transport import publisher_executor_type
from comparisons import comparison,matches,interval,native_frame,TOLERANCES
from contract import COLORS, JOINTS, EVAL_SEEDS, corrective_action, parameters, timing_audit
from checkpoint import progress
from storage import atomic_json,CANONICAL_ROOT, RootFileGrowth, SceneSafetyMarker, confined, prepare_environment, wait_for_storage
from protocol import StepGeneration, CancelledGeneration, OwnershipStalled
from scene_sync import physical_pose_caches, collision_pose_errors
from scene_reconcile import SceneReconciler, ros_messages
from success import SETTLED_TICKS, initial_arm, settled_containment
from cup_geometry import PROFILE,CUP,PRIMITIVES,primitive_world_poses
from diagnostics import final_stills
from query_service import StrictQueryClient
from recovery import REGRASP_POLICY, regrasp_slot, recovery_geometry, detect_miss
from dart import DEFAULT_NOISE, SmoothDart, TargetCandidate, PlannerSchedule
from command_geometry import choose_noise_scale, final_command, QUERY_POLICY, frame_capacity_seconds
from tick_contract import TickContract, TickContractError, require_snapshots, digest, verify_proof, CLOCK_POLICY, STEP_NS
from episode_policy import (POLICY, QualityReject, ConvergenceFailure, WorkerRegistry,
                            PLANNER_REJECTIONS, FailureStreak, disposition)

def ns(stamp):
    return stamp.sec * 1000000000 + stamp.nanosec


def mcp(tool, params=None):
    result = kit_call(tool, params or {}, host='127.0.0.1',
                      port=int(os.environ['SOARM_KIT_MCP_PORT']), timeout=30)
    if tool == 'run_python' and isinstance(result, dict) and 'result' in result:
        result = result['result']
    if isinstance(result, dict) and result.get('status') not in (None, 'success'):
        raise RuntimeError(f'{tool}: {result}')
    return result


def timeline(play):
    return mcp('run_python', {'code': 'import collection_runtime as c\nresult=c.runtime().' +
                              ('advance()' if play else 'cancel()')})


class Sensors:
    """Join first camera pixels and measured state to unique native ticks; retain raw stamps."""
    def __init__(self, node, generation):
        self.generation = generation
        self.condition = threading.Condition()
        self.pending = collections.OrderedDict()
        self.last = -1
        self.failure = None
        self.duplicate_messages = collections.Counter()
        self.consumed = collections.OrderedDict()
        self.seen_camera_frames = {'wrist':{},'top':{}}
        self.camera_publishers={}
        self.camera_first_frame={}
        self.rerender_messages=collections.Counter()
        self.stale_camera_messages=collections.Counter()
        self.rerender_log={}
        self.camera_candidates={}
        self.camera_native_ticks={}
        self.native_ticks=collections.OrderedDict()
        self.camera_boundary_floor=-1
        self.reference = None
        self.acks = {}
        self.subs = []
        for key, topic in [('wrist', '/wrist_camera_rgb_sim'), ('top', '/workspace_camera_sim')]:
            subscription=node.create_subscription(Image, topic,self.camera_callback(key),
                QoSProfile(depth=64,reliability=ReliabilityPolicy.RELIABLE))
            subscription.collection_publisher_info=True
            self.subs.append(subscription)
        self.subs.append(node.create_subscription(JointState, '/joint_states_measured',
                                                lambda msg: self.put('joint', msg), 100))
        self.subs.append(node.create_subscription(String, '/collection_tick',
                                                lambda msg: self.put('tick', msg), 64))
        self.subs.append(node.create_subscription(JointState, '/joint_states', self.reference_received, 100))
        self.subs.append(node.create_subscription(String, '/collection_command_ack', self.ack_received, 64))

    def camera_callback(self,key):
        def receive(message,info=None):self.put(key,message,info)
        return receive

    def ack_received(self, msg):
        ack = json.loads(msg.data)
        with self.condition:
            self.acks[(ack['epoch'], ack['observation_stamp_ns'])] = ack
            if len(self.acks) > 64:
                del self.acks[next(iter(self.acks))]
            self.condition.notify_all()

    def read_ack(self, epoch, stamp):
        with self.condition:
            if not self.condition.wait_for(lambda:(epoch, stamp) in self.acks, timeout=10):
                raise TimeoutError('Physical command acknowledgement exceeded 10 wall seconds')
            ack = self.acks.pop((epoch, stamp))
        if 'error' in ack:
            raise RuntimeError(ack['error'])
        return ack

    def reference_received(self, msg):
        stamp = ns(msg.header.stamp)
        # The raw reference topic has no generation field. Keep receipt epoch
        # separate and infer source epoch only when its sim stamp is beyond
        # the current reset barrier; never relabel a delayed diagnostic.
        epoch, floor = self.generation.identity
        source_epoch = epoch if epoch is not None and stamp > floor else None
        reference = dict(source_topic='/joint_states', stamp_ns=stamp,
            receipt_epoch=epoch, source_epoch=source_epoch, received_wall_ns=time.monotonic_ns(),
            source_epoch_basis='inferred from sim header and native reset barrier; topic has no epoch field',
            validity='current_generation' if source_epoch is not None else 'before_reset_or_unknown',
            positions=dict(zip(msg.name, msg.position)))
        with self.condition:
            self.reference = reference

    def reference_at(self, epoch, stamp):
        with self.condition:
            reference = dict(self.reference) if self.reference is not None else None
        if reference is not None:
            if not interval(reference['stamp_ns']-stamp,-math.inf,0,'stamp_ns'):
                reference['source_epoch'] = None
                reference['validity'] = 'future_stamp_or_unknown'
            reference['same_tick'] = reference['source_epoch'] == epoch and matches(reference['stamp_ns'],stamp,'stamp_ns')
            reference['age_sim_ns'] = stamp-reference['stamp_ns']
        return reference

    def camera_native_tick(self,key,first,required=False):
        token=(key,first['stamp_ns'])
        frames=self.camera_candidates[token]
        candidates={value for value in self.native_ticks if value[0]==first['receipt_epoch'] and
                    any(matches(stamp,value[1],'stamp_ns') for stamp in frames)}
        if token in self.camera_native_ticks:candidates.add(self.camera_native_ticks[token])
        if len(candidates)==1:
            native=next(iter(candidates))
            if all(matches(stamp,native[1],'stamp_ns') for stamp in frames):
                self.camera_native_ticks[token]=native
                return native
        if candidates or required:
            self.failure=SensorAmbiguity(dict(stream=key,reason='camera selection requires one unique native tick',
                candidates=list(frames.values()),native_ticks=[dict(epoch=e,sim_ns=n) for e,n in sorted(candidates)],
                owner_epoch=first['receipt_epoch'],stamp_rule='stamp_ns',tolerance_ns=TOLERANCES['stamp_ns']))
            self.condition.notify_all();raise self.failure
        earliest=self.camera_boundary_floor+STEP_NS-TOLERANCES['stamp_ns']-TOLERANCES['native_ns']
        if self.camera_boundary_floor>=0 and any(stamp<earliest for stamp in frames):
            self.failure=SensorAmbiguity(dict(stream=key,reason='late unresolved camera group behind durable boundary',
                candidates=list(frames.values()),native_floor_ns=self.camera_boundary_floor,
                earliest_next_header_ns=earliest,owner_epoch=first['receipt_epoch']))
            self.condition.notify_all();raise self.failure
        return None  # Provisional first pixels cannot be consumed before native tick delivery.

    def require_camera_boundary(self,floor,timeout=5.):
        deadline=time.monotonic()+timeout
        with self.condition:
            while True:
                if self.failure:raise self.failure
                unresolved=[]
                for camera,frames in self.seen_camera_frames.items():
                    for first in frames.values():
                        if self.camera_native_tick(camera,first) is None:
                            unresolved.extend(self.camera_candidates[(camera,first['stamp_ns'])].values())
                if not unresolved:
                    self.camera_boundary_floor=max(self.camera_boundary_floor,floor)
                    return dict(passed=True,native_floor_ns=floor,groups=sum(len(f) for f in self.seen_camera_frames.values()))
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    self.failure=SensorAmbiguity(dict(reason='unresolved camera groups at durable boundary',
                        candidates=unresolved,native_floor_ns=floor,timeout_wall_s=timeout))
                    self.condition.notify_all();raise self.failure
                self.condition.wait(min(remaining,.1))

    def put(self,key,msg,info=None):
        if key in ('wrist','top'):
            with self.generation.observe_identity() as identity:
                return self._put_frame(key,msg,info,identity)
        return self._put_frame(key,msg,info)

    def _put_frame(self, key, msg,info=None,identity=None):
        if isinstance(msg,BaseException):
            with self.condition:
                self.failure=self.failure or msg;self.condition.notify_all()
                raise self.failure
        receive_ns=time.monotonic_ns()
        if key == 'tick':
            msg = json.loads(msg.data)
            raw_stamp = msg['sim_ns']
        else:
            raw_stamp = ns(msg.header.stamp)
        if raw_stamp<=0:return
        if key in ('wrist','top') and (identity[0] is None or raw_stamp<=identity[1]+int(TOLERANCES['stamp_ns'])):
            with self.condition:self.stale_camera_messages[key]+=1
            return
        incoming=frame_identity(key,msg,raw_stamp)
        incoming['receipt_epoch']=self.generation.epoch
        with self.condition:
            if self.failure:return
            if key in ('wrist','top'):
                if identity[0] is None or raw_stamp<=identity[1]+int(TOLERANCES['stamp_ns']):
                    self.stale_camera_messages[key]+=1;return
                incoming['receipt_epoch']=identity[0]
                try:incoming['publisher']=publisher_identity(info)
                except SensorAmbiguity as exc:
                    self.failure=exc;self.condition.notify_all();raise
            if key=='tick':
                self.native_ticks[(msg['epoch'],raw_stamp)]=True
                while len(self.native_ticks)>240:self.native_ticks.popitem(last=False)
            if not hasattr(self,'consumed'):self.consumed=collections.OrderedDict()
            if not hasattr(self,'seen_camera_frames'):self.seen_camera_frames={'wrist':{},'top':{}}
            if key=='tick':
                for camera,frames in self.seen_camera_frames.items():
                    for bucket in range(raw_stamp//int(TOLERANCES['stamp_ns'])-3,raw_stamp//int(TOLERANCES['stamp_ns'])+4):
                        if bucket in frames:self.camera_native_tick(camera,frames[bucket])
            if key in self.seen_camera_frames:
                publisher=incoming['publisher'];bound=self.camera_publishers.get(key)
                if bound is not None and publisher_key(bound)!=publisher_key(publisher):
                    self.failure=SensorAmbiguity(dict(stream=key,reason='camera publisher changed or duplicated',
                        candidates=[self.camera_first_frame[key],incoming],owner_epoch=self.generation.epoch))
                    self.condition.notify_all();raise self.failure
                self.camera_publishers[key]=bound or publisher
                if bound is None:self.camera_first_frame[key]=incoming
                seen=self.seen_camera_frames[key];bucket=raw_stamp//int(TOLERANCES['stamp_ns'])
                candidates=[seen[b] for b in range(bucket-2,bucket+3) if b in seen and
                    max(raw_stamp,*self.camera_candidates[(key,seen[b]['stamp_ns'])])-min(raw_stamp,*self.camera_candidates[(key,seen[b]['stamp_ns'])])<=2*TOLERANCES['stamp_ns']]
                if candidates:
                    previous=candidates[0]
                    same=(len(candidates)==1 and
                        previous['layout']==incoming['layout'] and
                        publisher_key(previous['publisher'])==publisher_key(publisher))
                    if same:
                        self.camera_candidates[(key,previous['stamp_ns'])].setdefault(raw_stamp,incoming)
                        self.camera_native_tick(key,previous)
                        if exact_duplicate(previous,incoming):self.duplicate_messages[key]+=1
                        else:
                            self.rerender_messages[key]+=1
                            token=key+':'+str(raw_stamp)
                            event=self.rerender_log.setdefault(token,dict(stream=key,stamp_ns=raw_stamp,epoch=previous['receipt_epoch'],
                                publisher=previous['publisher'],selected_content_sha256=previous['content_sha256'],count=0))
                            event.update(count=event['count']+1,last_stamp_ns=raw_stamp,last_content_sha256=incoming['content_sha256'],
                                last_publication_sequence_number=publisher.get('publication_sequence_number'))
                        return  # Never replace the first pending OR consumed image.
                    self.failure=SensorAmbiguity(dict(stream=key,reason='distinct same-camera tick/layout candidates',
                        candidates=[*candidates,incoming],owner_epoch=self.generation.epoch))
                    self.condition.notify_all();raise self.failure
                seen[bucket]=incoming
                self.camera_candidates[(key,raw_stamp)]={raw_stamp:incoming}
                self.camera_native_tick(key,incoming)
            if self.last>=0 and raw_stamp<=self.last+int(TOLERANCES['stamp_ns']):
                # This sample was never eligible for a current joined frame.
                # Previously observed camera identities above still detect a
                # conflicting late alternative to any recorded/pending frame.
                return
            candidates=[]
            for anchor,entry in self.pending.items():
                stamps=[value[0]['sim_ns'] if name=='tick' else ns(value[0].header.stamp)
                        for name,value in entry.items()]
                if all(matches(raw_stamp,stamp,'stamp_ns') for stamp in stamps):candidates.append(anchor)
            if len(candidates)>1:
                self.failure=SensorAmbiguity(dict(stream=key,reason='multiple compatible groups',
                    candidates=[(value[2] if name in ('wrist','top') else frame_identity(name,value[0],value[0]['sim_ns'] if name=='tick' else ns(value[0].header.stamp)))
                                for anchor in candidates for name,value in self.pending[anchor].items()]+[incoming],
                    owner_epoch=self.generation.epoch))
                self.condition.notify_all();raise self.failure
            anchor=candidates[0] if candidates else raw_stamp
            entry=self.pending.setdefault(anchor,{})
            if key in entry:
                old=entry[key][0];old_stamp=old['sim_ns'] if key=='tick' else ns(old.header.stamp)
                previous=frame_identity(key,old,old_stamp)
                if exact_duplicate(previous,incoming):
                    self.duplicate_messages[key]+=1;return
                self.failure=SensorAmbiguity(dict(stream=key,reason='distinct candidates for one pending tick',
                    candidates=[previous,incoming],owner_epoch=self.generation.epoch))
                self.condition.notify_all();raise self.failure
            entry[key]=(msg,receive_ns,incoming) if key in ('wrist','top') else (msg,receive_ns)
            while len(self.pending)>240:self.pending.popitem(last=False)
            self.condition.notify_all()

    def camera_diagnostics(self,motion_epochs,duplicate_start,rerender_start,stale_camera_start):
        with self.condition:
            result={}
            result['camera_duplicate_messages'] = {key: value - duplicate_start.get(key, 0) for key, value in self.duplicate_messages.items()}
            result['camera_publishers']=dict(self.camera_publishers)
            result['camera_rerenders']={key:value-rerender_start.get(key,0) for key,value in self.rerender_messages.items()}
            result['camera_stale_boundary_frames']={key:value-stale_camera_start.get(key,0) for key,value in self.stale_camera_messages.items()}
            result['camera_rerender_log']=[dict(value) for value in self.rerender_log.values() if value['epoch'] in motion_epochs]
            return result

    def prune_durable_camera_history(self):
        # The caller has atomically published this attempt's receipt. A native
        # cancellation floor already excludes these frames from future joins.
        with self.generation.observe_identity() as (_,floor),self.condition:
            for frames in self.seen_camera_frames.values():
                for key in [k for k,v in frames.items() if v['stamp_ns']<=floor+int(TOLERANCES['stamp_ns'])]:del frames[key]
            for key in [k for k,v in self.rerender_log.items() if v['stamp_ns']<=floor+int(TOLERANCES['stamp_ns'])]:del self.rerender_log[key]
            for token in [t for t in self.camera_candidates if t[1]<=floor+int(TOLERANCES['stamp_ns'])]:
                del self.camera_candidates[token];self.camera_native_ticks.pop(token,None)
            for tick in [t for t in self.native_ticks if t[1]<=floor+int(TOLERANCES['stamp_ns'])]:del self.native_ticks[tick]

    def read(self, timeout=15):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if getattr(self,'closed',False):raise CancelledGeneration('Sensor join closed')
                if self.failure:
                    if isinstance(self.failure,BaseException):raise self.failure
                    raise RuntimeError(self.failure)
                ready = sorted(k for k,v in self.pending.items() if len(v)==4)
                if ready:
                    stamp = ready[0]
                    entry = self.pending.pop(stamp)
                    for key in ('wrist','top'):
                        message,received,first=entry[key]
                        native=self.camera_native_tick(key,first,required=True)
                        snapshot=entry['tick'][0]
                        if native!=(snapshot['epoch'],snapshot['sim_ns']):
                            self.failure=SensorAmbiguity(dict(stream=key,reason='camera native tick differs from joined snapshot',native_tick=native,snapshot_epoch=snapshot['epoch'],snapshot_sim_ns=snapshot['sim_ns']))
                            self.condition.notify_all();raise self.failure
                        entry[key]=(message,received,dict(first,native_tick=dict(epoch=native[0],sim_ns=native[1]),selection='first taken pixels for unique native tick'))
                    self.last = ns(entry['wrist'][0].header.stamp)
                    self.consumed[self.last]={key:(value[2] if key in ('wrist','top') else frame_identity(key,value[0],value[0]['sim_ns'] if key=='tick' else ns(value[0].header.stamp))) for key,value in entry.items()}
                    while len(self.consumed)>240:self.consumed.popitem(last=False)
                    for old in list(self.pending):
                        if old <= self.last+int(TOLERANCES['stamp_ns']):
                            del self.pending[old]
                    return ns(entry['wrist'][0].header.stamp), entry
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    counts = collections.Counter(key for v in self.pending.values() for key in v)
                    raise TimeoutError(f'No unique camera/state sim-tick match within 1 us: {dict(counts)}')
                self.condition.wait(min(remaining, .5))

    @staticmethod
    def image(msg):
        if msg.encoding != 'rgb8' or (msg.width, msg.height) != (640, 480):
            raise ValueError(f'Expected 640x480 rgb8, got {msg.width}x{msg.height} {msg.encoding}')
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(480, 640, 3)


class EpisodeWriter:
    """Bounded asynchronous encoding, with raw camera/state/action provenance in JSONL."""
    def __init__(self, path, fps):
        self.path = path
        path.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.write_profiles=[]
        self.queue = queue.Queue(maxsize=60)
        self.failure = None
        self.encoders = {}
        for key in ('wrist', 'top'):
            self.encoders[key] = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x480', '-r', str(fps),
                '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
                '-threads', '1', '-pix_fmt', 'yuv420p', str(path / f'{key}.mp4')], stdin=subprocess.PIPE,
                stderr=open(path / f'{key}-encode.log', 'w'))
        self.thread = threading.Thread(target=self.write_loop, daemon=True)
        self.thread.start()

    def write_loop(self):
        try:
            with (self.path / 'frames.jsonl').open('w') as out, (self.path/'write-profiles.jsonl').open('w') as timings:
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    row, images,enqueue_ns = item
                    write_start=time.monotonic_ns()
                    for key, img in images.items():
                        self.encoders[key].stdin.write(img.tobytes())
                    out.write(json.dumps(row) + '\n')
                    profile=row['wall_profile']
                    completed=dict(profile_id=profile['profile_id'],epoch=profile['epoch'],sim_ns=profile['sim_ns'],
                        native_sim_ns=profile.get('native_sim_ns',profile['sim_ns']),camera_stamp_ns=row['sim_ns'],
                        row_queue_ms=milliseconds(enqueue_ns,write_start),row_video_write_ms=milliseconds(write_start,time.monotonic_ns()))
                    timings.write(json.dumps(completed)+'\n');self.write_profiles.append(completed)
                    self.rows.append(row)
        except Exception as exc:
            self.failure = repr(exc)

    def add(self, row, images):
        if self.failure:
            raise RuntimeError(self.failure)
        self.queue.put((copy.deepcopy(row), images,time.monotonic_ns()), timeout=2)

    def close(self):
        self.queue.put(None, timeout=5)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise TimeoutError('Episode encoder did not finish')
        for encoder in self.encoders.values():
            encoder.stdin.close()
            if encoder.wait(timeout=30):
                raise RuntimeError('ffmpeg encoding failed')
        if self.failure:
            raise RuntimeError(self.failure)


class Servo:
    """Execute bounded DART disturbances; label every tick with an unperturbed corrective command."""
    def __init__(self, node, generation, step):
        self.node = node
        self.generation = generation
        self.step = step
        self.lock = threading.RLock()
        self.arm = node.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.grip = node.create_publisher(JointTrajectory, '/gripper_controller/joint_trajectory', 10)
        self.plan = None
        self.grip_plan = None
        self.reference = None
        self.reference_epoch = None
        self.previous_executed = None
        self.params = parameters(0, 'red')
        self.rng = random.Random(0)
        self.last_sim_ns = 0
        self.phase = 'hold'
        self.noise_specs = []
        self.noise_index = 0
        self.previous_command = None
        self.executed_hold = None
        self.last_command_clock_ns = 0
        self.planned_sim_seconds = 0.
        self.planned_operations = 0

    def execution_budget(self):
        with self.lock:
            return self.planned_sim_seconds, self.planned_operations

    def configure(self, params):
        with self.generation.own('episode parameters'):
            self.params = dict(params)
            self.rng = random.Random(params['noise_seed'])
            self.noise_specs = []
            self.noise_index = 0
            self.executed_hold = None

    def noise_program(self, duration, tag):
        config=self.params.get('dart',DEFAULT_NOISE)
        index=getattr(self,'noise_index',0);self.noise_index=index+1
        program=SmoothDart(duration,self.params['noise_seed']+104729*index,config,
                           enabled=tag not in ('grasp_home','grasp_home_data','home') and not tag.startswith('regrasp_miss'))
        if not hasattr(self,'noise_specs'):self.noise_specs=[]
        self.noise_specs.append(dict(index=index,phase=tag,**program.spec))
        return program

    def commit_command(self, candidate, epoch, stamp):
        self.generation.require(epoch,stamp)
        if candidate.epoch!=epoch or not matches(candidate.stamp_ns,stamp):raise ValueError('Stale execution candidate')
        self.previous_command=candidate
        self.executed_hold=(epoch,candidate.held_reference) if candidate.held_reference is not None else None

    def execute(self, trajectory, done, operation=None, executed_trajectory=None, hold_executed=False):
        epoch=done.motion_epoch
        if operation is not None and (operation.event is not done or operation.epoch!=epoch):
            raise ValueError('Planner result/event origin mismatch')
        with self.generation.own('execution origin'):
            self.generation.require(epoch)
            config=dict(self.params.get('dart',DEFAULT_NOISE))
            index=self.noise_index;self.noise_index+=1
            tag=operation.tag if operation is not None else getattr(done,'motion_tag','home')
            seed=self.params['noise_seed']+104729*index
            gripper=self.node._actual_positions['gripper_joint']
        schedule=PlannerSchedule(trajectory)
        executed_schedule=PlannerSchedule(executed_trajectory) if executed_trajectory is not None else schedule
        if not matches(schedule.duration,executed_schedule.duration,'world_s'):
            raise ValueError('Clean and executed planner time laws must have the same duration')
        duration=schedule.duration
        noise=SmoothDart(duration,seed,config,enabled=tag not in ('grasp_home','grasp_home_data','home') and not tag.startswith('regrasp_miss'))
        def check(q):
            self.generation.require(epoch)
            return self.node._check_state_valid_with_contacts(dict(zip(JOINTS,q)))
        scale,preflight=choose_noise_scale(executed_schedule,noise,gripper,check,
            [JOINT_LIMITS[n] for n in JOINTS],lambda:self.generation.require(epoch))
        with self.generation.own('collector motion'):
            self.generation.require(epoch)
            with self.lock:
                self.node._plan_motion_wait(done,duration)
                self.planned_sim_seconds+=duration;self.planned_operations+=1
                self.plan=dict(trajectory=trajectory,schedule=schedule,executed_schedule=executed_schedule,
                    duration=duration,hold_executed=hold_executed,done=done,start=None,epoch=epoch,noise=noise,noise_scale=scale)
                self.executed_hold=None;self.phase=tag
                self.noise_specs.append(dict(index=index,phase=tag,**noise.spec,scale=scale,
                    execution_geometry=preflight,expert_schedule=schedule.spec,executed_schedule=executed_schedule.spec))
            self.step(True)

    def gripper(self, position, duration, blocking, phase='gripper'):
        epoch = self.node._motion_token()
        with self.generation.own('collector motion'):
            self.generation.require(epoch)
            with self.lock:
                if abs(self.node._actual_positions['gripper_joint'] - position) < .0005:
                    if self.reference is None:
                        self.reference = [self.node._actual_positions[name] for name in JOINTS]
                    self.reference_epoch = epoch
                    self.reference[5] = position
                    return True if blocking else None
            done = self.node._new_motion_event()
            self.node._plan_motion_wait(done, duration)
            with self.lock:
                self.planned_sim_seconds += duration
                self.planned_operations += 1
                self.grip_plan = dict(target=position, duration=duration, done=done, start=None, epoch=epoch)
                self.phase = phase
            self.step(True)
        if not blocking:
            return None
        try:
            finished = self.node._wait_motion_event(done, 60.)
        except MotionWaitTimeout:
            finished = False
        with self.generation.own('gripper wait continuation'):
            self.generation.require(epoch)
            if not finished:
                raise TimeoutError('Sim-time gripper execution timed out')
            # Verdict belongs to this operation; another home/gripper motion
            # can never supply success through the node's display status.
            return bool(done.verdict and done.verdict.get('ok'))

    @staticmethod
    def interpolate(trajectory, elapsed):
        points = trajectory.points
        previous = points[0]
        for point in points[1:]:
            t0, t1 = ns(previous.time_from_start) / 1e9, ns(point.time_from_start) / 1e9
            if elapsed <= t1:
                alpha = max(0., min(1., (elapsed - t0) / max(t1 - t0, 1e-9)))
                return [a + alpha * (b - a) for a, b in zip(previous.positions, point.positions)]
            previous = point
        return list(points[-1].positions)

    def tick(self, stamp, state, drive_targets=None):
        completed=[]
        margin=getattr(getattr(self.node,'_motion_waiter',None),'sim_margin',5.)
        with self.lock:
            self.last_sim_ns=stamp
            if self.reference is None:
                self.reference=list(state);self.reference_epoch=self.generation.epoch
            self.generation.require(self.reference_epoch)
            active=self.plan is not None or self.grip_plan is not None
            noise=[0.]*6;arm_ready=False;arm_elapsed=arm_duration=0.
            hold=getattr(self,'executed_hold',None)
            if hold is not None and hold[0]!=self.generation.epoch:
                raise RuntimeError('Executed hold belongs to a retired epoch')
            execution_reference=list(hold[1])+[self.reference[5]] if hold is not None else list(self.reference)
            next_hold=hold[1] if hold is not None else None
            if self.plan:
                plan=self.plan
                if plan['start'] is None:plan['start']=stamp
                arm_elapsed=(stamp-plan['start'])/1e9
                if 'schedule' not in plan:plan['schedule']=PlannerSchedule(plan['trajectory'])
                executed_schedule=plan.get('executed_schedule',plan['schedule'])
                arm_duration=plan.get('duration',plan['schedule'].duration)
                positions=plan['schedule'].sample((arm_elapsed+1/30)*plan['schedule'].duration/arm_duration)
                execution_positions=executed_schedule.sample((arm_elapsed+1/30)*executed_schedule.duration/arm_duration)
                for name,value in zip(plan['trajectory'].joint_names,positions):self.reference[JOINTS.index(name)]=value
                execution_reference=list(execution_positions)+[self.reference[5]]
                if 'noise' not in plan:plan['noise']=self.noise_program(arm_duration,self.phase)
                noise=[v*plan.get('noise_scale',1.) for v in plan['noise'].sample(arm_elapsed+1/30)]
                arm_ready=(arm_elapsed>arm_duration and max(abs(a-b) for a,b in zip(state[:5],execution_reference[:5]))<.008)
            if self.grip_plan:
                grip=self.grip_plan
                if grip['start'] is None:grip['start']=stamp;grip['initial']=state[5]
                elapsed=(stamp-grip['start'])/1e9
                alpha=min(1.,(elapsed+1/30)/grip['duration'])
                self.reference[5]=grip['initial']+alpha*(grip['target']-grip['initial'])
                if elapsed>grip['duration'] and abs(state[5]-grip['target'])<.008:
                    completed.append(grip['done']);self.grip_plan=None
                elif elapsed>grip['duration']+margin:
                    raise ConvergenceFailure('gripper_nonconvergence',epoch=grip['epoch'],stamp_ns=stamp,
                        phase=self.phase,duration=grip['duration'],elapsed=elapsed,margin=margin,state=state,reference=self.reference)
            clean=corrective_action(state,self.reference)
            execution_reference[5]=clean[5]  # Gripper has exactly the clean expert command.
            desired=[q+n for q,n in zip(execution_reference,noise)]
            previous=getattr(self,'previous_command',None)
            if previous is not None and previous.epoch!=self.generation.epoch:raise RuntimeError('Previous execution belongs to a retired epoch')
            anchor=drive_targets if drive_targets is not None else (previous.positions if previous is not None else state)
            velocities=tuple((q-p)*30 for q,p in zip(desired[:5],anchor[:5]))
            if self.plan:
                if arm_ready:
                    next_hold=tuple(execution_reference[:5]) if self.plan.get('hold_executed') else None
                    completed.append(self.plan['done']);self.plan=None
                elif arm_elapsed>arm_duration+margin:
                    raise ConvergenceFailure('arm_nonconvergence',epoch=self.plan['epoch'],stamp_ns=stamp,
                        phase=self.phase,duration=arm_duration,elapsed=arm_elapsed,margin=margin,state=state,reference=execution_reference)
            candidate=TargetCandidate(self.generation.epoch,stamp,tuple(desired),velocities,
                tuple(e-c for e,c in zip(desired,clean)),(0.,)*6,tuple(desired),next_hold,
                tuple(execution_reference),plan.get('noise_scale',1.) if active and 'plan' in locals() else 0.)
            executed=list(candidate.positions)
            reference=list(self.reference)
            self.last_command_clock_ns=self.node.get_clock().now().nanoseconds
            phase=self.phase
        return (clean,executed,active,phase,reference,self.last_command_clock_ns,completed,
                self.plan is not None or self.grip_plan is not None,candidate)

    def publish_reference(self, executed):
        # Expert and noisy commands are independent arrays. Never derive labels from controller feedback.
        for pub, names, values in [(self.arm, JOINTS[:5], executed[:5]), (self.grip, JOINTS[5:], executed[5:])]:
            msg = JointTrajectory()
            msg.joint_names = list(names)
            offset = 0 if len(names) == 5 else 5
            velocities = ([30 * (value - self.previous_executed[offset + i]) for i, value in enumerate(values)]
                          if self.previous_executed is not None else [0.] * len(values))
            msg.points = [JointTrajectoryPoint(positions=values, velocities=velocities,
                time_from_start=Duration(sec=0, nanosec=0))]
            pub.publish(msg)
        self.previous_executed = executed


class Collector:
    def __init__(self, args):
        if args.resume:collection_status.require_finalized(confined(args.output))
        from so_arm101_control.pick_place import PickPlaceNode
        from so_arm101_control.scene_transactions import SceneExecution
        self.args = args
        self.scene_marker = SceneSafetyMarker(os.environ['SOARM_INSTANCE_ROOT'])
        self.scene_marker.require_fresh_stack()
        self.generation = StepGeneration()
        self.generation.retirement_timeout = frame_capacity_seconds()+5.
        # Node construction initializes guarded status before a native epoch exists.
        # Only this synchronous constructor thread owns that bootstrap state.
        self._constructing_thread = threading.get_ident()
        try:
            self.node = PickPlaceNode(use_sim_time=True, joint_states_topic='/joint_states_measured',
                execution=SceneExecution('collector', self.motion_effect, self.scene_incomplete, self.callback_failure))
        finally:
            self._constructing_thread = None  # Revoke before either collector thread starts.
        self.ros_context=self.node.context
        self.node.validity_client=StrictQueryClient(self.node.validity_client,'MoveIt state validity',
            on_failure=lambda exc:self.abort_stalled_owner(exc,'MoveIt validity query'))
        self.node._motion_context = threading.local()
        self.node._motion_generation_provider = lambda: self.generation.epoch
        self.sensors = Sensors(self.node, self.generation)
        # Only joined, generation-validated snapshots update expert state/cache.
        self.node.destroy_subscription(self.node.js_sub)
        with self.node.objects_lock, self.node._drop_lock:
            self.node._physical_pose_cache = True
        self.servo = Servo(self.node, self.generation, self.timeline)
        self.command_pub = self.node.create_publisher(String, '/collection_command', 64)
        self.node._trajectory_executor = self.servo.execute
        self.node._gripper_executor = self.servo.gripper
        self.executor = publisher_executor_type()(num_threads=4)
        self.executor.add_node(self.node)
        self.ros_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.writer = None
        self.writer_lock = threading.Lock()
        self.stop = threading.Event()
        self.failure = None
        self.wait_diagnostics = []
        self.capture_failures = []
        self.node._motion_waiter = MotionWaiter(
            clock=lambda:self.node.get_clock().now().nanoseconds/1e9,
            validate=self.generation.require, emit=self.motion_wait_diagnostic,
            abort=lambda:self.stop.is_set() or self.node._qs_abort_evt.is_set(),
            rtf_floor=args.wait_rtf_floor,wall_margin=frame_capacity_seconds()+30.)
        self.status_lock=threading.RLock()
        self.finalization_certificate=None
        self.startup_status_output=confined(Path(os.environ['SOARM_INSTANCE_ROOT'])/'collector-startup'/uuid.uuid4().hex)
        self.current_attempt_path=None
        self.stalled_abort_lock = threading.Lock()
        self.stalled_abort_started = False
        self.generation.on_stall = self.abort_stalled_owner
        self.tick_condition = threading.Condition()
        self.last_processed_ns = 0
        self.last_snapshot = None
        self.tick_audit = TickContract()
        self.node._simulation_held = lambda: mcp('run_python', {'code':
            'import collection_runtime as c\nr=c.runtime()\nresult=not r.busy and not r.timeline.is_playing()'})
        self.scene_reset = SceneReconciler(self.node, ros_messages(),
            (REPO/'vla_SO-ARM101/src/so_arm101_moveit_config/config/so_arm101.srdf').read_text(),primitive_cups=True)
        self.node._add_cup_collision_objects=self.add_cup_collisions
        self.workers = WorkerRegistry(self.generation, self.node._motion_thread, self.node._motion_token)
        self.node._motion_thread = self.workers.thread
        # Install terminal hooks and their locks before any executor callback.
        self.ros_thread.start()
        self.capture_thread = threading.Thread(target=self.capture, daemon=True)
        self.capture_thread.start()


    def gpu_footprint(self):
        proof=owned_gpu_audit(Path(os.environ['SOARM_INSTANCE_ROOT']),int(os.getenv('ISAACSIM_GPU','0')))
        self.last_gpu_footprint=proof
        if not proof['passed']:raise GpuIsolationError(proof)
        return proof

    def motion_wait_diagnostic(self, record):
        self.wait_diagnostics.append(record)
        print(json.dumps(record), flush=True)

    @contextmanager
    def motion_effect(self, epoch):
        if (getattr(self, '_constructing_thread', None) == threading.get_ident() and
                epoch is None and self.generation.epoch is None and not self.generation.stalled):
            yield
            return
        try:
            self.generation.require(epoch)
            with self.generation.own('scene/bookkeeping/status'):
                self.generation.require(epoch)
                yield
        except OwnershipStalled as exc:
            self.abort_stalled_owner(exc, 'scene/bookkeeping/status')
            raise

    def motion_call(self, callback, *args, epoch=Ellipsis):
        with self.generation.own('motion request'):
            if self.stop.is_set():
                raise CancelledGeneration('Collector is stopping')
            epoch = self.generation.epoch if epoch is Ellipsis else epoch
            self.generation.require(epoch)
        context = self.node._motion_context
        saved = {key:getattr(context,key) for key in ('epoch','event') if hasattr(context,key)}
        context.epoch = epoch
        try:
            return callback(*args)
        finally:
            for key in ('epoch','event'):
                if key in saved:setattr(context,key,saved[key])
                elif hasattr(context,key):delattr(context,key)

    def retire_motion(self, epoch=None):
        with self.servo.lock:
            idle = self.servo.plan is None and self.servo.grip_plan is None
            hold = (tuple(self.servo.reference) if idle and self.servo.reference is not None and
                    self.servo.reference_epoch == epoch else None)
            for plan in (self.servo.plan, self.servo.grip_plan):
                if plan:
                    if not getattr(plan['done'], 'verdict', None):
                        plan['done'].verdict = dict(ok=False, outcome='cancelled', epoch=plan['epoch'])
                    plan['done'].set()
            self.servo.plan = self.servo.grip_plan = None
            self.servo.previous_executed = None
            self.servo.reference = None
            self.servo.reference_epoch = None
            if hold is None:
                self.servo.previous_command = None
                self.servo.executed_hold = None
        with self.sensors.condition:
            self.sensors.pending.clear()
            self.sensors.acks.clear()
        return hold

    def scene_incomplete(self, exc, source='MoveIt'):
        self.node._scene_terminal_error = str(exc)
        self.generation.poison(exc)
        self.persist_terminal_fault(exc,source)
        self.stop.set()
        self.node._qs_abort_evt.set()
        try:
            self.scene_marker.record(exc, source=source)
        finally:
            self.generation.fail(exc, 'unresolved '+source+' scene mutation')

    def mutate_kit(self, tool, params=None):
        try:
            return mcp(tool, params)
        except (TimeoutError, ConnectionError, OSError) as exc:
            self.scene_incomplete(exc, source='Kit')
            raise

    def status_output(self):
        return Path(self.args.output) if self.finalization_certificate is not None else self.startup_status_output

    def persist_terminal_fault(self,exc,source):
        sensor=getattr(self.sensors,'failure',None)
        with self.status_lock:
            return collection_status.terminal(self.status_output(),dict(error=repr(exc),source=source,
                epoch=self.generation.epoch,sensor_evidence=getattr(sensor,'evidence',None)),
                attempt=self.current_attempt_path if self.finalization_certificate is not None else None)

    def callback_failure(self, name, exc):
        self.abort_stalled_owner(OwnershipStalled('ROS callback '+name+': '+repr(exc)), 'ROS callback')

    def abort_stalled_owner(self, exc, requester):
        # Invalidate every existing worker before any emergency RPC or wait.
        self.generation.poison(exc)
        self.stop.set()
        self.node._qs_abort_evt.set()
        with self.stalled_abort_lock:
            if self.stalled_abort_started:
                return
            self.stalled_abort_started = True
            self.failure = str(exc)
        with self.tick_condition:
            self.tick_condition.notify_all()
        try:self.persist_terminal_fault(exc,requester)
        except Exception as publication_error:self.failure+='; terminal status publication failed: '+repr(publication_error)
        # No generation lock: its owner may be stuck in publication forever.
        # Native cancellation only pauses and invalidates queued commands.
        try:
            timeline(False)
        except Exception as pause_error:
            self.failure += '; emergency pause failed: '+repr(pause_error)
        finally:
            print(json.dumps(dict(event='stalled_generation_owner', requester=requester,
                error=self.failure, action='abort; no reset or next episode permitted')), flush=True)

    def cancel_owner(self, native_cancel, rebase=False):
        try:
            with self.generation.own('cancel/hold handover'):
                epoch = self.generation.epoch
                handover = []
                previous_command=getattr(self.servo,'previous_command',None)
                if previous_command is not None and previous_command.epoch!=epoch:raise RuntimeError('Cannot hand over stale command history')
                executed_hold=getattr(self.servo,'executed_hold',None)
                if executed_hold is not None and executed_hold[0]!=epoch:
                    raise RuntimeError('Cannot hand over a stale executed hold')
                result = self.generation.cancel(native_cancel,
                    lambda:handover.append(self.retire_motion(epoch)))
                rebase=rebase or 'arm_restored' in result or 'command_baseline' in result
                if not hasattr(self,'tick_audit'):self.tick_audit=TickContract()
                baseline=self.tick_audit.install_boundary(result,restore=rebase,
                    pause_targets=previous_command.positions if previous_command is not None else None)
                # Only an idle clean reference crosses this barrier. Active
                # motion is retired and will freeze at fresh measured state.
                with self.servo.lock:
                    if handover[0] is not None:
                        self.servo.reference = list(handover[0])
                        self.servo.reference_epoch = result['epoch']
                        self.servo.executed_hold=(result['epoch'],executed_hold[1]) if executed_hold is not None else None
                        if getattr(self.servo,'previous_command',None) is not None:
                            self.servo.previous_command=replace(self.servo.previous_command,epoch=result['epoch'])
                    if rebase:
                        targets=tuple(baseline['drive_targets'])
                        self.servo.previous_command=TargetCandidate(result['epoch'],baseline['sim_ns'],targets,
                            (0.,)*5,(0.,)*6,(0.,)*6,targets,planner_reference=targets,noise_scale=0.)
                        if result['arm_restored']:self.servo.executed_hold=None
                return result
        except OwnershipStalled as exc:
            self.abort_stalled_owner(exc, 'cancel/reset')
            raise OwnershipStalled(self.failure) from exc
        except Exception as exc:
            self.abort_stalled_owner(exc,'cancel/restore baseline')
            raise

    def timeline(self, play):
        if not play:
            return self.cancel_owner(lambda: timeline(False))
        with self.generation.own('advance'):
            self.generation.require(self.generation.epoch)
            return timeline(True)

    def restore_physical(self, code):
        # Retirement owns the scene before setters. On a stalled owner, only
        # native cancellation is allowed; never execute the restore callback.
        return self.cancel_owner(lambda: self.mutate_kit('run_python', {'code':code}),rebase=True)

    def record_capture_failure(self, exc, epoch, failure_tick=None):
        # Called while owning this generation, before cancellation retires plans.
        plan = self.servo.plan or self.servo.grip_plan
        duration = (ns(plan['trajectory'].points[-1].time_from_start)/1e9
                    if plan and 'trajectory' in plan else plan.get('duration') if plan else None)
        if plan and 'schedule' in plan:duration=plan.get('duration',plan['schedule'].duration)
        record = dict(epoch=epoch, sim_ns=getattr(self.servo, 'last_sim_ns', None),
            wall_ns=time.monotonic_ns(), phase=getattr(self.servo, 'phase', None),
            exception_type=type(exc).__name__, error=repr(exc),
            measured_state=[self.node._actual_positions[name] for name in JOINTS],
            expert_reference=list(self.servo.reference) if self.servo.reference is not None else None,
            operation_start_sim_ns=plan.get('start') if plan else None,
            planned_sim_seconds=duration,
            expert_schedule=plan['schedule'].spec if plan and 'schedule' in plan else None,
            convergence_margin_sim_seconds=getattr(getattr(self.node, '_motion_waiter', None), 'sim_margin', 5.),
            cancellation=dict(cause=repr(exc), invalidated_epoch=epoch, requested=False, completed=False))
        if isinstance(exc, ConvergenceFailure):
            record['physical_convergence'] = dict(exc.evidence)
            record['failing_tick_contract']=failure_tick
            record['validated_sim_tick']=bool(failure_tick and not verify_proof(failure_tick) and
                failure_tick['row']['epoch']==exc.evidence['epoch']==epoch and matches(failure_tick['row']['sim_ns'],exc.evidence['stamp_ns'],'stamp_ns'))
        if isinstance(exc,SensorAmbiguity):record['sensor_join_failure']=exc.evidence
        if isinstance(exc,TickContractError):record['pipeline_contract_failure']=dict(errors=exc.errors,evidence=exc.evidence)
        if not hasattr(self, 'capture_failures'):
            self.capture_failures = []
        self.capture_failures.append(record)
        return record

    @staticmethod
    def attempt_error(exc, records, motion_epochs, current_epoch):
        # Only the cleanup cancellation caused by this exact recorded epoch
        # may refer back to the original error. Unrelated epoch/transport/scene
        # exceptions remain the primary failure and never become retry hints.
        result = dict(error=repr(exc))
        if isinstance(exc, CancelledGeneration):
            for record in reversed(records):
                cancel = record['cancellation']
                response = cancel.get('result') or {}
                if (record['epoch'] in motion_epochs and cancel.get('completed') and
                        response.get('epoch') == current_epoch and current_epoch is not None):
                    result.update(error=record['error'], cleanup_error=repr(exc),
                                  original_capture_error_preserved=True)
                    break
        return result

    def wait_tick_clock(self,stamp,epoch):
        deadline=time.monotonic()+2.
        while True:
            self.generation.require(epoch,stamp)
            current=self.node.get_clock().now().nanoseconds
            self.generation.require(epoch,stamp)
            if matches(current,stamp,'stamp_ns'):return
            if current>stamp and not matches(current,stamp,'stamp_ns'):raise TickContractError(['ROS clock leads the held physical tick'])
            if time.monotonic()>=deadline:raise TickContractError(['ROS clock did not join the held physical tick'])
            if self.stop.wait(.002):raise CancelledGeneration('Collector stopped during clock join')

    def validate_command(self,candidate,state,previous):
        timing=dict(validity_queries=0,validity_query_ms=0.)
        def check(q):
            self.generation.require(candidate.epoch,candidate.stamp_ns)
            start=time.monotonic_ns();timing['validity_queries']+=1
            try:return self.node._check_state_valid_with_contacts(dict(zip(JOINTS,q)))
            finally:timing['validity_query_ms']+=milliseconds(start,time.monotonic_ns())
        try:return final_command(candidate,state,previous,check,[JOINT_LIMITS[n] for n in JOINTS])
        except Exception as exc:
            exc.wall_profile_geometry=timing
            raise

    def reject_quality(self,kind,evidence,snapshots=None):
        if snapshots is None:snapshots=[self.held_snapshot(self.generation.epoch)]
        contract=require_snapshots(snapshots)
        raise QualityReject(kind,evidence,contract=contract)

    def capture(self):
        while not self.stop.is_set():
            epoch=None;failure_tick=None;profile=None;profile_frozen=False;profile_retained=False;keep_profile=True
            read_start=time.monotonic_ns()
            profile=dict(profile_id='unjoined:'+str(read_start),epoch=self.generation.epoch,sim_ns=None,
                capture_start_ns=read_start,recorded=False,stage='sensor_join',outcome='pending')
            try:
                read_start=time.monotonic_ns()
                try:stamp,entry=self.sensors.read(timeout=30)
                except TimeoutError:
                    if self.servo.plan or self.servo.grip_plan:raise TimeoutError('Motion stalled: no synchronized camera/state tick within 30 wall seconds')
                    keep_profile=False;continue
                read_end=time.monotonic_ns()
                epoch=entry['tick'][0]['epoch']
                self.generation.require(epoch,stamp)
                with self.generation.own('capture/dispatch'):
                    self.generation.require(epoch,stamp)
                    joint,received=entry['joint'];snapshot=entry['tick'][0]
                    profile=profile_tick(snapshot,entry,read_start,read_end)
                    profile.update(stage='observation_contract',outcome='pending')
                    preparation_start=time.monotonic_ns()
                    mapped=dict(zip(joint.name,joint.position));mapped_v=dict(zip(joint.name,joint.velocity))
                    state=[mapped[n] for n in JOINTS];velocity=[mapped_v[n] for n in JOINTS]
                    row=dict(tick_contract_version=1,observation_epoch=snapshot['epoch'],expert_epoch=epoch,epoch=epoch,
                        invalid_through_ns=self.generation.floor_ns,sim_ns=stamp,wall_ns=time.monotonic_ns(),
                        joint_stamp_ns=ns(joint.header.stamp),joint_receive_ns=received,
                        wrist_stamp_ns=ns(entry['wrist'][0].header.stamp),top_stamp_ns=ns(entry['top'][0].header.stamp),
                        wrist_receive_ns=entry['wrist'][1],top_receive_ns=entry['top'][1],
                        state=state,velocity=velocity,state_source=snapshot['state_source'],
                        physics_state=snapshot['state'],physics_velocity=snapshot['velocity'],
                        physics_drive_targets=snapshot['drive_targets'],pose_stamp_ns=snapshot['sim_ns'],
                        physics_time_s=snapshot['physics_time_s'],physical_bodies=copy.deepcopy(snapshot['bodies']),
                        episode_id=getattr(self,'current_episode_id',None),
                        camera_identity={key:copy.deepcopy(entry[key][2]) for key in ('wrist','top')},
                        camera_layout={k:dict(encoding=entry[k][0].encoding,width=entry[k][0].width,height=entry[k][0].height,step=entry[k][0].step,bytes=len(entry[k][0].data)) for k in ('wrist','top')})
                    if not hasattr(self,'tick_audit'):self.tick_audit=TickContract()
                    contract_start=time.monotonic_ns()
                    self.tick_audit.observe(row)
                    self.wait_tick_clock(stamp,epoch)
                    profile['contract_validation_ms']+=milliseconds(contract_start,time.monotonic_ns())
                    self.node._joint_states_callback(joint);self.update_pose_cache(snapshot)
                    for plan in (self.servo.plan,self.servo.grip_plan):
                        if plan:self.generation.require(plan['epoch'])
                    physical_failure=None
                    profile['capture_preparation_ms']=milliseconds(preparation_start,time.monotonic_ns())-profile['contract_validation_ms']
                    profile['stage']='expert_dart_compute'
                    compute_start=time.monotonic_ns()
                    try:
                        clean,executed,active,phase,reference,command_clock,completed,advance,candidate=self.servo.tick(stamp,state,snapshot['drive_targets'])
                    except ConvergenceFailure as exc:
                        physical_failure=exc;reference=list(self.servo.reference)
                        clean=corrective_action(state,reference);executed=list(snapshot['drive_targets'])
                        active=True;phase=self.servo.phase;command_clock=self.node.get_clock().now().nanoseconds
                        completed=[];advance=False
                        candidate=TargetCandidate(epoch,stamp,tuple(executed),(0.,)*5,
                            tuple(e-c for e,c in zip(executed,clean)),(0.,)*6,tuple(executed),planner_reference=tuple(executed))
                    profile['expert_dart_compute_ms']=milliseconds(compute_start,time.monotonic_ns())
                    profile['stage']='execution_geometry'
                    geometry=self.validate_command(candidate,state,snapshot['drive_targets'])
                    profile['validity_queries']=geometry['query_timing']['queries']
                    profile['validity_query_ms']=geometry['query_timing']['total_query_wall_s']*1000
                    self.generation.require(epoch,stamp)
                    profile['stage']='command_publish_ack'
                    command_start=time.monotonic_ns()
                    self.servo.publish_reference(executed)
                    self.generation.require(epoch,stamp)
                    self.command_pub.publish(String(data=json.dumps(dict(epoch=epoch,observation_stamp_ns=stamp,command=executed,advance=advance))))
                    ack=self.sensors.read_ack(epoch,stamp)
                    profile['command_publish_ack_ms']=milliseconds(command_start,time.monotonic_ns())
                    self.generation.require(epoch,stamp)
                    row.update(action=clean,executed_action=executed,expert_reference=reference,
                        expert_observation_stamp_ns=stamp,command_clock_ns=command_clock,executing=active,phase=phase,
                        planned_noise_offset=list(candidate.noise_offset),command_filter_delta=list(candidate.filter_delta),
                        command_velocity=list(candidate.velocities),planner_execution_reference=list(candidate.planner_reference),
                        dart_offset=[e-r for e,r in zip(executed,candidate.planner_reference)],noise_scale=candidate.noise_scale,
                        execution_geometry=geometry,recovery=self.servo.params.get('recovery','none'),
                        controller_reference=self.sensors.reference_at(epoch,stamp),declared_application_stamp_ns=stamp,**ack)
                    selected=self.node._selected_object
                    if selected in snapshot['bodies']:
                        body=snapshot['bodies'][selected]
                        row['target_object_pose']=dict(zip(('x','y','z'),body['position']))
                        row['target_body']=body;row['target_cup']=snapshot['bodies']['cup_'+self.servo.params['color']]
                    profile['stage']='final_contract'
                    contract_start=time.monotonic_ns()
                    proof=self.tick_audit.finish(row)
                    profile['contract_validation_ms']+=milliseconds(contract_start,time.monotonic_ns())
                    self.servo.commit_command(candidate,epoch,stamp)
                    profile.update(executing=active,phase=phase,recorded=False)
                    checked=copy.deepcopy(snapshot);checked['tick_contract']=proof
                    with self.writer_lock:
                        if self.writer:
                            profile['recorded']=True;profile['stage']='image_decode'
                            decode_start=time.monotonic_ns()
                            images={key:Sensors.image(entry[key][0]).copy() for key in ('wrist','top')}
                            profile['image_decode_ms']=milliseconds(decode_start,time.monotonic_ns())
                        profile.update(stage='convergence' if physical_failure else 'complete',
                            outcome='quality_reject' if physical_failure else 'captured')
                        profile['capture_processing_ms']=milliseconds(read_end,time.monotonic_ns())
                        profile['tick_total_ms']=milliseconds(profile.get('native_step_start_ns') or read_end,time.monotonic_ns())
                        # The row gets its own finalized capture profile. Encoder/queue
                        # completion is persisted separately, never by mutating this row.
                        row['wall_profile']=copy.deepcopy(profile);profile_frozen=True
                        if self.writer:
                            self.writer.add(row,images)
                            self.episode_snapshots.append(checked)
                    if physical_failure is not None:profile['error']=repr(physical_failure)
                    if not hasattr(self,'wall_profiles'):self.wall_profiles=[]
                    self.wall_profiles.append(copy.deepcopy(profile));profile_retained=True
                    with self.tick_condition:
                        self.last_processed_ns=stamp;self.last_snapshot=checked;self.tick_condition.notify_all()
                    if physical_failure is not None:
                        failure_tick=proof
                        raise physical_failure
                    for event in completed:
                        self.node._set_motion_status(dict(ok=True,outcome='physical_sim_tick_converged',epoch=epoch),event)
                        event.set()
            except OwnershipStalled as exc:
                profile.update(outcome='terminal',error=repr(exc));self.abort_stalled_owner(exc,'capture/dispatch');return
            except CancelledGeneration as exc:
                profile.update(outcome='cancelled',error=repr(exc));continue
            except Exception as exc:
                profile.update(outcome='failed',error=repr(exc))
                profile.update(getattr(exc,'wall_profile_geometry',{}))
                if self.generation.stalled:
                    self.abort_stalled_owner(OwnershipStalled(self.generation.stalled),'capture/dispatch');return
                with self.generation.own('collector motion'):
                    if epoch is not None and epoch!=self.generation.epoch:continue
                    if epoch is None:epoch=self.generation.epoch
                    record=self.record_capture_failure(exc,epoch,failure_tick)
                    record['wall_profile']=copy.deepcopy(profile)
                    message=record['error'];self.node._qs_abort_evt.set()
                    if isinstance(exc,ConvergenceFailure) and record['validated_sim_tick']:
                        with self.servo.lock:
                            done=[p['done'] for p in (self.servo.plan,self.servo.grip_plan) if p]
                            for event in done:event.verdict=dict(ok=False,outcome=exc.kind,epoch=epoch,msg=message)
                        record['cancellation']['requested']=True
                        try:
                            cancelled=self.timeline(False);record['cancellation'].update(completed=True,result=cancelled)
                            if not cancelled.get('held') or cancelled.get('failure'):raise RuntimeError('Native cancellation did not return a healthy held state')
                        except Exception as pause_error:
                            record['cancellation']['error']=repr(pause_error);self.failure=repr(pause_error)
                        for event in done:event.set()
                    else:self.abort_stalled_owner(exc,'capture pipeline fault')
                    if getattr(self.node,'_motion_event',None):self.node._motion_event.set()
                    if self.failure:return

            finally:
                if keep_profile and not profile_retained:
                    if not profile_frozen:
                        profile['capture_processing_ms']=milliseconds(profile['capture_start_ns'],time.monotonic_ns())
                        profile['tick_total_ms']=milliseconds(profile.get('native_step_start_ns') or read_start,time.monotonic_ns())
                    if not hasattr(self,'wall_profiles'):self.wall_profiles=[]
                    self.wall_profiles.append(copy.deepcopy(profile))

    def advance_ticks(self, count):
        snapshots = []
        for _ in range(count):
            with self.tick_condition:
                previous = self.last_processed_ns
                epoch = self.generation.epoch
            self.timeline(True)
            with self.tick_condition:
                if not self.tick_condition.wait_for(lambda: (self.last_snapshot and self.last_snapshot['epoch'] == epoch and self.last_processed_ns > previous) or self.failure, timeout=frame_capacity_seconds()):
                    raise TimeoutError('Held-frame capacity exceeded '+str(frame_capacity_seconds())+' wall seconds; query/ACK/clock/recording did not retire')
                if self.failure:
                    if isinstance(self.failure,BaseException):raise self.failure
                    raise RuntimeError(self.failure)
                snapshots.append(self.last_snapshot)
        return snapshots

    def held_snapshot(self, epoch):
        with self.tick_condition:
            snapshot=copy.deepcopy(self.last_snapshot)
        if snapshot is None:raise RuntimeError('No measured snapshot for recovery')
        self.generation.require(epoch,snapshot['sim_ns'])
        if snapshot['epoch']!=epoch:raise RuntimeError('Recovery snapshot belongs to another epoch')
        require_snapshots([snapshot])
        return snapshot

    def reset_release(self, object_name):
        # Release a failed physical grasp before planning any reset arm motion.
        self.node._select_grasp_object(object_name)
        opening,_=self.node._gripper_angle_for_object(object_name)
        if not self.servo.gripper(opening,1.5,True,phase='reset_open'):
            raise RuntimeError('Reset gripper did not open')
        self.node._detach_lego_sync()
        snapshot=self.advance_ticks(1)[0]
        self.sync_scene(snapshot=snapshot)

    def regrasp_prefix(self, params, epoch, result):
        """Execute the miss while labels follow the separate true-target plan."""
        self.generation.require(epoch)
        before=self.held_snapshot(epoch)
        name=params['object'];bbox=self.node._lookup_bbox(name)
        if not bbox:raise ValueError('Re-grasp target has no dimensions')
        with self.node.objects_lock:pose=dict(self.node.objects_data[name])
        geometry=recovery_geometry(pose,bbox,params['regrasp'],
            current_wrist_roll=before['state'][4],tcp_clearance=self.node._tcp_clearance_var.get()/1000,
            approach_height=params['approach_height_m'],approach_offset=params['approach_offset_m'])
        report=dict(epoch=epoch,episode_id=self.current_episode_id,kind='regrasp',miss_confirmed=False,geometry=geometry,steps=[])
        result['recovery_report']=report
        if geometry['yaw'] is None:self.reject_quality('planner_reject',dict(stage='regrasp_ik',geometry=geometry))
        # The same target-removal barrier and shared collision validator used
        # by Quickstart; both clean and executed paths are checked separately.
        self.node._remove_single_lego_from_world(name)
        selected={}
        for stage,solutions in geometry['solutions'].items():
            for target in solutions:
                self.generation.require(epoch)
                valid,contacts=self.node._check_state_valid_with_contacts(target)
                if valid:selected[stage]=dict(target);break
            if stage not in selected:
                self.reject_quality('planner_reject',dict(stage=stage,reason='all regrasp IK candidates collide'))
        duration=2.5
        def tag(value):
            with self.node._motion_guard():self.node._last_motion_tag=value
        def arm(stage,clean_key,executed_key):
            self.generation.require(epoch)
            snapshot=self.held_snapshot(epoch)
            current=dict(zip(JOINTS[:5],snapshot['state'][:5]))
            clean_target=dict(selected[clean_key]);executed_target=dict(selected[executed_key])
            clean_path=self.node._plan_linear_joint_path(current,clean_target,duration/params['speed_scale'])
            executed_path=self.node._plan_linear_joint_path(current,executed_target,duration/params['speed_scale'])
            self.generation.require(epoch)
            if clean_path is None or executed_path is None:
                self.reject_quality('planner_reject',dict(stage=stage,reason='regrasp path collision'))
            tag(stage)
            event=self.node._new_motion_event();self.node._begin_motion(event)
            operation=self.node._operation(clean_target,event,duration_s=duration)
            self.servo.execute(clean_path,event,operation=operation,executed_trajectory=executed_path,hold_executed=True)
            if not self.node._wait_motion_event(event,60.) or not event.verdict or not event.verdict.get('ok'):
                raise RuntimeError('Recovery arm execution failed')
            self.generation.require(epoch)
            report['steps'].append(dict(stage=stage,clean_target=clean_target,executed_target=executed_target,
                start_sim_ns=snapshot['sim_ns'],end_sim_ns=self.held_snapshot(epoch)['sim_ns']))
        opening,closing=self.node._gripper_angle_for_object(name)
        def grip(stage,target):
            tag(stage)
            if not self.servo.gripper(target,3.,True,phase=stage):raise RuntimeError('Recovery gripper execution failed')
            self.generation.require(epoch)
            report['steps'].append(dict(stage=stage,target=target,sim_ns=self.held_snapshot(epoch)['sim_ns']))
        grip('regrasp_miss_open',opening)
        arm('regrasp_miss_approach','true_approach','miss_approach')
        arm('regrasp_miss_descent','true_final','miss_final')
        before=self.held_snapshot(epoch)
        grip('regrasp_miss_close',closing) # Deliberately no MoveIt attach on this first attempt.
        arm('regrasp_miss_lift','true_final','miss_lift')
        after=self.held_snapshot(epoch)
        with self.writer_lock:
            window=[s for s in self.episode_snapshots if before['sim_ns']<=s['sim_ns']<=after['sim_ns']]
        require_snapshots(window)
        detection=detect_miss(window,name,closing)
        report['window_frame_sha256s']=[s['tick_contract']['row_sha256'] for s in window]
        report['detection']=detection
        report['measured_before']={k:v for k,v in before.items() if k!='tick_contract'}
        report['measured_after_lift']={k:v for k,v in after.items() if k!='tick_contract'}
        # Even an accidentally successful first grasp is released before the
        # quality refusal; it must not leave a held block for the next reset.
        grip('regrasp_reopen',opening)
        report['miss_confirmed']=detection['confirmed']
        if not detection['confirmed']:
            self.reject_quality('regrasp_miss_not_confirmed',detection,window)
        current=self.held_snapshot(epoch)
        self.sync_scene(snapshot=current)
        report['scene_after_reopen']=self.scene_pose_check
        report['corrective_attempt']='shared Quickstart cycle from fresh measured poses'

    def request_home(self):
        with self.node._motion_guard():
            self.node._last_motion_tag = 'grasp_home_data'
        return self.node._cmd_grasp_home_data()

    def hold_home(self):
        from so_arm101_control.grasp_geometry import _grasp_home_data_target
        target = _grasp_home_data_target()
        with self.generation.own('nominal home hold'):
            epoch = self.generation.epoch
            self.generation.require(epoch)
            with self.servo.lock:
                if self.servo.plan is not None or self.servo.grip_plan is not None:
                    raise RuntimeError('Cannot install home hold while motion is active')
                reference = (list(self.servo.reference) if self.servo.reference is not None else
                             [self.node._actual_positions[name] for name in JOINTS])
                for i,name in enumerate(JOINTS[:5]):reference[i] = target[name]
                self.servo.reference, self.servo.reference_epoch = reference, epoch
                self.servo.previous_executed = None
        return target


    def restore_failed_episode_home(self):
        try:return self._restore_failed_episode_home()
        except Exception as exc:
            self.abort_stalled_owner(exc,'failed episode home reset')
            raise

    def _restore_failed_episode_home(self):
        from so_arm101_control.grasp_geometry import _grasp_home_data_target
        with self.writer_lock:
            if self.writer is not None:raise RuntimeError('Cannot restore home while recording an episode')
        baseline=self.safe_home_baseline
        validate_baseline(baseline,_grasp_home_data_target())
        retirement=self.workers.retire()
        if not retirement.get('passed'):raise RuntimeError('Failed episode workers did not retire before reset')
        cleared=self.scene_reset.clear()  # MoveIt, including attachments, is the authority.
        if not cleared.get('verified'):raise RuntimeError('Failed episode scene clear did not verify')
        with self.motion_effect(self.generation.epoch):
            self.node._attached_lego_name=None;self.node._attached_lego_tcp_offset=None
        restored=self.restore_physical('import collection_runtime as c\nr=c.runtime()\nresult=r.restore('+
            repr(baseline['snapshot'])+',arm=True,restore_drives=True)')
        actual=restored['snapshot'];wanted=baseline['snapshot']
        for key,expected in (('state',wanted['state']),('velocity',[0.]*6),('drive_targets',wanted['drive_targets'])):
            if not matches(actual.get(key),expected,'float32_readback'):
                raise RuntimeError('Failed episode home restore readback differs: '+key)
        with self.generation.own('restored home reference'):
            self.generation.require(restored['epoch'])
            with self.servo.lock:
                self.servo.reference=list(wanted['drive_targets']);self.servo.reference_epoch=restored['epoch']
                self.servo.executed_hold=None;self.servo.previous_executed=None
        samples=self.advance_ticks(6)
        check=initial_arm(samples,baseline['home_target'])
        if not check['contract_passed']:raise TickContractError(check['contract']['errors'],dict(contract=check['contract'],snapshots=samples))
        if not check['passed']:raise RuntimeError('Failed episode restored home did not settle: '+repr(check))
        self.sync_scene(snapshot=samples[-1])
        return dict(passed=True,unrecorded=True,before_planning=True,source_episode_id=baseline['source_episode_id'],
                    source_baseline_sha256=baseline['snapshot_sha256'],retirement=retirement,scene_clear=cleared,
                    restore=restored,settle=check,scene_verify=self.scene_pose_check)

    def settle_home(self, label):
        target = self.hold_home()
        samples=[];checks=[]
        for _ in range(round(30*POLICY['home_settle_sim_seconds'])):
            sample=self.advance_ticks(1)[0];samples=(samples+[sample])[-6:]
            check=initial_arm(samples,[target[name] for name in JOINTS[:5]])
            if not check['contract_passed']:raise TickContractError(check['contract']['errors'],dict(contract=check['contract'],snapshots=samples))
            if len(samples)==6:
                checks.append(dict(sim_ns=sample['sim_ns'],**check))
                if check['passed']:
                    return dict(label=label,passed=True,target=target,checks=checks,
                                settled_sim_seconds=(len(checks)+5)/30,snapshots=samples)
        self.reject_quality('home_unstable',dict(label=label,target=target,checks=checks),samples)

    def camera_graph(self):
        result={}
        for key,topic in (('wrist','/wrist_camera_rgb_sim'),('top','/workspace_camera_sim')):
            endpoints=self.node.get_publishers_info_by_topic(topic)
            if len(endpoints)!=1:raise RuntimeError('Camera topic requires exactly one publisher: '+topic+' count='+str(len(endpoints)))
            endpoint=endpoints[0];gid=bytes(endpoint.endpoint_gid)
            if len(gid)!=24 or not any(gid):raise RuntimeError('Camera graph identity is unverified: '+topic)
            result[key]=dict(topic=topic,endpoint_gid=gid.hex(),node_name=endpoint.node_name,
                node_namespace=endpoint.node_namespace,topic_type=endpoint.topic_type,
                qos=dict(depth=endpoint.qos_profile.depth,reliability=int(endpoint.qos_profile.reliability),durability=int(endpoint.qos_profile.durability)))
        return dict(passed=True,endpoints=result,scope='graph endpoint GUIDs; separate namespace from packet RMW IDs')

    def wait_ready(self):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.failure:
                raise RuntimeError(self.failure)
            if self.node.objects_data and len(self.node._drop_data) == 3 and self.servo.last_sim_ns and self.sensors.reference:
                for client in (self.node.plan_client, self.node.validity_client, self.node._apply_scene_client):
                    if not client.wait_for_service(timeout_sec=5):
                        raise RuntimeError('MoveIt service unavailable')
                self.camera_graph()
                return
            self.stop.wait(.1)
        raise TimeoutError('Collection ROS graph did not become ready')

    def scene(self, params, baseline):
        self.timeline(False)
        self.scene_clear_check = self.scene_reset.clear()
        mcp('run_python', {'code': 'import collection_runtime as c\nc.runtime().require_scene()\nresult=True'})
        if self.args.replay_receipt:
            replay = json.loads(self.args.replay_receipt.read_text())
            restored = self.restore_physical('import collection_runtime as c\nresult=c.runtime().restore(' +
                repr(replay['initial_physical']) + ',arm=True)')
            with self.generation.own('replay initial conditions'):
                self.update_pose_cache(restored['snapshot'])
                self.servo.reference = list(replay['initial_expert_reference'])
                self.servo.reference_epoch = self.generation.epoch
            self.scene_restore = restored
            cups = replay['cup_randomization']
        else:
            cups = self.randomize_scene(params, baseline)
        # Apply the same unrecorded reset barrier for normal collection/replay.
        # It gives physical joint/pose streams and TF fresh clocked samples.
        self.episode_start_physical = self.advance_ticks(2)[-1]
        self.sync_scene()
        return cups

    def randomize_scene(self, params, baseline):
        # Restore a common baseline so seed reproduction does not depend on the previous episode.
        code = 'from so_arm101_dt.extension import get_instance, CUP_LAYOUT\nfrom pxr import Gf\nimport random\nimport numpy as np\ni=get_instance()\n'
        code += 'CUP_LAYOUT.clear()\nCUP_LAYOUT.update(' + repr(self.baseline_cup_layout) + ')\n'
        code += 'baseline=' + repr(baseline) + '\n'
        code += "for name, pos in baseline.items():\n    i._set_obj_prim_pose(i._get_prim_path(name), Gf.Vec3d(*pos), (0.,0.,0.))\n    p=__import__('omni.usd', fromlist=['']).get_context().get_stage().GetPrimAtPath(i._get_prim_path(name))\n    from pxr import UsdPhysics\n    rb=UsdPhysics.RigidBodyAPI(p)\n    rb.GetVelocityAttr().Set(Gf.Vec3f(0.,0.,0.))\n    rb.GetAngularVelocityAttr().Set(Gf.Vec3f(0.,0.,0.))\n"
        code += 'random.seed(' + str(params['seed']) + ')\nnp.random.seed(' + str(params['seed'] % 2**32) + ')\nresult=True'
        self.mutate_kit('run_python', {'code': code})
        cups = self.mutate_kit('soarm101.randomize_cups', dict(seed=params['cup_seed'], **params['cup_sampling']))
        self.mutate_kit('run_python', {'code': 'from so_arm101_dt.extension import get_instance\nimport random\nimport numpy as np\nrandom.seed(' + str(params['seed']) + ')\nnp.random.seed(' + str(params['seed'] % 2**32) + ')\nresult=get_instance()._cmd_randomize_object_poses(required_visible_color=' + repr(params['color']) + ', required_visible_count=2)'})
        # Resume only to get the post-teleport poses, then freeze while scene updates/planning run.
        self.mutate_kit('run_python', {'code': 'import collection_runtime as c\nresult=c.runtime().reset_velocities()'})
        self.advance_ticks(6)
        self.scene_restore = self.restore_physical('import collection_runtime as c\nr=c.runtime()\ns=r.snapshot()\nresult=r.restore(s,arm=True)')
        return cups

    def update_pose_cache(self, snapshot):
        objects, cups = physical_pose_caches(snapshot,primitive_cups=self.scene_reset.primitive_cups)
        with self.node.objects_lock, self.node._drop_lock:
            self.node._physical_pose_cache = True
            self.node.objects_data = objects
            self.node._drop_data = cups

    def add_cup_collisions(self,cups_dict=None):
        from geometry_msgs.msg import Pose
        from shape_msgs.msg import SolidPrimitive
        with self.node._motion_guard():
            if cups_dict is None:
                with self.node._drop_lock:items=dict(self.node._drop_data)
            else:items=dict(cups_dict)
            scene=self.scene_reset.diff()
            for name,drop in items.items():
                co=self.scene_reset.msg.Object();co.header.frame_id='base';co.id='cup_'+name;co.operation=co.ADD
                for shape in primitive_world_poses(drop):
                    box=SolidPrimitive();box.type=SolidPrimitive.BOX;box.dimensions=shape['dimensions']
                    pose=Pose()
                    for key,value in zip(('x','y','z'),shape['position']):setattr(pose.position,key,value)
                    for key,value in zip(('w','x','y','z'),shape['orientation']):setattr(pose.orientation,key,value)
                    co.primitives.append(box);co.primitive_poses.append(pose)
                scene.world.collision_objects.append(co)
            response=self.scene_reset.apply(scene)
            if response is None or not response.success:raise RuntimeError('Primitive cup scene update refused')
            self.node._cup_collision_names=[co.id for co in scene.world.collision_objects]
            self.node._acm_allow_ground_rest(self.node._cup_collision_names)

    def sync_scene(self,snapshot=None):
        snapshot=self.episode_start_physical if snapshot is None else snapshot
        self.update_pose_cache(snapshot)
        cleared = self.scene_reset.clear()
        self.node._cmd_toggle_ground_plane()
        self.node._add_cup_collision_objects(cups_dict=dict(self.node._drop_data))
        self.scene_reset.rebuild_legos(snapshot)
        world_ids = ['ground_plane'] + list(self.node._cup_collision_names) + list(self.node._lego_collision_names)
        with self.node._motion_guard():
            self.scene_reset.set_policy(world_ids)
        deadline = time.monotonic() + 15
        while True:
            check = self.scene_reset.verify(snapshot, JOINTS)
            if check['passed']:
                self.scene_pose_check = dict(check, cleared=cleared)
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f'Full planning scene reset verification failed: {check}')
            self.stop.wait(.05)

    def failed_finalization(self,receipt,error,motion_epochs,duplicate_start,rerender_start,stale_camera_start,profiles,writes):
        original=getattr(self.sensors,'failure',None) or receipt.get('error') or self.failure or error
        self.failure=self.failure or repr(original)
        receipt.update(accepted=False,error=repr(original) if isinstance(original,BaseException) else str(original),
            disposition=dict(kind='terminal',reason='attempt_finalization'),motion_epochs=sorted(motion_epochs))
        receipt.setdefault('finalization_errors',[]).append(dict(error=repr(error),exception_type=type(error).__name__,proofs=getattr(error,'proofs',None)))
        receipt['wall_tick_profile']=dict(failed=True,ticks=profiles,async_writes=writes)
        if isinstance(original,SensorAmbiguity):
            receipt.update(exception_type='SensorAmbiguity',original_sensor_error_preserved=True,sensor_join_failure=original.evidence)
        receipt.update(self.sensors.camera_diagnostics(motion_epochs,duplicate_start,rerender_start,stale_camera_start))
        try:self.persist_terminal_fault(original,'attempt finalization')
        except Exception as publication_error:receipt['terminal_publication_error']=repr(publication_error)

    def run(self):
        args = self.args
        root = confined(args.output)
        if (root / 'superseded.json').exists():
            raise ValueError('Superseded diagnostic run requires a fresh output directory')
        if args.resume:collection_status.require_finalized(root)
        root.mkdir(parents=True, exist_ok=args.resume)
        with self.status_lock:self.finalization_certificate=collection_status.begin(root)
        previous = json.loads((root / 'run.json').read_text()) if args.resume else None
        if previous and previous.get('diagnostic', False) != args.diagnostic:
            raise ValueError('Resume diagnostic retention mode differs from the existing run')
        wait_policy = dict(rtf_floor=args.wait_rtf_floor, sim_margin_s=5., wall_margin_s=frame_capacity_seconds()+30.,
                           episode_wall_min_s=args.episode_deadline)
        if previous and previous.get('motion_wait_policy') != wait_policy:
            raise ValueError('Resume motion wait policy differs from the existing run')
        if previous and previous.get('quality_policy') != POLICY:
            raise ValueError('Quality policy differs; use a fresh output directory')
        receipts = []
        ledger_rows = {}
        if previous:
            for trial, path in enumerate(sorted(root.glob('attempt-*'))):
                if path.name != f'attempt-{trial:04d}':
                    raise ValueError('Cannot resume a ledger with missing attempt directories')
                receipt = json.loads((path / 'receipt.json').read_text())
                if receipt['accepted']:
                    rows = [json.loads(line) for line in (path / 'frames.jsonl').read_text().splitlines()]
                    ledger_rows[trial]=rows
                    if not timing_audit(rows, 30)['passed'] or not all((path / f'{k}.mp4').exists() for k in ('wrist', 'top')):
                        raise ValueError(f'Cannot resume invalid accepted attempt {path}')
                receipts.append(receipt)
            logged = [json.loads(line) for line in (root / 'episodes.jsonl').read_text().splitlines()]
            if logged != receipts:
                raise ValueError('Resume attempt log differs from the receipt ledger')
        self.safe_home_baseline=copy.deepcopy(previous.get('safe_home_baseline')) if previous else None
        self.needs_home_restore=bool(receipts and receipts[-1].get('disposition',{}).get('kind')=='quality_reject')
        streak = FailureStreak(receipts)
        if streak.stopped:
            raise RuntimeError('Previous receipt ledger requires a stop; review before a new run')
        if previous and previous.get('regrasp_only',False)!=args.regrasp_only:
            raise ValueError('Resume recovery mode differs')
        attempted, accepted = progress(receipts,args.seed,args.regrasp_only,lambda r:ledger_rows[r['params']['trial']])
        if accepted > args.episodes:
            raise ValueError('Requested target is smaller than the accepted checkpoint')
        old_summary = json.loads((root / 'summary.json').read_text()) if (root / 'summary.json').exists() else {}
        # Older runs did not checkpoint elapsed time. Their last receipt mtime records
        # the end of active work; exclude the intervening shutdown/disk wait.
        previous_wall = old_summary.get('wall_seconds', max(0., (root / 'episodes.jsonl').stat().st_mtime - previous['started']) if previous and receipts else 0.)
        disk_wait = old_summary.get('disk_wait_seconds', 0.)
        clock_config = self.mutate_kit('run_python', {'code': (REPO / 'linux-env/scripts/collect/configure_clock.py').read_text() + '\nresult=configure(30)'})
        transport_config = self.mutate_kit('run_python', {'code': 'import sys\nsys.path.insert(0,' + repr(str(REPO / 'linux-env/scripts/collect')) + ')\nimport collection_runtime as c\nresult=c.install()'})
        self.generation.install(transport_config)
        expected_bodies = set(transport_config['local_corners']) | {'cup_red','cup_green','cup_blue'}
        offsets = transport_config['effective_shape_offsets']
        if set(offsets) != expected_bodies:
            raise RuntimeError('Effective shape offsets missing bodies')
        for name, values in offsets.items():
            kind = 'cup' if name.startswith('cup_') else 'block'
            expected_shapes=len(PRIMITIVES) if kind=='cup' else 1
            if any(len(values[key+'_m'])!=expected_shapes for key in ('contact','rest')):
                raise RuntimeError('Physical primitive shape count differs from the shared geometry: '+name)
            for key in ('contact','rest'):
                expected = PROFILE['expected_offsets_m'][kind+'_'+key]
                if not values[key+'_m'] or any(abs(value-expected)>1e-7 for value in values[key+'_m']):
                    raise RuntimeError('Effective shape offsets differ from measured profile metadata: '+name)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            counts = mcp('run_python', {'code': 'import collection_runtime as c\nr=c.runtime()\nresult=[r.node.count_subscribers("/joint_states_measured"),r.node.count_subscribers("/collection_tick"),r.node.count_subscribers("/collection_command_ack")]'})
            if counts[0] >= 2 and counts[1] >= 1 and counts[2] >= 1 and self.command_pub.get_subscription_count() >= 1:
                break
            self.stop.wait(.1)
        else:
            raise TimeoutError('Physical command/feedback ROS graph was not discovered')
        self.advance_ticks(1)
        self.wait_ready()
        self.episode_start_physical = self.last_snapshot
        self.sync_scene()
        atomic_json(root / 'first-reset-scene.json',self.scene_pose_check)
        self.timeline(False)
        if previous:
            baseline = previous['baseline']
            self.baseline_cup_layout = previous['baseline_cup_layout']
        elif args.replay_receipt:
            replay_run = json.loads((args.replay_receipt.parent.parent / 'run.json').read_text())
            baseline = replay_run['baseline']
            self.baseline_cup_layout = replay_run['baseline_cup_layout']
        else:
            initial = mcp('run_python', {'code':
                'from so_arm101_dt.extension import get_instance, CUP_LAYOUT\nfrom pxr import UsdGeom, Usd\nimport omni.usd\n'
                'i=get_instance()\nst=omni.usd.get_context().get_stage()\n'
                "result={'objects': {p.GetName(): list(UsdGeom.Xformable(st.GetPrimAtPath(i._get_prim_path(p.GetName()))).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()) for p in st.GetPrimAtPath('/World/Objects').GetChildren() if p.GetName() != 'PhysicsMaterial'}, 'cup_layout': dict(CUP_LAYOUT)}"})
            baseline = initial['objects']
            for name, position in baseline.items():
                bbox = self.node._lookup_bbox(name)
                if not bbox:
                    raise ValueError(f'Missing dimensions for baseline object {name}')
                position[2] = self.node._ground_z_var.get() + float(bbox['sz']) / 2
            self.baseline_cup_layout = initial['cup_layout']
        tracked_sources = ['linux-env/scripts/collect/image_abi.py','linux-env/scripts/collect/image_transport_abi.json','linux-env/scripts/collect/ros_lifecycle.py','linux-env/scripts/collect/protocol.py','linux-env/scripts/collect/wall_profile.py','linux-env/scripts/collect/image_take.c','linux-env/scripts/collect/image_transport.py','isaac/exts/soarm101-dt/so_arm101_dt/extension.py','isaac/exts/soarm101-dt/so_arm101_dt/primitive_colliders.py','linux-env/scripts/collect/cup_geometry.py','linux-env/scripts/collect/pose_math.py','linux-env/scripts/collect/collection_status.py','linux-env/scripts/collect/nvml_processes.py','linux-env/scripts/collect/renderer_device.py','linux-env/scripts/collect/wall_profile.py','linux-env/scripts/collect/gpu_device.py','linux-env/scripts/collect/safe_home.py','linux-env/scripts/collect/frame_join.py','linux-env/scripts/collect/comparisons.py','linux-env/scripts/collect/collect.py', 'linux-env/scripts/collect/contract.py',
                           'linux-env/scripts/collect/checkpoint.py',
                           'linux-env/scripts/collect/episode_policy.py',
                           'linux-env/scripts/collect/dart.py',
                           'linux-env/scripts/collect/recovery.py',
                           'linux-env/scripts/collect/query_service.py',
                           'linux-env/scripts/collect/tick_contract.py',
                           'linux-env/scripts/collect/command_geometry.py',
                           'vla_SO-ARM101/src/so_arm101_moveit_config/config/joint_limits.yaml',
                           'linux-env/scripts/collect/storage.py',
                           'linux-env/scripts/collect/collection_runtime.py',
                           'linux-env/scripts/collect/success.py',
                           'linux-env/scripts/collect/diagnostics.py',
                           'linux-env/scripts/collect/scene_sync.py',
                           'linux-env/scripts/collect/scene_reconcile.py',
                           'linux-env/scripts/collect/protocol.py',
                           'linux-env/scripts/collect/configure_clock.py',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pick_place.py',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/grasp_geometry.py',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/motion_operation.py',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/motion_wait.py',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/scene_transactions.py',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/ros_callbacks.py',
                           'vla_SO-ARM101/src/so_arm101_moveit_config/config/so_arm101.srdf',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/compute_workspace.py',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/ik_calibration.yaml',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/workspace_bounds.yaml']
        tracked_sources.append(PROFILE['source'])
        source_hashes = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in tracked_sources}
        image_source=self.executor.image_take.metadata
        if (image_source['source_sha256']!=source_hashes['linux-env/scripts/collect/image_take.c'] or
                image_source['python_source_sha256']!=source_hashes['linux-env/scripts/collect/image_transport.py']):
            raise RuntimeError('Loaded image adapter differs from the recording provenance')
        if source_hashes[PROFILE['source']] != PROFILE['source_sha256']:
            raise RuntimeError('Primitive cup asset differs from the loaded geometry contract')
        if transport_config['loaded_sha256'] != source_hashes['linux-env/scripts/collect/collection_runtime.py']:
            raise RuntimeError('Kit loaded a different collection runtime')
        if any(source_hashes['linux-env/scripts/collect/'+name] != digest
               for name,digest in transport_config['loaded_dependencies'].items()):
            raise RuntimeError('Kit loaded different transport dependencies')
        if (transport_config['renderer_gpu']!=int(os.environ['SOARM_RENDERER_GPU_INDEX']) or
                transport_config['renderer_multi_gpu'] is not False or transport_config['physics_gpu']!=0 or transport_config['cuda_device']!=0 or
                transport_config['cuda_device_count']!=1 or transport_config['cuda_visible_devices']!=gpu_uuid(int(os.getenv('ISAACSIM_GPU','0')))):
            raise RuntimeError('Renderer/physics/CUDA instance GPU isolation differs')
        if transport_config['physics_dynamics']!='cpu':raise RuntimeError('Collection requires CPU PhysX dynamics')
        try:initial_gpu_footprint=self.gpu_footprint()
        finally:atomic_json(root/'gpu-isolation-startup.json',getattr(self,'last_gpu_footprint',dict(passed=False,error='GPU audit did not complete')))
        provenance = dict(start_trial=attempted, started=time.time(), source_sha256=source_hashes,
            git_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip())
        if previous:
            for name, digest in previous['source_sha256'].items():
                if name != 'linux-env/scripts/collect/collect.py' and source_hashes[name] != digest:
                    raise ValueError(f'Resume motion/timing source changed: {name}')
            if previous['instance'] != args.instance or previous['clock_config'] != clock_config:
                raise ValueError('Resume instance/clock configuration differs')
            previous.setdefault('resume_sessions', []).append(provenance)
            atomic_json(root / 'run.json',previous)
        else:
            atomic_json(root / 'run.json',dict(instance=args.instance, host=socket.gethostname(), gpu_index=int(os.getenv('ISAACSIM_GPU', '0')),
            wall_profile_policy=PROFILE_POLICY,recording_version=6,camera_selection_policy=CAMERA_SELECTION_POLICY,image_transport=self.executor.image_take.metadata, gpu_isolation_initial=initial_gpu_footprint,transport_config=transport_config, observation_state_topic='/joint_states_measured',
            diagnostic=args.diagnostic, success_profile=PROFILE,
            motion_wait_policy=wait_policy, tick_clock_policy=CLOCK_POLICY,geometry_query_policy=QUERY_POLICY,quality_policy=POLICY,regrasp_policy=REGRASP_POLICY,regrasp_only=args.regrasp_only,
            application_contract='Held at observation tick; read-back verified PhysX targets before one-frame advance; maximum one sim tick latency',
            clock_config=clock_config, domain=90 + args.instance,
            bridge_port=8777 + 2 * args.instance, fps=30, baseline=baseline, baseline_cup_layout=self.baseline_cup_layout, eval_seeds=EVAL_SEEDS,
            **{k: v for k, v in provenance.items() if k != 'start_trial'}))
        start = time.monotonic()
        for trial in range(attempted, args.max_attempts):
            if accepted >= args.episodes:
                break
            if self.stop.is_set() or self.failure:
                raise RuntimeError(self.failure or 'Collection interrupted')
            disk_wait += wait_for_storage(root, self.stop, emit=lambda value: print(value, flush=True))
            wall_profile_start=len(getattr(self,'wall_profiles',[]))
            root_growth = RootFileGrowth(Path(os.environ['SOARM_INSTANCE_ROOT']))
            attempted += 1
            color = COLORS[accepted % 3]
            params = (json.loads(args.replay_receipt.read_text())['params'] if args.replay_receipt
                      else parameters(args.seed+trial,color,regrasp=args.regrasp_only or regrasp_slot(accepted)))
            params['trial'] = trial
            path = root / f'attempt-{trial:04d}'
            receipt = dict(physics=transport_config['physics_config'],params=params,accepted=False,episode_id=hashlib.sha256((str(root)+':'+str(trial)+':'+str(params['seed'])).encode()).hexdigest())
            wait_log_start = len(self.wait_diagnostics)
            failure_log_start = len(self.capture_failures)
            cycle_result = {}
            motion_epochs = set()
            duplicate_start = dict(self.sensors.duplicate_messages)
            rerender_start=dict(self.sensors.rerender_messages)
            stale_camera_start=dict(self.sensors.stale_camera_messages)
            writer = None
            receipt_cycle = None
            try:
                with self.generation.own('attempt start'):
                    self.current_episode_id=receipt['episode_id']
                    self.current_attempt_path=path
                    self.episode_snapshots=[]
                    self.node._qs_abort_evt.clear()
                    motion_epochs.add(self.generation.epoch)
                if self.needs_home_restore:
                    receipt['failed_episode_home_reset']=self.restore_failed_episode_home()
                    self.needs_home_restore=False
                receipt['gpu_isolation_before']=self.gpu_footprint()
                receipt['camera_graph']=self.camera_graph()
                self.motion_call(self.reset_release,params['object'])
                # Home with the same shared primitive used by Quickstart, with sim ticks active.
                home = self.motion_call(self.request_home)
                motion_epochs.add(home.motion_epoch)
                finished = self.motion_call(self.node._wait_motion_event, home, 90., epoch=home.motion_epoch)
                with self.generation.own('home wait continuation'):
                    self.generation.require(home.motion_epoch)
                    if not finished or not home.verdict or not home.verdict.get('ok'):
                        raise RuntimeError('Pre-episode home failed')
                self.servo.configure(params)
                receipt['home_before_reset'] = self.settle_home('before_reset')
                if self.safe_home_baseline is None:
                    self.safe_home_baseline=make_baseline(receipt['home_before_reset'],receipt['episode_id'])
                    run_config=json.loads((root/'run.json').read_text());run_config['safe_home_baseline']=self.safe_home_baseline
                    atomic_json(root/'run.json',run_config)
                receipt['cup_randomization'] = self.scene(params, baseline)
                # Replay uses the recorded initial arm/hold exactly; normal
                # collection verifies the calibrated home after every barrier.
                if args.replay_receipt:
                    receipt['home_after_reset'] = dict(passed=True,replayed=True,
                        restore_check=self.scene_restore['max_arm_position_error_rad'])
                else:
                    receipt['home_after_reset'] = self.settle_home('after_reset')
                    self.scene_restore = self.restore_physical(
                        'import collection_runtime as c\nr=c.runtime()\nresult=r.restore(r.snapshot(),arm=True)')
                    verification = self.advance_ticks(6)
                    self.episode_start_physical = verification[-1]
                    check=initial_arm(verification,
                        [receipt['home_after_reset']['target'][name] for name in JOINTS[:5]])
                    receipt['home_after_reset']['post_velocity_reset_check'] = check
                    if not check['contract_passed']:raise TickContractError(check['contract']['errors'],dict(contract=check['contract'],snapshots=verification))
                    if not check['passed']:
                        self.reject_quality('home_unstable',dict(label='post_velocity_reset',check=check),verification)
                    self.sync_scene()
                receipt['initial_arm_check'] = receipt['home_after_reset']
                self.node.approach_offset = params['approach_offset_m']
                self.node._grasp_approach_height_var.set(params['approach_height_m'])
                self.node.speed_scale = params['speed_scale']
                receipt['camera_duplicate_messages_before_episode'] = dict(self.sensors.duplicate_messages)
                receipt['initial_objects'] = dict(self.node.objects_data)
                receipt['initial_cups'] = dict(self.node._drop_data)
                receipt['initial_physical'] = self.scene_restore['snapshot']
                receipt['episode_start_physical'] = self.episode_start_physical
                receipt['planning_scene_pose_check'] = self.scene_pose_check
                receipt['planning_scene_clear_check'] = self.scene_clear_check
                receipt['initial_restore_check'] = {k:v for k,v in self.scene_restore.items() if k != 'snapshot'}
                receipt['initial_restore_error_m'] = self.scene_restore['max_position_error_m']
                receipt['initial_expert_reference'] = list(self.servo.reference)
                writer = EpisodeWriter(path, 30)
                with self.writer_lock:
                    self.writer = writer
                cycle_result = {}
                with self.generation.own('cycle request'):
                    cycle_epoch = self.generation.epoch
                    motion_epochs.add(cycle_epoch)
                def run_cycle(object_name=params['object'], epoch=cycle_epoch):
                    try:
                        def run_and_verdict():
                            if params.get('recovery')=='regrasp':
                                self.regrasp_prefix(params,epoch,cycle_result)
                            ok=self.node.run_cycle(object_name)
                            event=self.node._current_motion_event()
                            return dict(ok=ok, verdict=self.node._motion_verdict(event),
                                        verdict_epoch=event.motion_epoch if event is not None else None)
                        cycle_result.update(self.motion_call(run_and_verdict, epoch=epoch))
                    except Exception as worker_error:
                        cycle_result.update(ok=False,worker_error=repr(worker_error),exception=worker_error)
                cycle = threading.Thread(target=run_cycle, daemon=True)
                receipt_cycle = cycle
                budget_start = self.servo.execution_budget()
                def episode_budget():
                    current = self.servo.execution_budget()
                    return current[0]-budget_start[0], current[1]-budget_start[1]
                cycle.start()
                try:
                    self.node._motion_waiter.wait_thread(cycle, episode_budget, cycle_epoch, args.episode_deadline)
                finally:
                    if cycle.is_alive():
                        self.node._qs_abort_evt.set()
                        try:
                            with self.generation.own('cycle cleanup barrier'):
                                if self.generation.epoch == cycle_epoch:
                                    self.timeline(False)
                        finally:
                            # Retirement is a wall-time lifecycle barrier after motion cancellation.
                            cycle.join(timeout=5)
                            if cycle.is_alive():
                                self.failure = 'Cancelled cycle did not stop; refusing overlapping episodes'
                if not cycle_result.get('ok'):
                    if cycle_result.get('exception') is not None:raise cycle_result['exception']
                    verdict=cycle_result.get('verdict') or {}
                    if cycle_result.get('verdict_epoch')==cycle_epoch and verdict.get('outcome') in PLANNER_REJECTIONS:
                        self.reject_quality('planner_reject',verdict)
                    raise RuntimeError('Scripted cycle failed: '+repr(cycle_result))
                receipt['script_completed'] = True
                # Verification ticks are a trailing audit, outside the motion dataset.
                with self.writer_lock:
                    self.writer = None
                snapshots = self.advance_ticks(SETTLED_TICKS)
                receipt['settled_physical'] = snapshots
                receipt['score'] = settled_containment(snapshots, params['object'], params['color'])
                if not receipt['score']['contract_passed']:raise TickContractError(receipt['score']['contract']['errors'],dict(contract=receipt['score']['contract'],snapshots=snapshots))
                if not receipt['score']['passed']:
                    self.reject_quality('containment',receipt['score'],snapshots)
                receipt['accepted'] = True
            except Exception as exc:
                receipt['exception_type'] = type(exc).__name__
                if isinstance(exc,GpuIsolationError):receipt['gpu_isolation_failure']=exc.evidence
                if isinstance(exc,TickContractError):
                    receipt['pipeline_contract_failure']=dict(errors=exc.errors,evidence=exc.evidence)
                if isinstance(exc,QualityReject):
                    receipt['quality_reject'] = dict(kind=exc.kind,evidence=exc.evidence,contract=exc.contract)
                try:
                    with self.generation.own('failure provenance'):
                        receipt.update(self.attempt_error(exc, self.capture_failures[failure_log_start:],
                                                          motion_epochs, self.generation.epoch))
                except OwnershipStalled as provenance_error:
                    receipt['error'] = repr(provenance_error)
                    self.abort_stalled_owner(provenance_error, 'failure provenance')
            finally:
                try:
                    with self.writer_lock:
                        self.writer = None
                    try:
                        if self.generation.stalled:
                            raise OwnershipStalled(self.generation.stalled)
                        receipt['end_cancel'] = self.timeline(False)
                        if receipt_cycle is not None:
                            receipt_cycle.join(timeout=5.)
                            if receipt_cycle.is_alive():
                                raise RuntimeError('Episode worker did not retire')
                        receipt['retirement'] = self.workers.retire()
                    except Exception as retire_error:
                        receipt['retirement'] = dict(passed=False,error=repr(retire_error))
                        self.abort_stalled_owner(retire_error,'episode retirement')
                    if writer:
                        writer.close()
                        receipt['timing'] = timing_audit(writer.rows, 30)
                        receipt['accepted'] = receipt['accepted'] and receipt['timing']['passed']
                        if args.diagnostic:
                            try:
                                receipt['diagnostics'] = final_stills(path, writer.rows)
                            except Exception as exc:
                                receipt['diagnostics'] = dict(saved=False, error=repr(exc))
                                receipt['accepted'] = False
                                receipt['error'] = 'Diagnostic still capture failed: '+repr(exc)
                                self.failure = receipt['error']
                        if receipt.get('score') and receipt['score'].get('contract_passed'):
                            initial_z = receipt['initial_physical']['bodies'][params['object']]['position'][2]
                            maximum_z = max((row.get('target_object_pose', {}).get('z', initial_z) for row in writer.rows), default=initial_z)
                            receipt['score']['lift_m'] = maximum_z - initial_z
                            if maximum_z - initial_z < .025:
                                receipt['accepted'] = False
                                receipt['error'] = 'Object did not lift at least 25 mm before placement'
                                receipt['quality_reject'] = dict(kind='insufficient_lift',evidence=dict(lift_m=maximum_z-initial_z),contract=require_snapshots(snapshots))
                    receipt['capture_failures'] = list(self.capture_failures[failure_log_start:])
                    sensor_failure=getattr(self.sensors,'failure',None)
                    if isinstance(sensor_failure,SensorAmbiguity):
                        if receipt.get('error'):receipt['secondary_error']=receipt['error']
                        receipt.update(error=repr(sensor_failure),exception_type='SensorAmbiguity',
                            original_sensor_error_preserved=True,sensor_join_failure=sensor_failure.evidence)
                    receipt['noise_operations'] = list(self.servo.noise_specs)
                    if cycle_result.get('recovery_report'):
                        receipt['recovery_report']=cycle_result['recovery_report']
                    if cycle_result.get('worker_error'):
                        receipt['cycle_worker_error'] = cycle_result['worker_error']
                    if sensor_failure is not None:
                        self.failure=self.failure or repr(sensor_failure)
                        receipt['camera_boundary']=dict(passed=False,original_sensor_error=repr(sensor_failure))
                    else:receipt['camera_boundary']=self.sensors.require_camera_boundary(self.generation.floor_ns)
                    receipt.update(self.sensors.camera_diagnostics(motion_epochs,duplicate_start,rerender_start,stale_camera_start))
                    receipt['root_file_growth'] = root_growth.finish()
                    profiles=getattr(self,'wall_profiles',[])[wall_profile_start:]
                    writes=writer.write_profiles if writer else []
                    joined=joined_profiles(profiles,writes)
                    receipt['wall_tick_profile']=dict(summary=wall_profile_summary(joined),recorded_summary=wall_profile_summary([p for p in joined if p['recorded']]),ticks=profiles,async_writes=writes)
                    receipt['motion_wait_diagnostics'] = [r for r in self.wait_diagnostics[wait_log_start:]
                                                           if r['epoch'] in motion_epochs]
                    try:receipt['gpu_isolation_after']=self.gpu_footprint()
                    except Exception as gpu_error:
                        receipt['gpu_isolation_after']=getattr(gpu_error,'evidence',dict(passed=False,error=repr(gpu_error)))
                        self.failure=self.failure or repr(gpu_error)
                    receipt['disposition'] = disposition(receipt,self.failure,writer.rows if writer else None)
                    receipt['accepted'] = receipt['disposition']['kind']=='accepted'
                    self.needs_home_restore=receipt['disposition']['kind']=='quality_reject'
                    if receipt['disposition']['kind']=='terminal':self.persist_terminal_fault(receipt.get('error'),'attempt disposition')
                except Exception as finalization_error:
                    self.failed_finalization(receipt,finalization_error,motion_epochs,duplicate_start,rerender_start,stale_camera_start,
                        getattr(self,'wall_profiles',[])[wall_profile_start:],writer.write_profiles if writer else [])
                receipt['failure_streak']=streak.observe(receipt['disposition'])
                path.mkdir(parents=True, exist_ok=True)
                atomic_json(path / 'receipt.json',receipt)
                with (root / 'episodes.jsonl').open('a') as log:
                    log.write(json.dumps(receipt) + '\n')
                self.sensors.prune_durable_camera_history()
            if receipt['accepted']:
                accepted += 1
            elif not args.diagnostic:
                # Keep every failed seed and diagnosis; delete only this attempt's two known bulk files.
                for key in ('wrist', 'top'):
                    video = path / f'{key}.mp4'
                    if video.exists():
                        video.unlink()
            print(json.dumps(dict(trial=trial, accepted=accepted, target=args.episodes,
                yield_fraction=accepted / (trial + 1), error=receipt.get('error'), timing=receipt.get('timing'))), flush=True)
            elapsed = previous_wall + time.monotonic() - start
            if receipt['failure_streak']['stop']:
                stopped=dict(producer_invocation=self.finalization_certificate['invocation'],complete=False,episodes=accepted,attempts=attempted,wall_seconds=elapsed,
                    stop=receipt['disposition'],failure_streak=receipt['failure_streak'],end_marker='COLLECTION_STOPPED')
                atomic_json(root/'summary.json',stopped)
                self.stop.set();self.node._qs_abort_evt.set()
                self.generation.poison(RuntimeError(repr(stopped['stop'])))
                raise RuntimeError('Collection stopped: '+repr(stopped['stop']))
            atomic_json(root / 'summary.json',dict(producer_invocation=self.finalization_certificate['invocation'],complete=False, episodes=accepted,
                attempts=attempted, wall_seconds=elapsed, disk_wait_seconds=disk_wait,
                end_marker='COLLECTION_IN_PROGRESS'))
        summary = dict(producer_invocation=self.finalization_certificate['invocation'],complete=accepted == args.episodes, episodes=accepted, attempts=attempted,
            yield_fraction=accepted / max(attempted, 1), wall_seconds=previous_wall + time.monotonic() - start,
            disk_wait_seconds=disk_wait,
            episodes_per_wall_hour=accepted * 3600 / max(previous_wall + time.monotonic() - start, 1), end_marker='COLLECTION_FINISHED')
        atomic_json(root / 'summary.json',summary)
        print(json.dumps(summary), flush=True)
        if not summary['complete']:
            raise RuntimeError('Attempt budget exhausted before the requested success count')

    def close(self):
        self.stop.set();errors=[]
        status_root=self.status_output();status_root.mkdir(parents=True,exist_ok=True)
        with self.sensors.condition:
            self.sensors.closed=True;self.sensors.condition.notify_all()
        # Invalidate stale success BEFORE any cleanup RPC or executor operation.
        def note_terminal(source):
            fault=self.failure or getattr(self.sensors,'failure',None)
            if fault:self.persist_terminal_fault(fault,source)
        def cleanup(name,callback):
            try:
                callback()
                return True
            except BaseException as exc:
                errors.append(dict(stage=name,error=repr(exc)))
                self.generation.poison(exc);self.node._qs_abort_evt.set()
                try:self.persist_terminal_fault(exc,'cleanup '+name)
                except BaseException as publication_error:errors.append(dict(stage='terminal publication',error=repr(publication_error)))
        cleanup('pre-cleanup terminal status',lambda:note_terminal('before cleanup'))
        cleanup('native cancellation',lambda:timeline(False) if self.generation.stalled else self.timeline(False))
        cleanup('idle rate',lambda:mcp('run_python', {'code': "import carb\ns=carb.settings.get_settings()\ns.set_int('/app/runLoops/main/rateLimitFrequency',10)\ns.set_bool('/app/runLoops/main/rateLimitEnabled',True)\nresult=True"}))
        def retire_capture():
            self.capture_thread.join(timeout=frame_capacity_seconds()+5)
            if self.capture_thread.is_alive():raise RuntimeError('Capture worker did not retire')
        capture_retired=cleanup('capture retirement',retire_capture)
        retired=cleanup('executor shutdown',lambda:retire_executor(self.executor,self.ros_thread))
        if capture_retired and retired:
            destroyed=cleanup('node destruction',self.node.destroy_node)
            if destroyed:cleanup('ROS shutdown',lambda:shutdown_context(rclpy,self.ros_context))
        cleanup('post-cleanup terminal status',lambda:note_terminal('after cleanup'))
        cleanup('wall profile persistence',lambda:atomic_json(status_root/'wall-profile-finalization.json',
            dict(ticks=getattr(self,'wall_profiles',[]),capture_failures=self.capture_failures)))
        with self.status_lock:
            certificate=collection_status.finish(status_root,self.finalization_certificate,errors)
        if (self.finalization_certificate is not None and not certificate['passed']) or errors:
            raise RuntimeError('Collector finalization failed: '+repr(certificate))

    def request_stop(self):
        self.stop.set()
        self.node._qs_abort_evt.set()
        self.timeline(False)
        event = getattr(self.node, '_motion_event', None)
        if event is not None:
            event.verdict = dict(ok=False, outcome='collection_stopped')
            event.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', type=int, required=True, choices=range(10))
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=310000)
    parser.add_argument('--replay-receipt', type=Path)
    parser.add_argument('--max-attempts', type=int, default=250)
    parser.add_argument('--episode-deadline', type=float, default=180,
                        help='Minimum wall watchdog; grows with planned simulated execution time')
    parser.add_argument('--wait-rtf-floor', type=float, default=.1,
                        help='Conservative RTF floor used only for motion wall watchdogs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Re-audit and continue this output ledger')
    parser.add_argument('--regrasp-only',action='store_true',help='Use re-grasp recovery in every accepted slot for its smoke')
    parser.add_argument('--diagnostic', action='store_true', help='Keep failed videos and save final recorded-frame stills')
    args = parser.parse_args()
    if not math.isfinite(args.wait_rtf_floor) or args.wait_rtf_floor <= 0:
        parser.error('--wait-rtf-floor must be finite and positive')
    if not math.isfinite(args.episode_deadline) or args.episode_deadline <= 0:
        parser.error('--episode-deadline must be a finite positive watchdog')
    if args.replay_receipt:
        if args.resume:
            parser.error('Replay and resume are separate operations')
        args.episodes, args.max_attempts = 1, 1
    if args.episodes < 1 or args.episodes > 50:
        parser.error('This collector is limited to the validated 50-episode pilot')
    if os.environ.get('ROS_DOMAIN_ID') != str(90 + args.instance):
        parser.error('ROS_DOMAIN_ID does not match the assigned instance')
    confined(args.output)
    prepare_environment(args.instance)
    os.environ['SOARM_KIT_MCP_PORT'] = str(8777 + 2 * args.instance)
    rclpy.init()
    collector = Collector(args)
    signal.signal(signal.SIGINT, lambda *_: collector.request_stop())
    signal.signal(signal.SIGTERM, lambda *_: collector.request_stop())
    try:
        collector.run()
    except BaseException:
        try:collector.close()
        except BaseException as cleanup_error:print('Secondary collector cleanup failure: '+repr(cleanup_error),file=sys.stderr,flush=True)
        raise
    else:collector.close()


if __name__ == '__main__':
    main()
