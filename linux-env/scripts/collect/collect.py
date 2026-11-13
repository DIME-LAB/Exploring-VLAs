#!/usr/bin/env python3
"""ROS sim-tick collector. Run with system Humble Python; export with pixi separately."""
import argparse
from contextlib import contextmanager
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
from contract import COLORS, JOINTS, EVAL_SEEDS, corrective_action, parameters, timing_audit
from checkpoint import progress
from storage import CANONICAL_ROOT, RootFileGrowth, SceneSafetyMarker, confined, prepare_environment, wait_for_storage
from protocol import StepGeneration, CancelledGeneration, OwnershipStalled
from scene_sync import physical_pose_caches, collision_pose_errors
from scene_reconcile import SceneReconciler, ros_messages
from success import SETTLED_TICKS, initial_arm, settled_containment
from cup_profile import PROFILE
from diagnostics import final_stills

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
    """Exact sim-stamp join: each image pair and state are consumed once, without resampling."""
    def __init__(self, node, generation):
        self.generation = generation
        self.condition = threading.Condition()
        self.pending = collections.OrderedDict()
        self.last = -1
        self.failure = None
        self.duplicate_messages = collections.Counter()
        self.reference = None
        self.acks = {}
        self.subs = []
        for key, topic in [('wrist', '/wrist_camera_rgb_sim'), ('top', '/workspace_camera_sim')]:
            self.subs.append(node.create_subscription(Image, topic,
                lambda msg, key=key: self.put(key, msg), QoSProfile(depth=64, reliability=ReliabilityPolicy.RELIABLE)))
        self.subs.append(node.create_subscription(JointState, '/joint_states_measured',
                                                lambda msg: self.put('joint', msg), 100))
        self.subs.append(node.create_subscription(String, '/collection_tick',
                                                lambda msg: self.put('tick', msg), 64))
        self.subs.append(node.create_subscription(JointState, '/joint_states', self.reference_received, 100))
        self.subs.append(node.create_subscription(String, '/collection_command_ack', self.ack_received, 64))

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
            if reference['stamp_ns'] > stamp+1000:
                reference['source_epoch'] = None
                reference['validity'] = 'future_stamp_or_unknown'
            reference['same_tick'] = reference['source_epoch'] == epoch and abs(reference['stamp_ns']-stamp) <= 1000
            reference['age_sim_ns'] = stamp-reference['stamp_ns']
        return reference

    def put(self, key, msg):
        if key == 'tick':
            msg = json.loads(msg.data)
            raw_stamp = msg['sim_ns']
        else:
            raw_stamp = ns(msg.header.stamp)
        # Native image and rclpy clocks round the same sim tick by up to a few ns.
        stamp = round(raw_stamp / 1000)
        if raw_stamp <= 0:
            return
        with self.condition:
            if stamp <= self.last:
                if key in ('wrist', 'top'):
                    self.duplicate_messages[key] += 1
                return
            entry = self.pending.setdefault(stamp, {})
            if key in entry and key != 'joint':
                self.duplicate_messages[key] += 1
                return
            entry[key] = (msg, time.monotonic_ns())
            # Bound outstanding joins; paused cameras can repeat their last stamp.
            while len(self.pending) > 240:
                self.pending.popitem(last=False)
            self.condition.notify_all()

    def read(self, timeout=15):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if self.failure:
                    raise RuntimeError(self.failure)
                ready = sorted(k for k, v in self.pending.items() if len(v) == 4 and k > self.last)
                if ready:
                    stamp = ready[0]
                    entry = self.pending.pop(stamp)
                    self.last = stamp
                    for old in list(self.pending):
                        if old < stamp:
                            del self.pending[old]
                    return ns(entry['wrist'][0].header.stamp), entry
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    counts = collections.Counter(key for v in self.pending.values() for key in v)
                    raise TimeoutError(f'No exact camera/state sim-tick match: {dict(counts)}')
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
            with (self.path / 'frames.jsonl').open('w') as out:
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    row, images = item
                    for key, img in images.items():
                        self.encoders[key].stdin.write(img.tobytes())
                    out.write(json.dumps(row) + '\n')
                    self.rows.append(row)
        except Exception as exc:
            self.failure = repr(exc)

    def add(self, row, images):
        if self.failure:
            raise RuntimeError(self.failure)
        self.queue.put((row, images), timeout=2)

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
        self.previous_executed = None
        self.params = parameters(0, 'red')
        self.rng = random.Random(0)
        self.last_sim_ns = 0
        self.phase = 'hold'
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

    def execute(self, trajectory, done, operation=None):
        epoch = done.motion_epoch
        if operation is not None and (operation.event is not done or operation.epoch != epoch):
            raise ValueError('Planner result/event origin mismatch')
        with self.generation.own('collector motion'):
            self.generation.require(epoch)
            with self.lock:
                duration = ns(trajectory.points[-1].time_from_start)/1e9
                self.node._plan_motion_wait(done, duration)
                self.planned_sim_seconds += duration
                self.planned_operations += 1
                self.plan = dict(trajectory=trajectory, done=done, start=None, epoch=epoch)
                self.phase = operation.tag if operation is not None else getattr(done,'motion_tag','home')
            self.step(True)

    def gripper(self, position, duration, blocking):
        epoch = self.node._motion_token()
        with self.generation.own('collector motion'):
            self.generation.require(epoch)
            with self.lock:
                if abs(self.node._actual_positions['gripper_joint'] - position) < .0005:
                    if self.reference is None:
                        self.reference = [self.node._actual_positions[name] for name in JOINTS]
                    self.reference[5] = position
                    return True if blocking else None
            done = self.node._new_motion_event()
            self.node._plan_motion_wait(done, duration)
            with self.lock:
                self.planned_sim_seconds += duration
                self.planned_operations += 1
                self.grip_plan = dict(target=position, duration=duration, done=done, start=None, epoch=epoch)
                self.phase = 'gripper'
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

    def tick(self, stamp, state):
        completed = []
        margin = getattr(getattr(self.node, '_motion_waiter', None), 'sim_margin', 5.)
        with self.lock:
            self.last_sim_ns = stamp
            if self.reference is None:
                self.reference = list(state)
            noise_active = False
            active = self.plan is not None or self.grip_plan is not None
            if self.plan:
                plan = self.plan
                if plan['start'] is None:
                    plan['start'] = stamp
                elapsed = (stamp - plan['start']) / 1e9
                duration = ns(plan['trajectory'].points[-1].time_from_start) / 1e9
                positions = self.interpolate(plan['trajectory'], elapsed + 1 / 30)
                for name, value in zip(plan['trajectory'].joint_names, positions):
                    self.reference[JOINTS.index(name)] = value
                noise_active = elapsed < duration * .8
                if elapsed > duration and max(abs(a - b) for a, b in zip(state[:5], self.reference[:5])) < .008:
                    completed.append(plan['done']); self.plan = None
                elif elapsed > duration + margin:
                    raise TimeoutError(f'Arm did not converge after {elapsed:.3f} sim seconds: state={state[:5]} reference={self.reference[:5]}')
            if self.grip_plan:
                plan = self.grip_plan
                if plan['start'] is None:
                    plan['start'] = stamp
                    plan['initial'] = state[5]
                elapsed = (stamp - plan['start']) / 1e9
                alpha = min(1., (elapsed + 1 / 30) / plan['duration'])
                self.reference[5] = plan['initial'] + alpha * (plan['target'] - plan['initial'])
                if elapsed > plan['duration'] and abs(state[5] - plan['target']) < .008:
                    completed.append(plan['done']); self.grip_plan = None
                elif elapsed > plan['duration'] + margin:
                    raise TimeoutError('Gripper did not converge in sim time')
            clean = corrective_action(state, self.reference)
            executed = list(clean)
            if noise_active and self.rng.random() < self.params['noise_probability']:
                for i in range(5):
                    executed[i] += max(-.012, min(.012, self.rng.gauss(0., self.params['noise_std_rad'])))
            for i, name in enumerate(JOINTS):
                lo, hi = JOINT_LIMITS[name]
                executed[i] = max(lo, min(hi, executed[i]))
            reference = list(self.reference)
            self.last_command_clock_ns = self.node.get_clock().now().nanoseconds
            phase = self.phase

        return (clean, executed, active, phase, reference, self.last_command_clock_ns,
                completed, self.plan is not None or self.grip_plan is not None)

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
        from so_arm101_control.pick_place import PickPlaceNode
        self.args = args
        self.scene_marker = SceneSafetyMarker(os.environ['SOARM_INSTANCE_ROOT'])
        self.scene_marker.require_fresh_stack()
        self.node = PickPlaceNode(use_sim_time=True, joint_states_topic='/joint_states_measured')
        self.generation = StepGeneration()
        self.node._motion_context = threading.local()
        self.node._motion_generation_provider = lambda: self.generation.epoch
        self.node._motion_effect_guard = self.motion_effect
        self.node._scene_failure_handler = self.scene_incomplete
        self.sensors = Sensors(self.node, self.generation)
        # Only joined, generation-validated snapshots update expert state/cache.
        self.node.destroy_subscription(self.node.js_sub)
        with self.node.objects_lock, self.node._drop_lock:
            self.node._physical_pose_cache = True
        self.servo = Servo(self.node, self.generation, self.timeline)
        self.command_pub = self.node.create_publisher(String, '/collection_command', 64)
        self.node._trajectory_executor = self.servo.execute
        self.node._gripper_executor = self.servo.gripper
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        self.ros_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.ros_thread.start()
        self.writer = None
        self.writer_lock = threading.Lock()
        self.stop = threading.Event()
        self.failure = None
        self.wait_diagnostics = []
        self.node._motion_waiter = MotionWaiter(
            clock=lambda:self.node.get_clock().now().nanoseconds/1e9,
            validate=self.generation.require, emit=self.motion_wait_diagnostic,
            abort=lambda:self.stop.is_set() or self.node._qs_abort_evt.is_set(),
            rtf_floor=args.wait_rtf_floor)
        self.stalled_abort_lock = threading.Lock()
        self.stalled_abort_started = False
        self.generation.on_stall = self.abort_stalled_owner
        self.tick_condition = threading.Condition()
        self.last_processed_ns = 0
        self.last_snapshot = None
        self.node._simulation_held = lambda: mcp('run_python', {'code':
            'import collection_runtime as c\nr=c.runtime()\nresult=not r.busy and not r.timeline.is_playing()'})
        self.scene_reset = SceneReconciler(self.node, ros_messages(),
            (REPO/'vla_SO-ARM101/src/so_arm101_moveit_config/config/so_arm101.srdf').read_text())
        self.capture_thread = threading.Thread(target=self.capture, daemon=True)
        self.capture_thread.start()

    def motion_wait_diagnostic(self, record):
        self.wait_diagnostics.append(record)
        print(json.dumps(record), flush=True)

    @contextmanager
    def motion_effect(self, epoch):
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

    def retire_motion(self):
        with self.servo.lock:
            for plan in (self.servo.plan, self.servo.grip_plan):
                if plan:
                    plan['done'].verdict = dict(ok=False, outcome='cancelled', epoch=plan['epoch'])
                    plan['done'].set()
            self.servo.plan = self.servo.grip_plan = None
            self.servo.previous_executed = None
            self.servo.reference = None
        with self.sensors.condition:
            self.sensors.pending.clear()
            self.sensors.acks.clear()

    def scene_incomplete(self, exc, source='MoveIt'):
        self.node._scene_terminal_error = str(exc)
        self.generation.poison(exc)
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

    def abort_stalled_owner(self, exc, requester):
        self.stop.set()
        self.node._qs_abort_evt.set()
        with self.stalled_abort_lock:
            if self.stalled_abort_started:
                return
            self.stalled_abort_started = True
            self.failure = str(exc)
        # No generation lock: its owner may be stuck in publication forever.
        # Native cancellation only pauses and invalidates queued commands.
        try:
            timeline(False)
        except Exception as pause_error:
            self.failure += '; emergency pause failed: '+repr(pause_error)
        finally:
            print(json.dumps(dict(event='stalled_generation_owner', requester=requester,
                error=self.failure, action='abort; no reset or next episode permitted')), flush=True)

    def cancel_owner(self, native_cancel):
        try:
            return self.generation.cancel(native_cancel, self.retire_motion)
        except OwnershipStalled as exc:
            self.abort_stalled_owner(exc, 'cancel/reset')
            raise OwnershipStalled(self.failure) from exc

    def timeline(self, play):
        if not play:
            return self.cancel_owner(lambda: timeline(False))
        with self.generation.own('advance'):
            self.generation.require(self.generation.epoch)
            return timeline(True)

    def restore_physical(self, code):
        # Retirement owns the scene before setters. On a stalled owner, only
        # native cancellation is allowed; never execute the restore callback.
        return self.cancel_owner(lambda: self.mutate_kit('run_python', {'code':code}))

    def capture(self):
        while not self.stop.is_set():
            epoch = None
            try:
                try:
                    stamp, entry = self.sensors.read(timeout=30)
                except TimeoutError:
                    if self.servo.plan or self.servo.grip_plan:
                        raise TimeoutError('Motion stalled: no synchronized camera/state tick within 30 wall seconds')
                    continue
                epoch = entry['tick'][0]['epoch']
                self.generation.require(epoch, stamp)
                with self.generation.own('capture/dispatch'):
                    self.generation.require(epoch, stamp)
                    joint, received = entry['joint']
                    mapped = dict(zip(joint.name, joint.position))
                    if not all(name in mapped for name in JOINTS):
                        raise ValueError('Incomplete joint state')
                    state = [mapped[name] for name in JOINTS]
                    self.node._joint_states_callback(joint)
                    self.update_pose_cache(snapshot=entry['tick'][0])
                    snapshot = entry['tick'][0]
                    if (max(abs(a-b) for a,b in zip(state,snapshot['state'])) > 1e-7 or
                        max(abs(a-b) for a,b in zip(joint.velocity,snapshot['velocity'])) > 1e-7):
                        raise RuntimeError('Physical joint/pose feedback channels disagree')
                    for plan in (self.servo.plan, self.servo.grip_plan):
                        if plan:
                            self.generation.require(plan['epoch'])
                    clean, executed, active, phase, reference, command_clock, completed, advance = self.servo.tick(stamp, state)
                    self.generation.require(epoch, stamp)
                    self.servo.publish_reference(executed)
                    self.generation.require(epoch, stamp)
                    self.command_pub.publish(String(data=json.dumps(dict(epoch=epoch,
                        observation_stamp_ns=stamp, command=executed, advance=advance))))
                    ack = self.sensors.read_ack(epoch, stamp)
                    self.generation.require(epoch, stamp)
                    row = dict(observation_epoch=snapshot['epoch'], expert_epoch=epoch,
                        invalid_through_ns=self.generation.floor_ns, sim_ns=stamp, wall_ns=time.monotonic_ns(), joint_stamp_ns=ns(joint.header.stamp),
                        joint_receive_ns=received, wrist_stamp_ns=ns(entry['wrist'][0].header.stamp),
                        top_stamp_ns=ns(entry['top'][0].header.stamp), wrist_receive_ns=entry['wrist'][1],
                        top_receive_ns=entry['top'][1], state=state, action=clean, executed_action=executed,
                        expert_observation_stamp_ns=stamp, command_clock_ns=command_clock,
                        expert_reference=reference, executing=active, phase=phase,
                        state_source=snapshot['state_source'], velocity=list(joint.velocity),
                        physics_state=snapshot['state'], physics_velocity=snapshot['velocity'],
                        physics_drive_targets=snapshot['drive_targets'],
                        pose_stamp_ns=snapshot['sim_ns'], physics_time_s=snapshot['physics_time_s'],
                        controller_reference=self.sensors.reference_at(epoch, stamp),
                        declared_application_stamp_ns=stamp, **ack)
                    selected = self.node._selected_object
                    if selected in snapshot['bodies']:
                        body = snapshot['bodies'][selected]
                        row['target_object_pose'] = dict(zip(('x', 'y', 'z'), body['position']))
                        row['target_body'] = body
                        row['target_cup'] = snapshot['bodies']['cup_' + self.servo.params['color']]
                    with self.writer_lock:
                        if self.writer:
                            self.writer.add(row, {key: Sensors.image(entry[key][0]).copy() for key in ('wrist', 'top')})
                    with self.tick_condition:
                        self.last_processed_ns = stamp
                        self.last_snapshot = snapshot
                        self.tick_condition.notify_all()
                    if completed:
                        for event in completed:
                            self.node._set_motion_status(dict(ok=True, outcome='physical_sim_tick_converged', epoch=epoch), event)
                            event.set()
            except OwnershipStalled as exc:
                self.abort_stalled_owner(exc, 'capture/dispatch')
                return
            except CancelledGeneration:
                continue
            except Exception as exc:
                if self.generation.stalled:
                    self.abort_stalled_owner(OwnershipStalled(self.generation.stalled), 'capture/dispatch')
                    return
                with self.generation.own('collector motion'):
                    if epoch is not None and epoch != self.generation.epoch:
                        continue
                    message = repr(exc)
                    self.node._qs_abort_evt.set()
                    self.node._set_motion_status(dict(ok=False, outcome='capture_error', msg=message))
                    if isinstance(exc, TimeoutError):
                        # A step deadline fails this attempt; the next seed gets a fresh motion plan.
                        with self.servo.lock:
                            done = [plan['done'] for plan in (self.servo.plan, self.servo.grip_plan) if plan]
                            self.servo.plan = self.servo.grip_plan = None
                            self.servo.reference = [self.node._actual_positions[name] for name in JOINTS]
                        try:
                            self.timeline(False)
                        except Exception as pause_error:
                            self.failure = repr(pause_error)
                        for event in done:
                            event.verdict = dict(ok=False, outcome='capture_error', msg=message)
                            event.set()
                    else:
                        self.failure = message
                    if getattr(self.node, '_motion_event', None):
                        self.node._motion_event.set()
                    if self.failure:
                        return

    def advance_ticks(self, count):
        snapshots = []
        for _ in range(count):
            with self.tick_condition:
                previous = self.last_processed_ns
                epoch = self.generation.epoch
            self.timeline(True)
            with self.tick_condition:
                if not self.tick_condition.wait_for(lambda: (self.last_snapshot and self.last_snapshot['epoch'] == epoch and self.last_processed_ns > previous) or self.failure, timeout=20):
                    raise TimeoutError('One-frame capture/command handshake exceeded 20 wall seconds')
                if self.failure:
                    raise RuntimeError(self.failure)
                snapshots.append(self.last_snapshot)
        return snapshots

    def wait_ready(self):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.failure:
                raise RuntimeError(self.failure)
            if self.node.objects_data and len(self.node._drop_data) == 3 and self.servo.last_sim_ns and self.sensors.reference:
                for client in (self.node.plan_client, self.node.validity_client, self.node._apply_scene_client):
                    if not client.wait_for_service(timeout_sec=5):
                        raise RuntimeError('MoveIt service unavailable')
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
        objects, cups = physical_pose_caches(snapshot)
        with self.node.objects_lock, self.node._drop_lock:
            self.node._physical_pose_cache = True
            self.node.objects_data = objects
            self.node._drop_data = cups

    def sync_scene(self):
        self.update_pose_cache(self.episode_start_physical)
        cleared = self.scene_reset.clear()
        self.node._cmd_toggle_ground_plane()
        self.node._add_cup_collision_objects(cups_dict=dict(self.node._drop_data))
        self.node._add_lego_collision_objects(legos_dict=dict(self.node.objects_data))
        world_ids = ['ground_plane'] + list(self.node._cup_collision_names) + list(self.node._lego_collision_names)
        with self.node._motion_guard():
            self.scene_reset.set_policy(world_ids)
        deadline = time.monotonic() + 15
        while True:
            check = self.scene_reset.verify(self.episode_start_physical, JOINTS)
            if check['passed']:
                self.scene_pose_check = dict(check, cleared=cleared)
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f'Full planning scene reset verification failed: {check}')
            self.stop.wait(.05)

    def run(self):
        args = self.args
        root = confined(args.output)
        if (root / 'superseded.json').exists():
            raise ValueError('Superseded diagnostic run requires a fresh output directory')
        root.mkdir(parents=True, exist_ok=args.resume)
        previous = json.loads((root / 'run.json').read_text()) if args.resume else None
        if previous and previous.get('diagnostic', False) != args.diagnostic:
            raise ValueError('Resume diagnostic retention mode differs from the existing run')
        wait_policy = dict(rtf_floor=args.wait_rtf_floor, sim_margin_s=5., wall_margin_s=30.,
                           episode_wall_min_s=args.episode_deadline)
        if previous and previous.get('motion_wait_policy') != wait_policy:
            raise ValueError('Resume motion wait policy differs from the existing run')
        receipts = []
        if previous:
            for trial, path in enumerate(sorted(root.glob('attempt-*'))):
                if path.name != f'attempt-{trial:04d}':
                    raise ValueError('Cannot resume a ledger with missing attempt directories')
                receipt = json.loads((path / 'receipt.json').read_text())
                if receipt['accepted']:
                    rows = [json.loads(line) for line in (path / 'frames.jsonl').read_text().splitlines()]
                    if not timing_audit(rows, 30)['passed'] or not all((path / f'{k}.mp4').exists() for k in ('wrist', 'top')):
                        raise ValueError(f'Cannot resume invalid accepted attempt {path}')
                receipts.append(receipt)
            logged = [json.loads(line) for line in (root / 'episodes.jsonl').read_text().splitlines()]
            if logged != receipts:
                raise ValueError('Resume attempt log differs from the receipt ledger')
        attempted, accepted = progress(receipts, args.seed)
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
            for key in ('contact','rest'):
                expected = PROFILE['effective_offsets_m'][kind+'_'+key]
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
        (root / 'first-reset-scene.json').write_text(json.dumps(self.scene_pose_check, indent=2))
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
        tracked_sources = ['linux-env/scripts/collect/collect.py', 'linux-env/scripts/collect/contract.py',
                           'linux-env/scripts/collect/checkpoint.py',
                           'linux-env/scripts/collect/storage.py',
                           'linux-env/scripts/collect/collection_runtime.py',
                           'linux-env/scripts/collect/success.py',
                           'linux-env/scripts/collect/cup_profile.py',
                           'linux-env/scripts/collect/cup_profile.json',
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
                           'vla_SO-ARM101/src/so_arm101_moveit_config/config/so_arm101.srdf',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/compute_workspace.py',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/ik_calibration.yaml',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/workspace_bounds.yaml']
        tracked_sources.append(PROFILE['source'])
        source_hashes = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in tracked_sources}
        if source_hashes[PROFILE['source']] != PROFILE['source_sha256']:
            raise RuntimeError('Cup shell differs from the calibrated success profile; recalibrate before collecting')
        if transport_config['loaded_sha256'] != source_hashes['linux-env/scripts/collect/collection_runtime.py']:
            raise RuntimeError('Kit loaded a different collection runtime')
        if any(source_hashes['linux-env/scripts/collect/'+name] != digest
               for name,digest in transport_config['loaded_dependencies'].items()):
            raise RuntimeError('Kit loaded different transport dependencies')
        if any(transport_config[name] != int(os.getenv('ISAACSIM_GPU','0'))
               for name in ('renderer_gpu','physics_gpu','cuda_device')):
            raise RuntimeError('Renderer/physics/CUDA instance GPU assignment differs')
        provenance = dict(start_trial=attempted, started=time.time(), source_sha256=source_hashes,
            git_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip())
        if previous:
            for name, digest in previous['source_sha256'].items():
                if name != 'linux-env/scripts/collect/collect.py' and source_hashes[name] != digest:
                    raise ValueError(f'Resume motion/timing source changed: {name}')
            if previous['instance'] != args.instance or previous['clock_config'] != clock_config:
                raise ValueError('Resume instance/clock configuration differs')
            previous.setdefault('resume_sessions', []).append(provenance)
            (root / 'run.json').write_text(json.dumps(previous, indent=2))
        else:
            (root / 'run.json').write_text(json.dumps(dict(instance=args.instance, host=socket.gethostname(), gpu_index=int(os.getenv('ISAACSIM_GPU', '0')),
            recording_version=2, transport_config=transport_config, observation_state_topic='/joint_states_measured',
            diagnostic=args.diagnostic, success_profile=PROFILE,
            motion_wait_policy=wait_policy,
            application_contract='Held at observation tick; read-back verified PhysX targets before one-frame advance; maximum one sim tick latency',
            clock_config=clock_config, domain=90 + args.instance,
            bridge_port=8777 + 2 * args.instance, fps=30, baseline=baseline, baseline_cup_layout=self.baseline_cup_layout, eval_seeds=EVAL_SEEDS,
            **{k: v for k, v in provenance.items() if k != 'start_trial'}), indent=2))
        start = time.monotonic()
        for trial in range(attempted, args.max_attempts):
            if accepted >= args.episodes:
                break
            if self.stop.is_set() or self.failure:
                raise RuntimeError(self.failure or 'Collection interrupted')
            disk_wait += wait_for_storage(root, self.stop, emit=lambda value: print(value, flush=True))
            root_growth = RootFileGrowth(Path(os.environ['SOARM_INSTANCE_ROOT']))
            attempted += 1
            color = COLORS[accepted % 3]
            params = (json.loads(args.replay_receipt.read_text())['params'] if args.replay_receipt
                      else parameters(args.seed + trial, color))
            params['trial'] = trial
            path = root / f'attempt-{trial:04d}'
            receipt = dict(params=params, accepted=False)
            wait_log_start = len(self.wait_diagnostics)
            motion_epochs = set()
            duplicate_start = dict(self.sensors.duplicate_messages)
            writer = None
            try:
                with self.generation.own('attempt start'):
                    self.node._qs_abort_evt.clear()
                    motion_epochs.add(self.generation.epoch)
                # Home with the same shared primitive used by Quickstart, with sim ticks active.
                home = self.motion_call(self.node._cmd_grasp_home_data)
                motion_epochs.add(home.motion_epoch)
                finished = self.motion_call(self.node._wait_motion_event, home, 90., epoch=home.motion_epoch)
                with self.generation.own('home wait continuation'):
                    self.generation.require(home.motion_epoch)
                    if not finished or not home.verdict or not home.verdict.get('ok'):
                        raise RuntimeError('Pre-episode home failed')
                self.servo.configure(params)
                measured_home = self.advance_ticks(6)
                from so_arm101_control.grasp_geometry import _grasp_home_data_target
                receipt['initial_arm_check'] = initial_arm(measured_home, [_grasp_home_data_target()[name] for name in JOINTS[:5]])
                if not receipt['initial_arm_check']['passed']:
                    raise RuntimeError('Measured physical arm is not stable at the episode home pose')
                receipt['cup_randomization'] = self.scene(params, baseline)
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
                    cycle_result.update(ok=self.motion_call(self.node.run_cycle, object_name, epoch=epoch))
                cycle = threading.Thread(target=run_cycle, daemon=True)
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
                            self.timeline(False)
                        finally:
                            # Retirement is a wall-time lifecycle barrier after motion cancellation.
                            cycle.join(timeout=5)
                            if cycle.is_alive():
                                self.failure = 'Cancelled cycle did not stop; refusing overlapping episodes'
                if not cycle_result.get('ok'):
                    raise RuntimeError('Scripted cycle failed: ' + repr(self.node._last_motion_status))
                # Verification ticks are a trailing audit, outside the motion dataset.
                with self.writer_lock:
                    self.writer = None
                snapshots = self.advance_ticks(SETTLED_TICKS)
                receipt['settled_physical'] = snapshots
                receipt['score'] = settled_containment(snapshots, params['object'], params['color'])
                if not receipt['score']['passed']:
                    raise RuntimeError('Whole-block settled containment failed')
                receipt['accepted'] = True
            except Exception as exc:
                receipt['error'] = repr(exc)
                self.timeline(False)
            finally:
                with self.writer_lock:
                    self.writer = None
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
                    if receipt.get('score'):
                        initial_z = receipt['initial_physical']['bodies'][params['object']]['position'][2]
                        maximum_z = max((row.get('target_object_pose', {}).get('z', initial_z) for row in writer.rows), default=initial_z)
                        receipt['score']['lift_m'] = maximum_z - initial_z
                        if maximum_z - initial_z < .025:
                            receipt['accepted'] = False
                            receipt['error'] = 'Object did not lift at least 25 mm before placement'
                receipt['camera_duplicate_messages'] = {key: value - duplicate_start.get(key, 0) for key, value in self.sensors.duplicate_messages.items()}
                receipt['root_file_growth'] = root_growth.finish()
                receipt['motion_wait_diagnostics'] = [r for r in self.wait_diagnostics[wait_log_start:]
                                                       if r['epoch'] in motion_epochs]
                path.mkdir(parents=True, exist_ok=True)
                (path / 'receipt.json').write_text(json.dumps(receipt, indent=2))
                with (root / 'episodes.jsonl').open('a') as log:
                    log.write(json.dumps(receipt) + '\n')
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
            (root / 'summary.json').write_text(json.dumps(dict(complete=False, episodes=accepted,
                attempts=attempted, wall_seconds=elapsed, disk_wait_seconds=disk_wait,
                end_marker='COLLECTION_IN_PROGRESS'), indent=2))
        summary = dict(complete=accepted == args.episodes, episodes=accepted, attempts=attempted,
            yield_fraction=accepted / max(attempted, 1), wall_seconds=previous_wall + time.monotonic() - start,
            disk_wait_seconds=disk_wait,
            episodes_per_wall_hour=accepted * 3600 / max(previous_wall + time.monotonic() - start, 1), end_marker='COLLECTION_FINISHED')
        (root / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        if not summary['complete']:
            raise RuntimeError('Attempt budget exhausted before the requested success count')

    def close(self):
        self.stop.set()
        try:
            self.timeline(False)
            mcp('run_python', {'code': "import carb\ns=carb.settings.get_settings()\ns.set_int('/app/runLoops/main/rateLimitFrequency',10)\ns.set_bool('/app/runLoops/main/rateLimitEnabled',True)\nresult=True"})
        finally:
            self.executor.shutdown(timeout_sec=5)
            self.node.destroy_node()
            rclpy.shutdown()

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
    finally:
        collector.close()


if __name__ == '__main__':
    main()
