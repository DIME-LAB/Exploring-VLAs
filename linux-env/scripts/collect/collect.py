#!/usr/bin/env python3
"""ROS sim-tick collector. Run with system Humble Python; export with pixi separately."""
import argparse
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

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'vla_SO-ARM101/src/so_arm101_control'))
sys.path.insert(0, str(REPO / 'omni-kit-mcp'))
from kit_mcp.client import call as kit_call
from so_arm101_control.pick_place import PickPlaceNode, JOINT_LIMITS
from contract import COLORS, JOINTS, EVAL_SEEDS, corrective_action, parameters, timing_audit


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
    # Rate limiting a paused app saves rendering CPU while ROS plans on wall time.
    code = "import omni.timeline, carb\ns=carb.settings.get_settings()\n"
    if play:
        code += "s.set_bool('/app/runLoops/main/rateLimitEnabled',False)\n"
        code += "omni.timeline.get_timeline_interface().play()\n"
    else:
        code += "omni.timeline.get_timeline_interface().pause()\n"
        code += "s.set_int('/app/runLoops/main/rateLimitFrequency',10)\n"
        code += "s.set_bool('/app/runLoops/main/rateLimitEnabled',True)\n"
    return mcp('run_python', {'code': code + 'result=True'})


class Sensors:
    """Exact sim-stamp join: each image pair and state are consumed once, without resampling."""
    def __init__(self, node):
        self.condition = threading.Condition()
        self.pending = collections.OrderedDict()
        self.last = -1
        self.failure = None
        self.duplicate_messages = collections.Counter()
        self.subs = []
        for key, topic in [('wrist', '/wrist_camera_rgb_sim'), ('top', '/workspace_camera_sim')]:
            self.subs.append(node.create_subscription(Image, topic,
                lambda msg, key=key: self.put(key, msg), QoSProfile(depth=64, reliability=ReliabilityPolicy.RELIABLE)))
        self.subs.append(node.create_subscription(JointState, '/joint_states',
                                                lambda msg: self.put('joint', msg), 100))

    def put(self, key, msg):
        raw_stamp = ns(msg.header.stamp)
        # Native image and rclpy clocks round the same sim tick by up to a few ns.
        stamp = round(raw_stamp / 1000)
        if raw_stamp <= 0:
            return
        with self.condition:
            if stamp <= self.last:
                return
            entry = self.pending.setdefault(stamp, {})
            if key in entry and key != 'joint':
                self.duplicate_messages[key] += 1
                return
            entry[key] = (msg, time.monotonic_ns())
            # Keep odd physics ticks too: joint_states is faster than cameras.
            while len(self.pending) > 240:
                self.pending.popitem(last=False)
            self.condition.notify_all()

    def read(self, timeout=15):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if self.failure:
                    raise RuntimeError(self.failure)
                ready = sorted(k for k, v in self.pending.items() if len(v) == 3 and k > self.last)
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
    def __init__(self, node):
        self.node = node
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

    def configure(self, params):
        self.params = params
        self.rng = random.Random(params['noise_seed'])

    def execute(self, trajectory, done):
        with self.lock:
            self.plan = dict(trajectory=trajectory, done=done, start=None)
            self.phase = getattr(self.node, '_last_motion_tag', 'home')
        timeline(True)

    def gripper(self, position, duration, blocking):
        done = threading.Event()
        with self.lock:
            self.grip_plan = dict(target=position, duration=duration, done=done, start=None)
            self.phase = 'gripper'
        timeline(True)
        if not blocking:
            return None
        if not done.wait(timeout=60):
            raise TimeoutError('Sim-time gripper execution timed out')
        return bool(self.node._last_motion_status and self.node._last_motion_status.get('ok'))

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
                elif elapsed > duration + 5:
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
                elif elapsed > plan['duration'] + 5:
                    raise TimeoutError('Gripper did not converge in sim time')
            clean = corrective_action(state, self.reference)
            executed = list(clean)
            if noise_active and self.rng.random() < self.params['noise_probability']:
                for i in range(5):
                    executed[i] += max(-.012, min(.012, self.rng.gauss(0., self.params['noise_std_rad'])))
            for i, name in enumerate(JOINTS):
                lo, hi = JOINT_LIMITS[name]
                executed[i] = max(lo, min(hi, executed[i]))
            self.last_command_clock_ns = self.node.get_clock().now().nanoseconds
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
            phase = self.phase

        if completed:
            timeline(False)  # pause BEFORE letting the planner advance to the next step
            self.node._last_motion_status = dict(ok=True, outcome='sim_tick_converged')
            for done in completed:
                done.set()
        return clean, executed, active, phase, list(self.reference), self.last_command_clock_ns


class Collector:
    def __init__(self, args):
        self.args = args
        self.node = PickPlaceNode(use_sim_time=True)
        self.sensors = Sensors(self.node)
        self.servo = Servo(self.node)
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
        self.capture_thread = threading.Thread(target=self.capture, daemon=True)
        self.capture_thread.start()

    def capture(self):
        while not self.stop.is_set():
            try:
                try:
                    stamp, entry = self.sensors.read(timeout=30)
                except TimeoutError:
                    if self.servo.plan or self.servo.grip_plan:
                        raise TimeoutError('Motion stalled: no synchronized camera/state tick within 30 wall seconds')
                    continue
                joint, received = entry['joint']
                mapped = dict(zip(joint.name, joint.position))
                if not all(name in mapped for name in JOINTS):
                    raise ValueError('Incomplete joint state')
                state = [mapped[name] for name in JOINTS]
                clean, executed, active, phase, reference, command_clock = self.servo.tick(stamp, state)
                row = dict(sim_ns=stamp, wall_ns=time.monotonic_ns(), joint_stamp_ns=ns(joint.header.stamp),
                    joint_receive_ns=received, wrist_stamp_ns=ns(entry['wrist'][0].header.stamp),
                    top_stamp_ns=ns(entry['top'][0].header.stamp), wrist_receive_ns=entry['wrist'][1],
                    top_receive_ns=entry['top'][1], state=state, action=clean, executed_action=executed,
                    expert_observation_stamp_ns=stamp, command_clock_ns=command_clock,
                    expert_reference=reference, executing=active, phase=phase)
                with self.node.objects_lock:
                    row['target_object_pose'] = dict(self.node.objects_data.get(self.node._selected_object, {}))
                with self.writer_lock:
                    if self.writer:
                        self.writer.add(row, {key: Sensors.image(entry[key][0]).copy() for key in ('wrist', 'top')})
            except Exception as exc:
                message = repr(exc)
                self.node._qs_abort_evt.set()
                self.node._last_motion_status = dict(ok=False, outcome='capture_error', msg=message)
                if isinstance(exc, TimeoutError):
                    # A step deadline fails this attempt; the next seed gets a fresh motion plan.
                    with self.servo.lock:
                        done = [plan['done'] for plan in (self.servo.plan, self.servo.grip_plan) if plan]
                        self.servo.plan = self.servo.grip_plan = None
                        self.servo.reference = [self.node._actual_positions[name] for name in JOINTS]
                    try:
                        timeline(False)
                    except Exception as pause_error:
                        self.failure = repr(pause_error)
                    for event in done:
                        event.set()
                else:
                    self.failure = message
                if getattr(self.node, '_motion_event', None):
                    self.node._motion_event.set()
                if self.failure:
                    return

    def wait_ready(self):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.failure:
                raise RuntimeError(self.failure)
            if self.node.objects_data and len(self.node._drop_data) == 3 and self.servo.last_sim_ns:
                for client in (self.node.plan_client, self.node.validity_client, self.node._apply_scene_client):
                    if not client.wait_for_service(timeout_sec=5):
                        raise RuntimeError('MoveIt service unavailable')
                return
            self.stop.wait(.1)
        raise TimeoutError('Collection ROS graph did not become ready')

    def scene(self, params, baseline):
        timeline(False)
        if self.node._attached_lego_name:
            self.node._detach_lego_sync()
        # Restore a common baseline so seed reproduction does not depend on the previous episode.
        code = 'from so_arm101_dt.extension import get_instance, CUP_LAYOUT\nfrom pxr import Gf\nimport random\nimport numpy as np\ni=get_instance()\n'
        code += 'CUP_LAYOUT.clear()\nCUP_LAYOUT.update(' + repr(self.baseline_cup_layout) + ')\n'
        code += 'baseline=' + repr(baseline) + '\n'
        code += "for name, pos in baseline.items():\n    i._set_obj_prim_pose(i._get_prim_path(name), Gf.Vec3d(*pos), (0.,0.,0.))\n    p=__import__('omni.usd', fromlist=['']).get_context().get_stage().GetPrimAtPath(i._get_prim_path(name))\n    from pxr import UsdPhysics\n    rb=UsdPhysics.RigidBodyAPI(p)\n    rb.GetVelocityAttr().Set(Gf.Vec3f(0.,0.,0.))\n    rb.GetAngularVelocityAttr().Set(Gf.Vec3f(0.,0.,0.))\n"
        code += 'random.seed(' + str(params['seed']) + ')\nnp.random.seed(' + str(params['seed'] % 2**32) + ')\nresult=True'
        mcp('run_python', {'code': code})
        cups = mcp('soarm101.randomize_cups', dict(seed=params['cup_seed'], **params['cup_sampling']))
        mcp('run_python', {'code': 'from so_arm101_dt.extension import get_instance\nimport random\nimport numpy as np\nrandom.seed(' + str(params['seed']) + ')\nnp.random.seed(' + str(params['seed'] % 2**32) + ')\nresult=get_instance()._cmd_randomize_object_poses(required_visible_color=' + repr(params['color']) + ', required_visible_count=2)'})
        # Resume only to get the post-teleport poses, then freeze while scene updates/planning run.
        previous = self.servo.last_sim_ns
        timeline(True)
        deadline = time.monotonic() + 15
        while self.servo.last_sim_ns < previous + 200000000:
            if self.failure or time.monotonic() > deadline:
                raise RuntimeError(self.failure or 'Post-randomize sim did not advance')
            self.stop.wait(.02)
        timeline(False)
        self.node._remove_lego_collision_objects()
        self.node._remove_cup_collision_objects()
        self.node._add_cup_collision_objects()
        self.node._add_lego_collision_objects()
        # Verify the asynchronous planning-scene updates reached MoveIt before accepting the scene.
        expected = set(self.node._cup_collision_names + self.node._lego_collision_names)
        deadline = time.monotonic() + 15
        from moveit_msgs.srv import GetPlanningScene
        while True:
            req = GetPlanningScene.Request(); req.components.components = 16
            response = self.node._wait_future(self.node._get_scene_client.call_async(req), timeout_sec=3)
            actual = {c.id for c in response.scene.world.collision_objects} if response else set()
            if expected and expected <= actual:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f'Planning scene missing {expected - actual}')
            self.stop.wait(.05)
        return cups

    def run(self):
        args = self.args
        root = args.output.resolve()
        allowed_root = Path(os.getenv('SOARM_COLLECTION_ROOT', '/mnt/a4500-backup/exploring-vlas')).resolve()
        if not root.is_relative_to(allowed_root):
            raise ValueError(f'Bulk collection output must be inside {allowed_root}')
        root.mkdir(parents=True, exist_ok=False)
        timeline(True)
        self.wait_ready()
        clock_config = mcp('run_python', {'code': (REPO / 'linux-env/scripts/collect/configure_clock.py').read_text() + '\nresult=configure(30)'})
        self.node._cmd_toggle_ground_plane()
        timeline(False)
        if args.replay_receipt:
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
                           'linux-env/scripts/collect/configure_clock.py',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pick_place.py',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/grasp_geometry.py',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/compute_workspace.py',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/ik_calibration.yaml',
                           'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/workspace_bounds.yaml']
        source_hashes = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in tracked_sources}
        (root / 'run.json').write_text(json.dumps(dict(instance=args.instance, host=socket.gethostname(), gpu_index=int(os.getenv('ISAACSIM_GPU', '0')),
            clock_config=clock_config, domain=90 + args.instance,
            bridge_port=8777 + 2 * args.instance, fps=30, baseline=baseline, baseline_cup_layout=self.baseline_cup_layout, eval_seeds=EVAL_SEEDS,
            started=time.time(), source_sha256=source_hashes, git_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()), indent=2))
        accepted = 0
        attempted = 0
        start = time.monotonic()
        for trial in range(args.max_attempts):
            if accepted >= args.episodes:
                break
            if self.stop.is_set() or self.failure:
                raise RuntimeError(self.failure or 'Collection interrupted')
            if __import__('shutil').disk_usage('/').free < 90 * 1024**3:
                raise RuntimeError('Root filesystem below 90 GiB free')
            attempted += 1
            color = COLORS[accepted % 3]
            params = (json.loads(args.replay_receipt.read_text())['params'] if args.replay_receipt
                      else parameters(args.seed + trial, color))
            params['trial'] = trial
            path = root / f'attempt-{trial:04d}'
            receipt = dict(params=params, accepted=False)
            duplicate_start = dict(self.sensors.duplicate_messages)
            writer = None
            try:
                # Home with the same shared primitive used by Quickstart, with sim ticks active.
                self.node._motion_event = threading.Event()
                self.node._cmd_grasp_home_data()
                if not self.node._motion_event.wait(timeout=90) or not self.node._last_motion_status.get('ok'):
                    raise RuntimeError('Pre-episode home failed')
                self.servo.configure(params)
                receipt['cup_randomization'] = self.scene(params, baseline)
                self.node.approach_offset = params['approach_offset_m']
                self.node._grasp_approach_height_var.set(params['approach_height_m'])
                self.node.speed_scale = params['speed_scale']
                receipt['camera_duplicate_messages_before_episode'] = dict(self.sensors.duplicate_messages)
                receipt['initial_objects'] = dict(self.node.objects_data)
                receipt['initial_cups'] = dict(self.node._drop_data)
                writer = EpisodeWriter(path, 30)
                with self.writer_lock:
                    self.writer = writer
                cycle_result = {}
                cycle = threading.Thread(target=lambda: cycle_result.update(ok=self.node.run_cycle(params['object'])), daemon=True)
                cycle.start(); cycle.join(timeout=args.episode_deadline)
                if cycle.is_alive():
                    self.node._qs_abort_evt.set()
                    timeline(False)
                    cycle.join(timeout=5)
                    if cycle.is_alive():
                        self.failure = 'Timed-out cycle did not stop; refusing overlapping episodes'
                    raise TimeoutError('Episode wall deadline exceeded')
                if not cycle_result.get('ok'):
                    raise RuntimeError('Scripted cycle failed: ' + repr(self.node._last_motion_status))
                # The return-home segment gives the released block several sim seconds to settle.
                lego = self.node.objects_data[params['object']]
                cup_name = self.node._qs_auto_drop_for_lego(params['object'])
                cup = self.node._drop_data[cup_name]
                distance = math.hypot(lego['x'] - cup['x'], lego['y'] - cup['y'])
                receipt['score'] = dict(distance_m=distance, lego=lego, cup=cup)
                if distance > .033 or not (cup['z'] - .04825 < lego['z'] < cup['z'] + .060):
                    raise RuntimeError('Physical success check failed')
                receipt['accepted'] = True
            except Exception as exc:
                receipt['error'] = repr(exc)
                timeline(False)
            finally:
                with self.writer_lock:
                    self.writer = None
                if writer:
                    writer.close()
                    receipt['timing'] = timing_audit(writer.rows, 30)
                    receipt['accepted'] = receipt['accepted'] and receipt['timing']['passed']
                    if receipt.get('score'):
                        initial_z = receipt['initial_objects'][params['object']]['z']
                        maximum_z = max((row.get('target_object_pose', {}).get('z', initial_z) for row in writer.rows), default=initial_z)
                        receipt['score']['lift_m'] = maximum_z - initial_z
                        if maximum_z - initial_z < .025:
                            receipt['accepted'] = False
                            receipt['error'] = 'Object did not lift at least 25 mm before placement'
                receipt['camera_duplicate_messages'] = {key: value - duplicate_start.get(key, 0) for key, value in self.sensors.duplicate_messages.items()}
                path.mkdir(parents=True, exist_ok=True)
                (path / 'receipt.json').write_text(json.dumps(receipt, indent=2))
                with (root / 'episodes.jsonl').open('a') as log:
                    log.write(json.dumps(receipt) + '\n')
            if receipt['accepted']:
                accepted += 1
            else:
                # Keep every failed seed and diagnosis; delete only this attempt's two known bulk files.
                for key in ('wrist', 'top'):
                    video = path / f'{key}.mp4'
                    if video.exists():
                        video.unlink()
            print(json.dumps(dict(trial=trial, accepted=accepted, target=args.episodes,
                yield_fraction=accepted / (trial + 1), error=receipt.get('error'), timing=receipt.get('timing'))), flush=True)
        summary = dict(complete=accepted == args.episodes, episodes=accepted, attempts=attempted,
            yield_fraction=accepted / attempted, wall_seconds=time.monotonic() - start,
            episodes_per_wall_hour=accepted * 3600 / max(time.monotonic() - start, 1), end_marker='COLLECTION_FINISHED')
        (root / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        if not summary['complete']:
            raise RuntimeError('Attempt budget exhausted before the requested success count')

    def close(self):
        self.stop.set()
        timeline(False)
        self.executor.shutdown(timeout_sec=5)
        self.node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', type=int, required=True, choices=range(10))
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=310000)
    parser.add_argument('--replay-receipt', type=Path)
    parser.add_argument('--max-attempts', type=int, default=250)
    parser.add_argument('--episode-deadline', type=float, default=180)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.replay_receipt:
        args.episodes, args.max_attempts = 1, 1
    if args.episodes < 1 or args.episodes > 50:
        parser.error('This collector is limited to the validated 50-episode pilot')
    if os.environ.get('ROS_DOMAIN_ID') != str(90 + args.instance):
        parser.error('ROS_DOMAIN_ID does not match the assigned instance')
    os.environ['SOARM_KIT_MCP_PORT'] = str(8777 + 2 * args.instance)
    rclpy.init()
    collector = Collector(args)
    signal.signal(signal.SIGINT, lambda *_: collector.stop.set())
    signal.signal(signal.SIGTERM, lambda *_: collector.stop.set())
    try:
        collector.run()
    finally:
        collector.close()


if __name__ == '__main__':
    main()
