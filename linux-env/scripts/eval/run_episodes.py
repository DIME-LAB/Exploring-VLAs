#!/usr/bin/env python3
"""Interim episode runner; Isaac and ros2_control must already be running."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import queue
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatusArray
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState, Image
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectoryPoint

SCRIPTS = Path(__file__).resolve().parents[1]
REPO = SCRIPTS.parents[1]
sys.path.insert(0, str(SCRIPTS))
from run_async_inference_sim import (
    build_commands,
    build_parser,
    parse_args,
    rtc_environment,
    wait_for_port,
)

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
HOME_DEG = [0.874, -14.829, 14.829, 90.0, -90.0 + math.degrees(0.0487)]
CUP_R, CUP_H = 0.033, 0.0965  # conservative interior-centre test (outer radius ~.039m)


def major_faults(pid):
    if pid is None:
        return None
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[9])
    except (FileNotFoundError, PermissionError):
        return None


def mcp(tool, params=None):
    env = dict(os.environ, PYTHONPATH=str(REPO / "omni-kit-mcp"))
    proc = subprocess.run(
        [
            "/usr/bin/python3",
            "-m",
            "kit_mcp.client",
            tool,
            json.dumps(params or {}),
            "--port",
            os.getenv("SOARM_KIT_MCP_PORT", "8767"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"{tool}: {proc.stderr[-1000:]} {proc.stdout[-1000:]}")
    result = json.loads(proc.stdout)
    if tool == "run_python":
        return result["result"]
    if result.get("status") != "success":
        raise RuntimeError(f"{tool}: {result}")
    return result


def scrubbed_env(sim_time=False):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("ROS_", "AMENT_", "COLCON_", "PYTHON", "CMAKE_"))
    }
    env["LD_LIBRARY_PATH"] = ":".join(
        p
        for p in env.get("LD_LIBRARY_PATH", "").split(":")
        if p and "/opt/ros" not in p
    )
    env.update(
        RMW_IMPLEMENTATION="rmw_cyclonedds_cpp",
        CYCLONEDDS_URI=f"file://{REPO}/linux-env/cyclonedds.xml",
    )
    if sim_time:
        if os.environ.get("ROS_DOMAIN_ID") != "92":
            raise RuntimeError("Phase3 live eval requires explicit ROS_DOMAIN_ID=92")
        env.update(ROS_DOMAIN_ID="92",
                   CYCLONEDDS_URI=f"file://{REPO}/linux-env/cyclonedds-eval.xml")
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "0" if sim_time else "1", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if not uuid.startswith("GPU-"):
        raise RuntimeError("Requested policy GPU UUID unavailable")
    env["CUDA_VISIBLE_DEVICES"] = uuid
    return env


class Ros:
    def __init__(self, sim_time=False, joint_topic="/joint_states"):
        self.physical = None
        self.clock_s = None
        self.physical_samples = []
        rclpy.init()
        self.node = rclpy.create_node("policy_episode_harness")
        self.lock = threading.Lock()
        self.data = {"objects": {}, "drops": {}, "joints": {}}
        self.received = {k: 0.0 for k in self.data}
        self.controller_status = {}
        self.arrivals = {k: [] for k in ("clock", "wrist", "workspace")}
        for label, topic, msgtype in (("clock", "/clock", Clock),
                ("wrist", "/wrist_camera_rgb_sim", Image),
                ("workspace", "/workspace_camera_sim", Image)):
            self.node.create_subscription(msgtype, topic,
                lambda m, label=label: self.arrival(label, m), qos_profile_sensor_data)
        for label in ("arm", "gripper"):
            self.node.create_subscription(
                GoalStatusArray, f"/{label}_controller/follow_joint_trajectory/_action/status",
                lambda m, label=label: self.status(label, m),
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.node.create_subscription(
            TFMessage, "/objects_poses_sim", lambda m: self.tf("objects", m), 10
        )
        self.node.create_subscription(
            TFMessage, "/drop_poses", lambda m: self.tf("drops", m), 10
        )
        self.node.create_subscription(JointState, joint_topic, self.joints, 10)
        if sim_time:
            from std_msgs.msg import String
            self.node.create_subscription(String, "/collection_tick", self.physical_received, 100)
        self.arm = ActionClient(
            self.node, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )
        self.grip = ActionClient(
            self.node,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
        )
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.node)
        self.thread = threading.Thread(target=self.ex.spin, daemon=True)
        self.thread.start()

    def physical_received(self, message):
        snapshot = json.loads(message.data)
        with self.lock:
            self.physical = snapshot
            self.physical_samples.append(snapshot)

    def physical_drain(self):
        with self.lock:
            samples, self.physical_samples = self.physical_samples, []
        return samples

    def update(self, key, value):
        with self.lock:
            self.data[key] = value
            self.received[key] = time.monotonic()

    def tf(self, key, msg):
        out = {}
        for t in msg.transforms:
            p, q = t.transform.translation, t.transform.rotation
            out[t.child_frame_id] = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        self.update(key, out)

    def joints(self, msg):
        self.update("joints", dict(zip(msg.name, msg.position)))

    def status(self, label, msg):
        with self.lock:
            previous = self.controller_status.get(label, {})
            self.controller_status[label] = dict(
                latest_size=len(msg.status_list),
                max_size=max(len(msg.status_list), previous.get("max_size", 0)),
                messages=previous.get("messages", 0)+1)

    def status_snapshot(self):
        with self.lock:
            return {k: dict(v) for k,v in self.controller_status.items()}

    def arrival(self, label, msg):
        stamp = msg.clock if label == "clock" else msg.header.stamp
        with self.lock:
            self.arrivals[label].append((time.time(), stamp.sec+stamp.nanosec/1e9))
            if label == "clock":
                self.clock_s = stamp.sec+stamp.nanosec/1e9

    def reset_arrivals(self):
        with self.lock:
            self.arrivals = {k: [] for k in self.arrivals}

    def arrival_snapshot(self):
        with self.lock:
            data = {k: list(v) for k,v in self.arrivals.items()}
        return {k: dict(samples=v, max_wall_gap_s=max((b[0]-a[0] for a,b in zip(v,v[1:])),default=None))
                for k,v in data.items()}

    def snapshot(self):
        with self.lock:
            ages = {k: time.monotonic() - v for k, v in self.received.items()}
            if max(ages.values()) > 2:
                raise RuntimeError(f"Missing/stale ROS state (seconds): {ages}")
            return tuple(dict(self.data[k]) for k in ("objects", "drops", "joints"))

    @staticmethod
    def wait(future, seconds):
        deadline = time.monotonic() + seconds
        while not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError("ROS action response timed out")
            time.sleep(0.02)
        return future.result()

    def fjt(self, client, names, positions, duration):
        if not client.wait_for_server(timeout_sec=5):
            raise RuntimeError(f"FJT server missing: {names}")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = names
        goal.trajectory.points = [
            JointTrajectoryPoint(
                positions=positions,
                time_from_start=Duration(sec=int(duration), nanosec=0),
            )
        ]
        handle = self.wait(client.send_goal_async(goal), 5)
        if not handle.accepted:
            raise RuntimeError("Home trajectory rejected")
        result = self.wait(handle.get_result_async(), duration + 10)
        if result.result.error_code:
            raise RuntimeError(f"Home failed: {result.result.error_string}")

    def home(self):
        self.fjt(self.arm, ARM_JOINTS, [math.radians(x) for x in HOME_DEG], 3)
        self.fjt(self.grip, ["gripper_joint"], [math.radians(6.49)], 1)
        _, _, joints = self.snapshot()
        error = max(
            abs(math.degrees(joints.get(n, float("inf"))) - d)
            for n, d in zip(ARM_JOINTS, HOME_DEG)
        )
        if error > 2:
            raise RuntimeError(f"Home pose error {error:.2f}deg")
        return error

    def close(self):
        self.ex.shutdown()
        self.node.destroy_node()
        rclpy.shutdown()
        self.thread.join(timeout=3)


def blue_in_cup(objects, drops):
    if "drop_1" not in drops:
        raise RuntimeError("drop_1 missing from /drop_poses")
    cup = drops["drop_1"]
    cx, cy, cz = cup[:3]
    x, y, z, w = cup[3:]
    # Inverse quaternion rotates world displacement into cup-local coordinates.
    rot = (
        (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)),
        (2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)),
        (2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)),
    )
    candidates = []
    for name, p in objects.items():
        if not name.startswith("blue"):
            continue
        d = [p[0] - cx, p[1] - cy, p[2] - cz]
        local = [sum(a * b for a, b in zip(row, d)) for row in rot]
        r = math.hypot(*local[:2])
        h = local[2] + CUP_H / 2
        candidates.append(
            {
                "lego": name,
                "r_xy_m": r,
                "z_above_base_m": h,
                "inside": r < CUP_R and 0.004 < h < CUP_H - 0.006,
            }
        )
    if not candidates:
        raise RuntimeError("No blue lego poses")
    best = min(candidates, key=lambda c: (not c["inside"], c["r_xy_m"]))
    return best["inside"], best


def inference_health(server, streams, tails):
    """Promote native RPC/model errors before they can resemble starvation."""
    if server.poll() is not None:
        raise RuntimeError(f'Owned policy server exited: {server.returncode}')
    needles = ('Error in StreamActions:', 'Error receiving actions:')
    for i, (stream, needle) in enumerate(zip(streams, needles)):
        new_log = tails[i]+stream.read()
        tails[i] = new_log[-128:]
        if needle in new_log:
            raise RuntimeError('Native policy inference/RPC failed; see logs; episode is invalid')


class Client:
    def __init__(self, cmd, env, out):
        out.mkdir(parents=True, exist_ok=True)
        self.log = (out / "client.log").open("w")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.lines = queue.Queue()
        self.fatal = None
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(
            "FATAL " + json.dumps({"error": f"client exited {self.proc.poll()}"})
        )

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def tagged(self, tag, timeout):
        deadline = time.monotonic() + timeout
        while True:
            health = getattr(self, 'health_check', None)
            if health:
                health()
            try:
                line = self.lines.get(timeout=min(.2, max(0, deadline-time.monotonic())))
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise
                continue
            if line.startswith("FATAL"):
                self.fatal = json.loads(line[len("FATAL"):])
                self.log.write(line)
                self.log.flush()
                raise RuntimeError(self.fatal.get("error", str(self.fatal)))
            if line.startswith(tag + " "):
                return json.loads(line[len(tag) :])
            self.log.write(line)
            self.log.flush()

    def close(self):
        if self.proc.poll() is None:
            self.send("ABORT")
            self.send("QUIT")
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=10)
        self.log.close()


def save_result(out, result):
    with (out / "results.jsonl").open("a") as f:
        f.write(json.dumps(result) + "\n")
    fields = [
        "episode",
        "outcome",
        "condition",
        "seed",
        "rtc_enabled",
        "requested_fps",
        "sim_cap_hz",
        "time_to_grasp_s",
        "time_to_place_s",
        "empty_ticks_after_startup",
        "empty_ticks_pct",
        "first_chunk_s",
        "request_chunk_mean_ms",
        "request_chunk_p95_ms",
        "request_queue_depth_mean",
        "chunk_size_threshold",
        "actions_per_chunk",
        "time_s",
        "actions_sent",
        "chunks",
        "effective_action_fps",
        "max_blue_lift_m",
        "failure_note",
        "video", "clock_basis", "color", "task", "suite", "phrase", "wall_time_s", "sim_time_s",
        "grasp_success", "correct_cup", "any_placement", "closing_distance_m", "closest_target_distance_m",
        "sim_dispatch_jitter_p95_ms", "total_episode_wall_s",
    ]
    path = out / "results.csv"
    new = not path.exists()
    with path.open("a") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(result)


def add_telemetry(result, epdir, cfg):
    client = result.get("client", result.get("fatal_client", {}))
    metrics = client.get("metrics", client)
    empty = metrics.get("empty_action_ticks_after_startup", 0)
    opportunities = metrics.get("actions_sent", 0)+empty+metrics.get("missed_action_ticks_after_startup", 0)
    requests = {r["observation_timestamp"]: r for r in client.get("request_events", []) if "observation_timestamp" in r}
    chunks = []
    for event in client.get("chunk_events", []):
        request = requests.get(event["observation_timestamp"], {})
        chunks.append(dict(event,
            request_to_arrival_s=event["arrival_wall_time"]-request["trigger_wall_time"] if request else None,
            queue_depth_at_capture_request=request.get("queue_depth_at_capture_request"),
            trigger_wall_time=request.get("trigger_wall_time"),
            capture_rpc_s=request.get("capture_rpc_s")))
    latencies = sorted(1000*c["request_to_arrival_s"] for c in chunks if c["request_to_arrival_s"] is not None)
    depths = [c["queue_depth_at_capture_request"] for c in chunks if c["queue_depth_at_capture_request"] is not None]
    if "sim_dispatch_jitter_p95_ms" in metrics:
        result["sim_dispatch_jitter_p95_ms"] = metrics["sim_dispatch_jitter_p95_ms"]
    result.update(empty_ticks_after_startup=empty, empty_ticks_pct=100*empty/opportunities if opportunities else None,
        first_chunk_s=metrics.get("first_chunk_s"),
        request_chunk_mean_ms=statistics.mean(latencies) if latencies else None,
        request_chunk_p95_ms=latencies[math.ceil(.95*len(latencies))-1] if latencies else None,
        request_queue_depth_mean=statistics.mean(depths) if depths else None,
        chunk_trace=chunks, chunk_size_threshold=cfg.chunk_size_threshold, actions_per_chunk=cfg.actions_per_chunk)
    if chunks:
        with (epdir / "chunks.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(chunks[0]))
            writer.writeheader();writer.writerows(chunks)


def episode_client_command(command, corrected=False):
    return command[:2]+[str(Path(__file__).with_name("episode_client.py"))]+command[4:]+(
        ["--dataset-brightness"] if corrected else [])


def main():
    p = build_parser()
    p.description = __doc__
    p.add_argument("--episode-manifest", type=Path, help="Ordered Phase3 rows; changes task/colour only while idle")
    p.add_argument("--eval-actuation", choices=("native", "immediate"), default="native",
                   help="SIM-only immediate targets; real hardware requires its own decision")
    p.add_argument("--sim-time", action="store_true", help="Opt-in /clock dispatch and physical Phase3 scoring")
    p.add_argument("--diagnostic-reobserve-actions", type=int, choices=(10,25,50), default=10,
                   help="Execution-horizon diagnostic only; frozen MAIN remains 10")
    p.add_argument("--diagnostic-prediction-trace", action="store_true",
                   help="Keep postprocessed predicted chunks and capture sim stamps, off the dispatch path")
    p.add_argument("--training-start-bank-sha256", help="Explicit sealed five-training-scene start diagnostic; never MAIN")
    p.add_argument("--color", choices=("red", "green", "blue"), default="blue")
    p.add_argument("--warmup-actions", type=int, default=100)
    p.add_argument("--wall-watchdog-s", type=float, default=900)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-s", type=float, default=45)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--success-hold-s", type=float, default=2)
    p.add_argument("--condition", default="unnamed")
    p.add_argument("--seeds", help="Comma-separated scene seeds, exactly one per episode")
    p.add_argument("--scene-bank", type=Path, help="Shared physical scene bank for paired conditions")
    p.add_argument("--sim-rate-hz", type=float, default=17.4, help="Kit frame ceiling, applied after each reset")
    p.add_argument("--fresh-client-per-episode", action="store_true", help="Diagnose persistent-client resource accumulation")
    p.add_argument("--dataset-brightness", action="store_true", help="Diagnostic only: divide both policy RGB feeds by calibrated live gain")
    args = parse_args(parser=p)
    if args.eval_actuation == "immediate" and not args.sim_time:
        p.error("Immediate-target evaluation requires --sim-time")
    if args.diagnostic_reobserve_actions != 10 and not args.sim_time:
        p.error('Execution-horizon diagnostics require --sim-time')
    if args.diagnostic_prediction_trace and not args.sim_time:
        p.error('Prediction-chunk diagnostics require --sim-time')
    if args.training_start_bank_sha256 and not args.sim_time:
        p.error('Training-start diagnostics require --sim-time')
    if args.diagnostic_reobserve_actions == 50 and args.rtc_enabled:
        p.error('The 50-action execution diagnostic requires --no-rtc')
    if args.sim_time and args.fresh_client_per_episode:
        p.error("Phase3 uses one persistent client per window")
    if args.dataset_brightness and args.sim_time:
        p.error("Inverse-brightness diagnostic and sim-time capture cannot be combined")
    if args.dataset_brightness and not args.observation_process:
        p.error("--dataset-brightness requires isolated observation capture")
    args.seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(range(900000, 900000 + args.episodes))
    if len(args.seeds) != args.episodes or (not args.episode_manifest and len(set(args.seeds)) != args.episodes):
        p.error("Provide exactly one unique seed per episode")
    args.scene_bank = (args.scene_bank or (Path("/mnt/extreme4tb/aaugus11/exploring-vlas/eval/scene-bank-v1") if args.sim_time else Path.home() / "w_policy_phase1/1.9/scene-bank")).expanduser().resolve()
    episode_rows = json.loads(args.episode_manifest.read_text()) if args.episode_manifest else None
    if episode_rows is not None:
        if len(episode_rows) != args.episodes or [r["seed"] for r in episode_rows] != args.seeds:
            p.error("Episode manifest does not match requested seed order")
        if any(r["rtc"] != args.rtc_enabled for r in episode_rows):
            p.error("One persistent server must use one RTC arm")
    cfg = args
    if args.sim_time:
        from phase3 import validate_bank
        if args.training_start_bank_sha256:
            from training_starts import validate_bank as validate_training_bank, entry as training_entry
            training_manifest=validate_training_bank(args.scene_bank,args.training_start_bank_sha256)
            training_entries={seed:training_entry(args.scene_bank,seed,args.training_start_bank_sha256) for seed in args.seeds}
            if episode_rows is None or any(r['color']!=training_entries[r['seed']]['color'] or r['task']!=training_entries[r['seed']]['task'] or r['suite']!='seen-training-'+training_manifest['mode'] for r in episode_rows):
                p.error('Training diagnostic rows differ from accepted source episodes')
            args.scene_bank_sha256=args.training_start_bank_sha256
            args.training_start_mode=training_manifest['mode']
        else:args.scene_bank_sha256 = validate_bank(args.scene_bank)
        for name in ("ROS_LOCALHOST_ONLY", "ROS_AUTOMATIC_DISCOVERY_RANGE", "ROS_STATIC_PEERS"):
            os.environ.pop(name, None)
        os.environ["CYCLONEDDS_URI"] = f"file://{REPO}/linux-env/cyclonedds-eval.xml"
        os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
        if args.fps != 30 or args.joint_states_topic != "/joint_states_measured":
            p.error("--sim-time requires --fps 30 --joint-states-topic /joint_states_measured")
        args.cameras_json = json.dumps({k: json.loads(args.cameras_json)[k] for k in ("top", "wrist")})
    args.out = args.out.expanduser().resolve()
    if args.episodes < 1 or args.max_s <= 0 or args.sim_rate_hz <= 0:
        raise ValueError("Positive episodes/max-s required")
    if args.out.exists() and (args.out / "results.csv").exists():
        raise FileExistsError("Choose a new output directory; results already exist")
    if cfg.dry_run:
        server_cmd, client_cmd = build_commands(cfg, python="python3")
        client_cmd = episode_client_command(client_cmd, args.dataset_brightness)
        if args.sim_time:
            server_cmd = server_cmd[:2]+[str(Path(__file__).with_name("sim_time.py")), "sim"]+server_cmd[4:]
            client_cmd += ["--eval-clock=sim", f"--eval-observe-every-actions={args.diagnostic_reobserve_actions}", f"--eval-actuation={args.eval_actuation}"]
        print(
            json.dumps(
                {
                    "server": server_cmd,
                    "client": client_cmd,
                    "episodes": args.episodes,
                    "max_s": args.max_s,
                    "out": str(args.out),
                },
                indent=2,
            )
        )
        return
    args.out.mkdir(parents=True, exist_ok=True)
    env = scrubbed_env(args.sim_time)
    if args.diagnostic_prediction_trace:
        env['SOARM_EVAL_RECORD_CHUNKS'] = '1'
    env.update(
        {k: v for k, v in rtc_environment(cfg).items() if k.startswith("LEROBOT_RTC_")}
    )
    metadata = dict(vars(cfg))
    metadata.update(
        eval_actuation=args.eval_actuation,
        actuation_duration_s=0. if args.eval_actuation == "immediate" else cfg.action_duration_s,
        evaluation_arm="deployment-realism, actuation v2" if args.eval_actuation == "immediate" else "original deployment",
        gpu_uuid=env["CUDA_VISIBLE_DEVICES"],
        inference_mode=env.get("SOARM_EVAL_INFERENCE_MODE", "eager" if args.sim_time else "checkpoint-default"),
        host_cpu_count=os.cpu_count(),
        host_load_start=list(os.getloadavg()),
        started_utc=time.strftime("%FT%TZ", time.gmtime()),
        fork_commit=subprocess.check_output(
            ["git", "-C", str(REPO / "lerobot"), "rev-parse", "HEAD"], text=True
        ).strip(),
        repo_commit=subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [Path(__file__), Path(__file__).with_name("scene_reset.py"),
                                    Path(__file__).with_name("episode_client.py"),
                                    SCRIPTS / "run_async_inference_sim.py",
                                    Path(__file__).with_name("observation_transform.py")]+(
                                    [Path(__file__).with_name(name) for name in ("sim_time.py", "phase3.py", "phase3_metrics.py", "actuation.py")]+[
                                    REPO/"vla_SO-ARM101/src/so_arm101_control/so_arm101_control/compute_workspace.py",
                                    SCRIPTS/"collect/success.py", SCRIPTS/"collect/cup_geometry.py", SCRIPTS/"collect/pose_math.py"] if args.sim_time else [])},
    )
    if args.training_start_bank_sha256:
        source=Path(__file__).with_name('training_starts.py')
        metadata['source_sha256'][str(source.relative_to(REPO))]=hashlib.sha256(source.read_bytes()).hexdigest()
    (args.out / "run_config.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    commands = build_commands(cfg, python="python3")
    if args.sim_time:
        commands = (commands[0][:2]+[str(Path(__file__).with_name("sim_time.py")), "sim"]+commands[0][4:], commands[1])
    pixi = [
        str(Path.home() / ".pixi/bin/pixi"),
        "run",
        "--manifest-path",
        str(REPO / "linux-env/pixi.toml"),
    ]
    import socket

    try:
        with socket.create_connection((cfg.server_host, cfg.server_port), timeout=1):
            raise RuntimeError(
                "Policy port already occupied; choose --server-port (harness owns its server)"
            )
    except OSError:
        pass
    serverlog = (args.out / "server.log").open("w")
    server = subprocess.Popen(
        pixi + commands[0],
        env=env,
        stdout=serverlog,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    client = ros = None
    try:
        startup_deadline = time.monotonic()+cfg.server_startup_timeout
        while not wait_for_port(cfg.server_host, cfg.server_port, .5):
            if server.poll() is not None:
                raise RuntimeError(f"Policy server exited at startup: {server.returncode}; see server.log")
            if time.monotonic() >= startup_deadline:
                raise TimeoutError("Policy server startup deadline")
        # Same client configuration entrance as the deployment launcher.
        client_cmd = pixi+episode_client_command(commands[1], args.dataset_brightness)
        if args.sim_time:
            client_cmd += ["--eval-clock=sim", f"--eval-observe-every-actions={args.diagnostic_reobserve_actions}", f"--eval-actuation={args.eval_actuation}"]
        client = Client(client_cmd, env, args.out)
        health_log = (args.out/'server.log').open(errors='replace')
        client_health_log = (args.out/'client.log').open(errors='replace')
        health_tail = ['', '']
        def server_health():
            inference_health(server, (health_log, client_health_log), health_tail)
        client.health_check = server_health
        print("READY", client.tagged("READY", 900 if args.sim_time else 240), flush=True)
        ros = Ros(args.sim_time, args.joint_states_topic)
        deadline = time.monotonic() + 15
        while True:
            try:
                ros.snapshot()
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)
        if args.sim_time:
            from phase3 import require_deployment
            deployment = require_deployment(ros, mcp)
            (args.out / "deployment.json").write_text(json.dumps(deployment, indent=2)+"\n")
            client.send(f"WARM warmup 600 {args.warmup_actions}")
            client.tagged("EPSTART", 15)
            warmup = client.tagged("EPDONE", 900)
            if warmup["actions_sent"] < args.warmup_actions or warmup["chunks"] < 2:
                raise RuntimeError("Incomplete compiled/guided warmup")
            if not any(e.get("eval_rtc", {}).get("prefix_steps", 0)>0 for e in warmup.get("chunk_events", []) if e.get("eval_rtc")) and args.rtc_enabled:
                raise RuntimeError("Warmup did not exercise guided RTC prediction")
            (args.out / "warmup.json").write_text(json.dumps(warmup, indent=2)+"\n")
        for index in range(args.episodes):
            if index and args.fresh_client_per_episode:
                client.close()
                client = Client(client_cmd, env, args.out / f"client-{index:02d}")
                print("READY", client.tagged("READY", 900 if args.sim_time else 240), flush=True)
            if episode_rows is not None:
                row = episode_rows[index]
                cfg.color, cfg.task = row["color"], row["task"]
                cfg.condition = f'{row["suite"]}-{row["phrase"]}-{row["color"]}-rtc-{row["rtc"]}'
                client.send("TASK " + json.dumps(cfg.task))
            ep = f"ep{index:02d}"
            epdir = args.out / ep
            epdir.mkdir()
            result = {
                "episode": ep,
                "eval_actuation": args.eval_actuation,
                "evaluation_arm": "deployment-realism, actuation v2" if args.eval_actuation == "immediate" else "original deployment",
                "actuation_duration_s": 0. if args.eval_actuation == "immediate" else cfg.action_duration_s,
                "outcome": "error",
                "condition": args.condition,
                "seed": args.seeds[index],
                "rtc_enabled": cfg.rtc_enabled,
                "clock_basis": "simulation" if args.sim_time else "wall",
                "task": cfg.task, "color": args.color,
                "phrase": episode_rows[index]["phrase"] if episode_rows else "exact",
                "suite": episode_rows[index]["suite"] if episode_rows else args.condition,
                "requested_fps": cfg.fps,
                "observation_transform": {
                    "name": "inverse_dataset_gain" if args.dataset_brightness else "identity",
                    "enabled": args.dataset_brightness,
                    "divisor": [1.12162, 1.11828, 1.11650] if args.dataset_brightness else [1., 1., 1.],
                    "cameras": ["wrist", "top"],
                },
                "time_to_grasp_s": None,
                "time_to_place_s": None,
                "time_s": 0.0,
                "actions_sent": 0,
                "chunks": 0,
                "max_blue_lift_m": 0.0,
                "failure_note": "",
                "video": "",
            }
            episode_wall_started = time.monotonic()
            recording = False
            running = False
            try:
                result["home_error_deg"] = ros.home()
                if args.training_start_bank_sha256:
                    target=training_entries[args.seeds[index]]
                    ros.fjt(ros.arm,ARM_JOINTS,target['state'][:5],3)
                    ros.fjt(ros.grip,['gripper_joint'],target['state'][5:],1)
                    result['training_start']={k:v for k,v in target.items() if k!='bodies'}
                result["controller_status_start"] = ros.status_snapshot()
                result["randomize"] = mcp("run_python", {"code":
                    f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); "
                    f"import scene_reset, importlib; "
                    "import hashlib; from pathlib import Path; "
                    "scene_current=hashlib.sha256(Path(scene_reset.__file__).read_bytes()).hexdigest(); "
                    "scene_reset._FRAME_SUB = None if getattr(scene_reset, '_SOURCE_SHA256', None) != scene_current else scene_reset._FRAME_SUB; "
                    "scene_reset = importlib.reload(scene_reset) if getattr(scene_reset, '_SOURCE_SHA256', None) != scene_current else scene_reset; "
                    f"result = scene_reset.prepare({args.seeds[index]}, {str(args.scene_bank)!r}, {('red', 'green', 'blue') if args.sim_time else ('blue',)!r}, generate={not args.sim_time!r}"
                    +(f", training_start_sha256={args.training_start_bank_sha256!r}" if args.training_start_bank_sha256 else '')+")"})
                result["host_load_start"] = list(os.getloadavg())
                result["gpu_start"] = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv"], text=True)
                # Wait for a pose publication after randomization; monotonic receipt ages
                # guard data freshness despite zero ROS header stamps in older scenes.
                randomized = time.monotonic()
                deadline = randomized + 10
                while (
                    ros.received["objects"] < randomized
                    or ros.received["drops"] < randomized
                ):
                    if time.monotonic() > deadline:
                        raise TimeoutError("No post-randomization poses")
                    time.sleep(0.05)
                objects, drops, _ = ros.snapshot()
                result["sim_timing"] = mcp("run_python", {"code":
                    "s=carb.settings.get_settings(); "
                    "s.set_bool('/app/runLoops/main/syncToPresent', False); "
                    "s.set_bool('/app/runLoops/main/rateLimitEnabled', True); "
                    f"s.set_float('/app/runLoops/main/rateLimitFrequency', {args.sim_rate_hz!r}); "
                    "result={k:s.get(k) for k in ['/app/runLoops/main/rateLimitFrequency', "
                    "'/app/runLoops/main/rateLimitEnabled', '/app/runLoops/main/syncToPresent']}"})
                result["sim_cap_hz"] = result["sim_timing"]["/app/runLoops/main/rateLimitFrequency"]
                if abs(result["sim_cap_hz"] - args.sim_rate_hz) > 1e-4 or not result["sim_timing"]["/app/runLoops/main/rateLimitEnabled"]:
                    raise RuntimeError(f"Kit rate read-back mismatch: {result['sim_timing']}")
                result["isaac_probe_start"] = mcp("run_python", {"code":
                    "import importlib, scene_reset; "
                    "result = scene_reset.start_probe()"})
                starts = {k: v[2] for k, v in objects.items() if k.startswith("blue")}
                score = None
                if args.sim_time:
                    from phase3 import new_score
                    fence = time.monotonic()+15
                    while (ros.physical is None or ros.physical["epoch"] != result["randomize"]["reset_epoch"] or ros.physical["sim_ns"] <= result["randomize"]["reset_stamp_ns"]):
                        if time.monotonic()>fence:
                            raise TimeoutError("No post-reset physical scoring baseline")
                        time.sleep(.02)
                    baseline = dict(ros.physical, bodies=result["randomize"]["bodies"])
                    if args.training_start_bank_sha256:
                        target=training_entries[args.seeds[index]]['state']
                        handoff=ros.physical
                        qerror=max(abs(math.degrees(a-b)) for a,b in zip(handoff['state'],target))
                        derror=max(abs(math.degrees(a-b)) for a,b in zip(handoff['drive_targets'],target))
                        result['training_start']['held_handoff']=dict(sim_ns=handoff['sim_ns'],state=handoff['state'],drive_targets=handoff['drive_targets'],max_state_error_deg=qerror,max_drive_error_deg=derror)
                        if qerror>.25 or derror>1e-4:raise RuntimeError('Training start drifted or disagrees with controller-held targets')
                    result["physical_baseline"] = dict(epoch=ros.physical["epoch"], stamp_ns=ros.physical["sim_ns"], starts_from="verified saved reset bodies")
                    score = new_score(args.color, baseline)
                    ros.physical_drain()
                blue_in_cup(objects, drops)
                mcp(
                    "soarm101.start_recording",
                    {"file_name": str(epdir / "viewport_capture.mp4")},
                )
                recording = True
                rec_t = time.monotonic()
                ros.reset_arrivals()
                result["isaac_major_faults_start"] = major_faults(result["isaac_probe_start"]["pid"])
                client.send(f"RUN {ep} {args.max_s}")
                client.tagged("EPSTART", 15)
                running = True
                if args.sim_time:
                    ros.physical_drain()
                started = time.monotonic()
                started_sim = ros.clock_s
                if score is not None:
                    score.begin = started_sim
                inside_since = None
                lift_since = {k: None for k in starts}
                lifted = False
                max_lift = 0.0
                success = False
                done = {}
                with (epdir / "poses.jsonl").open("w") as poses, (epdir / "physical.jsonl").open("w") as physical_out:
                    while time.monotonic() - started < (args.wall_watchdog_s if args.sim_time else args.max_s + 15):
                        try:
                            done = client.tagged("EPDONE", 0.1)
                            server_health()
                            running = False
                            break
                        except queue.Empty:
                            pass
                        objects, drops, joints = ros.snapshot()
                        now = time.monotonic()
                        if score is not None:
                            for snapshot in ros.physical_drain():
                                if snapshot["epoch"] != result["randomize"]["reset_epoch"]:
                                    raise RuntimeError("Physical snapshot crossed reset epoch")
                                if not started_sim <= snapshot["sim_ns"]/1e9 <= started_sim+args.max_s:
                                    continue
                                score.update(snapshot)
                                physical_out.write(json.dumps(snapshot)+"\n")
                            scored = score.result()
                            if scored["success"] and not success:
                                success = True
                                client.send("ABORT")
                        inside, detail = blue_in_cup(objects, drops)
                        lift = max(
                            (
                                v[2] - starts[k]
                                for k, v in objects.items()
                                if k in starts
                            ),
                            default=0.0,
                        )
                        max_lift = max(max_lift, lift)
                        for name, z0 in starts.items():
                            is_lifted = name in objects and objects[name][2] - z0 > 0.024
                            lift_since[name] = (lift_since[name] or now) if is_lifted else None
                            if lift_since[name] is not None and now - lift_since[name] >= 0.5:
                                lifted = True
                                if result["time_to_grasp_s"] is None:
                                    result["time_to_grasp_s"] = lift_since[name] - started
                        if args.sim_time:
                            inside = False  # physical whole-block scorer above owns Phase3 outcomes
                        if inside and not lifted:
                            inside = False
                        inside_since = (inside_since or now) if inside else None
                        poses.write(
                            json.dumps(
                                {
                                    "t": (ros.clock_s-started_sim) if args.sim_time else now-started,
                                    "wall_t": now-started,
                                    "objects": objects,
                                    "drop_1": drops["drop_1"],
                                    "inside": inside,
                                    "detail": detail,
                                    "joints": joints,
                                }
                            )
                            + "\n"
                        )
                        if (
                            inside_since is not None
                            and now - inside_since >= args.success_hold_s
                            and not success
                        ):
                            success = True
                            result["time_to_place_s"] = inside_since - started
                            client.send("ABORT")
                    else:
                        raise TimeoutError("Client did not acknowledge episode stop")
                if done["actions_sent"] == 0 or done["chunks"] == 0:
                    raise RuntimeError("Policy produced no executable actions")
                if score is not None:
                    for snapshot in ros.physical_drain():
                        if snapshot["epoch"] != result["randomize"]["reset_epoch"]:
                            raise RuntimeError("Final physical snapshot crossed reset epoch")
                        if started_sim <= snapshot["sim_ns"]/1e9 <= started_sim+min(args.max_s, done["duration_s"]):
                            score.update(snapshot)
                objects, drops, _ = ros.snapshot()
                inside, detail = blue_in_cup(objects, drops)
                success = success and inside if not args.sim_time else score.result()["success"]
                result.update(
                    outcome="success"
                    if success
                    else "grasped-not-placed"
                    if lifted
                    else "no-grasp",
                    time_s=done["duration_s"],
                    actions_sent=done["actions_sent"],
                    chunks=done["chunks"],
                    effective_action_fps=done["effective_action_fps"],
                    max_blue_lift_m=max_lift,
                    final=detail,
                    host_load_end=list(os.getloadavg()),
                    controller_status_end=ros.status_snapshot(),
                    client=done,
                    failure_note=""
                    if success
                    else "timeout; sustained blue lift only"
                    if lifted
                    else "timeout; no sustained blue lift",
                )
                if score is not None:
                    result.update(score.result())
                    result["outcome"] = "success" if result["success"] else "grasped-not-placed" if result["grasp_success"] else "no-grasp"
                    result["failure_note"] = "" if result["success"] else "sim-time budget; see physical characterization"
                    result["wall_time_s"] = done["wall_duration_s"]
                    result["sim_time_s"] = done["duration_s"]
            except (Exception, KeyboardInterrupt) as error:
                result["outcome"] = "error"
                result["failure_note"] = f"{type(error).__name__}: {error}"
                if client.fatal:
                    result["fatal_client"] = client.fatal
                raise
            finally:
                cleanup_error = None
                result["transport_arrivals"] = ros.arrival_snapshot()
                result["isaac_major_faults_end"] = major_faults(result.get("isaac_probe_start",{}).get("pid"))
                try:
                    result["isaac_frames"] = mcp("run_python", {"code": "import scene_reset; result = scene_reset.probe()"})
                except Exception as probe_error:
                    result["isaac_probe_error"] = str(probe_error)
                if running:
                    try:
                        client.send("ABORT")
                        stopped_client = client.tagged("EPDONE", 30)
                        result.update(
                            time_s=stopped_client["duration_s"],
                            actions_sent=stopped_client["actions_sent"],
                            chunks=stopped_client["chunks"],
                            effective_action_fps=stopped_client["effective_action_fps"],
                            client=stopped_client,
                        )
                    except Exception as error:  # noqa: BLE001 -- recording still must stop
                        cleanup_error = error
                        result["failure_note"] += f"; client stop: {error}"
                try:
                    if recording:
                        stopped = mcp("soarm101.stop_recording")
                        wall = time.monotonic() - rec_t
                        source = Path(stopped["output_path"])
                        duration = float(
                            subprocess.check_output(
                                [
                                    "ffprobe",
                                    "-v",
                                    "error",
                                    "-show_entries",
                                    "format=duration",
                                    "-of",
                                    "default=nw=1:nk=1",
                                    str(source),
                                ],
                                text=True,
                            )
                        )
                        if stopped.get("frame_count", 0) < 2:
                            raise RuntimeError("Empty viewport recording")
                        video = epdir / "viewport.mp4"
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-v",
                                "error",
                                "-y",
                                "-itsscale",
                                str(wall / duration),
                                "-i",
                                str(source),
                                "-c",
                                "copy",
                                str(video),
                            ],
                            check=True,
                        )
                        result.update(
                            video=str(video), recording_wall_s=wall, recording=stopped
                        )
                        source.unlink()
                except Exception as error:  # noqa: BLE001 -- preserve result before propagating cleanup failure
                    cleanup_error = error
                    result["outcome"] = "fail"
                    result["failure_note"] += (
                        f"; cleanup {type(error).__name__}: {error}"
                    )
                finally:
                    if args.sim_time:
                        try:
                            server_health()
                        except Exception as health_error:
                            result["outcome"]="error"
                            result["failure_note"] += f"; final health: {health_error}"
                            cleanup_error=health_error
                    result["total_episode_wall_s"] = time.monotonic()-episode_wall_started
                    add_telemetry(result, epdir, cfg)
                    (epdir / "result.json").write_text(
                        json.dumps(result, indent=2) + "\n"
                    )
                    save_result(args.out, result)
                    print(json.dumps({k: result.get(k) for k in (
                        "episode", "condition", "seed", "outcome", "time_s",
                        "effective_action_fps", "failure_note", "video")}), flush=True)
                if cleanup_error is not None:
                    raise cleanup_error

        ros.home()
    finally:
        try:
            if client:
                try:
                    if args.sim_time:
                        try:
                            server_health()
                        except Exception as health_error:
                            (args.out/"window-health.json").write_text(json.dumps(dict(passed=False,error=str(health_error)))+"\n")
                            raise
                        (args.out/"window-health.json").write_text(json.dumps(dict(passed=True))+"\n")
                finally:
                    client.close()
        finally:
            try:
                if ros:
                    ros.close()
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait()
                serverlog.close()


if __name__ == "__main__":
    main()
