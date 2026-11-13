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
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
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


def scrubbed_env():
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
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "1", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if not uuid.startswith("GPU-"):
        raise RuntimeError("GPU 1 UUID unavailable")
    env["CUDA_VISIBLE_DEVICES"] = uuid
    return env


class Ros:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node("policy_episode_harness")
        self.lock = threading.Lock()
        self.data = {"objects": {}, "drops": {}, "joints": {}}
        self.received = {k: 0.0 for k in self.data}
        self.controller_status = {}
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
        self.node.create_subscription(JointState, "/joint_states", self.joints, 10)
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
            line = self.lines.get(timeout=max(0, deadline - time.monotonic()))
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
        "time_s",
        "actions_sent",
        "chunks",
        "effective_action_fps",
        "max_blue_lift_m",
        "failure_note",
        "video",
    ]
    path = out / "results.csv"
    new = not path.exists()
    with path.open("a") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(result)


def main():
    p = build_parser()
    p.description = __doc__
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-s", type=float, default=45)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--success-hold-s", type=float, default=2)
    p.add_argument("--condition", default="unnamed")
    p.add_argument("--seeds", help="Comma-separated scene seeds, exactly one per episode")
    p.add_argument("--scene-bank", type=Path, help="Shared physical scene bank for paired conditions")
    p.add_argument("--sim-rate-hz", type=float, default=17.4, help="Kit frame ceiling, applied after each reset")
    p.add_argument("--fresh-client-per-episode", action="store_true", help="Diagnose persistent-client resource accumulation")
    args = parse_args(parser=p)
    args.seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(range(900000, 900000 + args.episodes))
    if len(args.seeds) != args.episodes or len(set(args.seeds)) != args.episodes:
        p.error("Provide exactly one unique seed per episode")
    args.scene_bank = (args.scene_bank or Path.home() / "w_policy_phase1/1.9/scene-bank").expanduser().resolve()
    cfg = args
    args.out = args.out.expanduser().resolve()
    if args.episodes < 1 or args.max_s <= 0 or args.sim_rate_hz <= 0:
        raise ValueError("Positive episodes/max-s required")
    if args.out.exists() and (args.out / "results.csv").exists():
        raise FileExistsError("Choose a new output directory; results already exist")
    if cfg.dry_run:
        server_cmd, client_cmd = build_commands(cfg, python="python3")
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
    env = scrubbed_env()
    env.update(
        {k: v for k, v in rtc_environment(cfg).items() if k.startswith("LEROBOT_RTC_")}
    )
    metadata = dict(vars(cfg))
    metadata.update(
        gpu_uuid=env["CUDA_VISIBLE_DEVICES"],
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
                                    SCRIPTS / "run_async_inference_sim.py"]},
    )
    (args.out / "run_config.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    commands = build_commands(cfg, python="python3")
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
        if not wait_for_port(
            cfg.server_host, cfg.server_port, cfg.server_startup_timeout
        ):
            raise RuntimeError("Policy server startup failed")
        # Same client configuration entrance as the deployment launcher.
        client_cmd = (
            pixi
            + ["python3", "-u", str(Path(__file__).with_name("episode_client.py"))]
            + commands[1][4:]
        )
        client = Client(client_cmd, env, args.out)
        print("READY", client.tagged("READY", 240), flush=True)
        ros = Ros()
        deadline = time.monotonic() + 15
        while True:
            try:
                ros.snapshot()
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)
        for index in range(args.episodes):
            if index and args.fresh_client_per_episode:
                client.close()
                client = Client(client_cmd, env, args.out / f"client-{index:02d}")
                print("READY", client.tagged("READY", 240), flush=True)
            ep = f"ep{index:02d}"
            epdir = args.out / ep
            epdir.mkdir()
            result = {
                "episode": ep,
                "outcome": "error",
                "condition": args.condition,
                "seed": args.seeds[index],
                "rtc_enabled": cfg.rtc_enabled,
                "requested_fps": cfg.fps,
                "time_to_grasp_s": None,
                "time_to_place_s": None,
                "time_s": 0.0,
                "actions_sent": 0,
                "chunks": 0,
                "max_blue_lift_m": 0.0,
                "failure_note": "",
                "video": "",
            }
            recording = False
            running = False
            try:
                result["home_error_deg"] = ros.home()
                result["controller_status_start"] = ros.status_snapshot()
                result["randomize"] = mcp("run_python", {"code":
                    f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); "
                    f"import scene_reset; result = scene_reset.prepare({args.seeds[index]}, {str(args.scene_bank)!r})"})
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
                starts = {k: v[2] for k, v in objects.items() if k.startswith("blue")}
                blue_in_cup(objects, drops)
                mcp(
                    "soarm101.start_recording",
                    {"file_name": str(epdir / "viewport_capture.mp4")},
                )
                recording = True
                rec_t = time.monotonic()
                client.send(f"RUN {ep} {args.max_s}")
                client.tagged("EPSTART", 15)
                running = True
                started = time.monotonic()
                inside_since = None
                lift_since = {k: None for k in starts}
                lifted = False
                max_lift = 0.0
                success = False
                done = {}
                with (epdir / "poses.jsonl").open("w") as poses:
                    while time.monotonic() - started < args.max_s + 15:
                        try:
                            done = client.tagged("EPDONE", 0.1)
                            running = False
                            break
                        except queue.Empty:
                            pass
                        objects, drops, joints = ros.snapshot()
                        now = time.monotonic()
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
                        if inside and not lifted:
                            inside = False
                        inside_since = (inside_since or now) if inside else None
                        poses.write(
                            json.dumps(
                                {
                                    "t": now - started,
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
                objects, drops, _ = ros.snapshot()
                inside, detail = blue_in_cup(objects, drops)
                success = success and inside
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
            except Exception as error:
                result["outcome"] = "error"
                result["failure_note"] = f"{type(error).__name__}: {error}"
                if client.fatal:
                    result["fatal_client"] = client.fatal
                raise
            finally:
                cleanup_error = None
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
