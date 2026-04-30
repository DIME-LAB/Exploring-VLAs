#!/usr/bin/env python3
# Reference: /home/aaugus11/Projects/Exploring-VLAs/delete_after_ingest
#            (LeRobot async inference launcher, sim-adapted)
"""Sim-adapted launcher for LeRobot async inference (policy server + robot client).

Spawns both subprocesses, prefixes their output with [SERVER]/[CLIENT], waits
for the policy server's port to come up before launching the client, and
tears both down cleanly on Ctrl+C or client exit.

Differences vs the real-hardware reference:
  * `--robot.type=so101_ros2` (our fork's ROS2-backed plugin) instead of
    `so101_follower` (Feetech serial).
  * `--robot.actuate=true` enables our plugin's send_action → FJT action
    client dispatch (defaults to no-op for backward-compat with the
    recording path).
  * No `--robot.port` / `--robot.id` — sim has no serial / per-arm
    calibration; the URDF and ros2_control launch handle that.
  * Cameras point at ROS2 topics (`/wrist_camera_rgb_sim`,
    `/workspace_camera_sim`) instead of OpenCV / RealSense.
  * Server runs in the same pixi-Jazzy env as the client (no remote-server
    decoupling for sim). Easy to split later by changing `--server-host`.

Usage:
    python3 run_async_inference_sim.py \\
        --pretrained=anirudhrani/smolvla_sim_100ep_fft__10ksteps_h200 \\
        --task='Pick a blue lego and place it in blue cup'

    # Show what would run, without launching:
    python3 run_async_inference_sim.py --dry-run

Press Ctrl+C to stop. Both subprocesses get SIGTERM and a 5 s grace period.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# Sim defaults — matches record_sim_isaac.sh's topic + units contract.
DEFAULTS = dict(
    server_host="127.0.0.1",
    server_port=8080,
    fps=30,

    robot_type="so101_ros2",
    cameras={
        "wrist": {
            "type": "ros2",
            "topic": "/wrist_camera_rgb_sim",
            "encoding": "rgba8",
            "width": 640, "height": 480, "fps": 30,
        },
        "top": {
            "type": "ros2",
            "topic": "/workspace_camera_sim",
            "encoding": "rgba8",
            "width": 640, "height": 480, "fps": 30,
        },
    },

    policy_type="smolvla",
    pretrained="anirudhrani/smolvla_sim_100ep_fft__10ksteps_h200",
    policy_device="cuda",

    task="Pick a blue lego and place it in blue cup",

    # RTC params — same as the real-hardware reference. chunk_size_threshold
    # at 0.5 means request the next chunk once 50% of the executing one has
    # been consumed; aggregate_fn_name=weighted_average blends the new
    # chunk's leading actions with the executing chunk's tail for smoother
    # transitions (no hard-switch jitter).
    actions_per_chunk=50,
    chunk_size_threshold=0.5,
    aggregate_fn_name="weighted_average",
)


def wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def stream_output(prefix: str, proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        sys.stdout.write(f"[{prefix}] {line}\n")
        sys.stdout.flush()


def kill_proc(proc: subprocess.Popen, name: str, grace_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        print(f"[launcher] {name} did not exit in {grace_s:.0f}s; killing.")
        proc.kill()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Sim async inference launcher (so101_ros2 + ros2_control).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--server-host", default=DEFAULTS["server_host"])
    p.add_argument("--server-port", type=int, default=DEFAULTS["server_port"])
    p.add_argument("--fps", type=int, default=DEFAULTS["fps"])

    p.add_argument("--robot-type", default=DEFAULTS["robot_type"])
    p.add_argument(
        "--cameras-json", default=json.dumps(DEFAULTS["cameras"]),
        help="Cameras dict as JSON. Default points at Isaac Sim's "
             "/wrist_camera_rgb_sim and /workspace_camera_sim topics.",
    )

    p.add_argument("--policy-type", default=DEFAULTS["policy_type"])
    p.add_argument(
        "--pretrained", default=DEFAULTS["pretrained"],
        help="HF repo id OR path to a `pretrained_model` dir.",
    )
    p.add_argument("--policy-device", default=DEFAULTS["policy_device"])

    p.add_argument("--task", default=DEFAULTS["task"])
    p.add_argument("--actions-per-chunk", type=int,
                   default=DEFAULTS["actions_per_chunk"])
    p.add_argument("--chunk-size-threshold", type=float,
                   default=DEFAULTS["chunk_size_threshold"])
    p.add_argument("--aggregate-fn", default=DEFAULTS["aggregate_fn_name"])

    # ---- Real-Time Chunking (RTC) — server-side smoothness ----------------
    # RTC adds a guidance term to the flow-matching denoising step so the
    # next chunk's first ~execution_horizon actions are GENERATED to match
    # the executed tail of the previous chunk. Cuts inter-chunk velocity
    # discontinuities at the source (vs aggregate_fn which only blends
    # post-hoc). Docs: https://huggingface.co/docs/lerobot/rtc
    # Threaded into the policy_server subprocess via env vars (the upstream
    # server CLI doesn't expose RTC flags; our policy_server.py patch reads
    # them at startup — see LEROBOT_RTC_* env block there).
    p.add_argument("--rtc.enabled", dest="rtc_enabled", action="store_true",
                   default=True,
                   help="Enable RTC on the policy server. Default ON because "
                        "every flow-matching policy benefits and it's "
                        "additive with aggregate_fn=weighted_average.")
    p.add_argument("--no-rtc", dest="rtc_enabled", action="store_false",
                   help="Disable RTC (revert to aggregate-only smoothness).")
    p.add_argument("--rtc.execution-horizon", dest="rtc_execution_horizon",
                   type=int, default=10,
                   help="How many timesteps from prev chunk to maintain "
                        "consistency with. Higher = smoother but less reactive.")
    p.add_argument("--rtc.max-guidance-weight", dest="rtc_max_guidance_weight",
                   type=float, default=10.0,
                   help="Strength of consistency enforcement during denoising. "
                        "10.0 is optimal for 10-step flow matching (SmolVLA).")
    p.add_argument("--rtc.schedule", dest="rtc_schedule",
                   default="EXP", choices=["LINEAR", "EXP", "ONES", "ZEROS"],
                   help="prefix_attention_schedule: how guidance weight decays "
                        "across the overlap region. EXP is the docs-recommended "
                        "default.")

    p.add_argument("--server-startup-timeout", type=float, default=120.0,
                   help="Seconds to wait for server to start listening "
                        "(SmolVLA + base VLM cold-load is ~30-60 s).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print both commands and exit without launching.")
    args = p.parse_args()

    server_address = f"{args.server_host}:{args.server_port}"

    server_cmd = [
        sys.executable, "-u", "-m", "lerobot.async_inference.policy_server",
        f"--host={args.server_host}",
        f"--port={args.server_port}",
        f"--fps={args.fps}",
    ]
    client_cmd = [
        sys.executable, "-u", "-m", "lerobot.async_inference.robot_client",
        f"--robot.type={args.robot_type}",
        # Sim-specific: actuate ON (drives ros2_control), use_degrees ON
        # (matches recorded dataset's units convention).
        "--robot.actuate=true",
        "--robot.use_degrees=true",
        f"--robot.cameras={args.cameras_json}",
        f"--task={args.task}",
        f"--server_address={server_address}",
        f"--policy_type={args.policy_type}",
        f"--pretrained_name_or_path={args.pretrained}",
        f"--policy_device={args.policy_device}",
        f"--actions_per_chunk={args.actions_per_chunk}",
        f"--chunk_size_threshold={args.chunk_size_threshold}",
        f"--aggregate_fn_name={args.aggregate_fn}",
    ]

    print("=" * 72)
    print("  LeRobot Async Inference (sim) Launcher")
    print("=" * 72)
    print(f"  Server     : {server_address}  (fps={args.fps})")
    print(f"  Policy     : {args.policy_type}  device={args.policy_device}")
    print(f"  Checkpoint : {args.pretrained}")
    print(f"  Robot      : {args.robot_type}  actuate=true  use_degrees=true")
    print(f"  Cameras    : keys={list(json.loads(args.cameras_json).keys())}")
    print(f"  Task       : {args.task}")
    print(f"  RTC        : actions_per_chunk={args.actions_per_chunk}  "
          f"threshold={args.chunk_size_threshold}  agg={args.aggregate_fn}")
    print("=" * 72)

    if args.dry_run:
        print("\n[server cmd]")
        print("  " + " \\\n      ".join(server_cmd))
        print("\n[client cmd]")
        print("  " + " \\\n      ".join(client_cmd))
        return 0

    # Local pretrained-path validation. HF repo IDs (no '/' in the first
    # segment, OR a '/' but path doesn't exist locally) are passed through
    # to the client, which downloads them.
    if Path(args.pretrained).expanduser().is_dir():
        print(f"[launcher] using local checkpoint: {args.pretrained}")
    else:
        print(f"[launcher] treating --pretrained as HF repo id: {args.pretrained}")

    # Build server env: pass RTC flags through env vars so the patched
    # policy_server picks them up at startup (no upstream CLI plumbing needed).
    server_env = os.environ.copy()
    if args.rtc_enabled:
        server_env["LEROBOT_RTC_ENABLED"] = "1"
        server_env["LEROBOT_RTC_EXECUTION_HORIZON"] = str(args.rtc_execution_horizon)
        server_env["LEROBOT_RTC_MAX_GUIDANCE_WEIGHT"] = str(args.rtc_max_guidance_weight)
        server_env["LEROBOT_RTC_PREFIX_ATTENTION_SCHEDULE"] = args.rtc_schedule
        print(f"[launcher] RTC ON: horizon={args.rtc_execution_horizon}, "
              f"weight={args.rtc_max_guidance_weight}, schedule={args.rtc_schedule}")
    else:
        print("[launcher] RTC OFF (aggregate-only smoothness)")

    print("\n[launcher] Starting policy server...")
    server = subprocess.Popen(server_cmd, env=server_env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    server_thread = threading.Thread(
        target=stream_output, args=("SERVER", server), daemon=True,
    )
    server_thread.start()

    print(f"[launcher] Waiting up to {args.server_startup_timeout:.0f}s for "
          f"{server_address} to listen...")
    if not wait_for_port(args.server_host, args.server_port, args.server_startup_timeout):
        print("[launcher] Server did not become ready in time.")
        kill_proc(server, "server")
        server_thread.join(timeout=2)
        return 1
    print("[launcher] Server is up.\n")

    print("[launcher] Starting robot client...")
    client = subprocess.Popen(client_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    client_thread = threading.Thread(
        target=stream_output, args=("CLIENT", client), daemon=True,
    )
    client_thread.start()

    rc = 0
    try:
        rc = client.wait()
    except KeyboardInterrupt:
        print("\n[launcher] Ctrl+C received; shutting down.")
        kill_proc(client, "client")
        rc = 130
    finally:
        kill_proc(server, "server")
        client_thread.join(timeout=2)
        server_thread.join(timeout=2)

    print(f"\n[launcher] Done. Client exit code: {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
