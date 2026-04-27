"""L2 smoke test: full lerobot dataset written from ROS2 topics, then reloaded.

Simulates a SO-ARM101-ish 6-DOF observation (5 arm joints + gripper) plus one
camera, end-to-end through PR #866's ROS2Camera + standalone rclpy subscribers
for joint_states. Actions are synthesized as (state + small random delta) so we
exercise the action column meaningfully.

Pipeline:
  [smoke_publisher]   → /smoke/image             → ROS2Camera
  [js_publisher]      → /smoke/joint_states      → rclpy subscriber
  for each tick: add_frame(); → save_episode(); → LeRobotDataset.from_disk()
                                                   and validate shape/columns.

Run (three terminals):
  T1: pixi run python scripts/smoke_publisher.py --topic /smoke/image
  T2: pixi run python scripts/smoke_l2_dataset.py --help  (see below)
  (joint_states is synthesized INSIDE this script to keep the demo tight.)
"""
import argparse
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.cameras.configs import ROS2CameraConfig
from lerobot.common.robot_devices.cameras.ros2 import ROS2Camera

ARM_JOINT_NAMES = [
    "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw",
]


class JointStatePublisher(Node):
    """Synthesizes a 6-DOF JointState message stream at `fps`."""

    def __init__(self, topic: str, fps: int):
        super().__init__("smoke_js_publisher")
        self.pub = self.create_publisher(JointState, topic, 10)
        self.t0 = time.time()
        self.timer = self.create_timer(1.0 / fps, self.tick)

    def tick(self):
        t = time.time() - self.t0
        # Synth positions: each joint oscillates with a different period/amplitude.
        pos = np.array([
            0.5 * np.sin(t * 0.7),
            0.4 * np.sin(t * 0.9 + 0.3),
            0.3 * np.sin(t * 1.1 + 0.6),
            0.2 * np.sin(t * 1.3 + 0.9),
            0.6 * np.sin(t * 0.5 + 1.2),
            0.5 + 0.4 * np.sin(t * 2.0),  # gripper, 0..1-ish
        ], dtype=np.float32)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ARM_JOINT_NAMES
        msg.position = pos.tolist()
        self.pub.publish(msg)


class JointStateSubscriber(Node):
    """Stores the latest JointState position vector (in ARM_JOINT_NAMES order)."""

    def __init__(self, topic: str):
        super().__init__("smoke_js_subscriber")
        self.latest: np.ndarray | None = None
        self.sub = self.create_subscription(JointState, topic, self.cb, 10)

    def cb(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        self.latest = np.array(
            [name_to_pos[n] for n in ARM_JOINT_NAMES if n in name_to_pos],
            dtype=np.float32,
        )


def spin_nodes_in_thread(nodes: list[Node]) -> tuple[SingleThreadedExecutor, threading.Thread]:
    """Drive callbacks for multiple nodes via a single-threaded executor in a thread."""
    executor = SingleThreadedExecutor()
    for n in nodes:
        executor.add_node(n)
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()
    return executor, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-topic", default="/smoke/image")
    ap.add_argument("--js-topic", default="/smoke/joint_states")
    ap.add_argument("--repo-id", default="local/smoke_soarm101")
    ap.add_argument("--root", default="/tmp/lerobot_smoke_dataset")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    # Clean output
    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)

    # --- Camera first — ROS2Camera.__init__ owns rclpy.init() & its spin thread.
    cam_cfg = ROS2CameraConfig(
        topic=args.image_topic,
        encoding="bgr8",
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    cam = ROS2Camera(cam_cfg)  # rclpy.init() happens here
    cam.connect()

    # --- JS pub + sub now that rclpy is alive. -----------------------------
    js_pub = JointStatePublisher(args.js_topic, args.fps * 2)
    js_sub = JointStateSubscriber(args.js_topic)
    js_executor, _js_thread = spin_nodes_in_thread([js_pub, js_sub])

    # Wait for first joint_states.
    t0 = time.perf_counter()
    while js_sub.latest is None:
        if time.perf_counter() - t0 > 5.0:
            raise TimeoutError("no joint_states received within 5s")
        time.sleep(0.05)
    state_dim = js_sub.latest.shape[0]
    print(f"Got first joint_state vector of dim={state_dim}")

    # --- Dataset ----------------------------------------------------------
    features = {
        "observation.images.top": {
            "dtype": "video",
            "shape": (args.height, args.width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": list(ARM_JOINT_NAMES[:state_dim]),
        },
        "action": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": list(ARM_JOINT_NAMES[:state_dim]),
        },
    }
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=root,
        features=features,
        use_videos=True,
        robot_type="so_arm101_smoke",
    )

    rng = np.random.default_rng(0)
    period = 1.0 / args.fps
    for i in range(args.frames):
        loop_start = time.perf_counter()
        img = cam.read()
        state = js_sub.latest.copy()
        # Synth action: next-step state + noise.
        action = (state + rng.normal(0, 0.01, size=state.shape)).astype(np.float32)
        ds.add_frame({
            "observation.images.top": img,
            "observation.state": state,
            "action": action,
            "task": "smoke test: synth dataset from ROS2 topics",
        })
        if i % 5 == 0:
            print(f"  [{i}/{args.frames}] state={np.round(state, 3)}")
        # Pace to fps.
        elapsed = time.perf_counter() - loop_start
        if elapsed < period:
            time.sleep(period - elapsed)

    print("saving episode …")
    ds.save_episode()

    # Teardown sub/pub.
    cam.disconnect()
    js_executor.shutdown()
    js_pub.destroy_node()
    js_sub.destroy_node()
    # NOTE: rclpy.shutdown is owned by ROS2Camera.__del__ here.

    # --- Reload + verify --------------------------------------------------
    print("\nreloading dataset from disk …")
    reloaded = LeRobotDataset(args.repo_id, root=root)
    print(f"  num_episodes: {reloaded.num_episodes}")
    print(f"  num_frames:   {reloaded.num_frames}")
    print(f"  features:     {list(reloaded.features.keys())}")
    sample = reloaded[0]
    print(f"  sample keys:  {list(sample.keys())}")
    img_tensor = sample["observation.images.top"]
    state_tensor = sample["observation.state"]
    action_tensor = sample["action"]
    print(f"  image tensor: shape={tuple(img_tensor.shape)} dtype={img_tensor.dtype}")
    print(f"  state tensor: shape={tuple(state_tensor.shape)} dtype={state_tensor.dtype}")
    print(f"  action tensor: shape={tuple(action_tensor.shape)} dtype={action_tensor.dtype}")

    # On-disk layout check
    expected = [
        root / "meta" / "info.json",
        root / "meta" / "episodes.jsonl",
        root / "meta" / "tasks.jsonl",
        root / "data" / "chunk-000" / "episode_000000.parquet",
        root / "videos" / "chunk-000" / "observation.images.top" / "episode_000000.mp4",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        print("\n[L2 FAIL] missing on-disk files:")
        for p in missing:
            print(f"  - {p}")
        raise SystemExit(1)

    # Content assertions
    assert reloaded.num_episodes == 1, reloaded.num_episodes
    assert reloaded.num_frames == args.frames, reloaded.num_frames
    assert state_tensor.shape == (state_dim,), state_tensor.shape
    assert action_tensor.shape == (state_dim,), action_tensor.shape
    # Image: HF LeRobotDataset returns (C, H, W) float tensors after decoding.
    assert tuple(img_tensor.shape)[-2:] == (args.height, args.width), img_tensor.shape
    print("\n[L2 PASS] dataset written, reloaded, shape-checked")


if __name__ == "__main__":
    main()
