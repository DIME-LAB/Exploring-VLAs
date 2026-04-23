#!/usr/bin/env python3
"""Sim ground-truth publisher for SO-ARM101 worlds.

Subscribes to Gazebo world pose info via ``gz.transport13`` (NOT through
``ros_gz_bridge`` — the bridge's ``gz.msgs.Pose_V → tf2_msgs/TFMessage``
conversion drops ``pose.name`` and ``pose.id``, leaving us unable to tell
which transform belongs to which object). Filters to a curated catalog of
known objects and republishes as standard ROS2 topics:

* ``/objects_poses_sim`` (``tf2_msgs/TFMessage``) at 10 Hz — mirrors the
  ``aruco_camera_localizer`` real-side contract. ``child_frame_id`` = object
  name, ``header.frame_id`` = world name.
* ``/objects_bbox_sim`` (``std_msgs/String`` JSON) at 1 Hz — ``{"name":
  {"sx": ..., "sy": ..., "sz": ...}}``, dimensions in meters.

Together these let ``control_gui`` treat sim + real as interchangeable
data sources — same topic names, same message shapes, same callbacks, no
branching.
"""

import json
import os
import threading
from typing import Dict, Optional

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


DEFAULT_OBJECTS_YAML = os.path.join(
    get_package_share_directory('sim_ground_truth'), 'config',
    'lego_world_objects.yaml',
)


def _load_objects_catalog(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, 'r') as fh:
        data = yaml.safe_load(fh) or {}
    return data.get('objects', {}) or {}


def _load_drop_entities(path: str) -> list:
    """Load the drop_entities list from the catalog YAML (empty list if absent)."""
    with open(path, 'r') as fh:
        data = yaml.safe_load(fh) or {}
    raw = data.get('drop_entities', []) or []
    # Tolerate scalar / dict entries; only str names enter the filter set.
    return [str(n) for n in raw if isinstance(n, str)]


class GroundTruthPublisher(Node):
    def __init__(self) -> None:
        super().__init__('sim_ground_truth_publisher')

        self.declare_parameter('world_name', 'so_arm101_lego_world')
        self.declare_parameter('gz_pose_topic', '')  # '' → /world/<name>/pose/info
        self.declare_parameter('poses_topic', '/objects_poses_sim')
        self.declare_parameter('bbox_topic', '/objects_bbox_sim')
        self.declare_parameter('drop_poses_topic', '/drop_poses')
        self.declare_parameter('poses_rate_hz', 10.0)
        self.declare_parameter('bbox_rate_hz', 1.0)
        self.declare_parameter('drop_poses_rate_hz', 10.0)
        self.declare_parameter('objects_yaml', DEFAULT_OBJECTS_YAML)
        self.declare_parameter('log_unknown_entities', False)

        self._world_name = str(self.get_parameter('world_name').value)
        gz_topic = str(self.get_parameter('gz_pose_topic').value)
        if not gz_topic:
            gz_topic = f'/world/{self._world_name}/pose/info'
        poses_topic = str(self.get_parameter('poses_topic').value)
        bbox_topic = str(self.get_parameter('bbox_topic').value)
        drop_topic = str(self.get_parameter('drop_poses_topic').value)
        poses_rate = float(self.get_parameter('poses_rate_hz').value)
        bbox_rate = float(self.get_parameter('bbox_rate_hz').value)
        drop_rate = float(self.get_parameter('drop_poses_rate_hz').value)
        objects_yaml = str(self.get_parameter('objects_yaml').value)
        self._log_unknown = bool(self.get_parameter('log_unknown_entities').value)

        try:
            self._catalog = _load_objects_catalog(objects_yaml)
            drop_entities = _load_drop_entities(objects_yaml)
        except FileNotFoundError:
            self.get_logger().error(
                f"objects_yaml {objects_yaml!r} not found — publishing nothing"
            )
            self._catalog = {}
            drop_entities = []
        self._wanted = set(self._catalog.keys())
        self._drop_wanted = set(drop_entities)

        self._latest: Dict[str, TransformStamped] = {}
        self._latest_drops: Dict[str, TransformStamped] = {}
        self._latest_lock = threading.Lock()
        self._unknown_seen: set = set()

        self._poses_pub = self.create_publisher(TFMessage, poses_topic, 10)
        self._bbox_pub = self.create_publisher(String, bbox_topic, 1)
        self._drop_pub = self.create_publisher(TFMessage, drop_topic, 10)

        # Gazebo transport subscription — runs on gz's own thread pool.
        self._gz_node = GzNode()
        subscribed = self._gz_node.subscribe(Pose_V, gz_topic, self._on_gz_pose_v)
        if not subscribed:
            self.get_logger().error(f"gz.transport13 subscribe to {gz_topic!r} failed")
        else:
            self.get_logger().info(f"gz subscribed: {gz_topic}")

        if poses_rate > 0:
            self.create_timer(1.0 / poses_rate, self._publish_poses)
        if bbox_rate > 0:
            self.create_timer(1.0 / bbox_rate, self._publish_bbox)
        if drop_rate > 0 and self._drop_wanted:
            self.create_timer(1.0 / drop_rate, self._publish_drops)

        self.get_logger().info(
            f"Ground truth: {len(self._wanted)} objects + {len(self._drop_wanted)} "
            f"drop entities from {objects_yaml} | world={self._world_name} → "
            f"poses={poses_topic}@{poses_rate:g}Hz, "
            f"bbox={bbox_topic}@{bbox_rate:g}Hz, "
            f"drops={drop_topic}@{drop_rate:g}Hz"
        )

    def _on_gz_pose_v(self, msg: Pose_V) -> None:
        """Gazebo Transport callback — runs off the rclpy executor thread."""
        stamp = self.get_clock().now().to_msg()
        updated: Dict[str, TransformStamped] = {}
        updated_drops: Dict[str, TransformStamped] = {}
        for p in msg.pose:
            name = p.name
            if not name:
                continue
            if name in self._wanted or name in self._drop_wanted:
                ts = TransformStamped()
                ts.header.stamp = stamp
                ts.header.frame_id = self._world_name
                ts.child_frame_id = name
                ts.transform.translation.x = p.position.x
                ts.transform.translation.y = p.position.y
                ts.transform.translation.z = p.position.z
                ts.transform.rotation.x = p.orientation.x
                ts.transform.rotation.y = p.orientation.y
                ts.transform.rotation.z = p.orientation.z
                ts.transform.rotation.w = p.orientation.w
                if name in self._wanted:
                    updated[name] = ts
                if name in self._drop_wanted:
                    updated_drops[name] = ts
            elif self._log_unknown and name not in self._unknown_seen:
                self._unknown_seen.add(name)
                self.get_logger().info(f"seen entity not in catalog: {name!r}")
        if updated or updated_drops:
            with self._latest_lock:
                if updated:
                    self._latest.update(updated)
                if updated_drops:
                    self._latest_drops.update(updated_drops)

    def _publish_poses(self) -> None:
        with self._latest_lock:
            transforms = list(self._latest.values())
        if not transforms:
            return
        self._poses_pub.publish(TFMessage(transforms=transforms))

    def _publish_drops(self) -> None:
        with self._latest_lock:
            transforms = list(self._latest_drops.values())
        if not transforms:
            return
        self._drop_pub.publish(TFMessage(transforms=transforms))

    def _publish_bbox(self) -> None:
        payload: Dict[str, Dict[str, float]] = {
            name: {'sx': float(d.get('sx', 0.0)),
                   'sy': float(d.get('sy', 0.0)),
                   'sz': float(d.get('sz', 0.0))}
            for name, d in self._catalog.items()
        }
        self._bbox_pub.publish(String(data=json.dumps(payload)))


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = GroundTruthPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
