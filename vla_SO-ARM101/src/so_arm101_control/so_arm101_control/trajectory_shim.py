#!/usr/bin/env python3
"""trajectory_shim — execute MoveIt trajectories on real SO-ARM101 hardware.

Action servers (matching moveit_controllers.yaml):
  /arm_controller/follow_joint_trajectory     control_msgs/FollowJointTrajectory
  /gripper_controller/follow_joint_trajectory control_msgs/FollowJointTrajectory

Plays back trajectories at ``rate_hz`` on ``joint_commands_topic``.
``jointstatereader`` (with ``accept_command_input=true``) writes those
JointState setpoints to the Feetech follower servos.

Goal completion is hybrid:
  * Time: all waypoint times must elapse (scaled by 1/velocity_scale).
  * Position: final-waypoint joints must be within ``tolerance_rad``
    of the most recent ``/joint_states``.
  * Timeout: if tolerance isn't met within ``completion_timeout_s``
    after the trajectory ends, abort with GOAL_TOLERANCE_VIOLATED.

Parameters:
  rate_hz              (20.0)   how often to publish setpoints
  velocity_scale       (0.25)   multiplier on time_from_start.
                                 <1 plays the trajectory slower.
  tolerance_rad        (0.052)  completion tolerance per joint (3°)
  completion_timeout_s (1.5)    extra wait after trajectory ends
  joint_commands_topic (/joint_commands)
  joint_states_topic   (/joint_states)

Joint-name remap (MoveIt → URDF/jointstatereader):
  ``gripper`` → ``gripper_joint``   (config split between MoveIt and URDF)

Cancel: ``send_cancel_goal`` makes the playback loop exit on its next tick.
The last published setpoint sticks (servos remain torque-on at that pose).
For an emergency torque-off, kill the shim and restart the stack with
``read_only:=true``.

Preemption: a newer goal arriving for a group signals the in-flight goal
via a ``threading.Event``; the old playback loop sees the flag, calls
``goal_handle.canceled()``, releases the per-group lock, and the new goal
takes over. Lock-acquire timeout is 2s — if the old loop fails to drain
that fast, the new goal aborts. Designed for FK-tab streaming where a
slider can fire 20+ goals/s; old "lock.acquire(blocking=False) → reject"
behavior dropped 96% of those goals on bench testing.
"""

import bisect
import math
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState


# MoveIt's controller config names this joint 'gripper' but the URDF /
# jointstatereader / Feetech bus all call it 'gripper_joint'. Translate
# at the publish boundary so jointstatereader's name-keyed dict picks it up.
_NAME_REMAP = {
    'gripper': 'gripper_joint',
}


def _interpolate(times, positions, t):
    """Linear interp at time t across waypoints. Returns list[J]."""
    if t <= times[0]:
        return list(positions[0])
    if t >= times[-1]:
        return list(positions[-1])
    i = bisect.bisect_right(times, t) - 1
    j = i + 1
    span = times[j] - times[i] or 1e-9
    a = (t - times[i]) / span
    return [positions[i][k] + a * (positions[j][k] - positions[i][k])
            for k in range(len(positions[i]))]


def _remap_names(names):
    return [_NAME_REMAP.get(n, n) for n in names]


class TrajectoryShim(Node):
    def __init__(self):
        super().__init__('trajectory_shim')

        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('velocity_scale', 0.25)
        self.declare_parameter('tolerance_rad', math.radians(3.0))
        self.declare_parameter('completion_timeout_s', 1.5)
        self.declare_parameter('joint_commands_topic', '/joint_commands')
        self.declare_parameter('joint_states_topic', '/joint_states')

        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.velocity_scale = float(self.get_parameter('velocity_scale').value)
        self.tolerance_rad = float(self.get_parameter('tolerance_rad').value)
        self.completion_timeout_s = float(self.get_parameter('completion_timeout_s').value)
        self._cmd_topic = str(self.get_parameter('joint_commands_topic').value)
        self._state_topic = str(self.get_parameter('joint_states_topic').value)

        cb_group = ReentrantCallbackGroup()

        self.cmd_pub = self.create_publisher(JointState, self._cmd_topic, 10)

        self._state_lock = threading.Lock()
        self._latest_state = None
        self.create_subscription(
            JointState, self._state_topic,
            self._on_state, 10, callback_group=cb_group,
        )

        # One in-flight goal per group (arm, gripper). The Event is set
        # by an incoming goal to preempt the current loop; the lock then
        # serializes execute() callbacks so the new goal waits for the old
        # one to drain before taking over.
        self._arm_lock = threading.Lock()
        self._gripper_lock = threading.Lock()
        self._arm_preempt = threading.Event()
        self._gripper_preempt = threading.Event()

        self.arm_server = ActionServer(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            execute_callback=self._execute_arm,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=cb_group,
        )
        self.gripper_server = ActionServer(
            self, FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory',
            execute_callback=self._execute_gripper,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=cb_group,
        )

        self.get_logger().info(
            f"trajectory_shim ready | rate={self.rate_hz:.1f} Hz | "
            f"velocity_scale={self.velocity_scale:.3f} "
            f"({1.0 / max(self.velocity_scale, 1e-3):.2f}x slower than MoveIt) | "
            f"tolerance={math.degrees(self.tolerance_rad):.2f}° | "
            f"completion_timeout={self.completion_timeout_s:.2f}s"
        )

    def _on_state(self, msg: JointState):
        with self._state_lock:
            self._latest_state = dict(zip(msg.name, msg.position))

    def _execute_arm(self, goal_handle):
        return self._execute(goal_handle, self._arm_lock, self._arm_preempt, label='arm')

    def _execute_gripper(self, goal_handle):
        return self._execute(goal_handle, self._gripper_lock, self._gripper_preempt, label='gripper')

    def _execute(self, goal_handle, group_lock, preempt, *, label):
        traj = goal_handle.request.trajectory
        if not traj.joint_names or not traj.points:
            self.get_logger().warn(f"[{label}] empty trajectory; aborting")
            goal_handle.abort()
            r = FollowJointTrajectory.Result()
            r.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            return r

        # Signal any in-flight goal to bow out, then wait for the lock.
        # The previous goal's _play loop checks `preempt` each tick and
        # releases the lock via its `finally`, so this acquire usually
        # succeeds within ~50 ms.
        preempt.set()
        if not group_lock.acquire(timeout=2.0):
            self.get_logger().warn(
                f"[{label}] previous goal didn't release within 2s; aborting new"
            )
            goal_handle.abort()
            r = FollowJointTrajectory.Result()
            r.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            return r

        try:
            preempt.clear()
            return self._play(goal_handle, traj, preempt, label=label)
        finally:
            group_lock.release()

    def _play(self, goal_handle, traj, preempt, *, label):
        names_out = _remap_names(list(traj.joint_names))
        inv_scale = 1.0 / max(self.velocity_scale, 1e-3)

        times = [
            (pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9) * inv_scale
            for pt in traj.points
        ]
        positions = [list(pt.positions) for pt in traj.points]
        n_joints = len(names_out)

        if any(len(p) != n_joints for p in positions):
            self.get_logger().error(f"[{label}] waypoint joint count mismatch")
            goal_handle.abort()
            r = FollowJointTrajectory.Result()
            r.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            return r

        dt = 1.0 / self.rate_hz
        t_end = times[-1]
        self.get_logger().info(
            f"[{label}] {len(times)} waypoints, scaled duration {t_end:.2f}s "
            f"(MoveIt asked {t_end * self.velocity_scale:.2f}s); "
            f"joints={names_out}"
        )

        # ---- Playback ----
        t0 = self.get_clock().now().nanoseconds * 1e-9
        while True:
            if preempt.is_set():
                self.get_logger().info(f"[{label}] preempted by newer goal")
                # rclpy disallows EXECUTING → CANCELED unilaterally; that
                # transition is reserved for client-initiated cancel.
                # For server-initiated preemption we abort instead.
                goal_handle.abort()
                r = FollowJointTrajectory.Result()
                r.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                return r
            if goal_handle.is_cancel_requested:
                self.get_logger().info(f"[{label}] cancelled by client")
                goal_handle.canceled()
                r = FollowJointTrajectory.Result()
                r.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                return r
            t = self.get_clock().now().nanoseconds * 1e-9 - t0
            self._publish(names_out, _interpolate(times, positions, t))
            if t >= t_end:
                break
            time.sleep(dt)

        # ---- Completion check (position within tolerance) ----
        goal_pos = positions[-1]
        deadline = self.get_clock().now().nanoseconds * 1e-9 + self.completion_timeout_s
        max_err = float('inf')
        while self.get_clock().now().nanoseconds * 1e-9 < deadline:
            if preempt.is_set():
                self.get_logger().info(f"[{label}] preempted during completion-wait")
                goal_handle.abort()
                r = FollowJointTrajectory.Result()
                r.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                return r
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                r = FollowJointTrajectory.Result()
                r.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                return r
            with self._state_lock:
                state = self._latest_state
            if state is not None:
                errs = [abs(state[n] - goal_pos[k])
                        for k, n in enumerate(names_out) if n in state]
                if errs:
                    max_err = max(errs)
                    if max_err <= self.tolerance_rad:
                        self.get_logger().info(
                            f"[{label}] reached goal "
                            f"(max_err={math.degrees(max_err):.2f}°)"
                        )
                        goal_handle.succeed()
                        r = FollowJointTrajectory.Result()
                        r.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                        return r
            # Hold the final setpoint while waiting for the servos to catch up.
            self._publish(names_out, goal_pos)
            time.sleep(dt)

        self.get_logger().warn(
            f"[{label}] tolerance violated: max_err={math.degrees(max_err):.2f}° "
            f"> {math.degrees(self.tolerance_rad):.2f}° after "
            f"{self.completion_timeout_s:.2f}s — aborting"
        )
        goal_handle.abort()
        r = FollowJointTrajectory.Result()
        r.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
        return r

    def _publish(self, names, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = list(positions)
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = TrajectoryShim()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
