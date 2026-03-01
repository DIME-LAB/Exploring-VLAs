#!/usr/bin/env python3
"""
SO-ARM101 Control GUI — Tkinter GUI with MoveIt IK, trajectory execution,
object detection, and dual hardware/simulation modes.

Source: adapted from RoboSort/JETANK_description/jetank_control_gui.py (3133 lines)
        IK delegated to MoveIt compute_ik service (KDL solver) instead of custom scipy.
        Joint mapping from MuammerBay/SO-ARM_ROS2_URDF and SO-ARM101_MoveIt_IsaacSim.
"""

import math
import os
import random
import signal
import threading
import time
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer as TfBuffer, TransformListener
try:
    from moveit_msgs.srv import GetPositionIK, GetPositionFK, GetMotionPlan, GetStateValidity
    from moveit_msgs.msg import (
        PositionIKRequest, RobotState, Constraints, JointConstraint,
        MotionPlanRequest, PlanningScene as PlanningSceneMsg, CollisionObject,
        AllowedCollisionEntry,
    )
    from shape_msgs.msg import SolidPrimitive
    from moveit_msgs.srv import (
        ExecuteKnownTrajectory, ApplyPlanningScene, GetPlanningScene as GetPlanningSceneSrv,
    )
    MOVEIT_AVAILABLE = True
except ImportError:
    try:
        from moveit_msgs.srv import GetPositionIK, GetPositionFK, GetMotionPlan, GetStateValidity
        from moveit_msgs.msg import (
            PositionIKRequest, RobotState, Constraints, JointConstraint,
            MotionPlanRequest, PlanningScene as PlanningSceneMsg, CollisionObject,
            AllowedCollisionEntry,
        )
        from shape_msgs.msg import SolidPrimitive
        from moveit_msgs.srv import (
            ApplyPlanningScene, GetPlanningScene as GetPlanningSceneSrv,
        )
        MOVEIT_AVAILABLE = True
        ExecuteKnownTrajectory = None
    except ImportError:
        MOVEIT_AVAILABLE = False
        GetPositionIK = None
        GetPositionFK = None
        GetStateValidity = None
        GetMotionPlan = None


# ---------------------------------------------------------------------------
# Debug service auto-registration
# ---------------------------------------------------------------------------
_SERVICE_REGISTRY = {}  # func.__name__ -> service_suffix


def service_trigger(service_name):
    """Decorator: marks a method to be auto-registered as ~/service_name (Trigger)."""
    def decorator(func):
        _SERVICE_REGISTRY[func.__name__] = service_name
        return func
    return decorator


# ---------------------------------------------------------------------------
# Joint configuration for SO-ARM101
# ---------------------------------------------------------------------------
ARM_JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
GRIPPER_JOINT_NAME = 'gripper_joint'
ALL_JOINT_NAMES = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]

JOINT_LIMITS = {
    'shoulder_pan':    (-1.91986, 1.91986),
    'shoulder_lift':       (-1.74533, 1.74533),
    'elbow_flex':       (-1.69, 1.69),
    'wrist_flex': (-1.65806, 1.65806),
    'wrist_roll':  (-2.74385, 2.84121),
    'gripper_joint':   (-0.174533, 1.74533),
}

# Jaw geometry for single-moving-jaw gripper offset compensation.
# Derived from STL mesh analysis of moving_jaw_so101_v1.stl + FK chain.
# Linear fit: jaw_gap(m) = BASELINE_JAW_GAP + JAW_GAP_RATE * gripper_joint_angle(rad)
# At angle=0 the jaws are NOT touching — there is a 16.9mm baseline gap.
BASELINE_JAW_GAP = 0.0190           # jaw gap at gripper_joint=0 (m)
JAW_GAP_RATE = 0.0749               # gap increase per radian (m/rad)
JAW_OPEN_CLEARANCE_M = 0.005        # extra jaw gap on open beyond symmetric baseline (m)
JAW_CLOSE_CLEARANCE_M = 0.0         # extra jaw gap on close beyond symmetric baseline (m)
TCP_CLEARANCE_M = 0.001             # extra IK offset beyond grip_width/2 for jaw overhang (m)

# Wrist roll offset from geometric IK: θ₅ = pan + grasp_yaw - WRIST_ROLL_OFFSET
_WRIST_ROLL_OFFSET = math.pi / 2 - 0.0486795  # ≈ 1.5221 rad (87.2°)


def _normalize_grasp_yaw(yaw, pan):
    """Pick yaw or yaw±π that keeps wrist_roll within joint limits.

    Gripper jaws are symmetric about the grip axis, so yaw and yaw+π
    produce equivalent grasps. We pick whichever keeps wrist_roll
    closest to the center of its range.
    """
    wr_lo, wr_hi = JOINT_LIMITS['wrist_roll']
    wr_center = (wr_lo + wr_hi) / 2
    best, best_dist = yaw, abs(pan + yaw - _WRIST_ROLL_OFFSET - wr_center)
    for candidate in (yaw + math.pi, yaw - math.pi):
        dist = abs(pan + candidate - _WRIST_ROLL_OFFSET - wr_center)
        if dist < best_dist:
            best, best_dist = candidate, dist
    return best

# ---------------------------------------------------------------------------
# Workspace bounds — loaded from compute_workspace.py output
# ---------------------------------------------------------------------------

def _load_workspace_yaml(section_name):
    """Load a named section from workspace_bounds.yaml."""
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'workspace_bounds.yaml')
    bounds = {}
    in_section = False
    try:
        with open(yaml_path, 'r') as f:
            for line in f:
                stripped = line.strip()
                if stripped == f'{section_name}:':
                    in_section = True
                    continue
                if in_section and stripped and not line[0].isspace():
                    break
                if in_section and ':' in stripped and not stripped.startswith('#'):
                    key, val = stripped.split(':', 1)
                    try:
                        bounds[key.strip()] = float(val.strip())
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return bounds

_WS = _load_workspace_yaml('workspace_bounds')
WORKSPACE_BOUNDS = {
    'X': (_WS.get('x_min', -0.35), _WS.get('x_max', 0.35)),
    'Y': (_WS.get('y_min', -0.35), _WS.get('y_max', 0.25)),
    'Z': (_WS.get('z_min', -0.10), _WS.get('z_max', 0.45)),
}

_GWS = _load_workspace_yaml('grasp_workspace_bounds')
GRASP_WORKSPACE_BOUNDS = {
    'R_MIN': _GWS.get('r_min', 0.09),
    'R_MAX': _GWS.get('r_max', 0.31),
    'Z_MIN': _GWS.get('z_min', -0.20),
    'Z_MAX': _GWS.get('z_max', 0.07),
}


def check_grasp_reachable(x, y, z, ground_z=None):
    """Check if (x, y, z) is within the top-down grasp workspace.

    Bounds are computed by sweeping geometric_ik() over a (r, z, yaw) grid,
    so they represent the true IK-solvable region, not just FK-reachable.
    ground_z: if provided, reject targets at or below the ground plane.
    Returns (ok, reason_string). ok=True means reachable.
    """
    if ground_z is not None and z <= ground_z:
        return False, f'at/below ground: z={z:.3f}m <= ground={ground_z:.3f}m'
    r = math.sqrt(x * x + y * y)
    r_min = GRASP_WORKSPACE_BOUNDS['R_MIN']
    r_max = GRASP_WORKSPACE_BOUNDS['R_MAX']
    if r < r_min:
        return False, f'too close: r={r:.3f}m < {r_min:.3f}m'
    if r > r_max:
        return False, f'too far: r={r:.3f}m > {r_max:.3f}m'
    z_min = GRASP_WORKSPACE_BOUNDS['Z_MIN']
    z_max = GRASP_WORKSPACE_BOUNDS['Z_MAX']
    if z < z_min:
        return False, f'too low: z={z:.3f}m < {z_min:.3f}m'
    if z > z_max:
        return False, f'too high: z={z:.3f}m > {z_max:.3f}m'
    return True, ''


class SOArm101ControlGUI(Node):
    """ROS2 node with embedded Tkinter GUI for SO-ARM101 control."""

    def __init__(self):
        super().__init__('so_arm101_control_gui')

        # Callback group for service clients — allows responses to be
        # processed concurrently with other callbacks (requires MultiThreadedExecutor)
        self._service_cb_group = ReentrantCallbackGroup()

        # Hardware mode
        self.use_real_hardware = False
        self.hw_lock = threading.Lock()

        # Current joint positions (radians) — updated from joint_state_broadcaster
        self.joint_positions = {name: 0.0 for name in ALL_JOINT_NAMES}
        self.joint_lock = threading.Lock()
        # Actual robot state — always updated from /joint_states, never blocked
        self._actual_positions = {name: 0.0 for name in ALL_JOINT_NAMES}

        # Track last sent arm positions
        self._last_sent_arm = [0.0] * len(ARM_JOINT_NAMES)
        self._last_sent_jaw = 0.0

        # --- Action clients (proven reliable for JTC) ---
        self.arm_action_client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.gripper_action_client = ActionClient(
            self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')

        # Track active goals so we can cancel before sending new ones
        self._arm_goal_handle = None
        self._gripper_goal_handle = None
        self._arm_goal_lock = threading.Lock()
        self._gripper_goal_lock = threading.Lock()

        # --- Gripper topic publisher (works fine via topic) ---
        self.gripper_traj_pub = self.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)
        # Hardware commands (for real servo driver)
        self.joint_cmd_pub = self.create_publisher(JointState, 'joint_commands_hw', 10)

        # --- Subscribers ---
        self.js_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_states_callback, 10)
        self.real_js_sub = self.create_subscription(
            JointState, 'real_joint_states', self._real_js_callback, 10)
        self.ext_cmd_sub = self.create_subscription(
            JointState, 'joint_commands', self._ext_cmd_callback, 10)
        self.objects_data = {}
        self.objects_lock = threading.Lock()
        self.objects_sub = None  # Created by _build_grasp_tab → _update_grasp_topic
        self.objects_bbox = {}   # {name: {sx, sy, sz}} from bbox topic
        _default_bbox = '/objects_bbox_real' if self.use_real_hardware else '/objects_bbox_sim'
        self.bbox_sub = self.create_subscription(
            String, _default_bbox, self._bbox_callback, 1)
        self.ee_pose_sub = self.create_subscription(
            PoseStamped, '/ee_pose', self._ee_pose_callback, 10)

        # MoveIt service clients + publishers
        self.ik_client = None
        self.plan_client = None
        if MOVEIT_AVAILABLE:
            from moveit_msgs.msg import DisplayTrajectory
            self.ik_client = self.create_client(
                GetPositionIK, '/compute_ik',
                callback_group=self._service_cb_group)
            self.fk_client = self.create_client(
                GetPositionFK, '/compute_fk',
                callback_group=self._service_cb_group)
            self.plan_client = self.create_client(
                GetMotionPlan, '/plan_kinematic_path',
                callback_group=self._service_cb_group)
            self.validity_client = self.create_client(
                GetStateValidity, '/check_state_validity',
                callback_group=self._service_cb_group)
            self._display_traj_pub = self.create_publisher(
                DisplayTrajectory, '/display_planned_path', 10)
            # Publish goal state to RViz MotionPlanning plugin
            self._goal_state_pub = self.create_publisher(
                RobotState, '/rviz/moveit/update_custom_goal_state', 10)
            # Switch active planning group in RViz
            self._planning_group_pub = self.create_publisher(
                String, '/rviz/moveit/select_planning_group', 10)
            self._active_planning_group = None  # Force first publish
            # Planning scene services for collision objects (ground plane, etc.)
            self._apply_scene_client = self.create_client(
                ApplyPlanningScene, '/apply_planning_scene',
                callback_group=self._service_cb_group)
            self._get_scene_client = self.create_client(
                GetPlanningSceneSrv, '/get_planning_scene',
                callback_group=self._service_cb_group)

        # Trajectory lock
        self._traj_lock = threading.Lock()

        # Track whether we should update sliders from joint_states
        self._slider_driven = False

        # Parameter for set_ik_target service
        self.declare_parameter('ik_target', '')
        # Parameters for jaw tuning via service calls
        self.declare_parameter('jaw_open_clearance_mm', JAW_OPEN_CLEARANCE_M * 1000)
        self.declare_parameter('jaw_close_clearance_mm', JAW_CLOSE_CLEARANCE_M * 1000)
        self.declare_parameter('tcp_clearance_mm', TCP_CLEARANCE_M * 1000)

        # TF buffer for TCP pose lookups
        self._tf_buffer = TfBuffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # GUI
        self.running = True
        self._setup_gui_thread()

        # --- Debug services (auto-registered via @service_trigger) ---
        self._debug_services = []
        for method_name, srv_suffix in _SERVICE_REGISTRY.items():
            cb = self._make_trigger_callback(method_name)
            srv = self.create_service(Trigger, f'~/{srv_suffix}', cb)
            self._debug_services.append(srv)
            self.get_logger().info(f'  service: ~/{srv_suffix}')

        # Manual services (read/write UI fields directly)
        for name, cb in [
            ('get_joint_positions', self._srv_get_joint_positions),
            ('get_ik_target', self._srv_get_ik_target),
            ('set_ik_target', self._srv_set_ik_target),
            ('get_ee_pose', self._srv_get_ee_pose),
            ('get_tcp_pose', self._srv_get_tcp_pose),
            ('get_log', self._srv_get_log),
        ]:
            self._debug_services.append(
                self.create_service(Trigger, f'~/{name}', cb))
            self.get_logger().info(f'  service: ~/{name}')

        self.get_logger().info('SO-ARM101 Control GUI initialized')

    # ------------------------------------------------------------------
    # Async service helper (thread-safe future wait)
    # ------------------------------------------------------------------

    @staticmethod
    def _wait_future(future, timeout_sec=2.0):
        """Wait for a ROS2 future to complete by polling. Thread-safe.

        Unlike rclpy.spin_until_future_complete, this does NOT create a
        temporary executor and does NOT spin the node.  The main-thread
        executor (rclpy.spin) processes callbacks; we just poll.
        """
        end = time.monotonic() + timeout_sec
        while not future.done() and time.monotonic() < end:
            time.sleep(0.01)          # 10 ms poll
        return future.result()        # None if timed-out / not done

    # ------------------------------------------------------------------
    # Debug service helpers
    # ------------------------------------------------------------------

    def _make_trigger_callback(self, method_name):
        """Factory: returns a Trigger callback that dispatches to tkinter thread."""
        def _callback(request, response):
            done_event = threading.Event()
            result = {'ok': True, 'msg': ''}

            def _run():
                try:
                    getattr(self, method_name)()
                    result['msg'] = f'{method_name} executed'
                except Exception as e:
                    result['ok'] = False
                    result['msg'] = str(e)
                finally:
                    done_event.set()

            if getattr(self, '_gui_ready', False):
                self.root.after(0, _run)
            else:
                result['ok'] = False
                result['msg'] = 'GUI not available'
                done_event.set()

            done_event.wait(timeout=2.0)
            response.success = result['ok']
            response.message = result['msg']
            return response
        return _callback

    def _srv_get_joint_positions(self, request, response):
        """Return current joint positions as name=value pairs."""
        with self.joint_lock:
            pairs = [f'{n}={self.joint_positions[n]:.6f}'
                     for n in ALL_JOINT_NAMES]
        response.success = True
        response.message = ', '.join(pairs)
        return response

    def _srv_get_ik_target(self, request, response):
        """Read current IK target fields (XYZ + quaternion)."""
        done_event = threading.Event()
        result = {'msg': ''}

        def _read():
            try:
                parts = []
                for axis in ['X', 'Y', 'Z']:
                    parts.append(f'{axis}={self.xyz_vars[axis].get():.6f}')
                for comp in ['Roll', 'shoulder_lift', 'Yaw']:
                    parts.append(f'{comp}={self.rpy_vars[comp].get():.1f}')
                qx, qy, qz, qw = self._rpy_deg_to_quat(
                    self.rpy_vars['Roll'].get(),
                    self.rpy_vars['shoulder_lift'].get(),
                    self.rpy_vars['Yaw'].get())
                parts.extend([f'qx={qx:.6f}', f'qy={qy:.6f}',
                              f'qz={qz:.6f}', f'qw={qw:.6f}'])
                result['msg'] = ', '.join(parts)
            except Exception as e:
                result['msg'] = f'error: {e}'
            finally:
                done_event.set()

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _read)
        else:
            result['msg'] = 'GUI not available'
            done_event.set()

        done_event.wait(timeout=2.0)
        response.success = 'error' not in result['msg']
        response.message = result['msg']
        return response

    def _srv_set_ik_target(self, request, response):
        """Write IK target fields. Pass key=value pairs in request.message.
        Example: ros2 service call ... '{message: "x=0.12, z=0.15, qw=1.0"}'
        Supported keys: x, y, z, qx, qy, qz, qw (case-insensitive)."""
        done_event = threading.Event()
        result = {'ok': True, 'msg': ''}

        # Parse key=value pairs from the trigger message field
        raw = getattr(request, 'message', '') if hasattr(request, 'message') else ''
        # Trigger doesn't have a message field on request — use a workaround:
        # We'll accept the values from the service call's yaml string that gets
        # stuffed into the Trigger request. But Trigger.Request has no fields.
        # So we need to use a different approach — pass via ROS param or topic.
        # Actually, let's use a simple convention: the caller sets a parameter first.

        # Better approach: read from a latched parameter
        # For now, parse from the node's parameter
        raw = self.get_parameter('ik_target').get_parameter_value().string_value

        if not raw:
            response.success = False
            response.message = (
                'Set param first: ros2 param set /so_arm101_control_gui '
                'ik_target "x=0.12,y=0.0,z=0.15,qw=1.0" '
                'then call this service')
            return response

        # Parse
        updates = {}
        for part in raw.replace(' ', '').split(','):
            if '=' in part:
                k, v = part.split('=', 1)
                updates[k.lower()] = float(v)

        def _write():
            try:
                axis_map = {'x': 'X', 'y': 'Y', 'z': 'Z'}
                rpy_map = {'roll': 'Roll', 'pitch': 'shoulder_lift', 'yaw': 'Yaw'}
                for k, v in updates.items():
                    if k in axis_map and axis_map[k] in self.xyz_vars:
                        self.xyz_vars[axis_map[k]].set(v)
                    elif k in rpy_map and rpy_map[k] in self.rpy_vars:
                        self.rpy_vars[rpy_map[k]].set(v)
                set_keys = ', '.join(f'{k}={v:.4f}' for k, v in updates.items())
                result['msg'] = f'Set: {set_keys}'
            except Exception as e:
                result['ok'] = False
                result['msg'] = str(e)
            finally:
                done_event.set()

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _write)
        else:
            result['ok'] = False
            result['msg'] = 'GUI not available'
            done_event.set()

        done_event.wait(timeout=2.0)
        response.success = result['ok']
        response.message = result['msg']
        return response

    def _srv_get_ee_pose(self, request, response):
        """Read current End-Effector pose values (gripper link)."""
        parts = []
        for key in ['X', 'Y', 'Z', 'qx', 'qy', 'qz', 'qw']:
            parts.append(f'{key}={self.ee_labels[key].get()}')
        response.success = True
        response.message = ', '.join(parts)
        return response

    def _srv_get_tcp_pose(self, request, response):
        """Look up tcp_link pose in base frame via TF2."""
        try:
            t = self._tf_buffer.lookup_transform(
                'base', 'tcp_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            p = t.transform.translation
            q = t.transform.rotation
            response.success = True
            response.message = (
                f'X={p.x:.6f}, Y={p.y:.6f}, Z={p.z:.6f}, '
                f'qx={q.x:.6f}, qy={q.y:.6f}, qz={q.z:.6f}, qw={q.w:.6f}')
        except Exception as e:
            response.success = False
            response.message = f'TF lookup failed: {e}'
        return response

    def _srv_get_log(self, request, response):
        """Read last 20 lines from Process Log."""
        done_event = threading.Event()
        result = {'msg': ''}

        def _read():
            try:
                content = self._process_log.get('1.0', 'end').strip()
                lines = content.split('\n')
                result['msg'] = '\n'.join(lines[-20:]) if lines else '(empty)'
            except Exception as e:
                result['msg'] = f'error: {e}'
            finally:
                done_event.set()

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _read)
        else:
            result['msg'] = 'GUI not available'
            done_event.set()

        done_event.wait(timeout=2.0)
        response.success = 'error' not in result['msg']
        response.message = result['msg']
        return response

    # ------------------------------------------------------------------
    # Controller command publishing (via action interface)
    # ------------------------------------------------------------------

    def _send_arm_goal(self, positions, duration_s=0.5):
        """Send arm joint positions via FollowJointTrajectory action."""
        if not self.arm_action_client.server_is_ready():
            self._append_log('arm_controller action server not ready', 'warn')
            return

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = list(ARM_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [positions.get(n, 0.0) for n in ARM_JOINT_NAMES]
        point.velocities = [0.0] * len(ARM_JOINT_NAMES)
        point.time_from_start = Duration(
            sec=int(duration_s),
            nanosec=int((duration_s % 1) * 1e9))
        traj.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        # Cancel previous goal if still active
        with self._arm_goal_lock:
            if self._arm_goal_handle is not None:
                try:
                    self._arm_goal_handle.cancel_goal_async()
                except Exception:
                    pass
                self._arm_goal_handle = None

        future = self.arm_action_client.send_goal_async(goal)
        future.add_done_callback(self._arm_goal_response)

    def _arm_goal_response(self, future):
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                with self._arm_goal_lock:
                    self._arm_goal_handle = goal_handle
        except Exception:
            pass

    def _send_gripper_goal(self, jaw_position, duration_s=0.5):
        """Send gripper position via FollowJointTrajectory action."""
        if not self.gripper_action_client.server_is_ready():
            self._append_log('gripper_controller action server not ready', 'warn')
            return

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ['gripper_joint']
        point = JointTrajectoryPoint()
        point.positions = [jaw_position]
        point.velocities = [0.0]
        point.time_from_start = Duration(
            sec=int(duration_s),
            nanosec=int((duration_s % 1) * 1e9))
        traj.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        with self._gripper_goal_lock:
            if self._gripper_goal_handle is not None:
                try:
                    self._gripper_goal_handle.cancel_goal_async()
                except Exception:
                    pass
                self._gripper_goal_handle = None

        future = self.gripper_action_client.send_goal_async(goal)
        future.add_done_callback(self._gripper_goal_response)

    def _gripper_goal_response(self, future):
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                with self._gripper_goal_lock:
                    self._gripper_goal_handle = goal_handle
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Joint state feedback
    # ------------------------------------------------------------------

    def _joint_states_callback(self, msg):
        """Update internal state and GUI sliders from joint_state_broadcaster."""
        # Always track actual robot state (never blocked by _slider_driven)
        with self.joint_lock:
            for i, name in enumerate(msg.name):
                if name in self._actual_positions and i < len(msg.position):
                    self._actual_positions[name] = msg.position[i]
        if self._slider_driven:
            return
        with self.joint_lock:
            for i, name in enumerate(msg.name):
                if name in self.joint_positions and i < len(msg.position):
                    self.joint_positions[name] = msg.position[i]
            positions = dict(self.joint_positions)
        # Update sliders and goal state to reflect actual robot state
        if getattr(self, '_gui_ready', False):
            self.root.after(0, self._sync_all_sliders, positions)
        self._publish_goal_state()

    # ------------------------------------------------------------------
    # GUI setup
    # ------------------------------------------------------------------

    def _setup_gui_thread(self):
        self.gui_thread = threading.Thread(target=self._run_gui, daemon=True)
        self.gui_thread.start()

    def _run_gui(self):
        self.root = tk.Tk()
        self.root.title('SO-ARM101 Control')
        self.root.geometry('580x780')
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        # Status bar
        self.status_var = tk.StringVar(value='Mode: Simulation')
        status_frame = tk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(status_frame, textvariable=self.status_var, anchor='w',
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT)

        self.hw_var = tk.StringVar(value='sim')
        tk.Radiobutton(status_frame, text='Real Hardware', variable=self.hw_var,
                       value='real', command=self._toggle_hardware).pack(side=tk.RIGHT, padx=2)
        tk.Radiobutton(status_frame, text='Simulation', variable=self.hw_var,
                       value='sim', command=self._toggle_hardware).pack(side=tk.RIGHT, padx=2)

        # Ground plane collision toggle (common to all tabs)
        scene_frame = tk.Frame(self.root)
        scene_frame.pack(fill=tk.X, padx=5, pady=1)
        self._ground_plane_var = tk.BooleanVar(value=True)
        tk.Checkbutton(scene_frame, text='Ground Plane',
                       variable=self._ground_plane_var,
                       command=self._toggle_ground_plane).pack(side=tk.LEFT)
        tk.Label(scene_frame, text='  Z:').pack(side=tk.LEFT)
        self._ground_z_var = tk.DoubleVar(value=0.0)
        tk.Spinbox(scene_frame, from_=-0.5, to=0.5, increment=0.01,
                   textvariable=self._ground_z_var, width=6).pack(side=tk.LEFT)
        # Publish ground plane on startup after MoveIt is ready
        self.root.after(3000, self._toggle_ground_plane)

        # Notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))
        self._notebook = notebook

        self._build_individual_tab(notebook)
        self._build_arm_control_tab(notebook)
        self._build_grasp_tab(notebook)

        # Auto-populate IK fields when switching to IK tab
        notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

        # --- Log Panel (bottom) ---
        self._build_log_panel()

        self._gui_ready = True
        self.root.mainloop()
        self._gui_ready = False

    # ------------------------------------------------------------------
    # Tab 1: Individual Joint Control
    # ------------------------------------------------------------------

    def _build_individual_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='FK')

        self.sliders = {}
        self.slider_labels = {}

        # --- Arm Section ---
        arm_frame = ttk.LabelFrame(frame, text='Arm')
        arm_frame.pack(fill=tk.X, padx=10, pady=(5, 2))

        for name in ARM_JOINT_NAMES:
            lo, hi = JOINT_LIMITS[name]
            row = tk.Frame(arm_frame)
            row.pack(fill=tk.X, padx=5, pady=2)

            tk.Label(row, text=name, width=14, anchor='w').pack(side=tk.LEFT, padx=(5, 0))

            var = tk.DoubleVar(value=0.0)
            slider = tk.Scale(
                row, variable=var, from_=lo, to=hi,
                orient=tk.HORIZONTAL, resolution=0.001, length=300,
                command=lambda val, n=name: self._on_slider(n, float(val)))
            slider.pack(side=tk.LEFT, padx=5)

            lbl = tk.Label(row, text='0.000', width=8)
            lbl.pack(side=tk.LEFT)

            self.sliders[name] = var
            self.slider_labels[name] = lbl

        arm_btn_frame = tk.Frame(arm_frame)
        arm_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(arm_btn_frame, text='Reset Arm', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._zero_arm).pack(side=tk.LEFT, padx=5)
        tk.Button(arm_btn_frame, text='Randomize', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._randomize_arm).pack(side=tk.LEFT, padx=5)

        # --- Gripper Section ---
        gripper_frame = ttk.LabelFrame(frame, text='Gripper')
        gripper_frame.pack(fill=tk.X, padx=10, pady=(2, 2))

        jaw_row = tk.Frame(gripper_frame)
        jaw_row.pack(fill=tk.X, padx=5, pady=2)

        lo, hi = JOINT_LIMITS[GRIPPER_JOINT_NAME]
        tk.Label(jaw_row, text=GRIPPER_JOINT_NAME, width=14, anchor='w').pack(side=tk.LEFT, padx=(5, 0))

        jaw_var = tk.DoubleVar(value=0.0)
        jaw_slider = tk.Scale(
            jaw_row, variable=jaw_var, from_=lo, to=hi,
            orient=tk.HORIZONTAL, resolution=0.001, length=300,
            command=lambda val: self._on_slider(GRIPPER_JOINT_NAME, float(val)))
        jaw_slider.pack(side=tk.LEFT, padx=5)

        jaw_lbl = tk.Label(jaw_row, text='0.000', width=8)
        jaw_lbl.pack(side=tk.LEFT)

        self.sliders[GRIPPER_JOINT_NAME] = jaw_var
        self.slider_labels[GRIPPER_JOINT_NAME] = jaw_lbl

        gripper_btn_frame = tk.Frame(gripper_frame)
        gripper_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(gripper_btn_frame, text='Reset Gripper', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._zero_gripper).pack(side=tk.LEFT, padx=5)
        tk.Button(gripper_btn_frame, text='Open', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._gripper_open).pack(side=tk.LEFT, padx=5)
        tk.Button(gripper_btn_frame, text='Close', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._gripper_close).pack(side=tk.LEFT, padx=5)

        # --- Action Buttons (always visible) ---
        action_frame = tk.Frame(frame)
        action_frame.pack(fill=tk.X, padx=10, pady=(8, 5))
        self.set_joints_btn = tk.Button(
            action_frame, text='Set Joints', bg='#b0b0b0', fg='#1a1a1a', command=self._set_joints)
        self.set_joints_btn.pack(side=tk.LEFT, padx=5)
        self.execute_btn = tk.Button(
            action_frame, text='Plan & Execute', bg='#b0b0b0', fg='#1a1a1a', command=self._moveit_execute)
        self.execute_btn.pack(side=tk.LEFT, padx=5)
        tk.Label(action_frame, text='Speed:', font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 2))
        self.velocity_scale_var = tk.DoubleVar(value=0.5)
        self._last_speed_val = 0.5
        vcmd = (self.root.register(self._validate_speed), '%P')
        self.velocity_scale_spin = tk.Spinbox(
            action_frame, from_=0.1, to=1.0, increment=0.1,
            textvariable=self.velocity_scale_var, width=4,
            font=('Arial', 9), validate='all', validatecommand=vcmd)
        self.velocity_scale_spin.pack(side=tk.LEFT)

    def _validate_speed(self, value_str):
        """Validate speed spinbox: allow empty (during editing) and 0.1-1.0."""
        if value_str == '':
            return True
        try:
            v = float(value_str)
            if v > 1.0:
                self._append_log('Speed must be between 0.1 and 1.0 (100% of joint limits)', 'warn')
                return False
            if v < 0.1:
                self._append_log('Speed must be between 0.1 and 1.0', 'warn')
                return False
            if v != self._last_speed_val:
                self._last_speed_val = v
                pct = int(v * 100)
                self._append_log(f'Speed set to {v:.1f} ({pct}% of joint limits)')
            return True
        except ValueError:
            return False

    def _on_slider(self, joint_name, value):
        self._slider_driven = True
        with self.joint_lock:
            self.joint_positions[joint_name] = value
        if joint_name in self.slider_labels:
            self.slider_labels[joint_name].config(text=f'{value:.3f}')
        # Sync IK tab jaw label (sliders share the same DoubleVar)
        if joint_name == GRIPPER_JOINT_NAME and hasattr(self, '_ik_jaw_label'):
            self._ik_jaw_label.config(text=f'{value:.3f}')
        # Switch planning group based on which joint is being moved
        if joint_name == GRIPPER_JOINT_NAME:
            self._select_planning_group('gripper')
        else:
            self._select_planning_group('arm')
        # Update RViz goal state only — robot doesn't move until button click
        self._publish_goal_state()

    @service_trigger('zero_arm')
    def _zero_arm(self):
        """Reset arm joints to zero."""
        self._slider_driven = True
        self._select_planning_group('arm')
        with self.joint_lock:
            for name in ARM_JOINT_NAMES:
                self.joint_positions[name] = 0.0
        for name in ARM_JOINT_NAMES:
            if name in self.sliders:
                self.sliders[name].set(0.0)
                self.slider_labels[name].config(text='0.000')
        self._publish_goal_state()
        self.status_var.set('Arm zeroed')

    def _zero_gripper(self):
        """Reset gripper to zero (closed)."""
        self._gripper_command(0.0)
        self.status_var.set('Gripper zeroed')

    def _select_planning_group(self, group_name):
        """Switch the active planning group in RViz.

        On first call, cycles through gripper→arm to clear startup markers.
        """
        if not hasattr(self, '_planning_group_pub'):
            return
        from std_msgs.msg import String
        # First call: cycle gripper→arm to clear green startup markers
        if not hasattr(self, '_planning_group_initialized'):
            self._planning_group_initialized = True
            msg = String()
            msg.data = 'gripper'
            self._planning_group_pub.publish(msg)
        if hasattr(self, '_active_planning_group') and self._active_planning_group == group_name:
            return
        self._active_planning_group = group_name
        msg = String()
        msg.data = group_name
        self._planning_group_pub.publish(msg)
        # Republish goal state after RViz processes the group switch
        if getattr(self, '_gui_ready', False):
            self.root.after(150, self._publish_goal_state)

    @service_trigger('randomize_arm')
    def _randomize_arm(self):
        """Set arm joints to a random collision-free configuration."""
        self._select_planning_group('arm')
        self.status_var.set('Finding random valid state...')

        def _find_valid():
            max_attempts = 100
            for attempt in range(max_attempts):
                # Generate random joint values within limits
                positions = {}
                for name in ARM_JOINT_NAMES:
                    lo, hi = JOINT_LIMITS[name]
                    positions[name] = random.uniform(lo, hi)

                # Check validity via MoveIt
                if MOVEIT_AVAILABLE and hasattr(self, 'validity_client') \
                        and self.validity_client.service_is_ready():
                    req = GetStateValidity.Request()
                    req.robot_state.joint_state.name = list(ARM_JOINT_NAMES)
                    req.robot_state.joint_state.position = [
                        positions[n] for n in ARM_JOINT_NAMES]
                    req.group_name = 'arm'
                    future = self.validity_client.call_async(req)
                    self._wait_future(future, timeout_sec=1.0)
                    if future.result() is not None and not future.result().valid:
                        continue  # collision — retry
                # Valid (or no validity service available) — apply it
                if getattr(self, '_gui_ready', False):
                    self.root.after(0, self._apply_random_arm, positions)
                return

            # Exhausted attempts — apply last one anyway
            self._append_log(f'No collision-free state after {max_attempts} attempts', 'warn')
            if getattr(self, '_gui_ready', False):
                self.root.after(0, self._apply_random_arm, positions)

        threading.Thread(target=_find_valid, daemon=True).start()

    def _apply_random_arm(self, positions):
        """Apply validated random arm positions to sliders and goal state."""
        self._slider_driven = True
        with self.joint_lock:
            self.joint_positions.update(positions)
        for name in ARM_JOINT_NAMES:
            val = positions[name]
            if name in self.sliders:
                self.sliders[name].set(val)
                self.slider_labels[name].config(text=f'{val:.3f}')
        self._publish_goal_state()
        self.status_var.set('Arm randomized (valid)')

    # ------------------------------------------------------------------
    # Planning Mode
    # ------------------------------------------------------------------


    def _publish_goal_state(self):
        """Publish current slider positions (arm + gripper) as goal state to RViz."""
        if not MOVEIT_AVAILABLE or not hasattr(self, '_goal_state_pub'):
            return
        goal_state = RobotState()
        goal_state.is_diff = True  # Required by MotionPlanning plugin
        goal_state.joint_state.name = list(ALL_JOINT_NAMES)
        with self.joint_lock:
            goal_state.joint_state.position = [
                self.joint_positions[n] for n in ALL_JOINT_NAMES]
        self._goal_state_pub.publish(goal_state)

    @service_trigger('set_joints')
    def _set_joints(self):
        """Send current slider positions directly to arm + gripper controllers."""
        with self.joint_lock:
            positions = dict(self.joint_positions)
        self._send_arm_goal(positions, duration_s=0.5)
        jaw = positions.get(GRIPPER_JOINT_NAME, 0.0)
        self._send_gripper_goal(jaw, duration_s=0.5)
        if self.use_real_hardware:
            self._send_hw_command(positions)
        joints_str = ', '.join(f'{n}: {positions[n]:.3f}' for n in ARM_JOINT_NAMES)
        jaw_str = f'{GRIPPER_JOINT_NAME}: {jaw:.3f}'
        self.status_var.set('Joints set')
        self._append_log(f'Set Joints → {joints_str}, {jaw_str}')
        # Allow joint_states to sync sliders again after robot reaches goal
        self.root.after(1000, self._clear_slider_driven)

    def _clear_slider_driven(self):
        """Allow joint_states callback to sync sliders again."""
        self._slider_driven = False

    @service_trigger('plan_execute')
    def _moveit_execute(self):
        """Plan and execute: plan via MoveIt, then send trajectory to arm_controller."""
        if not MOVEIT_AVAILABLE or self.plan_client is None:
            self.status_var.set('MoveIt not available')
            return
        if not self.plan_client.service_is_ready():
            self.status_var.set('Planning service not ready...')
            self._append_log('/plan_kinematic_path service not ready', 'warn')
            return

        self.root.after(0, lambda: self.execute_btn.config(state=tk.DISABLED))
        self._set_status('Planning...')

        with self.joint_lock:
            target_positions = {n: self.joint_positions[n] for n in ARM_JOINT_NAMES}
            self._execute_jaw_target = self.joint_positions.get(GRIPPER_JOINT_NAME, 0.0)

        goal_str = ', '.join(f'{n}: {target_positions[n]:.3f}' for n in ARM_JOINT_NAMES)
        self._append_log(f'Plan & Execute → {goal_str}, {GRIPPER_JOINT_NAME}: {self._execute_jaw_target:.3f}')

        constraints = Constraints()
        for name in ARM_JOINT_NAMES:
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = target_positions[name]
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        request = GetMotionPlan.Request()
        mpr = MotionPlanRequest()
        mpr.group_name = 'arm'
        mpr.num_planning_attempts = 10
        mpr.allowed_planning_time = 10.0
        vel_scale = self.velocity_scale_var.get()
        mpr.max_velocity_scaling_factor = vel_scale
        mpr.max_acceleration_scaling_factor = vel_scale
        mpr.goal_constraints.append(constraints)
        request.motion_plan_request = mpr

        future = self.plan_client.call_async(request)
        future.add_done_callback(self._plan_and_execute_callback)

    def _plan_and_execute_callback(self, future):
        """Handle planning result — display trajectory in RViz, then execute."""
        try:
            resp = future.result()
        except Exception as e:
            self._set_status(f'Planning failed: {e}')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            return

        error_code = resp.motion_plan_response.error_code.val
        if error_code != 1:
            self._set_status(f'Planning failed (error {error_code})')
            self._append_log(f'Planning failed (error {error_code})', 'warn')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            return

        robot_trajectory = resp.motion_plan_response.trajectory
        pt = resp.motion_plan_response.planning_time
        n_pts = len(robot_trajectory.joint_trajectory.points)
        self._append_log(f'Plan found ({pt:.3f}s), {n_pts} points')

        # Display trajectory in RViz
        from moveit_msgs.msg import DisplayTrajectory
        display_msg = DisplayTrajectory()
        display_msg.trajectory.append(robot_trajectory)
        self._display_traj_pub.publish(display_msg)

        # Execute via arm_controller
        if not self.arm_action_client.server_is_ready():
            self._set_status('Arm controller not ready')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            return

        self._set_status(f'Executing ({n_pts} points)...')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = robot_trajectory.joint_trajectory
        send_future = self.arm_action_client.send_goal_async(goal)
        send_future.add_done_callback(self._execute_response)

        # Send gripper to its goal position too
        if hasattr(self, '_execute_jaw_target'):
            self._send_gripper_goal(self._execute_jaw_target, duration_s=1.0)

    def _execute_response(self, future):
        """Handle trajectory execution acceptance."""
        try:
            goal_handle = future.result()
        except Exception as e:
            self._set_status(f'Execution failed: {e}')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            return
        if not goal_handle.accepted:
            self._set_status('Execution rejected')
            self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))
            return
        self._set_status('Executing trajectory...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._execute_result)

    def _execute_result(self, future):
        """Handle trajectory execution result."""
        try:
            future.result()
            self._set_status('Execution complete')
            self._append_log('Trajectory execution complete')
        except Exception as e:
            self._set_status(f'Execution error: {e}')
            self._append_log(f'Execution error: {e}', 'error')
        self.root.after(0, lambda: self.execute_btn.config(state=tk.NORMAL))

    def _set_status(self, text):
        """Thread-safe status bar update."""
        if getattr(self, '_gui_ready', False):
            self.root.after(0, self.status_var.set, text)

    # ------------------------------------------------------------------
    # Log Panel
    # ------------------------------------------------------------------

    def _build_log_panel(self):
        """Build the bottom log panel with Process Log and System Errors tabs."""
        outer = tk.Frame(self.root)
        outer.pack(fill=tk.BOTH, padx=5, pady=(2, 5), expand=False)

        log_notebook = ttk.Notebook(outer)
        log_notebook.pack(fill=tk.BOTH, expand=True)
        self._log_notebook = log_notebook

        # Process Log tab — text + buttons inside
        proc_frame = tk.Frame(log_notebook, bg='#1e1e1e')
        log_notebook.add(proc_frame, text='Process Log')
        proc_btn = tk.Frame(proc_frame)
        proc_btn.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Button(proc_btn, text='Clear', width=6, bg='#b0b0b0', fg='#1a1a1a',
                  command=self._clear_active_log).pack(fill=tk.BOTH, expand=True, padx=3, pady=(3, 2))
        tk.Button(proc_btn, text='Copy', width=6, bg='#b0b0b0', fg='#1a1a1a',
                  command=self._copy_active_log).pack(fill=tk.BOTH, expand=True, padx=3, pady=(2, 3))
        proc_scroll = tk.Scrollbar(proc_frame)
        proc_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._process_log = tk.Text(proc_frame, height=6, wrap=tk.WORD,
                                     font=('Consolas', 9), state=tk.DISABLED,
                                     bg='#1e1e1e', fg='#d4d4d4', borderwidth=0)
        self._process_log.config(yscrollcommand=proc_scroll.set)
        proc_scroll.config(command=self._process_log.yview)
        self._process_log.tag_configure('info', foreground='#d4d4d4')
        self._process_log.tag_configure('warn', foreground='#ffa500')
        self._process_log.tag_configure('error', foreground='#ff4444')
        self._process_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # System Errors tab — text + buttons inside
        err_frame = tk.Frame(log_notebook, bg='#1e1e1e')
        log_notebook.add(err_frame, text='System Errors')
        err_btn = tk.Frame(err_frame)
        err_btn.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Button(err_btn, text='Clear', width=6, bg='#b0b0b0', fg='#1a1a1a',
                  command=self._clear_active_log).pack(fill=tk.BOTH, expand=True, padx=3, pady=(3, 2))
        tk.Button(err_btn, text='Copy', width=6, bg='#b0b0b0', fg='#1a1a1a',
                  command=self._copy_active_log).pack(fill=tk.BOTH, expand=True, padx=3, pady=(2, 3))
        err_scroll = tk.Scrollbar(err_frame)
        err_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._error_log = tk.Text(err_frame, height=6, wrap=tk.WORD,
                                   font=('Consolas', 9), state=tk.DISABLED,
                                   bg='#1e1e1e', fg='#ff6b6b', borderwidth=0)
        self._error_log.config(yscrollcommand=err_scroll.set)
        err_scroll.config(command=self._error_log.yview)
        self._error_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Capture stderr and uncaught exceptions into System Errors
        self._setup_stderr_capture()

    def _get_active_log_widget(self):
        """Return the Text widget for the currently selected log tab."""
        idx = self._log_notebook.index(self._log_notebook.select())
        return self._process_log if idx == 0 else self._error_log

    def _clear_active_log(self):
        widget = self._get_active_log_widget()
        widget.config(state=tk.NORMAL)
        widget.delete('1.0', tk.END)
        widget.config(state=tk.DISABLED)

    def _copy_active_log(self):
        widget = self._get_active_log_widget()
        content = widget.get('1.0', tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self._append_log('Log copied to clipboard')

    def _setup_stderr_capture(self):
        """Redirect stderr and sys.excepthook to System Errors log."""
        import sys

        gui = self  # prevent closure issues
        original_stderr = sys.stderr

        class StderrRedirector:
            def __init__(self):
                self._buffer = ''

            def write(self, text):
                original_stderr.write(text)  # still print to terminal
                if not text.strip():
                    return
                self._buffer += text
                if '\n' in self._buffer:
                    lines = self._buffer.split('\n')
                    for line in lines[:-1]:
                        if line.strip():
                            gui._append_system_error(line)
                    self._buffer = lines[-1]

            def flush(self):
                if self._buffer.strip():
                    gui._append_system_error(self._buffer)
                    self._buffer = ''
                original_stderr.flush()

        sys.stderr = StderrRedirector()

        original_excepthook = sys.excepthook

        def _excepthook(exc_type, exc_value, exc_tb):
            import traceback
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            gui._append_system_error(tb_str)
            original_excepthook(exc_type, exc_value, exc_tb)

        sys.excepthook = _excepthook

    def _append_system_error(self, text):
        """Thread-safe append to System Errors log only."""
        import datetime
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        line = f'[{timestamp}] {text}\n'

        def _do():
            if not hasattr(self, '_error_log'):
                return
            self._error_log.config(state=tk.NORMAL)
            self._error_log.insert(tk.END, line)
            self._error_log.see(tk.END)
            self._error_log.config(state=tk.DISABLED)
            # Auto-switch to System Errors tab
            self._log_notebook.select(1)

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _do)

    def _append_log(self, text, level='info'):
        """Thread-safe log append to Process Log. level: 'info', 'warn', 'error'."""
        import datetime
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        line = f'[{timestamp}] {text}\n'

        def _do_append():
            if not hasattr(self, '_process_log'):
                return
            self._process_log.config(state=tk.NORMAL)
            self._process_log.insert(tk.END, line, level)
            self._process_log.see(tk.END)
            self._process_log.config(state=tk.DISABLED)

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _do_append)

    # ------------------------------------------------------------------
    # Tab 2: Arm Control (IK via MoveIt)
    # ------------------------------------------------------------------

    def _build_arm_control_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='IK')
        self._ik_tab_frame = frame

        # --- Arm (XYZ + RPY) side by side ---
        coord_frame = ttk.LabelFrame(frame, text='Arm')
        coord_frame.pack(fill=tk.X, padx=10, pady=5)

        self.xyz_vars = {}
        self._ik_spinboxes = {}    # field -> Spinbox widget (for color changes)
        self._ik_last_valid = {}   # field -> last value that produced valid IK
        self._ik_solve_gen = 0     # generation counter for async IK results
        self._ik_solve_lock = threading.Lock()  # prevent concurrent rclpy.spin
        self._ik_debounce_id = None
        self._ik_trace_active = True  # guard to suppress traces during programmatic updates

        columns_row = tk.Frame(coord_frame)
        columns_row.pack(fill=tk.X, padx=5, pady=2)

        # Left column: XYZ
        xyz_col = tk.Frame(columns_row)
        xyz_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(xyz_col, text='Position', font=('Arial', 9, 'bold')).pack(anchor='w', padx=2)
        for label, default in [('X', 0.12), ('Y', 0.0), ('Z', 0.15)]:
            lo, hi = WORKSPACE_BOUNDS[label]
            row = tk.Frame(xyz_col)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f'{label}:', width=3).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=default)
            spin = tk.Spinbox(
                row, textvariable=var, from_=lo, to=hi,
                increment=0.01, width=8, format='%.3f')
            spin.pack(side=tk.LEFT, padx=3)
            var.trace_add('write', lambda *a, f=label: self._on_ik_var_changed(f))
            self.xyz_vars[label] = var
            self._ik_spinboxes[label] = spin
            self._ik_last_valid[label] = default

        # Right column: RPY
        rpy_col = tk.Frame(columns_row)
        rpy_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(rpy_col, text='Orientation', font=('Arial', 9, 'bold')).pack(anchor='w', padx=2)
        self.rpy_vars = {}
        for comp, default in [('Roll', 0.0), ('shoulder_lift', 0.0), ('Yaw', 0.0)]:
            row = tk.Frame(rpy_col)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f'{comp[0]}:', width=3).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=default)
            spin = tk.Spinbox(
                row, textvariable=var, from_=-180.0, to=180.0,
                increment=1.0, width=8, format='%.1f')
            spin.pack(side=tk.LEFT, padx=3)
            var.trace_add('write', lambda *a, f=comp: self._on_ik_var_changed(f))
            self.rpy_vars[comp] = var
            self._ik_spinboxes[comp] = spin
            self._ik_last_valid[comp] = default

        # Buttons inside Arm frame
        arm_btn_frame = tk.Frame(coord_frame)
        arm_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(arm_btn_frame, text='Reset Arm', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._ik_reset).pack(side=tk.LEFT, padx=5)
        tk.Button(arm_btn_frame, text='Randomize', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._ik_randomize).pack(side=tk.LEFT, padx=5)

        # --- Gripper Section (shares DoubleVar with FK tab) ---
        gripper_frame2 = ttk.LabelFrame(frame, text='Gripper')
        gripper_frame2.pack(fill=tk.X, padx=10, pady=(2, 2))

        jaw_row2 = tk.Frame(gripper_frame2)
        jaw_row2.pack(fill=tk.X, padx=5, pady=2)
        lo, hi = JOINT_LIMITS[GRIPPER_JOINT_NAME]
        tk.Label(jaw_row2, text=GRIPPER_JOINT_NAME, width=14, anchor='w').pack(side=tk.LEFT, padx=(5, 0))
        # Reuse the FK tab's DoubleVar so both sliders stay in sync automatically
        jaw_var_shared = self.sliders[GRIPPER_JOINT_NAME]
        self._ik_jaw_slider = tk.Scale(
            jaw_row2, variable=jaw_var_shared, from_=lo, to=hi,
            orient=tk.HORIZONTAL, resolution=0.001, length=300,
            command=lambda val: self._on_slider(GRIPPER_JOINT_NAME, float(val)))
        self._ik_jaw_slider.pack(side=tk.LEFT, padx=5)
        self._ik_jaw_label = tk.Label(jaw_row2, text='0.000', width=8)
        self._ik_jaw_label.pack(side=tk.LEFT)

        gripper_btn_frame2 = tk.Frame(gripper_frame2)
        gripper_btn_frame2.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(gripper_btn_frame2, text='Reset Gripper', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._zero_gripper).pack(side=tk.LEFT, padx=5)
        tk.Button(gripper_btn_frame2, text='Open', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._gripper_open).pack(side=tk.LEFT, padx=5)
        tk.Button(gripper_btn_frame2, text='Close', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._gripper_close).pack(side=tk.LEFT, padx=5)

        # --- Action buttons: Set Joints / Plan & Execute ---
        ik_btn_frame = tk.Frame(frame)
        ik_btn_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(ik_btn_frame, text='Set Joints', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._ik_btn_set_joints).pack(side=tk.LEFT, padx=5)
        tk.Button(ik_btn_frame, text='Plan & Execute', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._ik_btn_plan_execute).pack(side=tk.LEFT, padx=5)

        # IK state tracking
        self.ik_status_var = tk.StringVar(value='Ready')
        self._ik_valid = True
        self._ik_planned_target = None

        # Hidden EE pose storage — used by services
        self.ee_labels = {}
        for key in ['X', 'Y', 'Z', 'qx', 'qy', 'qz', 'qw']:
            self.ee_labels[key] = tk.StringVar(value='---')

    def _build_grasp_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Grasp')

        # --- Object Source ---
        topic_frame = ttk.LabelFrame(frame, text='Object Source')
        topic_frame.pack(fill=tk.X, padx=10, pady=5)

        topic_row = tk.Frame(topic_frame)
        topic_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(topic_row, text='Topic:', anchor='w').pack(side=tk.LEFT)
        default_topic = '/objects_poses_real' if self.use_real_hardware else '/objects_poses_sim'
        self._grasp_topic_var = tk.StringVar(value=default_topic)
        tk.Entry(topic_row, textvariable=self._grasp_topic_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        tk.Button(topic_row, text='Update Topic', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._update_grasp_topic).pack(side=tk.RIGHT, padx=(2, 0))
        tk.Button(topic_row, text='Refresh', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._refresh_objects).pack(side=tk.RIGHT, padx=(2, 0))

        opts_row = tk.Frame(topic_frame)
        opts_row.pack(fill=tk.X, padx=5, pady=2)
        default_bbox_topic = '/objects_bbox_real' if self.use_real_hardware else '/objects_bbox_sim'
        tk.Label(opts_row, text='BBox:', anchor='w').pack(side=tk.LEFT)
        self._bbox_topic_var = tk.StringVar(value=default_bbox_topic)
        tk.Entry(opts_row, textvariable=self._bbox_topic_var, width=22).pack(
            side=tk.LEFT, padx=(5, 5))
        self._bbox_enabled_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opts_row, text='TCP offset',
                       variable=self._bbox_enabled_var).pack(side=tk.LEFT, padx=5)

        # --- Detected Objects ---
        obj_frame = ttk.LabelFrame(frame, text='Detected Objects')
        obj_frame.pack(fill=tk.X, padx=10, pady=5)

        self.obj_listbox = tk.Listbox(obj_frame, height=5, font=('Consolas', 9),
                                       selectbackground='#d0d0d0',
                                       selectforeground='#1a1a1a')
        self.obj_listbox.pack(fill=tk.X, padx=5, pady=2)

        # --- Arm | Gripper columns ---
        ctrl_cols = ttk.Frame(frame)
        ctrl_cols.pack(fill=tk.X, padx=10, pady=5)

        # Left column: Arm
        arm_col = ttk.LabelFrame(ctrl_cols, text='Arm')
        arm_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))

        arm_dur_row = tk.Frame(arm_col)
        arm_dur_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(arm_dur_row, text='Duration (s):', anchor='w').pack(side=tk.LEFT)
        self._grasp_arm_duration_var = tk.DoubleVar(value=2.5)
        tk.Spinbox(arm_dur_row, textvariable=self._grasp_arm_duration_var,
                   from_=0.5, to=10.0, increment=0.5,
                   width=8, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        approach_row = tk.Frame(arm_col)
        approach_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(approach_row, text='Approach height (m):', anchor='w').pack(side=tk.LEFT)
        self._grasp_approach_height_var = tk.DoubleVar(value=0.05)
        tk.Spinbox(approach_row, textvariable=self._grasp_approach_height_var,
                   from_=0.00, to=0.20, increment=0.01,
                   width=6, format='%.2f').pack(side=tk.LEFT, padx=(5, 0))

        obj_z_row = tk.Frame(arm_col)
        obj_z_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(obj_z_row, text='Object Z (m):', anchor='w').pack(side=tk.LEFT)
        self._grasp_obj_z_var = tk.DoubleVar(value=0.0)
        tk.Spinbox(obj_z_row, textvariable=self._grasp_obj_z_var,
                   from_=-0.05, to=0.20, increment=0.005,
                   width=6, format='%.3f').pack(side=tk.LEFT, padx=(5, 0))

        self._grasp_cross_var = tk.BooleanVar(value=False)
        tk.Checkbutton(arm_col, text='Cross-axis grasp',
                       variable=self._grasp_cross_var).pack(
            fill=tk.X, padx=5, pady=1)

        tk.Button(arm_col, text='Home', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._grasp_reset).pack(fill=tk.X, padx=5, pady=2)
        self._grasp_move_btn = tk.Button(
            arm_col, text='Move to Grab', bg='#b0b0b0', fg='#1a1a1a',
            command=self._move_to_object)
        self._grasp_move_btn.pack(fill=tk.X, padx=5, pady=2)

        # Right column: Gripper
        grip_col = ttk.LabelFrame(ctrl_cols, text='Gripper')
        grip_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 0))

        _jaw_min_deg = math.degrees(JOINT_LIMITS['gripper_joint'][0])
        _jaw_max_deg = math.degrees(JOINT_LIMITS['gripper_joint'][1])
        grip_range_row = tk.Frame(grip_col)
        grip_range_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(grip_range_row, text='Range:', anchor='w').pack(side=tk.LEFT)
        self._grasp_grip_close_var = tk.DoubleVar(value=_jaw_min_deg)
        tk.Spinbox(grip_range_row, textvariable=self._grasp_grip_close_var,
                   from_=_jaw_min_deg, to=_jaw_max_deg,
                   increment=5, width=5, format='%.0f').pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(grip_range_row, text='-').pack(side=tk.LEFT)
        self._grasp_grip_open_var = tk.DoubleVar(value=_jaw_max_deg)
        tk.Spinbox(grip_range_row, textvariable=self._grasp_grip_open_var,
                   from_=_jaw_min_deg, to=_jaw_max_deg,
                   increment=5, width=5, format='%.0f').pack(side=tk.LEFT)
        tk.Label(grip_range_row, text='\u00b0').pack(side=tk.LEFT)

        grip_dur_row = tk.Frame(grip_col)
        grip_dur_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(grip_dur_row, text='Duration (s):', anchor='w').pack(side=tk.LEFT)
        self._grasp_grip_duration_var = tk.DoubleVar(value=3.0)
        tk.Spinbox(grip_dur_row, textvariable=self._grasp_grip_duration_var,
                   from_=0.2, to=5.0, increment=0.1,
                   width=8, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        clearance_row = tk.Frame(grip_col)
        clearance_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(clearance_row, text='Open clearance (mm):', anchor='w').pack(side=tk.LEFT)
        self._jaw_open_clearance_var = tk.DoubleVar(value=JAW_OPEN_CLEARANCE_M * 1000)
        tk.Spinbox(clearance_row, textvariable=self._jaw_open_clearance_var,
                   from_=-5.0, to=20.0, increment=0.5,
                   width=5, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        close_cl_row = tk.Frame(grip_col)
        close_cl_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(close_cl_row, text='Close clearance (mm):', anchor='w').pack(side=tk.LEFT)
        self._jaw_close_clearance_var = tk.DoubleVar(value=JAW_CLOSE_CLEARANCE_M * 1000)
        tk.Spinbox(close_cl_row, textvariable=self._jaw_close_clearance_var,
                   from_=-10.0, to=10.0, increment=0.5,
                   width=5, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        tcp_clear_row = tk.Frame(grip_col)
        tcp_clear_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(tcp_clear_row, text='TCP clearance (mm):', anchor='w').pack(side=tk.LEFT)
        self._tcp_clearance_var = tk.DoubleVar(value=TCP_CLEARANCE_M * 1000)
        tk.Spinbox(tcp_clear_row, textvariable=self._tcp_clearance_var,
                   from_=-5.0, to=10.0, increment=0.5,
                   width=5, format='%.1f').pack(side=tk.LEFT, padx=(5, 0))

        grip_btn_row1 = tk.Frame(grip_col)
        grip_btn_row1.pack(fill=tk.X, padx=5, pady=2)
        tk.Button(grip_btn_row1, text='Grasp Open', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._gripper_open_for_object).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(grip_btn_row1, text='Grasp Close', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._gripper_close_for_object).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        grip_btn_row2 = tk.Frame(grip_col)
        grip_btn_row2.pack(fill=tk.X, padx=5, pady=2)
        tk.Button(grip_btn_row2, text='Open', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._gripper_open_range).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(grip_btn_row2, text='Close', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._gripper_close_range).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # Initial subscription to default topic
        self._update_grasp_topic()

    # ------------------------------------------------------------------
    # IK tab: tab-change, FK, spinbox IK, buttons
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_to_rpy_deg(qx, qy, qz, qw):
        """Convert quaternion to Roll/shoulder_lift/Yaw in degrees."""
        sinr = 2.0 * (qw * qx + qy * qz)
        cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny, cosy)
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    @staticmethod
    def _rpy_deg_to_quat(roll_deg, pitch_deg, yaw_deg):
        """Convert Roll/shoulder_lift/Yaw in degrees to quaternion (x, y, z, w)."""
        r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
        cr, cp, cy = math.cos(r / 2), math.cos(p / 2), math.cos(y / 2)
        sr, sp, sy = math.sin(r / 2), math.sin(p / 2), math.sin(y / 2)
        return (sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy)

    def _get_ik_target_quat(self):
        """Read XYZ + RPY from spinboxes, return (x, y, z, qx, qy, qz, qw)."""
        x = self.xyz_vars['X'].get()
        y = self.xyz_vars['Y'].get()
        z = self.xyz_vars['Z'].get()
        qx, qy, qz, qw = self._rpy_deg_to_quat(
            self.rpy_vars['Roll'].get(),
            self.rpy_vars['shoulder_lift'].get(),
            self.rpy_vars['Yaw'].get())
        return x, y, z, qx, qy, qz, qw

    def _on_tab_changed(self, event):
        """Auto-populate IK spinboxes from current EE pose when switching to IK tab."""
        tab_text = self._notebook.tab(self._notebook.select(), 'text')
        if tab_text != 'IK':
            return
        self._compute_fk_to_spinboxes()
        with self.joint_lock:
            self._ik_planned_target = {n: self.joint_positions[n] for n in ARM_JOINT_NAMES}
        self._ik_valid = True

    def _compute_fk_to_spinboxes(self, joint_positions=None):
        """Compute FK and populate IK spinboxes. Uses current joints if None."""
        if not MOVEIT_AVAILABLE or not hasattr(self, 'fk_client'):
            return
        if not self.fk_client.service_is_ready():
            return

        if joint_positions is None:
            with self.joint_lock:
                positions = [self.joint_positions[n] for n in ARM_JOINT_NAMES]
        else:
            positions = [joint_positions[n] for n in ARM_JOINT_NAMES]

        def _call_fk():
            with self._ik_solve_lock:
                request = GetPositionFK.Request()
                request.header.frame_id = 'base'
                request.header.stamp = self.get_clock().now().to_msg()
                request.fk_link_names = ['tcp_link']
                request.robot_state.joint_state.name = list(ARM_JOINT_NAMES)
                request.robot_state.joint_state.position = list(positions)

                future = self.fk_client.call_async(request)
                self._wait_future(future, timeout_sec=2.0)

            if future.result() is None:
                return
            resp = future.result()
            if resp.error_code.val != 1 or not resp.pose_stamped:
                return

            p = resp.pose_stamped[0].pose.position
            o = resp.pose_stamped[0].pose.orientation
            if getattr(self, '_gui_ready', False):
                self.root.after(0, self._populate_ik_spinboxes,
                               p.x, p.y, p.z, o.x, o.y, o.z, o.w)

        threading.Thread(target=_call_fk, daemon=True).start()

    def _populate_ik_spinboxes(self, x, y, z, qx, qy, qz, qw):
        """Set IK spinbox values from FK result and mark state as valid."""
        self._ik_trace_active = False  # suppress IK solves during batch update
        self.xyz_vars['X'].set(round(x, 3))
        self.xyz_vars['Y'].set(round(y, 3))
        self.xyz_vars['Z'].set(round(z, 3))
        r, p, ya = self._quat_to_rpy_deg(qx, qy, qz, qw)
        self.rpy_vars['Roll'].set(round(r, 1))
        self.rpy_vars['shoulder_lift'].set(round(p, 1))
        self.rpy_vars['Yaw'].set(round(ya, 1))
        self._ik_trace_active = True
        # Store as last valid
        for key in ['X', 'Y', 'Z']:
            self._ik_last_valid[key] = self.xyz_vars[key].get()
        for key in ['Roll', 'shoulder_lift', 'Yaw']:
            self._ik_last_valid[key] = self.rpy_vars[key].get()
        # Mark valid, clear red
        self._ik_valid = True
        for spin in self._ik_spinboxes.values():
            spin.config(fg='black')

    def _ik_reset(self):
        """Reset IK tab: zero arm and populate spinboxes from resulting FK."""
        self._zero_arm()
        def _after_zero():
            with self.joint_lock:
                pos = {n: self.joint_positions[n] for n in ARM_JOINT_NAMES}
            self._ik_planned_target = dict(pos)
            self._ik_valid = True
            self._compute_fk_to_spinboxes(pos)
        if getattr(self, '_gui_ready', False):
            self.root.after(500, _after_zero)

    # --- Spinbox IK: debounced, serialized ---

    def _on_ik_var_changed(self, field):
        """Any IK spinbox variable changed. Debounce then solve IK."""
        if not self._ik_trace_active:
            self.get_logger().debug(f'IK trace suppressed: {field}')
            return  # suppress during programmatic batch updates
        self.get_logger().info(f'IK var changed: {field}')
        if self._ik_debounce_id is not None:
            self.root.after_cancel(self._ik_debounce_id)
        self._ik_debounce_id = self.root.after(
            150, lambda: self._ik_solve_interactive(field))

    def _ik_solve_interactive(self, changed_field, revert_on_fail=False):
        """Compute IK from current spinbox values. Serialized via lock."""
        if not MOVEIT_AVAILABLE or self.ik_client is None:
            self.get_logger().warn('IK solve skipped: MoveIt not available')
            return
        if not self.ik_client.service_is_ready():
            self.get_logger().warn('IK solve skipped: /compute_ik not ready')
            return

        self._ik_debounce_id = None
        x, y, z, qx, qy, qz, qw = self._get_ik_target_quat()
        self.get_logger().info(
            f'IK solve starting: {changed_field} -> ({x:.3f}, {y:.3f}, {z:.3f})')

        with self.joint_lock:
            current_joints = [self.joint_positions[n] for n in ARM_JOINT_NAMES]

        seeds = [
            list(current_joints),
            [math.atan2(-y, x) if abs(x) + abs(y) > 0.001 else 0.0,
             0.0, 0.0, 0.0, 0.0],
            [0.0] * len(ARM_JOINT_NAMES),
        ]

        self._ik_solve_gen += 1
        gen = self._ik_solve_gen

        def _make_ik_request(seed, avoid_collisions):
            request = GetPositionIK.Request()
            ik_req = PositionIKRequest()
            ik_req.group_name = 'arm'
            ik_req.avoid_collisions = avoid_collisions
            robot_state = RobotState()
            robot_state.joint_state.name = list(ARM_JOINT_NAMES)
            robot_state.joint_state.position = list(seed)
            ik_req.robot_state = robot_state
            pose = PoseStamped()
            pose.header.frame_id = 'base'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            ik_req.pose_stamped = pose
            ik_req.timeout.sec = 0
            ik_req.timeout.nanosec = 500_000_000  # 0.5s cap per request
            request.ik_request = ik_req
            return request

        def _extract_joints(result):
            sol = result.solution.joint_state
            target = {}
            for i, name in enumerate(sol.name):
                if name in ARM_JOINT_NAMES and i < len(sol.position):
                    target[name] = sol.position[i]
            return target

        def _solve():
            if not self._ik_solve_lock.acquire(blocking=False):
                self.get_logger().info('IK solve: lock busy, skipping')
                return  # another solve running, debounce will retry
            try:
                # Pass 1: collision-aware IK (valid + safe)
                for i, seed in enumerate(seeds):
                    request = _make_ik_request(seed, avoid_collisions=True)
                    future = self.ik_client.call_async(request)
                    result = self._wait_future(future, timeout_sec=2.0)
                    code = result.error_code.val if result else 'N/A'
                    self.get_logger().info(
                        f'IK pass1 seed{i}: done={future.done()}, code={code}')
                    if result is not None and result.error_code.val == 1:
                        target = _extract_joints(result)
                        self.get_logger().info('IK SUCCESS (collision-free)')
                        if getattr(self, '_gui_ready', False):
                            self.root.after(0, self._ik_interactive_success,
                                           target, gen)
                        return

                # Pass 2: collision-ignored IK (reachable but colliding)
                for i, seed in enumerate(seeds):
                    request = _make_ik_request(seed, avoid_collisions=False)
                    future = self.ik_client.call_async(request)
                    result = self._wait_future(future, timeout_sec=2.0)
                    code = result.error_code.val if result else 'N/A'
                    self.get_logger().info(
                        f'IK pass2 seed{i}: done={future.done()}, code={code}')
                    if result is not None and result.error_code.val == 1:
                        target = _extract_joints(result)
                        self.get_logger().info('IK COLLISION (goal shown red)')
                        if getattr(self, '_gui_ready', False):
                            self.root.after(0, self._ik_interactive_fail_with_goal,
                                           target, changed_field, gen)
                        return

                # Truly unreachable — no solution even ignoring collisions
                self.get_logger().info('IK UNREACHABLE (all seeds failed)')
                if getattr(self, '_gui_ready', False):
                    self.root.after(0, self._ik_interactive_fail,
                                   changed_field, revert_on_fail, gen)
            finally:
                self._ik_solve_lock.release()

        threading.Thread(target=_solve, daemon=True).start()

    def _ik_interactive_success(self, target, gen):
        """IK succeeded — update goal state, clear red."""
        if gen != self._ik_solve_gen:
            return  # stale

        self._ik_valid = True
        self._ik_planned_target = dict(target)

        for key in ['X', 'Y', 'Z']:
            self._ik_last_valid[key] = self.xyz_vars[key].get()
        for key in ['Roll', 'shoulder_lift', 'Yaw']:
            self._ik_last_valid[key] = self.rpy_vars[key].get()

        # Clear all red
        for spin in self._ik_spinboxes.values():
            spin.config(fg='black')

        # Update sliders + goal state
        self._slider_driven = True
        self._select_planning_group('arm')
        with self.joint_lock:
            for name in ARM_JOINT_NAMES:
                if name in target:
                    self.joint_positions[name] = target[name]
        for name in ARM_JOINT_NAMES:
            if name in target and name in self.sliders:
                self.sliders[name].set(target[name])
                self.slider_labels[name].config(text=f'{target[name]:.3f}')
        self._publish_goal_state()

    def _ik_interactive_fail_with_goal(self, target, changed_field, gen):
        """IK collision — publish colliding goal (RViz shows red), mark field red."""
        if gen != self._ik_solve_gen:
            return  # stale

        self._ik_valid = False
        self._ik_planned_target = None  # not executable

        # Mark the offending field red
        self._ik_spinboxes[changed_field].config(fg='red')

        # Publish the colliding solution so goal robot shows RED in RViz
        self._slider_driven = True
        self._select_planning_group('arm')
        with self.joint_lock:
            for name in ARM_JOINT_NAMES:
                if name in target:
                    self.joint_positions[name] = target[name]
        for name in ARM_JOINT_NAMES:
            if name in target and name in self.sliders:
                self.sliders[name].set(target[name])
                self.slider_labels[name].config(text=f'{target[name]:.3f}')
        self._publish_goal_state()

        self._append_log(
            f'IK collision — {changed_field} causes collision', 'warn')

    def _ik_interactive_fail(self, changed_field, revert_on_fail, gen):
        """IK truly unreachable — mark field red, log warning."""
        if gen != self._ik_solve_gen:
            return  # stale

        self._ik_valid = False

        # Mark the offending field red
        self._ik_spinboxes[changed_field].config(fg='red')
        self._append_log(
            f'IK unreachable — {changed_field} value out of workspace', 'warn')

    # --- IK buttons (always enabled, show warning if invalid) ---

    def _ik_btn_set_joints(self):
        """Send currently planned IK joints directly to controllers."""
        if not self._ik_valid or self._ik_planned_target is None:
            self._append_log('IK solution not found — adjust target first', 'warn')
            return
        self._set_joints()

    def _ik_btn_plan_execute(self):
        """Plan & Execute from currently planned IK joints via MoveIt."""
        if not self._ik_valid or self._ik_planned_target is None:
            self._append_log('IK solution not found — adjust target first', 'warn')
            return
        self._moveit_execute()

    # --- IK services (for programmatic access) ---

    @service_trigger('ik_set_joints')
    def _ik_set_joints(self):
        """Service: send IK joints to controllers."""
        if self._ik_valid and self._ik_planned_target is not None:
            self._set_joints()
        else:
            self._compute_ik_full(mode='set_joints')

    @service_trigger('ik_plan_execute')
    def _ik_plan_execute(self):
        """Service: plan & execute IK joints."""
        if self._ik_valid and self._ik_planned_target is not None:
            self._moveit_execute()
        else:
            self._compute_ik_full(mode='plan_execute')

    @service_trigger('ik_randomize')
    def _ik_randomize(self):
        """Randomize arm goal state, compute FK, populate spinboxes."""
        self._randomize_arm()
        def _fk_after_randomize():
            with self.joint_lock:
                pos = {n: self.joint_positions[n] for n in ARM_JOINT_NAMES}
            self._ik_planned_target = dict(pos)
            self._ik_valid = True
            joints_str = ', '.join(f'{n}: {pos[n]:.3f}' for n in ARM_JOINT_NAMES)
            self._append_log(f'Randomized goal: {joints_str}')
            self._compute_fk_to_spinboxes(pos)
        if getattr(self, '_gui_ready', False):
            self.root.after(1500, _fk_after_randomize)

    # --- Full IK solver (for services, 6 seeds + 2 passes) ---

    def _compute_ik_full(self, mode=None, target_pose=None, grasp_yaw=None):
        """IK computation: geometric solver for grasps, MoveIt for general moves.

        For grasp moves (grasp_yaw is not None), uses the analytical geometric
        IK solver which produces exact gripper-down solutions in ~100µs, then
        validates with MoveIt collision checking. No MoveIt fallback — if
        geometric IK fails, the position is unreachable for a top-down grasp.

        For non-grasp moves, uses the original multi-seed MoveIt KDL solver.

        target_pose: optional (x, y, z, qx, qy, qz, qw) to bypass spinboxes.
        grasp_yaw: desired jaw-line direction (rad) for top-down grasp alignment.
        """
        if target_pose is not None:
            x, y, z, qx, qy, qz, qw = target_pose
        else:
            x, y, z, qx, qy, qz, qw = self._get_ik_target_quat()

        # --- Grasp moves: geometric IK + collision check (no MoveIt fallback) ---
        if grasp_yaw is not None:
            def _try_geometric():
                with self._ik_solve_lock:
                    # Pre-check: is the target within the top-down grasp workspace?
                    ground_z = self._ground_z_var.get() if hasattr(self, '_ground_z_var') else None
                    ok, reason = check_grasp_reachable(x, y, z, ground_z=ground_z)
                    if not ok:
                        self._append_log(
                            f'Grasp rejected: {reason} '
                            f'({x:.3f}, {y:.3f}, {z:.3f})', 'warn')
                        return

                    from so_arm101_control.compute_workspace import geometric_ik
                    solutions = geometric_ik(x, y, z, grasp_yaw=grasp_yaw)
                    if not solutions:
                        self._append_log(
                            f'Grasp unreachable: no geometric IK solution '
                            f'({x:.3f}, {y:.3f}, {z:.3f})', 'warn')
                        return

                    # Try each solution (elbow-up first, then elbow-down)
                    for i, sol in enumerate(solutions):
                        config = 'elbow-up' if i == 0 else 'elbow-down'
                        if self._check_state_valid(sol):
                            self._append_log(
                                f'Geometric IK: {config}, '
                                f'wrist_roll='
                                f'{math.degrees(sol["wrist_roll"]):.1f}°')
                            self._ik_apply_and_act(sol, mode)
                            return
                        self._append_log(
                            f'Geometric IK: {config} collides', 'warn')

                    # All geometric solutions collide — reject (no MoveIt fallback)
                    self._append_log(
                        'Grasp unreachable: all geometric IK solutions collide '
                        f'({x:.3f}, {y:.3f}, {z:.3f})', 'warn')

            threading.Thread(target=_try_geometric, daemon=True).start()
            return

        # --- Non-grasp moves: MoveIt IK (original pipeline) ---
        if not MOVEIT_AVAILABLE or self.ik_client is None:
            self._append_log('moveit_msgs not installed', 'warn')
            return
        if not self.ik_client.service_is_ready():
            self._append_log('compute_ik service not ready', 'warn')
            return
        self._compute_ik_moveit(x, y, z, qx, qy, qz, qw, mode, grasp_yaw)

    def _compute_ik_moveit(self, x, y, z, qx, qy, qz, qw, mode, grasp_yaw):
        """MoveIt multi-seed IK solver — used for non-grasp moves only."""
        if not MOVEIT_AVAILABLE or self.ik_client is None:
            self._append_log('moveit_msgs not installed', 'warn')
            return
        if not self.ik_client.service_is_ready():
            self._append_log('compute_ik service not ready', 'warn')
            return

        with self.joint_lock:
            current_joints = [self.joint_positions[n] for n in ARM_JOINT_NAMES]

        bearing = math.atan2(-y, x) if abs(x) + abs(y) > 0.001 else 0.0

        seeds = [
            list(current_joints),
            [bearing, 0.0, 0.0, 0.0, 0.0],
            [0.0] * len(ARM_JOINT_NAMES),
        ]
        for _ in range(3):
            seeds.append([random.uniform(*JOINT_LIMITS[n]) for n in ARM_JOINT_NAMES])

        def _try_seeds():
            with self._ik_solve_lock:
                for avoid_col in [True, False]:
                    for seed in seeds:
                        request = GetPositionIK.Request()
                        ik_req = PositionIKRequest()
                        ik_req.group_name = 'arm'
                        ik_req.avoid_collisions = avoid_col

                        robot_state = RobotState()
                        robot_state.joint_state.name = list(ARM_JOINT_NAMES)
                        robot_state.joint_state.position = list(seed)
                        ik_req.robot_state = robot_state

                        pose = PoseStamped()
                        pose.header.frame_id = 'base'
                        pose.header.stamp = self.get_clock().now().to_msg()
                        pose.pose.position.x = x
                        pose.pose.position.y = y
                        pose.pose.position.z = z
                        pose.pose.orientation.x = qx
                        pose.pose.orientation.y = qy
                        pose.pose.orientation.z = qz
                        pose.pose.orientation.w = qw
                        ik_req.pose_stamped = pose
                        ik_req.timeout.sec = 0
                        ik_req.timeout.nanosec = 500_000_000  # 0.5s cap
                        request.ik_request = ik_req

                        future = self.ik_client.call_async(request)
                        self._wait_future(future, timeout_sec=2.0)

                        if future.result() is None:
                            continue
                        resp = future.result()
                        if resp.error_code.val == 1:
                            sol = resp.solution.joint_state
                            target = {}
                            for i, name in enumerate(sol.name):
                                if name in ARM_JOINT_NAMES \
                                        and i < len(sol.position):
                                    target[name] = sol.position[i]
                            self._ik_apply_and_act(target, mode)
                            return

            self._append_log(
                f'IK failed for ({x:.3f}, {y:.3f}, {z:.3f})', 'warn')

        threading.Thread(target=_try_seeds, daemon=True).start()

    def _check_state_valid(self, target):
        """Check if a joint state is collision-free via MoveIt's planning scene."""
        if not MOVEIT_AVAILABLE or not hasattr(self, 'validity_client') \
                or not self.validity_client.service_is_ready():
            return True  # no checker available — assume valid
        req = GetStateValidity.Request()
        req.robot_state.joint_state.name = list(ARM_JOINT_NAMES)
        req.robot_state.joint_state.position = [
            target.get(n, 0.0) for n in ARM_JOINT_NAMES]
        req.group_name = 'arm'
        future = self.validity_client.call_async(req)
        self._wait_future(future, timeout_sec=1.0)
        if future.result() is not None:
            return future.result().valid
        return True  # timeout — assume valid

    def _ik_apply_and_act(self, target, mode):
        """Apply IK solution to sliders/goal state and optionally execute."""
        def _apply():
            self._slider_driven = True
            self._select_planning_group('arm')
            with self.joint_lock:
                for name in ARM_JOINT_NAMES:
                    if name in target:
                        self.joint_positions[name] = target[name]
            for name in ARM_JOINT_NAMES:
                if name in target and name in self.sliders:
                    self.sliders[name].set(target[name])
                    self.slider_labels[name].config(text=f'{target[name]:.3f}')
            self._publish_goal_state()
            self._ik_planned_target = dict(target)
            self._ik_valid = True
            if mode == 'set_joints':
                self._set_joints()
            elif mode == 'plan_execute':
                self._moveit_execute()
            elif mode == 'grasp_approach':
                duration = getattr(self, '_grasp_arm_duration', 2.0)
                final_joints = getattr(self, '_grasp_final_joints', None)
                def _descend():
                    if final_joints is not None:
                        self._append_log('Approach complete, descending to grasp')
                        self._ik_apply_and_act(final_joints, 'grasp_execute')
                self._execute_trajectory(target, duration_s=duration,
                                         on_complete=_descend)
            elif mode == 'grasp_execute':
                duration = getattr(self, '_grasp_arm_duration', 2.0)
                self._execute_trajectory(target, duration_s=duration)

        if getattr(self, '_gui_ready', False):
            self.root.after(0, _apply)

    def _set_ik_status(self, text):
        self.ik_status_var.set(text)
        self._append_log(text)

    def _ee_pose_callback(self, msg):
        if getattr(self, '_gui_ready', False):
            p = msg.pose.position
            o = msg.pose.orientation
            self.root.after(0, self._update_ee_display,
                           p.x, p.y, p.z, o.x, o.y, o.z, o.w)

    def _update_ee_display(self, x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        self.ee_labels['X'].set(f'{x:.4f}')
        self.ee_labels['Y'].set(f'{y:.4f}')
        self.ee_labels['Z'].set(f'{z:.4f}')
        self.ee_labels['qx'].set(f'{qx:.4f}')
        self.ee_labels['qy'].set(f'{qy:.4f}')
        self.ee_labels['qz'].set(f'{qz:.4f}')
        self.ee_labels['qw'].set(f'{qw:.4f}')

    def _objects_callback(self, msg):
        with self.objects_lock:
            for tf in msg.transforms:
                name = tf.child_frame_id
                self.objects_data[name] = {
                    'x': tf.transform.translation.x,
                    'y': tf.transform.translation.y,
                    'z': tf.transform.translation.z,
                    'qx': tf.transform.rotation.x,
                    'qy': tf.transform.rotation.y,
                    'qz': tf.transform.rotation.z,
                    'qw': tf.transform.rotation.w,
                }

    def _bbox_callback(self, msg):
        """Cache world-aligned bounding boxes from /objects_bbox_sim."""
        import json
        try:
            self.objects_bbox = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    @service_trigger('grasp_update_topic')
    def _update_grasp_topic(self):
        """Switch object subscription to topic from GUI entry and auto-refresh."""
        new_topic = self._grasp_topic_var.get().strip()
        if not new_topic:
            return
        # Destroy old subscription and create new one
        if hasattr(self, 'objects_sub') and self.objects_sub is not None:
            self.destroy_subscription(self.objects_sub)
        with self.objects_lock:
            self.objects_data.clear()
        self.objects_sub = self.create_subscription(
            TFMessage, new_topic, self._objects_callback, 10)
        # Update button text
        if hasattr(self, '_grasp_move_btn'):
            if new_topic == '/drop_poses':
                self._grasp_move_btn.config(text='Move to Drop')
            else:
                self._grasp_move_btn.config(text='Move to Grab')
        self._append_log(f'Grasp topic: {new_topic}')

    @service_trigger('grasp_refresh')
    def _refresh_objects(self):
        if not hasattr(self, 'obj_listbox'):
            return
        # Clear stale data so only fresh messages populate the list
        with self.objects_lock:
            self.objects_data.clear()
        self.obj_listbox.delete(0, tk.END)
        # Wait briefly for new messages to arrive, then populate
        if getattr(self, '_gui_ready', False):
            self.root.after(500, self._populate_object_list)

    def _populate_object_list(self):
        if not hasattr(self, 'obj_listbox'):
            return
        self.obj_listbox.delete(0, tk.END)
        with self.objects_lock:
            for name, pos in self.objects_data.items():
                self.obj_listbox.insert(
                    tk.END,
                    f'{name}  ({pos["x"]:.3f}, {pos["y"]:.3f}, {pos["z"]:.3f})')
        count = self.obj_listbox.size()
        if count > 0:
            self._append_log(f'Objects refreshed: {count} found')

    @service_trigger('grasp_reset')
    def _grasp_reset(self):
        """Move arm to grasp-ready home: gripper pointing down.

        Plans from current joint state (not from zeros) to avoid flinging.
        """
        duration = self._grasp_arm_duration_var.get()
        target = {name: 0.0 for name in ARM_JOINT_NAMES}
        target['wrist_flex'] = math.pi / 2
        self._execute_trajectory(target, duration_s=duration)

    @service_trigger('grasp_select')
    def _grasp_select(self):
        """Select an object in the listbox by name (via ik_target param) or first item.
        Usage: ros2 param set ... ik_target "green_1" then call this service.
        """
        if not hasattr(self, 'obj_listbox') or self.obj_listbox.size() == 0:
            self._append_log('No objects to select', 'warn')
            return
        # Check if a name was specified via the ik_target parameter
        name_hint = self.get_parameter('ik_target').get_parameter_value().string_value.strip()
        target_idx = 0
        if name_hint and '=' not in name_hint:
            for i in range(self.obj_listbox.size()):
                if self.obj_listbox.get(i).split('  ')[0] == name_hint:
                    target_idx = i
                    break
        self.obj_listbox.selection_clear(0, tk.END)
        self.obj_listbox.selection_set(target_idx)
        text = self.obj_listbox.get(target_idx)
        self._append_log(f'Selected: {text.split("  ")[0]}')

    @classmethod
    def _grasp_orientation(cls, obj_x, obj_y, obj_qz, obj_qw):
        """Compute a top-down grasp quaternion for a table object.
        The gripper approaches from above (pitch=90°), with yaw aligned
        to the pan angle toward the object plus the object's z-rotation.
        Returns (qx, qy, qz, qw) for the EE in the base frame.
        """
        pan = math.atan2(obj_y, obj_x) if abs(obj_x) + abs(obj_y) > 0.001 else 0.0
        obj_yaw = math.atan2(2.0 * obj_qw * obj_qz, 1.0 - 2.0 * obj_qz * obj_qz)
        yaw_deg = math.degrees(pan + obj_yaw)
        return cls._rpy_deg_to_quat(0.0, 90.0, yaw_deg)

    def _get_selected_object_name(self):
        """Return the object name selected in the grasp listbox, or None."""
        sel = self.obj_listbox.curselection()
        if not sel:
            return None
        return self.obj_listbox.get(sel[0]).split('  ')[0]

    def _get_grip_width(self, obj_name):
        """Return the grip width for the object, respecting cross-axis checkbox."""
        bbox = self.objects_bbox.get(obj_name)
        if not bbox:
            return None
        cross = self._grasp_cross_var.get() if hasattr(self, '_grasp_cross_var') else False
        return max(bbox['sx'], bbox['sy']) if cross else min(bbox['sx'], bbox['sy'])

    def _compute_jaw_offset(self, obj_name, obj_yaw):
        """Compute (dx, dy) to shift TCP so the object center sits between jaws.

        TCP is at the fixed jaw tip. The moving jaw opens along the object's
        local +Y axis: (-sin(obj_yaw), cos(obj_yaw)) in world frame.
        (wrist_roll compensates for pan, so only obj_yaw matters.)
        We shift TCP from object center toward the fixed jaw by grip_width/2.
        Returns (0, 0) when TCP offset is disabled via the checkbox.
        """
        if hasattr(self, '_bbox_enabled_var') and not self._bbox_enabled_var.get():
            return 0.0, 0.0
        grip_width = self._get_grip_width(obj_name)
        if grip_width is None:
            return 0.0, 0.0
        tcp_clear = (self._tcp_clearance_var.get() / 1000
                     if hasattr(self, '_tcp_clearance_var') else TCP_CLEARANCE_M)
        half_offset = grip_width / 2 + tcp_clear
        # fixed_jaw_dir = (-sin(obj_yaw), +cos(obj_yaw))
        dx = -half_offset * math.sin(obj_yaw)
        dy = half_offset * math.cos(obj_yaw)
        return dx, dy

    def _gripper_angle_for_object(self, obj_name):
        """Return (open_angle, close_angle) in radians for the selected object.

        Uses jaw_gap = BASELINE_JAW_GAP + JAW_GAP_RATE * angle model derived
        from STL mesh analysis. Falls back to full range if no bbox data.
        """
        grip_width = self._get_grip_width(obj_name) if obj_name else None
        if grip_width is None:
            return JOINT_LIMITS['gripper_joint'][1], JOINT_LIMITS['gripper_joint'][0]
        # Read tunable values from UI (mm → m), fall back to module constants
        tcp_clear = (self._tcp_clearance_var.get() / 1000
                     if hasattr(self, '_tcp_clearance_var') else TCP_CLEARANCE_M)
        open_cl = (self._jaw_open_clearance_var.get() / 1000
                   if hasattr(self, '_jaw_open_clearance_var') else JAW_OPEN_CLEARANCE_M)
        close_cl = (self._jaw_close_clearance_var.get() / 1000
                    if hasattr(self, '_jaw_close_clearance_var') else JAW_CLOSE_CLEARANCE_M)
        # Symmetric baseline = grip_width + 2*tcp_clearance (tcp_clear gap each side)
        # Open/close clearances are extra gap on top of that baseline
        # angle = (desired_gap - baseline_jaw_gap) / rate
        open_gap = grip_width + 2 * tcp_clear + open_cl
        open_angle = (open_gap - BASELINE_JAW_GAP) / JAW_GAP_RATE
        open_angle = max(JOINT_LIMITS['gripper_joint'][0],
                         min(open_angle, JOINT_LIMITS['gripper_joint'][1]))
        close_gap = grip_width + 2 * tcp_clear + close_cl
        close_angle = (close_gap - BASELINE_JAW_GAP) / JAW_GAP_RATE
        close_angle = max(JOINT_LIMITS['gripper_joint'][0],
                          min(close_angle, JOINT_LIMITS['gripper_joint'][1]))
        return open_angle, close_angle

    @service_trigger('gripper_open_for_object')
    def _gripper_open_for_object(self):
        """Open gripper to the angle matching the selected object's width."""
        obj_name = self._get_selected_object_name()
        bbox = self.objects_bbox.get(obj_name) if obj_name else None
        if not bbox:
            self._append_log('Grasp Open: no object selected or no bbox data')
            return
        open_angle, _ = self._gripper_angle_for_object(obj_name)
        self._append_log(
            f'Grasp Open: {math.degrees(open_angle):.1f}° for {obj_name}')
        self._gripper_command(open_angle, execute=True,
                              duration_s=self._grasp_grip_duration_var.get())

    @service_trigger('gripper_close_for_object')
    def _gripper_close_for_object(self):
        """Close gripper to the object's width minus threshold."""
        obj_name = self._get_selected_object_name()
        bbox = self.objects_bbox.get(obj_name) if obj_name else None
        if not bbox:
            self._append_log('Grasp Close: no object selected or no bbox data')
            return
        _, close_angle = self._gripper_angle_for_object(obj_name)
        self._append_log(f'Grasp Close: {math.degrees(close_angle):.1f}° for {obj_name}')
        self._gripper_command(close_angle, execute=True,
                              duration_s=self._grasp_grip_duration_var.get())

    @service_trigger('grasp_move')
    def _move_to_object(self):
        sel = self.obj_listbox.curselection()
        if not sel:
            self._append_log('No object selected', 'warn')
            return
        text = self.obj_listbox.get(sel[0])
        obj_name = text.split('  ')[0]
        with self.objects_lock:
            pos = self.objects_data.get(obj_name)
        if pos is None:
            self._append_log(f'Object "{obj_name}" not found', 'warn')
            return

        topic = self._grasp_topic_var.get().strip()
        z_offset = 0.05 if topic == '/drop_poses' else 0.0

        # Use Object Z override if set, otherwise use detected z
        obj_z_override = self._grasp_obj_z_var.get()
        base_z = obj_z_override if abs(obj_z_override) > 1e-4 else pos['z']
        target_z = base_z + z_offset
        action = 'drop' if topic == '/drop_poses' else 'grab'

        # Compute object yaw for wrist_roll alignment (two-stage IK)
        obj_qz = pos.get('qz', 0.0)
        obj_qw = pos.get('qw', 1.0)
        obj_yaw = math.atan2(2.0 * obj_qw * obj_qz, 1.0 - 2.0 * obj_qz * obj_qz)

        # Minor-axis (cross) grasp: rotate 90° to close across the short axis
        cross = self._grasp_cross_var.get()
        if cross:
            obj_yaw += math.pi / 2

        # Normalize yaw so wrist_roll stays within joint limits
        # (gripper jaws are symmetric: yaw ≡ yaw+π for grasping)
        pan = math.atan2(pos['y'], pos['x']) if abs(pos['x']) + abs(pos['y']) > 0.001 else 0.0
        obj_yaw = _normalize_grasp_yaw(obj_yaw, pan)

        # Jaw offset: shift target so object sits between both jaws
        jaw_dx, jaw_dy = self._compute_jaw_offset(obj_name, obj_yaw)
        tx, ty = pos['x'] + jaw_dx, pos['y'] + jaw_dy
        if abs(jaw_dx) > 0.001 or abs(jaw_dy) > 0.001:
            gw = (self._get_grip_width(obj_name) or 0) * 1000
            self._append_log(
                f'  Jaw offset: ({jaw_dx*1000:+.1f}, {jaw_dy*1000:+.1f})mm '
                f'for {gw:.0f}mm grip{"[cross]" if cross else ""}')

        self._append_log(
            f'Grasp: {action} "{obj_name}" at '
            f'({tx:.3f}, {ty:.3f}, {target_z:.3f})'
            f'{" [cross]" if cross else ""}')

        self._grasp_arm_duration = self._grasp_arm_duration_var.get()
        # Compute top-down grasp orientation (pitch=90°, yaw aligned to object)
        gqx, gqy, gqz, gqw = self._grasp_orientation(
            tx, ty, obj_qz, obj_qw)

        approach_h = self._grasp_approach_height_var.get()

        # Pre-validate ALL stages before moving the arm
        def _prevalidate_and_execute():
            from so_arm101_control.compute_workspace import geometric_ik
            ground_z = self._ground_z_var.get() if hasattr(self, '_ground_z_var') else None

            poses_to_check = []
            if approach_h > 0:
                approach_z = target_z + approach_h
                poses_to_check.append(('approach', tx, ty, approach_z))
                poses_to_check.append(('final', tx, ty, target_z))
            else:
                poses_to_check.append(('final', tx, ty, target_z))

            # Validate each stage: workspace check → geometric IK → collision
            validated = {}  # stage_name -> joint solution dict
            for stage, px, py, pz in poses_to_check:
                ok, reason = check_grasp_reachable(px, py, pz, ground_z=ground_z)
                if not ok:
                    self._append_log(
                        f'Grasp rejected ({stage}): {reason} '
                        f'({px:.3f}, {py:.3f}, {pz:.3f})', 'warn')
                    return

                solutions = geometric_ik(px, py, pz, grasp_yaw=obj_yaw)
                if not solutions:
                    self._append_log(
                        f'Grasp unreachable ({stage}): no geometric IK '
                        f'({px:.3f}, {py:.3f}, {pz:.3f})', 'warn')
                    return

                found = False
                for i, sol in enumerate(solutions):
                    config = 'elbow-up' if i == 0 else 'elbow-down'
                    if self._check_state_valid(sol):
                        validated[stage] = sol
                        self._append_log(
                            f'  {stage}: {config}, '
                            f'wrist_roll={math.degrees(sol["wrist_roll"]):.1f}°')
                        found = True
                        break
                    self._append_log(
                        f'  {stage}: {config} collides', 'warn')

                if not found:
                    self._append_log(
                        f'Grasp unreachable ({stage}): all solutions collide '
                        f'({px:.3f}, {py:.3f}, {pz:.3f})', 'warn')
                    return

            # All stages validated — execute
            duration = self._grasp_arm_duration
            if approach_h > 0 and 'approach' in validated and 'final' in validated:
                self._append_log(f'  Both stages validated, executing approach')
                self._grasp_final_joints = validated['final']
                def _apply_approach():
                    self._ik_apply_and_act(validated['approach'], 'grasp_approach')
                if getattr(self, '_gui_ready', False):
                    self.root.after(0, _apply_approach)
            elif 'final' in validated:
                def _apply_final():
                    self._ik_apply_and_act(validated['final'], 'grasp_execute')
                if getattr(self, '_gui_ready', False):
                    self.root.after(0, _apply_final)

        threading.Thread(target=_prevalidate_and_execute, daemon=True).start()

    # ------------------------------------------------------------------
    # Tab 3: Gripper Control
    # ------------------------------------------------------------------

    @service_trigger('gripper_close')
    def _gripper_close(self):
        self._gripper_command(JOINT_LIMITS['gripper_joint'][0], execute=True)

    @service_trigger('gripper_open')
    def _gripper_open(self):
        self._gripper_command(JOINT_LIMITS['gripper_joint'][1], execute=True)

    @service_trigger('gripper_open_range')
    def _gripper_open_range(self):
        """Open gripper to the range spinbox upper value (grasp tab)."""
        angle = math.radians(self._grasp_grip_open_var.get())
        self._gripper_command(angle, execute=True,
                              duration_s=self._grasp_grip_duration_var.get())

    @service_trigger('gripper_close_range')
    def _gripper_close_range(self):
        """Close gripper to the range spinbox lower value (grasp tab)."""
        angle = math.radians(self._grasp_grip_close_var.get())
        self._gripper_command(angle, execute=True,
                              duration_s=self._grasp_grip_duration_var.get())

    @service_trigger('set_jaw_open_clearance')
    def _set_jaw_open_clearance(self):
        """Set jaw open clearance: ros2 param set ... jaw_open_clearance_mm 5.0"""
        val = self.get_parameter('jaw_open_clearance_mm').get_parameter_value().double_value
        if hasattr(self, '_jaw_open_clearance_var'):
            self._jaw_open_clearance_var.set(val)
        self._append_log(f'Jaw open clearance set to {val:.1f}mm')

    @service_trigger('set_jaw_close_clearance')
    def _set_jaw_close_clearance(self):
        """Set jaw close clearance: ros2 param set ... jaw_close_clearance_mm 0.0
        +ve = more gap, -ve = tighter"""
        val = self.get_parameter('jaw_close_clearance_mm').get_parameter_value().double_value
        if hasattr(self, '_jaw_close_clearance_var'):
            self._jaw_close_clearance_var.set(val)
        self._append_log(f'Jaw close clearance set to {val:+.1f}mm')

    @service_trigger('set_tcp_clearance')
    def _set_tcp_clearance(self):
        """Set TCP IK clearance from param: ros2 param set ... tcp_clearance_mm 1.0"""
        val = self.get_parameter('tcp_clearance_mm').get_parameter_value().double_value
        if hasattr(self, '_tcp_clearance_var'):
            self._tcp_clearance_var.set(val)
        self._append_log(f'TCP clearance set to {val:.1f}mm')

    @service_trigger('check_grasp_reachable')
    def _check_grasp_reachable_svc(self):
        """Check if the selected object is within the top-down grasp workspace."""
        obj_name = self._get_selected_object_name()
        if not obj_name:
            self._append_log('No object selected for reachability check', 'warn')
            return
        with self.objects_lock:
            pos = self.objects_data.get(obj_name)
        if pos is None:
            self._append_log(f'Object "{obj_name}" not found', 'warn')
            return
        obj_z_override = self._grasp_obj_z_var.get()
        z = obj_z_override if abs(obj_z_override) > 1e-4 else pos['z']
        ground_z = self._ground_z_var.get() if hasattr(self, '_ground_z_var') else None
        ok, reason = check_grasp_reachable(pos['x'], pos['y'], z, ground_z=ground_z)
        r = math.sqrt(pos['x']**2 + pos['y']**2)
        if ok:
            self._append_log(
                f'Grasp reachable: "{obj_name}" r={r:.3f}m z={z:.3f}m '
                f'[R: {GRASP_WORKSPACE_BOUNDS["R_MIN"]:.3f}-'
                f'{GRASP_WORKSPACE_BOUNDS["R_MAX"]:.3f}m, '
                f'Z: {GRASP_WORKSPACE_BOUNDS["Z_MIN"]:.3f}-'
                f'{GRASP_WORKSPACE_BOUNDS["Z_MAX"]:.3f}m]')
        else:
            self._append_log(
                f'Grasp unreachable: "{obj_name}" — {reason}', 'warn')

    def _gripper_command(self, jaw_target, execute=False, duration_s=1.0):
        """Set gripper goal. If execute=True, also send to controller."""
        self._slider_driven = True
        self._select_planning_group('gripper')
        with self.joint_lock:
            self.joint_positions['gripper_joint'] = jaw_target
        if 'gripper_joint' in self.sliders:
            self.sliders['gripper_joint'].set(jaw_target)
            self.slider_labels['gripper_joint'].config(text=f'{jaw_target:.3f}')
        if hasattr(self, '_ik_jaw_label'):
            self._ik_jaw_label.config(text=f'{jaw_target:.3f}')
        self._publish_goal_state()
        if execute:
            self._send_gripper_goal(jaw_target, duration_s=duration_s)

    # ------------------------------------------------------------------
    # Trajectory execution (arm joints via action interface)
    # ------------------------------------------------------------------

    def _execute_trajectory(self, target_positions, duration_s=2.0, on_complete=None):
        """Send trajectory to arm_controller via action interface.
        Source: trajectory logic adapted from JETANK_description/jetank_control_gui.py"""
        if not self._traj_lock.acquire(blocking=False):
            self._append_log('Trajectory already in progress', 'warn')
            return

        self._slider_driven = True

        # Send the full trajectory via action (controller handles interpolation)
        self._send_arm_goal(target_positions, duration_s=duration_s)

        # Animate the UI sliders to show progress
        with self.joint_lock:
            start = {n: self.joint_positions[n] for n in ARM_JOINT_NAMES}

        def _run():
            try:
                steps = int(duration_s * 50)
                for i in range(steps + 1):
                    if not self.running:
                        break
                    t = i / max(steps, 1)
                    t = t * t * (3.0 - 2.0 * t)

                    positions = {}
                    for name in ARM_JOINT_NAMES:
                        s = start.get(name, 0.0)
                        e = target_positions.get(name, s)
                        lo, hi = JOINT_LIMITS[name]
                        positions[name] = max(lo, min(hi, s + (e - s) * t))

                    with self.joint_lock:
                        self.joint_positions.update(positions)

                    if getattr(self, '_gui_ready', False):
                        self.root.after(0, self._sync_arm_sliders, dict(positions))

                    self._publish_goal_state()

                    if self.use_real_hardware:
                        self._send_hw_command(positions)

                    time.sleep(1.0 / 50)

                self._append_log('Trajectory complete')
                self._slider_driven = False
            finally:
                self._traj_lock.release()
            if on_complete is not None:
                on_complete()

        threading.Thread(target=_run, daemon=True).start()

    def _sync_arm_sliders(self, positions):
        for name, val in positions.items():
            if name in self.sliders:
                self.sliders[name].set(val)
                self.slider_labels[name].config(text=f'{val:.3f}')

    # ------------------------------------------------------------------
    # Hardware mode
    # ------------------------------------------------------------------

    def _toggle_hardware(self):
        use_real = self.hw_var.get() == 'real'
        with self.hw_lock:
            if use_real == self.use_real_hardware:
                return
            self.use_real_hardware = use_real

        if use_real:
            self._append_log('Switching to REAL hardware mode', 'warn')
            self.status_var.set('Mode: Real Hardware')
            self.set_joints_btn.config(state=tk.DISABLED)
        else:
            self._append_log('Switching to SIMULATION mode')
            self.status_var.set('Mode: Simulation')
            self.set_joints_btn.config(state=tk.NORMAL)

        # Update grasp topic to match hardware mode
        if hasattr(self, '_grasp_topic_var'):
            new_topic = '/objects_poses_real' if use_real else '/objects_poses_sim'
            self._grasp_topic_var.set(new_topic)
            self._update_grasp_topic()

    @service_trigger('toggle_ground_plane')
    def _toggle_ground_plane(self):
        """Add/remove a ground plane collision object in MoveIt's planning scene."""
        if not MOVEIT_AVAILABLE or not hasattr(self, '_apply_scene_client'):
            self._append_log('MoveIt not available — cannot update planning scene', 'warn')
            return

        def _apply():
            if not self._apply_scene_client.wait_for_service(timeout_sec=5.0):
                self._append_log('apply_planning_scene service not available', 'warn')
                return

            # Step 1: Add or remove the collision object
            scene = PlanningSceneMsg()
            scene.is_diff = True

            co = CollisionObject()
            co.header.frame_id = 'base'
            co.id = 'ground_plane'

            adding = self._ground_plane_var.get()
            if adding:
                co.operation = CollisionObject.ADD
                box = SolidPrimitive()
                box.type = SolidPrimitive.BOX
                box.dimensions = [2.0, 2.0, 0.01]
                co.primitives.append(box)
                pose = Pose()
                z = self._ground_z_var.get()
                pose.position.z = z - 0.005  # center of 0.01-thick box
                pose.orientation.w = 1.0
                co.primitive_poses.append(pose)
            else:
                co.operation = CollisionObject.REMOVE

            scene.world.collision_objects.append(co)
            req = ApplyPlanningScene.Request()
            req.scene = scene
            future = self._apply_scene_client.call_async(req)
            self._wait_future(future, timeout_sec=5.0)
            if future.result() is None or not future.result().success:
                action = 'add' if adding else 'remove'
                self._append_log(f'Failed to {action} ground plane', 'warn')
                return

            if not adding:
                self._append_log('Ground plane removed')
                return

            # Step 2: Allow base <-> ground_plane collision in the ACM
            # (base sits on the ground — contact is expected)
            if not self._get_scene_client.wait_for_service(timeout_sec=5.0):
                self._append_log(f'Ground plane added at z={z:.3f} (ACM not updated)', 'warn')
                return

            get_req = GetPlanningSceneSrv.Request()
            get_req.components.components = 128  # ALLOWED_COLLISION_MATRIX
            future = self._get_scene_client.call_async(get_req)
            self._wait_future(future, timeout_sec=5.0)
            if future.result() is None:
                self._append_log(f'Ground plane added at z={z:.3f} (ACM not updated)', 'warn')
                return

            acm = future.result().scene.allowed_collision_matrix
            gp_name = 'ground_plane'
            if gp_name not in acm.entry_names:
                # Add new column (False) to every existing row
                for entry in acm.entry_values:
                    entry.enabled.append(False)
                # Add new row for ground_plane
                gp_row = AllowedCollisionEntry()
                gp_row.enabled = [False] * len(acm.entry_names) + [True]  # self=True
                # Allow contact with base
                if 'base' in acm.entry_names:
                    base_idx = acm.entry_names.index('base')
                    gp_row.enabled[base_idx] = True
                    acm.entry_values[base_idx].enabled[-1] = True
                acm.entry_names.append(gp_name)
                acm.entry_values.append(gp_row)

            acm_scene = PlanningSceneMsg()
            acm_scene.is_diff = True
            acm_scene.allowed_collision_matrix = acm
            req2 = ApplyPlanningScene.Request()
            req2.scene = acm_scene
            future2 = self._apply_scene_client.call_async(req2)
            self._wait_future(future2, timeout_sec=5.0)
            if future2.result() is not None and future2.result().success:
                self._append_log(f'Ground plane added at z={z:.3f}')
            else:
                self._append_log(f'Ground plane added at z={z:.3f} (ACM update failed)', 'warn')

        threading.Thread(target=_apply, daemon=True).start()

    def _real_js_callback(self, msg):
        if not self.use_real_hardware:
            return
        with self.joint_lock:
            for i, name in enumerate(msg.name):
                if name in self.joint_positions and i < len(msg.position):
                    self.joint_positions[name] = msg.position[i]
            positions = dict(self.joint_positions)

        if getattr(self, '_gui_ready', False):
            self.root.after(0, self._sync_all_sliders, positions)

    def _sync_all_sliders(self, positions):
        for name, val in positions.items():
            if name in self.sliders:
                self.sliders[name].set(val)
                if name in self.slider_labels:
                    self.slider_labels[name].config(text=f'{val:.3f}')

    def _send_hw_command(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(positions.keys())
        msg.position = list(positions.values())
        self.joint_cmd_pub.publish(msg)

    def _ext_cmd_callback(self, msg):
        """Handle external joint commands — update sliders and send to controllers."""
        self._slider_driven = True
        with self.joint_lock:
            for i, name in enumerate(msg.name):
                if name in self.joint_positions and i < len(msg.position):
                    lo, hi = JOINT_LIMITS.get(name, (-math.pi, math.pi))
                    self.joint_positions[name] = max(lo, min(hi, msg.position[i]))
            positions = dict(self.joint_positions)
        # External commands directly move the robot
        self._send_arm_goal(positions, duration_s=0.2)
        jaw = positions.get(GRIPPER_JOINT_NAME, 0.0)
        self._send_gripper_goal(jaw, duration_s=0.2)
        self._publish_goal_state()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_close(self):
        self.running = False
        self.root.quit()
        self.root.destroy()

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=rclpy.SignalHandlerOptions.NO)
    node = SOArm101ControlGUI()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def _shutdown_handler(signum, frame):
        node.running = False
        # Tell tkinter to quit from its own thread
        if hasattr(node, 'root'):
            try:
                node.root.after(0, node._on_close)
            except Exception:
                pass
        executor.shutdown()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        if hasattr(node, 'root'):
            try:
                node.root.quit()
            except Exception:
                pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
