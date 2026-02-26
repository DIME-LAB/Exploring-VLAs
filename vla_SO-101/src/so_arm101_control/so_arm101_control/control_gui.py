#!/usr/bin/env python3
"""
SO-ARM101 Control GUI — Tkinter GUI with MoveIt IK, trajectory execution,
object detection, and dual hardware/simulation modes.

Source: adapted from RoboSort/JETANK_description/jetank_control_gui.py (3133 lines)
        IK delegated to MoveIt compute_ik service (KDL solver) instead of custom scipy.
        Joint mapping from MuammerBay/SO-ARM_ROS2_URDF and SO-ARM101_MoveIt_IsaacSim.
"""

import math
import threading
import time
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
try:
    from moveit_msgs.srv import GetPositionIK, GetMotionPlan, GetStateValidity
    from moveit_msgs.msg import (
        PositionIKRequest, RobotState, Constraints, JointConstraint,
        MotionPlanRequest,
    )
    from moveit_msgs.srv import ExecuteKnownTrajectory
    MOVEIT_AVAILABLE = True
except ImportError:
    try:
        from moveit_msgs.srv import GetPositionIK, GetMotionPlan, GetStateValidity
        from moveit_msgs.msg import (
            PositionIKRequest, RobotState, Constraints, JointConstraint,
            MotionPlanRequest,
        )
        MOVEIT_AVAILABLE = True
        ExecuteKnownTrajectory = None
    except ImportError:
        MOVEIT_AVAILABLE = False
        GetPositionIK = None
        GetStateValidity = None
        GetMotionPlan = None


# ---------------------------------------------------------------------------
# Joint configuration for SO-ARM101
# ---------------------------------------------------------------------------
ARM_JOINT_NAMES = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll']
GRIPPER_JOINT_NAME = 'Jaw'
ALL_JOINT_NAMES = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]

JOINT_LIMITS = {
    'Rotation':    (-1.91986, 1.91986),
    'Pitch':       (-1.74533, 1.74533),
    'Elbow':       (-1.74533, 1.5708),
    'Wrist_Pitch': (-1.65806, 1.65806),
    'Wrist_Roll':  (-2.79253, 2.79253),
    'Jaw':         (-0.174533, 1.74533),
}


class SOArm101ControlGUI(Node):
    """ROS2 node with embedded Tkinter GUI for SO-ARM101 control."""

    def __init__(self):
        super().__init__('so_arm101_control_gui')

        # Hardware mode
        self.use_real_hardware = False
        self.hw_lock = threading.Lock()

        # Current joint positions (radians) — updated from joint_state_broadcaster
        self.joint_positions = {name: 0.0 for name in ALL_JOINT_NAMES}
        self.joint_lock = threading.Lock()

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
        self.objects_sub = self.create_subscription(
            TFMessage, '/objects_poses', self._objects_callback, 10)
        self.ee_pose_sub = self.create_subscription(
            PoseStamped, '/ee_pose', self._ee_pose_callback, 10)

        # MoveIt service clients + publishers
        self.ik_client = None
        self.plan_client = None
        if MOVEIT_AVAILABLE:
            from moveit_msgs.msg import DisplayTrajectory
            self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
            self.plan_client = self.create_client(
                GetMotionPlan, '/plan_kinematic_path')
            self.validity_client = self.create_client(
                GetStateValidity, '/check_state_validity')
            self._display_traj_pub = self.create_publisher(
                DisplayTrajectory, '/display_planned_path', 10)
            # Publish goal state to RViz MotionPlanning plugin
            self._goal_state_pub = self.create_publisher(
                RobotState, '/rviz/moveit/update_custom_goal_state', 10)
            # Switch active planning group in RViz
            from std_msgs.msg import String
            self._planning_group_pub = self.create_publisher(
                String, '/rviz/moveit/select_planning_group', 10)
            self._active_planning_group = None  # Force first publish

        # Trajectory lock
        self._traj_lock = threading.Lock()

        # Track whether we should update sliders from joint_states
        self._slider_driven = False

        # GUI
        self.running = True
        self._setup_gui_thread()

        self.get_logger().info('SO-ARM101 Control GUI initialized')

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
        traj.joint_names = ['Jaw']
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
        if self._slider_driven:
            return
        with self.joint_lock:
            for i, name in enumerate(msg.name):
                if name in self.joint_positions and i < len(msg.position):
                    self.joint_positions[name] = msg.position[i]
            positions = dict(self.joint_positions)
        # Update sliders and goal state to reflect actual robot state
        if hasattr(self, 'root') and self.root.winfo_exists():
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
        self.root.geometry('580x680')
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

        # Notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))

        self._build_individual_tab(notebook)
        self._build_arm_control_tab(notebook)

        # --- Log Panel (bottom) ---
        self._build_log_panel()

        self.root.mainloop()

    # ------------------------------------------------------------------
    # Tab 1: Individual Joint Control
    # ------------------------------------------------------------------

    def _build_individual_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Robot Control')

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
        # Switch RViz planning group based on which joint is being moved
        if joint_name == GRIPPER_JOINT_NAME:
            self._select_planning_group('gripper')
        else:
            self._select_planning_group('arm')
        # Update RViz goal state only — robot doesn't move until button click
        self._publish_goal_state()
        # Reset slider_driven after user stops dragging
        if hasattr(self, '_slider_reset_id'):
            self.root.after_cancel(self._slider_reset_id)
        self._slider_reset_id = self.root.after(300, self._reset_slider_driven)

    def _reset_slider_driven(self):
        self._slider_driven = False

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
        self._slider_driven = True
        self._select_planning_group('gripper')
        with self.joint_lock:
            self.joint_positions[GRIPPER_JOINT_NAME] = 0.0
        if GRIPPER_JOINT_NAME in self.sliders:
            self.sliders[GRIPPER_JOINT_NAME].set(0.0)
            self.slider_labels[GRIPPER_JOINT_NAME].config(text='0.000')
        self._publish_goal_state()
        self.status_var.set('Gripper zeroed')

    def _select_planning_group(self, group_name):
        """Switch the active planning group in RViz."""
        if not hasattr(self, '_planning_group_pub'):
            return
        if hasattr(self, '_active_planning_group') and self._active_planning_group == group_name:
            return
        self._active_planning_group = group_name
        from std_msgs.msg import String
        msg = String()
        msg.data = group_name
        self._planning_group_pub.publish(msg)
        # Republish goal state after RViz processes the group switch
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(150, self._publish_goal_state)

    def _randomize_arm(self):
        """Set arm joints to a random collision-free configuration."""
        import random
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
                    rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                    if future.result() is not None and not future.result().valid:
                        continue  # collision — retry
                # Valid (or no validity service available) — apply it
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after(0, self._apply_random_arm, positions)
                return

            # Exhausted attempts — apply last one anyway
            self._append_log(f'No collision-free state after {max_attempts} attempts', 'warn')
            if hasattr(self, 'root') and self.root.winfo_exists():
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
        if hasattr(self, 'root') and self.root.winfo_exists():
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

        if hasattr(self, 'root') and self.root.winfo_exists():
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

        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, _do_append)

    # ------------------------------------------------------------------
    # Tab 2: Arm Control (IK via MoveIt)
    # ------------------------------------------------------------------

    def _build_arm_control_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text='Grasp')

        # XYZ inputs
        coord_frame = ttk.LabelFrame(frame, text='Target Position (meters)')
        coord_frame.pack(fill=tk.X, padx=10, pady=5)

        self.xyz_vars = {}
        for label, default in [('X', 0.0), ('Y', 0.0), ('Z', 0.15)]:
            row = tk.Frame(coord_frame)
            row.pack(fill=tk.X, padx=5, pady=2)
            tk.Label(row, text=f'{label}:', width=3).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=default)
            tk.Entry(row, textvariable=var, width=12).pack(side=tk.LEFT, padx=5)
            self.xyz_vars[label] = var

        # Gripper buttons
        gripper_frame2 = ttk.LabelFrame(frame, text='Gripper')
        gripper_frame2.pack(fill=tk.X, padx=10, pady=(2, 5))
        gripper_btn2 = tk.Frame(gripper_frame2)
        gripper_btn2.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(gripper_btn2, text='Open', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._gripper_open).pack(side=tk.LEFT, padx=5)
        tk.Button(gripper_btn2, text='Close', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._gripper_close).pack(side=tk.LEFT, padx=5)

        # Action buttons
        ik_btn_frame = tk.Frame(frame)
        ik_btn_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(ik_btn_frame, text='Set Joints', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._ik_set_joints).pack(side=tk.LEFT, padx=5)
        tk.Button(ik_btn_frame, text='Plan & Execute', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._ik_plan_execute).pack(side=tk.LEFT, padx=5)

        # IK status
        self.ik_status_var = tk.StringVar(value='Ready')
        tk.Label(frame, textvariable=self.ik_status_var, anchor='w',
                 font=('Arial', 9)).pack(fill=tk.X, padx=10, pady=2)

        # Current EE pose display
        ee_frame = ttk.LabelFrame(frame, text='Current End-Effector Pose')
        ee_frame.pack(fill=tk.X, padx=10, pady=5)
        self.ee_labels = {}
        for axis in ['X', 'Y', 'Z']:
            row = tk.Frame(ee_frame)
            row.pack(fill=tk.X, padx=5, pady=1)
            tk.Label(row, text=f'{axis}:', width=3).pack(side=tk.LEFT)
            lbl = tk.Label(row, text='---', width=12, anchor='w')
            lbl.pack(side=tk.LEFT)
            self.ee_labels[axis] = lbl

        # Object detection section
        obj_frame = ttk.LabelFrame(frame, text='Detected Objects')
        obj_frame.pack(fill=tk.X, padx=10, pady=5)

        self.obj_listbox = tk.Listbox(obj_frame, height=5)
        self.obj_listbox.pack(fill=tk.X, padx=5, pady=2)

        obj_btn = tk.Frame(obj_frame)
        obj_btn.pack(fill=tk.X, padx=5, pady=2)
        tk.Button(obj_btn, text='Refresh', bg='#b0b0b0', fg='#1a1a1a', command=self._refresh_objects).pack(side=tk.LEFT, padx=2)
        tk.Button(obj_btn, text='Move to Selected', bg='#b0b0b0', fg='#1a1a1a',
                  command=self._move_to_object).pack(side=tk.LEFT, padx=2)

    def _ik_set_joints(self):
        """Compute IK then send directly to controllers."""
        self._compute_ik(mode='set_joints')

    def _ik_plan_execute(self):
        """Compute IK then plan & execute via MoveIt."""
        self._compute_ik(mode='plan_execute')

    def _compute_ik(self, move=True, mode=None):
        if not MOVEIT_AVAILABLE or self.ik_client is None:
            self.ik_status_var.set('moveit_msgs not installed — IK unavailable')
            return
        if not self.ik_client.service_is_ready():
            self.ik_status_var.set('MoveIt /compute_ik service not available')
            self._append_log('compute_ik service not ready', 'warn')
            return

        self.ik_status_var.set('Computing IK...')

        x = self.xyz_vars['X'].get()
        y = self.xyz_vars['Y'].get()
        z = self.xyz_vars['Z'].get()

        request = GetPositionIK.Request()
        ik_req = PositionIKRequest()
        ik_req.group_name = 'arm'
        ik_req.avoid_collisions = True

        robot_state = RobotState()
        robot_state.joint_state.name = list(ARM_JOINT_NAMES)
        with self.joint_lock:
            robot_state.joint_state.position = [
                self.joint_positions[n] for n in ARM_JOINT_NAMES]
        ik_req.robot_state = robot_state

        pose = PoseStamped()
        pose.header.frame_id = 'base'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        ik_req.pose_stamped = pose

        request.ik_request = ik_req

        future = self.ik_client.call_async(request)
        future.add_done_callback(
            lambda f: self._ik_result_callback(f, move, mode))

    def _ik_result_callback(self, future, move, mode=None):
        """Handle IK service response. Called from ROS executor thread."""
        try:
            response = future.result()
        except Exception as e:
            self._set_ik_status(f'IK service error: {e}')
            return

        if response.error_code.val != 1:
            self._set_ik_status(
                f'IK failed (error code {response.error_code.val})')
            self._append_log(f'IK failed (error code {response.error_code.val})', 'warn')
            return

        sol = response.solution.joint_state
        target = {}
        for i, name in enumerate(sol.name):
            if name in ARM_JOINT_NAMES and i < len(sol.position):
                target[name] = sol.position[i]

        angles_str = ', '.join(f'{n}: {target.get(n, 0):.3f}' for n in ARM_JOINT_NAMES)
        self._set_ik_status(f'IK solution: {angles_str}')
        self._append_log(f'IK solution: {angles_str}')

        # Snap sliders to IK solution + publish goal state
        def _apply():
            self._select_planning_group('arm')
            for name in ARM_JOINT_NAMES:
                if name in target and name in self.sliders:
                    self.sliders[name].set(target[name])
            self._publish_goal_state()

        if mode == 'set_joints':
            # Snap sliders then send directly
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(0, _apply)
                self.root.after(50, self._set_joints)
        elif mode == 'plan_execute':
            # Snap sliders then plan & execute via MoveIt
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(0, _apply)
                self.root.after(50, self._moveit_execute)
        elif move:
            self._execute_trajectory(target, duration_s=2.0)

    def _set_ik_status(self, text):
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, self.ik_status_var.set, text)

    def _ee_pose_callback(self, msg):
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(0, self._update_ee_display,
                           msg.pose.position.x,
                           msg.pose.position.y,
                           msg.pose.position.z)

    def _update_ee_display(self, x, y, z):
        self.ee_labels['X'].config(text=f'{x:.4f}')
        self.ee_labels['Y'].config(text=f'{y:.4f}')
        self.ee_labels['Z'].config(text=f'{z:.4f}')

    def _objects_callback(self, msg):
        with self.objects_lock:
            for tf in msg.transforms:
                name = tf.child_frame_id
                self.objects_data[name] = {
                    'x': tf.transform.translation.x,
                    'y': tf.transform.translation.y,
                    'z': tf.transform.translation.z,
                }

    def _refresh_objects(self):
        self.obj_listbox.delete(0, tk.END)
        with self.objects_lock:
            for name, pos in self.objects_data.items():
                self.obj_listbox.insert(
                    tk.END,
                    f'{name}  ({pos["x"]:.3f}, {pos["y"]:.3f}, {pos["z"]:.3f})')

    def _move_to_object(self):
        sel = self.obj_listbox.curselection()
        if not sel:
            self.ik_status_var.set('No object selected')
            return
        text = self.obj_listbox.get(sel[0])
        obj_name = text.split('  ')[0]
        with self.objects_lock:
            pos = self.objects_data.get(obj_name)
        if pos is None:
            return
        self.xyz_vars['X'].set(pos['x'])
        self.xyz_vars['Y'].set(pos['y'])
        self.xyz_vars['Z'].set(pos['z'] + 0.05)
        self._ik_move()

    # ------------------------------------------------------------------
    # Tab 3: Gripper Control
    # ------------------------------------------------------------------

    def _gripper_close(self):
        self._slider_driven = True
        self._select_planning_group('gripper')
        target = JOINT_LIMITS['Jaw'][0]
        with self.joint_lock:
            self.joint_positions['Jaw'] = target
        if 'Jaw' in self.sliders:
            self.sliders['Jaw'].set(target)
            self.slider_labels['Jaw'].config(text=f'{target:.3f}')
        self._publish_goal_state()

    def _gripper_open(self):
        self._slider_driven = True
        self._select_planning_group('gripper')
        target = JOINT_LIMITS['Jaw'][1]
        with self.joint_lock:
            self.joint_positions['Jaw'] = target
        if 'Jaw' in self.sliders:
            self.sliders['Jaw'].set(target)
            self.slider_labels['Jaw'].config(text=f'{target:.3f}')
        self._publish_goal_state()

    # ------------------------------------------------------------------
    # Trajectory execution (arm joints via action interface)
    # ------------------------------------------------------------------

    def _execute_trajectory(self, target_positions, duration_s=2.0):
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

                    if hasattr(self, 'root') and self.root.winfo_exists():
                        self.root.after(0, self._sync_arm_sliders, dict(positions))

                    self._publish_goal_state()

                    if self.use_real_hardware:
                        self._send_hw_command(positions)

                    time.sleep(1.0 / 50)

                self._append_log('Trajectory complete')
            finally:
                self._traj_lock.release()

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

    def _real_js_callback(self, msg):
        if not self.use_real_hardware:
            return
        with self.joint_lock:
            for i, name in enumerate(msg.name):
                if name in self.joint_positions and i < len(msg.position):
                    self.joint_positions[name] = msg.position[i]
            positions = dict(self.joint_positions)

        if hasattr(self, 'root') and self.root.winfo_exists():
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
    rclpy.init(args=args)
    node = SOArm101ControlGUI()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
