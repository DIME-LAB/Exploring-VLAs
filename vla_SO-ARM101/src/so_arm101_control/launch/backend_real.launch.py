"""Real-hardware backend for the SO-ARM101 stack.

Brings up the producers of the topic contract control_gui consumes,
sourced from the physical robot + USB cameras + ArUco/YOLOE perception:

  * jointstatereader        — Feetech serial bridge.
                              accept_command_input=True so external
                              setpoints on /joint_commands write to the
                              follower; mirror_to_follower=False so leader
                              USB teleop is OFF (the planner drives, not
                              a leader arm).
                              Publishes /joint_states (follower position).
  * real_cameras.launch.py  — wrist + top USB cameras as
                              /wrist_camera, /top_camera (so_arm101_bringup).
  * localize_aruco          — ArUco detection on /top_camera → /drop_poses
                              (cup marker frames).
  * localize_yoloe          — YOLOE-based lego detection → /objects_poses_real
                              + /objects_bbox_real (TFMessage + String JSON).

  * trajectory_shim       — hosts /arm_controller/follow_joint_trajectory
                              and /gripper_controller/follow_joint_trajectory
                              action servers (matching MoveIt's
                              moveit_controllers.yaml). Interpolates
                              received trajectories at 20 Hz on
                              /joint_commands; jointstatereader writes
                              those to the follower servos. Velocity
                              scaling via ``velocity_scale_real`` lets
                              first-contact tests run slow.

Pass ``read_only:=true`` to flip jointstatereader to observation-only mode
(accept_command_input=false, publish_commands=false). First-contact
bringup uses this to verify the read path before granting write authority.

Usage (typically via stack.launch.py mode:=real):
  ros2 launch so_arm101_control backend_real.launch.py \\
      leader_port:=/dev/ttyACM1 follower_port:=/dev/ttyACM0 \\
      wrist_idx:=0 top_idx:=1
  # First-contact (no writes to the follower):
  ros2 launch so_arm101_control backend_real.launch.py \\
      follower_port:=/dev/ttyACM0 read_only:=true
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    bringup_pkg = get_package_share_directory('so_arm101_bringup')

    leader_port = LaunchConfiguration('leader_port')
    follower_port = LaunchConfiguration('follower_port')
    wrist_idx = LaunchConfiguration('wrist_idx')
    top_idx = LaunchConfiguration('top_idx')
    enable_top = LaunchConfiguration('enable_top')
    read_only = LaunchConfiguration('read_only')
    velocity_scale_real = LaunchConfiguration('velocity_scale_real')

    # --- Real arm: Feetech serial bridge ---
    # Two configurations chosen by ``read_only``:
    #   read_only:=false (default) — topic-driven write mode (mode 2 in
    #     joint_state_reader.py): trajectory_shim / control_gui /
    #     a policy publishes JointState setpoints to /joint_commands and
    #     this node writes them to the follower servos.
    #   read_only:=true — observation-only (mode 3): /joint_states is
    #     published from the follower's measured position; nothing writes.
    #     Use this for first-contact bringup (RViz mirror, sanity-check
    #     state path) before granting any write authority.
    jointstatereader_write = Node(
        package='jointstatereader',
        executable='joint_state_reader',
        name='joint_state_reader',
        output='screen',
        parameters=[{
            'leader_port': leader_port,
            'follower_port': follower_port,
            'mirror_to_follower': False,
            'accept_command_input': True,
            'publish_source': 'follower',
            'publish_commands': True,
            'use_sim_time': False,
        }],
        condition=UnlessCondition(read_only),
    )
    jointstatereader_observe = Node(
        package='jointstatereader',
        executable='joint_state_reader',
        name='joint_state_reader',
        output='screen',
        parameters=[{
            'leader_port': leader_port,
            'follower_port': follower_port,
            'mirror_to_follower': False,
            'accept_command_input': False,
            'publish_source': 'follower',
            'publish_commands': False,
            'use_sim_time': False,
        }],
        condition=IfCondition(read_only),
    )

    # --- Real cameras: wrist + (optional) top ---
    real_cameras = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_pkg, 'launch', 'real_cameras.launch.py')),
        launch_arguments={
            'wrist_idx': wrist_idx,
            'top_idx': top_idx,
            'enable_top': enable_top,
        }.items(),
    )

    # --- ArUco cup-marker detection → /drop_poses ---
    # Reads /top_camera (default), publishes drop frames at marker poses.
    # Tunables (aruco dictionary, marker size, calibration) live inside
    # aruco_camera_localizer's own params — defaults match the lego_world
    # marker spec.
    localize_aruco = Node(
        package='aruco_camera_localizer',
        executable='localize_aruco',
        name='localize_aruco',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    # --- YOLOE lego detection → /objects_poses_real, /objects_bbox_real ---
    # Topic names match the sim's /objects_poses_sim / /objects_bbox_sim
    # contract (with _real suffix), so control_gui's grasp_update_topic
    # service can switch sources.
    localize_yoloe = Node(
        package='aruco_camera_localizer',
        executable='localize_yoloe',
        name='localize_yoloe',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    # --- Trajectory shim: MoveIt FollowJointTrajectory → /joint_commands ---
    # Only spawned when read_only=false (no point in a write-path executor
    # when jointstatereader is in observation-only mode). Velocity scaling
    # defaults to 0.25 (4x slower than MoveIt's planned timing) so the
    # first real-arm motions are deliberate; raise toward 1.0 once you
    # trust the path.
    trajectory_shim = Node(
        package='so_arm101_control',
        executable='trajectory_shim',
        name='trajectory_shim',
        output='screen',
        parameters=[{
            'velocity_scale': velocity_scale_real,
            'rate_hz': 20.0,
            'use_sim_time': False,
        }],
        condition=UnlessCondition(read_only),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'leader_port', default_value='/dev/ttyACM1',
            description='Feetech leader serial port (ignored; teleop is OFF in real-mode).'),
        DeclareLaunchArgument(
            'follower_port', default_value='/dev/ttyACM0',
            description='Feetech follower serial port — the arm we drive.'),
        DeclareLaunchArgument(
            'wrist_idx', default_value='0',
            description='cv2.VideoCapture index for the wrist camera.'),
        DeclareLaunchArgument(
            'top_idx', default_value='1',
            description='cv2.VideoCapture index for the overhead camera.'),
        DeclareLaunchArgument(
            'enable_top', default_value='true',
            description='Launch the top camera publisher (required for ArUco/YOLOE).'),
        DeclareLaunchArgument(
            'read_only', default_value='false',
            description=(
                'When true, run jointstatereader in observation-only mode '
                '(accept_command_input=false, publish_commands=false) and '
                'skip the trajectory_shim. Use for first-contact bringup '
                'before granting write authority.')),
        DeclareLaunchArgument(
            'velocity_scale_real', default_value='0.25',
            description=(
                'trajectory_shim velocity multiplier on time_from_start. '
                'Default 0.25 plays MoveIt-planned trajectories 4x slower '
                'than commanded — safer for first real-arm motions. '
                'Raise toward 1.0 once tracking lag and torque limits feel '
                'safe.')),

        jointstatereader_write,
        jointstatereader_observe,
        real_cameras,
        localize_aruco,
        localize_yoloe,
        trajectory_shim,
    ])
