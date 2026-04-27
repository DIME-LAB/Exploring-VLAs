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

DEFERRED — not in this launch yet:
  * trajectory_shim node — hosts arm_controller/follow_joint_trajectory
    and gripper_controller/follow_joint_trajectory action servers,
    interpolates incoming MoveIt trajectories at fixed rate, and publishes
    one JointState setpoint per tick to /joint_commands (which
    jointstatereader writes to the follower). Without this shim,
    control_gui's _execute_full_trajectory has nowhere to send its goal —
    MoveIt plans land but nothing executes.

Until the shim exists, this launch is **state ingest + perception only**:
you'll see the arm's joint positions in RViz, see object detections on
/objects_poses_real, but motion commands from control_gui will time out
on the action client.

Usage (typically via stack.launch.py mode:=real):
  ros2 launch so_arm101_control backend_real.launch.py \\
      leader_port:=/dev/ttyACM1 follower_port:=/dev/ttyACM0 \\
      wrist_idx:=0 top_idx:=1
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
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

    # --- Real arm: Feetech serial bridge ---
    # Topic-driven mode (mode 2 in joint_state_reader.py docstring): an
    # external publisher (eventually our trajectory_shim) drives the follower
    # by publishing JointState setpoints to /joint_commands. We don't want
    # USB-leader teleop active here.
    jointstatereader = Node(
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

        LogInfo(msg=(
            '[backend_real] motion execution shim NOT YET IMPLEMENTED — '
            'control_gui can plan but cannot execute on real hardware until '
            'arm_controller/follow_joint_trajectory action server exists. '
            'See backend_real.launch.py docstring.'
        )),

        jointstatereader,
        real_cameras,
        localize_aruco,
        localize_yoloe,
    ])
