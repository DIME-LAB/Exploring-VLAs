"""Top-level dispatcher for the SO-ARM101 stack.

Always launches core.launch.py (mode-agnostic — RSP, move_group, control_gui,
ee/camera pose publishers, optional RViz) and conditionally one of three
backends based on the ``mode`` argument:

    mode:=gazebo  →  backend_gazebo.launch.py    (gz sim + bridge + controllers)
    mode:=isaac   →  backend_isaac.launch.py     (currently a no-op; Isaac runs externally)
    mode:=real    →  backend_real.launch.py      (jointstatereader + cameras + aruco)

The split lets you bring up just RViz + control_gui (mode:=isaac, no backend
processes) without paying the ~130 % CPU cost of an idle Gazebo server, and
gives the real-hardware path a self-contained launch graph that doesn't drag
in any simulator dependencies.

Usage:
  ros2 launch so_arm101_control stack.launch.py mode:=gazebo rviz:=true
  ros2 launch so_arm101_control stack.launch.py mode:=gazebo headless:=true
  ros2 launch so_arm101_control stack.launch.py mode:=isaac rviz:=true
  ros2 launch so_arm101_control stack.launch.py mode:=real follower_port:=/dev/ttyACM0
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    control_pkg = get_package_share_directory('so_arm101_control')
    launch_dir = os.path.join(control_pkg, 'launch')

    mode = LaunchConfiguration('mode')
    use_rviz = LaunchConfiguration('rviz')
    headless = LaunchConfiguration('headless')

    # use_sim_time = true for sim modes (gazebo, isaac), false for real.
    # Wrapped in PythonExpression so it resolves at launch time from `mode`.
    use_sim_time_expr = PythonExpression(
        ["'", mode, "' in ('gazebo', 'isaac')"])

    core = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'core.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time_expr,
            'rviz': use_rviz,
        }.items(),
    )

    backend_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'backend_gazebo.launch.py')),
        launch_arguments={'headless': headless}.items(),
        condition=IfCondition(PythonExpression(["'", mode, "' == 'gazebo'"])),
    )

    backend_isaac = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'backend_isaac.launch.py')),
        condition=IfCondition(PythonExpression(["'", mode, "' == 'isaac'"])),
    )

    backend_real = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, 'backend_real.launch.py')),
        # Pass-through args for the real backend.
        launch_arguments={
            'leader_port': LaunchConfiguration('leader_port'),
            'follower_port': LaunchConfiguration('follower_port'),
            'auto_detect': LaunchConfiguration('auto_detect'),
            'wrist_idx': LaunchConfiguration('wrist_idx'),
            'top_idx': LaunchConfiguration('top_idx'),
            'enable_top': LaunchConfiguration('enable_top'),
        }.items(),
        condition=IfCondition(PythonExpression(["'", mode, "' == 'real'"])),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'mode', default_value='gazebo',
            description='Backend: gazebo | isaac | real'),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='Launch RViz visualization (works in any mode).'),
        DeclareLaunchArgument(
            'headless', default_value='false',
            description='Gazebo-only: skip the Gazebo GUI client (-g).'),

        # Real-mode pass-through args, declared here so users can set them
        # at the dispatcher level. Ignored in gazebo/isaac modes.
        # Defaults match backend_real.launch.py — auto_detect resolves
        # wrist_idx via USB signature; enable_top defaults to false
        # because the documented real-hw rig is single-cam.
        DeclareLaunchArgument('leader_port',   default_value='/dev/ttyACM1'),
        DeclareLaunchArgument('follower_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('auto_detect',   default_value='true'),
        DeclareLaunchArgument('wrist_idx',     default_value='-1'),
        DeclareLaunchArgument('top_idx',       default_value='1'),
        DeclareLaunchArgument('enable_top',    default_value='false'),

        core,
        backend_gazebo,
        backend_isaac,
        backend_real,
    ])
