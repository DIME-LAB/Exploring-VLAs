"""Mode-agnostic core for the SO-ARM101 stack.

Launches the pieces that are identical regardless of where /joint_states +
/objects_poses_* + /drop_poses come from:

  * robot_state_publisher (URDF → /robot_description + TF)
  * move_group            (MoveIt planning)
  * control_gui           (the operator + planner/executor)
  * ee_pose_publisher     (publishes EE pose for downstream loggers)
  * camera_pose_publisher (publishes camera frames for perception)
  * RViz (optional)

The simulator/real backend (Gazebo / Isaac / real-arm) lives in a sibling
launch file (backend_gazebo.launch.py / backend_isaac.launch.py /
backend_real.launch.py) and is included by stack.launch.py based on the
``mode`` argument.

This split exists because the previous monolithic gazebo.launch.py made
"run RViz + control_gui without Gazebo" impossible — every mode toggle
still kept gz sim running at ~130 % CPU.

Usage (typically you launch stack.launch.py instead):
  ros2 launch so_arm101_control core.launch.py rviz:=true use_sim_time:=true
"""

import os
import xacro
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    desc_pkg = get_package_share_directory('so_arm101_description')
    moveit_pkg = get_package_share_directory('so_arm101_moveit_config')

    # Use the Gazebo xacro for the URDF body too — it includes the
    # ros2_control block which Gazebo needs and which other backends
    # ignore (no controller_manager loads in Isaac/real, so the block
    # is inert there).
    xacro_file = os.path.join(desc_pkg, 'urdf', 'so_arm101.gazebo.urdf.xacro')
    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc)
    robot_description = doc.toxml()

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('rviz')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }],
    )

    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_pkg, 'launch', 'move_group.launch.py')),
    )

    control_gui = Node(
        package='so_arm101_control',
        executable='control_gui',
        name='so_arm101_control_gui',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    ee_pose_publisher = Node(
        package='so_arm101_control',
        executable='ee_pose_publisher',
        name='ee_pose_publisher',
        parameters=[{
            'base_frame': 'base',
            'ee_frame': 'gripper',
            'publish_rate': 10.0,
            'startup_delay': 5.0,
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    camera_pose_publisher = Node(
        package='so_arm101_control',
        executable='camera_pose_publisher',
        name='camera_pose_publisher',
        parameters=[{
            'base_frame': 'base',
            'camera_frame': 'camera_link',
            'publish_rate': 10.0,
            'startup_delay': 5.0,
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_pkg, 'launch', 'moveit_rviz.launch.py')),
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='True for Gazebo/Isaac (use /clock), false for real hardware.'),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='Launch RViz visualization.'),

        robot_state_publisher,
        move_group,
        control_gui,
        ee_pose_publisher,
        camera_pose_publisher,
        rviz,
    ])
