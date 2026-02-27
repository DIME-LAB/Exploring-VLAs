"""
Full control stack launch for SO-ARM101.
Launches robot_state_publisher, ros2_control (mock hardware), MoveIt move_group,
control GUI, EE pose publisher, and optionally the servo driver for real hardware.

Usage:
  # Simulation mode (default):
  ros2 launch so_arm101_control control.launch.py

  # Real hardware mode:
  ros2 launch so_arm101_control control.launch.py real_hardware:=true serial_port:=/dev/ttyACM0
"""

import os
import subprocess
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    desc_pkg = get_package_share_directory('so_arm101_description')
    moveit_pkg = get_package_share_directory('so_arm101_moveit_config')

    # Process xacro URDF to include ros2_control tags
    xacro_file = os.path.join(moveit_pkg, 'config', 'so_arm101.urdf.xacro')
    robot_description = subprocess.check_output(
        ['xacro', xacro_file]).decode('utf-8')

    controllers_yaml = os.path.join(moveit_pkg, 'config', 'ros2_controllers.yaml')
    rviz_config = os.path.join(moveit_pkg, 'config', 'moveit.rviz')

    real_hardware = LaunchConfiguration('real_hardware')
    serial_port = LaunchConfiguration('serial_port')

    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument('real_hardware', default_value='false',
                              description='Use real servo hardware'),
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyACM0',
                              description='Serial port for servo driver'),

        # Robot State Publisher (with xacro-processed URDF including ros2_control)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen',
        ),

        # ros2_control controller manager (mock hardware for sim)
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[
                {'robot_description': robot_description},
                controllers_yaml,
            ],
            output='screen',
        ),

        # Spawn controllers via Node (waits for CM automatically)
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster'],
            output='screen',
        ),

        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['arm_controller'],
            output='screen',
        ),

        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['gripper_controller'],
            output='screen',
        ),

        # MoveIt move_group (for IK + planning + execution)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(moveit_pkg, 'launch', 'move_group.launch.py')),
        ),

        # Servo driver (only when real hardware)
        Node(
            package='so_arm101_control',
            executable='servo_driver',
            name='servo_driver',
            parameters=[{
                'serial_port': serial_port,
                'baud_rate': 1000000,
                'publish_rate_hz': 20.0,
            }],
            output='screen',
            condition=IfCondition(real_hardware),
        ),

        # Control GUI
        Node(
            package='so_arm101_control',
            executable='control_gui',
            name='so_arm101_control_gui',
            output='screen',
        ),

        # EE Pose Publisher
        Node(
            package='so_arm101_control',
            executable='ee_pose_publisher',
            name='ee_pose_publisher',
            parameters=[{
                'base_frame': 'base',
                'ee_frame': 'gripper',
                'publish_rate': 10.0,
                'startup_delay': 3.0,
            }],
            output='screen',
        ),

        # RViz (via MoveIt launch to pass robot_description_kinematics,
        # planning_pipelines, and joint_limits — required for MotionPlanning plugin)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(moveit_pkg, 'launch', 'moveit_rviz.launch.py')),
        ),
    ])
