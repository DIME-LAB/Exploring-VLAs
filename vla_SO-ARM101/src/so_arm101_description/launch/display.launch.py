"""
Display launch file for SO-ARM101 URDF visualization.
Launches robot_state_publisher, joint_state_publisher_gui, and RViz.
Source: adapted from MuammerBay/SO-ARM_ROS2_URDF
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('so_arm101_description')

    urdf_file = os.path.join(pkg_share, 'urdf', 'so_arm101.urdf')
    rviz_config = os.path.join(pkg_share, 'config', 'display.rviz')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen',
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        ),
    ])
